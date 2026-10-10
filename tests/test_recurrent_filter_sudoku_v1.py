import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transport_reasoning.config import Config as LegacyConfig
from transport_reasoning.data import prepare, TaskData
from transport_reasoning.filters import action, coefficients
from transport_reasoning.models import Encoder as LegacyEncoder
from research.recurrent_filter_sudoku_v1.config import Config, from_legacy, grid
from research.recurrent_filter_sudoku_v1.model import Encoder
from research.recurrent_filter_sudoku_v1.training import train, optimization


def batch(size=4):
    n = size * size
    gen = torch.Generator().manual_seed(991)
    return {'x': torch.randn(2, n, 4 * size + 2, generator=gen, dtype=torch.float64),
            'mask': torch.ones(2, n, n, dtype=torch.bool)}


def small(**kwargs):
    return Config(size=4, width=16, heads=4, recurrent_steps=3, device='cpu', **kwargs)


def equal(a, b):
    if torch.is_tensor(a): torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray): assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b


def test_new_grid_and_legacy_hash_stability():
    old = LegacyConfig(task='sudoku', size=9, method='recurrent', width=128, heads=4, layers=4, recurrent_steps=32)
    before = old.hash(); new = from_legacy(old)
    assert old.hash() == before and 'architecture' not in old.dict()
    assert new.architecture == 'recurrent' and new.attention_filter == 'onehop' and new.recurrent_steps == 32
    runs = grid()
    assert len(runs) == 27 and sum(c.stage == 'pilot' for c in runs) == 12
    assert all(c.planned_steps == 30000 for c in runs)
    assert {c.completion_steps for c in runs if c.stage == 'pilot'} == {10000}
    assert len(grid(True)) == 32


def test_hop_one_equal_values_gradients_and_initialization():
    c = small(); a = Encoder(c).double(); b = Encoder(replace(c, attention_filter='binomial')).double()
    equal(a.state_dict(), b.state_dict())
    x = batch(); ya, yb = a(x), b(x)
    torch.testing.assert_close(ya, yb, rtol=1e-12, atol=1e-12)
    ya.square().mean().backward(); yb.square().mean().backward()
    for pa, pb in zip(a.parameters(), b.parameters()): torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-12, atol=1e-12)
    other = Encoder(replace(c, attention_filter='binomial', hops=4)).double()
    for (name, p), (other_name, q) in zip(a.named_parameters(), other.named_parameters()):
        assert name == other_name; equal(p, q)


def test_single_iteration_parameter_sharing_and_legacy_equivalence():
    c = replace(small(), recurrent_steps=1)
    model = Encoder(c).double(); x = batch()
    direct = model.readout(model.norm(model.blocks[0](model.embedding(x['x']), x['mask'])))
    torch.testing.assert_close(model(x), direct, rtol=0, atol=0)
    long = Encoder(replace(c, recurrent_steps=32, attention_filter='binomial', hops=4))
    assert model.parameter_count() == long.parameter_count()
    legacy_c = LegacyConfig(task='sudoku', size=4, method='recurrent', width=16, heads=4, layers=4, recurrent_steps=3, seed=101)
    legacy = LegacyEncoder(legacy_c).double(); translated = Encoder(from_legacy(legacy_c)).double()
    equal(legacy.state_dict(), translated.state_dict())
    torch.testing.assert_close(legacy(x), translated(x), rtol=0, atol=0)
    legacy(x).sum().backward(); translated(x).sum().backward()
    for a, b in zip(legacy.parameters(), translated.parameters()): equal(a.grad, b.grad)


def test_actual_operation_counts_and_attention_refresh():
    c = small(attention_filter='binomial', hops=4)
    model = Encoder(c).double(); x = batch(); seen = []; counts = {'qkv': 0, 'ffn': 0, 'softmax': 0, 'transport': 0}
    def qkv_hook(module, args, output): counts['qkv'] += 1; seen.append(args[0].detach().clone())
    def ffn_hook(module, args, output): counts['ffn'] += 1
    hooks = [model.blocks[0].qkv.register_forward_hook(qkv_hook), model.blocks[0].ff.register_forward_hook(ffn_hook)]
    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._softmax.default: counts['softmax'] += 1
            if func == torch.ops.aten.bmm.default and args[0].shape[-2:] == (16, 16) and args[1].shape[-2:] == (16, 4): counts['transport'] += 1
            return func(*args, **(kwargs or {}))
    with Count(): model(x)
    for h in hooks: h.remove()
    assert counts == {'qkv': 3, 'ffn': 3, 'softmax': 3, 'transport': 12}
    assert not torch.equal(seen[0], seen[1]) and not torch.equal(seen[1], seen[2])
    # The input to QKV changes and softmax is recomputed at every outer step.
    assert c.operations()['transport_matmuls'] == counts['transport']


def test_polynomial_gradient_and_targets_excluded():
    torch.manual_seed(9)
    z = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    a = logits.softmax(-1); c = coefficients('binomial', 4)
    computed = action(a, z, c)
    explicit = sum(w * torch.linalg.matrix_power(a, i + 1) @ z for i, w in enumerate(c))
    torch.testing.assert_close(computed, explicit, rtol=1e-12, atol=1e-12)
    ga = torch.autograd.grad(computed.square().sum(), (logits, z), retain_graph=True)
    gb = torch.autograd.grad(explicit.square().sum(), (logits, z))
    for x, y in zip(ga, gb): torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-12)
    model = Encoder(small()).double(); inputs = batch()
    torch.testing.assert_close(model({**inputs, 'y': torch.zeros(2, 16)}), model({**inputs, 'y': torch.ones(2, 16)}), rtol=0, atol=0)


def test_exact_offgrid_resume_and_threshold_archives(tmp_path, monkeypatch):
    import research.recurrent_filter_sudoku_v1.training as training
    oldcwd = Path.cwd(); monkeypatch.chdir(tmp_path)
    legacy = LegacyConfig(task='sudoku', size=4, train_count=16, val_count=4, test_count=4)
    prepare(legacy); datadir = legacy.data_dir().resolve()
    monkeypatch.setenv('RECURRENT_DATA_DIR', str(datadir))
    c = small(attention_filter='binomial', hops=2, planned_steps=6, completion_steps=6, eval_every=2, batch_size=4, stage='checks')
    calls = [0]; evaluate = training.evaluate_model
    def forced_threshold(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        result['solved_rate'] = [.1, .2, .6, .7][calls[0]]; calls[0] += 1
        return result
    monkeypatch.setattr(training, 'evaluate_model', forced_threshold)
    monkeypatch.setenv('RECURRENT_OUTPUT', str(tmp_path / 'direct'))
    train(c); direct = torch.load(c.run_dir() / 'checkpoint_last.pt', weights_only=False)
    direct_best = torch.load(c.run_dir() / 'checkpoint_best.pt', weights_only=False)
    calls[0] = 0; monkeypatch.setenv('RECURRENT_OUTPUT', str(tmp_path / 'resumed'))
    result = train(c, stop_after=3); assert result['status'] == 'paused'
    assert [json.loads(s)['step'] for s in (c.run_dir() / 'metrics.jsonl').read_text().splitlines()] == [1, 2]
    train(c, resume=c.run_dir() / 'checkpoint_last.pt')
    resumed = torch.load(c.run_dir() / 'checkpoint_last.pt', weights_only=False)
    for key in ['model', 'optimizer', 'scheduler', 'python_rng', 'numpy_rng', 'torch_rng', 'data_rng', 'step', 'best']:
        equal(direct[key], resumed[key])
    equal(direct_best['model'], torch.load(c.run_dir() / 'checkpoint_best.pt', weights_only=False)['model'])
    assert torch.load(c.run_dir() / 'threshold_50_previous_failing.pt', weights_only=False)['step'] == 2
    assert torch.load(c.run_dir() / 'threshold_50_first_passing.pt', weights_only=False)['step'] == 4
    assert resumed['thresholds']['0.5']['confirmed_step'] == 6
    state = copy.deepcopy(resumed['data_rng']); g1 = np.random.default_rng(); g2 = np.random.default_rng()
    g1.bit_generator.state = state; g2.bit_generator.state = direct['data_rng']
    assert np.array_equal(g1.integers(0, 16, 64), g2.integers(0, 16, 64))


def test_budget_admission_freezes_optional_before_test(tmp_path, monkeypatch):
    from research.recurrent_filter_sudoku_v1.campaign import admission
    from transport_reasoning.config import write_json
    monkeypatch.setenv('RECURRENT_OUTPUT', str(tmp_path))
    write_json(tmp_path/'profile/budget_estimate.json', {'mandatory_seconds_with_20_percent_margin':1000,
               'optional_B32_additional_seconds_with_20_percent_margin':500, 'reuse_historical_R32_quality':True,'new_training_runs':22})
    assert admission(900) is None
    assert not (tmp_path/'admission.json').exists()
    value = admission(1200)
    assert not value['include_final_B32'] and value['new_training_runs'] == 22
    assert admission(1800) == value


def test_accounting_stops_at_absolute_gpu_cap(tmp_path, monkeypatch):
    from argparse import Namespace
    from research.recurrent_filter_sudoku_v1 import scheduler
    from transport_reasoning.config import write_json
    monkeypatch.setenv('RECURRENT_OUTPUT',str(tmp_path))
    value={'gpu_jobs':[{'job_id':'123','accounted':False,'reserved_seconds':600}], 'gpu_seconds_used':3500,
           'first_allocation_unix':__import__('time').time(),'status':'paused'}
    scheduler.save(value)
    write_json(tmp_path/'queue/budget_grant.json',{'gpu_seconds_limit':4100,'wall_seconds_limit':7200})
    commands=[]
    def command(args):
        commands.append(args)
        assert args[0]=='sacct'
        return '123|COMPLETED|600|gres/gpu=1|0:0|'
    monkeypatch.setattr(scheduler,'command',command)
    monkeypatch.setattr(scheduler,'batch',lambda *a,**k: pytest.fail('No budget for another allocation'))
    result=scheduler.account(Namespace(job='123'))
    assert result['gpu_seconds_used']==4100 and result['status']=='budget_stopped'
    assert len(commands)==1 and scheduler.account(Namespace(job='123'))['gpu_seconds_used']==4100


def test_mechanism_and_inference_measurement_execute(tmp_path, monkeypatch):
    from research.recurrent_filter_sudoku_v1.measure import diagnose, benchmark
    monkeypatch.chdir(tmp_path)
    legacy=LegacyConfig(task='sudoku',size=4,train_count=64,val_count=16,test_count=16)
    prepare(legacy); monkeypatch.setenv('RECURRENT_DATA_DIR',str(legacy.data_dir().resolve()))
    c=small(attention_filter='binomial',hops=4); model=Encoder(c); data=TaskData(c,'train')
    result=diagnose(model,data,c)
    assert len(result['rows'])==3 and result['shared_parameter_gradient_norms']['qkv_norm']>0
    assert result['rows'][1]['attention_change_rms']>0 and not result['causal_claim']
    result=benchmark(model,data,c,repeats=2,warmup=1)
    assert [r['batch_size'] for r in result]==[1,64]
    assert all(r['forward_batch_seconds']['median']>0 for r in result)
