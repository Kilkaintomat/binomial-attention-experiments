import json
import math
import time
from dataclasses import replace
import numpy as np
import torch
from transport_reasoning.config import write_json
from transport_reasoning.data import TaskData
from transport_reasoning.evaluation import evaluate_model, loss_for, sync, sudoku_valid
from transport_reasoning.training import atomic_checkpoint, environment
from .config import Config, code_hash, root
from .model import Encoder
from .training import setup, optimization, train


def distribution(values):
    return {'median': float(np.median(values)), 'p25': float(np.quantile(values, .25)),
            'p75': float(np.quantile(values, .75)), 'minimum': min(values), 'maximum': max(values), 'repeats': len(values)}


@torch.no_grad()
def benchmark(model, data, c, repeats=30, warmup=10):
    model.eval(); records = []
    for batch_size in [1, 64]:
        ids = np.arange(batch_size); batch = data.batch(ids, c.device)
        for _ in range(warmup): model(batch)
        sync(c.device)
        forward = []; full = []
        for _ in range(repeats):
            sync(c.device); tick = time.perf_counter(); model(batch); sync(c.device)
            forward.append(time.perf_counter() - tick)
            sync(c.device); tick = time.perf_counter()
            prepared = data.batch(ids, c.device); logits = model(prepared); sync(c.device)
            puzzle = prepared['puzzle'].cpu().numpy()
            pred = logits.argmax(-1).cpu().numpy() + 1
            pred = np.where(puzzle != 0, puzzle, pred)
            for p, q in zip(pred, puzzle): sudoku_valid(p, q)
            sync(c.device); full.append(time.perf_counter() - tick)
        records.append({'batch_size': batch_size, 'warmup_forwards': warmup, 'precision': 'float32',
                        'forward_batch_seconds': distribution(forward), 'end_to_end_batch_seconds': distribution(full),
                        'forward_examples_per_second': batch_size / float(np.median(forward)),
                        'end_to_end_examples_per_second': batch_size / float(np.median(full))})
    return records


def profile(device='cuda'):
    if device != 'cuda': raise ValueError('GPU budget admission requires a real allocated GPU profile')
    all_results = {}
    for label in ['R16', 'B16', 'R32', 'B32']:
        c = Config(stage='profile', seed=109, recurrent_steps=int(label[1:]),
                   attention_filter='binomial' if label[0] == 'B' else 'onehop', hops=4 if label[0] == 'B' else 1)
        setup(c); data = TaskData(c, 'train'); validation = TaskData(c, 'validation')
        model = Encoder(c).to(device); opt, scheduler = optimization(model, c)
        generator = np.random.default_rng(c.seed + 987123)
        times = []; torch.cuda.reset_peak_memory_stats()
        for step in range(70):
            sync(device); tick = time.perf_counter()
            batch = data.batch(generator.integers(0, len(data), c.batch_size), device)
            opt.zero_grad(set_to_none=True); loss = loss_for(model(batch), batch, c)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip, error_if_nonfinite=True)
            opt.step(); scheduler.step(); sync(device)
            if step >= 10: times.append(time.perf_counter() - tick)
        peak = torch.cuda.max_memory_allocated(); validations = []; checkpoints = []
        target = root() / 'profile'; target.mkdir(parents=True, exist_ok=True)
        for repetition in range(3):
            tick = time.perf_counter(); evaluate_model(model, validation, c, device, target / (label + '_predictions.npz')); sync(device)
            validations.append(time.perf_counter() - tick)
            tick = time.perf_counter()
            atomic_checkpoint(target / (label + '_checkpoint.pt'), {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict()})
            checkpoints.append(time.perf_counter() - tick)
        entry = {'label': label, 'config': c.dict(), 'parameters': model.parameter_count(), 'operations': c.operations(),
                 'training_step_seconds': distribution(times), 'full_validation_18000_seconds': distribution(validations),
                 'single_checkpoint_seconds': distribution(checkpoints), 'peak_training_memory_bytes': peak,
                 'training_examples_per_second': 64 / float(np.median(times)), 'inference': benchmark(model, validation, c)}
        # Three writes per evaluation conservatively cover last, best, and evaluated copies.
        entry['estimated_pilot_seconds'] = 10000 * np.median(times) + 41 * (np.median(validations) + 3 * np.median(checkpoints))
        entry['estimated_final_seconds'] = 30000 * np.median(times) + 122 * np.median(validations) + 121 * 3 * np.median(checkpoints)
        all_results[label] = entry
        write_json(root() / 'profile/gpu_profile.json', {'source_sha256': code_hash(), 'environment': environment(), 'configurations': all_results})
        print(json.dumps({'profile_completed': label, 'estimated_final_seconds': entry['estimated_final_seconds']}), flush=True)
    checks = json.loads((root() / 'checks/legacy_compatibility.json').read_text())
    reuse = checks['status'] == 'passed'
    pilot = sum(x['estimated_pilot_seconds'] for x in all_results.values()) * 3
    final = sum(all_results[k]['estimated_final_seconds'] for k in ['R16', 'B16'] + ([] if reuse else ['R32'])) * 5
    # Measured inference/diagnostics and process startup reserve in addition to 20% margin.
    base = pilot + final + 600
    optional = 5 * all_results['B32']['estimated_final_seconds']
    estimate = {'source_sha256': code_hash(), 'measured_on_allocated_gpu': True, 'reuse_historical_R32_quality': reuse,
                'mandatory_pilot_runs': 12, 'mandatory_final_results': 15, 'new_training_runs': 22 if reuse else 27,
                'mandatory_seconds_before_margin': float(base), 'mandatory_seconds_with_20_percent_margin': float(base * 1.2),
                'optional_B32_additional_seconds_with_20_percent_margin': float(optional * 1.2),
                'profile_allocation_charged_separately': True, 'historical_training_time_not_current_speed_evidence': True}
    write_json(root() / 'profile/budget_estimate.json', estimate)
    return estimate


def overfit(label_only=None):
    outcomes = []
    for label in ([label_only] if label_only else ['R16', 'B16', 'R32', 'B32']):
        c = Config(stage='overfit', seed=211, recurrent_steps=int(label[1:]), attention_filter='binomial' if label[0] == 'B' else 'onehop',
                   hops=4 if label[0] == 'B' else 1, device='cpu', overfit_count=64, planned_steps=1000, completion_steps=1000, eval_every=100)
        checkpoint = c.run_dir() / 'checkpoint_last.pt'
        result = train(c, resume=checkpoint if checkpoint.exists() else None, deadline=time.perf_counter()+3300)
        curve = [json.loads(s) for s in (c.run_dir() / 'metrics.jsonl').read_text().splitlines()]
        outcome = {'label': label, 'config_hash': c.hash(), 'completed_steps': result['completed_steps'],
                   'best_train_subset_solved': result['best_validation_solved_rate'], 'initial_loss': curve[0]['loss'],
                   'final_loss': curve[-1]['loss'], 'passes_95_percent_memorization': result['best_validation_solved_rate'] >= .95}
        outcomes.append(outcome)
        write_json(root() / ('checks/overfit_' + label + '.json'), {'status': 'passed' if outcome['passes_95_percent_memorization'] else 'failed',
                   'source_sha256': code_hash(), 'result': outcome})
        print(json.dumps(outcome), flush=True)
    if label_only:
        paths = [root() / ('checks/overfit_' + name + '.json') for name in ['R16','B16','R32','B32']]
        if not all(path.exists() for path in paths): return {'status': 'partial', 'result': outcomes[0]}
        records = [json.loads(path.read_text()) for path in paths]
        if any(r['source_sha256'] != code_hash() for r in records): raise RuntimeError('Overfit source mismatch')
        outcomes = [r['result'] for r in records]
    status = 'passed' if all(r['passes_95_percent_memorization'] for r in outcomes) else 'failed'
    result = {'status': status, 'source_sha256': code_hash(), 'seed': 211, 'criterion': .95,
              'split': 'first_64_training_puzzles', 'results': outcomes, 'generalization_claim': False}
    write_json(root() / 'checks/overfit.json', result)
    if status != 'passed': raise RuntimeError('Overfit criterion failed; mass queue remains blocked')
    return result


def diagnose(model, data, c):
    """Small validation-only mechanism observations, not a causal attribution."""
    batch = data.batch(np.arange(16), c.device); model.eval(); rows = []; previous = [None]
    def hook(block, inputs, output):
        x, mask = inputs; b, n, d = x.shape; hd = d // c.heads
        q, k, v = block.qkv(block.ln1(x)).reshape(b, n, 3, c.heads, hd).permute(2, 0, 3, 1, 4).unbind(0)
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        z = v; y = torch.zeros_like(v)
        for weight in block.coeff(): z = a @ z; y = y + weight * z
        attention = block.out(y.transpose(1, 2).reshape(b, n, d)); intermediate = x + attention
        ffn = block.ff(block.ln2(intermediate))
        normalized = torch.nn.functional.normalize(output, dim=-1)
        readout = model.readout(model.norm(output))
        prediction = readout.argmax(-1) + 1; prediction = torch.where(batch['puzzle'] != 0, batch['puzzle'], prediction)
        solved = np.mean([sudoku_valid(p, q) for p, q in zip(prediction.cpu().numpy(), batch['puzzle'].cpu().numpy())])
        rows.append({'iteration': len(rows) + 1, 'attention_entropy_nats': float(-(a * a.clamp_min(1e-30).log()).sum(-1).mean()),
                     'attention_change_rms': None if previous[0] is None else float((a - previous[0]).square().mean().sqrt()),
                     'state_update_relative_norm': float((output - x).norm() / x.norm().clamp_min(1e-20)),
                     'token_variance': float(output.var(dim=1, unbiased=False).mean()),
                     'mean_pairwise_token_cosine': float(((normalized @ normalized.transpose(-1, -2)).sum((1, 2)) - n).div(n * (n - 1)).mean()),
                     'transport_to_residual_norm': float(attention.norm() / x.norm().clamp_min(1e-20)),
                     'ffn_to_residual_norm': float(ffn.norm() / intermediate.norm().clamp_min(1e-20)),
                     'intermediate_readout_solved_rate': float(solved)})
        previous[0] = a.detach().clone()
    handle = model.blocks[0].register_forward_hook(hook)
    try:
        with torch.no_grad(): model(batch)
    finally: handle.remove()
    model.zero_grad(set_to_none=True); loss_for(model(batch), batch, c).backward()
    gradients = {'qkv_norm': float(model.blocks[0].qkv.weight.grad.norm()),
                 'ffn_norm': float(torch.sqrt(sum(p.grad.square().sum() for p in model.blocks[0].ff.parameters())))}
    model.zero_grad(set_to_none=True)
    return {'split': 'validation_first16', 'rows': rows, 'shared_parameter_gradient_norms': gradients,
            'readout_trained_on_final_iteration_only': True, 'causal_claim': False}
