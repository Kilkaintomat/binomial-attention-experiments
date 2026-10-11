"""Audit and export the completed MLP extension; private artifacts remain read-only."""
import argparse
import datetime
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from transport_reasoning.config import Config
from transport_reasoning.data import TaskData, file_hash
from transport_reasoning.models import Encoder
from transport_reasoning.training import source_hash

PUBLIC = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def independent_path(a, start, goal):
    assert np.array_equal(a, a.T) and np.diag(a).all()
    edges = a.copy(); np.fill_diagonal(edges, False)
    assert int(edges.sum()) == 2 * (len(a) - 1)
    parent = {int(start): None}; queue = [int(start)]
    for node in queue:
        for other in np.flatnonzero(edges[node]):
            other = int(other)
            if other not in parent:
                parent[other] = node; queue.append(other)
    assert len(parent) == len(a)
    result = np.zeros(len(a), dtype=bool); node = int(goal)
    while node is not None:
        result[node] = True; node = parent[node]
    return result


def sudoku_legal(pred, clues):
    grid = pred.reshape(-1, 9, 9)
    boxes = grid.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    digits = np.arange(1, 10)[None, None, :]
    return ((np.sort(grid, axis=2) == digits).all((1, 2))
            & (np.sort(grid.transpose(0, 2, 1), axis=2) == digits).all((1, 2))
            & (np.sort(boxes, axis=2) == digits).all((1, 2))
            & ((clues == 0) | (pred == clues)).all(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Audit requires a real CPU allocation'
    private = args.artifact_root.resolve()
    os.chdir(private.parents[1])
    torch.set_num_threads(4)
    protocol = read(private / 'mlp_final_protocol.json')
    configs = [Config(**c).validate() for c in read(private / 'planned_mlp_final_queue.json')]
    assert len(configs) == 20
    assert {(c.task, c.method, c.seed) for c in configs} == {
        (t, m, s) for t in ['maze', 'sudoku'] for m in ['binomial', 'exponential_truncated']
        for s in [11, 23, 37, 53, 71]}
    source = source_hash(); assert source == protocol['source_sha256']
    assert read(PUBLIC / 'results/protocols/mlp_final_protocol.json') == protocol
    datasets = {}; rng_states = {}; rows = []; exports = []
    for c in sorted(configs, key=lambda c: (c.task, c.method, c.seed)):
        folder = c.run_dir(); result = read(folder / 'result.json'); ev = read(folder / 'evaluation_test.json')
        base = Config(**protocol['baseline_configs_by_new_hash'][c.hash()])
        assert c.dict() == dict(base.dict(), inner_beta=.1, inner_width=128, max_seconds=4800)
        assert c.dict() == read(folder / 'config.json') == result['config']
        assert result['status'] == 'completed' and result['step'] == c.steps == 30000
        assert result['examples'] == 1920000 and result['config_hash'] == c.hash()
        assert result['source_sha256'] == source
        if c.task not in datasets:
            datasets[c.task] = {split: TaskData(c, split) for split in ['train', 'validation', 'test']}
            manifest = datasets[c.task]['train'].manifest
            assert manifest == protocol['data_manifests'][c.task]
            for name, sha in manifest['files'].items():
                assert file_hash(c.data_dir() / name) == sha
            if c.task == 'maze':
                d = datasets[c.task]['test'].data
                oracle = np.stack([independent_path(a, s, g) for a, s, g in zip(d['a'], d['start'], d['goal'])])
                assert np.array_equal(oracle, d['y'])
                datasets[c.task]['oracle'] = oracle
        test = datasets[c.task]['test']; train_count = len(datasets[c.task]['train'])
        assert ev['count'] == len(test) == (1000 if c.task == 'maze' else 18000)
        assert ev['split'] == 'test'
        assert result['data_hash'] == ev['data_hash'] == test.manifest['data_hash']
        curve = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert [v['step'] for v in curve] == [1] + list(range(250, 30001, 250))
        assert all(math.isfinite(v[k]) for v in curve for k in ['loss', 'solved_rate', 'train_loss'])
        selected = max(curve, key=lambda v: (v['solved_rate'], -v['loss'], -v['step']))
        best = torch.load(folder / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
        last = torch.load(folder / 'checkpoint_last.pt', map_location='cpu', weights_only=False)
        assert best['step'] == selected['step'] == ev['checkpoint_step'] and last['step'] == 30000
        for state in [best, last]:
            assert state['config'] == c.dict() and state['data_hash'] == result['data_hash']
            assert state['source_sha256'] == source
            assert all(torch.isfinite(v).all() for v in state['model'].values())
            assert all(k in state for k in ['optimizer', 'scheduler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng', 'data_rng'])
        key = (c.seed, train_count)
        if key not in rng_states:
            generator = np.random.default_rng(c.seed + 987123)
            for _ in range(30000): generator.integers(0, train_count, 64)
            rng_states[key] = generator.bit_generator.state
        assert last['data_rng'] == rng_states[key]
        assert last['scheduler']['last_epoch'] == 30000 and last['scheduler']['_last_lr'] == [0.]
        group = last['optimizer']['param_groups'][0]
        assert group['lr'] == 0 and group['weight_decay'] == .01 and tuple(group['betas']) == (.9, .999)
        paired = base.run_dir(); old_last = torch.load(paired / 'checkpoint_last.pt', map_location='cpu', weights_only=False)
        assert old_last['data_rng'] == last['data_rng']
        baseline_model, model = Encoder(base), Encoder(c)
        for name, value in baseline_model.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        assert model.parameter_count() == result['parameter_count'] == baseline_model.parameter_count() + 133120
        with np.load(folder / 'predictions_test.npz', allow_pickle=False) as saved:
            assert saved['ids'].tolist() == test.ids
            pred = saved['predictions']; data = test.data
            if c.task == 'maze':
                legal = (pred == datasets[c.task]['oracle']).all(1)
                exact = (pred == data['y']).all(1)
            else:
                legal = sudoku_legal(pred, data['puzzle']); exact = (pred == data['y']).all(1)
                blank = data['puzzle'] == 0
                assert abs(float(((pred == data['y']) & blank).sum() / blank.sum()) - ev['blank_accuracy']) < 1e-12
            assert np.array_equal(legal, saved['solved']) and np.array_equal(exact, saved['exact'])
            assert abs(float(legal.mean()) - ev['solved_rate']) < 1e-12
            assert abs(float(exact.mean()) - ev['exact_match']) < 1e-12
            sample = pred[:16].copy()
        model.load_state_dict(best['model']); model.eval()
        with torch.no_grad():
            batch = test.batch(np.arange(16), 'cpu'); logits = model(batch)
            if c.task == 'maze': replay = (logits.squeeze(-1) >= 0).numpy()
            else:
                replay = logits.argmax(-1).numpy() + 1
                replay = np.where(batch['puzzle'].numpy() != 0, batch['puzzle'].numpy(), replay)
        assert np.array_equal(replay, sample)
        thresholds = []
        for threshold in ([.9] if c.task == 'maze' else [.5, .9]):
            crossing = next((i for i in range(1, len(curve)) if curve[i-1]['solved_rate'] >= threshold and curve[i]['solved_rate'] >= threshold), None)
            record = {'threshold': threshold, 'status': 'not_reached' if crossing is None else 'confirmed'}
            if crossing is not None:
                directory = folder / ('threshold_' + str(round(threshold * 100)))
                archive = read(directory / 'archive.json')
                assert archive['status'] == 'confirmed' and archive['first_passing_step'] == curve[crossing-1]['step']
                assert archive['confirming_step'] == curve[crossing]['step']
                for filename, field in [('checkpoint_failing.pt', 'failing_step'), ('checkpoint_passing.pt', 'first_passing_step')]:
                    state = torch.load(directory / filename, map_location='cpu', weights_only=False)
                    assert state['step'] == archive[field] and state['config'] == c.dict()
                record.update(archive, both_threshold_weights_verified=True)
            thresholds.append(record)
        # Keep the earlier publication's partial first-run snapshot immutable.
        destination = PUBLIC / 'results/mlp_final/records' / c.task / c.method / c.hash() / ('seed_' + str(c.seed))
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['config.json', 'result.json', 'evaluation_test.json', 'metrics.jsonl']:
            original = folder / name; target = destination / name
            if target.exists(): assert target.read_bytes() == original.read_bytes(), 'Refuse to replace different archived record'
            else: shutil.copyfile(original, target)
            exports.append({'file': str(target.relative_to(PUBLIC)), 'sha256': file_hash(target)})
        # Explicitly expose only portable environment fields.
        environment = read(folder / 'environment.json')
        portable = {k: environment[k] for k in ['python', 'packages', 'cuda_build', 'gpu_model', 'gpu_memory_bytes', 'dtype', 'tf32', 'threads', 'backend', 'source_sha256'] if k in environment}
        write(destination / 'environment.json', portable)
        exports.append({'file': str((destination / 'environment.json').relative_to(PUBLIC)), 'sha256': file_hash(destination / 'environment.json')})
        rows.append({'task': c.task, 'method': c.method, 'seed': c.seed, 'config_hash': c.hash(), 'baseline_config_hash': base.hash(),
                     'data_hash': result['data_hash'], 'source_sha256': source, 'solved_rate': ev['solved_rate'],
                     'test_count': ev['count'], 'checkpoint_replay_examples': 16, 'checkpoint_replay_mismatches': 0,
                     'best_step': best['step'], 'completed_steps': last['step'], 'parameter_count': result['parameter_count'],
                     'batch_rng_matches_baseline': True, 'common_initialization_matches_baseline': True, 'thresholds': thresholds,
                     'private_artifact_sha256': {name: file_hash(folder/name) for name in ['checkpoint_best.pt', 'checkpoint_last.pt', 'predictions_test.npz']}})
        del model, baseline_model, best, last, old_last
        print(json.dumps({'audited': len(rows), 'task': c.task, 'method': c.method, 'seed': c.seed}), flush=True)
    budget = read(private / 'budget.json')
    allocations = [{k: a[k] for k in ['actual_seconds', 'gpu_count', 'purpose', 'status'] if k in a} for a in budget['allocations']]
    total = sum(a['actual_seconds'] * a['gpu_count'] for a in allocations)
    extension = sum(a['actual_seconds'] * a['gpu_count'] for a in allocations if a.get('purpose') == 'inner_mlp_final_v1')
    assert total == budget['used_gpu_seconds'] and total <= 24*3600
    result = {'status': 'passed', 'snapshot_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': source, 'new_completed_runs': 20, 'saved_test_predictions_revalidated': sum(r['test_count'] for r in rows),
              'checkpoint_replay_examples': 320, 'full_test_forward_repeated': False, 'common_initialization_and_batch_stream_verified': True,
              'analysis': {'kind': 'post-training descriptive paired seed analysis', 'bootstrap_seed': 20261011, 'resamples': 20000,
                           'interval': '95% percentile', 'multiplicity_correction': False, 'sample_sd_ddof': 1},
              'budget': {'allocation_seconds_mlp_extension_including_failed_attempt': extension, 'total_original_campaign_gpu_seconds': total,
                         'original_ceiling_gpu_hours': 24, 'allocations': allocations, 'new_sudoku_series_excluded': True},
              'runs': rows, 'exported_files': exports,
              'limitations': ['Complete test inference was not repeated; all saved decisions were independently revalidated.',
                              'CPU replay used the first16 test examples of every run.',
                              'Five paired training seeds; known benchmark; no parameter-matched outer-only or equal-wall-time retraining.']}
    write(PUBLIC / 'results/audit/mlp_final_publication_audit.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'new_completed_runs', 'saved_test_predictions_revalidated', 'checkpoint_replay_examples']}))


if __name__ == '__main__':
    main()
