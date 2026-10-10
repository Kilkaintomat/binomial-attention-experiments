"""Check published evidence consistency; does not imply a new checkpoint audit."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from build_public_assets import ROOT, METHODS, TASKS, csv_rows, read, record_dir


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12), (a, b)


def main():
    import numpy as np
    source = hashlib.sha256()
    for path in sorted((ROOT / 'src').rglob('*.py')):
        source.update(path.relative_to(ROOT).as_posix().encode())
        source.update(path.read_bytes())
    manifest = read('PUBLICATION_MANIFEST.json')
    assert source.hexdigest() == manifest['training_source_sha256']
    for item in manifest['public_adapter_changes']:
        assert hashlib.sha256((ROOT / item['file']).read_bytes()).hexdigest() == item['public_sha256']
    rows = csv_rows('results.csv')
    assert len(rows) == len({r['config_hash'] for r in rows}) == 152
    assert Counter((r['stage'], r['run_status']) for r in rows) == {
        ('final', 'completed'): 60, ('mlp_pilot', 'completed'): 24, ('horizon_pilot', 'completed'): 18,
        ('overfit', 'completed'): 12, ('pilot', 'completed'): 12, ('pilot', 'paused'): 24, ('smoke', 'completed'): 2}
    final = [r for r in rows if r['stage'] == 'final']
    snapshots = { (r['task'], r['method'], int(r['seed'])): r for r in final }
    assert len(snapshots) == 60
    assert set(snapshots) == {(t, m, s) for t in TASKS for m in METHODS for s in [11, 23, 37, 53, 71]}
    evaluations_checked = 0
    for row in rows:
        folder = record_dir(row)
        config = json.loads((folder / 'config.json').read_text())
        result = json.loads((folder / 'result.json').read_text())
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
        assert config_hash == row['config_hash'] == result['config_hash']
        assert config == result['config']
        assert int(row['updates']) == result['step']
        if row['stage'] == 'overfit':
            assert row['split'] == 'train_subset'
        if row['method'] == 'recurrent':
            assert int(row['L']) == int(row['unique_blocks']) == 1
        if row['stage'] != 'final':
            continue
        assert float(row['inner_beta']) == config['inner_beta'] == 0
        assert int(row['updates']) == config['steps'] == 30000
        assert int(row['examples']) == 1920000
        assert result['source_sha256'] == row['source_sha256'] == source.hexdigest()
        evaluation = json.loads((folder / 'evaluation_test.json').read_text())
        assert evaluation['split'] == row['split'] == 'test'
        assert evaluation['count'] == (1000 if row['task'] == 'maze' else 18000)
        assert evaluation['data_hash'] == result['data_hash'] == row['data_hash']
        assert row['data_hash'] == read('results/protocols/mlp_final_protocol.json')['data_manifests'][row['task']]['data_hash']
        for key in ['solved_rate', 'exact_match', 'loss', 'inference_seconds_per_example']:
            close(row[key], evaluation[key])
        if row['task'] == 'sudoku':
            assert [int(row[k]) for k in ['actual_train_count', 'actual_validation_count', 'actual_test_count']] == [180000, 18000, 18000]
        curve = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in curve] == [1] + list(range(250, 30001, 250))
        best = max(curve, key=lambda r: (r['solved_rate'], -r['loss'], -r['step']))
        assert best['step'] == evaluation['checkpoint_step'] == int(row['filter_checkpoint_step'])
        close(best['solved_rate'], result['best_validation_solved_rate'])
        assert row['filter_metadata_source'] == 'tested_best_checkpoint'
        assert json.loads(row['filters_by_block']) == best['filters']
        evaluations_checked += evaluation['count']
    for row in csv_rows('final_aggregates.csv'):
        selected = sorted([r for r in final if (r['task'], r['method']) == (row['task'], row['method'])], key=lambda r: int(r['seed']))
        values = [float(r['solved_rate']) for r in selected]
        close(mean(values), row['mean']); close(stdev(values), row['sample_sd'])
        assert {str(r['seed']): float(r['solved_rate']) for r in selected} == json.loads(row['per_seed'])
    bootstrap_rng = np.random.default_rng(7531)
    for row in csv_rows('paired_differences.csv'):
        left, right = row['comparison'].split('-')
        values = [float(snapshots[row['task'], left, s]['solved_rate']) - float(snapshots[row['task'], right, s]['solved_rate']) for s in json.loads(row['seeds'])]
        assert values == json.loads(row['paired_values'])
        close(mean(values), row['mean_difference']); close(stdev(values), row['sample_sd'])
        interval = np.quantile(bootstrap_rng.choice(values, (10000, len(values)), replace=True).mean(1), [.025, .975])
        close(interval[0], row['bootstrap_seed_CI_low']); close(interval[1], row['bootstrap_seed_CI_high'])
    links_checked = 0
    for name in ['README.md', 'REPORT_RU.md', 'docs/EXPERIMENT_TABLES_RU.md']:
        path = ROOT / name
        body = path.read_text()
        assert '@@' not in body
        # Formula notation inside code spans is not a Markdown link.
        prose = re.sub(r'```[\s\S]*?```', '', body)
        prose = re.sub(r'`[^`\n]+`', '', prose)
        for target in re.findall(r'\]\(([^)]+)\)', prose):
            if target.startswith(('https://', 'http://', '#')):
                continue
            target = target.split('#', 1)[0]
            destination = (path.parent / target).resolve()
            # This script writes its own successful report only after every check passes.
            if destination != ROOT / 'results/metadata/publication_validation.json':
                assert destination.exists(), (name, target)
            links_checked += 1
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['.git', '.cache', '__pycache__', '.pytest_cache', '.venv', 'experiments']):
            continue
        assert p.suffix.lower() not in ['.pt', '.npz', '.pdf', '.xlsx', '.log'], p.relative_to(ROOT)
    result = {'status': 'passed', 'source_sha256': source.hexdigest(), 'archived_records_checked': len(rows),
              'final_runs_checked': len(final), 'test_examples_represented_by_checked_aggregate_metrics': evaluations_checked,
              'local_document_links_checked': links_checked,
              'scope': 'Published config hashes, training source, archived scalar evaluations, best-step selection, aggregates and links. No raw predictions or checkpoints are included; this is not a new prediction replay.'}
    (ROOT / 'results/metadata/publication_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
