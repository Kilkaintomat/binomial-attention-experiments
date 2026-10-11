"""Rebuild the dated publication tables and figures from completed run records."""
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'results/updates_20261012'
TABLES = SNAPSHOT / 'tables'
FIGURES = SNAPSHOT / 'figures'
EXPECTED_SOURCE = {'local_mask': '647975e8907be8e5358b13e6258434d393a8028536156415e72e1dd350af5860',
                   'recurrent': 'c3b99115ee9b0a49b0a238609eec282ee722bf7bd09645d2d6013c20ea011ca7',
                   'reachability': 'a98bc726b261c66a552a50b3fcf5e7dc165f7ba3fe12a52a722fe377ba9a8ed8'}
MODELS = ['F4-one', 'F4-bin', 'F4-exp', 'F4-one-inner', 'F4-bin-inner', 'F4-exp-inner', 'R32-one', 'F4-geo']


def read(path): return json.loads(path.read_text())
def curve(path): return [json.loads(line) for line in path.read_text().splitlines()]
def close(a, b): assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def table(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (TABLES / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)


def aggregate(rows, keys, metrics):
    groups = collections.defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        row = dict(zip(keys, key)); row.update(seed_count=len(group), seeds=','.join(str(x['seed']) for x in sorted(group, key=lambda x: x['seed'])))
        for metric in metrics:
            values = [x[metric] for x in group]
            row[metric + '_mean'] = mean(values)
            row[metric + '_sample_sd'] = stdev(values) if len(values) > 1 else None
        output.append(row)
    return output


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Build and verify inside a CPU allocation'
    TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    snapshot = read(SNAPSHOT / 'snapshot.json')
    copied = 0
    for series, metadata in snapshot['series'].items():
        for item in metadata['files']:
            data = (SNAPSHOT / series / item['file']).read_bytes()
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
            copied += 1
    rows = []; pilots = []; checked = 0
    for series in EXPECTED_SOURCE:
        for path in sorted((SNAPSHOT / series / 'records').rglob('result.json')):
            result = read(path); config = read(path.parent / 'config.json'); stage = config['stage']
            assert config == result['config'] and result['status'] == 'completed'
            digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
            assert path.parent.name == digest
            source = result.get('source_hash', result.get('source_sha256'))
            assert source == EXPECTED_SOURCE[series]
            steps = result.get('step', result.get('completed_steps'))
            assert steps == config['completion_steps']
            checked += 1
            if series == 'reachability':
                points = curve(path.parent / 'validation.jsonl')
                metric = result['final_validation']
                assert metric['full_validation'] and metric['count'] == 40000
                assert points[-1] == metric
                best = max(points, key=lambda x: (x['balanced_accuracy'], -x['bce']))
                assert best['step'] == result['best_step']
                base = {'series': series, 'family': config['family'], 'model': config['model'], 'seed': config['seed'],
                        'stage': stage, 'steps': steps, 'lr': config['lr'], 'split': 'validation',
                        'balanced_accuracy': metric['balanced_accuracy'], 'bce': metric['bce'],
                        'f1': metric['f1'], 'auroc': metric['auroc'], 'count': metric['count'],
                        'parameters': result['operations']['parameters'], **result['times']}
            else:
                model = path.relative_to(SNAPSHOT / series / 'records').parts[-4] if series == 'local_mask' else path.relative_to(SNAPSHOT / series / 'records').parts[1]
                points = curve(path.parent / 'metrics.jsonl')
                assert [p['step'] for p in points] == [1] + list(range(250, steps + 1, 250))
                best = max(points, key=lambda x: (x['solved_rate'], -x['loss'], -x['step']))
                close(best['solved_rate'], result['best_validation_solved_rate'])
                metric = read(path.parent / 'evaluation_test.json') if stage == 'final' else result['last_validation']
                if stage == 'final':
                    assert metric['split'] == 'test' and metric['count'] == 18000
                    assert metric['checkpoint_step'] == best['step']
                    assert metric['source_sha256'] == source and metric['data_hash'] == result['data_hash']
                base = {'series': series, 'model': model, 'support': config['support'], 'seed': config['seed'],
                        'stage': stage, 'steps': steps, 'split': 'test' if stage == 'final' else 'validation',
                        'solved_rate': metric['solved_rate'], 'blank_accuracy': metric['blank_accuracy'], 'loss': metric['loss'],
                        'count': metric['count'], 'parameters': result['parameter_count'],
                        'training_seconds': result['timings']['training'], 'elapsed_seconds': result['timings']['elapsed'],
                        'checkpoint_step': metric.get('checkpoint_step')}
            base.update(config_hash=digest, source_hash=source, data_hash=result['data_hash'], record=str(path.parent.relative_to(ROOT)))
            (rows if stage == 'final' else pilots).append(base)
    assert len(rows) == 53 and len(pilots) == 78
    local = [r for r in rows if r['series'] == 'local_mask']
    recurrent = [r for r in rows if r['series'] == 'recurrent']
    graph = [r for r in rows if r['series'] == 'reachability']
    assert len(local) == 15 and len(recurrent) == 5 and len(graph) == 33
    assert {(r['model'], r['seed']) for r in local} == {(m, s) for m in ['F4-onehop', 'F4-binomial', 'R32-onehop'] for s in [11, 23, 37, 53, 71]}
    common = [r for r in graph if r['seed'] in [11, 23]]
    assert {(r['family'], r['model'], r['seed']) for r in common} == {(f, m, s) for f in ['er', 'dag'] for m in MODELS for s in [11, 23]}
    assert not list((SNAPSHOT / 'reachability/records').glob('**/evaluation/*.json'))
    stats = ['solved_rate', 'blank_accuracy', 'training_seconds', 'elapsed_seconds']
    local_agg = aggregate(local, ['model'], stats)
    recurrent_agg = aggregate(recurrent, ['model'], stats)
    graph_agg = aggregate(common, ['family', 'model'], ['balanced_accuracy', 'bce', 'training_seconds', 'elapsed_seconds'])
    table('local_mask_test_per_seed.csv', local); table('local_mask_test_aggregates.csv', local_agg)
    table('recurrent_test_per_seed.csv', recurrent); table('recurrent_test_aggregates.csv', recurrent_agg)
    table('reachability_final_validation_per_seed.csv', graph)
    table('reachability_validation_common_seeds.csv', graph_agg)
    table('pilots_validation.csv', pilots)
    for series in ['local_mask', 'recurrent']:
        runtime = list(csv.DictReader((SNAPSHOT / series / 'runtime_report/final_seeds.csv').open()))
        subset = local if series == 'local_mask' else recurrent
        for r in subset:
            matching = [x for x in runtime if x['config_hash'] == r['config_hash']]
            assert len(matching) == 1
            for key in ['solved_rate', 'blank_accuracy']: close(float(matching[0][key]), r[key])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ['F4-onehop', 'F4-binomial', 'R32-onehop']
    for ax, key, label in zip(axes, ['blank_accuracy', 'solved_rate'], ['Correct blank cells (%)', 'Solved puzzles (%)']):
        for i, model in enumerate(names):
            values = [100 * r[key] for r in local if r['model'] == model]
            ax.bar(i, mean(values), color=['#64748b', '#2563eb', '#ea580c'][i], alpha=.6)
            ax.scatter([i + (j - 2) * .07 for j in range(5)], values, color='black', s=18, zorder=3)
        ax.set(xticks=range(3), xticklabels=['One-hop', 'Binomial', 'Recurrent32'], ylabel=label, ylim=(0, 100 if key == 'blank_accuracy' else 1.6))
    fig.suptitle('Sudoku grid4: final test, five seeds; dots show individual seeds')
    fig.tight_layout(); fig.savefig(FIGURES / 'local_mask_test.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, family in zip(axes, ['er', 'dag']):
        for i, model in enumerate(MODELS):
            values = [100 * r['balanced_accuracy'] for r in common if r['family'] == family and r['model'] == model]
            ax.bar(i, mean(values), color='#2563eb', alpha=.55)
            ax.scatter([i-.06, i+.06], values, color='black', s=15)
        ax.set(xticks=range(len(MODELS)), xticklabels=MODELS, ylim=(90, 100.5), title=family.upper(), ylabel='Validation balanced accuracy (%)')
        ax.tick_params(axis='x', rotation=65)
    fig.suptitle('Reachability: validation only, n=18/19, final step; seeds 11 and 23')
    fig.tight_layout(); fig.savefig(FIGURES / 'reachability_validation.png', dpi=170); plt.close(fig)
    audit = read(SNAPSHOT / 'prediction_audit.json')
    assert audit['status'] == 'passed' and audit['test_predictions_revalidated'] == 360000
    result = {'status': 'passed', 'copied_file_hashes_checked': copied, 'completed_run_records_checked': checked,
              'new_sudoku_test_runs': 20, 'reachability_final_training_runs': 33, 'reachability_test_evaluations': 0,
              'same_seed_graph_summary': [11, 23], 'no_new_training': True,
              'scope': 'Raw Sudoku prediction audit, config/source/data hashes, checkpoint selection, curves, scalar tables and identical seed coverage; reachability validation is not a test result.'}
    (SNAPSHOT / 'publication_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
