"""Audit and rebuild the immutable 2026-10-12 08:24 UTC validation snapshot.

Run inside a CPU scheduler allocation. No training or evaluation is performed.
An optional --source directory verifies the published scientific source files.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / 'results/reachability_update_20261012'
SOURCE = 'a98bc726b261c66a552a50b3fcf5e7dc165f7ba3fe12a52a722fe377ba9a8ed8'
DATA = 'cf5ce937d591c8eea2e497f3bcef3bb111c2a7a9f5a2f6e5bfdaeb54921b0155'
MODELS = ['F4-one', 'F4-bin', 'F4-exp', 'F4-one-inner', 'F4-bin-inner', 'F4-exp-inner', 'R32-one', 'F4-geo']
SEEDS = [11, 23, 37, 53, 71]


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def validate_metric(m):
    tp, tn, fp, fn = (m[k] for k in ['tp', 'tn', 'fp', 'fn'])
    assert min(tp, tn, fp, fn) >= 0
    assert tp + tn + fp + fn == m['count']
    assert tp + fn == m['positive_count'] and tn + fp == m['negative_count']
    close(m['accuracy'], (tp + tn) / m['count'])
    close(m['positive_fraction'], (tp + fn) / m['count'])
    close(m['balanced_accuracy'], .5 * (tp / (tp + fn) + tn / (tn + fp)))
    close(m['f1'], 2 * tp / (2 * tp + fp + fn))
    for key in ['bce', 'balanced_accuracy', 'f1', 'auroc', 'average_precision']:
        assert math.isfinite(m[key])


def table(name, rows):
    with (SNAP / 'tables' / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['family'], row['model']].append(row)
    result = []
    for (family, model), group in sorted(groups.items()):
        row = dict(family=family, model=model, seed_count=len(group), seeds=','.join(str(r['seed']) for r in sorted(group, key=lambda r: r['seed'])))
        for metric in ['balanced_accuracy', 'accuracy', 'bce', 'training_seconds', 'elapsed_seconds']:
            values = [r[metric] for r in group]
            row[metric + '_mean'] = mean(values)
            row[metric + '_sample_sd'] = stdev(values) if len(values) > 1 else None
        result.append(row)
    return result


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use a CPU scheduler allocation'
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    for name in ['tables', 'figures']:
        (SNAP / name).mkdir(exist_ok=True)
    snapshot = read(SNAP / 'snapshot.json')
    state = snapshot['series']['reachability']
    for item in state['files']:
        data = (SNAP / 'reachability' / item['file']).read_bytes()
        assert len(data) == item['bytes']
        assert hashlib.sha256(data).hexdigest() == item['sha256']
    provenance = read(SNAP / 'source_provenance.json')
    if args.source:
        for item in provenance['files']:
            assert hashlib.sha256((args.source / item['file']).read_bytes()).hexdigest() == item['sha256'], item['file']
    records = SNAP / 'reachability/records'
    finals, pilots, partial, statuses, curves = [], [], [], collections.Counter(), 0
    for path in sorted(records.rglob('result.json')):
        r = read(path)
        c = read(path.parent / 'config.json')
        assert c == r['config'] and digest(c)[:16] == path.parent.name
        assert r['source_hash'] == SOURCE and r['data_hash'] == DATA
        assert c['planned_steps'] == 30000 and c['batch_size'] == 64
        assert r['examples_seen'] == r['step'] * c['batch_size']
        statuses[c['stage'] + '_' + r['status']] += 1
        if r['status'] != 'completed':
            assert r['status'] == 'paused'
            partial.append(dict(family=c['family'], model=c['model'], seed=c['seed'], saved_result_step=r['step'], status=r['status'], interpretation='previous slot result; live training may have advanced'))
            continue
        assert r['step'] == c['completion_steps'] == (30000 if c['stage'] == 'final' else 3000)
        points = [json.loads(line) for line in (path.parent / 'validation.jsonl').read_text().splitlines()]
        assert [m['step'] for m in points] == list(range(1000 if c['stage'] == 'final' else 500, r['step'] + 1, 1000 if c['stage'] == 'final' else 500))
        for metric in points:
            validate_metric(metric)
            assert metric['count'] == (40000 if metric['full_validation'] else 2048)
        curves += len(points)
        assert points[-1] == r['final_validation'] and points[-1]['full_validation']
        best = max(points, key=lambda m: (m['balanced_accuracy'], -m['bce']))
        assert r['best_step'] == best['step']
        close(r['best_validation_ba'], best['balanced_accuracy'])
        m = points[-1]
        row = dict(family=c['family'], model=c['model'], seed=c['seed'], stage=c['stage'], steps=r['step'], lr=c['lr'],
                   split='validation', n='18,19', **{k:m[k] for k in ['balanced_accuracy', 'accuracy', 'bce', 'f1', 'auroc', 'positive_fraction', 'count']},
                   best_step=r['best_step'], best_validation_ba=r['best_validation_ba'], parameters=r['operations']['parameters'],
                   **r['times'], config_hash=path.parent.name, record=str(path.parent.relative_to(ROOT)))
        (finals if c['stage'] == 'final' else pilots).append(row)
    assert dict(statuses) == state['completed_counts']
    assert len(finals) == 77 and len(pilots) == 48 and len(partial) == 1
    assert not list(records.glob('**/evaluation/*.json'))
    assert not (SNAP / 'reachability/metadata/frozen_checkpoints.json').exists()
    selection = read(SNAP / 'reachability/metadata/final_selection.json')
    assert len(selection['configs']) == 80 and selection['before_test']
    actual = {(r['family'], r['model'], r['seed']) for r in finals}
    expected = {(f, m, s) for f in ['er', 'dag'] for m in MODELS for s in SEEDS}
    assert len(actual) == len(finals) and actual <= expected
    missing = [dict(family=f, model=m, seed=s) for f, m, s in sorted(expected - actual)]
    common = [r for r in finals if r['seed'] != 71]
    assert {(r['family'], r['model'], r['seed']) for r in common} == {(f, m, s) for f in ['er', 'dag'] for m in MODELS for s in SEEDS[:-1]}
    # Verify selected learning rates directly from full pilot endpoints.
    for family in ['er', 'dag']:
        for model in MODELS:
            candidates = [r for r in pilots if r['family'] == family and r['model'] == model and r['stage'] == 'pilot']
            assert len(candidates) == 2
            chosen = max(candidates, key=lambda r: (r['balanced_accuracy'], -r['bce'], -r['lr']))['lr']
            assert all(r['lr'] == chosen for r in finals + pilots if r['family'] == family and r['model'] == model and r['stage'] != 'pilot')
    table('final_validation_per_seed.csv', finals)
    table('validation_all_completed.csv', aggregate(finals))
    table('validation_common_four_seeds.csv', aggregate(common))
    table('pilot_confirmation_validation.csv', pilots)
    table('unfinished_final_training.csv', missing)
    comparisons = []
    for family in ['er', 'dag']:
        for left, right in [('F4-bin', 'F4-one'), ('F4-exp', 'F4-bin'), ('F4-geo', 'F4-bin'), ('F4-bin-inner', 'F4-bin'), ('F4-bin-inner', 'F4-one-inner'), ('R32-one', 'F4-bin')]:
            values = {m: {r['seed']:r['balanced_accuracy'] for r in common if r['family'] == family and r['model'] == m} for m in [left, right]}
            differences = [100 * (values[left][s] - values[right][s]) for s in SEEDS[:-1]]
            comparisons.append(dict(family=family, comparison=left+' minus '+right, seeds='11,23,37,53', paired_seeds=4,
                                    difference_mean_pp=mean(differences), difference_sample_sd_pp=stdev(differences), minimum_pp=min(differences), maximum_pp=max(differences), split='validation; descriptive, no test inference'))
    table('paired_validation_differences.csv', comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
    for ax, family in zip(axes, ['er', 'dag']):
        for i, model in enumerate(MODELS):
            values = [100*r['balanced_accuracy'] for r in common if r['family'] == family and r['model'] == model]
            ax.bar(i, mean(values), color='#2563eb', alpha=.55)
            ax.scatter([i+(j-1.5)*.065 for j in range(4)], values, color='#111827', s=18, zorder=3)
        ax.set(xticks=range(8), xticklabels=MODELS, ylim=(90,100.5), ylabel='Validation balanced accuracy (%)', title=family.upper())
        ax.tick_params(axis='x', rotation=60)
    fig.suptitle('Graph reachability: validation only, n=18/19, final step 30,000\nCommon seeds 11, 23, 37, 53; dots = individual seeds')
    fig.tight_layout()
    fig.savefig(SNAP / 'figures/validation_common_four_seeds.png', dpi=160)
    plt.close(fig)
    summary = dict(status='passed', snapshot_utc=snapshot['capture_finished_utc'], copied_files_checked=len(state['files']),
                   completed_run_records_checked=len(finals)+len(pilots), validation_points_checked=curves,
                   finals_completed=len(finals), pilots_and_confirmations_completed=len(pilots), partial_records=partial, unfinished=missing,
                   test_evaluations=0, common_seeds=SEEDS[:-1], source_hash=SOURCE, data_hash=DATA,
                   source_files_checked=len(provenance['files']) if args.source else 0,
                   gpu_hours_accounted=state['gpu_seconds_accounted']/3600, active_allocation_excluded=True,
                   scope='File hashes; configs, source/data identity; all completed validation confusion matrices and checkpoint selection; LR choices; complete common seed coverage; aggregates. No new inference or prediction-level replay.',
                   training_modified=False)
    (SNAP / 'validation_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
