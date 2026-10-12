"""Recompute metrics from saved graph logits in a CPU allocation.

Requires the original private run directory and dataset, mounted read-only.
Does not perform inference, train, or modify the original scientific records.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'results/reachability_final_20261012'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metrics(y, z):
    y = np.asarray(y, dtype=bool)
    z = np.asarray(z, dtype=np.float64)
    assert np.isfinite(z).all()
    pred = z >= 0
    tp, tn = int((y & pred).sum()), int((~y & ~pred).sum())
    fp, fn = int((~y & pred).sum()), int((y & ~pred).sum())
    p, n = tp + fn, tn + fp
    if not len(y):
        return dict(count=0, balanced_accuracy=None, recall=None, auroc=None, bce=None)
    auc = ap = None
    if p and n:
        # Integrate ROC with tied-score groups, independently of production rank sums.
        order = np.argsort(-z, kind='stable')
        ys, zs = y[order], z[order]
        end = np.r_[np.flatnonzero(zs[:-1] != zs[1:]), len(zs)-1]
        cp = np.r_[0, np.cumsum(ys)[end]]
        cn = np.r_[0, (end+1)-np.cumsum(ys)[end]]
        auc = float(np.sum(np.diff(cn/n) * (cp[:-1]+cp[1:])/(2*p)))
        ap = float(np.sum(np.diff(cp)/p * cp[1:]/(cp[1:]+cn[1:])))
    return dict(count=len(y), positive_count=p, negative_count=n, positive_fraction=p/len(y),
                tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp+tn)/len(y),
                balanced_accuracy=.5*(tp/p+tn/n) if p and n else None,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/p if p else None,
                specificity=tn/n if n else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                auroc=auc, average_precision=ap, bce=float(np.mean(np.logaddexp(0,z)-y*z)))


def compare(actual, saved):
    for key, value in actual.items():
        if value is None:
            assert saved.get(key) is None, (key, value, saved.get(key))
        else:
            assert np.isclose(value, saved[key], rtol=1e-9, atol=1e-11), (key, value, saved[key])


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'CPU allocation required'
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    from transport_reasoning.reachability.data import Dataset
    from transport_reasoning.reachability.config import source_hash, file_hash
    original = args.source / 'experiments/reachability'
    records = BASE / 'reachability/records'
    frozen = read(BASE / 'reachability/metadata/frozen_checkpoints.json')
    cache = {}

    def expected(family, split, n, regime):
        key = family, split, n, regime
        if key not in cache:
            limit = (50 if split == 'stress' else 500) if regime == 'extra' else None
            data = Dataset('stress' if split == 'stress' else family, split, n=n, limit_graphs=limit)
            fields = {k: [] for k in ['labels', 'graph_ids', 'query_ids', 'n', 'distance', 'weak_same', 'reverse', 'ell', 'variant']}
            for meta, a, count, q, prefix in data.groups:
                length = count*q
                fields['graph_ids'].append(np.repeat([prefix+'/'+str(i) for i in range(count)], q))
                fields['query_ids'].append(np.tile(np.arange(q), count))
                fields['n'].append(np.full(length, meta['n']))
                fields['ell'].append(np.full(length, meta.get('ell') or -1))
                fields['variant'].append(np.full(length, meta.get('variant') or ''))
                for k in ['distance', 'weak_same', 'reverse']:
                    fields[k].append(np.asarray(a[k][:count]).reshape(-1))
                fields['labels'].append((a['distance'][:count] >= 0).reshape(-1))
            cache[key] = {k: np.concatenate(v) for k, v in fields.items()}
        return cache[key]

    checked, predictions, grouped, slices_checked = [], 0, 0, 0
    checkpoints, configurations = [], set()
    for path in sorted(records.glob('final/**/evaluation/*.json')):
        result = read(path.parent.parent/'result.json')
        c = result['config']
        config_hash = path.parent.parent.name
        if config_hash not in configurations:
            configurations.add(config_hash)
            checkpoint = original/'runs'/path.parent.parent.relative_to(records)/'checkpoint_best.pt'
            actual_hash = file_hash(checkpoint)
            assert actual_hash == frozen[config_hash]['checkpoint_sha256']
            assert result['best_step'] == frozen[config_hash]['step']
            checkpoints.append(dict(config_hash=config_hash, sha256=actual_hash, step=result['best_step']))
        saved = read(path)
        assert saved['source_hash'] == source_hash() == result['source_hash']
        assert saved['data_hash'] == result['data_hash']
        assert saved['checkpoint_step'] == frozen[config_hash]['step']
        raw = original/'runs'/path.relative_to(records).with_suffix('.npz')
        reference = expected(c['family'], saved['split'], saved['n'], saved['regime'])
        with np.load(raw, allow_pickle=False) as a:
            for k, value in reference.items():
                assert np.array_equal(a[k], value), (str(path.relative_to(BASE)), k)
            y, z = a['labels'], a['logits']
            compare(metrics(y, z), saved)
            for row in saved['by_size_or_stress_cell']:
                mask = (a['n'] == row['n']) & (a['ell'] == (row['ell'] or -1)) & (a['variant'] == (row['variant'] or ''))
                compare(metrics(y[mask], z[mask]), row)
                grouped += 1
            bounds = {'1': (1,1), '2': (2,2), '3-4': (3,4), '5-8': (5,8), '9-16': (9,16), '17-32': (17,32), '>32': (33,10000)}
            for row in saved['path_length_slices']:
                name = row['slice']
                if name.startswith('positive_distance_'):
                    lo, hi = bounds[name.removeprefix('positive_distance_')]
                    mask = (a['distance'] >= lo) & (a['distance'] <= hi)
                else:
                    weak, reverse = name.removeprefix('negative_weak_').split('_reverse_')
                    mask = (a['distance'] < 0) & (a['weak_same'] == (weak == 'True')) & (a['reverse'] == (reverse == 'True'))
                compare(metrics(y[mask], z[mask]), row)
                slices_checked += 1
            predictions += len(y)
            checked.append(dict(file=str(path.relative_to(BASE)), prediction_sha256=sha(raw), count=len(y)))
        if len(checked) % 100 == 0:
            print(json.dumps(dict(evaluations_checked=len(checked), predictions=predictions)), flush=True)
    assert len(checked) == 1400 and len(configurations) == 80
    out = dict(status='passed', evaluations_checked=len(checked), saved_predictions_checked=predictions,
               checkpoints_checked=len(checkpoints), grouped_metric_rows_checked=grouped, path_slice_rows_checked=slices_checked,
               labels_and_ids_checked_against_frozen_data=True, source_hash=source_hash(),
               metrics_recomputed=['confusion matrix','accuracy','balanced accuracy','precision','recall','specificity','F1','AUROC','average precision','BCE'],
               scope='All saved fixed/extra predictions, dataset labels and metadata, checkpoint hashes and metric summaries. No new inference, training, or fresh graph-label BFS recomputation.',
               inference_replayed=False, checkpoints=checkpoints, files=checked)
    (BASE/'prediction_audit.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['files','checkpoints']}), flush=True)


if __name__ == '__main__':
    main()
