"""Validate archived Sudoku predictions without training or touching live jobs."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'results/updates_20261012'


def read(path):
    return json.loads(path.read_text())


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use a CPU scheduler allocation'
    torch.set_num_threads(1)
    dataset = dict(np.load(Path(os.environ['SUDOKU_AUDIT_DATA']) / 'test.npz', allow_pickle=False))
    clues, target = dataset['puzzle'], dataset['y']
    assert clues.shape == target.shape == (18000, 81)
    rows = []
    for series, variable, namespace in [('local_mask', 'SUDOKU_AUDIT_LOCAL_ROOT', 'sudoku_local_mask_v1'),
                                         ('recurrent', 'SUDOKU_AUDIT_RECURRENT_ROOT', 'recurrent_filter_sudoku_v1')]:
        private = Path(os.environ[variable]) / 'experiments' / namespace / 'runs'
        for evaluation_path in sorted((SNAPSHOT / series / 'records/final').rglob('evaluation_test.json')):
            relative = evaluation_path.parent.relative_to(SNAPSHOT / series / 'records')
            source = private / relative
            evaluation = read(evaluation_path)
            assert evaluation == read(source / 'evaluation_test.json')
            saved = dict(np.load(source / 'predictions_test.npz', allow_pickle=False))
            assert np.array_equal(saved['ids'], dataset['ids'])
            pred = saved['predictions']
            grid = pred.reshape(-1, 9, 9)
            boxes = grid.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
            digits = np.arange(1, 10)[None, None, :]
            valid = ((np.sort(grid, axis=2) == digits).all((1, 2))
                     & (np.sort(grid.transpose(0, 2, 1), axis=2) == digits).all((1, 2))
                     & (np.sort(boxes, axis=2) == digits).all((1, 2))
                     & ((clues == 0) | (pred == clues)).all(1))
            exact = (pred == target).all(1)
            blank = clues == 0
            values = {'solved_rate': float(valid.mean()), 'exact_match': float(exact.mean()),
                      'blank_accuracy': float(((pred == target) & blank).sum() / blank.sum())}
            assert np.array_equal(valid, saved['solved']) and np.array_equal(exact, saved['exact'])
            for key, value in values.items():
                assert math.isclose(value, evaluation[key], rel_tol=1e-12, abs_tol=1e-12)
            config = read(evaluation_path.parent / 'config.json')
            best = torch.load(source / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
            assert best['config'] == config and best['step'] == evaluation['checkpoint_step']
            assert best['data_hash'] == evaluation['data_hash']
            rows.append({'series': series, 'record': relative.as_posix(), 'count': len(pred),
                         'solved_count': int(valid.sum()), 'selected_checkpoint_step': best['step'],
                         'evaluation_sha256': hashlib.sha256(evaluation_path.read_bytes()).hexdigest(), **values})
            del best
    assert len(rows) == 20 and sum(r['count'] for r in rows) == 360000
    result = {'status': 'passed', 'new_final_runs': len(rows), 'test_predictions_revalidated': 360000,
              'checkpoint_metadata_checked': len(rows), 'checkpoint_inference_replayed': False,
              'scope': 'Independent Sudoku legality, exact match and blank accuracy from saved predictions; IDs and selected checkpoint metadata. Historical controls retain their earlier audit.',
              'runs': rows}
    (SNAPSHOT / 'prediction_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}))


if __name__ == '__main__':
    main()
