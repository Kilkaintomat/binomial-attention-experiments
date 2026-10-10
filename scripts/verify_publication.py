"""Validate a portable publication, without accessing private data or the scheduler API."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    commands = [
        [sys.executable, '-m', 'pytest', '--junitxml=.cache/publication_tests.xml'],
        [sys.executable, 'scripts/build_public_assets.py'],
        [sys.executable, 'scripts/check_public_snapshot.py'],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    import torch
    from check_chunked_execution import check_resume, check_dispatch
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as temporary:
        resumed_tasks = check_resume(Path(temporary))
        # check_dispatch mocks every scheduler interaction; no job is submitted.
        with __import__('unittest.mock', fromlist=['patch']).patch.dict(os.environ, {'SLURM_JOB_ID': 'publication-test'}):
            check_dispatch(Path(temporary))
    suites = ET.parse(ROOT / '.cache/publication_tests.xml').getroot()
    totals = {key: sum(int(s.get(key, 0)) for s in suites.iter('testsuite')) for key in ['tests', 'failures', 'errors', 'skipped']}
    result = {'status': 'passed', 'pytest': totals, 'off_grid_cpu_resume_bitwise': resumed_tasks,
              'portable_dispatcher_mock_test': 'passed', 'report_builder': 'passed', 'snapshot_consistency': 'passed',
              'execution': 'CPU allocation' if os.environ.get('SLURM_JOB_ID') else 'local CPU',
              'no_new_gpu_training': True, 'no_private_data_required': True}
    (ROOT / 'results/metadata/publication_tests.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
