"""Small oracle/model checks and independent CPU generation shards."""
import os
import sys
import subprocess
from pathlib import Path
from importlib.metadata import distributions
from transport_reasoning.reachability.config import SIZES,root,write_json,freeze_protocol,allocation
from transport_reasoning.reachability.data import generate_size,generate_stress,audit_data

allocation();mode=sys.argv[1]
if mode=='small':
    result=subprocess.run([sys.executable,'-m','pytest','tests/test_reachability_data.py','tests/test_reachability_model.py',
                           '--junitxml=experiments/reachability/checks/data_model_tests.xml'])
    if result.returncode:raise SystemExit(result.returncode)
    os.environ['REACH_DATA']=str(root()/'small_data')
    for family in ['er','dag']:
        for n in sorted({n for sizes in SIZES.values() for n in sizes}):generate_size(family,n,small=True)
    generate_stress(small=True);audit=audit_data(small=True)
    write_json(root()/'checks/small_data.json',audit)
    Path('requirements-reachability.lock').write_text('\n'.join(sorted(set(d.metadata['Name']+'=='+d.version for d in distributions() if d.metadata['Name'])))+'\n')
    print('Small data and model checks passed.',flush=True)
elif mode in ['generate','generate-worker']:
    groups=[(f,n) for f in ['er','dag'] for n in sorted({n for sizes in SIZES.values() for n in sizes})]
    index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    indices=range(index,len(groups)+1,4) if mode=='generate-worker' else [index]
    for item in indices:
        if item==len(groups):generate_stress()
        else:generate_size(*groups[item])
elif mode=='audit':
    freeze_protocol();print(audit_data(),flush=True)
else:raise ValueError('Unknown CPU data job')
