"""Full CPU verification of the isolated package before GPU admission."""
import os
import subprocess
import sys
import shutil
from importlib import metadata
from pathlib import Path
import xml.etree.ElementTree as ET
from transport_reasoning.reachability.config import allocation,root,write_json,source_hash,freeze_protocol,read
from transport_reasoning.reachability.checks import resume_check
from transport_reasoning.reachability.campaign import build_grid

allocation();freeze_protocol();build_grid()
previous=root()/'checks/full_cpu.json'
if previous.exists() and read(previous)['source_hash']!=source_hash():
    archived=root()/'checks/archive'/read(previous)['source_hash'][:12];archived.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(previous,archived/'full_cpu.json')
    if (root()/'checks/resume').exists():shutil.move(str(root()/'checks/resume'),str(archived/'resume'))
result=subprocess.run([sys.executable,'-m','pytest','--junitxml='+str(root()/'checks/complete_tests.xml')])
if result.returncode:raise SystemExit(result.returncode)
resume=resume_check()
for args in [['--help'],['prepare-data','--family','er','--n','16','--dry-run']]:
    subprocess.run([sys.executable,'-m','transport_reasoning.reachability',*args],check=True)
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','--target','.cache/install_check','.'],check=True)
env=os.environ.copy();env['PYTHONPATH']='/workspace/.cache/install_check'
subprocess.run([sys.executable,'-c','import transport_reasoning.reachability.model; print("Installed package import passed")'],env=env,check=True)
names=sorted({d.metadata['Name'] for d in metadata.distributions() if d.metadata['Name']})
Path('requirements-reachability.lock').write_text('\n'.join(name+'=='+metadata.version(name) for name in names)+'\n')
xml=ET.parse(root()/'checks/complete_tests.xml').getroot();suites=list(xml.iter('testsuite'))
value={'status':'passed','source_hash':source_hash(),'tests':sum(int(s.attrib.get('tests',0)) for s in suites),
       'pytest_exit_code':0,'resume':resume,'cli_help_and_dry_run':'passed','package_install_import':'passed',
       'data_hash':read(root()/'data/manifest.json')['data_hash'],'allocation_id':os.environ['SLURM_JOB_ID']}
write_json(root()/'checks/full_cpu.json',value);print(value,flush=True)
