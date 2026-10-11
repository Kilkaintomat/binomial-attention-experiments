import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import distributions
from transport_reasoning.config import write_json
from .config import root,code_hash


def run():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Real CPU allocation required')
    target=root()/'checks'; target.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,'-m','pytest','--junitxml='+str(target/'pytest.xml')])
    suites=list(ET.parse(target/'pytest.xml').getroot().iter('testsuite'))
    write_json(target/'pytest.json',{'status':'passed' if result.returncode==0 else 'failed','source_sha256':code_hash(),
               'tests':sum(int(s.get('tests',0)) for s in suites),'returncode':result.returncode})
    if result.returncode: raise RuntimeError('Tests failed; mass training blocked')
    from .audit import audit
    from .measure import smoke
    from transport_reasoning.training import environment
    audit(); smoke()
    write_json(target/'environment.json',environment())
    packages=sorted(set((d.metadata['Name'],d.version) for d in distributions() if d.metadata['Name']))
    (target/'installed_versions.txt').write_text(''.join(name+'=='+version+'\n' for name,version in packages))
    return {'status':'passed','source_sha256':code_hash()}
