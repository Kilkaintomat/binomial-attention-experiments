"""Run only inside a CPU Slurm allocation; no existing campaign writes."""
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time

from common import ART,SIDE,ROOT,atomic,source_hash,rows,digest


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use Slurm for checks and data preparation'
    from data import make_data,WordTokenizer
    from engine import encode
    from transformers import AutoTokenizer
    started=time.time()
    proc=subprocess.run([sys.executable,'-m','unittest','-v','test_correctness','test_scheduler'],cwd=str(SIDE/'code'),capture_output=True,text=True)
    print(proc.stdout+proc.stderr,flush=True)
    (ART/'cpu_tests.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError('CPU correctness tests failed')
    data=make_data(SIDE/'data')
    for rel,info in data['files'].items():
        if digest(SIDE/'data'/rel)!=info['sha256']:raise RuntimeError('Dataset checksum mismatch')
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-1.7B'),local_files_only=True)
    checks=[]
    for task in ['vt','phop_conditional']:
        for split in ['train','validation','test']:
            pool=rows(SIDE/'data'/task/(split+'.jsonl'))
            selected=pool[::max(1,len(pool)//32)]
            for row in selected:
                ids,ans,diag=encode(tok,row)
                assert ans[-1]==tok.eos_token_id and len(ids)<4096 and len(ans)<=8
                if task=='vt':assert diag['source_word_positions']
                checks.append({'task':task,'split':split,'tokens':len(ids),'answer_tokens':len(ans)})
    oracle=[]
    for ell in [1,2,4,8,16,32]:
        for r in [1,2,4,8,16,32,64,80,96]:
            oracle.append({'ell':ell,'r':r,'theoretical_tail':sum(math.comb(r,s) for s in range(ell,r+1))/(2**r-1)})
    atomic(ART/'oracle_reference.json',oracle)
    try:
        import matplotlib
    except ImportError:
        # Campaign-local plotting dependencies; never upgrade the shared ML venv.
        subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check',
            '--no-deps','--target',str(SIDE/'.deps'),'--report',str(ART/'plot_install.json'),
            'matplotlib==3.9.4','contourpy==1.3.2','cycler==0.12.1','fonttools==4.55.3',
            'kiwisolver==1.4.7','pyparsing==3.2.1'])
        sys.path.insert(0,str(SIDE/'.deps'))
        importlib.invalidate_caches()
        import matplotlib
    from report import oracle_plot
    oracle_plot()
    env={x:importlib.metadata.version(x) for x in ['torch','transformers','peft','numpy']}
    env['matplotlib']=matplotlib.__version__
    env.update(python=platform.python_version(),model=__import__('json').loads((ROOT/'artifacts/models.json').read_text())['1.7B'])
    atomic(ART/'environment.json',env)
    atomic(ART/'cpu_preflight.json',{'status':'passed','source_sha256':source_hash(),'seconds':time.time()-started,
        'data_manifest_sha256':digest(SIDE/'data/manifest.json'),'tokenization_checks':checks,
        'slurm_job_id':os.environ['SLURM_JOB_ID']})
    from freeze_campaign import main as freeze_campaign
    freeze_campaign()
    print('PREPARATION PASSED',flush=True)


if __name__=='__main__':main()
