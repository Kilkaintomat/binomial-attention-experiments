"""Campaign-local IO. This module never imports or mutates predecessor controllers."""
import hashlib
import json
import os
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
SIDE = CODE.parent
ROOT = Path(os.environ.get('LT_PROJECT_ROOT', str(SIDE.parents[1])))
ART = SIDE/'artifacts'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,ensure_ascii=True);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def source_hash():
    return hashlib.sha256(json.dumps([(p.name,digest(p)) for p in sorted(CODE.iterdir())
        if p.suffix in ['.py','.sh','.sbatch']],sort_keys=True).encode()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]


def append(path, row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())


def freeze(path, value):
    if Path(path).exists():
        if read(path)!=value:raise RuntimeError('Immutable manifest differs: '+str(path))
    else:atomic(path,value)


def validate():
    if digest(ART/'manifest.json')!=(ART/'manifest.sha256').read_text().strip():raise RuntimeError('Manifest changed')
    plan=read(ART/'manifest.json')
    if source_hash()!=plan['source_sha256']:raise RuntimeError('Frozen source hash mismatch')
    if digest(SIDE/'PROTOCOL_RU.md')!=plan['protocol_sha256']:raise RuntimeError('Protocol changed')
    if digest(SIDE/'data/manifest.json')!=plan['data_manifest_sha256']:raise RuntimeError('Data manifest changed')
    for rel,sha in read(ART/'protected_files.json').items():
        if digest(ROOT/rel)!=sha:raise RuntimeError('Protected predecessor source changed: '+rel)
    return plan
