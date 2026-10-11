import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from transport_reasoning.config import Config as LegacyConfig, digest

SERIES = 'sudoku_local_mask_v1'
BASE_COMMIT = '9ea79b57259e9a6619098279ec4274b8efb4dd13'
LEGACY_SOURCE = 'd8b8974c94d1fbca8539a07380ddefc1d9894bd45a66f19c49c38860b449e1f6'
PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
PILOT_SEEDS = (101, 103, 107)
FINAL_SEEDS = (11, 23, 37, 53, 71)
LABELS = ('F4-onehop', 'F4-binomial', 'R32-onehop')
SUPPORTS = ('graph', 'grid4')


def root(): return Path(os.environ.get('LOCAL_MASK_OUTPUT', 'experiments/' + SERIES))


def hash_paths(paths):
    h = hashlib.sha256()
    for p in sorted(paths): h.update(str(p.relative_to(PROJECT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def code_hash():
    """Scientific computation hash; presentation and scheduler have separate hashes."""
    paths = list((PROJECT / 'src').rglob('*.py'))
    paths += [PACKAGE / (n + '.py') for n in ['config','data','model','training','measure']]
    return hash_paths(paths)


def snapshot_hash():
    return hash_paths(list((PROJECT/'src').rglob('*.py')) + list((PROJECT/'research').rglob('*.py')) + list((PROJECT/'research').rglob('*.sh')))


@dataclass
class Config:
    schema_version: int = 1
    series: str = SERIES
    architecture: str = 'feedforward'
    attention_filter: str = 'onehop'
    recurrent_steps: int = 1
    hops: int = 1
    weight_sharing: str = 'untied'
    layers: int = 4
    inner_beta: float = 0.
    supervision: str = 'final_only'
    task: str = 'sudoku'
    size: int = 9
    support: str = 'graph'
    width: int = 128
    heads: int = 4
    seed: int = 101
    stage: str = 'pilot'
    planned_steps: int = 30000
    completion_steps: int = 10000
    eval_every: int = 250
    batch_size: int = 64
    lr: float = .001
    weight_decay: float = .01
    clip: float = 1.
    warmup_fraction: float = .05
    device: str = 'cuda'
    threads: int = 4
    overfit_count: int = 0
    actual_train_count: int = 180000
    actual_validation_count: int = 18000
    actual_test_count: int = 18000

    def validate(self):
        if self.schema_version != 1 or self.series != SERIES: raise ValueError('Unknown series')
        if self.task != 'sudoku' or self.support not in SUPPORTS or self.size not in (4,9): raise ValueError('Invalid Sudoku mask')
        if self.attention_filter not in ('onehop','binomial'): raise ValueError('Only two filters in this ablation')
        if self.architecture not in ('feedforward','recurrent'): raise ValueError('Invalid architecture')
        if self.weight_sharing != ('tied' if self.architecture == 'recurrent' else 'untied'): raise ValueError('Sharing mismatch')
        if self.architecture == 'recurrent' and self.attention_filter != 'onehop': raise ValueError('Recurrent binomial belongs to a separate series')
        if self.inner_beta != 0 or self.supervision != 'final_only': raise ValueError('Additional mechanisms are excluded')
        if self.attention_filter == 'onehop' and self.hops != 1: raise ValueError('Record the actual one-hop count')
        if min(self.hops,self.layers,self.recurrent_steps,self.batch_size,self.eval_every,self.width,self.heads) < 1: raise ValueError('Invalid dimensions')
        if self.width % self.heads or self.width % 4: raise ValueError('Invalid width')
        if not 1 <= self.completion_steps <= self.planned_steps: raise ValueError('Invalid schedule')
        if self.lr <= 0 or self.weight_decay < 0 or self.clip <= 0 or not 0 < self.warmup_fraction < 1: raise ValueError('Invalid optimizer')
        if self.overfit_count and self.stage != 'overfit': raise ValueError('Subsets are reserved for overfit')
        return self

    def dict(self): return asdict(self)
    def hash(self): return digest(self.dict())[:16]
    def data_dir(self): return Path(os.environ['LOCAL_MASK_DATA_DIR'])
    def mask_label(self): return 'relational' if self.support == 'graph' else 'grid4'
    def label(self): return 'R32-onehop' if self.architecture == 'recurrent' else 'F4-' + self.attention_filter
    def key(self): return self.mask_label() + '/' + self.label()
    def run_dir(self): return root()/'runs'/self.stage/self.mask_label()/self.label()/f'seed_{self.seed}'/self.hash()
    def legacy_model_config(self):
        return LegacyConfig(task=self.task,size=self.size,method='recurrent' if self.architecture=='recurrent' else self.attention_filter,
                            width=self.width,heads=self.heads,layers=self.layers,hops=self.hops,recurrent_steps=self.recurrent_steps,seed=self.seed,inner_beta=0.)
    def operations(self):
        n = self.recurrent_steps if self.architecture=='recurrent' else self.layers
        return {'unique_blocks':1 if self.architecture=='recurrent' else self.layers,'qkv_evaluations':n,
                'attention_softmax_evaluations':n,'outer_ffn_evaluations':n,'transport_matmuls':n*self.hops,
                'maximum_sequential_transports':n*self.hops,'inner_ffn_evaluations':0}


def make(label, support='graph', seed=101, stage='pilot', **kwargs):
    if label not in LABELS: raise ValueError('Unknown model label')
    recurrent = label == 'R32-onehop'
    c = Config(architecture='recurrent' if recurrent else 'feedforward', attention_filter='binomial' if label=='F4-binomial' else 'onehop',
               recurrent_steps=32 if recurrent else 1,hops=4 if label=='F4-binomial' else 1,
               weight_sharing='tied' if recurrent else 'untied',layers=1 if recurrent else 4,support=support,seed=seed,stage=stage,
               completion_steps=30000 if stage=='final' else 10000)
    return replace(c, **kwargs).validate()


def from_legacy(old):
    label = {'onehop':'F4-onehop','binomial':'F4-binomial','recurrent':'R32-onehop'}[old.method]
    return make(label, old.support, old.seed, old.stage, size=old.size,width=old.width,heads=old.heads,
                planned_steps=old.steps,completion_steps=old.steps,eval_every=old.eval_every,batch_size=old.batch_size,
                lr=old.lr,weight_decay=old.weight_decay,clip=old.clip,warmup_fraction=old.warmup_fraction,device=old.device,threads=old.threads)


def load(path): return Config(**json.loads(Path(path).read_text())).validate()


def grid():
    return [make(label,support,seed,stage) for stage,seeds in [('pilot',PILOT_SEEDS),('final',FINAL_SEEDS)]
            for seed in seeds for label in LABELS for support in SUPPORTS]
