import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from transport_reasoning.config import Config as LegacyConfig, digest, write_json

SERIES = 'recurrent_filter_sudoku_v1'
BASE_COMMIT = '9ea79b57259e9a6619098279ec4274b8efb4dd13'
LEGACY_SOURCE = 'd8b8974c94d1fbca8539a07380ddefc1d9894bd45a66f19c49c38860b449e1f6'
PILOT_SEEDS = (101, 103, 107)
FINAL_SEEDS = (11, 23, 37, 53, 71)
PROJECT = Path(__file__).resolve().parents[2]


def root():
    return Path(os.environ.get('RECURRENT_OUTPUT', 'experiments/' + SERIES))


def code_hash():
    h = hashlib.sha256()
    paths = list((PROJECT / 'src').rglob('*.py')) + list((PROJECT / 'research').rglob('*.py')) + list((PROJECT / 'research').rglob('*.sh'))
    for p in sorted(paths):
        h.update(str(p.relative_to(PROJECT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


@dataclass
class Config:
    schema_version: int = 1
    series: str = SERIES
    architecture: str = 'recurrent'
    attention_filter: str = 'onehop'
    recurrent_steps: int = 16
    hops: int = 1
    weight_sharing: str = 'tied'
    layers: int = 1
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
        if self.schema_version != 1 or self.series != SERIES:
            raise ValueError('Unknown schema or series')
        if self.architecture not in ('recurrent', 'feedforward'):
            raise ValueError('Unknown architecture')
        if self.weight_sharing != ('tied' if self.architecture == 'recurrent' else 'untied'):
            raise ValueError('Architecture and sharing disagree')
        if self.attention_filter not in ('onehop', 'binomial', 'exponential', 'geometric'):
            raise ValueError('Unknown filter')
        if self.inner_beta != 0 or self.supervision != 'final_only':
            raise ValueError('Additional mechanisms require a separately versioned protocol')
        if self.task != 'sudoku' or self.size not in (4, 9) or self.support != 'graph':
            raise ValueError('Invalid task')
        if self.attention_filter == 'onehop' and self.hops != 1:
            raise ValueError('Onehop must record its actual hop count')
        if self.width % self.heads or self.width % 4:
            raise ValueError('Invalid width')
        if not 1 <= self.completion_steps <= self.planned_steps or min(self.recurrent_steps, self.hops, self.layers, self.batch_size, self.eval_every) < 1:
            raise ValueError('Invalid training limits')
        if self.overfit_count and self.stage != 'overfit':
            raise ValueError('Tiny training subsets are reserved for overfit checks')
        return self

    def dict(self): return asdict(self)
    def hash(self): return digest(self.dict())[:16]
    def data_dir(self): return Path(os.environ['RECURRENT_DATA_DIR'])
    def run_dir(self): return root() / 'runs' / self.stage / self.label() / f'seed_{self.seed}' / self.hash()
    def label(self):
        prefix = {'onehop': 'R', 'binomial': 'B', 'exponential': 'E', 'geometric': 'G'}[self.attention_filter]
        return prefix + str(self.recurrent_steps)

    def legacy_model_config(self):
        method = {'exponential': 'exponential_truncated', 'geometric': 'finite_geometric'}.get(self.attention_filter, self.attention_filter)
        return LegacyConfig(task=self.task, size=self.size, method=method, width=self.width, heads=self.heads,
                            layers=1 if self.architecture == 'recurrent' else self.layers,
                            hops=self.hops, recurrent_steps=self.recurrent_steps, seed=self.seed, inner_beta=0.)

    def operations(self):
        n = self.recurrent_steps if self.architecture == 'recurrent' else self.layers
        return {'unique_blocks': 1 if self.architecture == 'recurrent' else self.layers,
                'qkv_evaluations': n, 'attention_softmax_evaluations': n, 'outer_ffn_evaluations': n,
                'transport_matmuls': n * self.hops, 'inner_ffn_evaluations': 0}


def from_legacy(c):
    d = c.dict()
    recurrent = d['method'] == 'recurrent'
    filt = {'recurrent': 'onehop', 'exponential_truncated': 'exponential', 'finite_geometric': 'geometric'}.get(d['method'], d['method'])
    if filt == 'learned_polynomial' or d['inner_beta']:
        raise ValueError('This study has no compatible learned/inner-MLP configuration')
    return Config(architecture='recurrent' if recurrent else 'feedforward',
                  weight_sharing='tied' if recurrent else 'untied', attention_filter=filt,
                  recurrent_steps=d['recurrent_steps'], layers=d['layers'], hops=1 if filt == 'onehop' else d['hops'],
                  width=d['width'], heads=d['heads'], seed=d['seed'], stage=d['stage'], size=d['size'],
                  planned_steps=d['steps'], completion_steps=d['steps'], batch_size=d['batch_size'],
                  lr=d['lr'], weight_decay=d['weight_decay'], clip=d['clip'], warmup_fraction=d['warmup_fraction'],
                  eval_every=d['eval_every'], device=d['device'], threads=d['threads']).validate()


def load(path): return Config(**json.loads(Path(path).read_text())).validate()


def grid(include_b32=False):
    result = []
    for stage, seeds, labels, updates in [('pilot', PILOT_SEEDS, ['R16', 'B16', 'R32', 'B32'], 10000),
                                         ('final', FINAL_SEEDS, ['R16', 'B16', 'R32'] + (['B32'] if include_b32 else []), 30000)]:
        for seed in seeds:
            for label in labels:
                result.append(Config(attention_filter='binomial' if label[0] == 'B' else 'onehop',
                                     hops=4 if label[0] == 'B' else 1, recurrent_steps=int(label[1:]),
                                     seed=seed, stage=stage, completion_steps=updates).validate())
    return result
