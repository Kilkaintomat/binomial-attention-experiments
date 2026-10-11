"""Versioned task, architecture and resource specifications."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os

SERIES = 'directed_reachability_v1'
BASE = 'e9d676f3bd02a99c05913243c4590255e5a48394'
UPSTREAM = '22b880e8b00486e3d2ab916150d0138f538d5c50'
MASTER = 20261011
SEEDS = [11, 23, 37, 53, 71]
FAMILIES = ['er', 'dag']
PRIORITY = ['F4-one', 'F4-bin', 'F4-exp', 'F4-one-inner', 'F4-bin-inner',
            'F4-exp-inner', 'R32-one', 'F4-geo', 'F4-learned']
SIZES = {'train': {16: 50000, 17: 50000}, 'validation': {18: 2500, 19: 2500},
         'test_ID': {16: 2500, 17: 2500}, 'test_near': {n: 2500 for n in range(20, 28)},
         'test_far': {32: 2000, 64: 2000, 128: 2000}}
STRESS_CELLS = [(n, ell) for n in [16, 32, 64, 128] for ell in [2, 4, 8, 16, 32, 48] if 2*(ell+1) <= n]
PROJECT = Path(__file__).resolve().parents[3]


def root(): return Path(os.environ.get('REACH_ROOT', 'experiments/reachability'))
def data_root(): return Path(os.environ.get('REACH_DATA', str(root() / 'data')))
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()
def source_hash():
    paths = sorted(Path(__file__).parent.glob('*.py')) + sorted((PROJECT / 'scripts/reachability').glob('*'))
    return digest([(str(p.relative_to(PROJECT)), file_hash(p)) for p in paths if p.is_file()])
def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def allocation():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('A scheduler allocation is required')


@dataclass(frozen=True)
class Config:
    family: str = 'er'
    model: str = 'F4-one'
    seed: int = 11
    stage: str = 'final'
    lr: float = .001
    architecture: str = 'feedforward'
    filter: str = 'onehop'
    outer_steps: int = 4
    share_weights: bool = False
    hops: int = 1
    inner_beta: float = 0.
    width: int = 128
    heads: int = 4
    batch_size: int = 64
    planned_steps: int = 30000
    completion_steps: int = 30000
    weight_decay: float = .01
    clip: float = 1.
    warmup_fraction: float = .05
    tf32: bool = False
    device: str = 'cuda'
    threads: int = 4
    task: str = 'reachability'
    version: str = SERIES

    def validate(self):
        if self.family not in FAMILIES or self.model not in PRIORITY: raise ValueError('Unknown family/model')
        expected = model_spec(self.model)
        if any(getattr(self, k) != v for k, v in expected.items()): raise ValueError('Architecture fields disagree with model ID')
        if self.stage not in ['pilot', 'confirmation', 'final', 'smoke', 'overfit']: raise ValueError('Unknown stage')
        if self.task != 'reachability' or self.version != SERIES: raise ValueError('Wrong task/version')
        if self.width != 128 or self.heads != 4 or self.batch_size != 64: raise ValueError('Frozen dimensions/batch')
        if not 0 < self.completion_steps <= self.planned_steps or self.planned_steps != 30000: raise ValueError('Scheduler horizon changed')
        if self.stage == 'final' and (self.seed not in SEEDS or self.completion_steps != 30000): raise ValueError('Final protocol changed')
        return self
    def dict(self): return asdict(self)
    def hash(self): return digest(self.dict())[:16]
    def directory(self): return root() / 'runs' / self.stage / self.family / self.model / ('seed_'+str(self.seed)) / self.hash()


def model_spec(name):
    recurrent = name == 'R32-one'
    kind = name.split('-')[1]
    return dict(architecture='recurrent' if recurrent else 'feedforward', share_weights=recurrent,
                outer_steps=32 if recurrent else 4, filter={'one':'onehop','bin':'binomial','exp':'exponential','geo':'geometric','learned':'learned'}[kind],
                hops=1 if kind == 'one' else 4, inner_beta=.1 if name.endswith('-inner') else 0.)


def make(family, model, seed=11, stage='final', lr=.001, **kwargs):
    steps = 30000 if stage == 'final' else 3000 if stage in ['pilot','confirmation'] else 40
    return Config(family=family, model=model, seed=seed, stage=stage, lr=lr,
                  completion_steps=steps, **model_spec(model), **kwargs).validate()


def protocol():
    return {'version': SERIES, 'base_commit': BASE, 'upstream_commit': UPSTREAM,
            'data': {'master_seed': MASTER, 'families': ['er_exact_n_v1','dag_exact_n_v1'],
                     'split_graph_counts': SIZES, 'queries_per_graph': 8, 'query_sampling': 'uniform ordered non-self pairs without replacement',
                     'augmentation': 'fixed independent random relabeling at generation; no online augmentation',
                     'rng_streams': 'SeedSequence(master,family,split,n,graph_index,attempt,stream)',
                     'exact_duplicate_policy': 'reject any repeated labeled adjacency within family and n across all splits',
                     'isomorphism_audit': 'directed WL grouping; bounded exact isomorphism checks; partial coverage explicitly reported',
                     'er_d0': [1,1.5,2], 'dag_d0':[1,1.5,2,2.5,3], 'p':'d0/(n-1)',
                     'all_vertices_retained': True, 'labels':'directed BFS, negative distance stored as -1',
                     'features':['constant','source_marker','target_marker','log1p_indegree','log1p_outdegree'],
                     'attention_mask':'identity OR transpose(B); padded rows self-only', 'readout':'target only'},
            'models': {name: model_spec(name) for name in PRIORITY}, 'width':128, 'heads':4,
            'training': {'batch_size':64,'sampling':'uniform replacement over fixed graph-query list',
                         'optimizer':'AdamW','weight_decay':.01,'clip':1.,'dtype':'float32','TF32':False,
                         'warmup_steps':1500,'cosine_horizon':30000,'loss':'unweighted BCEWithLogits'},
            'pilots':{'learning_rates':[.0003,.001],'seed':101,'confirmation_seed':103,'steps':3000,'full_count':54,
                      'selection':'last full validation BA, then lower BCE, then lower LR',
                      'small_validation':'first 128 graphs of each validation size by manifest order; all 8 queries'},
            'finals':{'seeds':SEEDS,'steps':30000,'full_count':90,'validation_every':1000,
                      'best':'maximum validation BA, minimum BCE, earliest step', 'seed_rounds':True},
            'stress':{'cells':STRESS_CELLS,'variants':['dag','cyclic'],'templates_per_cell_variant':200,
                      'queries_per_template':2,'positive_negative_paired':True},
            'extra_compute':{'hops':[4,8,16,32],'recurrent_steps':[32,64,128],
                             'parameters':'lambda/rho stay fixed at training r=4','learned_extended':'unsupported',
                             'far_graphs_per_n':500,'stress_templates_per_cell_variant':50,
                             'subset':'first graph/template IDs in each manifest group, fixed before training'},
            'budget':{'gpu_seconds_limit':86400,'admission_reserve':1.2,'priority':PRIORITY,
                      'accounting':'actual allocated GPU seconds including checks, errors, evaluation',
                      'continuation':'retain omitted grid and budget estimate; no automatic cap increase'},
            'statistics':{'threshold':.5,'primary':'balanced_accuracy','seed_sd':'sample, ddof=1',
                          'paired_bootstrap':{'seed':20261011,'resamples':20000,'unit':'model seed'},
                          'graph_bootstrap_unit':'all eight queries of graph; both stress queries of template',
                          'claims':'exploratory; no familywise significance claim'},
            'upstream_adaptation':'exact-n query-conditioned binary reachability; not replication of published scores',
            'external_publication_authorized':False}


def freeze_protocol():
    p = root() / 'PROTOCOL_REACHABILITY.json'; value = protocol()
    if p.exists() and read(p) != json.loads(json.dumps(value)): raise RuntimeError('Protocol already frozen and differs')
    write_json(p, value)
    return value
