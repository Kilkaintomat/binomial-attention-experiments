"""A separately frozen, inference-only amendment to directed_reachability_v1."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from transport_reasoning.reachability.config import Config, read, write_json, digest, file_hash, root, source_hash
from transport_reasoning.reachability.data import Dataset
from transport_reasoning.reachability.model import Encoder, coefficients, fixed_parameter

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('REWEIGHT_OUT', 'experiments/reweight_v2'))
BASE_SOURCE = 'a98bc726b261c66a552a50b3fcf5e7dc165f7ba3fe12a52a722fe377ba9a8ed8'
BASE_DATA = 'cf5ce937d591c8eea2e497f3bcef3bb111c2a7a9f5a2f6e5bfdaeb54921b0155'
SEEDS = [11, 23, 37, 53, 71]
FAMILIES = ['er', 'dag']
MODELS = ['F4-bin', 'F4-exp', 'F4-geo']
HOPS = [4, 8, 16, 32]
MASTER = 2026101201
CELLS = [(n, ell) for n in [16, 32, 64, 128] for ell in [2, 4, 8, 16, 32, 48] if 2*(ell+1) <= n]
PROTOCOL = {
    'version': 'reachability_reweight_v2',
    'authorization': 'User requested testing exponential/geometric reweighting as r increases, without retraining.',
    'status': 'new hypothesis after observing v1 tests; new random test draws, same generator families',
    'base_commit': '2a10b8a8e56c7c04012ef8cdea3e164a9f9cf048',
    'base_source_hash': BASE_SOURCE, 'base_training_data_hash': BASE_DATA,
    'models': MODELS, 'families': FAMILIES, 'seeds': SEEDS, 'hops': HOPS,
    'training': 'none; all 30 existing best checkpoints frozen by v1 validation selection',
    'learned_parameters': 'unchanged; no optimizer, gradients, threshold fitting, or model selection',
    'coefficient_rules': {
        'original': 'v1 binomial; exp/geometric keep lambda/rho calibrated at train r=4',
        'matched': 'exp/geometric lambda/rho uniquely solve sum(k*c_k)=r*2**(r-1)/(2**r-1)',
        'solver': '100 bisection steps in log parameter [-40,40], float64 softmax; r=4 reuses original exactly',
        'r4': 'one common evaluation, shared by original/matched; not counted twice',
        'inner_MLP': 'excluded; plain filters only, as in the mathematical question',
    },
    'data': {
        'master_seed': MASTER, 'far_sizes': [32, 64, 128], 'far_graphs_per_size_family': 500,
        'far_queries_per_graph': 8, 'stress_cells': [list(x) for x in CELLS],
        'stress_variants': ['dag', 'cyclic'], 'stress_templates_per_cell_variant': 50,
        'stress_queries_per_template': 2, 'all_models_share_identical_test_draws': True,
        'exact_duplicates': 'reject labeled adjacencies against all same-size v1 data and new test draws',
        'isomorphism': 'no claim of isomorphic/topological disjointness; stress uses the original construction',
        'selection': 'all generated examples, no filtering by model predictions or class label',
        'expected_graphs': 4500, 'expected_unique_queries': 27000,
    },
    'evaluation': {'batch_size': 64, 'float32': True, 'TF32': False, 'threshold': 0.5,
        'primary': 'balanced accuracy', 'stress_pair_metric': 'fraction of templates with both answers correct',
        'save': 'raw logits, labels, graph/query IDs, metadata, state norms, timing, memory, actual coefficients',
        'expected_evaluations': 720, 'expected_prediction_entries': 2700000,
        'timing': 'observed forward time on the common workload, no claim of isolated latency benchmarking'},
    'comparisons': ['matched vs original at each r=8,16,32', 'matched vs own r=4', 'matched vs binomial at same r'],
    'statistics': {'summary': 'mean and sample SD over five model seeds, each cell separately',
        'intervals': 'exact 3125 resamples of paired model-seed differences; conditional on common test draws',
        'claims': 'exploratory; no multiplicity-adjusted significance claim; no best-r selection'},
    'limits': {'max_new_GPU_allocated_seconds': 7200, 'old_GPU_allocated_seconds': 73283,
        'combined_ceiling_seconds': 86400, 'slots': 4, 'seconds_per_slot': 1800,
        'concurrent_GPUs': 1, 'stop': 'no automatic budget increase or training fallback'},
}


def bundle_hash():
    return digest([(p.name, file_hash(p)) for p in sorted(HERE.iterdir()) if p.suffix in ['.py', '.sh']])


def freeze_protocol():
    value = {'protocol': PROTOCOL, 'protocol_hash': digest(PROTOCOL), 'extension_source_hash': bundle_hash()}
    path = OUT/'protocol.json'
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Frozen amendment or source changed')
    else:
        write_json(path, value)
    if source_hash() != BASE_SOURCE:
        raise RuntimeError('Original scientific source changed')
    return value


def parameter_for(kind, hops):
    if hops == 4:
        return fixed_parameter(kind)
    target = hops * 2**(hops-1) / (2**hops-1)
    k = torch.arange(1, hops+1, dtype=torch.float64)
    lo, hi = -40., 40.
    for _ in range(100):
        mid = (lo+hi)/2
        logw = k*mid-torch.lgamma(k+1) if kind == 'exponential' else (k-1)*mid
        if float((logw.softmax(0)*k).sum()) < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp((lo+hi)/2))


def settings(c):
    return [(r, rule) for r in HOPS for rule in (['original', 'matched'] if r > 4 and c.model != 'F4-bin' else ['original'])]


def weight_row(kind, hops, rule):
    p = (parameter_for(kind, hops) if rule == 'matched' else fixed_parameter(kind)) if kind != 'binomial' else None
    w = coefficients(kind, hops, p)
    k = torch.arange(1, hops+1, dtype=torch.float64)
    mean = float((k*w).sum())
    return {'filter': kind, 'r': hops, 'rule': rule, 'parameter': p, 'weights': w.tolist(),
            'mean_index': mean, 'variance_index': float(((k-mean)**2*w).sum()),
            'tail_mass_k_ge_r_half': float(w[k >= hops/2].sum())}


def apply_rule(model, hops, rule):
    model.extra_compute(hops=hops)
    if rule == 'matched':
        for block in model.blocks:
            block.fixed_coeff = coefficients(model.c.filter, hops, parameter_for(model.c.filter, hops)).to(block.fixed_coeff)
    return weight_row(model.c.filter, hops, rule)


def configs():
    all_configs = [Config(**d) for d in read(root()/'final_selection.json')['configs']]
    selected = [c for c in all_configs if c.model in MODELS]
    assert {(c.family, c.model, c.seed) for c in selected} == {(f, m, s) for f in FAMILIES for m in MODELS for s in SEEDS}
    assert len(selected) == 30
    return sorted(selected, key=lambda c: (SEEDS.index(c.seed), FAMILIES.index(c.family), MODELS.index(c.model)))


def checkpoint_record(c):
    frozen = read(root()/'frozen_checkpoints.json')[c.hash()]
    path = c.directory()/'checkpoint_best.pt'
    if file_hash(path) != frozen['checkpoint_sha256']:
        raise RuntimeError('Original checkpoint changed')
    result = read(c.directory()/'result.json')
    assert result['status'] == 'completed' and result['step'] == 30000
    return {'config': c.dict(), 'config_hash': c.hash(), **frozen}


def load_checkpoint(c, device):
    record = checkpoint_record(c)
    state = torch.load(c.directory()/'checkpoint_best.pt', map_location='cpu', weights_only=False)
    assert state['config'] == c.dict() and state['source_hash'] == BASE_SOURCE and state['data_hash'] == BASE_DATA
    assert state['step'] == record['step']
    model = Encoder(c).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, record


def learned_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class NewDataset(Dataset):
    def __init__(self, family, split, n=None):
        self.family, self.split = family, split
        self.groups, self.offsets = [], [0]
        base = OUT/'data'
        self.manifest = read(base/'manifest.json')
        paths = [base/family/('n'+str(n))] if split == 'far' else sorted((base/'stress').glob('*/*'))
        for path in paths:
            meta = read(path/'manifest.json')
            arrays = {p.stem: np.load(p, mmap_mode='r', allow_pickle=False) for p in path.glob('*.npy')}
            q = arrays['pairs'].shape[1]
            self.groups.append((meta, arrays, meta['graphs'], q, 'reweight_v2/'+str(path.relative_to(base))))
            self.offsets.append(self.offsets[-1]+meta['graphs']*q)


def workloads():
    return [('far', n) for n in [32, 64, 128]] + [('stress', None)]


def result_path(c, r, rule, split, n):
    return OUT/'evaluations'/c.family/c.model/('seed_'+str(c.seed))/f'{rule}_r{r}_{split}_{n or "all"}.json'
