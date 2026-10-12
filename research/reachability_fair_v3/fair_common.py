"""Frozen minimal comparison: equal radius, terminal-hop control, and richer tests."""
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from transport_reasoning.reachability import training as trainer
from transport_reasoning.reachability.config import Config as BaseConfig, read, write_json, digest, file_hash, source_hash as base_source_hash
from transport_reasoning.reachability.model import Encoder as BaseEncoder
from transport_reasoning.reachability.data import Dataset

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('FAIR_ROOT','experiments/fair_v3'))
BASE_ROOT = Path(os.environ.get('BASE_REACH_ROOT','/original/experiments/reachability'))
PREVIOUS_DATA = Path(os.environ.get('PREVIOUS_REWEIGHT_DATA','/reweight_previous/experiments/reweight_v2/data'))
BASE_SOURCE = 'a98bc726b261c66a552a50b3fcf5e7dc165f7ba3fe12a52a722fe377ba9a8ed8'
BASE_DATA = 'cf5ce937d591c8eea2e497f3bcef3bb111c2a7a9f5a2f6e5bfdaeb54921b0155'
VERSION = 'reachability_fair_v3'
MASTER = 2026101202
SEEDS = [11,23,37,53,71]
FAMILIES = ['er','dag']
NEW_MODELS = ['R16-one','F4-last-r4','F4-bin-r8']
BASE_MODELS = ['F4-one','F4-bin','F4-exp','F4-geo','R32-one']
LENGTHS = list(range(2,21))+[24,28,32,40,48]
VARIANTS = ['plain_dag','plain_cyclic','branch_dag','branch_cyclic']
SPECS = {
 'R16-one':dict(architecture='recurrent',filter='onehop',outer_steps=16,share_weights=True,hops=1,last_only=False),
 'F4-last-r4':dict(architecture='feedforward',filter='binomial',outer_steps=4,share_weights=False,hops=4,last_only=True),
 'F4-bin-r8':dict(architecture='feedforward',filter='binomial',outer_steps=4,share_weights=False,hops=8,last_only=False),
}
PROTOCOL = {
 'version':VERSION,'authorization':'User requested necessary fair comparisons and explicitly approved a separate limit of 10 GPU-hours.',
 'base_commit':'2a10b8a8e56c7c04012ef8cdea3e164a9f9cf048','base_source_hash':BASE_SOURCE,'training_data_hash':BASE_DATA,
 'new_models':SPECS,'reused_models':BASE_MODELS,'families':FAMILIES,'seeds':SEEDS,
 'training':{'new_finals':30,'reused_finals':50,'updates':30000,'batch_size':64,'width':128,'heads':4,
    'inner_MLP':False,'optimizer':'AdamW','weight_decay':.01,'clip':1.,'dtype':'float32','TF32':False,
    'warmup_updates':1500,'cosine_horizon':30000,'loss':'unweighted BCEWithLogits',
    'train_sizes':[16,17],'validation_sizes':[18,19],'data':'unchanged original train/validation with original sampling',
    'checkpoint':'maximum original validation BA at logit threshold0; then lower BCE; then earlier update'},
 'pilots':{'LRs':[.0003,.001],'seed':101,'confirmation_seed':103,'updates':3000,'total':18,
    'selection':'full original validation at update3000; highest BA, lowest BCE, lower LR; confirmation does not retune'},
 'comparisons':{
    'radius16':['R16-one vs F4-bin','R16-one vs F4-exp','R16-one vs F4-geo','R16-one vs F4-last-r4'],
    'radius32':['R32-one vs F4-bin-r8'],
    'same_blocks_parameters_transport_count':['F4-bin vs F4-last-r4','F4-exp vs F4-last-r4','F4-geo vs F4-last-r4'],
    'training_depth':['R32-one vs R16-one','F4-bin-r8 vs F4-bin'],
    'cost':'quality versus measured latency, throughput, memory and actual operation counts; no claim of equal cost at equal radius',
    'remaining_confound':'R models share weights; F models have independent blocks. Radius matching does not isolate recurrence or parameter sharing.'},
 'test':{'master_seed':MASTER,'far_sizes':[32,64,128],'graphs_per_family_size':500,'queries_per_graph':8,
    'stress_n':128,'stress_lengths':LENGTHS,'stress_variants':VARIANTS,'templates_per_cell':50,'queries_per_template':2,
    'expected_graphs':7800,'unique_queries':33600,'evaluation_workloads':320,'saved_prediction_entries':1728000,
    'selection':'all generated examples; no label balancing or model-score filtering for random far',
    'exact_duplicates':'exclude labeled adjacencies from v1 and reweight_v2 and within v3',
    'topology':'new random draws, not claimed isomorphically disjoint; plain stress remains a radius diagnostic',
    'branches':'mirrored auxiliary incoming trees attached to the two chains; source-target shortest path stays ell; no source access to auxiliary nodes',
    'closed':'new test model predictions only after all30 new finals and all80 best checkpoints are frozen'},
 'metrics':{'primary':'BA at sigmoid0.5 (logit0)',
    'secondary':['recall','specificity','AUROC','BCE','both stress queries correct','positive-negative logit margins'],
    'calibration':'one additional scalar threshold per checkpoint from original full validation only; maximize BA, tie closest to0, then lower threshold',
    'threshold_candidates':'0, all distinct validation logits, nextafter(max logit,+inf); prediction logit>=threshold',
    'calibration_not_used_for':'checkpoint/LR selection or test-dependent retuning',
    'margin_summary':['median','p10','p90','minimum','maximum','fraction_positive','fraction_gt_1e-6'],
    'timing':'warmup10 repeats30; n32,64,128; batch1 and64; state-norm instrumentation disabled only during timing'},
 'statistics':{'seeds':'five paired seeds; sample SD','interval':'exact3125 paired-seed bootstrap resamples',
    'claims':'exploratory, no multiplicity-adjusted significance; seeds do not replace independent graph topologies'},
 'budget':{'new_GPU_seconds_limit':36000,'includes':'GPU smoke/profile, pilots, finals, inference, failures and scheduler overhead',
    'admission':'full prespecified grid estimate with1.2reserve must fit; never silently reduce seeds or updates',
    'slot_seconds':1800,'concurrent_GPUs':1,'automatic_training_fallback':False},
}


def extension_hash():
    return digest({'base':base_source_hash(),'files':[(p.name,file_hash(p)) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.sh']]})


def freeze_protocol():
    assert base_source_hash()==BASE_SOURCE
    assert read(BASE_ROOT/'data/manifest.json')['data_hash']==BASE_DATA
    value={'protocol':PROTOCOL,'protocol_hash':digest(PROTOCOL),'source_hash':extension_hash()}
    p=OUT/'protocol.json'
    if p.exists():
        assert read(p)==value,'Frozen protocol/source changed'
    else:write_json(p,value)
    return value


@dataclass(frozen=True)
class Config(BaseConfig):
    version:str=VERSION
    last_only:bool=False

    def validate(self):
        assert self.model in SPECS and self.family in FAMILIES and self.version==VERSION
        assert all(getattr(self,k)==v for k,v in SPECS[self.model].items())
        assert self.width==128 and self.heads==4 and self.batch_size==64 and self.inner_beta==0
        assert self.task=='reachability' and self.stage in ['pilot','confirmation','final','smoke']
        assert self.planned_steps==30000 and 0<self.completion_steps<=30000
        assert self.device in ['cuda','cpu'] and not self.tf32
        assert self.weight_decay==.01 and self.clip==1 and self.warmup_fraction==.05
        if self.stage=='final':assert self.seed in SEEDS and self.completion_steps==30000
        return self


def make(family,model,seed=11,stage='final',lr=.001,**kwargs):
    steps=30000 if stage=='final' else 3000 if stage in ['pilot','confirmation'] else 40
    return Config(family=family,model=model,seed=seed,stage=stage,lr=lr,completion_steps=steps,
                  **SPECS[model],**kwargs).validate()


class Encoder(BaseEncoder):
    def __init__(self,c):
        super().__init__(c)
        if getattr(c,'last_only',False):
            for block in self.blocks:
                block.fixed_coeff.zero_()
                block.fixed_coeff[-1]=1.


def install_training_adapter():
    # Reuse the frozen optimizer/checkpoint/validation code; redirect only architecture and provenance.
    assert os.environ.get('REACH_ROOT')==str(OUT),'Training output must be isolated'
    trainer.Encoder=Encoder
    trainer.source_hash=extension_hash


def baseline_configs():
    cs=[BaseConfig(**c) for c in read(BASE_ROOT/'final_selection.json')['configs'] if c['model'] in BASE_MODELS]
    assert len(cs)==50 and {(c.family,c.model,c.seed) for c in cs}=={(f,m,s) for f in FAMILIES for m in BASE_MODELS for s in SEEDS}
    return cs


def base_directory(c):
    return BASE_ROOT/'runs'/c.stage/c.family/c.model/('seed_'+str(c.seed))/c.hash()


def load_checkpoint(c,baseline=False,device='cuda'):
    directory=base_directory(c) if baseline else c.directory()
    result=read(directory/'result.json')
    assert result['status']=='completed' and result['step']==30000
    path=directory/'checkpoint_best.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['config']==c.dict() and state['data_hash']==BASE_DATA
    assert state['source_hash']==(BASE_SOURCE if baseline else extension_hash())
    if baseline:
        frozen=read(BASE_ROOT/'frozen_checkpoints.json')[c.hash()]
        assert file_hash(path)==frozen['checkpoint_sha256'] and state['step']==frozen['step']
    model=(BaseEncoder(c) if baseline else Encoder(c)).to(device)
    model.load_state_dict(state['model'],strict=True)
    model.eval();model.requires_grad_(False)
    return model,{'config':c.dict(),'baseline':baseline,'checkpoint_sha256':file_hash(path),'step':state['step'],
                  'training_source_hash':state['source_hash']}


def learned_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def all_final_configs():
    new=[Config(**c) for c in read(OUT/'final_selection.json')['configs']]
    return [(c,False) for c in new]+[(c,True) for c in baseline_configs()]


class TestDataset(Dataset):
    def __init__(self,family,split,n=None):
        self.family,self.split=family,split;self.groups=[];self.offsets=[0]
        base=OUT/'data';self.manifest=read(base/'manifest.json')
        paths=[base/family/('n'+str(n))] if split=='far' else sorted((base/'stress').glob('*/*'))
        for path in paths:
            m=read(path/'manifest.json');a={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in path.glob('*.npy')}
            q=a['pairs'].shape[1];self.groups.append((m,a,m['graphs'],q,VERSION+'/'+str(path.relative_to(base))))
            self.offsets.append(self.offsets[-1]+m['graphs']*q)


def eval_path(c,split,n):
    return OUT/'evaluations'/c.family/c.model/('seed_'+str(c.seed))/(split+'_'+str(n or 'all')+'.json')


def workloads():return [('far',n) for n in [32,64,128]]+[('stress',None)]
