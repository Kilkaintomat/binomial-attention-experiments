"""Prespecified data intervention with immutable training code and old controls."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import numpy as np
import torch
from transport_reasoning.reachability import training as trainer
from transport_reasoning.reachability.config import Config as BaseConfig, read, write_json, digest, file_hash, source_hash as base_hash
from transport_reasoning.reachability.model import Encoder, coefficients
from transport_reasoning.reachability.data import Dataset, raw_inputs, torch_inputs

ROOT=Path(os.environ.get('CURRICULUM_ROOT','/workspace/experiments/curriculum_v4'))
BASE=Path('/original/experiments/reachability')
HERE=Path(__file__).resolve().parent
BASE_SOURCE='a98bc726b261c66a552a50b3fcf5e7dc165f7ba3fe12a52a722fe377ba9a8ed8'
BASE_DATA='cf5ce937d591c8eea2e497f3bcef3bb111c2a7a9f5a2f6e5bfdaeb54921b0155'
VERSION='reachability_curriculum_v4'
MASTER=2026101301
FAMILIES=['er','dag'];SEEDS=[11,23,37,53,71]
REGIMES=['class_balanced','short_aug','length_balanced']
TRAIN_FILES=['c4_common.py','c4_data.py','c4_train.py','c4_checks.py','c4_queue.py','host.sh','container.sh']
PROTOCOL={
 'version':VERSION,'base_source_hash':BASE_SOURCE,'base_data_hash':BASE_DATA,'master_seed':MASTER,
 'new_training_count':30,'reused_controls':10,'architecture':'F4-bin','families':FAMILIES,'seeds':SEEDS,
 'regimes':{'original':'reuse original last checkpoint at30000',
   'class_balanced':'same original data, equal probability classes',
   'short_aug':'50% original class-balanced,50% paired augmented n48; positive-length distribution from original train restricted to1..8',
   'length_balanced':'same50/50 mixture and same augmented pool; uniform augmented positive length1..16'},
 'primary_contrasts':['class_balanced-original','length_balanced-short_aug'],
 'confounding':'class_balanced to short_aug changes size, structure and negative-example difficulty; not a pure size effect',
 'model':{'outer_steps':4,'hops':4,'width':128,'heads':4,'share_weights':False,'inner_MLP':False},
 'training':{'updates':30000,'batch_size':64,'lr':'original F4-bin LR per family, fixed across regimes and seeds',
   'new_LR_search':False,'optimizer':'same AdamW,weight_decay0.01,clip1,warmup1500,cosine30000,float32,TF32false',
   'checkpoint':'last at30000 primary; original validation monitoring every1000, no data-dependent extensions',
   'sampling':'fixed distribution from step1; no staged curriculum; virtual index reproducible from trainer batch RNG'},
 'pool':{'n':48,'lengths':list(range(1,17)),'graphs_per_family_length_train':1024,'graphs_per_family_length_validation':128,
   'queries_per_graph':2,'generator':'mirrored chains with incoming trees and two-step detours; optional mirrored backward edges',
   'ER_augmentation_is_not_ER':'controlled cyclic graph family distinct from original ER distribution',
   'negative_controls':'same graph/weak component; matched source degrees; no directed path',
   'split_disjointness':'reject exact labeled adjacency duplicates; no isomorphism-disjointness claim'},
 'test_plan':{'new_master_seed':2026101302,'random_families':FAMILIES,'random_sizes':[64,128,256],
   'graphs_per_random_cell':500,'queries_per_random_graph':8,'stress_n':256,
   'stress_lengths':[8,12,16,20,24,32,48,64],'stress_variants':['plain_dag','plain_cyclic','detour_dag','detour_cyclic'],
   'stress_graphs_per_cell':100,'stress_queries_per_graph':2,'hops':[4,8,16,32],
   'closed':'freeze all40 final checkpoints before new test predictions',
   'extra_compute':'fixed learned tensors; binomial coefficients recomputed for each r',
   'primary':'BA at logit0, paired-seed differences; all5seeds retained',
   'secondary':['recall','specificity','AUROC','BCE','both queries correct','paired logit margins'],
   'calibration':'separate thresholds from original validation and augmented validation n48 length1..16; no test calibration',
   'length_transfer':'lengths>16; lengths8/12/16 are size-transfer controls',
   'inference_source':'separately frozen after queue launch but before new test predictions'},
 'resources':{'requested_new_gpu_seconds':36000,'grant_required':True,'whole_grid_reserve_factor':1.2,
   'one_GPU_per_series':True,'slot_seconds':1800,'includes':'GPU checks, profiling, training and inference; actual Slurm allocation accounting'},
}


def source_hash():return digest({'base':base_hash(),'files':[(name,file_hash(HERE/name)) for name in TRAIN_FILES]})


def freeze():
    assert base_hash()==BASE_SOURCE and read(BASE/'data/manifest.json')['data_hash']==BASE_DATA
    value={'protocol':PROTOCOL,'protocol_hash':digest(PROTOCOL),'source_hash':source_hash()}
    p=ROOT/'protocol.json'
    if p.exists():assert read(p)==value,'Frozen training source/protocol changed'
    else:write_json(p,value)
    return value


def controls():
    configs=[BaseConfig(**v) for v in read(BASE/'final_selection.json')['configs'] if v['model']=='F4-bin']
    assert len(configs)==10 and {(c.family,c.seed) for c in configs}=={(f,s) for f in FAMILIES for s in SEEDS}
    for f in FAMILIES:assert len({c.lr for c in configs if c.family==f})==1
    return configs


def control_directory(c):return BASE/'runs/final'/c.family/c.model/('seed_'+str(c.seed))/c.hash()


@dataclass(frozen=True)
class Config(BaseConfig):
    model:str='F4-bin'
    filter:str='binomial'
    hops:int=4
    version:str=VERSION
    regime:str='class_balanced'

    def validate(self):
        assert self.family in FAMILIES and self.regime in REGIMES and self.version==VERSION
        assert self.model=='F4-bin' and self.filter=='binomial' and self.hops==4
        assert self.architecture=='feedforward' and self.outer_steps==4 and not self.share_weights and self.inner_beta==0
        assert self.width==128 and self.heads==4 and self.batch_size==64 and not self.tf32 and self.threads==4
        assert self.planned_steps==30000 and self.stage in ['final','smoke']
        assert self.weight_decay==.01 and self.clip==1 and self.warmup_fraction==.05
        if self.stage=='final':assert self.seed in SEEDS and self.completion_steps==30000
        return self


def make(family,regime,seed=11,stage='final',**kwargs):
    lr=next(c.lr for c in controls() if c.family==family)
    return Config(family=family,regime=regime,seed=seed,stage=stage,lr=lr,
        completion_steps=30000 if stage=='final' else 40,**kwargs).validate()


def install():
    assert os.environ['REACH_ROOT']==str(ROOT),'Isolated output root required'
    trainer.source_hash=source_hash


class PoolDataset(Dataset):
    def __init__(self,family,split):
        self.family,self.split=family,split;self.groups=[];self.offsets=[0]
        self.manifest=read(ROOT/'data/manifest.json')
        for ell in range(1,17):
            p=ROOT/'data'/split/family/('ell_'+str(ell));m=read(p/'manifest.json')
            a={q.stem:np.load(q,mmap_mode='r',allow_pickle=False) for q in p.glob('*.npy')}
            self.groups.append((m,a,m['graphs'],2,VERSION+'/'+str(p.relative_to(ROOT/'data'))))
            self.offsets.append(self.offsets[-1]+m['graphs']*2)


class SampledDataset:
    """Random virtual IDs map to a fixed sampling law without mutable extra RNG."""
    def __init__(self,family,regime):
        self.family,self.regime=family,regime;assert regime in REGIMES
        self.original=Dataset(family,'train');self.pool=PoolDataset(family,'train')
        labels=np.concatenate([(a['distance'][:count]>=0).ravel() for m,a,count,q,prefix in self.original.groups])
        self.by_class=[np.flatnonzero(~labels),np.flatnonzero(labels)]
        counts=np.zeros(16,dtype=np.float64)
        for m,a,count,q,prefix in self.original.groups:
            for ell in range(1,9):counts[ell-1]+=(a['distance'][:count]==ell).sum()
        self.short_prob=counts/counts.sum();self.cdf=np.cumsum(self.short_prob);self.cdf[-1]=1.
        self.manifest={'data_hash':digest({'original':BASE_DATA,'pool':self.pool.manifest['data_hash'],
            'regime':regime,'family':family,'sampler':'v1_virtual_2pow50_lowbits_class_origin_then_length_graph'})}

    def __len__(self):return 2**50

    def mapping(self,ids):
        ids=np.asarray(ids,dtype=np.int64);assert ((ids>=0)&(ids<len(self))).all()
        label=ids%2;aug=((ids//2)%2==1) if self.regime!='class_balanced' else np.zeros(len(ids),bool)
        word=ids//4
        target=np.empty(len(ids),np.int64);ell=np.zeros(len(ids),np.int64)
        for y in [0,1]:
            selected=(~aug)&(label==y);target[selected]=self.by_class[y][word[selected]%len(self.by_class[y])]
        if aug.any():
            selector=(word[aug]%(2**20))/(2**20)
            ll=np.searchsorted(self.cdf,selector,side='right')+1 if self.regime=='short_aug' else (selector*16).astype(np.int64)+1
            gg=(word[aug]//(2**20))%1024
            ell[aug]=ll;target[aug]=(ll-1)*2048+2*gg+(1-label[aug])
        return aug,target,label,ell

    def rows(self,ids):
        aug,target,expected,ell=self.mapping(ids);graphs=[None]*len(ids);pairs=[None]*len(ids);meta=[None]*len(ids);labels=np.empty(len(ids),np.float32)
        for selected,data in [(~aug,self.original),(aug,self.pool)]:
            positions=np.flatnonzero(selected)
            if not len(positions):continue
            g,p,y,m=data.rows(target[positions])
            for j,i in enumerate(positions):graphs[i]=g[j];pairs[i]=p[j];labels[i]=y[j];meta[i]=m[j]
        assert np.array_equal(labels,expected)
        return graphs,pairs,labels,meta

    def batch(self,ids,device='cpu'):
        graphs,pairs,y,meta=self.rows(ids)
        return torch_inputs(raw_inputs(graphs,pairs),device),torch.as_tensor(y,device=device),meta
