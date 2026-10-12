"""Read-only posthoc diagnosis of v1 predictions; no new-series test access."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from transport_reasoning.reachability.config import allocation, file_hash, read, write_json
from transport_reasoning.reachability.data import bfs, raw_inputs
from transport_reasoning.reachability.metrics import binary_metrics

ROOT=Path('/original/experiments/reachability')
OUT=Path('/diagnosis')
MODELS=['F4-one','F4-bin','F4-exp','F4-geo','R32-one']


def summary(z):
    z=np.asarray(z,dtype=np.float64)
    return {name:float(value) for name,value in zip(['min','q10','median','q90','max'],np.quantile(z,[0,.1,.5,.9,1]))}


def main():
    allocation();manifest=read(ROOT/'data/manifest.json');distributions=[]
    for family in ['er','dag']:
        for split in ['train','validation','test_ID','test_far']:
            groups=[g for g in manifest['groups'] if g['family']==family and g['split']==split and (split!='test_far' or g['n']==128)]
            counts=Counter()
            for g in groups:counts.update({int(k):v for k,v in g['distance_counts'].items()})
            total=sum(counts.values());pos=total-counts[-1]
            distributions.append({'family':family,'split':split,'sizes':[g['n'] for g in groups],
                'queries':total,'positive_count':pos,'positive_fraction':pos/total,'negative_fraction':counts[-1]/total,
                'distance_counts':dict(sorted(counts.items())),'maximum_positive_distance':max(counts),
                'positive_distance_over8_count':sum(v for k,v in counts.items() if k>8),
                'fraction_of_positives_over8':sum(v for k,v in counts.items() if k>8)/pos})
    rows=[];random_rows=[];files=[]
    for c in read(ROOT/'final_selection.json')['configs']:
        if c['model'] not in MODELS:continue
        configs=list((ROOT/'runs/final'/c['family']/c['model']/('seed_'+str(c['seed']))).glob('*/config.json'))
        directories=[p.parent for p in configs if read(p)==c];assert len(directories)==1;directory=directories[0]
        for filename in ['fixed_test_ID','fixed_test_far_128','fixed_stress']:
            path=directory/'evaluation'/(filename+'.npz');a=np.load(path,allow_pickle=False);y=a['labels'];z=a['logits']
            metric=read(path.with_suffix('.json'));recomputed=binary_metrics(y,z)
            for field in ['tp','tn','fp','fn','count','balanced_accuracy']:assert recomputed[field]==metric[field]
            files.append({'family':c['family'],'model':c['model'],'seed':c['seed'],'evaluation':filename,'sha256':file_hash(path)})
            if filename!='fixed_stress':
                random_rows.append({'family':c['family'],'model':c['model'],'seed':c['seed'],'evaluation':filename,
                    **recomputed,'predicted_negative_fraction':float((z<0).mean())});continue
            for variant in ['dag','cyclic']:
                for ell in [2,4,8,16,32,48]:
                    ids=(a['n']==128)&(a['ell']==ell)&(a['variant']==variant)
                    yy=y[ids];zz=z[ids].astype(np.float64)
                    assert len(yy)==400 and np.array_equal(yy.reshape(-1,2),np.tile([1.,0.],(200,1)))
                    assert np.array_equal(a['graph_ids'][ids][::2],a['graph_ids'][ids][1::2])
                    pos=zz[yy==1];neg=zz[yy==0];delta=pos-neg
                    row={'family':c['family'],'model':c['model'],'seed':c['seed'],'n':128,'variant':variant,'ell':ell,
                        **binary_metrics(yy,zz),'predicted_negative_fraction':float((zz<0).mean()),
                        'positive_logits':summary(pos),'negative_logits':summary(neg),
                        'positive_probabilities':summary(1/(1+np.exp(-pos))),
                        'negative_probabilities':summary(1/(1+np.exp(-neg))),
                        'paired_logit_margin':summary(delta),'pair_margin_above_1e_6_fraction':float((delta>1e-6).mean()),
                        'pair_exactly_equal_fraction':float((delta==0).mean()),
                        'all_positive_min_minus_all_negative_max':float(pos.min()-neg.max()),
                        'positive_logit_spread':float(np.ptp(pos)),'negative_logit_spread':float(np.ptp(neg))}
                    rows.append(row)
    # On plain DAG stress, all model-visible ancestors and their original input
    # degrees can be checked in a canonical order given by distance to target.
    topology=[]
    for ell in [2,4,8,16,32,48]:
        path=ROOT/'data/stress/dag'/('n128_ell'+str(ell))
        graphs=np.load(path/'B.npy',mmap_mode='r');pairs=np.load(path/'pairs.npy',mmap_mode='r')
        signatures=[set(),set()];sizes=set();outside=[]
        for B,pair in zip(graphs,pairs):
            distance=bfs(B.T,int(pair[0,1]));anc=np.flatnonzero(distance>=0);anc=anc[np.argsort(-distance[anc])]
            sizes.add(len(anc));assert len(anc)==ell+1 and len(set(distance[anc]))==ell+1
            outside.append(int(len(B)-len(anc)))
            for query_index,(s,t) in enumerate(pair):
                x=raw_inputs([B],[(int(s),int(t))])['x'][0,anc]
                h=hashlib.sha256(B[np.ix_(anc,anc)].tobytes()+x.tobytes()).hexdigest();signatures[query_index].add(h)
        topology.append({'ell':ell,'templates':len(graphs),'ancestors_count':sorted(sizes),
            'unique_visible_positive_inputs':len(signatures[0]),'unique_visible_negative_inputs':len(signatures[1]),
            'outside_target_ancestor_set':outside[0],
            'features_include_original_graph_indegree_outdegree':True,
            'conclusion':'canonical ancestral inputs identical across templates, separately for each query label'})
        assert len(signatures[0])==len(signatures[1])==1
    out={'snapshot_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','scheduler_job':os.environ['SLURM_JOB_ID'],
        'source_sha256':file_hash(__file__),'new_inference':False,'new_training':False,'v3_test_accessed':False,
        'data_hash':manifest['data_hash'],'data_distributions':distributions,'stress_rows':rows,
        'random_rows':random_rows,'plain_dag_topology':topology,'prediction_files':files}
    write_json(OUT/'round_numbers_diagnosis.json',out)
    print(json.dumps({'status':'passed','stress_rows':len(rows),'random_rows':len(random_rows),'topology':topology}),flush=True)


if __name__=='__main__':main()
