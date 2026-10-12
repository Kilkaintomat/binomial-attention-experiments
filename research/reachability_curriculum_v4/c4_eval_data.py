"""Separate frozen evaluation source and unseen test graphs; no model selection."""
from c4_common import *
from c4_data import augmented_graph,key
from transport_reasoning.reachability.data import graph,sample_pairs,stress_graph,metadata,bfs,closure
from transport_reasoning.reachability.config import allocation

TEST=PROTOCOL['test_plan']
EVAL_FILES=['c4_eval_data.py','c4_evaluate.py']


def evaluation_hash():return digest({'training_source':source_hash(),'evaluation_files':[(p,file_hash(HERE/p)) for p in EVAL_FILES]})


def freeze_evaluation():
    freeze()
    value={'plan':TEST,'source_hash':evaluation_hash(),'training_source_hash':source_hash(),
        'thresholds':'one per checkpoint per r per validation set; max BA, tie closest0 then smaller threshold',
        'timing':'common architecture benchmark on original ER seed11; r4/8/16/32,n64/128/256,batch1/64,warmup10,repeats30',
        'paired_intervals':'all3125 resamples of5 paired seeds; exploratory, no multiplicity-adjusted claim',
        'expected_workloads':640,'expected_test_prediction_entries':2944000,
        'expected_calibration_prediction_entries':7055360,'no_test_based_changes':True}
    p=ROOT/'evaluation_protocol.json'
    if p.exists():assert read(p)==value,'Frozen evaluation source changed'
    else:write_json(p,value)
    return value


def old_keys(n):
    seen=set()
    for base in [BASE/'data',Path('/reweight_previous/experiments/reweight_v2/data'),Path('/fair_previous/experiments/fair_v3/data')]:
        for p in list(base.glob('**/n'+str(n)+'/B.npy'))+list(base.glob('**/n'+str(n)+'_ell*/B.npy')):
            seen.update(key(B) for B in np.load(p,mmap_mode='r'))
    return seen


def generate_group(family,n,seen,variant=None,ell=None):
    stress=family=='stress';count,q=(100,2) if stress else (500,8)
    path=ROOT/'test_data'/('stress/'+variant+'/n'+str(n)+'_ell'+str(ell) if stress else family+'/n'+str(n))
    if (path/'manifest.json').exists():
        m=read(path/'manifest.json')
        for name,sha in m['files'].items():assert file_hash(path/name)==sha
        seen.update(np.load(path/'adjacency_sha.npy').tolist());return m
    arrays={'B':np.empty((count,n,n),bool),'pairs':np.empty((count,q,2),np.int16),
        'distance':np.empty((count,q),np.int16),'reverse':np.empty((count,q),bool),'weak_same':np.empty((count,q),bool),
        'indegree':np.empty((count,n),np.int16),'outdegree':np.empty((count,n),np.int16),'adjacency_sha':np.empty(count,dtype='S64')}
    rejected=0;fi=['er','dag','stress'].index(family);vi=TEST['stress_variants'].index(variant) if stress else 0
    for i in range(count):
        attempt=0
        while True:
            def rng(stream):return np.random.default_rng(np.random.SeedSequence([TEST['new_master_seed'],fi,n,ell or 0,vi,i,attempt,stream]))
            if stress:
                if variant.startswith('detour'):B,pairs=augmented_graph(n,ell,variant.endswith('cyclic'),rng(0))
                else:B,pairs,_=stress_graph(n,ell,'cyclic' if variant.endswith('cyclic') else 'dag',rng(0))
            else:B,_=graph(family,n,rng(0),rng(2));pairs=sample_pairs(n,rng(1))
            sha=key(B)
            if sha not in seen:seen.add(sha);break
            attempt+=1;rejected+=1;assert attempt<10000
        meta=metadata(B,pairs);assert not np.diag(B).any()
        if stress:
            assert meta['distance'].tolist()==[ell,-1] and meta['weak_same'].all()
            a,b=pairs[:,0];assert meta['indegree'][a]==meta['indegree'][b] and meta['outdegree'][a]==meta['outdegree'][b]
        if i<32:
            C=closure(B)
            if family=='dag' or (stress and variant.endswith('dag')):assert not np.diag(C).any()
            for (s,t),d in zip(pairs,meta['distance']):assert bool(C[s,t])==bool(d>=0) and bfs(B,int(s))[t]==d
        arrays['B'][i]=B;arrays['pairs'][i]=pairs;arrays['adjacency_sha'][i]=sha
        for k in ['distance','reverse','weak_same','indegree','outdegree']:arrays[k][i]=meta[k]
    path.mkdir(parents=True,exist_ok=True);files={}
    for name,a in arrays.items():
        p=path/(name+'.npy')
        with p.with_suffix('.tmp').open('wb') as f:np.save(f,a,allow_pickle=False)
        p.with_suffix('.tmp').replace(p);files[p.name]=file_hash(p)
    m={'family':family,'n':n,'variant':variant,'ell':ell,'graphs':count,'queries':count*q,
        'positive_count':int((arrays['distance']>=0).sum()),'files':files,'exact_duplicates_rejected':rejected,'source_hash':evaluation_hash()}
    write_json(path/'manifest.json',m);print({'generated':str(path.relative_to(ROOT)),'graphs':count},flush=True);return m


def prepare():
    allocation();freeze_evaluation();seen={n:old_keys(n) for n in TEST['random_sizes']};groups=[]
    for family in FAMILIES:
        for n in TEST['random_sizes']:groups.append(generate_group(family,n,seen[n]))
    for variant in TEST['stress_variants']:
        for ell in TEST['stress_lengths']:groups.append(generate_group('stress',256,seen[256],variant,ell))
    m={'groups':groups,'graphs':sum(g['graphs'] for g in groups),'queries':sum(g['queries'] for g in groups),
        'source_hash':evaluation_hash(),'master_seed':TEST['new_master_seed'],'exact_labeled_duplicates_against_previous_and_current':0,
        'isomorphism_disjointness_claimed':False}
    assert m['graphs']==6200 and m['queries']==30400
    m['data_hash']=digest(m);write_json(ROOT/'test_data/manifest.json',m)
    result={'status':'passed','data_hash':m['data_hash'],'evaluation_source_hash':evaluation_hash(),'training_source_hash':source_hash(),
        'scheduler_job':os.environ['SLURM_JOB_ID'],'test_predictions_performed':False}
    write_json(ROOT/'test_ready.json',result);print(json.dumps(result),flush=True);return result


class TestDataset(Dataset):
    def __init__(self,family,split,n=None):
        self.family,self.split=family,split;self.groups=[];self.offsets=[0]
        self.manifest=read(ROOT/'test_data/manifest.json');base=ROOT/'test_data'
        paths=[base/family/('n'+str(n))] if split=='far' else sorted((base/'stress').glob('*/*'))
        for path in paths:
            m=read(path/'manifest.json');a={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in path.glob('*.npy')}
            q=a['pairs'].shape[1];self.groups.append((m,a,m['graphs'],q,VERSION+'/test/'+str(path.relative_to(base))))
            self.offsets.append(self.offsets[-1]+m['graphs']*q)


def workloads():return [('far',n) for n in TEST['random_sizes']]+[('stress',None)]
