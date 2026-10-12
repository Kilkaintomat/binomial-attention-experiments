"""Shared augmented graph pool for a controlled sampling-distribution intervention."""
from c4_common import *
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.data import metadata,bfs,closure
import hashlib


def key(B):return hashlib.sha256(np.packbits(B).tobytes()).hexdigest().encode()


def augmented_graph(n,ell,cyclic,rng):
    assert n%2==0 and 1<=ell<n//2
    half=n//2;B=np.zeros((n,n),bool)
    def edge(a,b):B[a,b]=B[half+a,half+b]=True
    for u in range(ell):edge(u,u+1)
    B[ell,half+ell]=True
    if cyclic:
        reverse=rng.random(ell)<.5;reverse[int(rng.integers(ell))]=True
        for u in np.flatnonzero(reverse):edge(int(u+1),int(u))
    incoming=[]
    for v in range(ell+1,half):
        if ell>=2 and rng.random()<.5:
            j=int(rng.integers(1,ell));edge(j-1,v);edge(v,j+1)
            if cyclic and rng.random()<.5:edge(v,j-1);edge(j+1,v)
        else:
            parent=int(rng.choice(list(range(1,ell+1))+incoming));edge(v,parent);incoming.append(v)
    order=rng.permutation(n);inv=np.argsort(order)
    return B[np.ix_(order,order)],np.array([[inv[0],inv[ell]],[inv[half],inv[ell]]],np.int16)


def generate():
    allocation();freeze();seen=set();groups=[]
    # Old series use no n48 random/stress groups; check the actual available directories.
    for old in [BASE/'data',Path('/fair_previous/experiments/fair_v3/data'),Path('/reweight_previous/experiments/reweight_v2/data')]:
        assert (old/'manifest.json').exists()
        for p in list(old.glob('**/n48/B.npy'))+list(old.glob('**/n48_ell*/B.npy')):
            seen.update(key(B) for B in np.load(p,mmap_mode='r'))
    for split,count in [('train',1024),('validation',128)]:
        for family in FAMILIES:
            for ell in range(1,17):
                path=ROOT/'data'/split/family/('ell_'+str(ell))
                if (path/'manifest.json').exists():
                    m=read(path/'manifest.json')
                    for name,sha in m['files'].items():assert file_hash(path/name)==sha
                    seen.update(np.load(path/'adjacency_sha.npy').tolist());groups.append(m);continue
                arrays={'B':np.empty((count,48,48),bool),'pairs':np.empty((count,2,2),np.int16),
                    'distance':np.empty((count,2),np.int16),'reverse':np.empty((count,2),bool),'weak_same':np.empty((count,2),bool),
                    'indegree':np.empty((count,48),np.int16),'outdegree':np.empty((count,48),np.int16),
                    'adjacency_sha':np.empty(count,dtype='S64')}
                rejected=0
                for i in range(count):
                    attempt=0
                    while True:
                        rng=np.random.default_rng(np.random.SeedSequence([MASTER,['train','validation'].index(split),FAMILIES.index(family),ell,i,attempt]))
                        B,pairs=augmented_graph(48,ell,family=='er',rng);sha=key(B)
                        if sha not in seen:seen.add(sha);break
                        attempt+=1;rejected+=1
                        assert attempt<10000
                    meta=metadata(B,pairs)
                    assert meta['distance'].tolist()==[ell,-1] and meta['weak_same'].all() and not np.diag(B).any()
                    a,b=pairs[:,0];assert meta['indegree'][a]==meta['indegree'][b] and meta['outdegree'][a]==meta['outdegree'][b]
                    if i<32:
                        C=closure(B)
                        if family=='dag':assert not np.diag(C).any()
                        for (s,t),d in zip(pairs,meta['distance']):assert bool(C[s,t])==bool(d>=0) and bfs(B,int(s))[t]==d
                    arrays['B'][i]=B;arrays['pairs'][i]=pairs;arrays['adjacency_sha'][i]=sha
                    for name in ['distance','reverse','weak_same','indegree','outdegree']:arrays[name][i]=meta[name]
                path.mkdir(parents=True,exist_ok=True);files={}
                for name,arr in arrays.items():
                    p=path/(name+'.npy')
                    with p.with_suffix('.tmp').open('wb') as f:np.save(f,arr,allow_pickle=False)
                    p.with_suffix('.tmp').replace(p);files[p.name]=file_hash(p)
                m={'family':family,'split':split,'n':48,'ell':ell,'variant':'augmented_cyclic' if family=='er' else 'augmented_dag',
                    'graphs':count,'queries':2*count,'positive_count':count,'files':files,'source_hash':source_hash(),
                    'duplicates_rejected':rejected,'generator':'mirrored chain, incoming trees and equal-length detours'}
                write_json(path/'manifest.json',m);groups.append(m);print({'generated':str(path.relative_to(ROOT)),'graphs':count},flush=True)
    manifest={'groups':groups,'source_hash':source_hash(),'master_seed':MASTER,'train_graphs':32768,'validation_graphs':4096,
        'total_graphs':36864,'queries':73728,'exact_labeled_duplicates':0,'isomorphism_disjointness_claimed':False}
    assert sum(g['graphs'] for g in groups)==36864
    manifest['data_hash']=digest(manifest);write_json(ROOT/'data/manifest.json',manifest)
    return manifest
