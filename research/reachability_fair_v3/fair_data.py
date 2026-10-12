"""Unseen graph draws and stress with variable structures inside the receptive field."""
import hashlib
from fair_common import *
from transport_reasoning.reachability.data import graph, sample_pairs, stress_graph, metadata, closure, bfs


def key(B):return hashlib.sha256(np.packbits(B).tobytes()).hexdigest().encode()


def branch_graph(n,ell,cyclic,rng):
    B=np.zeros((n,n),bool);offset=ell+1;base=2*(ell+1)
    for u in range(ell):B[u,u+1]=B[offset+u,offset+u+1]=True
    B[ell,offset+ell]=True
    if cyclic:
        reverse=rng.random(ell)<.5;reverse[int(rng.integers(ell))]=True
        for u in np.flatnonzero(reverse):B[u+1,u]=B[offset+u+1,offset+u]=True
    assert (n-base)%2==0
    # Each mirrored branch points INTO a chain or an older auxiliary vertex.
    # No edges leave a chain for a branch: the source cannot gain a shortcut.
    a_aux=[];b_aux=[]
    for i in range((n-base)//2):
        a=base+2*i;b=a+1
        candidates=list(range(1,ell+1))+a_aux
        j=int(rng.integers(len(candidates)))
        pa=candidates[j]
        pb=offset+pa if j<ell else b_aux[j-ell]
        B[a,pa]=B[b,pb]=True
        a_aux.append(a);b_aux.append(b)
    order=rng.permutation(n);inv=np.argsort(order)
    pairs=np.array([[inv[0],inv[ell]],[inv[offset],inv[ell]]],np.int16)
    return B[np.ix_(order,order)],pairs


def old_keys(n):
    seen=set()
    for base in [BASE_ROOT/'data',PREVIOUS_DATA]:
        assert (base/'manifest.json').exists()
        paths=list(base.glob('**/n'+str(n)+'/B.npy'))+list(base.glob('**/n'+str(n)+'_ell*/B.npy'))
        for p in paths:
            for B in np.load(p,mmap_mode='r'):seen.add(key(B))
    return seen


def group(family,n,seen,variant=None,ell=None):
    count,q=(50,2) if family=='stress' else (500,8)
    path=OUT/'data'/('stress/'+variant+'/n'+str(n)+'_ell'+str(ell) if family=='stress' else family+'/n'+str(n))
    if (path/'manifest.json').exists():
        m=read(path/'manifest.json')
        for name,sha in m['files'].items():assert file_hash(path/name)==sha
        seen.update(np.load(path/'adjacency_sha.npy').tolist());return m
    values={'B':np.empty((count,n,n),bool),'pairs':np.empty((count,q,2),np.int16),
            'distance':np.empty((count,q),np.int16),'reverse':np.empty((count,q),bool),'weak_same':np.empty((count,q),bool),
            'indegree':np.empty((count,n),np.int16),'outdegree':np.empty((count,n),np.int16),'adjacency_sha':np.empty(count,dtype='S64')}
    rejected=oracle=0;fi=['er','dag','stress'].index(family);vi=VARIANTS.index(variant) if variant else 0
    for i in range(count):
        attempt=0
        while True:
            def rng(stream):return np.random.default_rng(np.random.SeedSequence([MASTER,fi,n,ell or 0,vi,i,attempt,stream]))
            if family=='stress':
                if variant.startswith('branch'):B,pairs=branch_graph(n,ell,variant.endswith('cyclic'),rng(0))
                else:B,pairs,_=stress_graph(n,ell,'cyclic' if variant.endswith('cyclic') else 'dag',rng(0))
            else:
                B,_=graph(family,n,rng(0),rng(2));pairs=sample_pairs(n,rng(1))
            sha=key(B)
            if sha not in seen:seen.add(sha);break
            rejected+=1;attempt+=1
            if attempt>10000:raise RuntimeError('Cannot obtain new labeled adjacency')
        meta=metadata(B,pairs)
        assert not np.diag(B).any()
        if family=='stress':
            assert meta['distance'].tolist()==[ell,-1] and meta['weak_same'].all()
            s0,s1=pairs[:,0]
            assert meta['indegree'][s0]==meta['indegree'][s1] and meta['outdegree'][s0]==meta['outdegree'][s1]
            if variant.startswith('branch'):
                # More than ell+1 vertices can send information into the target.
                assert int((bfs(B.T,int(pairs[0,1]))>=0).sum())>ell+1
        if i<32:
            C=closure(B)
            if family=='dag' or (variant and variant.endswith('dag')):assert not np.diag(C).any()
            for (s,t),distance in zip(pairs,meta['distance']):
                assert bool(C[s,t])==bool(distance>=0) and bfs(B,int(s))[t]==distance;oracle+=1
        values['B'][i]=B;values['pairs'][i]=pairs;values['adjacency_sha'][i]=sha
        for k in ['distance','reverse','weak_same','indegree','outdegree']:values[k][i]=meta[k]
    path.mkdir(parents=True,exist_ok=True);files={}
    for k,v in values.items():
        p=path/(k+'.npy')
        with p.with_suffix('.tmp').open('wb') as f:np.save(f,v,allow_pickle=False)
        p.with_suffix('.tmp').replace(p);files[p.name]=file_hash(p)
    m={'family':family,'n':n,'variant':variant,'ell':ell,'graphs':count,'queries':count*q,'files':files,
       'positive_count':int((values['distance']>=0).sum()),'exact_duplicates_rejected':rejected,
       'independent_oracle_queries':oracle,'source_hash':extension_hash()}
    write_json(path/'manifest.json',m);print({'generated':str(path.relative_to(OUT)),'graphs':count},flush=True);return m


def prepare_data():
    from transport_reasoning.reachability.config import allocation
    allocation();freeze_protocol();seen={n:old_keys(n) for n in [32,64,128]};groups=[]
    for family in FAMILIES:
        for n in [32,64,128]:groups.append(group(family,n,seen[n]))
    for variant in VARIANTS:
        for ell in LENGTHS:groups.append(group('stress',128,seen[128],variant,ell))
    m={'groups':groups,'graphs':sum(x['graphs'] for x in groups),'queries':sum(x['queries'] for x in groups),
       'source_hash':extension_hash(),'master_seed':MASTER,'exact_labeled_duplicates_against_old_and_new':0,
       'isomorphic_disjointness_claimed':False,'independent_oracle_queries':sum(x['independent_oracle_queries'] for x in groups)}
    assert m['graphs']==7800 and m['queries']==33600
    m['data_hash']=digest(m);write_json(OUT/'data/manifest.json',m);return m
