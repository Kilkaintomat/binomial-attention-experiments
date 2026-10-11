"""Exact-size graphs, explicit directed oracles and separate model inputs."""
from collections import Counter, deque
import hashlib
import time
from pathlib import Path
import numpy as np
import networkx as nx
from .config import MASTER, SIZES, STRESS_CELLS, data_root, digest, file_hash, read, write_json, allocation

SPLITS = list(SIZES)


def rng_for(family, split, n, index, attempt=0, stream=0):
    return np.random.default_rng(np.random.SeedSequence([MASTER, ['er','dag','stress'].index(family),
                SPLITS.index(split) if split in SPLITS else 5, n, index, attempt, stream]))


def bfs(B, source):
    distance = np.full(len(B), -1, dtype=np.int16); distance[source] = 0; queue = deque([int(source)])
    neighbors = [np.flatnonzero(row) for row in B]
    while queue:
        u = queue.popleft()
        for v in neighbors[u]:
            if distance[v] < 0: distance[v] = distance[u] + 1; queue.append(int(v))
    return distance


def closure(B):
    out = B.copy()
    for k in range(len(B)): out |= out[:,k,None] & out[None,k,:]
    return out


def graph(family, n, graph_rng, permutation_rng):
    d0 = float(graph_rng.choice([1,1.5,2] if family == 'er' else [1,1.5,2,2.5,3]))
    B = graph_rng.random((n,n)) < d0/(n-1); np.fill_diagonal(B, False)
    if family == 'dag': B = np.triu(B, 1)
    order = permutation_rng.permutation(n)
    return B[np.ix_(order,order)], d0


def sample_pairs(n, rng):
    flat = rng.choice(n*(n-1), 8, replace=False)
    s = flat // (n-1); t = flat % (n-1); t = t+(t >= s)
    return np.stack([s,t], axis=1).astype(np.int16)


def metadata(B, pairs):
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    components = list(nx.weakly_connected_components(G)); weak = np.empty(len(B), dtype=np.int16)
    for i, vertices in enumerate(components): weak[list(vertices)] = i
    all_sources = {int(v): bfs(B, int(v)) for v in np.unique(pairs)}
    s,t = pairs.T
    return {'distance':np.array([all_sources[int(u)][v] for u,v in pairs],dtype=np.int16),
            'reverse':np.array([all_sources[int(v)][u] >= 0 for u,v in pairs]), 'weak_same':weak[s] == weak[t],
            'indegree':B.sum(0).astype(np.int16),'outdegree':B.sum(1).astype(np.int16),
            'weak_count':len(components),'strong_count':nx.number_strongly_connected_components(G),
            'wl':nx.weisfeiler_lehman_graph_hash(G, iterations=3)}


def raw_inputs(graphs, pairs, pad_to=None):
    count = len(graphs); size = max(map(len, graphs)) if pad_to is None else pad_to
    if size < max(map(len, graphs)): raise ValueError('Padding smaller than graph')
    x = np.zeros((count,size,5),np.float32)
    mask = np.repeat(np.eye(size,dtype=bool)[None],count,axis=0)
    valid = np.zeros((count,size),bool); targets = np.empty(count,np.int64)
    for i,(B,(s,t)) in enumerate(zip(graphs,pairs)):
        n = len(B)
        if s == t or not (0 <= s < n and 0 <= t < n): raise ValueError('Invalid query')
        valid[i,:n] = True; mask[i,:n,:n] |= B.T
        x[i,:n,0] = 1; x[i,s,1] = 1; x[i,t,2] = 1
        x[i,:n,3] = np.log1p(B.sum(0)); x[i,:n,4] = np.log1p(B.sum(1)); targets[i] = t
    return {'x':x, 'attention_mask':mask, 'node_valid':valid, 'target_index':targets}


def torch_inputs(inputs, device):
    import torch
    return {k:torch.as_tensor(v,device=device) for k,v in inputs.items()}


def group_path(family, split, n): return data_root() / family / split / ('n'+str(n))


def save_group(path, values, summary):
    path.mkdir(parents=True,exist_ok=True)
    files = {}
    for key,value in values.items():
        dest = path/(key+'.npy'); temp = path/(key+'.npy.tmp')
        with temp.open('wb') as f: np.save(f,value,allow_pickle=False)
        temp.replace(dest); files[dest.name] = file_hash(dest)
    summary.update(files=files, content_hash=digest(files), generator_sha256=file_hash(__file__))
    write_json(path/'manifest.json',summary)
    return summary


def generate_size(family, n, small=False):
    allocation(); started = time.perf_counter(); seen = set(); results = []
    for split,sizes in SIZES.items():
        if n not in sizes: continue
        count = min(sizes[n],32) if small else sizes[n]
        path = group_path(family,split,n)
        if (path/'manifest.json').exists():
            manifest = read(path/'manifest.json')
            if manifest['graphs'] != count or manifest['generator_sha256'] != file_hash(__file__): raise RuntimeError('Existing data differs')
            for filename,checksum in manifest['files'].items():
                if file_hash(path/filename) != checksum: raise RuntimeError('Existing data hash mismatch')
            seen.update(np.load(path/'adjacency_sha.npy',allow_pickle=False).tolist()); results.append(manifest); continue
        B_all = np.empty((count,n,n),bool); pairs_all = np.empty((count,8,2),np.int16)
        values = {'B':B_all,'pairs':pairs_all,'distance':np.empty((count,8),np.int16),
                  'reverse':np.empty((count,8),bool),'weak_same':np.empty((count,8),bool),
                  'indegree':np.empty((count,n),np.int16),'outdegree':np.empty((count,n),np.int16),
                  'weak_count':np.empty(count,np.int16),'strong_count':np.empty(count,np.int16),
                  'd0':np.empty(count,np.float32),'adjacency_sha':np.empty(count,dtype='S64'),'wl':np.empty(count,dtype='S32')}
        rejected = 0
        for i in range(count):
            attempt = 0
            while True:
                B,d0 = graph(family,n,rng_for(family,split,n,i,attempt,0),rng_for(family,split,n,i,attempt,2))
                key = hashlib.sha256(np.packbits(B).tobytes()).hexdigest().encode()
                if key not in seen: seen.add(key); break
                attempt += 1; rejected += 1
            pairs = sample_pairs(n,rng_for(family,split,n,i,attempt,1)); meta = metadata(B,pairs)
            if family == 'dag' and not nx.is_directed_acyclic_graph(nx.from_numpy_array(B,create_using=nx.DiGraph)): raise AssertionError('DAG has a cycle')
            B_all[i] = B; pairs_all[i] = pairs; values['d0'][i] = d0; values['adjacency_sha'][i] = key
            for k,v in meta.items(): values[k][i] = v
            if i % 5000 == 0: print({'family':family,'split':split,'n':n,'generated':i,'target':count},flush=True)
        distances = values['distance']; edges = B_all.sum((1,2))
        summary = {'family':family,'split':split,'n':n,'graphs':count,'queries':count*8,'small':small,
                   'positive_fraction':float((distances>=0).mean()),'distance_counts':dict(Counter(map(str,distances.ravel()))),
                   'indegree_counts':dict(Counter(map(str,values['indegree'].ravel()))),
                   'outdegree_counts':dict(Counter(map(str,values['outdegree'].ravel()))),
                   'weak_component_counts':dict(Counter(map(str,values['weak_count']))),
                   'strong_component_counts':dict(Counter(map(str,values['strong_count']))),
                   'mean_edges':float(edges.mean()),'mean_density':float(edges.mean()/(n*(n-1))),
                   'rejected_exact_graph_duplicates':rejected,'cpu_elapsed_seconds_cumulative':time.perf_counter()-started}
        results.append(save_group(path,values,summary))
    return results


def stress_graph(n, ell, variant, rng):
    B = np.zeros((n,n),bool); other = ell+1
    for u in range(ell): B[u,u+1] = True; B[other+u,other+u+1] = True
    B[ell,other+ell] = True
    if variant == 'cyclic':
        # Mirror reverse-edge choices, preserving equal source degrees.
        reverse = rng.random(ell) < .5
        reverse[int(rng.integers(ell))] = True
        for u in np.flatnonzero(reverse): B[u+1,u] = True; B[other+u+1,other+u] = True
    for v in range(2*(ell+1),n):
        parents = [2*ell+1] + list(range(2*(ell+1),v))
        B[int(rng.choice(parents)),v] = True
    structural_hash = hashlib.sha256(np.packbits(B).tobytes()).hexdigest()
    order = rng.permutation(n); inverse = np.argsort(order)
    pairs = np.array([[inverse[0],inverse[ell]],[inverse[other],inverse[ell]]],np.int16)
    return B[np.ix_(order,order)],pairs,structural_hash


def generate_stress(small=False):
    allocation(); count = 4 if small else 200; summaries=[]
    for variant in ['dag','cyclic']:
        for n,ell in STRESS_CELLS:
            path = data_root()/'stress'/variant/('n'+str(n)+'_ell'+str(ell))
            if (path/'manifest.json').exists():
                old=read(path/'manifest.json')
                if old['generator_sha256'] != file_hash(__file__) or old['graphs'] != count: raise RuntimeError('Stress data changed')
                summaries.append(old); continue
            values={'B':np.empty((count,n,n),bool),'pairs':np.empty((count,2,2),np.int16),
                    'distance':np.empty((count,2),np.int16),'reverse':np.empty((count,2),bool),
                    'weak_same':np.empty((count,2),bool),'indegree':np.empty((count,n),np.int16),
                    'outdegree':np.empty((count,n),np.int16),'weak_count':np.empty(count,np.int16),
                    'strong_count':np.empty(count,np.int16),'wl':np.empty(count,dtype='S32'),
                    'structural_hash':np.empty(count,dtype='S64')}
            for i in range(count):
                rng=rng_for('stress','stress',n,i,ell,0 if variant=='dag' else 1)
                B,pairs,template=stress_graph(n,ell,variant,rng);meta=metadata(B,pairs)
                assert meta['distance'].tolist()==[ell,-1] and bool(meta['weak_same'].all())
                a,b=pairs[:,0];assert meta['indegree'][a]==meta['indegree'][b] and meta['outdegree'][a]==meta['outdegree'][b]
                values['B'][i]=B;values['pairs'][i]=pairs;values['structural_hash'][i]=template.encode()
                for k,v in meta.items():values[k][i]=v
            summaries.append(save_group(path,values,{'family':'stress','variant':variant,'n':n,'ell':ell,'graphs':count,'queries':2*count,
                             'unique_labeled_construction_topologies':len(set(values['structural_hash'].tolist())),
                             'independent_topology_caveat':'random permutations are not independent topologies; reported template count is an upper bound on distinct isomorphism classes','small':small}))
    return summaries


def audit_data(small=False):
    allocation(); summaries=[]; independent_queries=0; exact_sets={}; iso_results=[]
    for family in ['er','dag']:
        for split,sizes in SIZES.items():
            for n,count in sizes.items():
                path=group_path(family,split,n);m=read(path/'manifest.json')
                assert m['graphs']==(min(count,32) if small else count)
                for name,sha in m['files'].items(): assert file_hash(path/name)==sha
                arrays={key:np.load(path/(key+'.npy'),mmap_mode='r') for key in ['B','pairs','distance','adjacency_sha']}
                hashes=set(arrays['adjacency_sha'].tolist());previous=exact_sets.setdefault((family,n),set())
                assert len(hashes)==m['graphs'] and not hashes & previous;previous.update(hashes)
                for i in np.linspace(0,m['graphs']-1,min(32,m['graphs']),dtype=int):
                    B=arrays['B'][i];C=closure(B);pairs=arrays['pairs'][i]
                    assert not np.diag(B).any() and len(set(map(tuple,pairs)))==8 and (pairs[:,0]!=pairs[:,1]).all()
                    for (s,t),dist in zip(pairs,arrays['distance'][i]):
                        assert (dist>=0)==bool(C[s,t]) and dist==bfs(B,int(s))[t];independent_queries+=1
                summaries.append(m)
        for n in [16,17]:
            train=group_path(family,'train',n);test=group_path(family,'test_ID',n)
            tw=np.load(train/'wl.npy');vw=np.load(test/'wl.npy');groups={}
            for i,k in enumerate(tw):groups.setdefault(bytes(k),[]).append(i)
            tb=np.load(train/'B.npy',mmap_mode='r');vb=np.load(test/'B.npy',mmap_mode='r')
            candidate=sum(len(groups.get(bytes(k),[])) for k in vw);checked=matches=0;started=time.perf_counter()
            for j,key in enumerate(vw):
                for i in groups.get(bytes(key),[]):
                    if checked>=5000 or time.perf_counter()-started>120:break
                    checked+=1;matches+=int(nx.is_isomorphic(nx.from_numpy_array(tb[i],create_using=nx.DiGraph),nx.from_numpy_array(vb[j],create_using=nx.DiGraph)))
                if checked>=5000 or time.perf_counter()-started>120:break
            iso_results.append({'family':family,'n':n,'cross_split_candidate_pairs':candidate,'exact_pairs_checked':checked,
                                'isomorphic_pairs_found':matches,'full_coverage':checked==candidate,
                                'limitation':'WL grouping alone is not proof of non-isomorphism; graph-level holdout may retain isomorphic topologies'})
    stress=[]
    for path in sorted((data_root()/'stress').glob('*/*/manifest.json')):
        m=read(path)
        for name,sha in m['files'].items():assert file_hash(path.parent/name)==sha
        stress.append(m)
    assert len(stress)==30
    manifest={'version':'directed_reachability_v1','small':small,'groups':summaries,'stress':stress,
              'generator_sha256':file_hash(__file__),'total_graphs':sum(m['graphs'] for m in summaries),
              'total_queries':sum(m['queries'] for m in summaries),'stress_graphs':sum(m['graphs'] for m in stress)}
    manifest['data_hash']=digest(manifest)
    write_json(data_root()/'manifest.json',manifest)
    result={'status':'passed','small':small,'independent_oracle_queries_checked':independent_queries,
            'exact_graph_duplicates_across_splits':0,'isomorphism_audit':iso_results,'data_hash':manifest['data_hash'],
            'full_isomorphic_disjointness_established':all(x['full_coverage'] and not x['isomorphic_pairs_found'] for x in iso_results)}
    write_json(data_root()/'DATA_AUDIT_REACHABILITY.json',result);return result


class Dataset:
    def __init__(self,family,split,n=None,limit_graphs=None):
        self.family=family;self.split=split;self.groups=[];self.offsets=[0]
        paths=([group_path(family,split,size) for size in SIZES[split] if n is None or size==n]
               if family!='stress' else sorted((data_root()/'stress').glob('*/*')))
        self.manifest=read(data_root()/'manifest.json')
        for path in paths:
            meta=read(path/'manifest.json');arrays={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in path.glob('*.npy')}
            count=min(meta['graphs'],limit_graphs) if limit_graphs else meta['graphs'];q=arrays['pairs'].shape[1]
            self.groups.append((meta,arrays,count,q,str(path.relative_to(data_root()))));self.offsets.append(self.offsets[-1]+count*q)
    def __len__(self):return self.offsets[-1]
    def rows(self,indices):
        graphs=[];pairs=[];labels=[];metadata_rows=[]
        for index in indices:
            group=int(np.searchsorted(self.offsets,int(index),side='right')-1)
            meta,a,count,q,prefix=self.groups[group];i,j=divmod(int(index)-self.offsets[group],q)
            graph_id=prefix+'/'+str(i);s,t=map(int,a['pairs'][i,j]);distance=int(a['distance'][i,j])
            graphs.append(a['B'][i]);pairs.append((s,t));labels.append(float(distance>=0))
            metadata_rows.append({'graph_id':graph_id,'query_id':j,'n':meta['n'],'s':s,'t':t,'distance':distance,
                                  'weak_same':bool(a['weak_same'][i,j]),'reverse':bool(a['reverse'][i,j]),
                                  'variant':meta.get('variant'),'ell':meta.get('ell')})
        return graphs,pairs,np.array(labels,np.float32),metadata_rows
    def batch(self,indices,device='cpu'):
        import torch
        graphs,pairs,y,metadata_rows=self.rows(indices)
        return torch_inputs(raw_inputs(graphs,pairs),device),torch.as_tensor(y,device=device),metadata_rows
