"""Supplement the bounded data audit without changing any graphs or labels."""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import networkx as nx
from transport_reasoning.reachability.config import allocation,data_root,root,read,write_json,file_hash
from transport_reasoning.reachability.data import group_path

allocation();rows=[]
for family in ['er','dag']:
    for n in [16,17]:
        start=time.perf_counter();train=group_path(family,'train',n);test=group_path(family,'test_ID',n)
        tw=np.load(train/'wl.npy');vw=np.load(test/'wl.npy');groups=defaultdict(list)
        for i,w in enumerate(tw):groups[bytes(w)].append(i)
        tb=np.load(train/'B.npy',mmap_mode='r');vb=np.load(test/'B.npy',mmap_mode='r')
        candidates=sum(len(groups.get(bytes(w),[])) for w in vw);checked=matches=0;affected_train=set();affected_test=set()
        for j,w in enumerate(vw):
            others=groups.get(bytes(w),[])
            if not others:continue
            test_graph=nx.from_numpy_array(vb[j],create_using=nx.DiGraph)
            for i in others:
                if time.perf_counter()-start>240:break
                checked+=1
                if nx.is_isomorphic(nx.from_numpy_array(tb[i],create_using=nx.DiGraph),test_graph):
                    matches+=1;affected_train.add(i);affected_test.add(j)
            if time.perf_counter()-start>240:break
        rows.append({'family':family,'n':n,'candidate_pairs':candidates,'exact_pairs_checked':checked,
                     'isomorphic_pairs':matches,'affected_train_graphs':len(affected_train),'affected_ID_test_graphs':len(affected_test),
                     'ID_test_graphs':len(vb),'complete':checked==candidates,'cpu_elapsed_seconds':time.perf_counter()-start})
result={'status':'completed' if all(r['complete'] for r in rows) else 'partial','rows':rows,
        'data_hash':read(data_root()/'manifest.json')['data_hash'],'audit_source_sha256':file_hash(__file__),
        'data_changed':False,'interpretation':'exact labeled graphs are disjoint; some ID-test graph topologies recur up to directed isomorphism; no topological-disjointness claim'}
write_json(root()/'DATA_ISOMORPHISM_COMPLETE.json',result);print(result,flush=True)
