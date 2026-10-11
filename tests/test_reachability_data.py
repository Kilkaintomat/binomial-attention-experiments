import numpy as np
import networkx as nx
from transport_reasoning.reachability.data import bfs,closure,raw_inputs,graph,rng_for,sample_pairs,stress_graph
from transport_reasoning.reachability.config import STRESS_CELLS


def test_chain_orientation_and_reachable_nonedge():
    B=np.array([[0,1,0],[0,0,1],[0,0,0]],bool)
    assert bfs(B,0)[2]==2 and bfs(B,2)[0]==-1 and not B[0,2]
    mask=raw_inputs([B],[(0,2)])['attention_mask'][0]
    A=mask/mask.sum(1,keepdims=True);z=np.array([1.,0.,0.])
    assert (A@z)[1]>0 and (A@z)[2]==0 and (A@A@z)[2]>0
    assert not mask[0,1] and mask[1,0]
    idx=np.stack(np.meshgrid(np.arange(3),np.arange(3))).reshape(2,-1)
    original=np.stack([bfs(B,s) for s in range(3)])
    recreated=np.empty_like(original);recreated[idx[0],idx[1]]=original.flatten()
    assert np.array_equal(recreated,original.T)  # Upstream flatten/meshgrid orientation.


def test_random_oracles_exact_size_and_separate_rng():
    for family in ['er','dag']:
        for i in range(12):
            B,_=graph(family,17,rng_for(family,'train',17,i,0,0),rng_for(family,'train',17,i,0,2))
            assert B.shape==(17,17) and not B.diagonal().any()
            if family=='dag':assert nx.is_directed_acyclic_graph(nx.from_numpy_array(B,create_using=nx.DiGraph))
            C=closure(B);pairs=sample_pairs(17,rng_for(family,'train',17,i,0,1))
            assert len(set(map(tuple,pairs)))==8 and (pairs[:,0]!=pairs[:,1]).all()
            for s,t in pairs:assert bool(C[s,t])==(bfs(B,int(s))[t]>=0)


def test_stress_oracles_and_source_degree_equality():
    assert len(STRESS_CELLS)==15
    for variant in ['dag','cyclic']:
        for n,ell in STRESS_CELLS:
            B,pairs,_=stress_graph(n,ell,variant,np.random.default_rng(20261011))
            assert [int(bfs(B,int(s))[t]) for s,t in pairs]==[ell,-1]
            a,b=pairs[:,0];assert B[a].sum()==B[b].sum() and B[:,a].sum()==B[:,b].sum()
            G=nx.from_numpy_array(B,create_using=nx.DiGraph)
            assert nx.is_weakly_connected(G)
            assert nx.is_directed_acyclic_graph(G)==(variant=='dag')
