import copy
import json
import numpy as np
import pytest
import torch
from transport_reasoning.config import Config
from transport_reasoning.filters import coefficients,action,binomial_stable,moments,exponential_full_action,sparse_action
from transport_reasoning.data import tree_maze,topology_id,path_label,sudoku_solutions,make_sudoku,prepare,TaskData
from transport_reasoning.evaluation import maze_valid,sudoku_valid,loss_for
from transport_reasoning.models import Encoder
from transport_reasoning.models.mlp_audit import audit
from transport_reasoning.training import train,evaluate_checkpoint


@pytest.mark.parametrize("h",[1,2,6,8,16])
def test_filters(h):
    torch.manual_seed(2)
    a=torch.randn(7,7,dtype=torch.float64).softmax(-1); v=torch.randn(7,3,dtype=torch.float64)
    eye=torch.eye(7,dtype=torch.float64)
    for method in ["onehop","binomial","exponential_truncated","finite_geometric","learned_polynomial"]:
        c=coefficients(method,h)
        explicit=sum(float(w)*torch.linalg.matrix_power(a,i+1)@v for i,w in enumerate(c))
        torch.testing.assert_close(action(a,v,c),explicit,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(action(eye,v,c),v,atol=1e-12,rtol=1e-12)
        operator=action(a,eye,c)
        assert operator.min()>=0
        torch.testing.assert_close(operator.sum(-1),torch.ones(7,dtype=torch.float64))
        if method!="onehop": assert abs(moments(c)["mu"]-h/(2*(1-2.**-h)))<1e-11
    torch.testing.assert_close(binomial_stable(a,v,h),action(a,v,coefficients("binomial",h)),atol=1e-12,rtol=1e-12)


def test_exponential():
    torch.manual_seed(3)
    a=torch.randn(6,6,dtype=torch.float64).softmax(-1); v=torch.randn(6,3,dtype=torch.float64)
    for lam in [1e-8,1.,3.,8.]:
        y,certificate=exponential_full_action(a,v,lam)
        # matrix_exp subtraction is poorly conditioned near zero, so use the analytic limit there.
        ref=(torch.matrix_exp(lam*a)-torch.eye(6,dtype=torch.float64))@v/np.expm1(lam) if lam>1e-6 else a@v
        torch.testing.assert_close(y,ref,atol=1e-8,rtol=1e-8)
        assert certificate["omitted_mass"]<=1e-12


def test_gradients_and_sparse():
    torch.manual_seed(4); dtype=torch.float64
    mask=torch.eye(4,dtype=torch.bool)|torch.roll(torch.eye(4,dtype=torch.bool),1,0)
    logits=torch.randn(4,4,dtype=dtype,requires_grad=True); v=torch.randn(4,2,dtype=dtype,requires_grad=True)
    theta=torch.randn(3,dtype=dtype,requires_grad=True)
    def dense(l,z,t): return action(l.masked_fill(~mask,-torch.inf).softmax(-1),z,t.softmax(-1))
    assert torch.autograd.gradcheck(dense,(logits,v,theta))
    idx=mask.nonzero().T
    def sparse(l,z,t):
        a=l.masked_fill(~mask,-torch.inf).softmax(-1)
        sp=torch.sparse_coo_tensor(idx,a[mask],a.shape)
        return sparse_action(sp,z,t.softmax(-1))
    y=dense(logits,v,theta); z=sparse(logits,v,theta)
    torch.testing.assert_close(y,z,rtol=1e-12,atol=1e-12)
    gy=torch.autograd.grad(y.square().sum(),(logits,v,theta),retain_graph=True)
    gz=torch.autograd.grad(z.square().sum(),(logits,v,theta))
    for a,b in zip(gy,gz): torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)


def test_chain_tail():
    a=torch.zeros(9,9,dtype=torch.float64); a[0,0]=1
    for i in range(1,9): a[i,i-1]=1
    v=torch.zeros(9,1,dtype=torch.float64); v[0]=1
    c=coefficients("binomial",8); y=action(a,v,c).flatten()
    for d in range(1,9): assert abs(y[d]-c[d-1:].sum())<1e-12
    shift=a.clone(); shift[0,0]=0
    z=action(shift,v,c).flatten()
    torch.testing.assert_close(z[1:],c)


def test_validators_and_topology():
    rng=np.random.default_rng(1)
    for generator in ["dfs","wilson"]:
        a=tree_maze(4,rng,generator); y=path_label(a,0,15)
        assert maze_valid(a,y,0,15)
        assert not maze_valid(a,np.zeros(16,bool),0,15)
        rotated=np.rot90(np.arange(16).reshape(4,4)).ravel()
        assert topology_id(a,4)==topology_id(a[np.ix_(rotated,rotated)],4)
        assert (a.sum()-16)//2==15
    q,y=make_sudoku(4,rng,.5)
    assert len(sudoku_solutions(q))==1
    assert sudoku_valid(y,q)
    assert not sudoku_valid(np.ones(16,int),q)


def test_mlp_algebra():
    result=audit()
    assert result["direct_slot_max_error"]<1e-12
    assert result["direct_slot_max_gradient_error"]<1e-12
    assert result["frozen_polynomial_max_error"]<1e-12
    assert all(a>b for a,b in zip(result["euler_to_exp_errors_fixed_tau"],result["euler_to_exp_errors_fixed_tau"][1:]))


def test_compact_sudoku_targets_are_long(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    c=Config(task="sudoku",size=4,train_count=8,val_count=4,test_count=4,width=16,layers=1,batch_size=4)
    prepare(c)
    data=TaskData(c,"train")
    data.data["y"]=data.data["y"].astype(np.int8)
    data.data["puzzle"]=data.data["puzzle"].astype(np.int8)
    batch=data.batch(np.arange(4),"cpu")
    assert batch["y"].dtype==torch.long
    assert torch.isfinite(loss_for(Encoder(c)(batch),batch,c))


@pytest.mark.parametrize("task,size",[("maze",4),("sudoku",4)])
def test_smoke_resume(tmp_path,monkeypatch,task,size):
    monkeypatch.chdir(tmp_path)
    c=Config(task=task,size=size,train_count=16,val_count=4,test_count=4,steps=4,eval_every=2,batch_size=4,width=16,layers=1,hops=2)
    prepare(c)
    train_ids=set(TaskData(c,"train").ids); val_ids=set(TaskData(c,"validation").ids)
    assert not train_ids&val_ids
    first=train(c,stop_after=2); assert first["status"]=="paused"
    resumed=train(c,resume=c.run_dir()/"checkpoint_last.pt"); assert resumed["status"]=="completed"
    weights=copy.deepcopy(torch.load(c.run_dir()/"checkpoint_last.pt",weights_only=False)["model"])
    result=evaluate_checkpoint(c,c.run_dir()/"checkpoint_best.pt")
    assert 0<=result["solved_rate"]<=1
    other=tmp_path/"uninterrupted"; other.mkdir(); monkeypatch.chdir(other)
    prepare(c); train(c)
    direct=torch.load(c.run_dir()/"checkpoint_last.pt",weights_only=False)["model"]
    for key in weights: torch.testing.assert_close(weights[key],direct[key],rtol=0,atol=0)


def test_common_initialization_inner_identity_and_gradients(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    c=Config(train_count=16,val_count=4,test_count=4,width=16,layers=1,hops=2,batch_size=4)
    prepare(c); data=TaskData(c,"train"); batch=data.batch(np.arange(4),"cpu")
    m=Encoder(c)
    inner_c=copy.deepcopy(c); inner_c.inner_beta=.1
    inner=Encoder(inner_c)
    altered_batch={**batch,"y":torch.flip(batch["y"],[0])}
    torch.testing.assert_close(m(batch),m(altered_batch),rtol=0,atol=0)
    for key,value in m.state_dict().items(): torch.testing.assert_close(value,inner.state_dict()[key])
    inner.blocks[0].c.inner_beta=0
    torch.testing.assert_close(m(batch),inner(batch),rtol=0,atol=0)
    inner.blocks[0].c.inner_beta=.1
    loss=loss_for(inner(batch),batch,c,data.manifest["pos_weight"]); loss.backward()
    for name,p in inner.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    initial=loss_for(m(batch),batch,c,data.manifest["pos_weight"]).item()
    original=m.blocks[0].qkv.weight.detach().clone(); opt=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(30):
        opt.zero_grad(); loss=loss_for(m(batch),batch,c,data.manifest["pos_weight"]); loss.backward(); opt.step()
    assert loss.item()<initial
    assert not torch.equal(original,m.blocks[0].qkv.weight)
