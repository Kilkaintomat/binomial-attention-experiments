from dataclasses import replace
import numpy as np
import pytest
import torch
from transport_reasoning.reachability.config import PRIORITY,make
from transport_reasoning.reachability.data import raw_inputs,torch_inputs,stress_graph
from transport_reasoning.reachability.model import Encoder,Block,coefficients,fixed_parameter
from transport_reasoning.filters import action


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


@pytest.mark.parametrize('name',PRIORITY)
def test_padding_permutation_finite_gradients_and_parameter_counts(name):
    c=make('er',name,stage='smoke',device='cpu');m=Encoder(c)
    B=np.zeros((6,6),bool);B[0,1]=B[1,2]=B[2,0]=B[3,4]=True
    original=torch_inputs(raw_inputs([B],[(0,2)]),'cpu')
    padded=torch_inputs(raw_inputs([B],[(0,2)],pad_to=9),'cpu')
    p=np.array([3,5,1,0,2,4]);inv=np.argsort(p)
    permuted=torch_inputs(raw_inputs([B[np.ix_(p,p)]],[(inv[0],inv[2])]),'cpu')
    a=m(original);assert torch.allclose(a,m(padded),atol=2e-6,rtol=2e-6)
    assert torch.allclose(a,m(permuted),atol=2e-6,rtol=2e-6)
    a.sum().backward();assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
    calls={'qkv':0,'inner':0,'outer':0};handles=[]
    def hook(key):
        def count(*_):calls[key]+=1
        return count
    for b in m.blocks:
        handles.append(b.qkv.register_forward_hook(hook('qkv')));handles.append(b.ff.register_forward_hook(hook('outer')))
        if hasattr(b,'inner'):handles.append(b.inner.register_forward_hook(hook('inner')))
    m(original)
    for h in handles:h.remove()
    expected=m.operations();assert calls=={'qkv':expected['qkv_calls'],'inner':expected['inner_ffn_calls'],'outer':expected['outer_ffn_calls']}
    assert len(m.blocks)==(1 if c.share_weights else 4)


def test_coefficients_common_initialization_and_fixed_extra_parameters():
    for method in ['binomial','exponential','geometric','learned']:
        c=coefficients(method);assert torch.isfinite(c).all() and abs(c.sum()-1)<1e-12
        assert abs((c*torch.arange(1,5)).sum()-32/15)<1e-11
        assert coefficients(method,1).tolist()==[1.]
    base=Encoder(make('er','F4-one',device='cpu')).state_dict()
    for name in PRIORITY:
        other=Encoder(make('er',name,device='cpu')).state_dict()
        for key in base.keys()&other.keys():
            if 'fixed_coeff' not in key:assert torch.equal(base[key],other[key])
    for kind in ['exponential','geometric']:
        p=fixed_parameter(kind);w=coefficients(kind,32,p)
        assert abs(float(w.sum())-1)<1e-12 and w.isfinite().all()
        assert float((w*torch.arange(1,33)).sum())<4  # No mean rematching at inference.


def test_linear_action_and_metadata_boundary():
    c=make('er','F4-bin',device='cpu');b=Block(c,0);x=torch.randn(2,5,128);mask=torch.eye(5,dtype=torch.bool).expand(2,-1,-1);valid=torch.ones(2,5,dtype=torch.bool)
    q,k,v=b.qkv(b.ln1(x)).reshape(2,5,3,4,32).permute(2,0,3,1,4).unbind(0)
    a=((q@k.transpose(-1,-2))/32**.5).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
    y=action(a,v,b.coeff());u=x+b.out(y.transpose(1,2).reshape_as(x));expected=u+b.ff(b.ln2(u))
    assert torch.allclose(b(x,mask,valid),expected,atol=1e-6)
    assert torch.count_nonzero(a.masked_select(~mask[:,None].expand_as(a)))==0
    model=Encoder(c);inputs=torch_inputs(raw_inputs([np.zeros((3,3),bool)],[(0,2)]),'cpu')
    logits=model(inputs);a=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros(1));z=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.ones(1))
    assert torch.equal(logits,model(inputs)) and not torch.equal(a,z)
    with pytest.raises(ValueError):model({**inputs,'label':torch.ones(1)})


@pytest.mark.parametrize('name,ell',[('F4-one',8),('F4-one-inner',8),('F4-bin',32),('F4-bin-inner',32)])
def test_source_outside_local_radius_is_indistinguishable(name,ell):
    B,pairs,_=stress_graph(128,ell,'cyclic',np.random.default_rng(17))
    model=Encoder(make('er',name,device='cpu'));out=model(torch_inputs(raw_inputs([B,B],pairs),'cpu'))
    assert torch.allclose(out[0],out[1],atol=2e-6,rtol=2e-6)


def test_inner_zero_recovers_linear_and_extra_learned_rejected():
    linear=Encoder(make('er','F4-bin',device='cpu'));inner=Encoder(make('er','F4-bin-inner',device='cpu'))
    for b in inner.blocks:b.beta=0.
    B=np.triu(np.ones((5,5),bool),1);x=torch_inputs(raw_inputs([B],[(0,4)]),'cpu')
    assert torch.equal(linear(x),inner(x))
    with pytest.raises(ValueError):Encoder(make('er','F4-learned',device='cpu')).extra_compute(hops=8)
    with pytest.raises(ValueError):linear.extra_compute(outer_steps=32)
