import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transport_reasoning.config import Config as LegacyConfig
from transport_reasoning.data import prepare, TaskData as LegacyTaskData
from transport_reasoning.filters import action, coefficients
from transport_reasoning.models import Encoder as LegacyEncoder
from research.sudoku_local_mask_v1.config import Config, make, grid, from_legacy
from research.sudoku_local_mask_v1.data import allowed_mask, mask_hash, diameter, TaskData
from research.sudoku_local_mask_v1.model import Encoder
from research.sudoku_local_mask_v1.training import train


def equal(a,b):
    if torch.is_tensor(a): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


def inputs(support='grid4',size=9):
    g=torch.Generator().manual_seed(811)
    return {'x':torch.randn(2,size*size,4*size+2,generator=g,dtype=torch.float64),
            'mask':torch.tensor(np.broadcast_to(allowed_mask(support,size),(2,size*size,size*size)).copy())}


def test_mask_geometry_exact():
    gridmask=allowed_mask('grid4'); relational=allowed_mask('graph')
    assert np.array_equal(gridmask,gridmask.T) and np.diag(gridmask).all()
    unique,counts=np.unique(gridmask.sum(1),return_counts=True)
    assert dict(zip(unique,counts))=={3:4,4:28,5:49}
    assert gridmask.sum()==369 and (gridmask.sum()-81)//2==144 and diameter(gridmask)==16
    assert np.all(relational.sum(1)==21) and diameter(relational)==2
    assert not gridmask[8,9] and not gridmask[0,10] and not gridmask[0,8]
    assert mask_hash(gridmask)!=mask_hash(relational)


def test_config_grid_and_legacy_hash_preserved():
    c=LegacyConfig(task='sudoku',size=9,method='recurrent',width=128,heads=4,layers=4,recurrent_steps=32)
    before=c.hash(); new=from_legacy(c)
    assert c.hash()==before and 'series' not in c.dict()
    assert new.layers==1 and new.hops==1 and new.recurrent_steps==32
    rows=grid(); assert len(rows)==48 and sum(c.stage=='pilot' for c in rows)==18
    assert len({(c.support,c.label()) for c in rows})==6
    assert all(c.planned_steps==30000 for c in rows)
    assert make('F4-onehop','graph').hash()!=make('F4-onehop','grid4').hash()
    with pytest.raises(ValueError): make('R32-onehop',attention_filter='binomial',hops=4)


def test_far_input_cannot_reach_f4_onehop():
    c=make('F4-onehop','grid4',width=16,device='cpu'); model=Encoder(c).double()
    batch=inputs(); batch['x'].requires_grad_(True)
    output=model(batch)
    changed={**batch,'x':batch['x'].detach().clone()}; changed['x'][:,80,:]+=7
    torch.testing.assert_close(output[:,0],model(changed)[:,0],rtol=0,atol=0)
    derivative=torch.autograd.grad(output[0,0,0],batch['x'])[0]
    assert torch.count_nonzero(derivative[0,80])==0
    row,col=np.divmod(np.arange(81),9); outside=torch.tensor(row+col>4)
    assert torch.count_nonzero(derivative[0,outside])==0
    assert derivative[0,1].abs().sum()>1e-12


@pytest.mark.parametrize('label,updates,transports',[('F4-onehop',4,4),('F4-binomial',4,16),('R32-onehop',32,32)])
def test_actual_attention_mask_heads_and_operations(label,updates,transports):
    c=make(label,'grid4',width=16,device='cpu'); model=Encoder(c).double(); batch=inputs()
    counts={'qkv':0,'ffn':0,'softmax':0,'av':0}; attention=[]; qkv_inputs=[]; hooks=[]
    def qhook(module,args,out): counts['qkv']+=1; qkv_inputs.append(args[0].detach().clone())
    def fhook(module,args,out): counts['ffn']+=1
    for block in model.blocks:
        hooks.extend([block.qkv.register_forward_hook(qhook),block.ff.register_forward_hook(fhook)])
    class Count(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            if func==torch.ops.aten._softmax.default: counts['softmax']+=1; attention.append(result.detach().clone())
            if func==torch.ops.aten.bmm.default and args[0].shape[-2:]==(81,81) and args[1].shape[-2:]==(81,4): counts['av']+=1
            return result
    with Count(): result=model(batch)
    for hook in hooks: hook.remove()
    assert counts=={'qkv':updates,'ffn':updates,'softmax':updates,'av':transports}
    for a in attention:
        assert a.shape==(2,4,81,81)
        forbidden=(~batch['mask'][:,None]).expand_as(a)
        assert torch.count_nonzero(a[forbidden])==0
        torch.testing.assert_close(a.sum(-1),torch.ones_like(a.sum(-1)),rtol=0,atol=1e-12)
    assert not torch.equal(qkv_inputs[0],qkv_inputs[1])
    result.square().mean().backward()
    assert torch.isfinite(result).all() and all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert c.operations()['transport_matmuls']==transports
    if label=='R32-onehop': assert len(model.blocks)==1


def test_same_initialization_parameters_and_r1_gradients():
    a=make('F4-onehop','graph',width=16,device='cpu')
    b=replace(a,support='grid4'); left,right=Encoder(a).double(),Encoder(b).double()
    equal(left.state_dict(),right.state_dict()); assert left.parameter_count()==right.parameter_count()
    binomial=Encoder(replace(b,attention_filter='binomial',hops=1)).double(); batch=inputs()
    equal(right.state_dict(),binomial.state_dict())
    torch.testing.assert_close(right(batch),binomial(batch),rtol=1e-12,atol=1e-12)
    right(batch).square().mean().backward(); binomial(batch).square().mean().backward()
    for p,q in zip(right.parameters(),binomial.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=1e-12,atol=1e-12)
    for label in ['F4-binomial','R32-onehop']:
        p,q=Encoder(make(label,'graph')),Encoder(make(label,'grid4'))
        equal(p.state_dict(),q.state_dict()); assert p.parameter_count()==q.parameter_count()
    assert Encoder(make('R32-onehop')).parameter_count()==204169
    assert Encoder(make('F4-binomial')).parameter_count()==797449


def test_polynomial_r4_and_gradients():
    torch.manual_seed(113); logits=torch.randn(5,5,dtype=torch.float64,requires_grad=True)
    v=torch.randn(5,3,dtype=torch.float64,requires_grad=True); a=logits.softmax(-1)
    c=coefficients('binomial',4)
    torch.testing.assert_close(c,torch.tensor([4,6,4,1],dtype=torch.float64)/15,rtol=0,atol=1e-15)
    actual=action(a,v,c); explicit=sum(w*torch.linalg.matrix_power(a,i+1)@v for i,w in enumerate(c))
    torch.testing.assert_close(actual,explicit,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(actual.square().sum(),(logits,v),retain_graph=True)
    gb=torch.autograd.grad(explicit.square().sum(),(logits,v))
    for p,q in zip(ga,gb): torch.testing.assert_close(p,q,rtol=1e-11,atol=1e-12)


def test_targets_do_not_enter_features_or_masks(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path); old=LegacyConfig(task='sudoku',size=4,train_count=16,val_count=4,test_count=4)
    prepare(old); monkeypatch.setenv('LOCAL_MASK_DATA_DIR',str(old.data_dir().resolve()))
    for support in ['graph','grid4']:
        c=make('F4-onehop',support,size=4,width=16,device='cpu'); masks=[]
        for split in ['train','validation','test']:
            data=TaskData(c,split); masks.append(data.graph)
            before=data.batch(np.arange(2),'cpu'); data.data['y'][:]=1; after=data.batch(np.arange(2),'cpu')
            equal(before['x'],after['x']); equal(before['mask'],after['mask'])
            model=Encoder(c); equal(model(before),model(after))
        assert all(np.array_equal(masks[0],m) for m in masks)
    assert make('F4-onehop','graph').data_dir()==make('F4-onehop','grid4').data_dir()


def test_legacy_relational_forward_exact():
    for method in ['onehop','binomial','recurrent']:
        old=LegacyConfig(task='sudoku',size=9,width=16,heads=4,layers=4,hops=4,recurrent_steps=32,method=method,seed=11)
        legacy=LegacyEncoder(old).double(); new=Encoder(from_legacy(old)).double(); batch=inputs('graph')
        equal(legacy.state_dict(),new.state_dict()); equal(legacy(batch),new(batch))


def test_interrupted_resume_restores_mask_optimizer_and_rng(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path); old=LegacyConfig(task='sudoku',size=4,train_count=16,val_count=4,test_count=4)
    prepare(old); monkeypatch.setenv('LOCAL_MASK_DATA_DIR',str(old.data_dir().resolve()))
    c=make('F4-binomial','grid4',stage='checks',size=4,width=16,device='cpu',planned_steps=6,completion_steps=6,eval_every=2,batch_size=4)
    monkeypatch.setenv('LOCAL_MASK_OUTPUT',str(tmp_path/'direct')); train(c)
    direct=torch.load(c.run_dir()/'checkpoint_last.pt',weights_only=False)
    monkeypatch.setenv('LOCAL_MASK_OUTPUT',str(tmp_path/'resumed')); result=train(c,stop_after=3)
    assert result['completed_steps']==3 and result['status']=='paused'
    assert [json.loads(x)['step'] for x in (c.run_dir()/'metrics.jsonl').read_text().splitlines()]==[1,2]
    train(c,resume=c.run_dir()/'checkpoint_last.pt'); resumed=torch.load(c.run_dir()/'checkpoint_last.pt',weights_only=False)
    for key in ['model','optimizer','scheduler','data_rng','python_rng','numpy_rng','torch_rng','mask_sha256','step','best']: equal(direct[key],resumed[key])
    g1,g2=np.random.default_rng(),np.random.default_rng(); g1.bit_generator.state=direct['data_rng'];g2.bit_generator.state=resumed['data_rng']
    assert np.array_equal(g1.integers(0,16,64),g2.integers(0,16,64))


def test_primary_interaction_is_paired_difference_of_differences():
    from research.sudoku_local_mask_v1.reporting import contrast
    rows=[]
    for i,seed in enumerate([11,23,37,53,71]):
        for support,one,binomial,recurrent in [('relational',.10,.12,.9),('grid4',.01,.08,.5)]:
            for model,value in [('F4-onehop',one),('F4-binomial',binomial),('R32-onehop',recurrent)]:
                rows.append({'stage':'final','status':'completed','support':support,'model':model,'seed':seed,'solved_rate':value+i*.001})
    values,summary=contrast(rows);lookup={r['contrast']:r for r in summary}
    assert lookup['I']['paired_seeds']==5 and lookup['I']['complete_five_seeds']
    assert lookup['G_grid']['mean_pp']==pytest.approx(7)
    assert lookup['G_rel']['mean_pp']==pytest.approx(2)
    assert lookup['I']['mean_pp']==pytest.approx(5)
    assert lookup['I']['lower95_pp']==pytest.approx(5)
    rows=[r for r in rows if not (r['support']=='grid4' and r['model']=='F4-binomial' and r['seed']==71)]
    _,summary=contrast(rows);row=next(r for r in summary if r['contrast']=='I')
    assert row['paired_seeds']==4 and not row['complete_five_seeds']


def test_admission_and_accounting_never_expand_limit(tmp_path,monkeypatch):
    from argparse import Namespace
    from transport_reasoning.config import write_json
    from research.sudoku_local_mask_v1 import scheduler
    from research.sudoku_local_mask_v1.campaign import admission
    monkeypatch.setenv('LOCAL_MASK_OUTPUT',str(tmp_path))
    write_json(tmp_path/'profile/budget_estimate.json',{'mandatory_seconds_with_20_percent_margin':1000,
               'reuse_15_relational':True,'new_training_runs':33})
    assert admission(900) is None and not (tmp_path/'admission.json').exists()
    value={'status':'profile_completed','gpu_seconds_used':3500,'first_allocation_unix':__import__('time').time(),
           'gpu_jobs':[{'job_id':'123','reserved_seconds':600,'accounted':False}]}
    scheduler.save(value);write_json(tmp_path/'queue/budget_grant.json',{'gpu_seconds_limit':4100,'wall_seconds_limit':7200})
    def command(args):
        assert args[0]=='sacct'
        return '123|COMPLETED|600|gres/gpu=1|0:0|'
    monkeypatch.setattr(scheduler,'command',command)
    monkeypatch.setattr(scheduler,'batch',lambda *a,**k:pytest.fail('No unreserved allocation'))
    result=scheduler.account(Namespace(job='123'))
    assert result['gpu_seconds_used']==4100 and result['status']=='budget_stopped'
    assert scheduler.account(Namespace(job='123'))['gpu_seconds_used']==4100


def test_predecessor_waits_for_existing_queue_without_mutation(tmp_path,monkeypatch):
    from transport_reasoning.config import write_json
    from research.sudoku_local_mask_v1 import scheduler
    old=tmp_path/'old';prior=tmp_path/'prior'
    monkeypatch.setenv('RUN_LEGACY_PROJECT',str(old));monkeypatch.setenv('RUN_PRIOR_SERIES_PROJECT',str(prior))
    legacy=old/'experiments/transport_reasoning/mlp_final_submission.json'
    previous=prior/'experiments/recurrent_filter_sudoku_v1/queue/state.json'
    write_json(legacy,{'gpu_job':'10','post_job_accounting_job':'11'})
    write_json(previous,{'status':'held','accounting_job':'21','gpu_jobs':[{'job_id':'20','accounted':False}]})
    before=(legacy.read_bytes(),previous.read_bytes())
    monkeypatch.setattr(scheduler,'active',lambda job:str(job)=='21')
    result=scheduler.predecessor()
    assert result['dependency']=='21' and not result['ready']
    assert before==(legacy.read_bytes(),previous.read_bytes())


def test_position_accuracy_uses_blank_group_denominators():
    from research.sudoku_local_mask_v1.measure import position_accuracy
    c=make('F4-onehop','grid4',device='cpu',width=16)
    class Data:
        ids=[str(i) for i in range(64)]
        def batch(self,ids,device):
            return {'x':torch.zeros(64,81,38),'mask':torch.tensor(np.broadcast_to(allowed_mask('grid4'),(64,81,81)).copy()),
                    'puzzle':torch.zeros(64,81,dtype=torch.long),'y':torch.ones(64,81,dtype=torch.long)}
    result=position_accuracy(Encoder(c),Data(),c)
    assert [g['blank_cells'] for g in result['groups']]==[64*4,64*28,64*49]
    assert sum(g['blank_cells'] for g in result['groups'])==64*81
