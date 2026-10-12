"""CPU invariants, GPU smoke/resume checks, and measured resource admission."""
import argparse
import time
from fair_common import *
from fair_data import prepare_data, branch_graph
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.data import raw_inputs, torch_inputs
from transport_reasoning.reachability.checks import TinyDataset, timing


def cpu_prepare():
    allocation();torch.set_num_threads(1);freeze_protocol();install_training_adapter()
    checks=[]
    B=np.zeros((7,7),bool);B[np.arange(6),np.arange(1,7)]=True
    x=torch_inputs(raw_inputs([B,B],[(0,6),(6,0)]),'cpu')
    for name in NEW_MODELS:
        c=make('er',name,seed=211,stage='smoke',device='cpu',threads=1);model=Encoder(c)
        out=model(x);assert torch.isfinite(out).all();out.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        # Permutation and padding preserve predictions; this catches graph/target convention errors.
        p=np.array([2,6,1,5,0,4,3]);inv=np.argsort(p)
        xp=torch_inputs(raw_inputs([B[np.ix_(p,p)]]*2,[(inv[0],inv[6]),(inv[6],inv[0])]),'cpu')
        pad=torch_inputs(raw_inputs([B,B],[(0,6),(6,0)],pad_to=11),'cpu')
        assert torch.allclose(out,model(xp),atol=2e-6,rtol=2e-6) and torch.allclose(out,model(pad),atol=2e-6,rtol=2e-6)
        if c.last_only:
            assert all(b.fixed_coeff.tolist()==[0.,0.,0.,1.] for b in model.blocks)
            # Independent last-power block formula; includes residual/output/outer FFN.
            b=model.blocks[0];v=torch.randn(2,7,128);mask=x['attention_mask']
            q,k,z=b.qkv(b.ln1(v)).reshape(2,7,3,4,32).permute(2,0,3,1,4).unbind(0)
            a=((q@k.transpose(-1,-2))/32**.5).masked_fill(~mask[:,None],-torch.inf).softmax(-1)
            z=torch.linalg.matrix_power(a,4)@z
            u=v+b.out(z.transpose(1,2).reshape_as(v));expected=u+b.ff(b.ln2(u))
            assert torch.allclose(b(v,mask,x['node_valid']),expected,atol=2e-6,rtol=2e-6)
        expected_radius=c.outer_steps*c.hops
        assert model.operations()['transport_matmuls']==expected_radius
        for variant in [False,True]:
            graph,pairs=branch_graph(128,40,variant,np.random.default_rng(773))
            z=model(torch_inputs(raw_inputs([graph,graph],pairs),'cpu'))
            assert torch.allclose(z[0],z[1],atol=2e-6,rtol=2e-6)
        checks.append({'model':name,'operations':model.operations(),'radius':expected_radius,'status':'passed'})
    # Common initial learned tensors are exactly equal where the architecture shares them.
    r16=Encoder(make('er','R16-one',device='cpu'));r32=BaseEncoder(next(c for c in baseline_configs() if c.model=='R32-one' and c.family=='er' and c.seed==11))
    for k,v in r16.state_dict().items():assert torch.equal(v,r32.state_dict()[k])
    last=Encoder(make('er','F4-last-r4',device='cpu'));b4=BaseEncoder(next(c for c in baseline_configs() if c.model=='F4-bin' and c.family=='er' and c.seed==11))
    b8=Encoder(make('er','F4-bin-r8',device='cpu'))
    for (name,p),(name2,q),(name3,z) in zip(last.named_parameters(),b4.named_parameters(),b8.named_parameters()):
        assert name==name2==name3 and torch.equal(p,q) and torch.equal(q,z)
    from fair_runner import calibrated_threshold
    threshold,info=calibrated_threshold(np.array([0,1,0,1]),np.array([-3.,-1.,-2.,-.5]))
    assert info['validation_BA']==1. and threshold==-1.
    threshold,info=calibrated_threshold(np.array([0,1]),np.array([0.,0.]))
    assert threshold==0. and info['validation_BA']==.5
    m=prepare_data()
    value={'status':'passed','architecture_checks':checks,'initialization_equality':True,'terminal_power_formula':True,
           'padding_permutation':True,'locality_outside_radius':True,'threshold_selection_tests':True,
           'source_hash':extension_hash(),'data_hash':m['data_hash'],'scheduler_job':os.environ['SLURM_JOB_ID']}
    write_json(OUT/'cpu_preflight.json',value);print(json.dumps(value),flush=True)


def resume_check():
    original=os.environ['REACH_ROOT'];tiny=TinyDataset();c=make('er','F4-last-r4',seed=211,stage='smoke')
    # Use a GPU allocation; exact replay must use the same deterministic batch sequence and optimizer state.
    try:
        os.environ['REACH_ROOT']=str(OUT/'checks/continuous')
        trainer.train(c,train_data=tiny,validation=tiny)
        full=torch.load(c.directory()/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        os.environ['REACH_ROOT']=str(OUT/'checks/resumed')
        trainer.train(c,stop_after=20,train_data=tiny,validation=tiny)
        trainer.train(c,train_data=tiny,validation=tiny)
        resumed=torch.load(c.directory()/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        assert full['step']==resumed['step']==40 and full['batch_rng']==resumed['batch_rng'] and full['scheduler']==resumed['scheduler']
        for k,v in full['model'].items():assert torch.equal(v,resumed['model'][k]),k
        for k,v in full['optimizer']['state'].items():
            for field,t in v.items():
                u=resumed['optimizer']['state'][k][field]
                assert torch.equal(t,u) if torch.is_tensor(t) else t==u
    finally:os.environ['REACH_ROOT']=original
    return {'status':'passed','steps':40,'resume_at':20,'comparison':'bitwise tensors, optimizer, scheduler, batch RNG'}


def gpu_preflight():
    allocation();freeze_protocol();install_training_adapter()
    assert read(OUT/'cpu_preflight.json')['status']=='passed'
    rows=[];smoke=[]
    resumed=resume_check()
    for family in FAMILIES:
        train=Dataset(family,'train',limit_graphs=32);validation=Dataset(family,'validation',limit_graphs=32)
        for name in NEW_MODELS:
            c=make(family,name,seed=211,stage='smoke')
            result=trainer.train(c,train_data=train,validation=validation)
            assert result['status']=='completed' and result['step']==40
            smoke.append({'family':family,'model':name,'steps':40,'validation':result['final_validation']})
            dataset=Dataset(family,'train');device=trainer.setup(c);model=Encoder(c).to(device)
            optimizer,_=trainer.optimizer_for(model,c);rng=np.random.default_rng(551);times=[]
            for i in range(70):
                trainer.sync(device);tick=time.perf_counter();inputs,y,_=dataset.batch(rng.integers(len(dataset),size=64),device)
                optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(inputs),y)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();trainer.sync(device)
                if i>=10:times.append(time.perf_counter()-tick)
            measurements=[timing(model,n,64,device,warmup=2,repeats=5) for n in [19,32,64,128]]
            rate={r['n']:r['end_to_end_median_seconds']/64 for r in measurements};step=float(np.median(times))
            val=40000*rate[19];pilot=3000*step+(5*2048+40000)*rate[19]+10
            final=30000*step+30*val+60
            evaluation=sum(4000*rate[n] for n in [32,64,128])+9600*rate[128]+val+8
            rows.append({'family':family,'model':name,'step_seconds':step,'pilot_seconds':pilot,'final_seconds':final,
                         'evaluation_seconds':evaluation,'operations':model.operations(),'profile_inference':measurements})
    # Conservative reused-checkpoint inference reserve; no test scores enter admission.
    estimate=sum(3*r['pilot_seconds']+5*(r['final_seconds']+r['evaluation_seconds']) for r in rows)+1800
    result={'status':'passed','resume':resumed,'smoke':smoke,'profile':rows,'estimated_full_seconds':estimate,
            'estimated_full_seconds_with_reserve':1.2*estimate,'test_scores_used':False,
            'source_hash':extension_hash(),'scheduler_job':os.environ['SLURM_JOB_ID']}
    write_json(OUT/'gpu_preflight.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['smoke','profile']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['cpu','gpu']);args=parser.parse_args()
    cpu_prepare() if args.mode=='cpu' else gpu_preflight()
