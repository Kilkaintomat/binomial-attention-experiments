import json
import math
import time
from dataclasses import replace
import numpy as np
import torch
from transport_reasoning.config import write_json
from .data import TaskData
from transport_reasoning.evaluation import evaluate_model, loss_for, sync, sudoku_valid
from transport_reasoning.training import atomic_checkpoint, environment
from .config import Config, code_hash, root, make, LABELS, SUPPORTS
from .model import Encoder
from .training import setup, optimization, train


def distribution(values):
    return {'median': float(np.median(values)), 'p25': float(np.quantile(values, .25)),
            'p75': float(np.quantile(values, .75)), 'minimum': min(values), 'maximum': max(values), 'repeats': len(values), 'raw_seconds': [float(x) for x in values]}


@torch.no_grad()
def benchmark(model, data, c, repeats=30, warmup=10):
    model.eval(); records = []
    for batch_size in [1, 64]:
        ids = np.arange(batch_size); batch = data.batch(ids, c.device)
        for _ in range(warmup): model(batch)
        sync(c.device)
        forward = []; full = []
        for _ in range(repeats):
            sync(c.device); tick = time.perf_counter(); model(batch); sync(c.device)
            forward.append(time.perf_counter() - tick)
            sync(c.device); tick = time.perf_counter()
            prepared = data.batch(ids, c.device); logits = model(prepared); sync(c.device)
            puzzle = prepared['puzzle'].cpu().numpy()
            pred = logits.argmax(-1).cpu().numpy() + 1
            pred = np.where(puzzle != 0, puzzle, pred)
            for p, q in zip(pred, puzzle): sudoku_valid(p, q)
            sync(c.device); full.append(time.perf_counter() - tick)
        records.append({'batch_size': batch_size, 'warmup_forwards': warmup, 'precision': 'float32',
                        'forward_batch_seconds': distribution(forward), 'end_to_end_batch_seconds': distribution(full),
                        'forward_examples_per_second': batch_size / float(np.median(forward)),
                        'end_to_end_examples_per_second': batch_size / float(np.median(full))})
    return records


def smoke():
    outcomes=[]
    for label in LABELS:
        for support in SUPPORTS:
            c=make(label,support,seed=227,stage='smoke',device='cpu'); setup(c)
            data=TaskData(c,'train').subset(64); model=Encoder(c); opt,scheduler=optimization(model,c)
            initial={n:p.detach().clone() for n,p in model.named_parameters()}
            losses=[]; generator=np.random.default_rng(c.seed+987123)
            for step in range(10):
                batch=data.batch(generator.integers(0,64,64),'cpu'); opt.zero_grad(set_to_none=True)
                loss=loss_for(model(batch),batch,c)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite smoke loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),c.clip,error_if_nonfinite=True)
                opt.step(); scheduler.step(); losses.append(float(loss))
            changed=any(not torch.equal(initial[n],p) for n,p in model.named_parameters())
            if not changed: raise RuntimeError('Optimizer did not change parameters')
            outcomes.append({'model':label,'support':c.mask_label(),'seed':227,'updates':10,'losses':losses,'parameters_changed':changed})
    result={'status':'passed','source_sha256':code_hash(),'results':outcomes,'generalization_claim':False}
    write_json(root()/'checks/smoke.json',result); return result


def overfit(label, support):
    c=make(label,support,seed=229,stage='overfit',device='cpu',overfit_count=64,planned_steps=1000,completion_steps=1000,eval_every=100)
    checkpoint=c.run_dir()/'checkpoint_last.pt'
    result=train(c,resume=checkpoint if checkpoint.exists() else None,deadline=time.perf_counter()+3300)
    curve=[json.loads(line) for line in (c.run_dir()/'metrics.jsonl').read_text().splitlines()]
    finite=all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in curve)
    status='passed' if finite and result['completed_steps']==1000 else 'incomplete'
    outcome={'status':status,'source_sha256':code_hash(),'model':label,'support':c.mask_label(),'seed':229,
             'completed_steps':result['completed_steps'],'best_train_subset_solved':result['best_validation_solved_rate'],
             'initial_loss':curve[0]['loss'],'final_loss':curve[-1]['loss'],'finite':finite,
             'memorized_95_percent':result['best_validation_solved_rate']>=.95,
             'low_memorization_is_not_automatically_a_code_error':True,'generalization_claim':False}
    write_json(root()/f'checks/overfit_{c.mask_label()}_{label}.json',outcome)
    return outcome


def collect_overfit():
    paths=[root()/f'checks/overfit_{make(label,support).mask_label()}_{label}.json' for label in LABELS for support in SUPPORTS]
    rows=[json.loads(p.read_text()) for p in paths if p.exists()]
    complete=len(rows)==6 and all(r['status']=='passed' and r['source_sha256']==code_hash() for r in rows)
    result={'status':'passed' if complete else 'incomplete','source_sha256':code_hash(),'results':rows,
            'criterion':'1000 finite updates on 64 train puzzles; low solved alone does not block this locality ablation'}
    write_json(root()/'checks/overfit.json',result); return result


def profile():
    results={}; target=root()/'profile'; target.mkdir(parents=True,exist_ok=True)
    for label in LABELS:
        for support in SUPPORTS:
            c=make(label,support,seed=109,stage='profile'); setup(c)
            train_data=TaskData(c,'train'); val=TaskData(c,'validation'); model=Encoder(c).to(c.device)
            opt,scheduler=optimization(model,c); generator=np.random.default_rng(c.seed+987123)
            times=[]; torch.cuda.reset_peak_memory_stats()
            for step in range(70):
                sync(c.device); tick=time.perf_counter()
                batch=train_data.batch(generator.integers(0,len(train_data),64),c.device)
                opt.zero_grad(set_to_none=True); loss=loss_for(model(batch),batch,c); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),c.clip,error_if_nonfinite=True)
                opt.step(); scheduler.step(); sync(c.device)
                if step>=10: times.append(time.perf_counter()-tick)
            peak=torch.cuda.max_memory_allocated(); validations=[]; checkpoints=[]
            name=c.mask_label()+'_'+label
            for _ in range(3):
                tick=time.perf_counter(); evaluate_model(model,val,c,c.device,target/(name+'_predictions.npz')); sync(c.device)
                validations.append(time.perf_counter()-tick)
                tick=time.perf_counter(); atomic_checkpoint(target/(name+'_checkpoint.pt'),{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict()})
                checkpoints.append(time.perf_counter()-tick)
            entry={'support':c.mask_label(),'model':label,'config':c.dict(),'parameters':model.parameter_count(),
                   'mask_sha256':train_data.mask_sha256,'data_hash':train_data.manifest['data_hash'],'operations':c.operations(),
                   'step_seconds':distribution(times),'validation_18000_seconds':distribution(validations),
                   'checkpoint_seconds':distribution(checkpoints),'peak_memory_bytes':peak,
                   'training_examples_per_second':64/float(np.median(times)),'inference':benchmark(model,val,c)}
            entry['pilot_seconds']=float(10000*np.median(times)+41*(np.median(validations)+3*np.median(checkpoints)))
            entry['final_seconds']=float(30000*np.median(times)+122*np.median(validations)+121*3*np.median(checkpoints))
            results[c.key()]=entry
            write_json(target/'gpu_profile.json',{'source_sha256':code_hash(),'environment':environment(),'configurations':results})
            print(json.dumps({'profile_completed':c.key(),'estimated_final_seconds':entry['final_seconds']}),flush=True)
    checks=json.loads((root()/'checks/legacy_compatibility.json').read_text())
    reuse=checks['reuse_all_15_relational_allowed']
    pilots=3*sum(r['pilot_seconds'] for r in results.values())
    finals=5*sum(r['final_seconds'] for key,r in results.items() if key.startswith('grid4/') or not reuse)
    estimate={'source_sha256':code_hash(),'measured_on_allocated_gpu':True,'reuse_15_relational':reuse,
              'mandatory_pilots':18,'mandatory_final_results':30,'new_training_runs':33 if reuse else 48,
              'mandatory_seconds_before_margin':pilots+finals+900,'mandatory_seconds_with_20_percent_margin':1.2*(pilots+finals+900),
              'profile_allocation_charged_separately':True,'reserve_includes_current_inference_and_position_diagnostics':True,
              'old_training_time_not_current_speed_evidence':True}
    write_json(target/'budget_estimate.json',estimate); return estimate


@torch.no_grad()
def position_accuracy(model,data,c):
    model.eval(); ids=np.arange(64); batch=data.batch(ids,c.device)
    pred=model(batch).argmax(-1)+1; blank=batch['puzzle']==0; correct=pred==batch['y']
    row,col=np.divmod(np.arange(81),9)
    corner=((row==0)|(row==8))&((col==0)|(col==8))
    boundary=((row==0)|(row==8)|(col==0)|(col==8))&~corner
    groups={'corners':corner,'other_edges':boundary,'interior':~(corner|boundary)}
    records=[]
    for name,mask in groups.items():
        selected=blank & torch.as_tensor(mask,device=c.device)[None,:]
        denominator=int(selected.sum()); numerator=int((correct&selected).sum())
        records.append({'group':name,'cells_per_puzzle':int(mask.sum()),'puzzles':64,'blank_cells':denominator,
                        'correct_blank_cells':numerator,'blank_accuracy':numerator/denominator if denominator else None,
                        'aggregation':'micro over original blank cells'})
    return {'split':'validation_first64','ids':data.ids[:64],'groups':records,
            'validator':'unchanged full Sudoku constraints; this diagnostic is raw blank accuracy'}
