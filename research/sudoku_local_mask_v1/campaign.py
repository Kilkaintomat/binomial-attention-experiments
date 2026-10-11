"""Prespecified six-condition queue with strict separation of masks and stages."""
import json
import os
import time
from pathlib import Path
from transport_reasoning.config import write_json
from .config import BASE_COMMIT, LABELS, SUPPORTS, code_hash, snapshot_hash, grid, make, root


def read(path): return json.loads(Path(path).read_text())


def prepare():
    target=root(); target.mkdir(parents=True,exist_ok=True)
    protocol={'series':'sudoku_local_mask_v1','base_commit':BASE_COMMIT,'code_commit':os.environ.get('LOCAL_MASK_CODE_COMMIT'),
              'source_sha256':code_hash(),'snapshot_sha256':snapshot_hash(),
              'masks':{'graph':'relational rows,columns,boxes+self','grid4':'Manhattan distance <=1, no wrap'},
              'models':['F4-onehop','F4-binomial','R32-onehop'],
              'inputs':'same 38 token-local features; no global information channel','dataset_counts':[180000,18000,18000],
              'training':'AdamW lr .001 wd .01 clip1, batch64 replacement, warmup5%, cosine horizon30000, float32 TF32 off',
              'batch_rng':'numpy default_rng(seed+987123); paired across masks and methods',
              'pilots':{'count':18,'seeds':[101,103,107],'completion_steps':10000,'planned_steps':30000,'test_allowed':False},
              'finals':{'results':30,'seeds':[11,23,37,53,71],'steps':30000,'fresh_initialization':True},
              'historical_reuse':'all 15 relational finals only after effective-configuration and artifact audit; otherwise 48 new trainings',
              'checkpoint':'max validation solved, min blank CE, earliest exact tie; full validation step1 and every250',
              'primary_estimand_pp':'I=(S(F4-binomial,grid4)-S(F4-onehop,grid4))-(S(F4-binomial,relational)-S(F4-onehop,relational))',
              'uncertainty':{'method':'paired seed percentile bootstrap','seed':90421,'resamples':20000,'interval':[.025,.975],'sample_sd_ddof':1},
              'interpretation':'I is effect of this graph change, not pure diameter; absolute quality and cost always shown',
              'thresholds':'50%,90%; two consecutive full validations; censor failures and pending crossings',
              'position_diagnostic':'first64 validation; corners/other_edges/interior; micro accuracy among original blank cells',
              'timing':'warmup10, repeats30, synchronized forward and full preparation/argmax/clue copy/validator; batches1,64',
              'resource_policy':'one GPU; preserve all prior queues; no release before their completion, checks, and separate budget grant',
              'budget':'absolute approved GPU allocation seconds and wall limit; profile+validation+checkpoint+test+20% reserve',
              'test_caveat':'follow-up on a known published benchmark; no claim of untouched confirmatory test',
              'extensions':'none; recurrent binomial, inner-MLP, gating and deep supervision excluded'}
    path=target/'protocol.json'
    if path.exists() and read(path)!=protocol: raise RuntimeError('Frozen protocol differs; archive explicitly before any new version')
    write_json(path,protocol)
    compatible=read(target/'checks/legacy_compatibility.json') if (target/'checks/legacy_compatibility.json').exists() else {}
    reuse=compatible.get('reuse_all_15_relational_allowed',False) and compatible.get('source_sha256')==code_hash()
    rows=[]
    for c in grid():
        filename=f'configs/{c.stage}/{c.mask_label()}_{c.label()}_seed{c.seed}.json'; write_json(target/filename,c.dict())
        historical=reuse and c.stage=='final' and c.support=='graph'
        rows.append({'config_file':filename,'config_hash':c.hash(),'model':c.label(),'support':c.mask_label(),
                     'stage':c.stage,'seed':c.seed,'status':'completed' if historical else 'planned',
                     'origin':'historical_quality_only' if historical else 'new_training'})
    if not (target/'manifest.json').exists(): write_json(target/'manifest.json',{'source_sha256':code_hash(),'rows':rows})
    return {'configs':48,'pilot_trainings':18,'final_results':30,'planned_new_trainings':33 if reuse else 48,
            'historical_relational_reused':15 if reuse else 0,'source_sha256':code_hash(),'snapshot_sha256':snapshot_hash()}


def collect_overfit():
    paths=[root()/f'checks/overfit_{make(label,support).mask_label()}_{label}.json' for label in LABELS for support in SUPPORTS]
    rows=[read(p) for p in paths if p.exists()]
    passed=len(rows)==6 and all(r['status']=='passed' and r['source_sha256']==code_hash() for r in rows)
    result={'status':'passed' if passed else 'incomplete','source_sha256':code_hash(),'results':rows,
            'criterion':'1000 finite updates on 64 train puzzles; low solved alone is not classified as a software error'}
    write_json(root()/'checks/overfit.json',result); return result


def preflight():
    protocol=read(root()/'protocol.json')
    if protocol['source_sha256']!=code_hash() or protocol['snapshot_sha256']!=snapshot_hash(): raise RuntimeError('Frozen implementation changed')
    for name in ['pytest','data','legacy_compatibility','smoke']:
        check=read(root()/f'checks/{name}.json')
        if check['status']!='passed' or check['source_sha256']!=code_hash(): raise RuntimeError('Preflight failed: '+name)
    if collect_overfit()['status']!='passed': raise RuntimeError('Overfit execution incomplete; low quality alone does not block')


def admission(remaining_seconds):
    if (root()/'admission.json').exists(): return read(root()/'admission.json')
    estimate=read(root()/'profile/budget_estimate.json'); required=estimate['mandatory_seconds_with_20_percent_margin']
    if remaining_seconds<required:
        write_json(root()/'needs_budget.json',{'status':'budget_stopped','required_remaining_seconds':required,
                   'available_seconds':remaining_seconds,'additional_seconds_needed':required-remaining_seconds})
        return None
    if list((root()/'runs/final').glob('**/evaluation_test.json')): raise RuntimeError('Budget decision must precede new test')
    decision={'source_sha256':code_hash(),'reuse_15_relational':estimate['reuse_15_relational'],
              'new_training_runs':estimate['new_training_runs'],'remaining_at_decision_seconds':remaining_seconds,
              'required_with_margin_seconds':required,'grid_frozen_before_new_test':True}
    write_json(root()/'admission.json',decision); return decision


def update_row(c,status,**details):
    p=root()/'manifest.json'; manifest=read(p)
    for row in manifest['rows']:
        if row['config_hash']==c.hash(): row.update(status=status,**details)
    write_json(p,manifest)


def legacy_folder(c):
    method='recurrent' if c.architecture=='recurrent' else c.attention_filter
    base=Path(os.environ['LOCAL_MASK_LEGACY_ROOT']); candidates=[]
    for path in (base/'runs/sudoku'/method).glob(f'*/seed_{c.seed}/result.json'):
        result=read(path)
        if result['config']['stage']=='final' and result['config']['inner_beta']==0: candidates.append(path.parent)
    if len(candidates)!=1: raise RuntimeError('Missing or ambiguous historical artifact')
    return candidates[0]


def run_slot(seconds,remaining_seconds):
    from .measure import profile,benchmark,position_accuracy
    from .training import train,evaluate,setup
    from .model import load_checkpoint_model
    from .data import TaskData
    preflight(); deadline=time.perf_counter()+seconds
    if not (root()/'profile/budget_estimate.json').exists(): profile(); return 'profile_completed'
    decision=admission(remaining_seconds)
    if decision is None: return 'budget_stopped'
    for c in grid():
        if c.stage=='final' and c.support=='graph' and decision['reuse_15_relational']:
            update_row(c,'completed',origin='historical_quality_only'); continue
        if time.perf_counter()>=deadline-180: return 'paused'
        update_row(c,'running'); checkpoint=c.run_dir()/'checkpoint_last.pt'
        try: result=train(c,resume=checkpoint if checkpoint.exists() else None,deadline=deadline)
        except Exception as error:
            update_row(c,'failed',error_type=type(error).__name__); raise
        update_row(c,result['status'],completed_steps=result['completed_steps'])
        if result['status']!='completed': return 'paused'
    for c in [x for x in grid() if x.stage=='final']:
        if time.perf_counter()>=deadline-300: return 'paused'
        folder=c.run_dir(); folder.mkdir(parents=True,exist_ok=True)
        historical=c.support=='graph' and decision['reuse_15_relational']
        if not historical and not (folder/'evaluation_test.json').exists(): evaluate(c)
        if not (folder/'measurement.json').exists():
            setup(c); source=legacy_folder(c) if historical else folder
            model,state=load_checkpoint_model(c,source/'checkpoint_best.pt');model.to(c.device);data=TaskData(c,'validation')
            write_json(folder/'measurement.json',{'source_sha256':code_hash(),'support':c.mask_label(),'model':c.label(),'seed':c.seed,
                       'checkpoint_step':state['step'],'historical_weights':historical,'checkpoint_source_sha256':state['source_sha256'],
                       'mask_sha256':data.mask_sha256,'inference':benchmark(model,data,c),'position_accuracy':position_accuracy(model,data,c)})
    from .reporting import report
    report(); write_json(root()/'completion.json',{'status':'completed','source_sha256':code_hash(),'decision':decision})
    return 'completed'
