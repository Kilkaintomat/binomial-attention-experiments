"""Budget admission before test, symmetric families, and round-wise seeds."""
import os
import time
from .config import PRIORITY,FAMILIES,SEEDS,make,freeze_protocol,read,root,source_hash,write_json,file_hash,allocation


def build_grid():
    freeze_protocol();rows=[]
    for model in PRIORITY:
        for family in FAMILIES:
            for lr in [.0003,.001]:
                c=make(family,model,101,'pilot',lr);p=root()/'configs'/('pilot_'+c.hash()+'.json');write_json(p,c.dict())
                rows.append({'stage':'pilot','family':family,'model':model,'seed':101,'lr':lr,'config_file':str(p.relative_to(root()))})
            rows.append({'stage':'confirmation','family':family,'model':model,'seed':103,'lr':'selected_by_pilot'})
    for seed in SEEDS:
        for model in PRIORITY:
            for family in FAMILIES:rows.append({'stage':'final','family':family,'model':model,'seed':seed,'lr':'selected_by_pilot'})
    write_json(root()/'campaign_grid.json',{'rows':rows,'full_neural_training_count':144,'source_hash':source_hash()})
    return {'pilot_count':54,'final_count':90,'full_training_count':144}


def choose_admission(profile,remaining):
    costs={(r['family'],r['model']):r for r in profile['rows']}
    def cost(models,seeds):return 1.2*sum(3*costs[f,m]['pilot_seconds_estimate']+len(seeds)*costs[f,m]['final_including_evaluation_seconds_estimate'] for m in models for f in FAMILIES)
    chosen=None
    # Prefer five seeds on the longest affordable model-priority prefix.
    for length in range(2,len(PRIORITY)+1):
        if cost(PRIORITY[:length],SEEDS)<=remaining:chosen=(PRIORITY[:length],SEEDS)
    if chosen is None:
        for rounds in range(1,len(SEEDS)+1):
            if cost(PRIORITY[:2],SEEDS[:rounds])<=remaining:chosen=(PRIORITY[:2],SEEDS[:rounds])
    if chosen is None:return {'status':'budget_stopped','minimum_required_seconds':cost(PRIORITY[:2],SEEDS[:1]),'remaining_seconds':remaining}
    models,seeds=chosen
    return {'status':'admitted','models':models,'seeds':seeds,'families':FAMILIES,'required_with_reserve_seconds':cost(models,seeds),
            'remaining_seconds_at_admission':remaining,'selection_rule':'longest priority prefix with five seeds; minimum two models, reduced seed rounds only if needed',
            'source_hash':source_hash(),'full_grid_admitted':len(models)==9 and len(seeds)==5,
            'omitted_models':PRIORITY[len(models):],'omitted_seeds':SEEDS[len(seeds):],
            'additional_seconds_for_full_grid':max(0.,profile['full_grid_with_reserve_seconds']-remaining),'before_any_new_test':True}


def run_queue(seconds,remaining,dry_run=False):
    allocation();build_grid();started=time.perf_counter();deadline=started+seconds
    if dry_run:return {'status':'dry_run','models':PRIORITY,'families':FAMILIES,'final_seeds':SEEDS,'ceiling_gpu_hours':24}
    from argparse import Namespace
    from .checks import preflight,profile
    from .training import train,evaluate_config
    if not (root()/'PREFLIGHT_REACHABILITY.json').exists():
        result=preflight(Namespace(seconds=max(0,int(deadline-time.perf_counter()))))
        if result['status']!='passed':return 'paused'
    if time.perf_counter()>deadline-300:return 'paused'
    if not (root()/'profile.json').exists():
        profile(Namespace())
        # Charge this allocation before committing to training cost.
        return 'profile_completed'
    path=root()/'admission.json'
    if not path.exists():
        decision=choose_admission(read(root()/'profile.json'),remaining-(time.perf_counter()-started));write_json(path,decision)
    decision=read(path)
    if decision['status']!='admitted':return 'budget_stopped'
    lrs={};pilot_rows=[]
    for model in decision['models']:
        for family in FAMILIES:
            candidates=[]
            for lr in [.0003,.001]:
                if time.perf_counter()>deadline-120:return 'paused'
                c=make(family,model,101,'pilot',lr);result=train(c,deadline)
                if result['status']!='completed':return 'paused'
                metric=result['final_validation'];assert metric['full_validation'] and metric['step']==3000
                candidates.append((metric['balanced_accuracy'] if metric['balanced_accuracy'] is not None else -1.,-metric['bce'],-lr,lr))
                pilot_rows.append({'family':family,'model':model,'seed':101,'lr':lr,'validation':metric})
            lrs[family+'/'+model]=max(candidates)[-1]
    write_json(root()/'pilot_selection.json',{'source_hash':source_hash(),'learning_rates':lrs,'rows':pilot_rows,'selection_uses':'full final pilot validation only'})
    for model in decision['models']:
        for family in FAMILIES:
            if time.perf_counter()>deadline-120:return 'paused'
            c=make(family,model,103,'confirmation',lrs[family+'/'+model]);result=train(c,deadline)
            if result['status']!='completed':return 'paused'
    finals=[make(family,model,seed,'final',lrs[family+'/'+model]) for seed in decision['seeds'] for model in decision['models'] for family in FAMILIES]
    selection={'source_hash':source_hash(),'admission':decision,'configs':[c.dict() for c in finals],'before_test':True}
    selection_path=root()/'final_selection.json'
    if selection_path.exists() and read(selection_path)!=selection:raise RuntimeError('Final selection changed')
    write_json(selection_path,selection)
    for c in finals:
        write_json(root()/'configs'/('final_'+c.hash()+'.json'),c.dict())
        if time.perf_counter()>deadline-120:return 'paused'
        result=train(c,deadline)
        if result['status']!='completed':return 'paused'
    frozen={c.hash():{'checkpoint_sha256':file_hash(c.directory()/'checkpoint_best.pt'),'step':read(c.directory()/'result.json')['best_step']} for c in finals}
    frozen_path=root()/'frozen_checkpoints.json'
    if frozen_path.exists() and read(frozen_path)!=frozen:raise RuntimeError('Previously frozen best checkpoints changed')
    write_json(frozen_path,frozen)
    for c in finals:
        if time.perf_counter()>deadline-180:return 'paused'
        if not evaluate_config(c,deadline):return 'paused'
    from .reporting import report
    report();write_json(root()/'completion.json',{'status':'completed','source_hash':source_hash(),'admission':decision,'new_final_trainings':len(finals),
                                                'full_protocol_complete':decision['full_grid_admitted']})
    return 'completed'
