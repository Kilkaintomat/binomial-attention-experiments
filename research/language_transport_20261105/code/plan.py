"""Small prespecified matrix; deep tests never enter selection."""
from common import SIDE,ART,read,rows,freeze,atomic

SEEDS=[11,22,33]
METHODS=['baseline','binomial','geometric','looped']
TASKS=['vt','phop_conditional','phop_fixed16','phop_fixed32']


def spec(model,task,method,seed,layers=2,phase='main',lr=.001,steps=8000):
    return dict(id='%s/%s/%s/%s_l%d/seed%d'%(phase,model,task,method,layers,seed),
        phase=phase,model=model,task=task,method=method,seed=seed,layers=layers,
        r=8 if method in ['binomial','geometric'] else 1,loops=2 if method=='looped' else 1,
        lr=lr,steps=steps,batch=32 if model=='scratch' else 8,micro=8 if model=='scratch' else 1,
        validation_steps=[x for x in [500,1000,2000,4000,8000,16000] if x<=steps],validation_n=64)


def pilots():
    result=[]
    for lr in [.0003,.001]:
        for method in ['baseline','binomial']:
            x=spec('scratch','vt',method,901,phase='pilot_lr'+str(lr),lr=lr,steps=2000)
            x['validation_steps']=[500,1000,2000];result.append(x)
    for method in ['baseline','binomial']:
        result.append(spec('scratch','phop_conditional',method,901,phase='pilot',lr=.001,steps=2000))
    return result


def completed_training(s):
    p=SIDE/'runs'/s['id']/'training.json'
    return p.exists() and read(p).get('status')=='completed' and read(p)['step']==s['steps']


def complete(s):
    p=SIDE/'runs'/s['id']/'result.json'
    return (s.get('frozen') or completed_training(s)) and p.exists() and read(p).get('status')=='completed'


def selection():
    path=ART/'selected_protocol.json'
    if path.exists():return read(path)
    candidates=[]
    for s in pilots():
        if s['task']=='vt' and s['method']=='baseline':
            v=rows(SIDE/'runs'/s['id']/'validation.jsonl')[-1]
            candidates.append((v['accuracy'],-s['lr'],s,v))
    _,_,best,val=max(candidates,key=lambda t:(t[0],t[1]))
    choice=dict(lr=best['lr'],r_train=8,steps=8000 if val['accuracy']>=.5 else 16000,
        basis='Last validation accuracy of ordinary VT baseline, ties prefer lower LR; only depths <=4',
        pilot_id=best['id'],pilot_accuracy=val['accuracy'],deep_test_used=False,
        r_choice='r=8 fixed a priori; no inference-grid or test-based selection')
    freeze(path,choice)
    return choice


def scratch():
    c=selection()
    return [spec('scratch',task,method,seed,lr=c['lr'],steps=c['steps'])
            for task in TASKS for method in METHODS for seed in SEEDS]


def pretrained():
    out=[]
    for task in ['vt','phop_conditional']:
        frozen=spec('qwen',task,'baseline',0,phase='frozen',steps=0)
        frozen['frozen']=True;out.append(frozen)
        for method,nlayers in [('baseline',2),('binomial',2),('binomial',4)]:
            for seed in SEEDS:out.append(spec('qwen',task,method,seed,nlayers,lr=1e-4,steps=2000))
    return out


def lm_ready():
    checks=[]
    for task in ['vt','phop_conditional']:
        for s in scratch():
            if s['task']!=task or s['method']!='baseline':continue
            val=rows(SIDE/'runs'/s['id']/'validation.jsonl')[-1]
            pool=rows(SIDE/'data'/task/'validation.jsonl')[:64]
            majority=max(sum(x['answer']==a for x in pool) for a in {x['answer'] for x in pool})/len(pool)
            checks.append(dict(task=task,seed=s['seed'],accuracy=val['accuracy'],majority=majority,
                passed=val['accuracy']>=max(.25,majority+.10)))
    ok=all(x['passed'] for x in checks)
    atomic(ART/'lm_learnability_gate.json',dict(passed=ok,checks=checks,deep_test_used=False))
    return ok


def next_stage():
    if not (ART/'gpu_preflight.json').exists():return 'preflight',[]
    if read(ART/'gpu_preflight.json')['status']!='passed':return 'blocked_preflight',[]
    p=pilots()
    if not all(completed_training(s) for s in p):return 'pilot',p
    main=scratch()
    freeze(ART/'scratch_matrix.json',main)
    if not all(complete(s) for s in main):return 'scratch',main
    if not lm_ready():return 'blocked_learnability',[]
    lm=pretrained();freeze(ART/'pretrained_matrix.json',lm)
    if not all(complete(s) for s in lm):return 'pretrained',lm
    return 'completed',[]
