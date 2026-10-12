"""Reuse fair-v3 validation predictions to audit thresholds on the original tests."""
from collections import defaultdict
from datetime import datetime,timezone
from c4_common import *
from c4_evaluate import threshold,calibrated,audit_predictions,equal
from transport_reasoning.reachability.metrics import binary_metrics
from transport_reasoning.reachability.config import allocation

FAIR=Path('/fair_previous/experiments/fair_v3')
MODELS=['F4-one','F4-bin','F4-exp','F4-geo','R32-one']


def run():
    allocation();torch.set_num_threads(1);out=ROOT/'no_training';out.mkdir(parents=True,exist_ok=True)
    configs=[BaseConfig(**v) for v in read(BASE/'final_selection.json')['configs'] if v['model'] in MODELS]
    assert len(configs)==50
    missing=[];rows=[];provenance=[];validation={}
    for c in configs:
        p=FAIR/'calibration'/c.family/c.model/('seed_'+str(c.seed)+'.json')
        if not p.exists():missing.append({'family':c.family,'model':c.model,'seed':c.seed});continue
        calibration=read(p);directory=control_directory(c)
        # The directory convention is shared by all original model IDs.
        frozen=read(BASE/'frozen_checkpoints.json')[c.hash()]
        assert calibration['checkpoint_sha256']==frozen['checkpoint_sha256']==file_hash(directory/'checkpoint_best.pt')
        assert calibration['predictions_sha256']==file_hash(p.with_suffix('.npz'))
        if c.family not in validation:validation[c.family]=Dataset(c.family,'validation')
        val=audit_predictions(p.with_suffix('.npz'),validation[c.family]);t,ba=threshold(val['labels'],val['logits'])
        assert t==calibration['threshold'] and ba==calibration['validation_BA']
        provenance.append({'family':c.family,'model':c.model,'seed':c.seed,'validation_threshold':t,
            'checkpoint_sha256':frozen['checkpoint_sha256'],'validation_predictions_sha256':file_hash(p.with_suffix('.npz'))})
        for name in ['fixed_test_ID','fixed_test_far_128','fixed_stress']:
            path=directory/'evaluation'/(name+'.npz');a=np.load(path,allow_pickle=False);y=a['labels'];z=a['logits']
            original=read(path.with_suffix('.json'));equal(original,binary_metrics(y,z))
            groups=[(None,None,None,np.ones(len(y),bool))] if name!='fixed_stress' else [
                (128,ell,variant,(a['n']==128)&(a['ell']==ell)&(a['variant']==variant))
                for ell in [2,4,8,16,32,48] for variant in ['dag','cyclic']]
            for n,ell,variant,ids in groups:
                before=binary_metrics(y[ids],z[ids]);after=calibrated(y[ids],z[ids],t)
                rows.append({'family':c.family,'model':c.model,'seed':c.seed,'evaluation':name,'n':n,'ell':ell,'variant':variant,
                    'threshold':t,'default_BA':before['balanced_accuracy'],'calibrated_BA':after['balanced_accuracy'],
                    'auroc':before['auroc'],'count':before['count'],'tp_default':before['tp'],'tn_default':before['tn'],
                    'fp_default':before['fp'],'fn_default':before['fn'],'tp_calibrated':after['tp'],'tn_calibrated':after['tn'],
                    'fp_calibrated':after['fp'],'fn_calibrated':after['fn'],'predictions_sha256':file_hash(path)})
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in ['family','model','evaluation','n','ell','variant'])].append(r)
    aggregate=[]
    for key,values in groups.items():
        if sorted(v['seed'] for v in values)!=SEEDS:continue
        a={'family':key[0],'model':key[1],'evaluation':key[2],'n':key[3],'ell':key[4],'variant':key[5],'seeds':5}
        for field in ['default_BA','calibrated_BA','auroc']:
            x=np.array([v[field] for v in values]);a[field+'_mean']=float(x.mean());a[field+'_sd']=float(x.std(ddof=1))
        aggregate.append(a)
    status={'status':'completed' if not missing else 'waiting_for_fair_v3_calibration',
        'snapshot_utc':datetime.now(timezone.utc).isoformat(),'completed_checkpoints':len(provenance),'expected_checkpoints':50,
        'missing':missing,'source_hash':file_hash(__file__),'scheduler_job':os.environ['SLURM_JOB_ID'],
        'new_training':False,'new_inference':False,'new_test_accessed':False,
        'validation_source':'original n18/19, already computed by fair-v3; never fit threshold on test'}
    write_json(out/'status.json',status);write_json(out/'per_seed.json',rows);write_json(out/'aggregate.json',aggregate);write_json(out/'provenance.json',provenance)
    if not missing:
        assert len(rows)==700
        text=['# Проверка порога без нового обучения или инференса','\nСнимок UTC: '+status['snapshot_utc'],
            '\nИспользованы50 исходных лучших checkpoints и их validation-предсказания из fair-v3. '
            'Для каждого checkpoint порог максимизирует BA на исходной validation n18/19; при равенстве выбран ближайший к0, затем меньший. '
            'Затем пересчитаны исходные ID, random far n128 и stress n128 тесты. Тестовые метки не участвуют в выборе порога.',
            '\n## Ациклический stress ℓ16, n128: BA, среднее пяти seeds',
            '\n| Обучение | Модель | Исходный порог | Порог validation | AUROC |\n|---|---|---:|---:|---:|']
        for family in FAMILIES:
            for model in MODELS:
                r=next(r for r in aggregate if r['family']==family and r['model']==model and r['evaluation']=='fixed_stress' and r['ell']==16 and r['variant']=='dag')
                text.append(f"| {family} | {model} | {100*r['default_BA_mean']:.2f}% | {100*r['calibrated_BA_mean']:.2f}% | {100*r['auroc_mean']:.2f}% |")
        text += ['\nПолные значения и все seeds сохранены в per_seed.json и aggregate.json. '
            'Высокий AUROC может сопровождаться очень малыми зазорами; улучшение или отсутствие улучшения BA следует оценивать отдельно. '
            'Исходная validation не содержит столь длинных положительных путей, поэтому отсутствие исправления не исключает пользу другой заранее заданной калибровочной выборки.',
            '\nУвеличение r и перевзвешивание exponential/geometric уже проверены в завершённой серии reweight-v2: '
            'https://github.com/Kilkaintomat/binomial-attention-experiments/blob/main/REPORT_REWEIGHT_RU.md. '
            'Её результаты не требуют повторного запуска. Общая проверка новых режимов обучения при r4/8/16/32 выполняется отдельным закрытым тестом curriculum-v4.']
        (out/'REPORT_THRESHOLD_RU.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in status.items() if k!='missing'}),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--check']:
        from c4_eval_checks import run as check
        check()
    else:
        assert not sys.argv[1:]
        run()
