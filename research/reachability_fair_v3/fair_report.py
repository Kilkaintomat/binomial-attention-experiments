"""Independent saved-prediction audit and fixed comparison tables; no inference."""
import csv
import itertools
from datetime import datetime, timezone
from fair_common import *
from fair_runner import calibrated_threshold, threshold_metrics
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.metrics import binary_metrics

CONTRASTS = [('R16-one','F4-bin'),('R16-one','F4-exp'),('R16-one','F4-geo'),
             ('R16-one','F4-last-r4'),('R32-one','F4-bin-r8'),
             ('F4-bin','F4-last-r4'),('F4-exp','F4-last-r4'),('F4-geo','F4-last-r4'),
             ('R32-one','R16-one'),('F4-bin-r8','F4-bin')]


def equal_metrics(actual, expected):
    for k,v in expected.items():
        if v is None:assert actual[k] is None,k
        elif isinstance(v,(int,np.integer)):assert actual[k]==v,k
        else:assert np.isclose(actual[k],v,atol=1e-10,rtol=1e-10),(k,actual[k],v)


def csv_write(name, rows):
    path=OUT/'report/tables'/name;path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def audit_predictions(path, data):
    a=np.load(path,allow_pickle=False)
    assert len(a['labels'])==len(data) and np.isfinite(a['logits']).all()
    for offset in range(0,len(data),1024):
        ids=np.arange(offset,min(offset+1024,len(data)));_,_,y,meta=data.rows(ids)
        assert np.array_equal(a['labels'][ids],y)
        for key,field in [('graph_ids','graph_id'),('query_ids','query_id'),('n','n'),
                          ('distance','distance'),('weak_same','weak_same'),('reverse','reverse')]:
            assert np.array_equal(a[key][ids],np.array([r[field] for r in meta])),key
        assert np.array_equal(a['ell'][ids],np.array([r.get('ell') or -1 for r in meta]))
        assert np.array_equal(a['variant'][ids],np.array([r.get('variant') or '' for r in meta]))
    return a


def report():
    allocation();torch.set_num_threads(1);frozen=freeze_protocol()
    completion=read(OUT/'completion.json');assert completion['status']=='completed'
    checkpoint_manifest=read(OUT/'frozen_checkpoints.json');assert len(checkpoint_manifest)==80
    manifest=read(OUT/'data/manifest.json');assert completion['data_hash']==manifest['data_hash']
    # Verify immutable data files before trusting metadata in saved predictions.
    for p in (OUT/'data').glob('**/manifest.json'):
        for filename,sha in read(p).get('files',{}).items():assert file_hash(p.parent/filename)==sha
    assert read(BASE_ROOT/'data/manifest.json')['data_hash']==BASE_DATA
    pilots=[read(p) for stage in ['pilot','confirmation'] for p in (OUT/'runs'/stage).glob('**/result.json')]
    assert len(pilots)==18
    assert all(p['status']=='completed' and p['step']==3000 and p['source_hash']==extension_hash() for p in pilots)
    rows=[];timings=[];training=[];cache={};entries=0;cal_entries=0;workload_count=0
    for c,baseline in all_final_configs():
        directory=base_directory(c) if baseline else c.directory()
        expected_sha=checkpoint_manifest[c.hash()]['checkpoint_sha256']
        assert file_hash(directory/'checkpoint_best.pt')==expected_sha
        if baseline:assert expected_sha==read(BASE_ROOT/'frozen_checkpoints.json')[c.hash()]['checkpoint_sha256']
        result=read(directory/'result.json');assert result['status']=='completed' and result['step']==30000
        assert result['config']==c.dict() and result['data_hash']==BASE_DATA
        assert result['source_hash']==(BASE_SOURCE if baseline else extension_hash())
        training.append({'family':c.family,'model':c.model,'seed':c.seed,'reused':baseline,
            'best_step':result['best_step'],'lr':c.lr,**result['times'],**result['operations']})
        cp=OUT/'calibration'/c.family/c.model/('seed_'+str(c.seed)+'.json');cal=read(cp)
        assert cal['checkpoint_sha256']==expected_sha and cal['source_hash']==extension_hash()
        assert cal['predictions_sha256']==file_hash(cp.with_suffix('.npz'))
        val_key=(c.family,'validation',None)
        if val_key not in cache:cache[val_key]=Dataset(c.family,'validation')
        pred=audit_predictions(cp.with_suffix('.npz'),cache[val_key]);cal_entries+=len(pred['labels'])
        threshold,info=calibrated_threshold(pred['labels'],pred['logits'])
        assert threshold==cal['threshold'] and info['validation_BA']==cal['validation_BA']
        equal_metrics(cal['default_validation'],binary_metrics(pred['labels'],pred['logits']))
        tp=OUT/'timing'/c.family/c.model/('seed_'+str(c.seed)+'.json');t=read(tp)
        assert t['checkpoint_sha256']==expected_sha and t['source_hash']==extension_hash()
        assert {(x['n'],x['batch_size']) for x in t['rows']}=={(n,b) for n in [32,64,128] for b in [1,64]}
        for x in t['rows']:timings.append({'family':c.family,'model':c.model,'seed':c.seed,**x,**result['operations']})
        for split,n in workloads():
            p=eval_path(c,split,n);metric=read(p)
            assert metric['config']==c.dict() and metric['checkpoint_sha256']==expected_sha
            assert metric['source_hash']==extension_hash() and metric['data_hash']==manifest['data_hash']
            assert metric['predictions_sha256']==file_hash(p.with_suffix('.npz'))
            assert metric['learned_parameters_unchanged'] and metric['validation_threshold']==threshold
            key=('stress' if split=='stress' else c.family,split,n)
            if key not in cache:cache[key]=TestDataset(*key)
            pred=audit_predictions(p.with_suffix('.npz'),cache[key]);y=pred['labels'];z=pred['logits']
            entries+=len(y);workload_count+=1
            equal_metrics(metric,binary_metrics(y,z));equal_metrics(metric['validation_threshold_metrics'],threshold_metrics(y,z,threshold))
            assert sum(x['count'] for x in metric['by_group'])==len(y)
            for cell in metric['by_group']:
                ids=(pred['n']==cell['n']) & (pred['ell']==(cell['ell'] or -1)) & (pred['variant']==(cell['variant'] or ''))
                yy,zz=y[ids],z[ids];equal_metrics(cell,binary_metrics(yy,zz))
                equal_metrics(cell['validation_threshold_metrics'],threshold_metrics(yy,zz,threshold))
                row={'family':c.family,'model':c.model,'seed':c.seed,'split':split,'n':cell['n'],
                     'ell':cell['ell'],'variant':cell['variant'],'radius':c.outer_steps*c.hops,
                     **{k:cell[k] for k in binary_metrics(yy,zz)},'validation_threshold':threshold,
                     'calibrated_BA':cell['validation_threshold_metrics']['balanced_accuracy']}
                if split=='stress':
                    assert np.array_equal(yy.reshape(-1,2),np.tile([1.,0.],(len(yy)//2,1)))
                    assert np.array_equal(pred['graph_ids'][ids][::2],pred['graph_ids'][ids][1::2])
                    margin=zz.astype(np.float64).reshape(-1,2)[:,0]-zz.astype(np.float64).reshape(-1,2)[:,1]
                    for field,threshold_value in [('both_queries_correct',0.),('both_queries_correct_calibrated',threshold)]:
                        score=float(((zz.astype(np.float64)>=threshold_value)==(yy==1)).reshape(-1,2).all(1).mean())
                        assert score==cell[field];row[field]=score
                    expected={'median':np.median(margin),'p10':np.quantile(margin,.1),'p90':np.quantile(margin,.9),
                        'minimum':margin.min(),'maximum':margin.max(),'fraction_positive':(margin>0).mean(),
                        'fraction_gt_1e-6':(margin>1e-6).mean()}
                    equal_metrics(cell['logit_margin'],expected)
                    row.update({'margin_'+k:float(v) for k,v in expected.items()})
                rows.append(row)
    assert workload_count==320 and entries==1728000 and cal_entries==3200000
    # One independent unit is a training seed. Never treat sizes or stress cells as extra seeds.
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in ['family','model','split','n','ell','variant']),[]).append(row)
    aggregates=[]
    for key,values in groups.items():
        assert sorted(r['seed'] for r in values)==SEEDS
        row=dict(zip(['family','model','split','n','ell','variant'],key));row['seeds']=5
        for field in ['balanced_accuracy','calibrated_BA','recall','specificity','auroc','bce',
                      'both_queries_correct','both_queries_correct_calibrated','margin_median','margin_fraction_gt_1e-6']:
            if all(v.get(field) is not None for v in values):
                a=np.array([v[field] for v in values]);row[field+'_mean']=float(a.mean());row[field+'_sd']=float(a.std(ddof=1))
        aggregates.append(row)
    look={tuple(r[k] for k in ['family','model','split','n','ell','variant','seed']):r for r in rows}
    resamples=np.array(list(itertools.product(range(5),repeat=5)));contrasts=[]
    cells=sorted({(r['family'],r['split'],r['n'],r['ell'] or -1,r['variant'] or '') for r in rows})
    for family,split,n,ell,variant in cells:
        for a,b in CONTRASTS:
            for field in ['balanced_accuracy','calibrated_BA']:
                delta=np.array([look[(family,a,split,n,None if ell==-1 else ell,variant or None,s)][field]-
                                look[(family,b,split,n,None if ell==-1 else ell,variant or None,s)][field] for s in SEEDS])
                boot=delta[resamples].mean(1)
                contrasts.append({'family':family,'split':split,'n':n,'ell':None if ell==-1 else ell,'variant':variant,
                    'model_a':a,'model_b':b,'metric':field,'mean_delta_pp':100*float(delta.mean()),
                    'sample_sd_pp':100*float(delta.std(ddof=1)),
                    'paired_seed_lower95_pp':100*float(np.quantile(boot,.025)),
                    'paired_seed_upper95_pp':100*float(np.quantile(boot,.975)),'seeds':5,'resamples':3125})
    for name,values in [('per_seed.csv',rows),('aggregate.csv',aggregates),('paired_comparisons.csv',contrasts),
                        ('timing.csv',timings),('training.csv',training)]:csv_write(name,values)
    audit={'status':'passed','snapshot_utc':datetime.now(timezone.utc).isoformat(),'checkpoints':80,
        'new_final_trainings':30,'reused_final_trainings':50,'test_workloads':workload_count,
        'test_predictions_checked':entries,'validation_predictions_checked':cal_entries,
        'source_hash':extension_hash(),'data_hash':manifest['data_hash'],'protocol_hash':frozen['protocol_hash'],
        'scheduler_job':os.environ['SLURM_JOB_ID'],'new_inference':False}
    write_json(OUT/'report/audit.json',audit)
    text=['# Сравнение при одинаковом максимальном числе переходов',
        '\nСнимок UTC: '+audit['snapshot_utc']+'. Аудит сохранённых предсказаний завершён.',
        '\n30 новых финальных обучений, 18 пилотов и подтверждений; 50 старых финальных моделей повторно проверены на новом тесте. '
        'Каждая новая модель обучена 30 000 шагов. Все пять seeds сохранены: 11, 23, 37, 53, 71.',
        '\nR16 сравнивается с четырёхблочными фильтрами r=4 (радиус16); R32 — с биномиальной r=8 (радиус32). '
        'Контроль A⁴V имеет те же четыре независимых блока и число переносов, что фильтры r=4. '
        'R16/R32 используют общие веса, F4 — независимые. Совпадение радиуса не означает равенства времени, числа attention/FFN или параметров.',
        '\nОбучение и выбор LR/checkpoint используют только исходные train/validation. Новый тест открыт после фиксации всех80 checkpoints. '
        'Основная метрика — balanced accuracy при logit≥0. Дополнительный порог выбран отдельно для каждого checkpoint по полной исходной validation, без тестовых меток.',
        '\nТест: 7800 новых размеченных графов, 33600 уникальных запросов. Длины stress: '+', '.join(map(str,LENGTHS))+'. '
        'В каждом stress-условии50 пар положительного/отрицательного запросов. Plain-графы являются диагностикой радиуса, '
        'перестановки вершин не создают независимые видимые топологии. Branch-варианты добавляют случайные входящие деревья, сохраняя длину кратчайшего пути и парный контроль степеней источника. '
        'Проверено отсутствие совпадающих размеченных матриц смежности с v1/v2 и внутри v3; изоморфная независимость не утверждается.',
        '\n## Случайные графы: BA, % (среднее ± выборочное SD по пяти seeds)',
        '\n| Обучение | Модель | n=32 | n=64 | n=128 |\n|---|---|---:|---:|---:|']
    def fmt(row,metric='balanced_accuracy'):
        return f"{100*row[metric+'_mean']:.2f} ± {100*row[metric+'_sd']:.2f}"
    for family in FAMILIES:
        for model in BASE_MODELS+NEW_MODELS:
            cells_a=[next(r for r in aggregates if r['family']==family and r['model']==model and r['split']=='far' and r['n']==n) for n in [32,64,128]]
            text.append('| '+family+' | '+model+' | '+' | '.join(fmt(r) for r in cells_a)+' |')
    for family in FAMILIES:
        for variant in VARIANTS:
            text+=['\n## Stress: '+family+', '+variant,
                '\n| Модель | ℓ=8 | ℓ=12 | ℓ=16 | ℓ=20 | ℓ=24 | ℓ=32 | ℓ=48 |\n|---|---:|---:|---:|---:|---:|---:|---:|']
            for model in BASE_MODELS+NEW_MODELS:
                cs=[next(r for r in aggregates if r['family']==family and r['model']==model and r['variant']==variant and r['ell']==ell) for ell in [8,12,16,20,24,32,48]]
                text.append('| '+model+' | '+' | '.join(fmt(r) for r in cs)+' |')
    text+=['\n## Интерпретация и файлы',
        '\n50% BA может означать постоянный ответ одного класса; это проверяется по TP/TN/FP/FN, AUROC, парным зазорам логитов и validation-порогу. '
        '70% при пяти seeds может быть средним нескольких дискретных исходов. CSV содержит все seeds и все длины, включая неудобные результаты.',
        '\n`tables/per_seed.csv`: полные метрики каждого seed, матрицы ошибок, калибровка и зазоры; '
        '`aggregate.csv`: средние/SD; `paired_comparisons.csv`: десять заранее заданных парных сравнений, '
        'все3125 bootstrap-перевыборок пяти seeds; `timing.csv`: фактические задержки/пропускная способность/память и операции; '
        '`training.csv`: время обучения и выбранные checkpoints. Интервалы исследовательские, без поправки на множественные сравнения. '
        'Они описывают вариацию seeds на фиксированных тестах и не заменяют неопределённость по новым графовым топологиям.',
        '\nАудит: 1 728 000 тестовых и 3 200 000 validation-предсказаний; метки/ID/расстояния проверены по данным, '
        'метрики пересчитаны, hashes данных, кода, предсказаний и80 checkpoints проверены. Ресурсы учитываются отдельно по Slurm в queue/state.json.']
    (OUT/'report/REPORT_FAIR_COMPARISON_RU.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':report()
