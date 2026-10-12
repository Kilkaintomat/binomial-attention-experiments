"""Frozen-checkpoint evaluation, validation-only calibration and prediction audit."""
import argparse
import csv
import hashlib
import itertools
import time
from datetime import datetime,timezone
from c4_common import *
from c4_eval_data import TestDataset,TEST,prepare,freeze_evaluation,evaluation_hash,workloads
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.metrics import binary_metrics,slices
from transport_reasoning.reachability.checks import timing


def threshold(y,z):
    y=np.asarray(y,dtype=np.int64);z=np.asarray(z,dtype=np.float64)
    assert np.isfinite(z).all() and set(np.unique(y))=={0,1}
    order=np.argsort(z,kind='stable');zs=z[order];ys=y[order];prefix=np.r_[0,np.cumsum(ys)]
    positive=int(y.sum());negative=len(y)-positive;candidates=np.unique(np.r_[0.,zs,np.nextafter(zs[-1],np.inf)])
    cut=np.searchsorted(zs,candidates,side='left');tp=positive-prefix[cut];tn=cut-prefix[cut];score=tp*negative+tn*positive
    choice=min(np.flatnonzero(score==score.max()),key=lambda i:(abs(candidates[i]),candidates[i]))
    return float(candidates[choice]),float(score[choice]/(2*positive*negative))


def calibrated(y,z,t):
    m=binary_metrics(y,np.asarray(z,dtype=np.float64)-t)
    return {k:m[k] for k in ['count','positive_count','negative_count','balanced_accuracy','accuracy','precision','recall','specificity','f1','tp','tn','fp','fn']}


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def all_configs():
    assert read(ROOT/'training_completed.json')['status']=='completed'
    out=[(c,'original',True) for c in controls()]
    out += [(Config(**v),v['regime'],False) for v in read(ROOT/'final_selection.json')['configs']]
    assert len(out)==40
    return sorted(out,key=lambda x:(x[0].seed,x[0].family,x[1]))


def load(c,regime,baseline,device):
    frozen=read(ROOT/'frozen_checkpoints.json')[c.hash()];p=(control_directory(c) if baseline else c.directory())/'checkpoint_last.pt'
    assert frozen['config']==c.dict() and frozen['regime']==regime and file_hash(p)==frozen['checkpoint_sha256']
    state=torch.load(p,map_location='cpu',weights_only=False)
    expected_data=BASE_DATA if baseline else digest({'original':BASE_DATA,'pool':read(ROOT/'data/manifest.json')['data_hash'],
        'regime':regime,'family':c.family,'sampler':'v1_virtual_2pow50_lowbits_class_origin_then_length_graph'})
    assert state['step']==30000 and state['config']==c.dict() and state['data_hash']==expected_data
    assert state['source_hash']==(BASE_SOURCE if baseline else source_hash())
    model=Encoder(c).to(device);model.load_state_dict(state['model'],strict=True);model.eval();model.requires_grad_(False)
    return model,frozen


def path_for(c,regime,r,split,n):return ROOT/'evaluations'/c.family/regime/('seed_'+str(c.seed))/('r'+str(r))/(split+'_'+str(n or 'all')+'.json')


def calibration_path(c,regime,r,kind):return ROOT/'calibration'/c.family/regime/('seed_'+str(c.seed))/('r'+str(r))/(kind+'.json')


def benchmarks(r,device):
    p=ROOT/'timing'/('r'+str(r)+'.json')
    if p.exists():
        t=read(p);assert t['evaluation_source_hash']==evaluation_hash();return t
    c=next(c for c in controls() if c.family=='er' and c.seed==11);model,frozen=load(c,'original',True,device)
    model.extra_compute(hops=r)
    t={'rows':[timing(model,n,b,device,warmup=10,repeats=30) for n in TEST['random_sizes'] for b in [1,64]],
        'reference_family':'er','reference_regime':'original','reference_seed':11,'checkpoint_sha256':frozen['checkpoint_sha256'],
        'r':r,'evaluation_source_hash':evaluation_hash(),'operations':model.operations(),
        'scope':'same F4-bin architecture and dimensions in every regime; shared benchmark, not per-checkpoint timing'}
    write_json(p,t);return t


def evaluate(seconds,remaining):
    allocation();freeze_evaluation();deadline=time.perf_counter()+seconds
    assert read(ROOT/'test_ready.json')['evaluation_source_hash']==evaluation_hash()
    manifest=read(ROOT/'test_data/manifest.json');cache={};done=0
    for c,regime,baseline in all_configs():
        for r in TEST['hops']:
            paths=[path_for(c,regime,r,split,n) for split,n in workloads()]
            if all(p.exists() for p in paths):
                for p in paths:
                    m=read(p);assert m['evaluation_source_hash']==evaluation_hash() and m['data_hash']==manifest['data_hash']
                    assert m['checkpoint_sha256']==read(ROOT/'frozen_checkpoints.json')[c.hash()]['checkpoint_sha256']
                    assert m['predictions_sha256']==file_hash(p.with_suffix('.npz'))
                done+=4;continue
            if time.perf_counter()>deadline-180:return 'paused'
            device=trainer.setup(c);bench=benchmarks(r,device);model,frozen=load(c,regime,baseline,device)
            learned=parameter_hash(model);model.extra_compute(hops=r);cal={}
            for kind in ['original','augmented']:
                p=calibration_path(c,regime,r,kind)
                if p.exists():
                    cal[kind]=read(p);assert cal[kind]['checkpoint_sha256']==frozen['checkpoint_sha256']
                    assert cal[kind]['evaluation_source_hash']==evaluation_hash() and cal[kind]['r']==r
                    assert cal[kind]['predictions_sha256']==file_hash(p.with_suffix('.npz'))
                    continue
                key=(c.family,'calibration',kind)
                if key not in cache:cache[key]=Dataset(c.family,'validation') if kind=='original' else PoolDataset(c.family,'validation')
                m,y,z,_=trainer.predict(model,cache[key],device,64,p.with_suffix('.npz'));t,ba=threshold(y,z)
                cal[kind]={'threshold':t,'validation_BA_at_selected_threshold':ba,'kind':kind,'r':r,
                    'validation_metrics_at0':m,'checkpoint_sha256':frozen['checkpoint_sha256'],
                    'evaluation_source_hash':evaluation_hash(),'predictions_sha256':file_hash(p.with_suffix('.npz')),
                    'selection':'maximum validation BA, tie closest0 then lower; no test data'}
                write_json(p,cal[kind])
            for split,n in workloads():
                p=path_for(c,regime,r,split,n)
                if p.exists():done+=1;continue
                if time.perf_counter()>deadline-150:return 'paused'
                key=('stress' if split=='stress' else c.family,split,n)
                if key not in cache:cache[key]=TestDataset(*key)
                metric,y,z,meta=trainer.predict(model,cache[key],device,64,p.with_suffix('.npz'))
                assert parameter_hash(model)==learned
                groups={}
                for i,m in enumerate(meta):groups.setdefault((m['n'],m['ell'],m['variant']),[]).append(i)
                cells=[]
                for (nn,ell,variant),ids in groups.items():
                    ids=np.array(ids);yy=y[ids];zz=z[ids].astype(np.float64)
                    cell={'n':nn,'ell':ell,'variant':variant,**binary_metrics(yy,zz),
                        'original_validation_threshold_metrics':calibrated(yy,zz,cal['original']['threshold']),
                        'augmented_validation_threshold_metrics':calibrated(yy,zz,cal['augmented']['threshold'])}
                    if split=='stress':
                        assert np.array_equal(yy.reshape(-1,2),np.tile([1.,0.],(len(yy)//2,1)))
                        assert all(meta[int(a)]['graph_id']==meta[int(b)]['graph_id'] for a,b in ids.reshape(-1,2))
                        margin=zz.reshape(-1,2)[:,0]-zz.reshape(-1,2)[:,1]
                        cell['both_queries_correct']=float(((zz>=0)==(yy==1)).reshape(-1,2).all(1).mean())
                        cell['paired_logit_margin']={'median':float(np.median(margin)),'p10':float(np.quantile(margin,.1)),
                            'p90':float(np.quantile(margin,.9)),'min':float(margin.min()),'max':float(margin.max()),
                            'fraction_positive':float((margin>0).mean()),'fraction_gt_1e_6':float((margin>1e-6).mean())}
                    cells.append(cell)
                metric.update(family=c.family,regime=regime,seed=c.seed,config=c.dict(),r=r,radius=4*r,split=split,n=n,
                    checkpoint_step=30000,checkpoint_sha256=frozen['checkpoint_sha256'],baseline_reused=baseline,
                    training_source_hash=BASE_SOURCE if baseline else source_hash(),evaluation_source_hash=evaluation_hash(),
                    data_hash=manifest['data_hash'],predictions_sha256=file_hash(p.with_suffix('.npz')),
                    learned_parameter_sha256=learned,learned_parameters_unchanged=True,
                    coefficients=model.blocks[0].coeff().detach().cpu().tolist(),operations=model.operations(),by_group=cells,
                    original_validation_threshold=cal['original']['threshold'],augmented_validation_threshold=cal['augmented']['threshold'],
                    original_validation_threshold_metrics=calibrated(y,z,cal['original']['threshold']),
                    augmented_validation_threshold_metrics=calibrated(y,z,cal['augmented']['threshold']),
                    path_length_slices=slices(y,z,meta),timing_reference='timing/r'+str(r)+'.json',scheduler_job=os.environ['SLURM_JOB_ID'])
                write_json(p,metric);done+=1
                write_json(ROOT/'progress.json',{'stage':'evaluation','completed':done,'expected':640,'family':c.family,'regime':regime,'seed':c.seed,'r':r})
            assert file_hash((control_directory(c) if baseline else c.directory())/'checkpoint_last.pt')==frozen['checkpoint_sha256']
            del model;torch.cuda.empty_cache()
    assert done==640
    write_json(ROOT/'completion.json',{'status':'completed','new_finals':30,'reused_controls':10,'evaluations':640,
        'evaluation_source_hash':evaluation_hash(),'data_hash':manifest['data_hash'],'scheduler_job':os.environ['SLURM_JOB_ID']})
    return 'completed'


def equal(actual,expected):
    for k,v in expected.items():
        if v is None:assert actual[k] is None,k
        elif isinstance(v,(int,np.integer)):assert actual[k]==v,k
        else:assert np.isclose(actual[k],v,atol=1e-10,rtol=1e-10),(k,actual[k],v)


def audit_predictions(path,data):
    a=np.load(path,allow_pickle=False);assert len(a['labels'])==len(data) and np.isfinite(a['logits']).all()
    for start in range(0,len(data),1024):
        ids=np.arange(start,min(start+1024,len(data)));_,_,y,meta=data.rows(ids);assert np.array_equal(a['labels'][ids],y)
        for k,f in [('graph_ids','graph_id'),('query_ids','query_id'),('n','n'),('distance','distance'),('weak_same','weak_same'),('reverse','reverse')]:
            assert np.array_equal(a[k][ids],np.array([m[f] for m in meta])),k
        assert np.array_equal(a['ell'][ids],np.array([m.get('ell') or -1 for m in meta]))
        assert np.array_equal(a['variant'][ids],np.array([m.get('variant') or '' for m in meta]))
    return a


def csv_write(name,rows):
    p=ROOT/'report/tables'/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def report():
    allocation();torch.set_num_threads(1);freeze_evaluation();assert read(ROOT/'completion.json')['status']=='completed'
    rows=[];training=[];cache={};entries=cal_entries=count=0;frozen=read(ROOT/'frozen_checkpoints.json')
    for base in [ROOT/'data',ROOT/'test_data']:
        for p in base.glob('**/manifest.json'):
            for name,sha in read(p).get('files',{}).items():assert file_hash(p.parent/name)==sha
    test_hash=read(ROOT/'test_data/manifest.json')['data_hash']
    for c,regime,baseline in all_configs():
        directory=control_directory(c) if baseline else c.directory();sha=frozen[c.hash()]['checkpoint_sha256']
        assert file_hash(directory/'checkpoint_last.pt')==sha
        train=read(directory/'result.json');assert train['status']=='completed' and train['step']==30000 and train['config']==c.dict()
        training.append({'family':c.family,'regime':regime,'seed':c.seed,'steps':30000,'lr':c.lr,**train['times'],**train['operations']})
        for r in TEST['hops']:
            thresholds={}
            for kind in ['original','augmented']:
                p=calibration_path(c,regime,r,kind);m=read(p)
                assert m['checkpoint_sha256']==sha and m['evaluation_source_hash']==evaluation_hash() and m['r']==r
                assert m['predictions_sha256']==file_hash(p.with_suffix('.npz'))
                key=(c.family,'calibration',kind)
                if key not in cache:cache[key]=Dataset(c.family,'validation') if kind=='original' else PoolDataset(c.family,'validation')
                a=audit_predictions(p.with_suffix('.npz'),cache[key]);cal_entries+=len(a['labels'])
                t,ba=threshold(a['labels'],a['logits']);assert t==m['threshold'] and ba==m['validation_BA_at_selected_threshold']
                equal(m['validation_metrics_at0'],binary_metrics(a['labels'],a['logits']));thresholds[kind]=t
            for split,n in workloads():
                p=path_for(c,regime,r,split,n);m=read(p)
                assert m['config']==c.dict() and m['checkpoint_sha256']==sha and m['evaluation_source_hash']==evaluation_hash() and m['data_hash']==test_hash
                assert m['predictions_sha256']==file_hash(p.with_suffix('.npz')) and m['learned_parameters_unchanged']
                assert np.array_equal(np.array(m['coefficients'],np.float32),coefficients('binomial',r).float().numpy())
                key=('stress' if split=='stress' else c.family,split,n)
                if key not in cache:cache[key]=TestDataset(*key)
                a=audit_predictions(p.with_suffix('.npz'),cache[key]);y=a['labels'];z=a['logits'];entries+=len(y);count+=1
                equal(m,binary_metrics(y,z))
                for kind,t in thresholds.items():
                    assert m[kind+'_validation_threshold']==t
                    equal(m[kind+'_validation_threshold_metrics'],calibrated(y,z,t))
                assert sum(cell['count'] for cell in m['by_group'])==len(y)
                for cell in m['by_group']:
                    ids=(a['n']==cell['n'])&(a['ell']==(cell['ell'] or -1))&(a['variant']==(cell['variant'] or ''))
                    yy=y[ids];zz=z[ids];equal(cell,binary_metrics(yy,zz))
                    row={'family':c.family,'regime':regime,'seed':c.seed,'r':r,'radius':4*r,'split':split,'n':cell['n'],
                        'ell':cell['ell'],'variant':cell['variant'],**binary_metrics(yy,zz)}
                    for kind,t in thresholds.items():
                        value=calibrated(yy,zz,t);equal(cell[kind+'_validation_threshold_metrics'],value)
                        row[kind+'_calibrated_BA']=value['balanced_accuracy'];row[kind+'_threshold']=t
                    if split=='stress':
                        assert np.array_equal(yy.reshape(-1,2),np.tile([1.,0.],(len(yy)//2,1)))
                        assert np.array_equal(a['graph_ids'][ids][::2],a['graph_ids'][ids][1::2])
                        d=zz.astype(np.float64).reshape(-1,2);delta=d[:,0]-d[:,1]
                        both=float(((zz>=0)==(yy==1)).reshape(-1,2).all(1).mean());assert both==cell['both_queries_correct']
                        expected={'median':float(np.median(delta)),'p10':float(np.quantile(delta,.1)),'p90':float(np.quantile(delta,.9)),
                            'min':float(delta.min()),'max':float(delta.max()),'fraction_positive':float((delta>0).mean()),'fraction_gt_1e_6':float((delta>1e-6).mean())}
                        equal(cell['paired_logit_margin'],expected);row['both_queries_correct']=both
                        row.update({'margin_'+k:v for k,v in expected.items()})
                    rows.append(row)
    assert count==640 and entries==2944000 and cal_entries==7055360
    groups={};keys=['family','regime','r','split','n','ell','variant'];aggregates=[]
    for row in rows:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    for key,values in groups.items():
        assert sorted(x['seed'] for x in values)==SEEDS
        row=dict(zip(keys,key));row['seeds']=5
        for field in ['balanced_accuracy','original_calibrated_BA','augmented_calibrated_BA','recall','specificity','auroc','bce','both_queries_correct','margin_median','margin_fraction_gt_1e_6']:
            if all(v.get(field) is not None for v in values):
                a=np.array([v[field] for v in values]);row[field+'_mean']=float(a.mean());row[field+'_sd']=float(a.std(ddof=1))
        aggregates.append(row)
    lookup={tuple(row[k] for k in keys+['seed']):row for row in rows};resamples=np.array(list(itertools.product(range(5),repeat=5)));contrasts=[]
    cells={(x['family'],x['r'],x['split'],x['n'],x['ell'],x['variant']) for x in rows}
    for family,r,split,n,ell,variant in sorted(cells,key=str):
        for a,b in [('class_balanced','original'),('length_balanced','short_aug')]:
            for field in ['balanced_accuracy','original_calibrated_BA','augmented_calibrated_BA']:
                delta=np.array([lookup[(family,a,r,split,n,ell,variant,s)][field]-lookup[(family,b,r,split,n,ell,variant,s)][field] for s in SEEDS])
                boot=delta[resamples].mean(1)
                contrasts.append({'family':family,'r':r,'split':split,'n':n,'ell':ell,'variant':variant,'regime_a':a,'regime_b':b,
                    'metric':field,'mean_delta_pp':100*float(delta.mean()),'sd_pp':100*float(delta.std(ddof=1)),
                    'paired_seed_lower95_pp':100*float(np.quantile(boot,.025)),'paired_seed_upper95_pp':100*float(np.quantile(boot,.975)),
                    'seeds':5,'resamples':3125})
    for name,values in [('per_seed.csv',rows),('aggregate.csv',aggregates),('paired_comparisons.csv',contrasts),('training.csv',training)]:csv_write(name,values)
    audit={'status':'passed','snapshot_utc':datetime.now(timezone.utc).isoformat(),'checkpoints':40,'workloads':640,
        'test_predictions_checked':entries,'validation_predictions_checked':cal_entries,'evaluation_source_hash':evaluation_hash(),
        'training_source_hash':source_hash(),'test_data_hash':test_hash,'scheduler_job':os.environ['SLURM_JOB_ID'],'new_inference':False}
    write_json(ROOT/'report/audit.json',audit)
    text=['# Влияние баланса классов и длин путей','\nСнимок UTC: '+audit['snapshot_utc']+'. Аудит сохранённых предсказаний пройден.',
        '\n30 новых обучений и10 исходных контролей; F4-bin, ER/DAG, пять seeds, checkpoint_last на30000 шагах. '
        'A=original; B=class_balanced; C=short_aug; D=length_balanced. '
        'Основные контрасты B−A и D−C. В C/D одинаковый дополнительный пул n48 и его доля50%; различаются вероятности длин. '
        'B→C меняет размер, структуру и отрицательные примеры одновременно. Все режимы используют фиксированное распределение с первого шага.',
        '\nTrain-длины дополнения1–16. Test-длины>16 проверяют перенос по длине;8/12/16 — известные длины на большем n. '
        'Циклическое дополнение к ER является отдельным управляемым генератором, не случайным ER. '
        'Основные метрики используют logit0. Два дополнительных порога на каждый checkpoint/r выбраны по исходной или расширенной validation, без test.',
        '\n## Random far n256: BA, % (среднее ± SD пяти seeds)',
        '\n| Семейство | Режим | r4 | r8 | r16 | r32 |\n|---|---|---:|---:|---:|---:|']
    def fmt(v):return f"{100*v['balanced_accuracy_mean']:.2f} ± {100*v['balanced_accuracy_sd']:.2f}"
    for family in FAMILIES:
        for regime in ['original']+REGIMES:
            rs=[next(x for x in aggregates if x['family']==family and x['regime']==regime and x['r']==r and x['split']=='far' and x['n']==256) for r in TEST['hops']]
            text.append('| '+family+' | '+regime+' | '+' | '.join(fmt(x) for x in rs)+' |')
    for family in FAMILIES:
        for variant in TEST['stress_variants']:
            text += ['\n## '+family+', '+variant+', n256, r32',
                '\n| Режим | ℓ8 | ℓ12 | ℓ16 | ℓ20 | ℓ24 | ℓ32 | ℓ48 | ℓ64 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for regime in ['original']+REGIMES:
                rs=[next(x for x in aggregates if x['family']==family and x['regime']==regime and x['r']==32 and x['variant']==variant and x['ell']==ell) for ell in TEST['stress_lengths']]
                text.append('| '+regime+' | '+' | '.join(fmt(x) for x in rs)+' |')
    text+=['\n## Полные данные и ограничения',
        '\nCSV сохраняют все r, размеры, длины, seeds, матрицы ошибок, калибровки, AUROC и зазоры. '
        'Парные интервалы основаны на3125 перевыборках пяти seeds, исследовательские, без поправки на множественные сравнения. '
        'Они условны на фиксированных тестовых графах. Перестановки plain-шаблонов не дают независимые видимые топологии. '
        'Данные имеют разные размеченные матрицы смежности; изоморфная независимость не утверждается.',
        '\nНовые test-предсказания выполнены только после фиксации всех40 checkpoints. При изменении r обучаемые параметры неизменны. '
        'Порог по расширенной validation использует длины до16, а не длинные test-метки. '
        'Равное число шагов не означает равенства стоимости обучения при изменении n. Фактические времена обучения находятся в training.csv; '
        'общие измерения одинаковой архитектуры — в timing/r*.json. Slurm учитывает GPU-секунды отдельно в queue/state.json.']
    (ROOT/'report/REPORT_CURRICULUM_RU.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate','report']);p.add_argument('--seconds',type=int);p.add_argument('--remaining',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='report':report()
    else:
        result=evaluate(a.seconds,a.remaining);write_json(ROOT/('slot_'+os.environ['SLURM_JOB_ID']+'.json'),{'status':result});print(json.dumps({'status':result}),flush=True)
