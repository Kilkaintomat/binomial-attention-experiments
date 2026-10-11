"""Dated, completeness-aware summaries; no outcome is inferred from another task."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .config import allocation,read,root,source_hash,PRIORITY,FAMILIES,SEEDS


def table(path,rows,minimum):
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(minimum+[k for r in rows for k in r]))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)


def report():
    allocation();target=root()/'report';target.mkdir(exist_ok=True);final=[];timing=[];distance=[];extra=[];pilots=[];curves=[]
    for p in sorted((root()/'runs').glob('**/result.json')):
        r=read(p);c=r['config'];stage=c['stage']
        if stage not in ['pilot','confirmation','final']:continue
        base={'family':c['family'],'model':c['model'],'seed':c['seed'],'stage':stage,'status':r['status'],'steps':r['step'],
              'parameters':r['operations']['parameters'],'source_hash':r['source_hash'],'data_hash':r['data_hash'],
              'config_hash':p.parent.name,'train_n':'16,17','train_peak_memory_bytes':r.get('train_peak_memory_bytes'),**r['times']}
        if stage!='final':pilots.append({**base,'lr':c['lr'],**(r['final_validation'] or {})})
        v=p.parent/'validation.jsonl'
        if v.exists():
            for line in v.read_text().splitlines():curves.append({**base,**json.loads(line)})
        if stage!='final':continue
        evaluations=sorted((p.parent/'evaluation').glob('*.json'))
        if not evaluations:final.append({**base,'evaluation_status':'not_evaluated'})
        for e in evaluations:
            value=read(e);meta={**base,**{k:value.get(k) for k in ['regime','split','n','setting','value','subset','checkpoint_step','device_name']}}
            rows=value['by_size_or_stress_cell'];numeric={k:value.get(k) for k in ['balanced_accuracy','accuracy','f1','auroc','average_precision','bce','count','positive_fraction','max_observed_state_norm']}
            if value['regime']=='fixed':
                for row in rows:final.append({**meta,**row,'evaluation_status':'completed'})
                vals=[row['balanced_accuracy'] for row in rows if row['balanced_accuracy'] is not None]
                final.append({**meta,**numeric,'n':'macro','balanced_accuracy':float(np.mean(vals)) if vals else None,'aggregation':'unweighted mean of size/stress-cell BA','evaluation_status':'completed'})
            else:
                for row in rows:extra.append({**meta,**row,'max_observed_state_norm':value.get('max_observed_state_norm')})
            for row in value.get('timing',[]):timing.append({**meta,**row})
            for row in value.get('path_length_slices',[]):distance.append({**meta,**row})
    grouped=defaultdict(list)
    for row in final:
        if row.get('evaluation_status')=='completed':grouped[tuple(row.get(k) for k in ['family','model','split','n','ell','variant'])].append(row)
    aggregate=[]
    for key,rows in grouped.items():
        row=dict(zip(['family','model','split','n','ell','variant'],key));row['seed_count']=len(rows)
        for metric in ['balanced_accuracy','f1','auroc','bce','training_seconds','elapsed_seconds']:
            values=[r[metric] for r in rows if r.get(metric) is not None]
            row[metric+'_mean']=float(np.mean(values)) if values else None;row[metric+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        aggregate.append(row)
    contrasts=[];rng=np.random.default_rng(20261011)
    pairs=[('F4-bin','F4-one'),('F4-exp','F4-bin'),('F4-geo','F4-bin'),('F4-bin-inner','F4-bin'),('F4-exp-inner','F4-exp'),('F4-bin-inner','F4-one-inner')]
    cells={tuple(r.get(k) for k in ['family','split','n','ell','variant']) for r in final if r.get('evaluation_status')=='completed'}
    for cell in cells:
        for left,right in pairs:
            by_model={m:{r['seed']:r['balanced_accuracy'] for r in final if r.get('evaluation_status')=='completed' and r['model']==m and tuple(r.get(k) for k in ['family','split','n','ell','variant'])==cell and r.get('balanced_accuracy') is not None} for m in [left,right]}
            seeds=sorted(by_model[left].keys()&by_model[right].keys());diff=np.array([100*(by_model[left][s]-by_model[right][s]) for s in seeds])
            row={**dict(zip(['family','split','n','ell','variant'],cell)),'comparison':left+' minus '+right,'paired_seeds':len(seeds),'seeds':str(seeds),'mean_difference_pp':float(diff.mean()) if len(diff) else None,'lower95_pp':None,'upper95_pp':None,'interpretation':'exploratory paired-seed bootstrap; no multiplicity adjustment'}
            if len(diff)>=2:
                draws=rng.choice(diff,size=(20000,len(diff)),replace=True).mean(1);row['lower95_pp'],row['upper95_pp']=map(float,np.quantile(draws,[.025,.975]))
            contrasts.append(row)
    outputs=[('pilot_selection.csv',pilots,['family','model','seed','lr','status']),('final_per_seed.csv',final,['family','model','seed','split','n','status']),
             ('aggregate_results.csv',aggregate,['family','model','split','n','seed_count']),('timing.csv',timing,['family','model','n','batch_size']),
             ('path_length_slices.csv',distance,['family','model','slice','count']),('extra_compute.csv',extra,['family','model','n','setting','value']),
             ('paired_differences.csv',contrasts,['family','comparison','paired_seeds']),('validation_curves.csv',curves,['family','model','seed','step'])]
    for name,rows,minimum in outputs:table(target/name,rows,minimum)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for family in FAMILIES:
        fig,ax=plt.subplots(figsize=(8,5))
        for model in PRIORITY:
            rows=[r for r in aggregate if r['family']==family and r['model']==model and isinstance(r['n'],int) and r['split'] in ['test_ID','test_near','test_far']]
            rows=sorted(rows,key=lambda r:r['n'])
            if rows:ax.plot([r['n'] for r in rows],[r['balanced_accuracy_mean'] for r in rows],marker='o',label=model)
        ax.set(xlabel='Graph vertices',ylabel='Test balanced accuracy',title=family+'; fixed compute')
        if ax.lines:ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(target/(family+'_ba_vs_n.png'),dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5));bins=['1','2','3-4','5-8','9-16','17-32','>32']
        for model in PRIORITY:
            values=[]
            for name in bins:
                vals=[r['recall'] for r in distance if r['family']==family and r['model']==model and r['regime']=='fixed' and r['split']=='test_far' and r['n']==128 and r['slice']=='positive_distance_'+name and r.get('recall') is not None]
                values.append(float(np.mean(vals)) if vals else np.nan)
            if np.isfinite(values).any():ax.plot(bins,values,marker='o',label=model)
        ax.set(xlabel='Positive directed shortest distance',ylabel='Recall',title=family+'; far n=128')
        if ax.lines:ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(target/(family+'_recall_vs_distance.png'),dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5))
        for model in PRIORITY:
            rows=[r for r in final if r['family']==family and r['model']==model and r.get('split')=='test_far' and r.get('n')==128 and r.get('balanced_accuracy') is not None]
            if rows:ax.scatter([r['training_seconds'] for r in rows],[r['balanced_accuracy'] for r in rows],label=model)
        ax.set(xlabel='Training seconds, completed 30000 updates',ylabel='Far n=128 balanced accuracy',title=family+'; quality and training cost')
        if ax.collections:ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(target/(family+'_quality_vs_time.png'),dpi=160);plt.close(fig)
        for model in PRIORITY:
            for variant in ['dag','cyclic']:
                matrix=np.full((4,6),np.nan)
                for i,n in enumerate([16,32,64,128]):
                    for j,ell in enumerate([2,4,8,16,32,48]):
                        vals=[r['balanced_accuracy'] for r in final if r['family']==family and r['model']==model and r.get('split')=='stress' and r.get('n')==n and r.get('ell')==ell and r.get('variant')==variant]
                        if vals:matrix[i,j]=np.mean(vals)
                if not np.isfinite(matrix).any():continue
                fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(matrix,vmin=0,vmax=1,aspect='auto')
                ax.set(xticks=range(6),xticklabels=[2,4,8,16,32,48],yticks=range(4),yticklabels=[16,32,64,128],xlabel='Directed distance',ylabel='Vertices',title=family+' '+model+' '+variant)
                fig.colorbar(im,ax=ax,label='Paired positive/negative accuracy');fig.tight_layout();fig.savefig(target/(family+'_'+model+'_'+variant+'_stress.png'),dpi=160);plt.close(fig)
    admission=read(root()/'admission.json') if (root()/'admission.json').exists() else None
    queue=read(root()/'queue/state.json') if (root()/'queue/state.json').exists() else {'status':'preparing','gpu_seconds_used':0}
    trained={(r['family'],r['model'],r['seed']) for r in final if r['status']=='completed'}
    lines=['# Directed reachability — фактический статус','',
           'Это адаптация ER/DAG к query-conditioned бинарной достижимости с сохранением всех вершин. Она не воспроизводит опубликованные scores исходной статьи.',
           '',f'Состояние очереди: {queue["status"]}. Завершено новых финальных обучений: {len(trained)}. Полная исходная сетка — 90 финалов и 54 пилота.',
           f'Учтённое GPU-время завершённых выделений: {queue["gpu_seconds_used"]/3600:.4f} ч из 24; активное выделение добавляется при accounting.',
           '', '## Дизайн','',
           'Два независимых семейства er_exact_n_v1 и dag_exact_n_v1. Train n=16–17, validation n=18–19, ID n=16–17, near n=20–27, far n=32/64/128. По восемь равномерных разных запросов на граф.',
           'Пять признаков вершины; маска I∨Bᵀ переносит маркер источника по направлению рёбер; readout только в target. Label, расстояния и component metadata не передаются в forward.',
           'AdamW, batch64, 30000 обновлений, одинаковая последовательность batches и общие начальные тензоры при одинаковом seed. Pilots3000 сохраняют scheduler horizon30000.',
           '', '## Покрытие и результаты','',
           'Выбранная по измеренной стоимости подматрица: '+(json.dumps(admission,ensure_ascii=False) if admission else 'ещё не определена; test-модели не оценены.'),
           '', '| family | model | split | n | seeds | BA mean | SD |','|---|---|---|---|---:|---:|---:|']
    for r in aggregate:
        if r['n']=='macro':lines.append('| '+' | '.join(str(r.get(k)) for k in ['family','model','split','n','seed_count','balanced_accuracy_mean','balanced_accuracy_sample_sd'])+' |')
    lines += ['', '## Проверки и ограничения','',
              'Фактические проверки: DATA_AUDIT_REACHABILITY.json, PREFLIGHT_REACHABILITY.json и JUnit. Изоморфизм проверяется отдельно от точного равенства adjacency; неполный аудит не доказывает отсутствие изоморфных дублей.',
              'Основные CSV содержат все фактически записанные seeds. Отсутствующие и частичные запуски не заменяются ожидаемыми результатами.',
              'Extra compute использует фиксированную подвыборку; lambda/rho сохраняют значения обучения. Learned r>4 — unsupported. Таблицы fixed/extra разделены.',
              'Mean±SD относится к seeds. Парные интервалы exploratory. Все запросы одного графа являются зависимыми; восемь пар не считаются восемью независимыми графами. Graph-bootstrap интервалы в основных таблицах пока не рассчитываются.',
              'Внутренний MLP меняет параметры и объём вычислений. Recurrent имеет другую архитектуру. Границы локальности и degree shortcuts ограничивают причинные выводы.',
              'До завершения оценок нет ответа о переносе на большие n, длинные пути и эффекте inner MLP; результаты Maze/Sudoku не подставляются.',
              '', '## Воспроизведение','', 'См. docs/REACHABILITY_RU.md; большие графы, checkpoints, logits, зависимости upstream и приватные настройки исключены из git. Внешняя публикация этой серии не выполнялась.']
    Path('REPORT_REACHABILITY_RU.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'source_hash':source_hash(),'completed_new_final_trainings':len(trained),'evaluated_cells':len(aggregate),'queue_status':queue['status']}
