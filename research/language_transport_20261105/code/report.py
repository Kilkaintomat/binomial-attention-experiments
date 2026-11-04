"""Rebuild dated tables/plots from raw records, never fill in missing runs."""
import csv
import math
import statistics
from collections import defaultdict

from common import SIDE,ART,rows,read,atomic


def csv_write(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not records:return
    fields=sorted(set().union(*(x.keys() for x in records)))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)


def oracle_plot():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from attention import action
    out=SIDE/'report';out.mkdir(exist_ok=True)
    data=[]
    for ell in [1,2,4,8,16,32]:
        n=ell+1;a=torch.zeros(n,n,dtype=torch.float64);a[0,0]=1
        for i in range(1,n):a[i,i-1]=1
        v=torch.zeros(n,1,dtype=torch.float64);v[0]=1
        for r in [1,2,4,8,16,32,64,80,96]:
            theory=sum(math.comb(r,s) for s in range(ell,r+1))/(2**r-1)
            measured=float(action(a,v,'binomial',r)[-1,0])
            data.append(dict(ell=ell,r=r,theoretical_tail=theory,measured_mass=measured,zero=theory==0))
    csv_write(out/'oracle_mass.csv',data)
    fig,ax=plt.subplots(figsize=(6,5));zeros=sum(x['zero'] for x in data)
    for ell in [1,2,4,8,16,32]:
        g=[x for x in data if x['ell']==ell and not x['zero']]
        ax.scatter([x['theoretical_tail'] for x in g],[x['measured_mass'] for x in g],label='ell='+str(ell),s=25)
    ax.plot([1e-10,1],[1e-10,1],'k--',label='y=x');ax.set(xscale='log',yscale='log',
        xlabel='Theoretical conditional binomial tail',ylabel='Measured absorbing-chain mass',
        title='Implementation check; zero pairs excluded: '+str(zeros))
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'oracle_mass.png',dpi=180);plt.close(fig)
    atomic(out/'oracle_checks.json',{'max_abs_error':max(abs(x['theoretical_tail']-x['measured_mass']) for x in data),
        'zero_pairs':zeros,'n':len(data),'scope':'analytic oracle only, no trained language model'})


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=SIDE/'report';out.mkdir(exist_ok=True);allrows=[];train=[];status=[];val=[]
    for p in sorted((SIDE/'runs').rglob('manifest.json')):
        spec=read(p);run=p.parent;result=read(run/'result.json') if (run/'result.json').exists() else {}
        training=read(run/'training.json') if (run/'training.json').exists() else {}
        info={k:spec.get(k) for k in ['id','phase','model','task','method','layers','seed','r','loops']}
        status.append(dict(info,status=result.get('status',training.get('status','not_started')),
                           evaluation_status=result.get('status','not_started'),step=training.get('step',0)))
        train.append(dict(info,**{k:v for k,v in training.items() if not isinstance(v,(dict,list))}))
        if (run/'validation.jsonl').exists():
            for x in rows(run/'validation.jsonl'):val.append(dict(info,**{k:v for k,v in x.items() if k!='by_hops'}))
        if result.get('status')!='completed':continue
        for x in rows(run/'predictions.jsonl'):
            allrows.append(dict(info,**x))
    csv_write(out/'run_status.csv',status);csv_write(out/'training.csv',train);csv_write(out/'validation_curves.csv',val)
    keys=['phase','model','task','method','layers','seed','slice','r_eval','loops_eval','hops','context_words','sequence_length','input_tokens','chains','distribution']
    groups=defaultdict(list)
    # Conditional unary p changes length; both raw token length and context factor retained.
    for x in allrows:groups[tuple(x.get(k) for k in keys)].append(x)
    metrics=[]
    for key,g in sorted(groups.items(),key=lambda x:str(x[0])):
        info=dict(zip(keys,key))
        for subset in ['all','nontrivial','missing','self_loop']:
            gg=g if subset=='all' else [x for x in g if x['pointer'].get(subset,False)]
            metrics.append(dict(info,subset=subset,n=len(gg),accuracy=sum(x['correct'] for x in gg)/len(gg) if gg else None,
                mean_prefill_seconds=statistics.mean(x['prefill_seconds'] for x in gg) if gg else None,
                mean_answer_seconds=statistics.mean(x['answer_seconds'] for x in gg) if gg else None,
                mean_answer_sequence_probability=statistics.mean(x['answer_sequence_probability_without_eos'] for x in gg) if gg else None))
    index={tuple(x[k] for k in keys)+ (x['subset'],):x for x in metrics}
    for x in metrics:
        base=dict(x,method='baseline',layers=2,r_eval=1,loops_eval=1)
        ref=index.get(tuple(base[k] for k in keys)+(x['subset'],)) if x['slice']=='main' else None
        x['delta_baseline_pp']=100*(x['accuracy']-ref['accuracy']) if ref and ref['accuracy'] is not None and x['accuracy'] is not None else None
    csv_write(out/'metrics_by_seed.csv',metrics)
    matrix=defaultdict(dict)
    rowkeys=['phase','model','task','seed','hops','context_words','sequence_length','input_tokens','chains','distribution','subset']
    for x in metrics:
        if x['slice']!='main':continue
        key=tuple(x[k] for k in rowkeys);label=x['method']+('_4layers' if x['layers']==4 else '')
        score='n=0' if x['accuracy'] is None else ('%.2f / %+.2f'%(100*x['accuracy'],x['delta_baseline_pp'])) if x['delta_baseline_pp'] is not None else ('%.2f / n.a.'%(100*x['accuracy']))
        matrix[key][label]=score
    csv_write(out/'attention_columns.csv',[dict(zip(rowkeys,key),**values) for key,values in matrix.items()])
    agg=defaultdict(list)
    for x in metrics:
        if x['accuracy'] is None:continue
        key=tuple((k,v) for k,v in x.items() if k not in ['seed','n','accuracy','mean_prefill_seconds','mean_answer_seconds','mean_answer_sequence_probability','delta_baseline_pp'])
        agg[key].append(x)
    summaries=[dict(key,n_seeds=len(g),mean_accuracy=statistics.mean(x['accuracy'] for x in g),
                    sample_sd=statistics.stdev(x['accuracy'] for x in g) if len(g)>1 else None,
                    min_seed_accuracy=min(x['accuracy'] for x in g),max_seed_accuracy=max(x['accuracy'] for x in g)) for key,g in agg.items()]
    csv_write(out/'seed_aggregates.csv',summaries)
    # Each line is one seed. No hand-picked best heads or seeds.
    for xname,slice_name,filename in [('hops','main','accuracy_hops.png'),('r_eval','r_sweep','accuracy_r.png'),
                                     ('context_words','main','accuracy_context.png'),('chains','main','accuracy_distractors.png')]:
        fig,ax=plt.subplots(figsize=(10,6));curves=defaultdict(lambda:defaultdict(list))
        for x in allrows:
            if x['slice']!=slice_name or (xname in ['chains','context_words'] and x['hops'] not in [4,16]):continue
            if xname=='chains' and x['task']!='vt':continue
            if xname=='chains' and x['context_words']!=1024:continue
            if xname=='context_words' and x['task']=='vt' and x['chains']!=4:continue
            if xname=='hops' and ((x['task']=='vt' and (x['context_words']!=512 or x['chains']!=4)) or
                                 (x['task']!='vt' and x['sequence_length']!=256)):continue
            label='%s/%s/%s/L%s/s%s/%s'%(x['model'],x['task'],x['method'],x['layers'],x['seed'],x['distribution'])
            if xname!='hops':label+='/h'+str(x['hops'])
            xx=x['sequence_length'] if xname=='context_words' and x['task']!='vt' else x[xname]
            curves[label][xx].append(x['correct'])
        for label,points in curves.items():
            xx=sorted(points);ax.plot(xx,[sum(points[z])/len(points[z]) for z in xx],label=label,alpha=.7)
        ax.set(xlabel=xname,ylabel='Exact match',ylim=(0,1));ax.legend(fontsize=4,ncol=3,loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout();fig.savefig(out/filename,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6));curves=defaultdict(list)
    for x in val:
        if str(x['phase']).startswith('pilot'):continue
        curves[(x['model'],x['task'],x['method'],x['layers'],x['seed'])].append(x)
    for key,g in curves.items():
        g=sorted(g,key=lambda x:x['step']);ax.plot([x['training_seconds'] for x in g],[x['accuracy'] for x in g],label=str(key),alpha=.6)
    ax.set(xlabel='Measured training seconds (same allocated GPU type)',ylabel='Short-depth validation accuracy',ylim=(0,1))
    ax.legend(fontsize=4,ncol=2,loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(out/'quality_time.png',dpi=180);plt.close(fig)
    masses=[]
    for x in allrows:
        if x['slice']!='r_sweep' or x['task']!='vt':continue
        # Annotation adds query -> terminal record and root record -> value.
        ell=x['hops']+2;r=x['r_eval'];theory=sum(math.comb(r,s) for s in range(ell,r+1))/(2**r-1)
        masses.append(dict(model=x['model'],method=x['method'],seed=x['seed'],hops=x['hops'],r=r,ell_annotation=ell,
            theoretical_reference=theory,measured_mass=statistics.mean(d['operator_mass_all_heads_mean'] for d in x['diagnostics']),
            answer_probability=x['answer_sequence_probability_without_eos'],correct=x['correct']))
    csv_write(out/'learned_mass.csv',masses)
    if masses:
        fig,ax=plt.subplots(figsize=(6,5));floor=1e-12
        ax.scatter([max(floor,x['theoretical_reference']) for x in masses],[max(floor,x['measured_mass']) for x in masses],s=8,alpha=.25)
        ax.plot([floor,1],[floor,1],'k--');ax.set(xscale='log',yscale='log',xlabel='Ideal tail at annotated ell (reference)',
            ylabel='Learned operator mass, all heads/layers mean',title='No equality assumed; zeros displayed at 1e-12')
        fig.tight_layout();fig.savefig(out/'learned_mass_tail.png',dpi=180);plt.close(fig)
    oracle_plot()
    done=sum(x['evaluation_status']=='completed' for x in status)
    (out/'REPORT_RU.md').write_text('# Языковые эксперименты: измеренный срез\n\n'
        +'Завершённых записей с оценкой: '+str(done)+'. Незавершённые запуски перечислены в run_status.csv и не включены в итоговые accuracy.\n\n'
        +'metrics_by_seed.csv содержит все seed и отдельные срезы самопетель, отсутствующих ссылок и нетривиальных маршрутов. '
        +'seed_aggregates.csv показывает среднее и выборочное стандартное отклонение между seed, отдельно от размера тестовой выборки. '
        +'При одном seed стандартное отклонение не задано. Отрицательные результаты сохраняются.\n\n'
        +'quality_time.png сравнивает качество на короткой validation по измеренному времени обучения; это не подбор по глубокому test. '
        +'Биномиальный хвост относится к поглощающей oracle-цепочке. Для обученной модели показана отдельно измеренная масса, а не предсказанная формулой accuracy. '
        +'Агрегирование голов и выбранных слоёв фиксировано: арифметическое среднее.\n\n'
        +'Численные архитектурные сравнения следует читать по одинаковым model/task/seed/context/distribution; canonical p=16/32 и условная адаптация не объединяются. '
        +'Срез без всех трёх seed не является завершённым основным сравнением.\n',encoding='utf-8')


if __name__=='__main__':main()
