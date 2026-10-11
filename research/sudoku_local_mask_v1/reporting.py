"""Per-mask tables and the prespecified paired difference of differences."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from transport_reasoning.config import write_json
from .config import LABELS, FINAL_SEEDS, code_hash, root, grid
from .campaign import read,legacy_folder,collect_overfit


def table(path,rows,default_fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or default_fields
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def contrast(rows):
    lookup={(r['support'],r['model'],r['seed']):r['solved_rate'] for r in rows
            if r['stage']=='final' and r['status']=='completed' and r.get('solved_rate') is not None}
    paired=[]; generator=np.random.default_rng(90421)
    definitions={'G_grid': [('grid4','F4-binomial',1),('grid4','F4-onehop',-1)],
                 'G_rel':[('relational','F4-binomial',1),('relational','F4-onehop',-1)],
                 'I':[('grid4','F4-binomial',1),('grid4','F4-onehop',-1),('relational','F4-binomial',-1),('relational','F4-onehop',1)]}
    for model in LABELS: definitions[model+'_grid_minus_rel']=[('grid4',model,1),('relational',model,-1)]
    for support in ['relational','grid4']:
        for model in ['F4-onehop','F4-binomial']:
            definitions[support+'_'+model+'_minus_R32']=[(support,model,1),(support,'R32-onehop',-1)]
    summaries=[]
    for name,terms in definitions.items():
        values=[];seeds=[]
        for seed in FINAL_SEEDS:
            if not all((support,model,seed) in lookup for support,model,weight in terms): continue
            value=100*sum(weight*lookup[support,model,seed] for support,model,weight in terms)
            values.append(value);seeds.append(seed);paired.append({'contrast':name,'seed':seed,'difference_pp':value})
        row={'contrast':name,'paired_seeds':len(seeds),'complete_five_seeds':len(seeds)==5,
             'mean_pp':float(np.mean(values)) if values else None,
             'sample_sd_pp':float(np.std(values,ddof=1)) if len(values)>1 else None,'lower95_pp':None,'upper95_pp':None}
        if len(values)>1:
            boot=generator.choice(values,size=(20000,len(values)),replace=True).mean(1)
            row['lower95_pp'],row['upper95_pp']=[float(x) for x in np.quantile(boot,[.025,.975])]
        summaries.append(row)
    return paired,summaries


def historical_threshold(curve,threshold):
    pending=None;last_fail=None
    for row in curve:
        if row['solved_rate']>=threshold:
            if pending is not None:
                return {'status':'reached','first_passing_step':pending['step'],'confirmed_step':row['step'],
                        'training_seconds_upper':pending.get('training_seconds'),'last_failing_step':last_fail['step'] if last_fail else None,
                        'training_seconds_lower':last_fail.get('training_seconds') if last_fail else None}
            pending=row
        else: last_fail=row;pending=None
    return {'status':'pending' if pending else 'not_reached','last_failing_step':last_fail['step'] if last_fail else None}


def report():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Analysis requires a CPU or GPU scheduler allocation')
    target=root()/'report';target.mkdir(parents=True,exist_ok=True);collect_overfit()
    audit=read(root()/'checks/legacy_compatibility.json')
    data_audit=read(root()/'checks/data.json')
    reuse=audit['reuse_all_15_relational_allowed']
    history={(r['model'],r['seed']):r for r in audit['historical_final_results']} if reuse else {}
    manifest=read(root()/'manifest.json');statuses={r['config_hash']:r for r in manifest['rows']}
    rows=[];curves=[];thresholds=[];positions=[]
    for c in grid():
        historical=c.stage=='final' and c.support=='graph' and (c.label(),c.seed) in history
        r=None;metric=None;status=statuses[c.hash()]['status'];origin='historical_quality_only' if historical else 'new_training'
        curve=[]
        if historical:
            status='completed';old=history[c.label(),c.seed];metric=old['evaluation'];curve_path=legacy_folder(c)/'metrics.jsonl'
            source=old['legacy_source_sha256']
        else:
            source=code_hash();curve_path=c.run_dir()/'metrics.jsonl'
            if (c.run_dir()/'result.json').exists():
                r=read(c.run_dir()/'result.json');status=r['status']
                if c.stage=='pilot': metric=r['last_validation']
            elif (c.run_dir()/'checkpoint_last.pt').exists(): status='running'
            if (c.run_dir()/'evaluation_test.json').exists(): metric=read(c.run_dir()/'evaluation_test.json')
        if curve_path.exists(): curve=[json.loads(line) for line in curve_path.read_text().splitlines()]
        row={'stage':c.stage,'support':c.mask_label(),'model':c.label(),'seed':c.seed,'status':status,'origin':origin,
             'source_sha256':source,'config_hash':c.hash(),'completed_steps':30000 if historical else r['completed_steps'] if r else curve[-1]['step'] if curve else 0,
             'planned_steps':c.planned_steps,'completion_steps':c.completion_steps,'parameters':204169 if c.architecture=='recurrent' else 797449,
             'data_hash':data_audit['data_hash'],'mask_sha256':data_audit['masks'][c.mask_label()]['mask_sha256'],
             'r':c.hops,'T':c.recurrent_steps,'layers':c.layers,**c.operations()}
        for key in ['solved_rate','exact_match','blank_accuracy','loss']: row[key]=metric.get(key) if metric else None
        row['checkpoint_step']=metric.get('checkpoint_step') if metric else None
        if r:
            row.update(mask_sha256=r['mask_sha256'],data_hash=r['data_hash'],training_seconds=r['timings']['training'],
                       validation_seconds=r['timings']['validation'],checkpoint_seconds=r['timings']['checkpoints'],elapsed_seconds=r['timings']['elapsed'],
                       training_examples_per_second=r['examples']/max(1e-12,r['timings']['training']),
                       peak_memory_bytes=(r['last_validation'] or {}).get('train_peak_memory_bytes'))
            row.update(training_gpu_hours=r['timings']['training']/3600,
                       active_training_call_gpu_hours=r['timings']['elapsed']/3600,
                       gpu_hours_basis='active run; shared allocation overhead and final inference charged separately by scheduler')
        for threshold in [.5,.9]:
            crossing=r['thresholds'][str(threshold)] if r else historical_threshold(curve,threshold) if curve else {'status':'not_started'}
            thresholds.append({'stage':c.stage,'support':c.mask_label(),'model':c.label(),'seed':c.seed,'threshold':threshold,'origin':origin,**crossing})
        mpath=c.run_dir()/'measurement.json'
        if mpath.exists():
            measurement=read(mpath)
            for measurement_row in measurement['inference']:
                b=measurement_row['batch_size']
                for kind,prefix in [('forward_batch_seconds','forward'),('end_to_end_batch_seconds','full')]:
                    for stat in ['median','p25','p75']: row[f'{prefix}_b{b}_{stat}_seconds']=measurement_row[kind][stat]
                row[f'forward_b{b}_examples_per_second']=measurement_row['forward_examples_per_second']
            for group in measurement['position_accuracy']['groups']:
                positions.append({'support':c.mask_label(),'model':c.label(),'seed':c.seed,**group})
        for point in curve:
            curves.append({'stage':c.stage,'support':c.mask_label(),'model':c.label(),'seed':c.seed,'origin':origin,
                           'step':point['step'],'solved_rate':point['solved_rate'],'loss':point['loss'],'blank_accuracy':point.get('blank_accuracy'),
                           'training_seconds':point.get('training_seconds'),'elapsed_seconds':point.get('elapsed_seconds')})
        rows.append(row)
    summary=[]
    for stage in ['pilot','final']:
        for support in ['relational','grid4']:
            for model in LABELS:
                group=[r for r in rows if (r['stage'],r['support'],r['model'])==(stage,support,model) and r['status']=='completed' and r['solved_rate'] is not None]
                item={'stage':stage,'support':support,'model':model,'completed_evaluated_seeds':len(group),
                      'origin':'historical_quality_only' if group and all(r['origin']=='historical_quality_only' for r in group) else 'new_training'}
                for key in ['solved_rate','blank_accuracy','loss','parameters','training_seconds','elapsed_seconds','forward_b1_median_seconds','forward_b64_median_seconds','peak_memory_bytes']:
                    vals=[r[key] for r in group if r.get(key) is not None]
                    item[key+'_mean']=float(np.mean(vals)) if vals else None
                    item[key+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                summary.append(item)
    paired,contrasts=contrast(rows)
    for stage in ['pilot','final']: table(target/(stage+'_seeds.csv'),[r for r in rows if r['stage']==stage],['support','model','seed','status'])
    for name,data in [('summary',summary),('contrasts_per_seed',paired),('contrasts_summary',contrasts),('thresholds',thresholds),('position_accuracy',positions),('validation_curves',curves)]:
        table(target/(name+'.csv'),data,['support','model','seed','status'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage in ['pilot','final']:
        for xkey in ['step','training_seconds']:
            for metric in ['solved_rate','loss','blank_accuracy']:
                fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
                for ax,support in zip(axes,['relational','grid4']):
                    for model,color in zip(LABELS,['C0','C1','C2']):
                        subset=[r for r in curves if (r['stage'],r['support'],r['model'])==(stage,support,model) and r[xkey] is not None and r[metric] is not None]
                        for n,seed in enumerate(sorted({r['seed'] for r in subset})):
                            vals=[r for r in subset if r['seed']==seed];historic=vals[0]['origin']=='historical_quality_only'
                            ax.plot([r[xkey] for r in vals],[r[metric] for r in vals],color=color,alpha=.55,linestyle='--' if historic else '-',
                                    label=model+(' historical' if historic else '') if n==0 else None)
                    ax.set(xlabel=xkey,ylabel=metric,title=support)
                    if ax.lines: ax.legend(fontsize=8)
                fig.suptitle(stage+'; fixed update budget');fig.tight_layout();fig.savefig(target/f'{stage}_{metric}_{xkey}.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for support,marker in [('relational','o'),('grid4','s')]:
        for model,color in zip(LABELS,['C0','C1','C2']):
            selected=[r for r in rows if r['stage']=='final' and r['support']==support and r['model']==model and r.get('forward_b64_median_seconds') and r['solved_rate'] is not None]
            if selected: ax.scatter([1000*r['forward_b64_median_seconds']/64 for r in selected],[100*r['solved_rate'] for r in selected],label=support+' '+model,color=color,marker=marker)
    ax.set(xlabel='Forward ms per puzzle, batch 64 throughput cost',ylabel='Test solved (%)',title='Absolute quality and measured cost')
    if ax.collections: ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(target/'quality_cost.png',dpi=140);plt.close(fig)
    queue=read(root()/'queue/state.json') if (root()/'queue/state.json').exists() else {'status':'prepared','gpu_seconds_used':0,'gpu_jobs':[]}
    new_done=sum(r['status']=='completed' and r['origin']=='new_training' for r in rows);planned=33 if reuse else 48
    interaction=next(r for r in contrasts if r['contrast']=='I')
    conclusion='Основное сравнение I пока недоступно: нет пяти парных новых grid4 результатов. Улучшение не установлено.'
    if interaction['complete_five_seeds']:
        conclusion=f'I = {interaction["mean_pp"]:.4f} п.п.; 95% seed-bootstrap [{interaction["lower95_pp"]:.4f}, {interaction["upper95_pp"]:.4f}]. '
        conclusion += 'Интервал включает ноль; данных недостаточно для подтверждения преимущества.' if interaction['lower95_pp']<=0<=interaction['upper95_pp'] else 'Интервал не включает ноль; знак и абсолютное качество оцениваются совместно.'
    lines=['# Sudoku: абляция локальной маски — текущий отчёт','',
           f'Серия `sudoku_local_mask_v1`; научный код `{code_hash()}`. Завершено новых обучений: **{new_done}/{planned}**. Исторических relational-контролей: **{15 if reuse else 0}/15**.',
           f'Состояние очереди: `{queue["status"]}`. Учтено завершённых GPU-выделений: **{queue["gpu_seconds_used"]/3600:.4f} ч**. Активное выделение, если есть, добавляется после accounting.',
           '', '## Вопрос и изменение','',
           'Меняется только коммуникационная маска: relational соединяет общую строку/столбец/квадрат, grid4 — self и четыре ближайших соседа. Диаметр 2 против 16; 21 связь в строке против 3/4/5. У grid4 369 разрешённых элементов и 144 недиагональных неориентированных ребра.',
           'Проверяются F4-onehop, F4-binomial и R32-onehop на обеих масках. Четыре независимых блока у F4, один общий блок с 32 полными обновлениями у R32. Width128, 4 головы, внешний FFN128→512→128, pre-LN/residual; без inner-MLP, gate и промежуточного loss.',
           'Фильтр binomial: (4,6,4,1)/15, r=4, средний индекс переноса 32/15. Максимум последовательных переносов — 4/16/32; это не гарантия качества или массы дальнего сигнала.',
           '', '## Данные и протокол','',
           'Официальный Sudoku-Hard, 180000 train / 18000 validation / 18000 test. Общие puzzle/target массивы и 38 признаков; mask hash отдельный. Маска не зависит от цифр/targets. Никакого глобального pooling, кандидатов, solver или repair.',
           '18 пилотов: 6 условий × seeds101,103,107, 10000 обновлений при scheduler horizon30000. Финалы: 6×seeds11,23,37,53,71, 30000 обновлений; новые финалы стартуют с нуля.',
           'Batch64 с возвращением; AdamW lr.001, wd.01, clip1, warmup5%, cosine. Validation полностью на шаге1 и каждые250. Выбор best: максимум solved, минимум CE, самый ранний шаг.',
           'Loss — CE только пустых клеток, среднее внутри каждой задачи, затем по batch. Decoding копирует clues, остальные argmax. Validator сохраняет настоящие правила Sudoku.',
           'Исторические source/config hashes сохранены; неактивные поля legacy объяснены в checks/legacy_compatibility.json. Старые времена не используются как доказательство ускорения нового кода.',
           '', '## Выполненные проверки','']
    for name in ['pytest','data','legacy_compatibility','smoke','overfit']:
        p=root()/f'checks/{name}.json';status=read(p)['status'] if p.exists() else 'not_executed'
        lines.append(f'- {name}: {status}.')
    lines += ['','Низкий solved на overfit у локальной F4-onehop сам по себе не означает ошибку. Маска, отсутствие дальнего влияния и нулевая производная, конечные gradients, legacy forward и точный resume проверяются отдельно.',
              '', '## Результаты','', '| stage | mask | model | completed seeds | solved mean ± sample SD, % | origin |','|---|---|---|---:|---:|---|']
    for item in summary:
        mean=item['solved_rate_mean'];sd=item['solved_rate_sample_sd']
        score='нет измерений' if mean is None else f'{100*mean:.4f}'+(f' ± {100*sd:.4f}' if sd is not None else '')
        lines.append(f'| {item["stage"]} | {item["support"]} | {item["model"]} | {item["completed_evaluated_seeds"]} | {score} | {item["origin"]} |')
    lines += ['', 'Пилотные строки используют последний validation, финальные — test выбранного checkpoint. Неполные seeds не считаются завершёнными.',
              '', '## Главное сравнение','',conclusion,
              'G_grid, G_rel, I, изменения моделей между масками и сравнения с R32 приведены в contrasts_per_seed.csv и contrasts_summary.csv. Метод: percentile bootstrap по парным обучающим seeds, 20000 повторов, seed90421, 95%; sample SD с ddof1. Пять seeds дают ограниченную точность; 18000 задач не заменяют независимые обучения.',
              '', '## Стоимость и диагностика','',
              'Forward-only и полный inference оцениваются после warmup отдельно при batch1 и64. Raw повторы, медианы и квартили сохраняются после фактического измерения; отсутствие файла означает отсутствие замера. Batch time/64 — стоимость throughput, а не latency одиночного запроса.',
              'Плотные masked PyTorch tensors остаются плотными вычислениями: grid4 не гарантирует ускорения. Старое качество помечено как историческое, inference прежних весов измеряется заново.',
              'Кривые solved, CE и blank accuracy строятся против updates и накопленного training time. Исторические времена помечены отдельно; это не отдельные equal-wall-time обучения.',
              'Пороги50%/90% подтверждаются двумя последовательными validation. Не достигнутые и pending наблюдения не выбрасываются. Для новых запусков сохраняются предыдущие failing/первые passing веса; исторические пороги восстанавливаются только по кривым.',
              ('На первых64 validation-задачах сохранена micro blank accuracy по четырём углам,28 остальным краям и49 внутренним клеткам; сохранены число пустых клеток, числитель и знаменатель.' if positions else
               'Запланирована micro blank accuracy на первых64 validation-задачах по четырём углам,28 остальным краям и49 внутренним клеткам. Эти измерения ещё не выполнялись; после них будут сохранены числитель и знаменатель каждой группы.'),
              '', '## Ограничения','',
              'Положительное I может быть следствием падения onehop; без абсолютного качества это не успех метода. Если на grid4 solved близок к нулю у всех моделей, результат так и интерпретируется.',
              'Смена графа одновременно меняет диаметр, степень и соответствие Sudoku-ограничениям. Это эффект данной маски, а не чистый причинный эффект диаметра.',
              'Это последующее исследование на известном benchmark с опубликованными test-результатами. Оно не является независимым подтверждением на никогда не виденном test.',
              '', '## Продолжение','',
              'Инструкции установки, подготовки, проверки, budget grant, resume и evaluation: research/sudoku_local_mask_v1/README.md.',
              'Основная очередь сохраняется в manifest.json; задания и фактически выделенное время — queue/state.json. Без отдельного подтверждённого лимита GPU-задание остаётся held. Точная смета и недостающий ресурс появляются после разрешённого GPU-профиля.',
              'Предыдущие кампании и их бюджеты не изменяются. Внешняя публикация этой серии не выполняется.']
    Path('REPORT_SUDOKU_LOCAL_MASK_RU.md').write_text('\n'.join(lines)+'\n')
    write_json(target/'report_manifest.json',{'source_sha256':code_hash(),'new_trainings_completed':new_done,'new_trainings_planned':planned,
               'historical_relational':15 if reuse else 0,'primary_interaction':interaction,'accounted_gpu_seconds':queue['gpu_seconds_used']})
    return {'new_trainings_completed':new_done,'new_trainings_planned':planned,'historical_relational':15 if reuse else 0,'primary_interaction':interaction}
