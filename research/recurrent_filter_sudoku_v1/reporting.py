"""Rebuild tables and figures from immutable per-seed artifacts, including missing rows."""
import csv
import json
from pathlib import Path
import numpy as np
from transport_reasoning.config import write_json
from .config import Config, code_hash, grid, root
from .campaign import read, legacy_folder


def table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)


def paired(rows):
    final = {(r['label'], r['seed']): r for r in rows if r['stage'] == 'final' and r.get('solved_rate') is not None}
    result = []; generator = np.random.default_rng(90421)
    for left, right in [('B16','R16'),('B16','R32'),('B32','R32')]:
        seeds = sorted({s for label,s in final if label == left} & {s for label,s in final if label == right})
        diffs = np.array([100*(final[left,s]['solved_rate']-final[right,s]['solved_rate']) for s in seeds])
        row = {'comparison': left + '-' + right, 'paired_seeds': len(seeds), 'seeds': str(seeds),
               'mean_difference_pp': float(diffs.mean()) if len(diffs) else None,
               'sample_sd_pp': float(diffs.std(ddof=1)) if len(diffs)>1 else None,
               'lower95_pp': None, 'upper95_pp': None, 'complete_five_seed_comparison': len(seeds)==5}
        if len(diffs) >= 2:
            samples = generator.choice(diffs, size=(20000,len(diffs)), replace=True).mean(1)
            row['lower95_pp'], row['upper95_pp'] = [float(x) for x in np.quantile(samples,[.025,.975])]
        result.append(row)
    return result


def report():
    if not __import__('os').environ.get('SLURM_JOB_ID'): raise RuntimeError('Analysis requires scheduler allocation')
    target = root() / 'report'; target.mkdir(parents=True, exist_ok=True)
    checks_path = root() / 'checks/legacy_compatibility.json'
    history = {r['seed']: r for r in read(checks_path)['historical_final_results']} if checks_path.exists() else {}
    manifest = read(root() / 'manifest.json'); statuses = {r['config_hash']: r for r in manifest['rows']}
    decision = read(root() / 'admission.json') if (root() / 'admission.json').exists() else None
    rows = []; curves = []; thresholds = []; mechanisms = []
    for c in grid(include_b32=True):
        historical = c.stage == 'final' and c.label() == 'R32' and c.seed in history
        status = statuses[c.hash()]['status']; r = None; metric = None
        if historical:
            status = 'completed'; metric = history[c.seed]['evaluation']
            try: curve_path = legacy_folder(c.seed) / 'metrics.jsonl'
            except (KeyError, RuntimeError): curve_path = None
        else:
            curve_path = c.run_dir() / 'metrics.jsonl'
            if (c.run_dir() / 'result.json').exists():
                r = read(c.run_dir() / 'result.json'); status = r['status']
                if c.stage == 'pilot': metric = r['last_validation']
            if (c.run_dir() / 'evaluation_test.json').exists(): metric = read(c.run_dir() / 'evaluation_test.json')
        row = {'stage': c.stage, 'label': c.label(), 'seed': c.seed, 'status': status,
               'origin': 'historical_quality_only' if historical else 'new',
               'config_hash': c.hash(), 'completed_steps': 30000 if historical else r['completed_steps'] if r else 0,
               'planned_steps': c.planned_steps, 'completion_steps': c.completion_steps,
               'optional': c.stage == 'final' and c.label() == 'B32',
               'parameters': 204169, 'T': c.recurrent_steps, 'r': c.hops, **c.operations()}
        for key in ['solved_rate','exact_match','blank_accuracy','loss']: row[key] = metric.get(key) if metric else None
        if r:
            row.update(train_seconds=r['timings']['training'], validation_seconds=r['timings']['validation'],
                       checkpoint_seconds=r['timings']['checkpoints'], elapsed_seconds=r['timings']['elapsed'],
                       training_examples_per_second=r['examples']/max(r['timings']['training'],1e-12),
                       peak_memory_bytes=(r['last_validation'] or {}).get('train_peak_memory_bytes'))
            for p, val in r['thresholds'].items(): thresholds.append({'label': c.label(),'seed':c.seed,'stage':c.stage,'threshold':p,**val})
        measurement = c.run_dir() / 'measurement.json'
        if measurement.exists():
            measured = read(measurement)
            for b in measured['inference']:
                size = b['batch_size']; row[f'forward_b{size}_median_seconds'] = b['forward_batch_seconds']['median']
                row[f'forward_b{size}_p25_seconds'] = b['forward_batch_seconds']['p25']
                row[f'forward_b{size}_p75_seconds'] = b['forward_batch_seconds']['p75']
                row[f'full_b{size}_median_seconds'] = b['end_to_end_batch_seconds']['median']
            for val in measured['mechanism']['rows']: mechanisms.append({'label':c.label(),'seed':c.seed,**val})
        if curve_path and curve_path.exists():
            for line in curve_path.read_text().splitlines():
                val = json.loads(line)
                curves.append({'label':c.label(),'seed':c.seed,'stage':c.stage,'origin':row['origin'],
                               'step':val['step'],'solved_rate':val['solved_rate'],
                               'elapsed_seconds':val.get('elapsed_seconds'),'training_seconds':val.get('training_seconds')})
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    for stage in ['pilot','final']: table(target / (stage + '_seeds.csv'),[r for r in rows if r['stage']==stage],fields)
    summary = []
    for stage in ['pilot','final']:
        for label in ['R16','B16','R32','B32']:
            group = [r for r in rows if r['stage']==stage and r['label']==label and r['status']=='completed' and r['solved_rate'] is not None]
            item = {'stage':stage,'label':label,'completed_evaluated_seeds':len(group)}
            for key in ['solved_rate','blank_accuracy','loss','parameters','train_seconds','elapsed_seconds','forward_b1_median_seconds','forward_b64_median_seconds','peak_memory_bytes']:
                vals = [r[key] for r in group if r.get(key) is not None]
                item[key + '_mean'] = float(np.mean(vals)) if vals else None
                item[key + '_sample_sd'] = float(np.std(vals,ddof=1)) if len(vals)>1 else None
            summary.append(item)
    comparisons = paired(rows)
    table(target / 'summary.csv',summary,list(summary[0])); table(target / 'paired.csv',comparisons,list(comparisons[0]))
    table(target / 'validation_curves.csv',curves,['stage','label','seed','origin','step','solved_rate','elapsed_seconds','training_seconds'])
    table(target / 'thresholds.csv',thresholds,list(dict.fromkeys(k for r in thresholds for k in r)) or ['stage','label','seed','threshold','status'])
    table(target / 'mechanism.csv',mechanisms,list(dict.fromkeys(k for r in mechanisms for k in r)) or ['label','seed','iteration'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage in ['pilot','final']:
        for horizontal in ['step','elapsed_seconds']:
            fig, ax = plt.subplots(figsize=(8,5))
            for label,color in zip(['R16','B16','R32','B32'],['C0','C1','C2','C3']):
                subset = [r for r in curves if r['stage']==stage and r['label']==label and r[horizontal] is not None]
                for i,seed in enumerate(sorted({r['seed'] for r in subset})):
                    vals = [r for r in subset if r['seed']==seed]
                    ax.plot([r[horizontal] for r in vals],[100*r['solved_rate'] for r in vals],color=color,alpha=.5,
                            label=label + (' (historical)' if vals[0]['origin']=='historical_quality_only' else '') if i==0 else None)
            ax.set(xlabel=horizontal, ylabel='Validation solved (%)',title=stage + '; fixed updates protocol')
            if ax.lines: ax.legend()
            fig.tight_layout(); fig.savefig(target / f'{stage}_quality_{horizontal}.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for label in ['R16','B16','R32','B32']:
        vals=[r for r in rows if r['stage']=='final' and r.get('forward_b64_median_seconds') and r.get('solved_rate') is not None]
        vals=[r for r in vals if r['label']==label]
        if vals: ax.scatter([1000*r['forward_b64_median_seconds']/64 for r in vals],[100*r['solved_rate'] for r in vals],label=label)
    ax.set(xlabel='Forward milliseconds per puzzle, batch 64',ylabel='Test solved (%)',title='Measured quality and inference cost')
    if ax.collections: ax.legend()
    fig.tight_layout(); fig.savefig(target/'quality_cost.png',dpi=160); plt.close(fig)
    gpu = read(root()/'queue/state.json').get('gpu_seconds_used',0)/3600 if (root()/'queue/state.json').exists() else 0
    new_completed=sum(r['origin']=='new' and r['status']=='completed' and not r['optional'] for r in rows)
    lines=['# Рекуррентный фильтр на Sudoku — текущий отчёт','',
           'Это последующее исследование на ранее оценённом benchmark. Исходная серия сохранена без изменений.',
           '',f'Исходный код: `{code_hash()}`. Завершено новых обязательных обучений: {new_completed}/22. Исторических совместимых R32: {len(history)}/5.',
           f'Учтено завершённых GPU-выделений: {gpu:.4f} ч; активное выделение, если есть, оплачивается отдельно до следующего accounting.',
           '', '## Вопросы и протокол','',
           'Q1: эффект фильтра при одинаковом T. Q2: B16 против R32 по качеству и фактическому времени. Q3: ограниченная диагностика оптимизации и смешивания.',
           'Один общий блок, width=128, 4 головы; 204169 параметров. R16/R32 используют один перенос внутри каждой рекуррентной итерации; B16/B32 — четыре с коэффициентами (4,6,4,1)/15.',
           'Пилоты: 4×3 seeds, 10000 обновлений, scheduler horizon 30000. Финалы R16/B16/R32: 5 seeds, 30000; новые финальные веса не продолжают пилоты.',
           '180000/18000/18000 задач, 38 входных признаков. Loss только по пустым клеткам, нормирование внутри задачи. Validation полностью каждые 250 шагов и на шаге 1. Checkpoint: максимум solved, минимум loss, самый ранний шаг.',
           '', '## Проверки','']
    for name in ['pytest','data','legacy_compatibility','overfit']:
        path=root()/f'checks/{name}.json'; check=read(path) if path.exists() else {'status':'not_executed'}
        lines.append(f'- {name}: {check["status"]}.')
    lines += ['', '## Результаты','', '| stage | model | completed seeds | solved, % mean ± sample SD |', '|---|---|---:|---:|']
    for s in summary:
        mean=s['solved_rate_mean']; sd=s['solved_rate_sample_sd']
        text='нет измерений' if mean is None else f'{100*mean:.4f}' + (f' ± {100*sd:.4f}' if sd is not None else '')
        lines.append(f'| {s["stage"]} | {s["label"]} | {s["completed_evaluated_seeds"]} | {text} |')
    lines += ['', 'Пилотная таблица использует последний validation; финальная — test выбранного по validation checkpoint. R32 final помечен как исторический в CSV.',
              '', '## Качество, стоимость и выводы','',
              'Парные разности в `paired.csv`: процентные пункты, percentile bootstrap 20000 выборок по seeds, 95% интервал; при пяти seeds интервалы нестабильны.',
              'Формальная цель B16−R32: нижняя граница >−1 п.п.; близкое среднее само по себе не доказывает не меньшее качество.',
              'Текущая скорость R32 измеряется на тех же ресурсах. Старое training time не используется как доказательство ускорения нового кода.',
              'Forward-only и полный inference, batch 1 и 64, записаны раздельно. В `measurement.json` сохранены повторы, квартили и диагностика.',
              'Кривые по elapsed — дополнительный анализ исходного обучения с фиксированным числом обновлений; отдельного equal-wall-time обучения нет.',
              'Недостигнутые 50%/90% не исключаются из анализа; строки порогов сохраняют цензурирование. Диагностика дисперсии/энтропии не устанавливает причинность.',
              '', 'Q1/Q2/Q3 пока не имеют окончательного ответа.' if new_completed<22 else 'Численные сравнения завершённой серии приведены в CSV; причинное объяснение требует отдельных вмешательств.',
              '', '## Воспроизведение и продолжение','',
              'Инструкции, команды resume и порядок контроля бюджета: `research/recurrent_filter_sudoku_v1/README.md`.',
              'Новая серия не публикуется автоматически. Большие артефакты находятся в отдельном игнорируемом namespace.',
              'При отсутствии budget_grant очередь остаётся held. Причина ожидания и задания: `queue/state.json`; фактический запрос ресурсов после профиля: `profile/budget_estimate.json`.']
    (Path('REPORT_RECURRENT_SUDOKU_RU.md')).write_text('\n'.join(lines)+'\n')
    write_json(target/'report_manifest.json',{'source_sha256':code_hash(),'new_mandatory_trainings_completed':new_completed,'historical_R32':len(history),'paired':comparisons,'admission':decision})
    return {'new_mandatory_trainings_completed':new_completed,'historical_R32':len(history),'accounted_gpu_hours':gpu}
