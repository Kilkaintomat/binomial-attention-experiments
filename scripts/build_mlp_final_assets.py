"""Rebuild the dated MLP extension report from published per-run artifacts."""
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from build_public_assets import ROOT, table

SEEDS = [11, 23, 37, 53, 71]
TASKS = ['maze', 'sudoku']
METHODS = ['binomial', 'exponential_truncated']
LABELS = {'binomial': 'Binomial', 'exponential_truncated': 'Exponential'}


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / 'results/tables' / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)


def folder(config, config_hash, mlp=False):
    records_root = ROOT / ('results/mlp_final/records' if mlp else 'results/records')
    return records_root / config['task'] / config['method'] / config_hash / ('seed_' + str(config['seed']))


def records():
    audit = read(ROOT / 'results/audit/mlp_final_publication_audit.json')
    assert audit['status'] == 'passed' and audit['new_completed_runs'] == 20
    all_rows = []; curves = []; thresholds = []
    for item in audit['runs']:
        for beta, key in [(0., 'baseline_config_hash'), (.1, 'config_hash')]:
            directory = folder(item, item[key], mlp=bool(beta)); result = read(directory / 'result.json')
            config = result['config']; evaluation = read(directory / 'evaluation_test.json')
            assert config['inner_beta'] == beta and result['status'] == 'completed' and result['step'] == 30000
            row = {'task': config['task'], 'method': config['method'], 'seed': config['seed'], 'inner_beta': beta,
                   'config_hash': item[key], 'paired_mlp_config_hash': item['config_hash'], 'stage': 'final', 'split': 'test',
                   'updates': result['step'], 'test_count': evaluation['count'], 'checkpoint_step': evaluation['checkpoint_step'],
                   'H': config['hops'], 'layers': config['layers'], 'lr': config['lr'], 'batch_size': config['batch_size'],
                   'parameter_count': result['parameter_count'], 'data_hash': result['data_hash'], 'source_sha256': result['source_sha256'],
                   'origin': 'new_mlp_final' if beta else 'historical_paired_control',
                   'inference_timing_scope': 'synchronized forward only; batch 64 seconds per example; not single-request latency'}
            for name in ['solved_rate', 'exact_match', 'loss', 'blank_accuracy', 'vertex_f1', 'vertex_iou',
                         'inference_seconds_per_example', 'inference_peak_memory_bytes']:
                row[name] = evaluation.get(name)
            for name in ['training_seconds', 'evaluation_seconds', 'checkpoint_seconds', 'elapsed_seconds', 'train_peak_memory_bytes']:
                row[name] = result.get(name)
            all_rows.append(row)
            curve = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            for point in curve:
                curves.append({'task': row['task'], 'method': row['method'], 'seed': row['seed'], 'inner_beta': beta,
                               'step': point['step'], 'solved_rate': point['solved_rate'], 'loss': point['loss'],
                               'blank_accuracy': point.get('blank_accuracy'), 'training_seconds': point['training_seconds']})
            for level in ([.9] if row['task'] == 'maze' else [.5, .9]):
                crossing = next((i for i in range(1, len(curve)) if curve[i-1]['solved_rate'] >= level and curve[i]['solved_rate'] >= level), None)
                entry = {'task': row['task'], 'method': row['method'], 'seed': row['seed'], 'inner_beta': beta, 'threshold': level,
                         'status': 'not_reached' if crossing is None else 'confirmed', 'first_passing_step': None,
                         'confirming_step': None, 'training_seconds_upper': None, 'last_failing_step': None,
                         'training_seconds_lower': None, 'threshold_weights': 'historical_curve_only' if beta == 0 else 'not_applicable'}
                if crossing is not None:
                    entry.update(first_passing_step=curve[crossing-1]['step'], confirming_step=curve[crossing]['step'],
                                 training_seconds_upper=curve[crossing-1]['training_seconds'])
                    if crossing >= 2:
                        entry.update(last_failing_step=curve[crossing-2]['step'], training_seconds_lower=curve[crossing-2]['training_seconds'])
                    if beta: entry['threshold_weights'] = 'both_archives_verified_by_private_audit'
                thresholds.append(entry)
    return audit, all_rows, curves, thresholds


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Build research figures inside a scheduler allocation'
    audit, rows, curves, thresholds = records()
    rows.sort(key=lambda r: (r['task'], r['method'], r['seed'], r['inner_beta']))
    lookup = {(r['task'], r['method'], r['seed'], r['inner_beta']): r for r in rows}
    aggregates = []
    for task in TASKS:
        for method in METHODS:
            for beta in [0., .1]:
                selected = [lookup[task, method, seed, beta] for seed in SEEDS]
                item = {'task': task, 'method': method, 'inner_beta': beta, 'completed_seeds': 5,
                        'seeds': json.dumps(SEEDS), 'parameter_count': selected[0]['parameter_count']}
                for field in ['solved_rate', 'exact_match', 'loss', 'blank_accuracy', 'vertex_f1', 'training_seconds',
                              'evaluation_seconds', 'checkpoint_seconds', 'elapsed_seconds', 'inference_seconds_per_example', 'train_peak_memory_bytes']:
                    values = [r[field] for r in selected if r[field] is not None]
                    item[field+'_mean'] = mean(values) if values else None
                    item[field+'_sample_sd'] = stdev(values) if len(values) > 1 else None
                item['solved_per_seed'] = json.dumps({r['seed']: r['solved_rate'] for r in selected})
                aggregates.append(item)
    agg = {(r['task'], r['method'], r['inner_beta']): r for r in aggregates}
    pairs = []; contrasts = []; rng = np.random.default_rng(20261011)

    def summarize_contrast(task, comparison, values):
        interval = np.quantile(rng.choice(values, (20000, 5), replace=True).mean(1), [.025, .975])
        contrasts.append({'task': task, 'comparison': comparison, 'paired_seeds': 5, 'seeds': json.dumps(SEEDS),
                          'paired_differences_pp': json.dumps(values), 'mean_difference_pp': mean(values), 'sample_sd_pp': stdev(values),
                          'lower95_pp': float(interval[0]), 'upper95_pp': float(interval[1]), 'analysis': 'descriptive post-training seed bootstrap; unadjusted'})

    for task in TASKS:
        for method in METHODS:
            differences = []
            for seed in SEEDS:
                on = lookup[task, method, seed, .1]; off = lookup[task, method, seed, 0.]
                difference = 100*(on['solved_rate']-off['solved_rate']); differences.append(difference)
                pairs.append({'task': task, 'method': method, 'seed': seed, 'mlp_config_hash': on['config_hash'],
                              'baseline_config_hash': off['config_hash'], 'baseline_solved_percent': 100*off['solved_rate'],
                              'mlp_solved_percent': 100*on['solved_rate'], 'difference_pp': difference,
                              'training_seconds_ratio_on_off': on['training_seconds']/off['training_seconds']})
            summarize_contrast(task, method+'_MLP_minus_no_MLP', differences)
        summarize_contrast(task, 'binomial_minus_exponential_MLP',
                           [100*(lookup[task, 'binomial', seed, .1]['solved_rate']-lookup[task, 'exponential_truncated', seed, .1]['solved_rate']) for seed in SEEDS])
    write_csv('mlp_final_per_seed.csv', rows)
    write_csv('mlp_final_aggregates.csv', aggregates)
    write_csv('mlp_final_pairs.csv', pairs)
    write_csv('mlp_final_contrasts.csv', contrasts)
    write_csv('mlp_final_validation_curves.csv', curves)
    write_csv('mlp_final_thresholds.csv', thresholds)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'mlp-final-20261011'})
    figures = ROOT / 'results/figures'

    def save(fig, name):
        fig.savefig(figures/(name+'.png'), dpi=160, bbox_inches='tight')
        fig.savefig(figures/(name+'.svg'), metadata={'Date': None}, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for i, method in enumerate(METHODS):
            for beta, offset, color in [(0., -.18, 'C0'), (.1, .18, 'C1')]:
                values = np.array([100*lookup[task, method, s, beta]['solved_rate'] for s in SEEDS])
                ax.bar(i+offset, values.mean(), yerr=values.std(ddof=1), capsize=4, width=.32,
                       color=color, alpha=.7, label=('MLP off' if beta == 0 else 'MLP on') if i == 0 else None)
                ax.scatter(i+offset+np.linspace(-.08, .08, 5), values, color='black', s=14, zorder=3)
        if task == 'sudoku':
            recurrent = [r for r in csv.DictReader((ROOT/'results/tables/results.csv').open()) if r['task'] == task and r['method'] == 'recurrent' and r['stage'] == 'final']
            ax.axhline(mean(float(r['solved_rate']) for r in recurrent)*100, color='C2', linestyle=':', label='Historical recurrent T=32')
        ax.set(title=task.title(), xticks=range(2), xticklabels=[LABELS[m] for m in METHODS], ylabel='Test puzzles solved (%)', ylim=(0, 105))
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Completed inner-MLP extension: 30,000 updates; mean, sample SD and five seeds')
    save(fig, 'mlp_final_solved')
    for xfield in ['step', 'training_seconds']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for row_index, task in enumerate(TASKS):
            for column, metric in enumerate(['solved_rate', 'loss']):
                ax = axes[row_index, column]
                for method, color in zip(METHODS, ['C0', 'C1']):
                    for beta in [0., .1]:
                        for index, seed in enumerate(SEEDS):
                            selected = [r for r in curves if (r['task'], r['method'], r['inner_beta'], r['seed']) == (task, method, beta, seed)]
                            ax.plot([r[xfield] for r in selected], [r[metric]*(100 if metric == 'solved_rate' else 1) for r in selected],
                                    color=color, linestyle='-' if beta else '--', alpha=.55,
                                    label=LABELS[method]+(' MLP on' if beta else ' MLP off') if index == 0 else None)
                ax.set(title=task.title(), xlabel='Optimizer updates' if xfield == 'step' else 'Accumulated training seconds',
                       ylabel='Validation solved (%)' if metric == 'solved_rate' else 'Validation loss')
                if metric == 'solved_rate': ax.set_ylim(0, 102)
                else: ax.set_yscale('log')
                ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle('MLP final validation: individual seeds; fixed update budget')
        save(fig, 'mlp_final_curves_'+xfield)

    def score(task, method, beta):
        a = agg[task, method, beta]
        return f"{100*a['solved_rate_mean']:.4f} ± {100*a['solved_rate_sample_sd']:.4f}"

    main_table = []; time_table = []; seed_table = []
    for task in TASKS:
        for method in METHODS:
            c = next(r for r in contrasts if r['task'] == task and r['comparison'] == method+'_MLP_minus_no_MLP')
            main_table.append([task, LABELS[method], score(task, method, 0.), score(task, method, .1),
                               f"{c['mean_difference_pp']:+.4f}", f"[{c['lower95_pp']:+.4f}; {c['upper95_pp']:+.4f}]"])
            on, off = agg[task, method, .1], agg[task, method, 0.]
            ratios = [r['training_seconds_ratio_on_off'] for r in pairs if (r['task'], r['method']) == (task, method)]
            time_table.append([task, LABELS[method], f"{off['training_seconds_mean']:.2f}", f"{on['training_seconds_mean']:.2f}",
                               f'{mean(ratios):.3f}', f"{on['elapsed_seconds_mean']:.2f}",
                               f"{on['inference_seconds_per_example_mean']*1e6:.2f}", f"{on['train_peak_memory_bytes_mean']/2**20:.1f}"])
            seed_table.append([task, LABELS[method]]+[f"{lookup[task, method, s, .1]['solved_rate']*100:.4f}" for s in SEEDS])
    main_text = table(['Задача', 'Фильтр', 'Без MLP, %', 'С MLP, %', 'Δ, п.п.', '95% CI Δ, п.п.'], main_table)
    budget = audit['budget']; mlp_hours = budget['allocation_seconds_mlp_extension_including_failed_attempt']/3600
    total_hours = budget['total_original_campaign_gpu_seconds']/3600
    bin_s = agg['sudoku', 'binomial', .1]['solved_rate_mean']*100; exp_s = agg['sudoku', 'exponential_truncated', .1]['solved_rate_mean']*100
    summary_sentence = f'Внутренний MLP увеличил средний Sudoku solved до {bin_s:.2f}% у binomial и {exp_s:.2f}% у exponential; без него было около 5–6%. На Maze оба варианта решают более 99,7% задач в среднем.'
    lines = ['# Финальная серия с внутренним MLP', '',
             f'Срез: **{audit["snapshot_utc"]}**. Завершены **20/20** новых обучений по **30 000 optimizer updates**. Ни один seed не исключён. Это финальные test-результаты; короткие 500-step пилоты сюда не входят.', '',
             summary_sentence, '', main_text, '',
             'В таблице mean ± sample SD по пяти обучающим seeds: 11, 23, 37, 53, 71. Δ — парная разность MLP on − off на одинаковом seed. Test: 1000 лабиринтов или 18000 Sudoku. Единица независимого повторения — обучение, а не отдельная задача.', '',
             '![Качество с MLP и без него](results/figures/mlp_final_solved.png)', '',
             '## Все seeds с MLP', '', table(['Задача', 'Фильтр']+[str(s) for s in SEEDS], seed_table), '',
             'Полные исходные значения, CE, exact match, blank accuracy/vertex F1, best step, время и память: [CSV по seeds](results/tables/mlp_final_per_seed.csv). В нём 40 строк: 20 новых результатов и 20 соответствующих исторических контролей.', '',
             '## Постановка и реализация', '',
             'Парная абляция четырёх независимых внешних Transformer-блоков: меняется inner_beta с 0 на 0,1, inner_width=128. В каждом блоке матрица attention вычисляется один раз; между переносами добавляется общий для шагов этого блока MLP с GELU и LayerNorm, смешивающий головы. Следующий внешний блок пересчитывает attention. Внешний FFN сохранён. Это отдельная серия от новых рекуррентных фильтров и локальных Sudoku-масок.', '',
             'Maze: 8×8, 10000/1000/1000, H=16; Sudoku: официальный Sudoku-Hard, 180000/18000/18000, H=4 и стандартная маска строки/столбца/квадрата. Общие данные, именованная инициализация общих весов, поток minibatch с возвращением, batch 64, AdamW, warmup/cosine и 30000 updates сохранены. LR Maze/binomial=0,0003; остальные включённые конфигурации=0,001. Изначальная сетка и соответствие пар зафиксированы [до обучения](results/protocols/mlp_final_protocol.json).', '',
             'Validation выполнялась на шаге 1 и каждые 250 обновлений. Test посчитан у checkpoint с максимальным validation solved, затем минимальным validation loss и самым ранним шагом при точном равенстве. Заданные цифры Sudoku копируются; остальные — argmax, без solver/repair. Проверка решений использует полные правила Sudoku.', '',
             'MLP добавляет 133120 параметров: Maze 792577→925697, Sudoku 797449→930569. Число FFN-вызовов на forward возрастает с 4 до 68 на Maze и до 20 на Sudoku. Поэтому результат не изолирует эффект нелинейности от дополнительных параметров и вычислений.', '',
             '## Неопределённость и сравнение фильтров', '',
             table(['Задача', 'Сравнение', 'Среднее, п.п.', 'Sample SD, п.п.', '95% CI, п.п.'],
                   [[r['task'], r['comparison'], f"{r['mean_difference_pp']:+.4f}", f"{r['sample_sd_pp']:.4f}",
                     f"[{r['lower95_pp']:+.4f}; {r['upper95_pp']:+.4f}]"] for r in contrasts]), '',
             'Описательный анализ после завершения обучения: percentile bootstrap по пяти парным seeds, 20000 повторов, RNG seed 20261011. Интервалы не скорректированы за множественные сравнения и не являются новым заранее зарегистрированным подтверждающим тестом. Пять seeds ограничивают точность; интервал, включающий ноль, не доказывает эквивалентность.', '',
             'Добавление MLP помогает обоим фильтрам, но отдельного преимущества биномиального фильтра эти результаты не устанавливают. На Maze различается также LR. На Sudoku 39–40% остаются существенно ниже исторических 94,77% у recurrent T=32; это сравнение архитектур с разной глубиной, параметрами и вычислительной стоимостью.', '',
             '## Время, память и бюджет', '',
             table(['Задача', 'Фильтр', 'Train без MLP, с', 'Train с MLP, с', 'Среднее парное on/off', 'Elapsed с MLP, с', 'Forward с MLP, мкс/задачу при batch 64', 'Peak train с MLP, MiB mean'], time_table), '',
             f'Фактические выделения для MLP-расширения: **{budget["allocation_seconds_mlp_extension_including_failed_attempt"]} с = {mlp_hours:.4f} GPU-часа**, включая неудачную попытку сохранения и служебные расходы. Вся прежняя кампания: **{budget["total_original_campaign_gpu_seconds"]} с = {total_hours:.4f} GPU-часа из 24**. Новые рекуррентная и локальная серии сюда не включены.', '',
             'Training seconds включают batch, forward/backward и optimizer. Elapsed добавляет validation/checkpoint внутри обучающих вызовов; это не время календарного ожидания в очереди. Полные GPU-выделения учитываются отдельно. Forward timing исключает подготовку признаков, decode и validator; batch time/64 не является задержкой одиночного запроса. Приведены сохранённые измерения исходных обучений, без нового speed benchmark и без equal-wall-time retraining.', '',
             '![Validation против шагов](results/figures/mlp_final_curves_step.png)', '',
             '![Validation против накопленного training time](results/figures/mlp_final_curves_training_seconds.png)', '',
             'Пороги: Maze90%; Sudoku50%/90%, подтверждение двумя последовательными validation. Все достигнутые и недостигнутые исходы сохранены в [таблице порогов](results/tables/mlp_final_thresholds.csv); времена не усредняются только по успешным seeds. Для достигнутых новых порогов проверены веса до/после первого прохождения, для старых контролей используются кривые.', '',
             '## Выполненный аудит и границы вывода', '',
             'В отдельном CPU allocation проверены все 20 новых final configs, source/data hashes,121 validation-точка на каждый run, выбор best, полные 30000 шагов, конечный optimizer/scheduler и RNG. Совпадение общих начальных весов и потока обучающих batches с парным контролем проверено для каждого seed. Независимо перепроверены все **190000 сохранённых test-предсказаний**; ответы лучших checkpoint повторены на **320 примерах** — по 16 на run. Ошибок в этих проверках не найдено. Полный test-forward повторно не выполнялся.', '',
             'Machine-readable evidence: [новый аудит](results/audit/mlp_final_publication_audit.json). Прежний аудит 570000 предсказаний относится к основной серии и не переименован в аудит MLP. Архивные протоколы и результаты контролей сохранены. В публикации отсутствуют веса, массивы предсказаний, исходные datasets, внешние документы и приватные настройки.', '',
             'Официальный test уже известен из прежней работы. Это последующее исследование известного benchmark. Не выполнены outer-only parameter-matched контроль, финальный общий-LR контроль между Maze-фильтрами и отдельные обучения при одинаковом wall time. Нельзя приписать весь прирост исключительно чередованию переноса и нелинейности либо объявить все варианты эквивалентными.', '',
             '## Воспроизведение отчёта', '',
             'В CPU allocation из корня публичной копии:', '',
             '```bash\npython scripts/build_mlp_final_assets.py\npython scripts/check_mlp_final_snapshot.py\n```', '',
             'Для независимого повторения проверки private checkpoints нужны отдельно сохранённые артефакты и команда `scripts/audit_mlp_final_publication.py --artifact-root <artifact_root>`; CSV не заменяет эти веса. Новое обучение для построения таблиц не требуется.', '',
             '[Основной отчёт](REPORT_RU.md) · [Агрегаты](results/tables/mlp_final_aggregates.csv) · [Парные разности](results/tables/mlp_final_pairs.csv) · [Интервалы](results/tables/mlp_final_contrasts.csv)']
    (ROOT/'REPORT_MLP_FINAL_RU.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    readme_path=ROOT/'README.md';readme=readme_path.read_text()
    start='<!-- MLP_FINAL_RESULTS_START -->';end='<!-- MLP_FINAL_RESULTS_END -->'
    assert readme.count(start)==readme.count(end)==1
    before,rest=readme.split(start);_,after=rest.split(end)
    readme_path.write_text(before+start+'\n'+main_text+'\n'+end+after,encoding='utf-8')
    metadata = {'snapshot_utc': audit['snapshot_utc'], 'completed_new_runs': 20, 'historical_paired_controls': 20,
                'all_primary_plus_mlp_final_runs': 80, 'source_sha256': audit['source_sha256'],
                'budget': budget, 'analysis': audit['analysis'], 'main_results': main_table,
                'figures': ['mlp_final_solved', 'mlp_final_curves_step', 'mlp_final_curves_training_seconds'],
                'summary_ru': summary_sentence}
    (ROOT/'results/metadata/mlp_final_publication.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'completed': 20, 'summary': main_table, 'budget_gpu_hours': mlp_hours}, ensure_ascii=False))


if __name__ == '__main__':
    main()
