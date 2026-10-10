"""Build the publication from archived measurements, without running training."""
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['onehop', 'binomial', 'exponential_truncated', 'finite_geometric', 'learned_polynomial', 'recurrent']
LABELS = dict(zip(METHODS, ['One hop', 'Binomial', 'Truncated exp.', 'Finite geometric', 'Learned polynomial', 'Recurrent']))
TASKS = ['maze', 'sudoku']


def read(name):
    return json.loads((ROOT / name).read_text())


def csv_rows(name):
    with (ROOT / 'results/tables' / name).open(newline='') as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    def cells(row):
        return '| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in row) + ' |'
    return '\n'.join([cells(headers), cells(['---'] * len(headers)), *(cells(r) for r in rows)])


def record_dir(row):
    return ROOT / 'results/records' / row['task'] / row['method'] / row['config_hash'] / ('seed_' + row['seed'])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'svg.hashsalt': 'binomial-attention-snapshot'})
    rows = csv_rows('results.csv')
    final = sorted([r for r in rows if r['stage'] == 'final' and float(r['inner_beta']) == 0],
                   key=lambda r: (TASKS.index(r['task']), METHODS.index(r['method']), int(r['seed'])))
    pilots = [r for r in rows if r['stage'] == 'mlp_pilot']
    aggregates = {(r['task'], r['method']): r for r in csv_rows('final_aggregates.csv')}
    groups = {(t, m): [r for r in final if (r['task'], r['method']) == (t, m)] for t in TASKS for m in METHODS}
    figures = ROOT / 'results/figures'
    figures.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(figures / (name + '.png'), bbox_inches='tight')
        fig.savefig(figures / (name + '.svg'), bbox_inches='tight', metadata={'Date': None})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for i, method in enumerate(METHODS):
            values = np.array([float(r['solved_rate']) * 100 for r in groups[task, method]])
            ax.bar(i, values.mean(), yerr=values.std(ddof=1), capsize=4, color=f'C{i}', alpha=.72)
            ax.scatter(i + np.linspace(-.12, .12, len(values)), values, color='black', s=14, zorder=3)
        ax.set(xticks=range(6), xticklabels=[LABELS[m] for m in METHODS], ylim=(0, 105),
               ylabel='Test puzzles solved (%)', title=task.title() + ': 30,000 updates, five seeds')
        ax.tick_params(axis='x', rotation=35)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Final primary series: mean, sample SD, and individual seeds')
    save(fig, 'final_solved')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for i, method in enumerate(METHODS):
            records = [[json.loads(line) for line in (record_dir(r) / 'metrics.jsonl').read_text().splitlines()] for r in groups[task, method]]
            steps = [x['step'] for x in records[0]]
            assert all([x['step'] for x in rec] == steps for rec in records)
            values = np.array([[x['solved_rate'] * 100 for x in rec] for rec in records])
            avg, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(steps, avg, label=LABELS[method], color=f'C{i}')
            ax.fill_between(steps, np.maximum(0, avg - sd), np.minimum(100, avg + sd), color=f'C{i}', alpha=.1)
        ax.set(title=task.title(), xlabel='Optimizer updates', ylabel='Validation solved (%)', ylim=(0, 102))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Validation trajectories: mean ± sample SD across five seeds')
    save(fig, 'learning_curves')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    moment_rows = []
    for ax, task in zip(axes, TASKS):
        for method in METHODS[1:4]:
            r = groups[task, method][0]
            f = json.loads(r['filters_by_block'])[0]
            coeff = f['coefficients']
            ax.plot(range(1, len(coeff) + 1), coeff, marker='o', label=LABELS[method])
            moment_rows.append([task, method, len(coeff), f"{f['mu']:.8f}", f"{f['nu']:.8f}", f"{f['variance']:.8f}", f"{coeff[-1]:.8f}"])
        ax.set(title=f"{task.title()}, H={len(coeff)}", xlabel='Hop s', ylabel='Coefficient c(s)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed filters: equal mean hop count, different tails')
    save(fig, 'filter_weights')

    mlp_table = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for i, method in enumerate(METHODS[1:4]):
            pair = []
            for beta in [0., .1]:
                rr = sorted([r for r in pilots if r['task'] == task and r['method'] == method and float(r['inner_beta']) == beta], key=lambda r: int(r['seed']))
                values = [float(r['solved_rate']) * 100 for r in rr]
                pair.append((rr, values))
                x = i + (-.18 if beta == 0 else .18)
                ax.bar(x, mean(values), width=.32, color='C0' if beta == 0 else 'C1', alpha=.75,
                       label=('MLP off' if beta == 0 else 'MLP on') if i == 0 else None)
                ax.scatter([x - .035, x + .035], values, color='black', s=18, zorder=3)
            (off, ov), (on, nv) = pair
            mlp_table.append([task, method, ', '.join(f'{v:.4f}' for v in ov), ', '.join(f'{v:.4f}' for v in nv),
                              f'{mean(ov):.4f}', f'{mean(nv):.4f}', f'{mean(nv)-mean(ov):+.4f}',
                              f"{mean(float(r['training_seconds']) for r in off):.3f}", f"{mean(float(r['training_seconds']) for r in on):.3f}"])
        ax.set(xticks=range(3), xticklabels=[LABELS[m] for m in METHODS[1:4]], ylabel='Validation solved (%)', title=task.title())
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.suptitle('MLP pilot only: 500 updates, two seeds; different y-axis scales')
    save(fig, 'mlp_pilot')

    counts = Counter((r['stage'], r['run_status'], r['updates']) for r in rows)
    run_counts = table(['Этап', 'Статус записи', 'Обновлений', 'Число конфигураций'],
                       [list(k) + [n] for k, n in sorted(counts.items())])
    main_rows = []
    time_rows = []
    threshold_rows = []
    for task in TASKS:
        for method in METHODS:
            a = aggregates[task, method]; rr = groups[task, method]
            main_rows.append([task, method, f"{float(a['mean'])*100:.4f} ± {float(a['sample_sd'])*100:.4f}",
                              ', '.join(f"{float(r['solved_rate'])*100:.4f}" for r in rr), rr[0]['parameter_count']])
            time_rows.append([task, method] + [f'{mean(float(r[k]) for r in rr):.3f}' for k in ['training_seconds', 'elapsed_seconds']] +
                             [f"{mean(float(r['inference_seconds_per_example']) for r in rr)*1e6:.2f}", f"{max(float(r['train_memory']) for r in rr)/2**20:.1f}"])
            for threshold in ([90] if task == 'maze' else [50, 90]):
                states = [json.loads(r[f'threshold_{threshold}']) for r in rr]
                reached = [x for x in states if x['status'] == 'reached']
                threshold_rows.append([task, method, threshold, f'{len(reached)}/5',
                                       ', '.join(str(x.get('first_passing_step', 'ценз.')) for x in states)])
    pairs = csv_rows('paired_differences.csv')
    paired_table = table(['Задача', 'Разность', 'Среднее, п.п.', '95% bootstrap CI, п.п.'],
                         [[r['task'], r['comparison'], f"{float(r['mean_difference'])*100:+.4f}",
                           f"[{float(r['bootstrap_seed_CI_low'])*100:+.4f}; {float(r['bootstrap_seed_CI_high'])*100:+.4f}]"] for r in pairs])
    horizons = read('results/protocols/horizon_selection.json')
    horizon_table = table(['Задача', 'H', 'Средний validation solved, %', 'Средний validation loss', 'Выбран'],
                          [[t, h, f'{v*100:.6f}', f'{-loss:.6f}', 'да' if horizons['chosen_H'][t] == h else '']
                           for t in TASKS for v, loss, h in horizons['scores'][t]])
    pointers = csv_rows('pointer_bridge.csv')
    pointer_rows = []
    for condition in sorted({r['condition'] for r in pointers}):
        for method in METHODS[1:4]:
            rr = [r for r in pointers if r['condition'] == condition and r['method'] == method]
            pointer_rows.append([condition, method, rr[0]['H']] + [f'{mean(float(r[k]) for r in rr):.8f}'
                                for k in ['conditional_accuracy', 'terminal_mass', 'unreached_accuracy']])
    modules = [
        ('src/transport_reasoning/config.py', 'Валидация конфигураций, hashes и пути артефактов'),
        ('src/transport_reasoning/data/__init__.py', 'Maze, RRN CSV, manifests, входные признаки'),
        ('src/transport_reasoning/models/__init__.py', 'Encoder, cached transport, recurrent и inner MLP'),
        ('src/transport_reasoning/filters/__init__.py', 'Коэффициенты, согласование среднего, моменты'),
        ('src/transport_reasoning/training/__init__.py', 'Обучение, RNG, checkpoint, восстановление'),
        ('src/transport_reasoning/evaluation/__init__.py', 'Loss, decode, независимые валидаторы'),
        ('scripts/campaign.py', 'Первичная последовательность пилотов и финальных запусков'),
        ('scripts/revision_audit.py', 'Повторная проверка данных, весов, предсказаний и протокола'),
        ('scripts/mlp_final_campaign.py', 'Замороженная очередь из 20 MLP запусков'),
        ('scripts/audited_mlp_runtime.py', 'Архивирование порогов и восстановление'),
        ('scripts/chunked_mlp_runtime.py', 'Выполнение очереди в отдельных временных слотах'),
        ('scripts/mlp_chunk_scheduler.py', 'Последовательное планирование и бюджет'),
        ('scripts/build_public_assets.py', 'Публичный отчёт и рисунки из сохранённых результатов')]
    replacements = {
        'SNAPSHOT': read('PUBLICATION_MANIFEST.json')['snapshot_utc'], 'RUN_COUNTS': run_counts,
        'FILTER_MOMENTS': table(['Задача', 'Фильтр', 'H', 'μ', 'ν=E[s(s−1)]', 'Var(s)', 'c(H)'], moment_rows),
        'HORIZON_TABLE': horizon_table,
        'MAIN_RESULTS': table(['Задача', 'Метод', 'Solved %, mean ± SD', 'Seeds 11, 23, 37, 53, 71 (%)', 'Параметров'], main_rows),
        'PAIRED_RESULTS': paired_table,
        'TIMING_TABLE': table(['Задача', 'Метод', 'Train s, mean', 'Elapsed s, mean', 'Forward μs/example, mean', 'Peak train MiB, max'], time_rows),
        'THRESHOLD_TABLE': table(['Задача', 'Метод', 'Порог %', 'Достигли', 'Первый passing step по пяти seeds'], threshold_rows),
        'MLP_PILOT_TABLE': table(['Задача', 'Метод', 'Off seeds %', 'On seeds %', 'Off mean %', 'On mean %', 'Δ п.п.', 'Off train s', 'On train s'], mlp_table),
        'POINTER_TABLE': table(['Условие', 'Метод', 'H eval', 'Conditional accuracy', 'Terminal mass', 'Unreached accuracy'], pointer_rows),
        'MODULE_TABLE': table(['Файл', 'Назначение'], [[f'[{p}]({p})', desc] for p, desc in modules])}
    text = (ROOT / 'scripts/report_template_ru.txt').read_text()
    for key, value in replacements.items():
        assert '@@' + key + '@@' in text, key
        text = text.replace('@@' + key + '@@', value)
    assert '@@' not in text
    (ROOT / 'REPORT_RU.md').write_text(text)
    appendix = ['# Полные таблицы экспериментов', 'Автоматическое приложение к [отчёту](../REPORT_RU.md). Solved приведён в процентах. Основная серия: test; пилоты: validation; overfit: train_subset. Пауза LR-кандидата означает завершение его отборочного бюджета, а не финальное обучение.',
                '## 60 финальных запусков', table(['Задача', 'Метод', 'Seed', 'Solved %', 'Loss', 'Best step', 'Train s', 'Config hash'],
                [[r['task'], r['method'], r['seed'], f"{float(r['solved_rate'])*100:.6f}", r['loss'], r['filter_checkpoint_step'], f"{float(r['training_seconds']):.3f}", r['config_hash']] for r in final])]
    for stage in ['mlp_pilot', 'overfit', 'pilot', 'horizon_pilot', 'smoke']:
        rr = sorted([r for r in rows if r['stage'] == stage], key=lambda r: (r['task'], r['method'], r['seed'], r['LR'], r['inner_beta'], r['H']))
        appendix += ['## ' + stage, table(['Задача', 'Метод', 'Seed', 'H', 'β', 'LR', 'Updates', 'Status', 'Split', 'Solved %', 'Config hash'],
                     [[r['task'], r['method'], r['seed'], r['H'], r['inner_beta'], r['LR'], r['updates'], r['run_status'], r['split'], f"{float(r['solved_rate'])*100:.6f}", r['config_hash']] for r in rr])]
    appendix += ['## Время и память', replacements['TIMING_TABLE'], '## Пороги качества', replacements['THRESHOLD_TABLE'],
                 '## Pointer: все 36 оценок', table(['Условие', 'Метод', 'Seed', 'H', 'Conditional accuracy', 'Terminal mass', 'Unreached accuracy'],
                 [[r[k] for k in ['condition', 'method', 'seed', 'H', 'conditional_accuracy', 'terminal_mass', 'unreached_accuracy']] for r in pointers]),
                 '## Идентификаторы данных', 'Manifest содержит исходные CSV hashes и hashes полученных NPZ. Байты NPZ могут зависеть от версии упаковки; воспроизведение на другом окружении следует проверять также по содержимому массивов.',
                 '```json\n' + json.dumps(read('results/protocols/mlp_final_protocol.json')['data_manifests'], indent=2, ensure_ascii=False) + '\n```']
    (ROOT / 'docs').mkdir(exist_ok=True)
    (ROOT / 'docs/EXPERIMENT_TABLES_RU.md').write_text('\n\n'.join(appendix) + '\n')
    print(json.dumps({'report': 'REPORT_RU.md', 'appendix': 'docs/EXPERIMENT_TABLES_RU.md', 'final_runs': len(final), 'all_records': len(rows), 'figures': 4}))


if __name__ == '__main__':
    main()
