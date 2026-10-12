"""Versioned audit repair; frozen inference sources and predictions stay unchanged.

Require identical actual float32 coefficients, with 1e-14 absolute tolerance
only for float64 diagnostic metadata recomputed on another CPU.
"""
import csv
import itertools
from collections import defaultdict
from common import *
from transport_reasoning.reachability.config import allocation


COEFFICIENT_AUDIT = {'checked': 0, 'max_absolute_difference': {},
                     'float32_weights_bitwise_equal': True, 'float64_absolute_tolerance': 1e-14}


def assert_same_coefficients(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key in ['filter', 'rule', 'r'] or actual[key] is None:
            assert actual[key] == expected[key]
        else:
            a = np.asarray(actual[key], dtype=np.float64)
            b = np.asarray(expected[key], dtype=np.float64)
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            delta = float(np.max(np.abs(a-b)))
            assert delta <= 1e-14, (key, delta)
            COEFFICIENT_AUDIT['max_absolute_difference'][key] = max(
                COEFFICIENT_AUDIT['max_absolute_difference'].get(key, 0.), delta)
    # These are the coefficients used by the float32 network; exact equality remains mandatory.
    assert np.array_equal(np.asarray(actual['weights'], dtype=np.float32),
                          np.asarray(expected['weights'], dtype=np.float32))
    COEFFICIENT_AUDIT['checked'] += 1


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def direct_metrics(y,z):
    positive = y == 1
    pred = z >= 0
    tp = int((positive & pred).sum())
    tn = int((~positive & ~pred).sum())
    fp = int((~positive & pred).sum())
    fn = int((positive & ~pred).sum())
    return {'tp':tp,'tn':tn,'fp':fp,'fn':fn,'count':len(y),
        'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None,
        'accuracy':(tp+tn)/len(y)}


def main():
    allocation()
    freeze_protocol()
    all_configs = configs()
    manifest = read(OUT/'data/manifest.json')
    checks = {r['config_hash']:r for r in read(OUT/'checkpoints.json')}
    expected_count = PROTOCOL['evaluation']['expected_evaluations']
    per_seed, paths, datasets = [], [], {}
    prediction_count = 0
    immutable_checks = []
    for c in all_configs:
        rec = checkpoint_record(c)
        assert rec['checkpoint_sha256'] == checks[c.hash()]['checkpoint_sha256']
        immutable_checks.append(rec)
        for r,rule in settings(c):
            for split,n in workloads():
                path = result_path(c,r,rule,split,n)
                if not path.exists():
                    continue
                row = read(path)
                from evaluate import validate_result
                validate_result(path,c,r,rule,split,n,manifest['data_hash'],rec)
                assert row['learned_parameter_sha256'] == checks[c.hash()]['learned_parameter_sha256']
                assert row['learned_parameters_unchanged']
                assert_same_coefficients(row['coefficients'], weight_row(c.filter,r,rule))
                key = ('stress' if split == 'stress' else c.family,split,n)
                if key not in datasets:
                    data = NewDataset(*key)
                    _,_,labels,meta = data.rows(np.arange(len(data)))
                    datasets[key] = (labels,meta)
                labels,meta = datasets[key]
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                    assert np.array_equal(a['labels'],labels)
                    assert np.isfinite(a['logits']).all()
                    for field in ['graph_ids','query_ids','n','distance','weak_same','reverse','ell','variant']:
                        singular = {'graph_ids':'graph_id','query_ids':'query_id'}.get(field,field)
                        expected = [m[singular] if m[singular] is not None else -1 if field=='ell' else '' for m in meta]
                        assert np.array_equal(a[field],np.array(expected)), field
                    y,z = a['labels'],a['logits']
                    for name,value in direct_metrics(y,z).items():
                        assert row[name] == value or (value is not None and abs(row[name]-value)<1e-12), name
                    for cell in row['by_group']:
                        ids = np.array([m['n']==cell['n'] and m['ell']==cell['ell'] and m['variant']==cell['variant'] for m in meta])
                        for name,value in direct_metrics(y[ids],z[ids]).items():
                            assert cell[name] == value or (value is not None and abs(cell[name]-value)<1e-12)
                        if split == 'stress':
                            both = float(((z[ids]>=0)==(y[ids]==1)).reshape(-1,2).all(1).mean())
                            assert both == cell['both_queries_correct']
                        per_seed.append({'family':c.family,'model':c.model,'seed':c.seed,'rule':rule,'r':r,
                            'split':split,'n':cell['n'],'ell':cell['ell'],'variant':cell['variant'],
                            'balanced_accuracy':cell['balanced_accuracy'],'both_queries_correct':cell.get('both_queries_correct'),
                            'count':cell['count'],'mean_index':row['coefficients']['mean_index'],
                            'parameter':row['coefficients']['parameter']})
                    prediction_count += len(y)
                paths.append({'path':str(path.relative_to(OUT)),'metrics_sha256':file_hash(path),
                    'predictions_sha256':row['predictions_sha256']})
    # Reject unexpected evaluation artifacts; never silently omit extra runs.
    actual_paths = {str(p.relative_to(OUT)) for p in (OUT/'evaluations').glob('**/*.json')}
    assert actual_paths == {row['path'] for row in paths}
    complete = len(paths) == expected_count
    if complete:
        assert prediction_count == PROTOCOL['evaluation']['expected_prediction_entries']
        assert read(OUT/'completion.json')['status'] == 'completed'
    fields = ['family','model','rule','r','split','n','ell','variant']
    groups = defaultdict(list)
    index = {}
    for row in per_seed:
        groups[tuple(row[k] for k in fields)].append(row)
        key = tuple(row[k] for k in ['family','model','seed','rule','r','split','n','ell','variant'])
        assert key not in index
        index[key] = row
    aggregates = []
    for key,rows in groups.items():
        assert len({x['seed'] for x in rows}) == len(rows)
        out = dict(zip(fields,key))
        out['seed_count'] = len(rows)
        if complete:
            assert len(rows) == 5
        for metric in ['balanced_accuracy','both_queries_correct']:
            values = [row[metric] for row in rows if row[metric] is not None]
            out[metric+'_mean'] = float(np.mean(values)) if values else None
            out[metric+'_sd'] = float(np.std(values,ddof=1)) if len(values)>1 else None
        aggregates.append(out)
    contrasts = []
    # Paired model-seed uncertainty is not uncertainty over new test graphs.
    draws = np.array(list(itertools.product(range(5),repeat=5)))
    for key,rows in groups.items():
        f,m,rule,r,split,n,ell,variant = key
        if rule != 'matched' or len(rows) != 5:
            continue
        for label,base_model,base_rule,base_r in [('matched_minus_original',m,'original',r),
                ('matched_minus_own_r4',m,'original',4),('matched_minus_binomial', 'F4-bin','original',r)]:
            diffs = []
            for s in SEEDS:
                x = index.get((f,m,s,rule,r,split,n,ell,variant))
                b = index.get((f,base_model,s,base_rule,base_r,split,n,ell,variant))
                if x is None or b is None or x['balanced_accuracy'] is None or b['balanced_accuracy'] is None:
                    break
                diffs.append(100*(x['balanced_accuracy']-b['balanced_accuracy']))
            if len(diffs) != 5:
                continue
            d = np.array(diffs)
            interval = np.quantile(d[draws].mean(1),[.025,.975])
            contrasts.append({**dict(zip(fields,key)),'contrast':label,'seed_count':5,
                'difference_pp':float(d.mean()),'paired_seed_sd_pp':float(d.std(ddof=1)),
                'lower95_pp':float(interval[0]),'upper95_pp':float(interval[1]),
                'differences_by_seed':json.dumps(dict(zip(SEEDS,diffs)))})
    write_csv(OUT/'tables/per_seed.csv',per_seed)
    write_csv(OUT/'tables/aggregate.csv',aggregates)
    write_csv(OUT/'tables/paired_differences.csv',contrasts)
    audit = {'status':'passed','series_complete':complete,'evaluations':len(paths),'expected_evaluations':expected_count,
        'prediction_entries_checked':prediction_count,'original_checkpoints_unchanged':len(immutable_checks),
        'labels_ids_and_metadata_verified':True,'confusion_and_BA_independently_recomputed':True,
        'stress_paired_success_verified':True,'learned_parameter_hashes_verified':True,
        'coefficient_rules_verified':True,'protocol_hash':digest(PROTOCOL),'data_hash':manifest['data_hash'],
        'extension_source_hash':bundle_hash(),'scheduler_job':os.environ['SLURM_JOB_ID'],'files':paths}
    correction = {'reason':'Exact float64 dictionary comparison was too strict across CPU implementations; stored float32 coefficients are identical.',
        'original_failed_job':8491517, 'diagnostic_job':8491570,
        'replacement_audit_job':os.environ['SLURM_JOB_ID'],
        'original_inference_source_unchanged':True,'predictions_unchanged':True,'new_inference':False,
        'original_report_sha256':file_hash(HERE/'report.py'),
        'corrected_report_sha256':file_hash(__file__), 'coefficient_validation':COEFFICIENT_AUDIT}
    assert COEFFICIENT_AUDIT['checked'] == len(paths)
    write_json(OUT/'audit_correction.json',correction)
    audit['coefficient_comparison'] = COEFFICIENT_AUDIT
    audit['audit_implementation_sha256'] = file_hash(__file__)
    write_json(OUT/'audit.json',audit)
    summary = {tuple(row[k] for k in fields):row for row in aggregates}
    def cell(f,m,rule,r,split,n,ell=None,variant=None):
        if r == 4:
            rule = 'original'
        row = summary.get((f,m,rule,r,split,n,ell,variant))
        if row is None:
            return '—'
        a,b = row['balanced_accuracy_mean'],row['balanced_accuracy_sd']
        return f'{100*a:.2f} ± {100*b:.2f}' if b is not None else f'{100*a:.2f} (seed count={row["seed_count"]})'
    lines = ['# Reachability: перевзвешивание exponential и geometric без дообучения', '',
        f'Статус: {"завершено" if complete else "частичный результат"}; {len(paths)}/{expected_count} оценок. Новых обучений: 0.', '',
        'Это новая исследовательская итерация после результатов v1. Правило коэффициентов, сетка r, seeds и новые данные зафиксированы до первых предсказаний этой итерации.', '',
        'Использованы 30 готовых best checkpoints: F4-bin, F4-exp, F4-geo × ER/DAG × seeds 11,23,37,53,71. '
        'Все обучаемые параметры сохранены; меняются только фиксированные коэффициенты фильтра. Inner MLP в эту проверку не входит.', '',
        'Для matched-режима lambda(r) и rho(r) определяются без использования labels или scores из равенства '
        'sum(k*c_k)=r*2^(r-1)/(2^r-1). При r=4 режимы совпадают точно. Original сохраняет параметры lambda/rho при обучении r=4; binomial служит общим контролем.', '',
        'Новые тестовые данные: 500 графов на семейство и n=32,64,128, по 8 запросов; '
        '50 шаблонов на каждую из 30 stress-ячеек с двумя запросами. Всего 4500 графов/шаблонов, 27000 уникальных запросов. '
        'Новый seed генератора; точные labeled adjacency исключены относительно прежних данных и внутри новой выборки. '
        'Изоморфная непересекаемость не заявляется; stress сохраняет прежний тип конструкции, поэтому это новые случайные реализации, а не новый класс задач.', '',
        'Во всех таблицах balanced accuracy, %, среднее ± sample SD пяти seeds. Все r сохраняются, лучший r по test не выбирается. '
        'На сбалансированном stress BA совпадает с accuracy отдельных запросов; доля полностью правильно решённых пар сохранена отдельно в CSV.', '',
        '## Случайные графы n=128', '', '| Обучение | Модель | r | Original | Matched |', '|---|---|---:|---:|---:|']
    for f in FAMILIES:
        for m in MODELS:
            for r in HOPS:
                lines.append(f'| {f} | {m} | {r} | {cell(f,m,"original",r,"far",128)} | {cell(f,m,"matched",r,"far",128) if m != "F4-bin" else "—"} |')
    for f in FAMILIES:
        for variant in ['dag','cyclic']:
            lines += ['',f'## Длинные пути: обучение {f}, stress {variant}, n=128','',
                '| Модель | Режим | r | ℓ=16 | ℓ=32 | ℓ=48 |','|---|---|---:|---:|---:|---:|']
            for m in MODELS:
                for r in HOPS:
                    for rule in (['original','matched'] if r>4 and m!='F4-bin' else ['original']):
                        vals = [cell(f,m,rule,r,'stress',128,ell,variant) for ell in [16,32,48]]
                        lines.append(f'| {m} | {rule} | {r} | '+ ' | '.join(vals)+' |')
    lines += ['', '## Проверки и границы вывода','',
        f'Аудит перепроверил {prediction_count:,} сохранённых предсказаний: labels, IDs, metadata, confusion matrix, BA и правильность обеих сторон stress-пар. '
        'Проверены hashes всех 30 исходных checkpoints и неизменность обученных параметров. '
        'CPU preflight проверил нормировку коэффициентов, согласование среднего, независимую скалярную формулу и точное совпадение logits при r=4.', '',
        'Полные per-seed таблицы, результаты для n=32/64/128 и всех stress-ячеек, а также парные разности лежат в tables/. '
        '95% интервалы в paired_differences.csv построены полным перебором 3125 bootstrap-выборок пяти paired seeds. '
        'Это условная неопределённость по обучениям при общей тестовой выборке; поправка на множественные сравнения не применяется, выводы exploratory.', '',
        'Равенство среднего индекса не означает равенства дисперсии или формы весов и не равно реальному расстоянию переноса сигнала. '
        'Четыре внешних блока, attention mask, learned weights и порог сохранены. При r шагов выполняются 4r transport matmuls. '
        'Forward seconds и память сохранены для каждого workload; это наблюдаемое время, без отдельного latency-бенчмарка.', '',
        'Все прежние результаты v1 сохранены. Новая проверка ограничена 7200 выделенными GPU-секундами; ресурсный учёт scheduler хранится отдельно. '
        'Отсутствие эффекта или ухудшение сохраняются наравне с улучшением.']
    lines += ['', 'Первоначальный CPU-аудит остановился на слишком строгом сравнении float64-метаданных коэффициентов. '
        'Диагностика показала максимальное расхождение 3,55e-15 и точное совпадение всех используемых float32-коэффициентов. '
        'Отдельная версия аудита сохраняет точное сравнение float32 и допускает до 1e-14 абсолютной погрешности только для float64-метаданных. '
        'Исходный код запуска, checkpoints, тестовые данные и предсказания не изменены. Полная повторная проверка зафиксирована в audit.json и audit_correction.json.']
    (OUT/'REPORT_REWEIGHT_RU.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='files'}),flush=True)


if __name__ == '__main__':
    main()
