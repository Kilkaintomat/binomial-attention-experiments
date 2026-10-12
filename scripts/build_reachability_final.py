"""Build complete graph-test tables without treating graph sizes as seeds.

Use a CPU allocation. The original run records remain immutable. Macro scores
are computed within each training seed before mean/sample SD across five seeds.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from build_reachability_update import SOURCE, DATA, MODELS, SEEDS, digest, read, validate_metric

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/reachability_final_20261012'
METRICS = ['balanced_accuracy', 'accuracy', 'f1', 'auroc', 'bce']
KEYS = ['family', 'model', 'split', 'n', 'ell', 'variant', 'setting', 'value']


def table(name, rows):
    with (BASE/'tables'/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys=KEYS, metrics=METRICS):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    out = []
    for key, group in groups.items():
        assert len(group) == 5 and {r['seed'] for r in group} == set(SEEDS), key
        r = dict(zip(keys, key))
        r.update(seed_count=5, seeds='11,23,37,53,71')
        for metric in metrics:
            values = [x[metric] for x in group]
            r[metric+'_mean'] = mean(values)
            r[metric+'_sample_sd'] = stdev(values)
        out.append(r)
    return out


def add_macro(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['family'], r['model'], r['seed'], r['split'], r['setting'], r['value']].append(r)
    macros = []
    for (family, model, seed, split, setting, value), group in groups.items():
        r = dict(family=family, model=model, seed=seed, split=split, n='macro', ell=None, variant=None,
                 setting=setting, value=value, cells=len(group), aggregation='equal size/stress-cell mean within seed')
        for metric in METRICS:
            r[metric] = mean(x[metric] for x in group)
        macros.append(r)
    return rows + macros


def workloads(c):
    fixed = [('fixed','test_ID',None,None,None), ('fixed','test_near',None,None,None)]
    fixed += [('fixed','test_far',n,None,None) for n in [32,64,128]] + [('fixed','stress',None,None,None)]
    settings = [('hops',r) for r in [4,8,16,32]] if c['filter'] in ['binomial','exponential','geometric'] else [('outer',t) for t in [32,64,128]] if c['model']=='R32-one' else []
    return set(fixed + [('extra',split,n,k,v) for k,v in settings for split,n in [('test_far',32),('test_far',64),('test_far',128),('stress',None)]])


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'CPU allocation required'
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    for name in ['tables','figures']:
        (BASE/name).mkdir(exist_ok=True)
    snapshot = read(BASE/'snapshot.json')
    state = snapshot['series']['reachability']
    assert state['status_at_capture'] == 'completed' and not state['active_gpu_allocation']
    assert all(j['accounted'] and j['scheduler_state']=='COMPLETED' and j['exit_code']=='0:0' for j in state['allocated_slots'])
    for item in state['files']:
        data = (BASE/'reachability'/item['file']).read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    provenance = read(BASE/'source_provenance.json')
    if args.source:
        for item in provenance['files']:
            assert hashlib.sha256((args.source/item['file']).read_bytes()).hexdigest()==item['sha256']
    completion = read(BASE/'reachability/metadata/completion.json')
    assert completion['status']=='completed' and completion['new_final_trainings']==80
    assert read(BASE/'reachability/metadata/postprocess_completed.json')['baselines_and_report']
    frozen = read(BASE/'reachability/metadata/frozen_checkpoints.json')
    assert len(frozen)==80
    trains, pilots, fixed, extra, pooled = [], [], [], [], []
    curves, evals = 0, collections.Counter()
    for path in sorted((BASE/'reachability/records').rglob('result.json')):
        result = read(path)
        c = read(path.parent/'config.json')
        assert result['config']==c and digest(c)[:16]==path.parent.name
        assert result['source_hash']==SOURCE and result['data_hash']==DATA and result['status']=='completed'
        assert result['step']==c['completion_steps']==(30000 if c['stage']=='final' else 3000)
        points = [json.loads(line) for line in (path.parent/'validation.jsonl').read_text().splitlines()]
        interval = 1000 if c['stage']=='final' else 500
        assert [m['step'] for m in points]==list(range(interval,result['step']+1,interval))
        for m in points:
            validate_metric(m)
            assert m['count']==(40000 if m['full_validation'] else 2048)
        curves += len(points)
        assert points[-1]==result['final_validation']
        best = max(points,key=lambda m:(m['balanced_accuracy'],-m['bce']))
        assert best['step']==result['best_step']
        base = dict(family=c['family'],model=c['model'],seed=c['seed'])
        train = dict(**base, stage=c['stage'], steps=result['step'], lr=c['lr'], best_step=result['best_step'],
                     parameters=result['operations']['parameters'], **result['times'],
                     validation_ba=points[-1]['balanced_accuracy'], validation_bce=points[-1]['bce'], config_hash=path.parent.name)
        if c['stage']!='final':
            pilots.append(train)
            continue
        trains.append(train)
        assert frozen[path.parent.name]['step']==best['step']
        observed = set()
        for e in sorted((path.parent/'evaluation').glob('*.json')):
            v = read(e)
            assert v['source_hash']==SOURCE and v['data_hash']==DATA and v['checkpoint_step']==best['step']
            key = tuple(v[k] for k in ['regime','split','n','setting','value'])
            assert key not in observed
            observed.add(key)
            evals[v['regime']] += 1
            pooled.append(dict(**base, **{k:v[k] for k in ['regime','split','n','setting','value','count','positive_fraction',*METRICS]}))
            for row in v['by_size_or_stress_cell']:
                validate_metric(row)
                out = dict(**base, split=v['split'],setting=v['setting'],value=v['value'], **row)
                (fixed if v['regime']=='fixed' else extra).append(out)
        assert observed==workloads(c), (base, workloads(c)-observed)
    assert len(trains)==80 and len(pilots)==48 and evals=={'fixed':480,'extra':920}
    assert {(r['family'],r['model'],r['seed']) for r in trains}=={(f,m,s) for f in ['er','dag'] for m in MODELS for s in SEEDS}
    for family in ['er','dag']:
        for model in MODELS:
            candidates = [r for r in pilots if r['family']==family and r['model']==model and r['stage']=='pilot']
            assert len(candidates)==2
            chosen = max(candidates,key=lambda r:(r['validation_ba'],-r['validation_bce'],-r['lr']))['lr']
            assert all(r['lr']==chosen for r in trains+pilots if r['family']==family and r['model']==model and r['stage']!='pilot')
    fixed = add_macro(fixed)
    extra = add_macro(extra)
    fixed_agg, extra_agg = aggregate(fixed), aggregate(extra)
    table('fixed_test_per_seed.csv',fixed)
    table('fixed_test_aggregate.csv',fixed_agg)
    table('extra_subset_per_seed.csv',extra)
    table('extra_subset_aggregate.csv',extra_agg)
    table('evaluation_pooled_metrics.csv',pooled)
    table('training_per_seed.csv',trains)
    table('training_aggregate.csv',aggregate(trains,['family','model'],['training_seconds','elapsed_seconds','validation_ba']))
    table('pilots_and_confirmation.csv',pilots)
    baselines = read(BASE/'reachability/baselines/results.json')
    assert baselines['source_hash']==SOURCE and len(baselines['rows'])==72
    table('baselines.csv',baselines['rows'])
    pairs = [('F4-bin','F4-one'),('F4-exp','F4-bin'),('F4-geo','F4-bin'),('F4-bin-inner','F4-bin'),('F4-exp-inner','F4-exp'),('F4-bin-inner','F4-one-inner'),('R32-one','F4-bin')]
    comparisons = []
    draw_ids = np.array(list(itertools.product(range(5),repeat=5)))
    for family in ['er','dag']:
        for split,n in [('test_ID','macro'),('test_near','macro'),('test_far',32),('test_far',64),('test_far',128),('test_far','macro'),('stress','macro')]:
            for left,right in pairs:
                by = {m:{r['seed']:r['balanced_accuracy'] for r in fixed if r['family']==family and r['model']==m and r['split']==split and r['n']==n} for m in [left,right]}
                d = np.array([100*(by[left][s]-by[right][s]) for s in SEEDS])
                ci = np.quantile(d[draw_ids].mean(1),[.025,.975])
                comparisons.append(dict(family=family,split=split,n=n,comparison=left+' minus '+right,paired_seeds=5,
                                        mean_difference_pp=float(d.mean()),lower95_pp=float(ci[0]),upper95_pp=float(ci[1]),
                                        method='exact enumeration of 3125 empirical paired-seed bootstrap samples; exploratory; no multiplicity adjustment'))
    table('paired_test_differences.csv',comparisons)
    # Natural per-size summaries agree with the original runtime report.
    runtime = list(csv.DictReader((BASE/'reachability/runtime_report/aggregate_results.csv').open()))
    runtime_lookup = {(r['family'],r['model'],r['split'],r['n'],r.get('ell') or '',r.get('variant') or ''):r for r in runtime}
    for r in fixed_agg:
        if r['n']=='macro':
            continue
        old = runtime_lookup[r['family'],r['model'],r['split'],str(r['n']),str(r['ell'] or ''),r['variant'] or '']
        assert np.isclose(float(old['balanced_accuracy_mean']),r['balanced_accuracy_mean'])
    correction = dict(issue='Runtime far/macro combined three sizes and five seeds as fifteen observations.',
                      affected_original='reachability/runtime_report/aggregate_results.csv: test_far, n=macro; seed_count=15 and SD over size-seed cells',
                      correction='Average sizes 32/64/128 within each seed, then compute mean and sample SD across five independent training seeds.',
                      corrected='tables/fixed_test_aggregate.csv: test_far, n=macro; seed_count=5',
                      per_size_scores_changed=False, saved_predictions_changed=False, original_records_preserved=True)
    (BASE/'aggregation_correction.json').write_text(json.dumps(correction,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    for ax,family in zip(axes,['er','dag']):
        for model in MODELS:
            rows = sorted([r for r in fixed_agg if r['family']==family and r['model']==model and r['split']!='stress' and isinstance(r['n'],int)],key=lambda r:r['n'])
            ax.plot([r['n'] for r in rows],[100*r['balanced_accuracy_mean'] for r in rows],marker='.',label=model)
        ax.axhline(50,color='gray',ls=':',lw=1)
        ax.set(title=family.upper(),xlabel='Vertices (training: 16/17)',ylabel='Test balanced accuracy (%)',ylim=(45,101),xscale='log',xticks=[16,32,64,128],xticklabels=['16','32','64','128'])
        ax.minorticks_off()
    axes[1].legend(fontsize=8,loc='lower left')
    fig.suptitle('Directed reachability: full fixed-compute tests; mean of five seeds')
    fig.tight_layout();fig.savefig(BASE/'figures/fixed_test_vs_size.png',dpi=170);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    for ax,variant in zip(axes,['dag','cyclic']):
        for model in ['F4-one','F4-bin','F4-exp','F4-geo','F4-bin-inner','R32-one']:
            rows=sorted([r for r in fixed_agg if r['family']=='er' and r['model']==model and r['split']=='stress' and r['n']==128 and r['variant']==variant],key=lambda r:r['ell'])
            ax.plot([r['ell'] for r in rows],[100*r['balanced_accuracy_mean'] for r in rows],marker='o',label=model)
        ax.set(title='ER-trained, '+variant+' templates',xlabel='Positive directed distance',ylabel='Paired balanced accuracy (%)',ylim=(40,102),xticks=[2,4,8,16,32,48])
    axes[1].legend(fontsize=8)
    fig.suptitle('Stress n=128: full fixed-compute tests, five seeds')
    fig.tight_layout();fig.savefig(BASE/'figures/stress_er_n128.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    for ax,family in zip(axes,['er','dag']):
        for model in ['F4-bin','F4-exp','F4-geo','F4-bin-inner','F4-exp-inner']:
            rows=sorted([r for r in extra_agg if r['family']==family and r['model']==model and r['split']=='test_far' and r['n']==128],key=lambda r:r['value'])
            ax.plot([r['value'] for r in rows],[100*r['balanced_accuracy_mean'] for r in rows],marker='o',label=model)
        ax.set(title=family.upper(),xlabel='Hops inside each of four blocks',ylabel='Subset balanced accuracy (%)',ylim=(45,101),xticks=[4,8,16,32])
    axes[1].legend(fontsize=8)
    fig.suptitle('Extra compute at n=128: same first 500 graphs; mean of five seeds')
    fig.tight_layout();fig.savefig(BASE/'figures/extra_far128.png',dpi=170);plt.close(fig)
    audit=read(BASE/'prediction_audit.json')
    assert audit['status']=='passed' and audit['evaluations_checked']==1400
    summary=dict(status='passed',snapshot_utc=snapshot['capture_finished_utc'],completed_finals=80,completed_pilots_confirmations=48,
                 fixed_evaluations=480,extra_evaluations=920,baseline_rows=72,validation_points_checked=curves,
                 captured_file_hashes_checked=len(state['files']),source_files_checked=len(provenance['files']) if args.source else 0,
                 gpu_seconds=state['gpu_seconds_accounted'],gpu_hours=state['gpu_seconds_accounted']/3600,
                 full_admitted_grid_complete=True,full_original_grid_complete=False,omitted_model='F4-learned',
                 macro_aggregation_corrected=True,all_test_groups_have_five_seeds=True,training_modified=False)
    (BASE/'publication_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
