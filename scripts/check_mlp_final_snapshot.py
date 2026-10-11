"""Independently check the portable MLP tables, pairing, bootstrap and evidence."""
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from statistics import mean, stdev

import numpy as np

from build_public_assets import ROOT
from build_mlp_final_assets import SEEDS, TASKS, METHODS, folder


def read(path): return json.loads(path.read_text())
def rows(name): return list(csv.DictReader((ROOT/'results/tables'/name).open()))
def close(a, b): assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Verification requires a CPU allocation'
    audit = read(ROOT/'results/audit/mlp_final_publication_audit.json')
    assert audit['status'] == 'passed'
    assert audit['saved_test_predictions_revalidated'] == 190000 and audit['checkpoint_replay_examples'] == 320
    for item in audit['exported_files']:
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest() == item['sha256'], item['file']
    source = hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*.py')):
        source.update(path.relative_to(ROOT).as_posix().encode()); source.update(path.read_bytes())
    assert source.hexdigest() == audit['source_sha256']
    data = rows('mlp_final_per_seed.csv')
    assert len(data) == 40
    lookup = {(r['task'], r['method'], int(r['seed']), float(r['inner_beta'])): r for r in data}
    assert set(lookup) == {(t,m,s,b) for t in TASKS for m in METHODS for s in SEEDS for b in [0., .1]}
    for key, row in lookup.items():
        directory = folder(row, row['config_hash'], mlp=key[3] > 0)
        result = read(directory/'result.json'); config = read(directory/'config.json'); evaluation = read(directory/'evaluation_test.json')
        digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
        assert digest == row['config_hash'] == result['config_hash']
        assert config == result['config'] and config['inner_beta'] == key[3]
        assert int(row['updates']) == result['step'] == 30000 and result['status'] == 'completed'
        assert row['data_hash'] == result['data_hash'] == evaluation['data_hash']
        assert row['source_sha256'] == result['source_sha256'] == source.hexdigest()
        for field in ['solved_rate', 'exact_match', 'loss', 'inference_seconds_per_example']:
            close(row[field], evaluation[field])
        for field in ['training_seconds', 'elapsed_seconds', 'parameter_count']:
            close(row[field], result[field])
        curve = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in curve] == [1] + list(range(250, 30001, 250))
        best = max(curve, key=lambda x: (x['solved_rate'], -x['loss'], -x['step']))
        assert best['step'] == evaluation['checkpoint_step'] == int(row['checkpoint_step'])
    for item in audit['runs']:
        c = read(folder(item,item['config_hash'],mlp=True)/'config.json')
        base = read(folder(item,item['baseline_config_hash'])/'config.json')
        assert c == dict(base,inner_beta=.1,inner_width=128,max_seconds=4800)
    for aggregate in rows('mlp_final_aggregates.csv'):
        selected = [lookup[aggregate['task'],aggregate['method'],s,float(aggregate['inner_beta'])] for s in SEEDS]
        for field in ['solved_rate','loss','training_seconds','elapsed_seconds','inference_seconds_per_example']:
            values = [float(r[field]) for r in selected]
            close(aggregate[field+'_mean'],mean(values));close(aggregate[field+'_sample_sd'],stdev(values))
    for pair in rows('mlp_final_pairs.csv'):
        task,method,seed=pair['task'],pair['method'],int(pair['seed'])
        on,off=lookup[task,method,seed,.1],lookup[task,method,seed,0.]
        close(pair['difference_pp'],100*(float(on['solved_rate'])-float(off['solved_rate'])))
        close(pair['training_seconds_ratio_on_off'],float(on['training_seconds'])/float(off['training_seconds']))
    rng=np.random.default_rng(20261011)
    contrasts=rows('mlp_final_contrasts.csv');assert len(contrasts)==6
    for contrast in contrasts:
        task=contrast['task'];name=contrast['comparison']
        if name=='binomial_minus_exponential_MLP':
            values=[100*(float(lookup[task,'binomial',s,.1]['solved_rate'])-float(lookup[task,'exponential_truncated',s,.1]['solved_rate'])) for s in SEEDS]
        else:
            method=name.removesuffix('_MLP_minus_no_MLP')
            values=[100*(float(lookup[task,method,s,.1]['solved_rate'])-float(lookup[task,method,s,0.]['solved_rate'])) for s in SEEDS]
        assert values==json.loads(contrast['paired_differences_pp'])
        close(contrast['mean_difference_pp'],mean(values));close(contrast['sample_sd_pp'],stdev(values))
        interval=np.quantile(rng.choice(values,(20000,5),replace=True).mean(1),[.025,.975])
        close(contrast['lower95_pp'],interval[0]);close(contrast['upper95_pp'],interval[1])
    assert len(rows('mlp_final_thresholds.csv'))==60
    curves=rows('mlp_final_validation_curves.csv')
    assert len(curves)==40*121
    assert all(v==121 for v in Counter((r['task'],r['method'],r['seed'],r['inner_beta']) for r in curves).values())
    links=0
    for name in ['REPORT_MLP_FINAL_RU.md','README.md','REPORT_RU.md']:
        path=ROOT/name;body=re.sub(r'```[\s\S]*?```','',path.read_text());body=re.sub(r'`[^`\n]+`','',body)
        for target in re.findall(r'\]\(([^)]+)\)',body):
            if target.startswith(('http://','https://','#')):continue
            assert (path.parent/target.split('#')[0]).exists(), (name,target)
            links+=1
    result={'status':'passed','new_final_records':20,'historical_paired_controls':20,'record_hashes_checked':len(audit['exported_files']),
            'validation_points_checked':len(curves),'paired_contrasts_recomputed':6,'links_checked':links,
            'scope':'Portable files, hashes, full final coverage, paired scalar metrics, checkpoint selection, aggregates and seed bootstrap. Private audit is separate.'}
    (ROOT/'results/metadata/mlp_final_publication_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
