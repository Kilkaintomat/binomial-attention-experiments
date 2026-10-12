"""Freeze the amendment, test its math, then create untouched random test draws."""
import time
import hashlib
import numpy as np
import torch
from common import *
from transport_reasoning.reachability.config import allocation, data_root
from transport_reasoning.reachability.data import graph, sample_pairs, metadata, stress_graph, closure, bfs, raw_inputs, torch_inputs


def adjacency_key(B):
    return hashlib.sha256(np.packbits(B).tobytes()).hexdigest().encode()


def preflight():
    rows = []
    for kind in ['binomial', 'exponential', 'geometric']:
        for r in HOPS:
            for rule in (['original', 'matched'] if kind != 'binomial' else ['original']):
                row = weight_row(kind, r, rule)
                w = np.array(row['weights'])
                assert np.isfinite(w).all() and (w > 0).all() and abs(w.sum()-1) < 1e-12
                if rule == 'matched' or kind == 'binomial':
                    assert abs(row['mean_index']-r*2**(r-1)/(2**r-1)) < 1e-11
                if r == 4 and rule == 'matched':
                    assert row['weights'] == weight_row(kind, r, 'original')['weights']
                # Independent scalar recurrence checks the log-space implementation.
                scalar = []
                for k in range(1, r+1):
                    import math
                    scalar.append(math.comb(r,k) if kind == 'binomial' else row['parameter']**k/math.factorial(k) if kind == 'exponential' else row['parameter']**(k-1))
                assert np.allclose(w, np.array(scalar)/sum(scalar), rtol=1e-12, atol=1e-15)
                rows.append(row)
    checked = []
    B = np.zeros((6,6), dtype=bool)
    B[np.arange(5), np.arange(1,6)] = True
    inputs = torch_inputs(raw_inputs([B,B], [(0,5),(5,0)]), 'cpu')
    for c in configs():
        model, record = load_checkpoint(c, 'cpu')
        before = learned_hash(model)
        with torch.inference_mode():
            z0 = model(inputs).clone()
            apply_rule(model,4,'matched' if c.model != 'F4-bin' else 'original')
            assert torch.equal(z0, model(inputs)), 'r=4 changed checkpoint outputs'
            for r, rule in settings(c):
                row = apply_rule(model, r, rule)
                assert torch.isfinite(model(inputs)).all()
                assert model.operations()['transport_matmuls'] == 4*r
                for block in model.blocks:
                    assert np.allclose(block.fixed_coeff.cpu().numpy(), row['weights'], rtol=1e-6, atol=1e-10)
        assert before == learned_hash(model)
        checked.append({**record, 'learned_parameter_sha256': before})
    write_json(OUT/'coefficients.json', rows)
    write_json(OUT/'checkpoints.json', checked)
    return {'status': 'passed', 'coefficient_cases':len(rows), 'frozen_checkpoints':len(checked),
        'r4_exact_logit_equivalence':True, 'learned_parameter_immutability':True,
        'independent_scalar_weight_check':True, 'finite_outputs_all_settings':True}


def old_hashes(n):
    seen = set()
    # Exclude both families and old stress at the same size; no old file is modified.
    for p in data_root().glob('*/test_far/n'+str(n)+'/adjacency_sha.npy'):
        seen.update(np.load(p, allow_pickle=False).tolist())
    for p in data_root().glob('stress/*/n'+str(n)+'_ell*/B.npy'):
        seen.update(adjacency_key(B) for B in np.load(p, mmap_mode='r'))
    if n in [16,17]:
        for p in data_root().glob('*/*/n'+str(n)+'/adjacency_sha.npy'):
            seen.update(np.load(p, allow_pickle=False).tolist())
    return seen


def generate_group(family, n, seen, variant=None, ell=None):
    count, q = (50,2) if family == 'stress' else (500,8)
    path = OUT/'data'/('stress/'+variant+'/n'+str(n)+'_ell'+str(ell) if family == 'stress' else family+'/n'+str(n))
    if (path/'manifest.json').exists():
        m = read(path/'manifest.json')
        for name, sha in m['files'].items():
            assert file_hash(path/name) == sha
        seen.update(np.load(path/'adjacency_sha.npy').tolist())
        return m
    values = {'B': np.empty((count,n,n),bool), 'pairs': np.empty((count,q,2),np.int16),
        'distance': np.empty((count,q),np.int16), 'reverse': np.empty((count,q),bool),
        'weak_same': np.empty((count,q),bool), 'indegree': np.empty((count,n),np.int16),
        'outdegree': np.empty((count,n),np.int16), 'adjacency_sha':np.empty(count,dtype='S64')}
    rejected, independent_checks = 0, 0
    family_id = ['er','dag','stress'].index(family)
    variant_id = 0 if variant != 'cyclic' else 1
    for i in range(count):
        attempt = 0
        while True:
            def rng(stream):
                return np.random.default_rng(np.random.SeedSequence([MASTER,family_id,n,ell or 0,variant_id,i,attempt,stream]))
            if family == 'stress':
                B, pairs, _ = stress_graph(n,ell,variant,rng(0))
            else:
                B, _ = graph(family,n,rng(0),rng(2))
                pairs = sample_pairs(n,rng(1))
            key = adjacency_key(B)
            if key not in seen:
                seen.add(key)
                break
            attempt += 1
            rejected += 1
            if attempt > 10000:
                raise RuntimeError('Exact novelty rejection exhausted')
        meta = metadata(B,pairs)
        assert not np.diag(B).any()
        if family == 'stress':
            assert meta['distance'].tolist() == [ell,-1] and meta['weak_same'].all()
            s0, s1 = pairs[:,0]
            assert meta['indegree'][s0] == meta['indegree'][s1] and meta['outdegree'][s0] == meta['outdegree'][s1]
        if i < 32:
            C = closure(B)
            if family == 'dag' or variant == 'dag':
                assert not np.diag(C).any()
            for (s,t), d in zip(pairs, meta['distance']):
                assert bool(C[s,t]) == bool(d >= 0) and bfs(B,int(s))[t] == d
                independent_checks += 1
        values['B'][i], values['pairs'][i], values['adjacency_sha'][i] = B, pairs, key
        for name in ['distance','reverse','weak_same','indegree','outdegree']:
            values[name][i] = meta[name]
    path.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, value in values.items():
        dest = path/(key+'.npy')
        with dest.with_suffix('.tmp').open('wb') as f:
            np.save(f,value,allow_pickle=False)
        dest.with_suffix('.tmp').replace(dest)
        files[dest.name] = file_hash(dest)
    m = {'family':family,'n':n,'variant':variant,'ell':ell,'graphs':count,'queries':count*q,
        'files':files,'rejected_exact_duplicates':rejected,'independent_oracle_queries':independent_checks,
        'positive_count':int((values['distance']>=0).sum()),'extension_source_hash':bundle_hash()}
    write_json(path/'manifest.json',m)
    print({'generated':str(path.relative_to(OUT)), 'graphs':count},flush=True)
    return m


def main():
    allocation()
    torch.set_num_threads(1)
    freeze_protocol()
    started = time.perf_counter()
    checks = preflight()
    seen = {n:old_hashes(n) for n in [16,32,64,128]}
    groups = []
    for family in FAMILIES:
        for n in [32,64,128]:
            groups.append(generate_group(family,n,seen[n]))
    for variant in ['dag','cyclic']:
        for n,ell in CELLS:
            groups.append(generate_group('stress',n,seen[n],variant,ell))
    manifest = {'groups':groups,'master_seed':MASTER,'protocol_hash':digest(PROTOCOL),
        'graphs':sum(g['graphs'] for g in groups),'queries':sum(g['queries'] for g in groups),
        'independent_oracle_queries':sum(g['independent_oracle_queries'] for g in groups),
        'extension_source_hash':bundle_hash(),'new_labeled_adjacencies_disjoint':True,
        'isomorphic_disjointness_claimed':False}
    assert manifest['graphs'] == 4500 and manifest['queries'] == 27000
    manifest['data_hash'] = digest(manifest)
    write_json(OUT/'data/manifest.json',manifest)
    checks.update(data_hash=manifest['data_hash'],elapsed_seconds=time.perf_counter()-started,
        scheduler_job=os.environ['SLURM_JOB_ID'],all_data_frozen_before_new_predictions=True)
    write_json(OUT/'preflight.json',checks)
    print(json.dumps(checks),flush=True)


if __name__ == '__main__':
    main()
