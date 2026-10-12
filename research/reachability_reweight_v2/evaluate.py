"""Resumable GPU inference. Old checkpoints/source/data are mounted read-only."""
import argparse
import time
from common import *
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.training import predict, setup
from transport_reasoning.reachability.metrics import binary_metrics, slices


def validate_result(path, c, r, rule, split, n, data_hash, record):
    row = read(path)
    assert row['config_hash'] == c.hash() and row['r'] == r and row['rule'] == rule
    assert row['split'] == split and row['n'] == n and row['data_hash'] == data_hash
    assert row['checkpoint_sha256'] == record['checkpoint_sha256']
    assert row['protocol_hash'] == digest(PROTOCOL) and row['extension_source_hash'] == bundle_hash()
    assert row['predictions_sha256'] == file_hash(path.with_suffix('.npz'))


def evaluate(seconds):
    allocation()
    frozen = freeze_protocol()
    assert read(OUT/'preflight.json')['status'] == 'passed'
    manifest = read(OUT/'data/manifest.json')
    assert manifest['data_hash'] == read(OUT/'preflight.json')['data_hash']
    for p in (OUT/'data').glob('**/manifest.json'):
        if p == OUT/'data/manifest.json':
            continue
        for filename, checksum in read(p)['files'].items():
            assert file_hash(p.parent/filename) == checksum
    if (OUT/'failure.json').exists():
        raise RuntimeError('Previous failure requires review before resuming')
    started = time.perf_counter()
    deadline = started+seconds
    completed = 0
    total = PROTOCOL['evaluation']['expected_evaluations']
    cache = {}
    all_configs = configs()
    for c in all_configs:
        device = setup(c)
        model, record = load_checkpoint(c,device)
        initial_hash = learned_hash(model)
        for r, rule in settings(c):
            weight_info = apply_rule(model,r,rule)
            assert learned_hash(model) == initial_hash
            assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
            for split,n in workloads():
                path = result_path(c,r,rule,split,n)
                if path.exists():
                    validate_result(path,c,r,rule,split,n,manifest['data_hash'],record)
                    completed += 1
                    continue
                if time.perf_counter() > deadline-150:
                    write_json(OUT/'progress.json', {'status':'paused','completed':completed,'expected':total,
                        'scheduler_job':os.environ['SLURM_JOB_ID'],'slot_elapsed_seconds':time.perf_counter()-started})
                    return
                key = ('stress' if split == 'stress' else c.family,split,n)
                if key not in cache:
                    cache[key] = NewDataset(*key)
                data = cache[key]
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    metric,y,z,meta = predict(model,data,device,64,path.with_suffix('.npz'))
                assert learned_hash(model) == initial_hash
                by_group = []
                groups = {}
                for i,row in enumerate(meta):
                    groups.setdefault((row['n'],row['ell'],row['variant']),[]).append(i)
                for (size,ell,variant),ids in groups.items():
                    ids = np.array(ids)
                    entry = {'n':size,'ell':ell,'variant':variant,**binary_metrics(y[ids],z[ids])}
                    if split == 'stress':
                        assert len(ids)%2 == 0
                        assert all(meta[int(a)]['graph_id'] == meta[int(b)]['graph_id'] for a,b in ids.reshape(-1,2))
                        entry['both_queries_correct'] = float(((z[ids] >= 0) == (y[ids] == 1)).reshape(-1,2).all(1).mean())
                    by_group.append(entry)
                metric.update(family=c.family,model=c.model,seed=c.seed,config_hash=c.hash(),r=r,rule=rule,
                    split=split,n=n,coefficients=weight_info,operations=model.operations(),
                    checkpoint_sha256=record['checkpoint_sha256'],checkpoint_step=record['step'],
                    learned_parameter_sha256=initial_hash,learned_parameters_unchanged=True,
                    protocol_hash=frozen['protocol_hash'],extension_source_hash=frozen['extension_source_hash'],
                    data_hash=manifest['data_hash'],base_source_hash=BASE_SOURCE,base_training_data_hash=BASE_DATA,
                    predictions_sha256=file_hash(path.with_suffix('.npz')),by_group=by_group,
                    path_length_slices=slices(y,z,meta),device_name=torch.cuda.get_device_name(),
                    peak_memory_bytes=torch.cuda.max_memory_allocated(),scheduler_job=os.environ['SLURM_JOB_ID'])
                write_json(path,metric)
                completed += 1
                status = {'status':'running','completed':completed,'expected':total,'family':c.family,'model':c.model,
                    'seed':c.seed,'r':r,'rule':rule,'split':split,'n':n,'scheduler_job':os.environ['SLURM_JOB_ID']}
                write_json(OUT/'progress.json',status)
                print(json.dumps(status),flush=True)
        # Validate the original file again after using this checkpoint.
        assert checkpoint_record(c) == record
        del model
        torch.cuda.empty_cache()
    assert completed == total
    result = {'status':'completed','completed':completed,'expected':total,'new_trainings':0,
        'data_hash':manifest['data_hash'],'protocol_hash':frozen['protocol_hash'],
        'extension_source_hash':frozen['extension_source_hash'],'scheduler_job':os.environ['SLURM_JOB_ID']}
    write_json(OUT/'completion.json',result)
    write_json(OUT/'progress.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds',type=int,default=1700)
    args = parser.parse_args()
    try:
        evaluate(args.seconds)
    except Exception as exc:
        write_json(OUT/'failure.json', {'exception':type(exc).__name__,'message':str(exc),
            'scheduler_job':os.environ.get('SLURM_JOB_ID')})
        raise
