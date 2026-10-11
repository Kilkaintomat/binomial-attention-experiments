import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from transport_reasoning.config import Config as LegacyConfig, write_json, digest
from transport_reasoning.data import file_hash
from .data import TaskData, allowed_mask, mask_hash, diameter
from transport_reasoning.models import Encoder as LegacyEncoder
from transport_reasoning.training import source_hash
from .config import Config, LEGACY_SOURCE, FINAL_SEEDS, code_hash, from_legacy, root, make, LABELS, SUPPORTS
from .model import Encoder


def legal_grids(values):
    grids = values.reshape(-1, 9, 9)
    target = np.arange(1, 10)[None, None, :]
    rows = (np.sort(grids, axis=2) == target).all(axis=(1, 2))
    cols = (np.sort(grids.transpose(0, 2, 1), axis=2) == target).all(axis=(1, 2))
    boxes = grids.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    return rows & cols & (np.sort(boxes, axis=2) == target).all(axis=(1, 2))


def audit():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('CPU allocation required')
    torch.set_num_threads(4)
    assert source_hash() == LEGACY_SOURCE
    c = Config(device='cpu'); datasets = {s: TaskData(c, s) for s in ['train', 'validation', 'test']}
    assert [len(datasets[s]) for s in datasets] == [180000, 18000, 18000]
    manifest = datasets['train'].manifest
    recorded = json.loads(Path('results/protocols/mlp_final_protocol.json').read_text())['data_manifests']['sudoku']
    assert manifest == recorded
    split_rows = {}; ids = {}; digit_ids = {}
    for split, data in datasets.items():
        q, y = data.data['puzzle'], data.data['y']
        assert q.shape == y.shape == (len(data), 81)
        assert np.all((q == 0) | (q == y)) and legal_grids(y).all()
        assert file_hash(c.data_dir() / (split + '.npz')) == manifest['files'][split + '.npz']
        arrays = {}
        for name, value in data.data.items():
            h = hashlib.sha256(); h.update(str(value.dtype).encode()); h.update(json.dumps(value.shape).encode()); h.update(np.ascontiguousarray(value).tobytes())
            arrays[name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'content_sha256': h.hexdigest()}
        ids[split] = set(data.ids); assert len(ids[split]) == len(data)
        canonical = set()
        for row in q:
            mapping = {0: 0}; nxt = 1; key = []
            for digit in row:
                digit = int(digit)
                if digit not in mapping: mapping[digit] = nxt; nxt += 1
                key.append(mapping[digit])
            canonical.add(bytes(key))
        digit_ids[split] = canonical
        split_rows[split] = {'count': len(data), 'arrays': arrays, 'valid_targets': True, 'clues_preserved': True}
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not ids[left] & ids[right]
        assert not digit_ids[left] & digit_ids[right]
    mask = datasets['validation'].graph
    assert bool((mask.astype(int) @ mask.astype(int) > 0).all())
    data_result = {'status': 'passed', 'source_sha256': code_hash(), 'preprocessing_source_sha256': LEGACY_SOURCE,
                   'data_hash': manifest['data_hash'], 'splits': split_rows, 'cross_split_exact_duplicates': 0,
                   'cross_split_digit_relabel_duplicates': 0, 'full_symmetry_certification': False,
                   'full_uniqueness_certification': False, 'graph_diameter': 2, 'features': 38}
    masks = {}
    for support in SUPPORTS:
        other = make('F4-onehop', support, device='cpu')
        for split, baseline in datasets.items():
            candidate = TaskData(other, split)
            assert candidate.manifest == baseline.manifest
            left, right = baseline.batch(np.arange(16), 'cpu'), candidate.batch(np.arange(16), 'cpu')
            torch.testing.assert_close(left['x'], right['x'], rtol=0, atol=0)
            assert candidate.ids == baseline.ids
            for name in ['puzzle','y']: assert np.array_equal(candidate.data[name], baseline.data[name])
        a = allowed_mask(support)
        masks[other.mask_label()] = {'mask_sha256': mask_hash(a), 'diameter': diameter(a), 'allowed_elements': int(a.sum()), 'row_counts': a.sum(1).tolist()}
    assert masks['grid4']['diameter'] == 16 and masks['grid4']['allowed_elements'] == 369
    data_result.update(masks=masks, same_puzzle_target_arrays_for_both_masks=True, no_cache_regeneration=True)
    write_json(root() / 'checks/data.json', data_result)
    historical=[]; rejected=[]
    legacy_root=Path(os.environ['LOCAL_MASK_LEGACY_ROOT'])
    for method,label in [('onehop','F4-onehop'),('binomial','F4-binomial'),('recurrent','R32-onehop')]:
        for seed in FINAL_SEEDS:
            candidates=[]
            for path in (legacy_root/'runs/sudoku'/method).glob('*/seed_'+str(seed)+'/result.json'):
                result=json.loads(path.read_text())
                if result['config']['stage']=='final' and result['config']['inner_beta']==0: candidates.append((path,result))
            if len(candidates)!=1:
                rejected.append({'model':label,'seed':seed,'reason':'missing_or_ambiguous_artifacts'}); continue
            try:
                path,result=candidates[0]; folder=path.parent; old=LegacyConfig(**result['config'])
                assert old.hash()==result['config_hash'] and result['source_sha256']==LEGACY_SOURCE
                assert old.support=='graph' and old.width==128 and old.heads==4 and old.inner_beta==0
                assert (old.lr,old.weight_decay,old.clip,old.warmup_fraction,old.batch_size,old.steps,old.eval_every)==(.001,.01,1.,.05,64,30000,250)
                assert old.recurrent_steps==32 if method=='recurrent' else old.layers==4
                if method=='binomial': assert old.hops==4
                best=torch.load(folder/'checkpoint_best.pt',map_location='cpu',weights_only=False)
                last=torch.load(folder/'checkpoint_last.pt',map_location='cpu',weights_only=False)
                assert best['source_sha256']==last['source_sha256']==LEGACY_SOURCE
                assert best['data_hash']==last['data_hash']==manifest['data_hash']
                assert best['config']==last['config']==old.dict() and last['step']==30000
                old_model=LegacyEncoder(old); model=Encoder(from_legacy(old))
                for name,value in old_model.state_dict().items(): torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
                old_model.load_state_dict(best['model']); model.load_state_dict(best['model'])
                batch=datasets['test'].batch(np.arange(16),'cpu')
                with torch.no_grad():
                    torch.testing.assert_close(old_model(batch),model(batch),rtol=0,atol=0)
                    sample=model(batch).argmax(-1).numpy()+1
                generator=np.random.default_rng(seed+987123)
                for _ in range(30000): generator.integers(0,180000,64)
                assert generator.bit_generator.state==last['data_rng']
                assert last['scheduler']['last_epoch']==30000 and last['scheduler']['_last_lr']==[0.]
                group=last['optimizer']['param_groups'][0]
                assert group['weight_decay']==.01 and group['eps']==1e-8 and tuple(group['betas'])==(.9,.999) and not group['amsgrad']
                curve=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
                assert [r['step'] for r in curve]==[1]+list(range(250,30001,250))
                ranked=max(curve,key=lambda r:(r['solved_rate'],-r['loss'],-r['step']))
                assert ranked['step']==best['step']
                metric=json.loads((folder/'evaluation_test.json').read_text())
                assert metric['checkpoint_step']==best['step'] and metric['count']==18000
                with np.load(folder/'predictions_test.npz',allow_pickle=False) as stored:
                    assert stored['ids'].tolist()==datasets['test'].ids
                    pred=stored['predictions']; puzzle=datasets['test'].data['puzzle']
                    valid=legal_grids(pred)&np.all((puzzle==0)|(puzzle==pred),axis=1)
                    assert np.array_equal(valid,stored['solved']) and float(valid.mean())==metric['solved_rate']
                    assert np.array_equal(np.where(puzzle[:16]!=0,puzzle[:16],sample),pred[:16])
                hashes={name:file_hash(folder/name) for name in ['config.json','result.json','checkpoint_best.pt','checkpoint_last.pt','evaluation_test.json','predictions_test.npz','metrics.jsonl']}
                historical.append({'model':label,'support':'relational','seed':seed,'origin':'historical_quality_only',
                                   'legacy_config_hash':old.hash(),'legacy_source_sha256':LEGACY_SOURCE,
                                   'mapped_new_config_hash':make(label,'graph',seed,'final').hash(),
                                   'evaluation':metric,'artifact_hashes':hashes,'replay_examples':16,
                                   'current_source_not_assigned_to_old_results':True,'old_timing_as_current_speed_evidence':False})
            except (AssertionError,KeyError,FileNotFoundError,RuntimeError,ValueError) as error:
                rejected.append({'model':label,'seed':seed,'reason_type':type(error).__name__})
    result={'status':'passed','source_sha256':code_hash(),'reuse_all_15_relational_allowed':len(historical)==15 and not rejected,
            'historical_final_results':historical,'rejected':rejected,'saved_predictions_revalidated':len(historical)*18000,
            'checkpoint_replay_examples':len(historical)*16,'same_named_initialization_verified':True,
            'inactive_legacy_fields':{'train_count/val_count/test_count':'RRN loader uses all 180000/18000/18000 rows',
                                      'hops for onehop/recurrent':'effective [1] regardless of stored hops=4',
                                      'layers for recurrent':'one shared block; stored layers=4 ignored',
                                      'recurrent_steps for F4':'no recurrent loop; stored value ignored',
                                      'inner_width':'inactive with inner_beta=0','max_seconds':'orchestration limit; verified completion is 30000 updates',
                                      'generator/clue_fraction':'inactive for official supplied RRN arrays'}}
    write_json(root()/'checks/legacy_compatibility.json',result)
    print(json.dumps({'data_and_masks':'passed','historical_compatible':len(historical),'rejected':len(rejected)}),flush=True)
    return result
