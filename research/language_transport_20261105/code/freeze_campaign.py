"""Freeze the verified campaign once. Never rewrite older experiment protocols."""
import os
import time
from common import ART,SIDE,ROOT,atomic,freeze,digest,source_hash,read
from plan import pilots,pretrained,spec,TASKS,METHODS,SEEDS


def main():
    assert os.environ.get('SLURM_JOB_ID')
    for name in ['claims','workers','recovered_claims']:(ART/name).mkdir(exist_ok=True)
    check=read(ART/'cpu_preflight.json')
    assert check['status']=='passed' and check['source_sha256']==source_hash()
    plan=dict(campaign='language_transport_20261105',source_sha256=source_hash(),protocol_sha256=digest(SIDE/'PROTOCOL_RU.md'),
        data_manifest_sha256=digest(SIDE/'data/manifest.json'),seeds=SEEDS,main_scratch_runs=48,main_lora_runs=18,
        frozen_evaluations=2,pilot_runs=6,training_slice_seconds=7200,max_concurrent_gpus=2,
        inference_r=[1,2,4,8,16,32,64,80,96],inference_loops=[1,2,4,8,16],
        predecessor='research/qbm_mlp_20261102',predecessor_total=36,
        model_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        pilot_selection='VT baseline validation <=4 only; LR .0003/.001; r fixed 8; steps 8000 or 16000 by prespecified rule')
    freeze(ART/'manifest.json',plan)
    (ART/'manifest.sha256').write_text(digest(ART/'manifest.json')+'\n')
    freeze(ART/'pilot_matrix.json',pilots())
    freeze(ART/'planned_pretrained_matrix.json',pretrained())
    preview=[spec('scratch',task,method,seed) for task in TASKS for method in METHODS for seed in SEEDS]
    for x in preview:x['lr']='selected_on_pilot_validation';x['steps']='8000_or_16000_selected_on_pilot_validation'
    freeze(ART/'planned_scratch_matrix.json',preview)
    protected={}
    old=ROOT/'research/qbm_mlp_20261102'
    for p in sorted((old/'code').rglob('*')):
        if p.is_file() and p.suffix in ['.py','.sh','.sbatch']:protected[str(p.relative_to(ROOT))]=digest(p)
    for p in [old/'PROTOCOL_RU.md',old/'artifacts/manifest.json']:
        if p.exists():protected[str(p.relative_to(ROOT))]=digest(p)
    freeze(ART/'protected_files.json',protected)
    if not (ART/'budget.json').exists():atomic(ART/'budget.json',{'allocations':[]})
    if not (ART/'budget_authorization.json').exists():
        atomic(ART/'budget_authorization.json',dict(campaign_gpu_hours=240,global_gpu_hours=500,
            prior_reserved_gpu_hours=200,max_concurrent_gpus=2,
            basis='User instruction 2026-11-05: prepare and launch both language task families, download as necessary, autonomously choose interpretation and increase GPU-hours as needed',
            automatic_adjustment='Raise new campaign reservation if measured benchmark estimate plus prescribed margins exceeds initial ceiling; old authorization/ledgers remain read-only',
            epoch=time.time()))
    atomic(ART/'ready.json',dict(status='ready',manifest_sha256=digest(ART/'manifest.json'),
        prepared_epoch=time.time(),cpu_job_id=os.environ['SLURM_JOB_ID'],main_trainings=66,frozen_evaluations=2,pilot_trainings=6))


if __name__=='__main__':main()
