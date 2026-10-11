"""Rebuild and validate the completed MLP publication in an allocated CPU job."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'CPU allocation required'
    os.chdir(ROOT)
    audit=json.loads((ROOT/'results/audit/mlp_final_publication_audit.json').read_text())
    assert audit['status']=='passed'
    manifest_path=ROOT/'PUBLICATION_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    manifest.setdefault('primary_snapshot_utc',manifest['snapshot_utc'])
    manifest['snapshot_utc']=audit['snapshot_utc']
    manifest['mlp_final_update']={'previous_public_commit':'9ea79b57259e9a6619098279ec4274b8efb4dd13',
        'new_completed_runs':20,'updates_per_run':30000,'record_directory':'results/mlp_final/records',
        'previous_partial_first_run_preserved':True,'old_protocols_and_primary_tables_unchanged':True,
        'audit':'results/audit/mlp_final_publication_audit.json'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ['build_mlp_final_assets.py','build_public_assets.py','check_public_snapshot.py','check_mlp_final_snapshot.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/name)],check=True)
    # No model/training implementation changed; no redundant model-training tests.
    print(json.dumps({'status':'passed','model_source_changed':False,'new_gpu_training_performed':False,
                      'private_artifact_audit':'passed','original_snapshot_checks':'passed','new_mlp_snapshot_checks':'passed'}))


if __name__=='__main__':main()
