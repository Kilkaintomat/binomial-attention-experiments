"""Bound the unchanged campaign runner by a shorter scheduler allocation."""
import json
import os
from pathlib import Path
import shutil

from transport_reasoning.config import ROOT
from transport_reasoning.data import file_hash


def run():
    import mlp_final_campaign as campaign
    plan = json.loads((ROOT / "mlp_chunk_execution.json").read_text())
    checks = json.loads((ROOT / "revision_audit/chunk_checks.json").read_text())
    if checks.get("status") != "passed" or checks.get("source_hashes") != plan["source_hashes"]:
        raise RuntimeError("Chunk execution checks are missing or stale")
    for path, expected in plan["source_hashes"].items():
        if file_hash(path) != expected:
            raise RuntimeError("Chunk execution source changed after verification")
    job = os.environ["SLURM_JOB_ID"]
    slot = next(row for row in plan["chunks"] if str(row["gpu_job"]) == job)
    limit = int(slot["walltime_seconds"])
    if not 600 <= limit <= 7200:
        raise RuntimeError("Invalid chunk wall time")
    # The original runner already stops training 210 seconds before its budget
    # boundary. Add 300 seconds for container startup and final report generation.
    original = campaign.RESERVATION_SECONDS
    campaign.RESERVATION_SECONDS = limit - 300
    print(json.dumps({"chunk": slot["index"], "walltime_seconds": limit,
                      "python_reservation_seconds": campaign.RESERVATION_SECONDS}), flush=True)
    try:
        campaign.run()
    finally:
        campaign.RESERVATION_SECONDS = original
        archive = ROOT / "chunks" / ("job_" + job)
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("mlp_final_progress.json", "mlp_final_result.json", "mlp_final_state.json",
                     "mlp_final_failure.json", "budget.json", "mlp_final_environment.json"):
            path = ROOT / name
            if path.exists():
                shutil.copy2(path, archive / name)
