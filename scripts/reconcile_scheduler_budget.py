"""Reconcile allocated GPU wall time after a scheduler job has finished."""
import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path


def main(job):
    root = Path("experiments/transport_reasoning")
    with (root / "campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_path = root / "budget.json"
        budget = json.loads(budget_path.read_text())
        jobs = sorted({a["job_id"] for a in budget["allocations"]} | {str(job)})
        records = {}
        for attempt in range(6):
            output = subprocess.check_output(["sacct", "-j", ",".join(jobs), "--noheader", "--parsable2",
                       "--format=JobID,State,ElapsedRaw,AllocTRES"], universal_newlines=True)
            for line in output.splitlines():
                parts = line.split("|")
                if parts[0] in jobs:
                    gpu = next((int(v.split("=")[1]) for v in parts[3].split(",") if v.startswith("gres/gpu=")), 0)
                    records[parts[0]] = {"state": parts[1], "elapsed_seconds": int(parts[2]), "gpu_count": gpu}
            if str(job) in records and records[str(job)]["state"] not in ("PENDING", "RUNNING", "COMPLETING"):
                break
            time.sleep(5)
        else:
            raise RuntimeError("Scheduler accounting is not final yet; reservation remains charged")
        if str(job) not in {a["job_id"] for a in budget["allocations"]}:
            budget["allocations"].append({"job_id": str(job), "reserved_seconds": 33300,
                                         "status": "finished", "purpose": "inner_mlp_final_v1"})
        for allocation in budget["allocations"]:
            record = records.get(allocation["job_id"])
            if not record or record["state"] in ("PENDING", "RUNNING", "COMPLETING"):
                continue
            if "actual_seconds" in allocation:
                allocation.setdefault("python_measured_seconds", allocation["actual_seconds"])
            allocation.update(actual_seconds=record["elapsed_seconds"] * record["gpu_count"],
                              gpu_count=record["gpu_count"], status="finished",
                              accounting_source="Slurm ElapsedRaw times allocated GPU count")
        budget["used_gpu_seconds"] = sum(a.get("actual_seconds", a["reserved_seconds"]) for a in budget["allocations"])
        budget["observed_gpu_seconds"] = budget["used_gpu_seconds"]
        temporary = budget_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n"); temporary.replace(budget_path)
        (root / "revision_audit" / "scheduler_accounting.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"gpu_seconds": budget["used_gpu_seconds"], "ceiling_seconds": 86400,
                          "within_ceiling": budget["used_gpu_seconds"] <= 86400}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--job", type=int, required=True)
    main(parser.parse_args().job)
    from mlp_chunk_scheduler import continue_after
    continue_after(parser.parse_args().job)
