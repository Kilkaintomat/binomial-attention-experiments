"""Serial CPU-side dispatch after authoritative accounting of each GPU chunk."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path("experiments/transport_reasoning")


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def decide(state, outcome, budget):
    if state != "COMPLETED":
        return {"status": "stopped_after_failed_chunk", "scheduler_state": state}
    if outcome.get("completed") == outcome.get("expected") == 20 and outcome.get("status") == "completed":
        return {"status": "completed"}
    if outcome.get("status") != "budget_paused" or outcome.get("expected") != 20:
        return {"status": "stopped_invalid_chunk_result"}
    if any(a.get("status") != "finished" or "actual_seconds" not in a for a in budget["allocations"]):
        return {"status": "stopped_unreconciled_budget"}
    remaining = int(86400 - sum(a["actual_seconds"] for a in budget["allocations"]))
    # A shorter last chunk must still have useful training time after margins.
    if remaining < 900:
        return {"status": "budget_exhausted", "remaining_seconds": remaining}
    return {"status": "continue", "walltime_seconds": min(7200, remaining), "remaining_seconds": remaining}


def call(args, **kwargs):
    return subprocess.check_output(args, universal_newlines=True, timeout=45, **kwargs).strip()


def continue_after(job):
    path = ROOT / "mlp_chunk_execution.json"
    if not path.exists() or not read(path).get("enabled"):
        return
    with (ROOT / "chunk_dispatch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(path)
        if str(plan["chunks"][-1]["gpu_job"]) != str(job):
            print("Chunk already superseded; no duplicate submission")
            return
        if plan.get("dispatch_in_progress"):
            raise RuntimeError("Previous dispatch is incomplete; inspect before retry")
        for source, expected in plan["source_hashes"].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Chunk source differs from verified source")
        checks = read(ROOT / "revision_audit/chunk_checks.json")
        if checks.get("status") != "passed" or checks.get("source_hashes") != plan["source_hashes"]:
            raise RuntimeError("Chunk verification is missing or stale")
        raw = call(["sacct", "-j", str(job), "-nP", "--format=JobID,State"])
        states = {line.split("|")[0]: line.split("|")[1].split()[0].rstrip("+")
                  for line in raw.splitlines() if "|" in line}
        progress = read(ROOT / "mlp_final_progress.json")
        outcome = read(ROOT / "mlp_final_result.json") if (ROOT / "mlp_final_result.json").exists() else {}
        if str(progress.get("job_id")) != str(job):
            outcome = {}
        action = decide(states.get(str(job)), outcome, read(ROOT / "budget.json"))
        plan["chunks"][-1]["outcome"] = action
        plan["status"] = action["status"]
        write(path, plan)
        if action["status"] != "continue":
            print(json.dumps(action))
            return
        accounting_job = os.environ.get("SLURM_JOB_ID")
        if not accounting_job:
            raise RuntimeError("Automatic dispatch requires a CPU scheduler allocation")
        wall = action["walltime_seconds"]
        duration = "%02d:%02d:%02d" % (wall // 3600, wall % 3600 // 60, wall % 60)
        environment = dict(os.environ)
        environment.update(TRANSPORT_ROOT=str(Path.cwd()),
                           TRANSPORT_INPUTS=os.environ["TRANSPORT_INPUTS"],
                           TRANSPORT_IMAGE=os.environ["TRANSPORT_IMAGE"])
        plan["dispatch_in_progress"] = {"after_gpu_job": int(job), "accounting_job": accounting_job}
        write(path, plan)
        gpu = call(["sbatch", "--parsable", "--hold", "--account=" + os.environ["TRANSPORT_ACCOUNT"], "--partition=" + os.environ["TRANSPORT_GPU_PARTITION"],
                    "--gres=gpu:a100:1", "--cpus-per-task=4", "--mem=48G", "--time=" + duration,
                    "--dependency=afterok:" + accounting_job, "--job-name=transport-mlp-chunk",
                    "--output=experiments/transport_reasoning/mlp-final-%j.log",
                    "scripts/runtime.sbatch", "scripts/audited_mlp_runtime.py", "run"], env=environment).split(";")[0]
        plan["dispatch_in_progress"]["gpu_job"] = gpu
        write(path, plan)
        wrap = "cd " + shlex.quote(str(Path.cwd())) + " && /usr/bin/python3 scripts/reconcile_scheduler_budget.py --job " + gpu
        post = call(["sbatch", "--parsable", "--account=" + os.environ["TRANSPORT_ACCOUNT"], "--partition=" + os.environ["TRANSPORT_CPU_PARTITION"],
                     "--cpus-per-task=1", "--mem=512M", "--time=00:05:00", "--dependency=afterany:" + gpu,
                     "--job-name=transport-accounting", "--output=experiments/transport_reasoning/accounting-%j.log",
                     "--wrap=" + wrap], env=environment).split(";")[0]
        index = len(plan["chunks"]) + 1
        plan["chunks"].append({"index": index, "gpu_job": int(gpu), "accounting_job": int(post),
                               "walltime_seconds": wall, "previous_gpu_job": int(job)})
        plan["status"] = "queued"
        write(path, plan)
        submission = read(ROOT / "mlp_final_submission.json")
        submission.update(gpu_job=int(gpu), post_job_accounting_job=int(post), chunk_index=index,
                          scheduler_walltime_seconds=wall, previous_chunk_job=int(job),
                          recorded_utc=datetime.datetime.utcnow().isoformat() + "Z")
        write(ROOT / "mlp_final_submission.json", submission)
        call(["scontrol", "release", gpu])
        plan.pop("dispatch_in_progress")
        write(path, plan)
        print(json.dumps({"status": "next_chunk_queued", "gpu_job": gpu, "accounting_job": post,
                          "walltime_seconds": wall, "chunk_index": index}))
