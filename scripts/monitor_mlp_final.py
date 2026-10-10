"""Read-only experiment monitoring; no training, resubmission or model imports."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1] / "experiments/transport_reasoning"
TAG = "# transport-mlp-watch-20261010"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED", "SPECIAL_EXIT"}


def read(path, default=None):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def command(args):
    return subprocess.check_output(args, universal_newlines=True, timeout=20,
                                   env=dict(os.environ, LC_ALL="C", LANG="C"))


def scheduler(jobs):
    records = {}
    output = command(["/usr/bin/sacct", "-j", ",".join(jobs), "--noheader", "--parsable2",
                      "--format=JobID,State,ElapsedRaw,ExitCode,AllocTRES"])
    for line in output.splitlines():
        fields = line.split("|")
        if fields[0] not in jobs:
            continue
        gpu = next((int(x.split("=")[1]) for x in fields[4].split(",")
                    if x.startswith("gres/gpu=")), 0)
        records[fields[0]] = dict(state=fields[1].split()[0].rstrip("+"),
                                 elapsed_seconds=int(fields[2]), exit_code=fields[3], gpu_count=gpu)
    # The live controller takes precedence over lagging accounting.
    output = command(["/usr/bin/squeue", "-u", str(os.getuid()), "--noheader", "--format=%i|%T|%r"])
    for line in output.splitlines():
        job, state, reason = line.strip().split("|", 2)
        if job in jobs:
            records.setdefault(job, {}).update(state=state, reason=reason)
    return records


def tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        return stream.read().decode("utf-8", errors="replace")


def snapshot(root, records, now):
    submission = read(root / "mlp_final_submission.json")
    protocol = read(root / "mlp_final_protocol.json")
    job = str(submission["gpu_job"])
    post = str(submission.get("post_job_accounting_job", ""))
    progress = read(root / "mlp_final_progress.json", {})
    current_attempt = str(progress.get("job_id")) == job
    state = records.get(job, {}).get("state", "UNKNOWN")
    result = {"checked_utc": datetime.datetime.utcfromtimestamp(now).isoformat() + "Z",
              "job_id": job, "scheduler": records.get(job, {"state": "UNKNOWN"}),
              "accounting_job_id": post, "accounting_job": records.get(post, {}),
              "expected_runs": protocol["run_count"], "completed_runs": 0,
              "progress_matches_job": current_attempt, "current": None, "alerts": []}
    alerts = result["alerts"]
    for config_hash, config in protocol["baseline_configs_by_new_hash"].items():
        run = root / "runs" / config["task"] / config["method"] / config_hash / ("seed_%s" % config["seed"])
        item = read(run / "result.json", {})
        if item.get("status") == "completed" and item.get("step") == protocol["updates_per_run"]:
            if (run / "evaluation_test.json").exists():
                result["completed_runs"] += 1
    if current_attempt:
        result["progress_message"] = progress.get("message")
        current = progress.get("current")
        result["current"] = current
        if current:
            run = root / "runs" / current["task"] / current["method"] / current["config_hash"] / ("seed_%s" % current["seed"])
            metrics = run / "metrics.jsonl"
            for line in reversed(tail(metrics).splitlines()):
                try:
                    result["last_metrics"] = json.loads(line)
                    break
                except ValueError:
                    continue
            if metrics.exists():
                result["metrics_age_seconds"] = round(max(0, now - metrics.stat().st_mtime))
            checkpoint = run / "checkpoint_last.pt"
            if checkpoint.exists():
                result["checkpoint_age_seconds"] = round(max(0, now - checkpoint.stat().st_mtime))
        failure = read(root / "mlp_final_failure.json")
        if failure:
            alerts.append("training_failure:" + failure.get("error_type", "unknown"))
    elif progress:
        result["ignored_previous_progress_job"] = progress.get("job_id")
    log = root / ("mlp-final-%s.log" % job)
    recent_log = tail(log)
    for marker in ("Traceback (most recent call last)", "CUDA out of memory", "PermissionError", "slurmstepd: error"):
        if marker in recent_log:
            alerts.append("job_log_error:" + marker)
    if state == "RUNNING":
        activity = [p.stat().st_mtime for p in (log, root / "mlp_final_progress.json") if p.exists()]
        if "metrics_age_seconds" in result:
            activity.append(now - result["metrics_age_seconds"])
        if "checkpoint_age_seconds" in result:
            activity.append(now - result["checkpoint_age_seconds"])
        age = now - max(activity) if activity else records.get(job, {}).get("elapsed_seconds", 0)
        result["activity_age_seconds"] = round(max(0, age))
        if age > 1800:
            alerts.append("no_file_activity_for_30_minutes")
    if state in TERMINAL and state != "COMPLETED":
        alerts.append("gpu_job_" + state.lower())
    chunk_plan = read(root / "mlp_chunk_execution.json", {})
    outcome = read(root / "mlp_final_result.json", {})
    planned_pause = (chunk_plan.get("enabled") and current_attempt
                     and outcome.get("status") == "budget_paused"
                     and records.get(post, {}).get("state") in ("PENDING", "RUNNING", "COMPLETING"))
    if chunk_plan.get("enabled"):
        result["execution_mode"] = "serial_checkpointed_chunks"
        result["chunk_index"] = next((row["index"] for row in chunk_plan["chunks"] if str(row["gpu_job"]) == job), None)
        result["chunk_dispatch_status"] = chunk_plan.get("status")
    if state == "COMPLETED" and result["completed_runs"] != result["expected_runs"] and not planned_pause:
        alerts.append("series_incomplete_after_job_completion")
    if state == "UNKNOWN":
        alerts.append("scheduler_job_not_found")
    budget = read(root / "budget.json", {})
    consumed = 0
    for allocation in budget.get("allocations", []):
        record = records.get(str(allocation["job_id"]), {})
        consumed += (record["elapsed_seconds"] * record["gpu_count"]
                     if "elapsed_seconds" in record and "gpu_count" in record
                     else allocation.get("actual_seconds", allocation.get("observed_seconds", 0)))
    result["observed_gpu_hours"] = round(consumed / 3600, 4)
    result["remaining_gpu_hours"] = round(max(0, 86400 - consumed) / 3600, 4)
    if consumed > 86400:
        alerts.append("gpu_budget_exceeded")
    if records.get(post, {}).get("state") in TERMINAL - {"COMPLETED"}:
        alerts.append("accounting_job_failed")
    result["monitor_finished"] = state in TERMINAL and (not post or records.get(post, {}).get("state") in TERMINAL)
    result["health"] = "attention" if alerts else ("waiting" if state == "PENDING" else state.lower())
    if planned_pause and state == "COMPLETED" and not alerts:
        result["health"] = "between_chunks"
    return result


def remove_schedule():
    before = command(["/usr/bin/crontab", "-l"])
    lines = [line for line in before.splitlines() if not line.endswith(TAG)]
    if len(lines) == len(before.splitlines()):
        return
    if command(["/usr/bin/crontab", "-l"]) != before:
        raise RuntimeError("Crontab changed concurrently; retry cleanup next time")
    subprocess.run(["/usr/bin/crontab", "-"], input="\n".join(lines) + "\n",
                   universal_newlines=True, timeout=20, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduled", action="store_true")
    args = parser.parse_args()
    output = ROOT / "monitoring"
    output.mkdir(exist_ok=True)
    with (output / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        now = time.time()
        try:
            submission = read(ROOT / "mlp_final_submission.json")
            budget = read(ROOT / "budget.json", {})
            jobs = {str(a["job_id"]) for a in budget.get("allocations", [])}
            jobs.add(str(submission["gpu_job"]))
            if submission.get("post_job_accounting_job"):
                jobs.add(str(submission["post_job_accounting_job"]))
            status = snapshot(ROOT, scheduler(sorted(jobs)), now)
        except Exception as exc:
            status = {"checked_utc": datetime.datetime.utcfromtimestamp(now).isoformat() + "Z",
                      "health": "monitor_error", "alerts": [type(exc).__name__], "monitor_finished": False}
        status["trigger"] = "cron" if args.scheduled else "manual"
        if args.scheduled and status["monitor_finished"]:
            try:
                remove_schedule()
                status["schedule_removed"] = True
            except Exception as exc:
                status["alerts"].append("schedule_cleanup:" + type(exc).__name__)
        write(output / "latest.json", status)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(status, ensure_ascii=True, sort_keys=True) + "\n")
        print(json.dumps(status, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
