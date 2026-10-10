"""Small metadata-only monitor checks, executed in a CPU allocation."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location("monitor", Path(__file__).with_name("monitor_mlp_final.py"))
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Checks require a CPU allocation")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        put = lambda name, data: monitor.write(root / name, data)
        put("mlp_final_submission.json", {"gpu_job": 2, "post_job_accounting_job": 3})
        put("mlp_final_protocol.json", {"run_count": 1, "updates_per_run": 30000,
            "baseline_configs_by_new_hash": {"example": {"task": "maze", "method": "binomial", "seed": 11}}})
        put("budget.json", {"allocations": [{"job_id": "1", "actual_seconds": 100},
                                            {"job_id": "2", "observed_seconds": 20}]})
        put("mlp_final_progress.json", {"job_id": "1", "message": "Stopped: PermissionError"})
        put("mlp_final_failure.json", {"error_type": "PermissionError"})
        records = {"2": {"state": "PENDING", "elapsed_seconds": 0, "gpu_count": 0}, "3": {"state": "PENDING"}}
        result = monitor.snapshot(root, records, 10000)
        assert result["health"] == "waiting" and result["alerts"] == []
        assert result["ignored_previous_progress_job"] == "1" and not result["monitor_finished"]
        current = {"task": "maze", "method": "binomial", "config_hash": "example", "seed": 11}
        put("mlp_final_progress.json", {"job_id": "2", "current": current})
        (root / "mlp_final_failure.json").unlink()
        run = root / "runs/maze/binomial/example/seed_11"
        run.mkdir(parents=True)
        metrics = run / "metrics.jsonl"
        metrics.write_text('{"step":250}\n{"step":500}\n{"step":')
        os.utime(metrics, (9500, 9500))
        os.utime(root / "mlp_final_progress.json", (7000, 7000))
        records["2"].update(state="RUNNING", elapsed_seconds=40, gpu_count=1)
        result = monitor.snapshot(root, records, 10000)
        assert result["last_metrics"]["step"] == 500 and result["alerts"] == []
        assert result["observed_gpu_hours"] == round(140 / 3600, 4)
        os.utime(metrics, (7000, 7000))
        result = monitor.snapshot(root, records, 10000)
        assert "no_file_activity_for_30_minutes" in result["alerts"]
        records["2"]["state"] = "FAILED"
        assert "gpu_job_failed" in monitor.snapshot(root, records, 10000)["alerts"]
        records["2"]["state"] = "COMPLETED"
        assert "series_incomplete_after_job_completion" in monitor.snapshot(root, records, 10000)["alerts"]
        put("mlp_chunk_execution.json", {"enabled": True, "status": "queued", "chunks": [{"index": 1, "gpu_job": 2}]})
        put("mlp_final_result.json", {"status": "budget_paused"})
        result = monitor.snapshot(root, records, 10000)
        assert result["health"] == "between_chunks" and not result["alerts"]
        (root / "mlp_chunk_execution.json").unlink()
        monitor.write(run / "result.json", {"status": "completed", "step": 30000})
        monitor.write(run / "evaluation_test.json", {})
        assert not monitor.snapshot(root, records, 10000)["monitor_finished"]
        records["3"]["state"] = "COMPLETED"
        result = monitor.snapshot(root, records, 10000)
        assert result["completed_runs"] == 1 and result["monitor_finished"] and not result["alerts"]
    print(json.dumps({"status": "passed", "scheduler_job_id": os.environ["SLURM_JOB_ID"],
                      "checks": ["old_attempt_ignored", "partial_metric_line", "live_budget", "stale_activity",
                                 "failed_job", "incomplete_series", "normal_pause_between_chunks", "completion_after_accounting"]}))


if __name__ == "__main__":
    main()
