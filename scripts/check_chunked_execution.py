"""CPU checks for exact off-grid continuation and bounded serial dispatch."""
import copy
import json
import os
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import torch
from transport_reasoning.config import Config, ROOT, write_json
from transport_reasoning.data import prepare, file_hash
import transport_reasoning.training as training
import audited_mlp_runtime as observed
import mlp_chunk_scheduler as dispatch

SOURCES = ("scripts/audited_mlp_runtime.py", "scripts/chunked_mlp_runtime.py",
           "scripts/mlp_chunk_scheduler.py", "scripts/reconcile_scheduler_budget.py",
           "scripts/check_chunked_execution.py")


def equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def check_resume(parent):
    cwd = Path.cwd()
    original_save = training.atomic_checkpoint
    verified = []
    try:
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            for task in ("maze", "sudoku"):
                c = Config(task=task, size=4, train_count=16, val_count=4, test_count=4,
                           width=16, layers=1, heads=4, hops=2, inner_beta=.1, inner_width=16,
                           steps=6, eval_every=2, batch_size=4)
                direct = Path(temporary) / (task + "_direct"); direct.mkdir(); os.chdir(direct)
                prepare(c)
                training.atomic_checkpoint = observed.checkpoint_wrapper(original_save)
                training.train(c)
                expected = copy.deepcopy(torch.load(c.run_dir() / "checkpoint_last.pt", weights_only=False))
                expected_best = copy.deepcopy(torch.load(c.run_dir() / "checkpoint_best.pt", weights_only=False))
                expected_steps = [json.loads(line)["step"] for line in (c.run_dir() / "metrics.jsonl").read_text().splitlines()]
                resumed = Path(temporary) / (task + "_resumed"); resumed.mkdir(); os.chdir(resumed)
                prepare(c)
                counter = [0]
                loss = training.loss_for

                def counted_loss(*args, **kwargs):
                    counter[0] += 1
                    return loss(*args, **kwargs)

                class Deadline:
                    def __sub__(self, margin):
                        return -float("inf") if counter[0] >= 3 else float("inf")

                with mock.patch.object(training, "loss_for", counted_loss):
                    partial = training.train(c, deadline=Deadline())
                assert partial["status"] == "budget_paused" and partial["step"] == 3
                assert [json.loads(line)["step"] for line in (c.run_dir() / "metrics.jsonl").read_text().splitlines()] == [1, 2]
                training.train(c, resume=c.run_dir() / "checkpoint_last.pt")
                actual = torch.load(c.run_dir() / "checkpoint_last.pt", weights_only=False)
                best = torch.load(c.run_dir() / "checkpoint_best.pt", weights_only=False)
                for key in ("model", "optimizer", "scheduler", "data_rng", "torch_rng", "numpy_rng", "python_rng", "step", "best"):
                    equal(expected[key], actual[key])
                equal(expected_best["model"], best["model"])
                assert expected_best["step"] == best["step"]
                assert [json.loads(line)["step"] for line in (c.run_dir() / "metrics.jsonl").read_text().splitlines()] == expected_steps
                verified.append(task)
    finally:
        os.chdir(cwd)
        training.atomic_checkpoint = original_save
    return verified


def check_dispatch(parent):
    budget = {"allocations": [{"job_id": "21", "actual_seconds": 53136, "status": "finished"}]}
    paused = {"status": "budget_paused", "completed": 3, "expected": 20}
    assert dispatch.decide("COMPLETED", paused, budget)["walltime_seconds"] == 7200
    assert dispatch.decide("FAILED", paused, budget)["status"] == "stopped_after_failed_chunk"
    assert dispatch.decide("COMPLETED", {}, budget)["status"] == "stopped_invalid_chunk_result"
    assert dispatch.decide("COMPLETED", dict(paused, completed=20, status="completed"), budget)["status"] == "completed"
    assert dispatch.decide("COMPLETED", paused, {"allocations": [{"actual_seconds": 85800, "status": "finished"}]})["status"] == "budget_exhausted"
    assert dispatch.decide("COMPLETED", paused, {"allocations": [{"actual_seconds": 84600, "status": "finished"}]})["walltime_seconds"] == 1800
    assert dispatch.decide("COMPLETED", paused, {"allocations": [{"status": "active"}]})["status"] == "stopped_unreconciled_budget"
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        root = Path(temporary)
        plan = {"enabled": True, "source_hashes": {}, "chunks": [{"index": 1, "gpu_job": 21, "accounting_job": 22, "walltime_seconds": 7200}]}
        write_json(root / "mlp_chunk_execution.json", plan)
        write_json(root / "revision_audit/chunk_checks.json", {"status": "passed", "source_hashes": {}})
        write_json(root / "budget.json", budget)
        write_json(root / "mlp_final_progress.json", {"job_id": "21"})
        write_json(root / "mlp_final_result.json", paused)
        write_json(root / "mlp_final_submission.json", {"gpu_job": 21})
        calls = []

        def fake_call(args, **kwargs):
            calls.append(args)
            if args[0] == "sacct":
                return "21|COMPLETED|"
            if args[0] == "sbatch":
                return "1001" if "--gres=gpu:a100:1" in args else "1002"
            if args == ["scontrol", "release", "1001"]:
                assert json.loads((root / "mlp_final_submission.json").read_text())["gpu_job"] == 1001
                return ""
            raise AssertionError(args)

        with mock.patch.dict(os.environ, {"TRANSPORT_ACCOUNT": "research", "TRANSPORT_GPU_PARTITION": "gpu", "TRANSPORT_CPU_PARTITION": "cpu", "TRANSPORT_INPUTS": "external-inputs", "TRANSPORT_IMAGE": "runtime.sif"}), mock.patch.object(dispatch, "ROOT", root), mock.patch.object(dispatch, "call", fake_call):
            dispatch.continue_after(21)
            n = len(calls)
            dispatch.continue_after(21)
            assert len(calls) == n
        gpu = next(args for args in calls if "--gres=gpu:a100:1" in args)
        cpu = next(args for args in calls if "--job-name=transport-accounting" in args)
        assert "--hold" in gpu and "--time=02:00:00" in gpu
        assert "--dependency=afterok:" + os.environ["SLURM_JOB_ID"] in gpu
        assert "--dependency=afterany:1001" in cpu
        assert json.loads((root / "mlp_chunk_execution.json").read_text())["chunks"][-1]["accounting_job"] == 1002


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Checks require a CPU scheduler allocation")
    torch.set_num_threads(4)
    before = file_hash("scripts/mlp_final_campaign.py")
    observed.verify()
    parent = (ROOT / "revision_audit/filesystem_tests").resolve()
    tasks = check_resume(parent)
    check_dispatch(parent)
    import chunked_mlp_runtime as chunk
    import mlp_final_campaign as campaign
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        root = Path(temporary)
        write_json(root / "mlp_chunk_execution.json", {"source_hashes": {}, "chunks": [
            {"index": 1, "gpu_job": os.environ["SLURM_JOB_ID"], "walltime_seconds": 7200}]})
        write_json(root / "revision_audit/chunk_checks.json", {"status": "passed", "source_hashes": {}})
        reservations = []
        original = campaign.RESERVATION_SECONDS
        with mock.patch.object(chunk, "ROOT", root), mock.patch.object(campaign, "run", lambda: reservations.append(campaign.RESERVATION_SECONDS)):
            chunk.run()
        assert reservations == [6900] and campaign.RESERVATION_SECONDS == original
    assert before == file_hash("scripts/mlp_final_campaign.py")
    result = {"status": "passed", "scheduler_job_id": os.environ["SLURM_JOB_ID"],
              "off_grid_resume_bitwise_equal_including_optimizer_rng_and_best_checkpoint": tasks,
              "validation_grid_unchanged": True, "threshold_pause_archives_verified": True,
              "dispatcher_serial_dependencies_budget_cap_and_idempotency": True,
              "actual_runner_receives_shorter_deadline_budget": True,
              "source_hashes": {p: file_hash(p) for p in SOURCES},
              "unchanged_training_source_sha256": training.source_hash(), "unchanged_base_runner_sha256": before}
    write_json(ROOT / "revision_audit/chunk_checks.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
