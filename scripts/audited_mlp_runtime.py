"""Observation-only extension: threshold weight archives and audited reporting."""
import argparse
import copy
import json
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transport_reasoning.config import Config, ROOT, write_json
import transport_reasoning.training as training


def copy_snapshot(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".copy.tmp")
    if temporary.exists():
        temporary.unlink()
    with source.open("rb") as incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    temporary.replace(destination)


def checkpoint_wrapper(original):
    def save(path, state):
        if path.name != "checkpoint_last.pt":
            return original(path, state)
        metrics_path = path.parent / "metrics.jsonl"
        curves = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
        active = bool(curves) and curves[-1]["step"] == state["step"]
        evaluated = path.parent / "checkpoint_last_evaluated.pt"
        # A deadline can pause between validation steps. Preserve the checkpoint
        # corresponding to the previous metric for any later threshold crossing.
        if not active and curves and path.exists() and not evaluated.exists():
            copy_snapshot(path, evaluated)
        pending = []
        if active:
            thresholds = (.5, .9) if state["config"]["task"] == "sudoku" else (.9,)
            for threshold in thresholds:
                directory = path.parent / ("threshold_" + str(round(100 * threshold)))
                metadata = directory / "archive.json"
                prior = json.loads(metadata.read_text()) if metadata.exists() else {}
                if prior.get("status") == "confirmed":
                    continue
                current = curves[-1]; previous = curves[-2] if len(curves) > 1 else None
                passing = current["solved_rate"] >= threshold
                prior_passing = previous is not None and previous["solved_rate"] >= threshold
                if passing and not prior_passing:
                    if prior.get("status") == "candidate" and prior.get("first_passing_step") == current["step"]:
                        continue  # Re-saving the same step at a deadline is idempotent.
                    failure_step = previous["step"] if previous is not None else None
                    if failure_step is not None:
                        if not path.exists():
                            raise RuntimeError("Previous threshold checkpoint is missing")
                        copy_snapshot(evaluated if evaluated.exists() else path,
                                      directory / "checkpoint_failing.pt")
                    pending.append((directory, {"threshold": threshold, "status": "candidate",
                                   "failing_step": failure_step, "first_passing_step": current["step"]}, True))
                elif passing and prior_passing:
                    if prior.get("status") != "candidate":
                        raise RuntimeError("Threshold confirmation has no archived first passing checkpoint")
                    prior.update(status="confirmed", confirming_step=current["step"])
                    pending.append((directory, prior, False))
                elif prior.get("status") == "candidate":
                    prior.update(status="unconfirmed", failed_confirmation_step=current["step"])
                    write_json(metadata, prior)
        original(path, state)
        for directory, metadata, first_pass in pending:
            if first_pass:
                copy_snapshot(path, directory / "checkpoint_passing.pt")
            write_json(directory / "archive.json", metadata)
        if active and evaluated.exists():
            evaluated.unlink()
    return save


def verify():
    from transport_reasoning.data import prepare, file_hash
    torch.set_num_threads(4)
    original = training.atomic_checkpoint
    wrapped = checkpoint_wrapper(original)
    # Exercise the SAME project filesystem as real checkpoints, not node-local /tmp.
    test_parent = (ROOT / "revision_audit" / "filesystem_tests").resolve()
    test_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=test_parent) as temporary:
        root = Path(temporary)
        metric_path = root / "metrics.jsonl"
        for step, score in [(250, .2), (500, .7), (750, .95), (1000, .8), (1250, .95), (1500, .96), (1750, .97)]:
            with metric_path.open("a") as f:
                f.write(json.dumps({"step": step, "solved_rate": score}) + "\n")
            wrapped(root / "checkpoint_last.pt", {"config": {"task": "sudoku"}, "step": step,
                                                  "model": {"value": torch.tensor(step)}})
        for threshold, fail, passing, confirmation in [(50, 250, 500, 750), (90, 1000, 1250, 1500)]:
            directory = root / ("threshold_" + str(threshold))
            metadata = json.loads((directory / "archive.json").read_text())
            assert metadata["status"] == "confirmed" and metadata["confirming_step"] == confirmation
            assert torch.load(directory / "checkpoint_failing.pt", weights_only=False)["step"] == fail
            assert torch.load(directory / "checkpoint_passing.pt", weights_only=False)["step"] == passing
        # An off-grid budget pause must not modify confirmed threshold archives.
        wrapped(root / "checkpoint_last.pt", {"config": {"task": "sudoku"}, "step": 1751})
    with tempfile.TemporaryDirectory(dir=test_parent) as temporary:
        root = Path(temporary)
        for step, score in [(250, .8), (251, None), (500, .95), (500, None), (501, None), (750, .96)]:
            if score is not None:
                with (root / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, "solved_rate": score}) + "\n")
            wrapped(root / "checkpoint_last.pt", {"config": {"task": "maze"}, "step": step})
        directory = root / "threshold_90"
        assert torch.load(directory / "checkpoint_failing.pt", weights_only=False)["step"] == 250
        assert torch.load(directory / "checkpoint_passing.pt", weights_only=False)["step"] == 500
        assert json.loads((directory / "archive.json").read_text())["confirming_step"] == 750
    cwd = Path.cwd(); exact_resume = []
    try:
        with tempfile.TemporaryDirectory(dir=test_parent) as temporary:
            for task in ("maze", "sudoku"):
                c = Config(task=task, size=4, train_count=16, val_count=4, test_count=4,
                           width=16, layers=1, heads=4, hops=2, inner_beta=.1, inner_width=16,
                           steps=6, eval_every=2, batch_size=4)
                direct = Path(temporary) / (task + "_direct"); direct.mkdir(); os.chdir(direct)
                training.atomic_checkpoint = original; prepare(c); training.train(c)
                expected = copy.deepcopy(torch.load(c.run_dir() / "checkpoint_last.pt", weights_only=False))
                resumed = Path(temporary) / (task + "_resumed"); resumed.mkdir(); os.chdir(resumed)
                training.atomic_checkpoint = wrapped; prepare(c)
                training.train(c, stop_after=2)
                training.train(c, resume=c.run_dir() / "checkpoint_last.pt")
                actual = torch.load(c.run_dir() / "checkpoint_last.pt", weights_only=False)
                for name, value in expected["model"].items():
                    torch.testing.assert_close(value, actual["model"][name], rtol=0, atol=0)
                assert expected["data_rng"] == actual["data_rng"]
                assert expected["scheduler"] == actual["scheduler"]
                exact_resume.append(task)
    finally:
        os.chdir(cwd); training.atomic_checkpoint = original
    resumable = []
    for spec in json.loads((ROOT / "planned_mlp_final_queue.json").read_text()):
        c = Config(**spec); checkpoint = c.run_dir() / "checkpoint_last.pt"
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["config"] == c.dict()
        assert state["source_sha256"] == training.source_hash()
        assert state["data_hash"] == json.loads((c.data_dir() / "manifest.json").read_text())["data_hash"]
        assert all(key in state for key in ("optimizer", "scheduler", "data_rng", "torch_rng", "cuda_rng"))
        with tempfile.TemporaryDirectory(dir=test_parent) as temporary:
            target = Path(temporary) / "checkpoint_copy.pt"
            copy_snapshot(checkpoint, target)
            assert file_hash(checkpoint) == file_hash(target)
            assert torch.load(target, map_location="cpu", weights_only=False)["step"] == state["step"]
        resumable.append({"task": c.task, "method": c.method, "seed": c.seed,
                          "config_hash": c.hash(), "checkpoint_step": state["step"]})
    write_json(ROOT / "revision_audit" / "runtime_checks.json", {"status": "passed",
               "threshold_false_start_and_confirmation": True, "copied_snapshots_survive_atomic_replacement": True,
               "tested_on_project_filesystem": True, "resumable_checkpoints_verified": resumable,
               "off_grid_pause_preserves_threshold_bracket": True,
               "inner_mlp_resume_and_observation_hooks_bitwise_equal": exact_resume,
               "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
               "additional_source_hashes": {p: file_hash(p) for p in
                   ("scripts/audited_mlp_runtime.py", "scripts/revision_reporting.py")}})
    from revision_reporting import refresh
    refresh()
    print("AUDITED_RUNTIME_READY", flush=True)


def run():
    from transport_reasoning.data import file_hash
    amendment = json.loads((ROOT / "mlp_final_observation_amendment.json").read_text())
    for path, expected in amendment["additional_source_hashes"].items():
        if file_hash(path) != expected:
            raise RuntimeError("Observation extension source changed after verification")
    if json.loads((ROOT / "revision_audit" / "audit_result.json").read_text())["errors"]:
        raise RuntimeError("Scientific audit has unresolved errors")
    checks = json.loads((ROOT / "revision_audit" / "runtime_checks.json").read_text())
    if checks["status"] != "passed" or checks.get("additional_source_hashes") != amendment["additional_source_hashes"]:
        raise RuntimeError("Observation extension tests did not pass")
    training.atomic_checkpoint = checkpoint_wrapper(training.atomic_checkpoint)
    sys.argv = ["scripts/mlp_final_campaign.py", "run"]
    try:
        chunk_path = ROOT / "mlp_chunk_execution.json"
        if chunk_path.exists() and json.loads(chunk_path.read_text()).get("enabled"):
            from chunked_mlp_runtime import run as run_chunk
            run_chunk()
        else:
            runpy.run_path("scripts/mlp_final_campaign.py", run_name="__main__")
    finally:
        from revision_reporting import refresh
        refresh()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("verify", "run"))
    args = parser.parse_args()
    if args.mode == "verify":
        verify()
    else:
        run()
