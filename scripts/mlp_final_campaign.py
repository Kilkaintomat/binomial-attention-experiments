"""Frozen five-seed inner-MLP extension of the completed primary campaign."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("experiments/transport_reasoning")
METHODS = ("binomial", "exponential_truncated")
SEEDS = (11, 23, 37, 53, 71)
RESERVATION_SECONDS = 33300  # Also enforced by a 09:15:00 scheduler limit.


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configs_and_costs():
    from transport_reasoning.config import Config

    original = read(ROOT / "planned_final_queue.json")
    by_key = {(c["task"], c["method"], c["seed"]): c for c in original}
    pilots = read(ROOT / "planned_mlp_queue.json")
    costs = {}
    for task in ("maze", "sudoku"):
        for method in METHODS:
            seconds = []
            for d in pilots:
                if (d["task"], d["method"], d["inner_beta"]) != (task, method, .1):
                    continue
                c = Config(**d).validate()
                r = read(c.run_dir() / "result.json")
                if r["status"] != "completed":
                    raise RuntimeError("Required MLP pilot is incomplete")
                evaluations = len((c.run_dir() / "metrics.jsonl").read_text().splitlines())
                seconds.append(r["training_seconds"] / r["step"] * 30000
                               + r["evaluation_seconds"] / evaluations * 122
                               + r["checkpoint_seconds"] / evaluations * 121 + 5)
            if len(seconds) != 2:
                raise RuntimeError("Expected two measured pilot seeds")
            costs[task + "/" + method] = statistics.mean(seconds)
    configs, controls = [], {}
    for seed in SEEDS:
        for task in ("maze", "sudoku"):
            for method in METHODS:
                base = Config(**by_key[task, method, seed]).validate()
                d = base.dict()
                d.update(inner_beta=.1, inner_width=128, max_seconds=4800)
                c = Config(**d).validate()
                configs.append(c)
                controls[c.hash()] = base.dict()
    return configs, controls, costs


def preflight():
    import numpy as np
    import torch
    from transport_reasoning.config import Config, write_json
    from transport_reasoning.data import TaskData, file_hash
    from transport_reasoning.evaluation import loss_for
    from transport_reasoning.models import Encoder
    from transport_reasoning.training import source_hash

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Preflight requires a CPU scheduler allocation")
    if (ROOT / "mlp_final_protocol.json").exists():
        raise RuntimeError("Extension protocol already exists; do not overwrite it")
    subprocess.run([sys.executable, "-m", "pytest", "-q",
                    "--junitxml=experiments/transport_reasoning/mlp_final_tests.xml"], check=True)
    torch.set_num_threads(4)
    configs, controls, costs = configs_and_costs()
    source = source_hash()
    data_manifests = {}
    audits = []
    for c in configs:
        base = Config(**controls[c.hash()])
        old = read(base.run_dir() / "result.json")
        if old["status"] != "completed" or old["step"] != 30000:
            raise RuntimeError("A paired baseline is incomplete")
        if old["source_sha256"] != source:
            raise RuntimeError("Model/training source changed since paired baseline")
        if not (base.run_dir() / "evaluation_test.json").exists():
            raise RuntimeError("A paired baseline lacks its frozen test evaluation")
        if c.data_spec() != base.data_spec():
            raise RuntimeError("Data differs between a paired baseline and MLP run")
        if c.task in data_manifests:
            continue
        manifest = read(c.data_dir() / "manifest.json")
        for filename, checksum in manifest["files"].items():
            if file_hash(c.data_dir() / filename) != checksum:
                raise RuntimeError("Dataset checksum mismatch")
        if old["data_hash"] != manifest["data_hash"]:
            raise RuntimeError("Data changed since paired baseline")
        data_manifests[c.task] = manifest
        data = TaskData(c, "train")
        batch = data.batch(np.arange(4), "cpu")
        baseline, inner = Encoder(base), Encoder(c)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, inner.state_dict()[name], rtol=0, atol=0)
        c.inner_beta = 0
        torch.testing.assert_close(baseline(batch), inner(batch), rtol=0, atol=0)
        c.inner_beta = .1
        output = inner(batch)
        changed_labels = dict(batch, y=torch.flip(batch["y"], [0]))
        torch.testing.assert_close(output, inner(changed_labels), rtol=0, atol=0)
        loss = loss_for(output, batch, c, manifest.get("pos_weight", 1.))
        loss.backward()
        for name, p in inner.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError("Invalid MLP gradient: " + name)
        audits.append({"task": c.task, "beta_zero_identity": True,
                       "common_initialization_exact": True, "all_gradients_finite": True,
                       "labels_do_not_affect_forward": True,
                       "baseline_parameters": baseline.parameter_count(),
                       "inner_mlp_parameters": inner.parameter_count()})
    budget = read(ROOT / "budget.json")
    charged = sum(a.get("actual_seconds", a["reserved_seconds"]) for a in budget["allocations"])
    estimated = sum(costs[c.task + "/" + c.method] for c in configs)
    available = min(86400 - charged, RESERVATION_SECONDS)
    if estimated * 1.10 + 300 > available:
        raise RuntimeError("Measured estimate plus margin exceeds remaining GPU budget")
    if any(c.run_dir().exists() for c in configs):
        raise RuntimeError("An extension run already exists before protocol freeze")
    protocol = {
        "name": "inner_mlp_final_v1", "created_utc_epoch": time.time(),
        "source_sha256": source, "runner_sha256": sha(__file__),
        "primary_protocol_sha256": sha(ROOT / "frozen_protocol.json"),
        "frozen_before_new_mlp_training_or_test": True,
        "prior_beta_zero_final_results_already_observed": True,
        "selection_basis": "External plan priority: binomial and exponential; pilot runtime; no new tuning",
        "methods": list(METHODS), "seeds": list(SEEDS), "tasks": ["maze", "sudoku"],
        "run_count": 20, "updates_per_run": 30000, "batch_size": 64,
        "inner_beta": .1, "inner_width": 128,
        "common_H": {"maze": 16, "sudoku": 4},
        "architecture": "Fixed attention per block; shared inner GELU FFN across hops; cross-head mixing; outer FFN retained",
        "learning_rates": "Unchanged paired primary configs; maze binomial 0.0003, other included configs 0.001",
        "initialization_and_data_order": "Same named initialization streams and training-data RNG as paired controls; fresh training from step zero",
        "checkpoint_rule": "maximum validation solved rate; ties minimum validation loss; earliest exact tie",
        "validation_every_updates": 250,
        "thresholds": {"maze": [.9], "sudoku": [.5, .9]}, "threshold_confirmations": 2,
        "practical_effect_solved_fraction": .02, "equivalence_claim_authorized": False,
        "data_manifests": data_manifests, "baseline_configs_by_new_hash": controls,
        "estimated_seconds_by_task_method": costs, "estimated_total_seconds": estimated,
        "budget_remaining_before_extension_seconds": 86400 - charged,
        "scheduler_reservation_seconds": RESERVATION_SECONDS, "cumulative_ceiling_gpu_hours": 24,
        "runtime_policy": "Save full-state checkpoints before deadline; no seed dropping, update shortening or tuning after freeze",
        "comparison_limits": ["On/off parameter counts and FFN compute differ",
                              "LR differs between the two maze filters; this is not a coefficient-only causal comparison",
                              "Existing recurrent runs have a fixed update budget, not a newly matched wall-time schedule"],
        "deferred": ["inner-MLP geometric and learned-polynomial final series",
                     "outer-only parameter-matched MLP control",
                     "dedicated equal-wall-time recurrent retraining"],
    }
    for c in configs:
        write_json(ROOT / "configs" / (c.hash() + ".json"), c.dict())
    write_json(ROOT / "planned_mlp_final_queue.json", [c.dict() for c in configs])
    write_json(ROOT / "mlp_final_preflight.json", {"status": "passed", "audits": audits,
               "tests_xml": "mlp_final_tests.xml", "source_sha256": source,
               "scheduler_job_id": os.environ["SLURM_JOB_ID"]})
    write_json(ROOT / "mlp_final_protocol.json", protocol)
    print(json.dumps({"status": "ready", "runs": len(configs), "estimated_gpu_hours": estimated / 3600}), flush=True)


def extension_summary(configs, protocol):
    from transport_reasoning.config import Config, write_json
    rows = []
    for c in configs:
        result_path = c.run_dir() / "result.json"
        evaluation_path = c.run_dir() / "evaluation_test.json"
        if not result_path.exists() or not evaluation_path.exists():
            continue
        r = read(result_path)
        if r["status"] != "completed":
            continue
        ev = read(evaluation_path)
        base = Config(**protocol["baseline_configs_by_new_hash"][c.hash()])
        old, old_ev = read(base.run_dir() / "result.json"), read(base.run_dir() / "evaluation_test.json")
        rows.append({"task": c.task, "method": c.method, "seed": c.seed,
                     "config_hash": c.hash(), "baseline_hash": base.hash(), "updates": r["step"],
                     "inner_test_solved": ev["solved_rate"], "baseline_test_solved": old_ev["solved_rate"],
                     "paired_difference": ev["solved_rate"] - old_ev["solved_rate"],
                     "inner_training_seconds": r["training_seconds"],
                     "baseline_training_seconds": old["training_seconds"],
                     "training_time_ratio": r["training_seconds"] / old["training_seconds"]})
    path = ROOT / "summaries" / "mlp_final_pairs.csv"
    path.parent.mkdir(exist_ok=True)
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    state = {"completed": len(rows), "expected": len(configs), "completed_hashes": [r["config_hash"] for r in rows],
             "remaining_configs": [c.dict() for c in configs if c.hash() not in {r["config_hash"] for r in rows}]}
    write_json(ROOT / "mlp_final_state.json", state)
    return state


def run():
    import torch
    from transport_reasoning.config import Config, write_json
    from transport_reasoning.training import train, evaluate_checkpoint, environment, source_hash
    from transport_reasoning.reporting import summarize

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one scheduler-allocated GPU is required")
    protocol = read(ROOT / "mlp_final_protocol.json")
    if read(ROOT / "mlp_final_preflight.json")["status"] != "passed":
        raise RuntimeError("Preflight did not pass")
    if protocol["source_sha256"] != source_hash() or protocol["runner_sha256"] != sha(__file__):
        raise RuntimeError("Source changed after extension protocol freeze")
    configs = [Config(**d).validate() for d in read(ROOT / "planned_mlp_final_queue.json")]
    if len(configs) != 20 or {(c.task, c.method, c.seed) for c in configs} != {
            (t, m, s) for t in ("maze", "sudoku") for m in METHODS for s in SEEDS}:
        raise RuntimeError("Final queue coverage mismatch")
    for c in configs:
        base = protocol["baseline_configs_by_new_hash"][c.hash()]
        expected = dict(base, inner_beta=.1, inner_width=128, max_seconds=4800)
        if c.dict() != expected:
            raise RuntimeError("Final queue differs from frozen paired protocol")
    with (ROOT / "campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = read(ROOT / "budget.json")
        charged = sum(a.get("actual_seconds", a["reserved_seconds"]) for a in budget["allocations"])
        available = min(86400 - charged, RESERVATION_SECONDS)
        if available <= 300:
            raise RuntimeError("Insufficient remaining cumulative GPU budget")
        start = time.perf_counter()
        deadline = start + available - 120
        allocation = {"job_id": os.environ["SLURM_JOB_ID"], "gpu_count": 1,
                      "started_utc_epoch": time.time(), "reserved_seconds": available,
                      "status": "active", "purpose": "inner_mlp_final_v1"}
        budget["allocations"].append(allocation)
        write_json(ROOT / "budget.json", budget)
        write_json(ROOT / "mlp_final_environment.json", environment())

        def status(message, current=None):
            allocation["observed_seconds"] = time.perf_counter() - start
            budget["observed_gpu_seconds"] = charged + allocation["observed_seconds"]
            write_json(ROOT / "budget.json", budget)
            event = {"message": message, "job_id": os.environ["SLURM_JOB_ID"],
                     "elapsed_seconds": allocation["observed_seconds"], "current": current}
            write_json(ROOT / "mlp_final_progress.json", event)
            (ROOT / "MLP_FINAL_STATUS.md").write_text(
                "# Inner-MLP final series\n\n" + message + "\n\n"
                + "20 planned runs; two tasks, two filters, five seeds; 30000 updates each.\n",
                encoding="utf-8")
            print(json.dumps(event), flush=True)

        try:
            status("Final series started")
            for i, c in enumerate(configs, 1):
                if time.perf_counter() >= deadline - 180:
                    break
                current = {"index": i, "total": len(configs), "task": c.task,
                           "method": c.method, "seed": c.seed, "config_hash": c.hash()}
                status("Training final run", current)
                checkpoint = c.run_dir() / "checkpoint_last.pt"
                result = train(c, resume=checkpoint if checkpoint.exists() else None, deadline=deadline)
                event = dict(current, stage="final", series="inner_mlp_final_v1", inner_beta=.1,
                             status=result["status"], step=result["step"])
                with (ROOT / "run_registry.jsonl").open("a") as f:
                    f.write(json.dumps(event) + "\n")
                if result["status"] == "completed":
                    if not (c.run_dir() / "evaluation_test.json").exists():
                        evaluate_checkpoint(c, c.run_dir() / "checkpoint_best.pt", split="test")
                state = extension_summary(configs, protocol)
                status("Completed {}/{} final runs".format(state["completed"], len(configs)), event)
                if result["status"] != "completed":
                    break
            state = extension_summary(configs, protocol)
            outcome = dict(state, status="completed" if state["completed"] == len(configs) else "budget_paused")
            write_json(ROOT / "mlp_final_result.json", outcome)
            summarize()
            status("{}: {}/{} final runs".format(outcome["status"], state["completed"], len(configs)))
        except Exception as exc:
            write_json(ROOT / "mlp_final_failure.json", {"error_type": type(exc).__name__,
                       "message": str(exc).replace(str(Path.cwd()), ".").replace(str(Path.home()), "~")})
            status("Stopped: " + type(exc).__name__)
            raise
        finally:
            allocation.update(actual_seconds=time.perf_counter() - start, status="finished")
            budget["used_gpu_seconds"] = sum(a.get("actual_seconds", a["reserved_seconds"]) for a in budget["allocations"])
            write_json(ROOT / "budget.json", budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight()
    else:
        run()
