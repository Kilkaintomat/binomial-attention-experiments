"""Independent audit of completed evidence and the frozen future MLP queue."""
import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from transport_reasoning.config import Config, ROOT, digest, write_json
from transport_reasoning.data import TaskData, file_hash
from transport_reasoning.filters import coefficients
from transport_reasoning.models import Block, Encoder
from transport_reasoning.training import source_hash

OUT = ROOT / "revision_audit"
ERRORS = []
WARNINGS = []


def read(path):
    return json.loads(Path(path).read_text())


def check(condition, code, detail=None):
    if not condition:
        ERRORS.append({"code": code, "detail": detail})


def table(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sudoku_legal(pred, clues):
    grid = pred.reshape(-1, 9, 9)
    boxes = grid.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    expected = np.arange(1, 10)[None, None, :]
    return ((np.sort(grid, axis=2) == expected).all((1, 2))
            & (np.sort(grid.transpose(0, 2, 1), axis=2) == expected).all((1, 2))
            & (np.sort(boxes, axis=2) == expected).all((1, 2))
            & ((clues == 0) | (pred == clues)).all(1))


def independent_maze(a, start, goal):
    n = len(a)
    edges = a.copy(); np.fill_diagonal(edges, False)
    check(np.array_equal(a, a.T) and np.diag(a).all(), "maze_graph_symmetry")
    check(int(edges.sum()) == 2 * (n - 1), "maze_tree_edge_count")
    previous = np.full(n, -2, dtype=int); previous[start] = -1
    frontier = [start]
    for vertex in frontier:
        for other in np.flatnonzero(edges[vertex]):
            if previous[other] == -2:
                previous[other] = vertex; frontier.append(int(other))
    check(len(frontier) == n, "maze_connected")
    path = np.zeros(n, dtype=bool); vertex = goal
    for _ in range(n):
        path[vertex] = True
        vertex = previous[vertex]
        if vertex < 0:
            break
    distances = np.zeros(n, dtype=int)
    for v in frontier[1:]:
        distances[v] = distances[previous[v]] + 1
    return path, distances[goal]


def canonical_maze(a, size):
    grid = np.arange(size * size).reshape(size, size)
    encodings = []
    for transposed in (False, True):
        base = grid.T if transposed else grid
        for flip_rows in (False, True):
            for flip_cols in (False, True):
                order = base[::-1] if flip_rows else base
                order = order[:, ::-1] if flip_cols else order
                indices = order.ravel()
                encodings.append(np.packbits(a[indices][:, indices]).tobytes())
    return hashlib.sha256(min(encodings)).hexdigest()


def audit_data(configs):
    cached, rows = {}, []
    for task in ("maze", "sudoku"):
        c = next(c for c in configs if c.task == task)
        manifest = read(c.data_dir() / "manifest.json")
        check(digest({k: v for k, v in manifest.items() if k != "data_hash"}) == manifest["data_hash"], "manifest_hash", task)
        seen, digit_seen = set(), set()
        for split in ("train", "validation", "test"):
            path = c.data_dir() / (split + ".npz")
            check(file_hash(path) == manifest["files"][path.name], "data_file_hash", task + "/" + split)
            data = dict(np.load(path, allow_pickle=False)); ids = data["ids"].tolist()
            check(len(ids) == manifest["splits"][split], "split_size", task + "/" + split)
            check(len(ids) == len(set(ids)), "within_split_duplicates", task + "/" + split)
            check(not (set(ids) & seen), "cross_split_duplicates", task + "/" + split)
            seen.update(ids)
            row = {"task": task, "split": split, "count": len(ids)}
            if task == "maze":
                paths, lengths = [], []
                for i, (a, start, goal) in enumerate(zip(data["a"], data["start"], data["goal"])):
                    independent, length = independent_maze(a, int(start), int(goal))
                    check(np.array_equal(independent, data["y"][i]), "maze_oracle_path", split + "/" + str(i))
                    check(canonical_maze(a, c.size) == ids[i], "maze_D4_identity", split + "/" + str(i))
                    paths.append(independent); lengths.append(length)
                data["independent_paths"] = np.array(paths)
                data["independent_lengths"] = np.array(lengths)
                row.update(path_length_mean=float(np.mean(lengths)),
                           path_longer_than_onehop_radius_4=float(np.mean(np.array(lengths) > 4)),
                           path_longer_than_recurrent_radius_32=float(np.mean(np.array(lengths) > 32)),
                           path_longer_than_final_multihop_radius_64=float(np.mean(np.array(lengths) > 64)))
                if split == "train":
                    positives = int(data["y"].sum())
                    check(abs(manifest["pos_weight"] - (data["y"].size - positives) / positives) < 1e-12, "maze_pos_weight")
            else:
                check(sudoku_legal(data["y"], data["puzzle"]).all(), "sudoku_all_targets_legal", split)
                digit_ids = set()
                for puzzle in data["puzzle"]:
                    mapping = {0: 0}; next_digit = 1; normalized = []
                    for v in puzzle:
                        v = int(v)
                        if v not in mapping:
                            mapping[v] = next_digit; next_digit += 1
                        normalized.append(mapping[v])
                    digit_ids.add(bytes(normalized))
                overlap = len(digit_ids & digit_seen)
                check(overlap == 0, "sudoku_digit_permutation_cross_split_duplicates", split)
                digit_seen.update(digit_ids)
                row.update(all_target_grids_valid=True, digit_relabeling_cross_split_duplicates=overlap,
                           full_row_column_transpose_symmetry_certificate=False,
                           every_puzzle_unique_solution_certified=False)
            cached[task, split] = data
            rows.append(row)
        # Alter labels BEFORE feature construction, not merely after batching.
        dataset = TaskData(c, "train"); before = dataset.batch(np.arange(4), "cpu")
        dataset.data["y"] = np.zeros_like(dataset.data["y"])
        after = dataset.batch(np.arange(4), "cpu")
        for key in ("x", "mask", "puzzle"):
            check(torch.equal(before[key], after[key]), "labels_enter_features", task + "/" + key)
    write_json(OUT / "data_checks.json", rows)
    print("AUDIT data complete", flush=True)
    return cached


def reference_block(block, x, mask):
    c = block.c; head_width = c.width // c.heads
    normalized = F.layer_norm(x, (c.width,), block.ln1.weight, block.ln1.bias, block.ln1.eps)
    q, k, v = F.linear(normalized, block.qkv.weight).chunk(3, dim=-1)
    qs, ks, zs = q.split(head_width, -1), k.split(head_width, -1), list(v.split(head_width, -1))
    routing = [torch.softmax(torch.einsum("bnd,bmd->bnm", qi, ki).div(math.sqrt(head_width))
                             .masked_fill(~mask, -torch.inf), -1) for qi, ki in zip(qs, ks)]
    weights = block.theta.softmax(0) if hasattr(block, "theta") else block.fixed_coeff
    contributions = []
    for weight in weights:
        transported = torch.cat([torch.einsum("bnm,bmd->bnd", a, z) for a, z in zip(routing, zs)], -1)
        if c.inner_beta:
            h = F.layer_norm(transported, (c.width,), block.inner_ln.weight, block.inner_ln.bias, block.inner_ln.eps)
            h = F.linear(F.gelu(F.linear(h, block.inner[0].weight, block.inner[0].bias)), block.inner[2].weight, block.inner[2].bias)
            transported = transported + c.inner_beta * h
        zs = list(transported.split(head_width, -1)); contributions.append(weight * transported)
    y = torch.stack(contributions).sum(0)
    residual = x + F.linear(y, block.out.weight)
    h = F.layer_norm(residual, (c.width,), block.ln2.weight, block.ln2.bias, block.ln2.eps)
    return residual + F.linear(F.gelu(F.linear(h, block.ff[0].weight, block.ff[0].bias)), block.ff[2].weight, block.ff[2].bias)


def audit_math():
    rows = []
    for hops in (1, 4, 8, 16):
        target = hops / (2 * (1 - 2. ** -hops))
        exact = np.array([math.comb(hops, s) / (2 ** hops - 1) for s in range(1, hops + 1)])
        check(np.allclose(coefficients("binomial", hops).numpy(), exact, atol=1e-14, rtol=1e-14), "binomial_coefficients", hops)
        for method in ("binomial", "exponential_truncated", "finite_geometric"):
            c = coefficients(method, hops).numpy(); s = np.arange(1, hops + 1)
            check(abs(float(c @ s) - target) < 1e-11, "mean_matching", [method, hops])
            check(abs(float(c.sum()) - 1) < 1e-14 and bool((c >= 0).all()), "coefficient_normalization", [method, hops])
    for method in ("binomial", "exponential_truncated", "finite_geometric", "learned_polynomial"):
        for beta in (0., .1):
            c = Config(width=16, heads=4, layers=1, hops=4, method=method, inner_beta=beta, inner_width=16)
            torch.manual_seed(491); block = Block(c, 0).double()
            x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
            mask = torch.eye(5, dtype=torch.bool).expand(2, -1, -1).clone()
            mask |= torch.roll(mask, 1, 2)
            a, b = block(x, mask), reference_block(block, x, mask)
            params = [x, *block.parameters()]
            ga = torch.autograd.grad(a.square().mean(), params, retain_graph=True)
            gb = torch.autograd.grad(b.square().mean(), params)
            value_error = float((a - b).abs().max())
            gradient_error = max(float((u - v).abs().max()) for u, v in zip(ga, gb))
            check(value_error < 1e-11 and gradient_error < 1e-10, "independent_MLP_reference", [method, beta, value_error, gradient_error])
            rows.append({"method": method, "inner_beta": beta, "value_max_error": value_error, "gradient_max_error": gradient_error})
    write_json(OUT / "math_checks.json", rows)
    print("AUDIT math complete", flush=True)


def audit_runs(configs, data):
    rows, coefficient_rows, sample_rows, threshold_rows = [], [], [], []
    rng_states = {}
    current_source = source_hash()
    source_mismatches = []
    summary = {r["config_hash"]: r for r in csv.DictReader((ROOT / "summaries/results.csv").open())}
    for index, c in enumerate(configs, 1):
        directory = c.run_dir(); r = read(directory / "result.json"); ev = read(directory / "evaluation_test.json")
        check(r["status"] == "completed" and r["step"] == 30000 and r["examples"] == 1920000, "run_completion", c.hash())
        check(r["config"] == c.dict() and r["config_hash"] == c.hash(), "run_config", c.hash())
        if r["source_sha256"] != current_source:
            source_mismatches.append(c.hash())
        curves = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        check([v["step"] for v in curves] == [1] + list(range(250, 30001, 250)), "validation_grid", c.hash())
        selected = max(curves, key=lambda v: (v["solved_rate"], -v["loss"]))
        best = torch.load(directory / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        last = torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        check(best["step"] == selected["step"] == ev["checkpoint_step"], "best_validation_checkpoint", c.hash())
        check(best["config"] == c.dict() == last["config"] and last["step"] == 30000, "checkpoint_identity", c.hash())
        check(best["source_sha256"] == r["source_sha256"] == last["source_sha256"], "checkpoint_source", c.hash())
        check(best["data_hash"] == r["data_hash"] == ev["data_hash"] == last["data_hash"], "checkpoint_data", c.hash())
        check(all(torch.isfinite(v).all() for v in last["model"].values()), "nonfinite_final_weights", c.hash())
        # Public data preserves its actual split sizes; config count fields are
        # generator defaults and are not the sampling upper bound for RRN.
        train_count = len(data[c.task, "train"]["ids"])
        key = (c.seed, train_count, c.steps, c.batch_size)
        if key not in rng_states:
            generator = np.random.default_rng(c.seed + 987123)
            for _ in range(c.steps):
                generator.integers(0, train_count, size=c.batch_size)
            rng_states[key] = generator.bit_generator.state
        check(last["data_rng"] == rng_states[key], "training_data_rng", c.hash())
        expected_lr = c.lr * .5 * (1 + math.cos(math.pi))
        check(abs(last["optimizer"]["param_groups"][0]["lr"] - expected_lr) < 1e-15, "final_cosine_lr", c.hash())
        for key in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng", "optimizer", "scheduler"):
            check(key in last, "resume_state_missing", [c.hash(), key])
        with np.load(directory / "predictions_test.npz", allow_pickle=False) as saved:
            pred = saved["predictions"]; test = data[c.task, "test"]
            check(np.array_equal(saved["ids"], test["ids"]), "prediction_ids", c.hash())
            if c.task == "maze":
                oracle = test["independent_paths"]
                degrees = np.einsum("bij,bj->bi", test["a"].astype(np.int16), pred.astype(np.int16)) - pred
                expected_degrees = np.full(pred.shape, 2, dtype=int)
                indices = np.arange(len(pred)); expected_degrees[indices, test["start"]] = 1; expected_degrees[indices, test["goal"]] = 1
                legal = pred[indices, test["start"]] & pred[indices, test["goal"]] & ((degrees == expected_degrees) | ~pred).all(1)
                exact = (pred == oracle).all(1)
                check(np.array_equal(legal, exact), "independent_maze_legality_vs_oracle", c.hash())
            else:
                legal = sudoku_legal(pred, test["puzzle"]); exact = (pred == test["y"]).all(1)
            score = float(legal.mean())
            check(np.array_equal(legal, saved["solved"]) and np.array_equal(exact, saved["exact"]), "per_example_metrics", c.hash())
            check(abs(score - ev["solved_rate"]) < 1e-12 and abs(float(exact.mean()) - ev["exact_match"]) < 1e-12, "evaluation_aggregate", c.hash())
            check(abs(score - float(summary[c.hash()]["solved_rate"])) < 1e-12, "summary_solved_rate", c.hash())
            sample_pred = pred[:16].copy()
        model = Encoder(c); model.load_state_dict(best["model"]); model.eval()
        with torch.no_grad():
            dataset = TaskData(c, "test"); batch = dataset.batch(np.arange(16), "cpu"); logits = model(batch)
            if c.task == "maze":
                replay = (logits.squeeze(-1) >= 0).numpy()
            else:
                replay = logits.argmax(-1).numpy() + 1
                replay = np.where(batch["puzzle"].numpy() != 0, batch["puzzle"].numpy(), replay)
        mismatches = int(np.count_nonzero(replay != sample_pred))
        check(mismatches == 0, "checkpoint_CPU_prediction_replay", [c.hash(), mismatches])
        sample_rows.append({"config_hash": c.hash(), "task": c.task, "method": c.method, "seed": c.seed,
                           "examples_replayed": 16, "cell_mismatches": mismatches})
        filt = best["filters"][0]
        reported_mu = float(summary[c.hash()]["mu"]); reported_nu = float(summary[c.hash()]["nu"])
        if abs(reported_mu - filt["mu"]) > 1e-7 or abs(reported_nu - filt["nu"]) > 1e-7:
            coefficient_rows.append({"config_hash": c.hash(), "task": c.task, "method": c.method, "seed": c.seed,
                                     "tested_checkpoint_step": best["step"], "reported_last_mu": reported_mu,
                                     "tested_best_mu": filt["mu"], "reported_last_nu": reported_nu, "tested_best_nu": filt["nu"]})
        rows.append({"task": c.task, "method": c.method, "seed": c.seed, "inner_beta": c.inner_beta,
                     "config_hash": c.hash(), "data_hash": r["data_hash"], "source_sha256": r["source_sha256"],
                     "test_count": len(pred), "solved_rate": score, "exact_match": float(exact.mean()),
                     "best_checkpoint_step": best["step"], "validation_solved_at_best": selected["solved_rate"],
                     "test_mu_best_checkpoint": filt["mu"], "test_nu_best_checkpoint": filt["nu"],
                     "training_seconds": r["training_seconds"], "parameter_count": r["parameter_count"],
                     "graph_radius": c.recurrent_steps if c.method == "recurrent" else c.layers * (1 if c.method == "onehop" else c.hops),
                     "LR": c.lr, "verified": True})
        for threshold in ((.5, .9) if c.task == "sudoku" else (.9,)):
            reached = next((i for i in range(1, len(curves)) if curves[i - 1]["solved_rate"] >= threshold and curves[i]["solved_rate"] >= threshold), None)
            if reached is not None:
                failing = curves[reached - 2]["step"] if reached >= 2 else 0
                passing = curves[reached - 1]["step"]
                stored_steps = {best["step"], last["step"]}
                threshold_rows.append({"config_hash": c.hash(), "threshold": threshold, "last_failing_step": failing,
                                       "first_passing_step": passing, "failing_weights_retained": failing in stored_steps,
                                       "passing_weights_retained": passing in stored_steps})
        del model, best, last, dataset
        print("AUDIT completed run {}/60 {} {} s{}".format(index, c.task, c.method, c.seed), flush=True)
    check(not source_mismatches, "completed_source_mismatches", source_mismatches)
    table("verified_final_results.csv", rows); table("coefficient_metadata_errors.csv", coefficient_rows)
    table("checkpoint_replay.csv", sample_rows); table("threshold_checkpoint_audit.csv", threshold_rows)
    if coefficient_rows:
        WARNINGS.append({"code": "learned_filter_metadata_from_last_not_tested_best", "affected_runs": len(coefficient_rows)})
    if threshold_rows:
        WARNINGS.append({"code": "threshold_weights_not_retained", "reached_thresholds": len(threshold_rows),
                         "both_weights_retained": sum(r["failing_weights_retained"] and r["passing_weights_retained"] for r in threshold_rows),
                         "logged_metric_intervals_remain_available": True})
    aggregates = []
    for task in ("maze", "sudoku"):
        for method in sorted({c.method for c in configs}):
            selected = [r for r in rows if r["task"] == task and r["method"] == method]
            values = [r["solved_rate"] for r in selected]
            aggregates.append({"task": task, "method": method, "seeds": json.dumps([r["seed"] for r in selected]),
                               "per_seed": json.dumps(values), "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)})
    table("verified_final_aggregates.csv", aggregates)
    return rows


def audit_future():
    protocol = read(ROOT / "mlp_final_protocol.json")
    queue = [Config(**d).validate() for d in read(ROOT / "planned_mlp_final_queue.json")]
    check(len(queue) == 20, "future_queue_count")
    expected = {(t, m, s) for t in ("maze", "sudoku") for m in ("binomial", "exponential_truncated") for s in (11, 23, 37, 53, 71)}
    check({(c.task, c.method, c.seed) for c in queue} == expected, "future_queue_coverage")
    check(protocol["source_sha256"] == source_hash(), "future_source_hash")
    check(protocol["runner_sha256"] == file_hash("scripts/mlp_final_campaign.py"), "future_runner_hash")
    for c in queue:
        base = Config(**protocol["baseline_configs_by_new_hash"][c.hash()])
        check(c.dict() == dict(base.dict(), inner_beta=.1, inner_width=128, max_seconds=4800), "future_pair_mismatch", c.hash())
        baseline, inner = Encoder(base), Encoder(c)
        for name, value in baseline.state_dict().items():
            check(torch.equal(value, inner.state_dict()[name]), "future_common_init", [c.hash(), name])
        check(c.steps == 30000 and c.batch_size == 64 and c.stage == "final", "future_training_budget", c.hash())
        check(c.hops == {"maze": 16, "sudoku": 4}[c.task], "future_horizon", c.hash())
    write_json(OUT / "future_queue_checks.json", {"runs": len(queue), "common_parameters_exactly_paired": True,
               "updates": 30000, "batch_size": 64, "inner_beta": .1, "inner_width": 128,
               "protocol_unchanged": True, "not_a_parameter_matched_architecture_ablation": True})


def audit_all_recorded_metadata():
    counts = {}; rows = []
    for path in sorted((ROOT / "runs").rglob("result.json")):
        r = read(path); c = Config(**r["config"]); key = c.stage + "/" + r["status"]
        counts[key] = counts.get(key, 0) + 1
        curves = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
        check(r["config_hash"] == c.hash() and read(path.parent / "config.json") == c.dict(), "all_run_config", c.hash())
        check(r["step"] == curves[-1]["step"], "all_run_step", c.hash())
        check(r["status"] != "completed" or r["step"] == c.steps, "all_run_completion", c.hash())
        check(all(math.isfinite(v[k]) for v in curves for k in ("loss", "train_loss", "gradient_norm", "solved_rate")), "all_run_finite_logs", c.hash())
        best = torch.load(path.parent / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        selected = max(curves, key=lambda v: (v["solved_rate"], -v["loss"]))
        check(best["step"] == selected["step"], "all_run_best_selection", c.hash())
        check(best["source_sha256"] == r["source_sha256"], "all_run_recorded_source", c.hash())
        check(best["config"] == c.dict(), "all_run_checkpoint_config", c.hash())
        actual = read(c.data_dir() / "manifest.json")["splits"]
        rows.append({"config_hash": c.hash(), "task": c.task, "stage": c.stage, "method": c.method,
                     "seed": c.seed, "inner_beta": c.inner_beta, "status": r["status"], "step": r["step"],
                     "actual_train_count": c.overfit_count or actual["train"],
                     "actual_validation_count": c.overfit_count or actual["validation"],
                     "configuration_train_count": c.train_count,
                     "evaluation_split": "train_subset" if c.overfit_count else "validation"})
    pointer = list(csv.DictReader((ROOT / "summaries/pointer_bridge.csv").open()))
    check(len(pointer) == 36, "pointer_bridge_row_count")
    check(all(r["implementation"] == "reimplementation" for r in pointer), "pointer_bridge_label")
    for r in pointer:
        for key in ("conditional_accuracy", "raw_correct_class_mass", "terminal_mass", "unreached_accuracy"):
            check(0 <= float(r[key]) <= 1 + 1e-6, "pointer_metric_range", [r["seed"], r["method"], key])
    table("all_recorded_runs.csv", rows)
    write_json(OUT / "all_recorded_metadata.json", {"run_count": len(rows), "stage_status_counts": counts,
               "pointer_bridge_rows": len(pointer), "pointer_training_not_repeated": True})
    WARNINGS.append({"code": "RRN_config_count_fields_are_unused_generator_defaults",
                     "actual_splits": [180000, 18000, 18000], "config_fields": [100000, 5000, 5000],
                     "sampling_checked_against_actual_data_size": True})


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Audit requires a scheduler allocation")
    OUT.mkdir(exist_ok=True); torch.set_num_threads(4); started = time.perf_counter()
    configs = [Config(**d).validate() for d in read(ROOT / "planned_final_queue.json")]
    check(len(configs) == 60, "primary_final_count")
    audit_math(); data = audit_data(configs); rows = audit_runs(configs, data); audit_future(); audit_all_recorded_metadata()
    result = {"status": "passed_with_limitations" if not ERRORS else "failed", "errors": ERRORS, "warnings": WARNINGS,
              "completed_runs_audited": len(rows), "saved_test_predictions_revalidated": sum(r["test_count"] for r in rows),
              "checkpoint_replay_examples": 16 * len(rows), "future_runs_checked": 20,
              "elapsed_seconds": time.perf_counter() - started, "scheduler_job_id": os.environ["SLURM_JOB_ID"],
              "source_sha256": source_hash(), "audit_script_sha256": file_hash(__file__),
              "limitations": ["Full test inference was not repeated; every saved prediction was independently validated",
                              "CPU checkpoint replay uses 16 test examples per completed run",
                              "Sudoku digit relabeling was checked; full row/column/transpose canonicalization and full uniqueness proofs were not performed",
                              "Protocol omissions and reporting defects are documented separately; passing this audit is not full plan completion"]}
    write_json(OUT / "audit_result.json", result)
    print(json.dumps(result), flush=True)
    if ERRORS:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        write_json(OUT / "audit_failure.json", {"error_type": type(exc).__name__,
                   "message": str(exc).replace(str(Path.cwd()), ".").replace(str(Path.home()), "~")})
        raise
