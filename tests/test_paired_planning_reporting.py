"""Independent fixtures for task/source gates and fixed-seed paired inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_report_test", ROOT / "scripts/summarize_paired_planning.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(environment):
    return {"environment": environment, "action_block": 5, "episodes": [
        {"trajectory_id": f"test-s{seed}-d{d}", "seed": seed, "dynamics_id": d,
         "split": "test", "steps": 64, "sha256": f"episode-{seed}-{d}"}
        for seed in range(1000, 1064) for d in range(4)]}


def sign(record):
    protocol = record["planning"]["protocol"]
    record["planning"]["signature"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def make_record(root, mode, training_seed, split, source, manifest):
    data_sha = sha(root / "data/world/pusht_relative/manifest.json")
    evaluator_sha = sha(root / "src/shiftwm/evaluate.py")
    identity = {"checkpoint_sha256": sha(source / "model.pt"), "data_manifest_sha256": data_sha}
    protocol = {**identity, "run_identity": identity, "split": split, "episodes_per_dynamics": 64,
                "horizon": 5, "samples": 300, "iterations": 30, "elites": 30, "native_budget": 50,
                "goal_offset": 5, "seed": 1701, "policy": "world_model",
                "search_coordinates": reporter.aggregate.MAIN_PLANNER_IDENTITY["search_coordinates"],
                "evaluator_sha256": evaluator_sha}
    rows = []
    # Construct the population independently from the reporter's selection code.
    combinations = [(o, d) for o in range(3) for d in range(3)] if split == "test" else [(3, 0), (0, 3), (3, 3)]
    for seed in range(1000, 1064):
        for observation, dynamics in combinations:
            support = int(seed == 1000)
            success = int(support or (mode == "factorized" and seed % 4 <= training_seed)
                          or (mode != "factorized" and seed % 4 == 0))
            rows.append({"trajectory_id": f"test-s{seed}-d{dynamics}", "seed": seed,
                         "observation_id": observation, "dynamics_id": dynamics, "goal_index": 7,
                         "policy": "world_model", "success": success, "final_success": success,
                         "success_during_context": support, "policy_eligible": 1 - support,
                         "native_steps": 3 if support else 50, "num_replans": 0 if support else 8,
                         "initial_distance_after_context": 3.0 + seed,
                         "initial_block_translation_error_px": 1.0 + dynamics,
                         "initial_block_angle_error_rad": 0.2,
                         "initial_agent_position_error_px": 2.0 + seed})
    record = {"status": "complete", "environment": "pusht", "model_mode": mode,
              "training_seed": training_seed, "checkpoint_epoch": 2, **identity,
              "evaluator_sha256": evaluator_sha, "planning": {
                  "status": "complete", "kind": "closed_loop_goal_image_planning", "split": split,
                  "policy": "world_model", "protocol": protocol,
                  "planner": {**reporter.aggregate.MAIN_PLANNER_BUDGET, **reporter.aggregate.MAIN_PLANNER_IDENTITY},
                  "summary": {"all": {"success": {"mean": 0.99999}}}, "records": rows}}
    sign(record)
    return record


@pytest.fixture
def completed_tree(tmp_path):
    root = tmp_path
    (root / "src/shiftwm").mkdir(parents=True)
    (root / "src/shiftwm/evaluate.py").write_text("fixture evaluator; never executed")
    for environment, data_name in (("pusht", "pusht_relative"), ("reacher", "reacher")):
        write(root / "data/world" / data_name / "manifest.json", make_manifest(environment))
    manifest = make_manifest("pusht")
    data_sha = sha(root / "data/world/pusht_relative/manifest.json")
    stats = root / "action_stats.json"
    write(stats, {"mean": [0.0], "std": [1.0]})
    for mode in ("factorized", "framewise"):
        for seed in range(3):
            package = root / "runs/world" / f"pusht_{mode}_s{seed}" / "best"
            package.mkdir(parents=True)
            (package / "model.pt").write_bytes(f"opaque unpickled weights {mode} {seed}".encode())
            run = {"seed": seed, "model": {"mode": mode}, "epochs": 2}
            write(package.parent / "run_config.json", run)
            write(package.parent / "training_summary.json", {
                "status": "completed", "completed_epochs": 2, "best_validation_prediction_loss": 0.25})
            (package.parent / "metrics.jsonl").write_text('\n'.join(json.dumps({"epoch": e, "val": {"prediction_loss": v}})
                                                                   for e, v in ((1, 0.5), (2, 0.25))))
            write(package / "config.json", {"model_config": {"mode": mode, "history_length": 3},
                "metadata": {"config": run}, "base_config": {"dimension": 192}, "action_mean": [0.0] * 10,
                "action_std": [1.0] * 10, "provenance": {"data_manifest_sha256": data_sha,
                    "weights_sha256": "fixed upstream", "download": {"repo": "quentinll/lewm-pusht"},
                    "action_stats": str(stats), "action_stats_sha256": sha(stats)}})
            for split in ("test", "extrapolation"):
                write(root / "results/world" / f"pusht_{mode}_s{seed}" / f"planning_{split}.json",
                      make_record(root, mode, seed, split, package, manifest))
    return root


def result_path(root, mode="framewise", seed=2, split="test"):
    return root / "results/world" / f"pusht_{mode}_s{seed}" / f"planning_{split}.json"


def contrast(report, split="test", metric="raw_success"):
    return next(row for row in report["results"] if row["environment"] == "pusht"
                and row["comparator"] == "framewise" and row["split"] == split and row["metric"] == metric)


def mutate(path, fn):
    record = json.loads(path.read_text())
    fn(record)
    write(path, record)


def test_complete_counts_ignore_invented_summaries_and_use_sample_sd(completed_tree):
    report = reporter.build_report(completed_tree)
    assert len(report["results"]) == 24
    assert sum(row["status"] == "complete" for row in report["results"]) == 4
    measured = contrast(report)["metrics"]
    # The independent fixture has 0,16,32 additional wins out of64 by training seed.
    assert measured["mean_difference_pp"] == 25.0
    assert measured["training_seed_sd_pp"] == 25.0
    assert [row["wins"] for row in measured["per_training_seed"]] == [0, 16, 32]
    assert measured["n_tasks_per_training_seed"] == 64
    assert contrast(report, "extrapolation")["metrics"]["n_tasks_per_training_seed"] == 192
    eligible = contrast(report, metric="policy_eligible_success")["metrics"]
    assert eligible["n_tasks_per_training_seed"] == 63
    assert eligible["mean_difference_pp"] == pytest.approx(100 * 16 / 63)
    assert all(row["metrics"] is None for row in report["results"] if row["comparator"] != "framewise")


def test_missing_seed_suppresses_all_numerical_contrasts(completed_tree):
    result_path(completed_tree).unlink()
    report = reporter.build_report(completed_tree)
    for metric in ("raw_success", "policy_eligible_success"):
        row = contrast(report, metric=metric)
        assert row["status"] == "pending" and row["metrics"] is None
        assert row["availability"]["framewise"]["2"] == "missing"
    assert contrast(report, "extrapolation")["status"] == "complete"
    assert "--" in reporter.render_tex(report)


def test_realistic_interrupted_file_without_planner_remains_pending(completed_tree):
    def interrupt(record):
        record["status"] = record["planning"]["status"] = "interrupted"
        record["planning"].pop("planner")
        record["planning"]["records"] = record["planning"]["records"][:17]
    mutate(result_path(completed_tree), interrupt)
    report = reporter.build_report(completed_tree)
    assert contrast(report)["metrics"] is None
    assert contrast(report)["availability"]["framewise"]["2"] == "interrupted"


def test_malformed_present_partial_planner_is_rejected(completed_tree):
    def spoil(record):
        record["status"] = record["planning"]["status"] = "interrupted"
        record["planning"]["planner"] = None
    mutate(result_path(completed_tree), spoil)
    with pytest.raises(ValueError, match="malformed planner"):
        reporter.build_report(completed_tree)


@pytest.mark.parametrize("field,value", [("training_seed", 1), ("checkpoint_sha256", "other weights"),
                                          ("data_manifest_sha256", "other data"), ("evaluator_sha256", "other evaluator"),
                                          ("checkpoint_epoch", 1)])
def test_source_identity_and_validation_best_are_enforced(completed_tree, field, value):
    mutate(result_path(completed_tree), lambda row: row.__setitem__(field, value))
    with pytest.raises(ValueError):
        reporter.build_report(completed_tree)


@pytest.mark.parametrize("change", ["summary", "protocol", "coordinates", "partial_summary"])
def test_planner_budget_and_search_cannot_be_spoofed(completed_tree, change):
    def spoil(record):
        if change in ("summary", "partial_summary"):
            record["planning"]["planner"]["samples"] = 128
        if change == "partial_summary":
            record["status"] = record["planning"]["status"] = "interrupted"
        if change == "protocol":
            record["planning"]["protocol"]["iterations"] = 5
            sign(record)
        if change == "coordinates":
            record["planning"]["protocol"]["search_coordinates"] = "raw actions"
            sign(record)
    mutate(result_path(completed_tree), spoil)
    with pytest.raises(ValueError):
        reporter.build_report(completed_tree)


@pytest.mark.parametrize("change", ["missing", "duplicate", "substitute", "swap_seeds", "goal", "support", "distance"])
def test_exact_tasks_and_support_pairing_across_training_seeds(completed_tree, change):
    def spoil(record):
        rows = record["planning"]["records"]
        if change == "missing": rows.pop()
        if change == "duplicate": rows[-1] = copy.deepcopy(rows[0])
        if change == "substitute": rows[0]["trajectory_id"] = "test-s999999-d0"
        if change == "swap_seeds": rows[0]["seed"], rows[9]["seed"] = rows[9]["seed"], rows[0]["seed"]
        if change == "goal": rows[0]["goal_index"] = 8
        if change == "support":
            rows[0].update(success_during_context=0, policy_eligible=1, native_steps=50, num_replans=8)
        if change == "distance": rows[0]["initial_block_translation_error_px"] += 0.01
    mutate(result_path(completed_tree, mode="factorized"), spoil)
    with pytest.raises(ValueError):
        reporter.build_report(completed_tree)


def test_identical_checkpoint_cannot_masquerade_as_distinct_training_seeds(completed_tree):
    package0 = completed_tree / "runs/world/pusht_factorized_s0/best/model.pt"
    package1 = completed_tree / "runs/world/pusht_factorized_s1/best/model.pt"
    package1.write_bytes(package0.read_bytes())
    for split in ("test", "extrapolation"):
        def update(record):
            record["checkpoint_sha256"] = sha(package1)
            record["planning"]["protocol"]["run_identity"]["checkpoint_sha256"] = sha(package1)
            sign(record)
        mutate(result_path(completed_tree, "factorized", 1, split), update)
    with pytest.raises(ValueError, match="Identical checkpoint"):
        reporter.build_report(completed_tree)


@pytest.mark.parametrize("change", ["unfinished", "wrong_seed", "bad_stats", "bad_best"])
def test_completed_training_source_is_required(completed_tree, change):
    run = completed_tree / "runs/world/pusht_framewise_s2"
    if change == "unfinished":
        mutate(run / "training_summary.json", lambda row: row.__setitem__("status", "running"))
    if change == "wrong_seed":
        mutate(run / "run_config.json", lambda row: row.__setitem__("seed", 9))
    if change == "bad_stats":
        write(completed_tree / "action_stats.json", {"different": "statistics"})
    if change == "bad_best":
        mutate(run / "training_summary.json", lambda row: row.__setitem__("best_validation_prediction_loss", 0.1))
    with pytest.raises(ValueError):
        reporter.build_report(completed_tree)


def small_pairs():
    pairs = {}
    # Perfectly correlated across conditions AND training seeds: effective N=4,
    # never 36 independent task/model observations.
    for training_seed in range(3):
        left, right = {}, {}
        for trajectory_seed in range(4):
            for condition in range(3):
                key = (f"s{trajectory_seed}-d{condition}", condition)
                metadata = {"seed": trajectory_seed, "policy_eligible": 1}
                left[key] = {**metadata, "success": int(trajectory_seed == 0)}
                right[key] = {**metadata, "success": int(trajectory_seed in (1, 2))}
        pairs[training_seed] = left, right
    return pairs


def test_bootstrap_keeps_correlated_conditions_and_fixed_training_seeds_together():
    measured = reporter.paired_statistics(small_pairs())
    rng = np.random.default_rng(1701)
    draw_indices = rng.integers(0, 4, size=(20000, 4))
    # Independently compute each cluster's average difference; actual is -25pp.
    distribution = np.array([1, -1, -1, 0])[draw_indices].mean(axis=1) * 100
    assert measured["conditional_ci95_pp"] == list(np.quantile(distribution, [0.025, 0.975]))
    assert measured["mean_difference_pp"] == -25.0
    assert measured["training_seed_sd_pp"] == 0.0
    assert measured["n_initial_state_clusters"] == 4
    assert measured["per_training_seed"][0]["losses"] == 6
    assert reporter.paired_statistics(small_pairs()) == measured


def test_bootstrap_requires_three_seeds_and_retains_degenerate_warning():
    pairs = small_pairs()
    with pytest.raises(ValueError, match="all three"):
        reporter.paired_statistics({0: pairs[0], 1: pairs[1]})
    same = {seed: (left, copy.deepcopy(left)) for seed, (left, _) in pairs.items()}
    result = reporter.paired_statistics(same)
    assert result["conditional_ci95_pp"] == [0.0, 0.0] and result["degenerate_interval"]
    assert "does not establish equivalence" in reporter.UNCERTAINTY


def test_eligible_bootstrap_uses_task_counts_when_clusters_have_different_sizes():
    pairs = small_pairs()
    # Cluster0 has1 eligible condition, cluster1 has3, cluster2 has2, cluster3
    # contributes none. Every training seed retains the same eligibility mask.
    sizes = [1, 3, 2, 0]
    for left, right in pairs.values():
        for key in left:
            seed, condition = left[key]["seed"], key[1]
            left[key]["policy_eligible"] = right[key]["policy_eligible"] = int(condition < sizes[seed])
    measured = reporter.paired_statistics(pairs, eligible=True)
    draw_indices = np.random.default_rng(1701).integers(0, 3, size=(20000, 3))
    differences, counts = np.array([1, -3, -2]), np.array([1, 3, 2])
    distribution = 100 * differences[draw_indices].sum(axis=1) / counts[draw_indices].sum(axis=1)
    assert measured["conditional_ci95_pp"] == list(np.quantile(distribution, [0.025, 0.975]))
    assert measured["mean_difference_pp"] == pytest.approx(-100 * 4 / 6)
    assert measured["n_tasks_per_training_seed"] == 6
    assert measured["n_initial_state_clusters"] == 3


def test_hash_cache_invalidates_changed_weights(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"first")
    cache = reporter.campaign.FileDigestCache()
    first = cache.sha256(path)
    path.write_bytes(b"changed length")
    assert cache.sha256(path) != first


@pytest.mark.parametrize("change", ["duplicate", "missing_seed", "different_seed_set"])
def test_manifest_population_must_be_complete_and_crossed(change):
    manifest = make_manifest("pusht")
    if change == "duplicate": manifest["episodes"].append(copy.deepcopy(manifest["episodes"][0]))
    if change == "missing_seed": manifest["episodes"].pop(0)
    if change == "different_seed_set": manifest["episodes"][0]["seed"] = 2000
    with pytest.raises(ValueError):
        reporter.manifest_tasks(manifest, "test")
