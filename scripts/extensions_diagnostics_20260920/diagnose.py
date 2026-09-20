"""Fixed development-only observation geometry and saved tissue-trace audit.

No training, simulator execution, model selection, or final-test access. The
original geometry helper is extracted verbatim by AST, avoiding its top-level
execution and writes. All 12 already-selected packages use the same 16 goals.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.stats import spearmanr
import torch

OUT = Path("reports/extensions_diagnostics_20260920")
HERE = Path("scripts/extensions_diagnostics_20260920")
COMPLETE = "reports/completed_extension_results.json"
PROBE = "reports/evidence/extensions_mechanism_probe.json"
HELPER = "reports/evidence/extension_mechanism_probe.py"
MODES = ("constant_dynamics", "factorized")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(root, name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "Unsafe relative input")
    resolved = (root / p).resolve(strict=True)
    require(resolved.is_relative_to(root.resolve()), "Input escapes workspace")
    return root / p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never replace original reports, a prior completed result, or registration.
    encoded = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as f:
        f.write(encoded)


def validate_roster(rows):
    actual = [(r["family"], r["mode"], r["seed"]) for r in rows]
    expected = {(f, m, s) for f in ("narrow", "wide") for m in MODES for s in range(3)}
    require(len(actual) == 12 and set(actual) == expected, "Expected exact 12-package grid")
    require(all(r["architecture"] == "transformer" and r["domain"] == "drone"
                for r in rows), "Wrong domain or architecture")


def dev_episodes(manifest):
    episodes = sorted([e for e in manifest["episodes"] if e["split"] == "development"],
                      key=lambda e: (e["seed"], e["dynamics_id"]))
    require(len(episodes) == 48, "Expected 48 development trajectories")
    require(len({e["seed"] for e in episodes}) == 16, "Expected 16 development seeds")
    require(len({(e["seed"], e["dynamics_id"]) for e in episodes}) == 48
            and {e["dynamics_id"] for e in episodes} == {0, 1, 2}, "Wrong development grid")
    require(all("development" in e["file"] and "development" in e["audit_file"]
                for e in episodes), "Non-development payload path")
    return episodes


def prepare(root=ROOT):
    require(not (root / OUT / "registration.json").exists(), "Registration already exists")
    completed, probe = read(root / COMPLETE), read(root / PROBE)
    sources = {}

    def pin(name, expected=None):
        digest = sha(relative(root, name))
        require(expected is None or digest == expected, f"Source changed: {name}")
        sources[str(name)] = digest

    for name in (COMPLETE, PROBE, HELPER, "reports/evidence/extension_planning_probe.py",
                 "reports/geometry_revision_protocol.md", "reports/extensions_mechanism_diagnosis.md"):
        pin(name, probe["sources_sha256"].get(name))
    # Local model, loader, and immutable upstream implementation sources. No data
    # below unrelated studies is opened by this source-only recursive inventory.
    for path in sorted((root / "src/shiftwm").rglob("*.py")):
        pin(str(path.relative_to(root)))
    for name in ("src/shiftwm/vendor/lewm/NOTICE.json", "external/le-wm/module.py", "external/le-wm/jepa.py"):
        if (root / name).exists():
            pin(name)
    for path in sorted((root / HERE).iterdir()):
        if path.is_file():
            pin(str(path.relative_to(root)))
    for name in ("data/extensions/drone_v1/manifest.json", "data/features/drone_v1/manifest.json"):
        pin(name, probe["sources_sha256"][name])
    manifest = read(root / "data/extensions/drone_v1/manifest.json")
    episodes = dev_episodes(manifest)
    for ep in episodes:
        for prefix, key in (("data/features/drone_v1/", "file"), ("data/extensions/drone_v1/", "audit_file")):
            name = prefix + ep[key]
            pin(name, probe["sources_sha256"][name])
    models = []
    for family, group in (("narrow", "original_runs"), ("wide", "geometry_runs")):
        for row in completed[group]:
            if row["domain"] != "drone" or row["architecture"] != "transformer" or row["mode"] not in MODES:
                continue
            summary = row["training_summary"]
            require(summary["status"] == "completed" and summary["completed_epochs"] == 30,
                    "Checkpoint training is incomplete")
            package = row["packages"]["best"]
            directory = package["resolved_directory"]
            require(package["files"]["model.pt"] == row["checkpoint_sha256"], "Selected package differs")
            for name, digest in package["files"].items():
                pin(directory + "/" + name, digest)
            pin(directory + "/package_manifest.json", package["package_manifest_sha256"])
            config = read(root / directory / "config.json")
            require(config["extension_format_version"] == (1 if family == "narrow" else 2), "Wrong package family")
            require(config["mode"] == row["mode"] and config["architecture"] == "transformer", "Wrong model identity")
            models.append({"family": family, "mode": row["mode"], "seed": row["training_seed"],
                           "domain": "drone", "architecture": "transformer", "run_id": row["name"],
                           "package": directory, "checkpoint_sha256": row["checkpoint_sha256"],
                           "training_identity": summary["training_identity"], "completed_epochs": 30,
                           "checkpoint_selection": row["checkpoint_selection"]})
    validate_roster(models)
    tissue = []
    for row in completed["original_runs"]:
        if row["domain"] != "surgery":
            continue
        name = row["planning_result"]
        pin(name, completed["source_sha256"][name])
        result = read(root / name)
        require(result["status"] == "completed" and result["identity"]["arguments"]["split"] == "development",
                "Tissue result is not completed development")
        require(len(result["records"]) == result["tasks"] == 8, "Unexpected tissue population")
        for record in result["records"]:
            trace = Path(record["trace_file"])
            if not trace.is_absolute():
                trace = root / Path(name).parent / trace
            pin(str(trace.relative_to(root)), record["trace_sha256"])
        tissue.append({"run_id": row["name"], "architecture": row["architecture"], "mode": row["mode"],
                       "seed": row["training_seed"], "path": name})
    require(len(tissue) == 18 and len({r["run_id"] for r in tissue}) == 18, "Expected 18 tissue runs")
    registration = {"status": "registered_before_new_diagnostic", "schema_version": 1,
                    "scope": "Development-only descriptive diagnosis; no final test, training, retuning, or promotion",
                    "selection": {"support_windows": [0, 16], "appearance_id": 1, "goal_dynamics_id": 1,
                                  "goal_count": 16, "pair_count": 120, "fixed_support_contexts": 16,
                                  "reported_support_window": 0, "episodes": episodes},
                    "models": models, "tissue_runs": tissue, "sources_sha256": sources,
                    "resources": {"cpus": 2, "memory_gib": 8, "minutes": 15, "gpu": 0}}
    write_new(root / OUT / "registration.json", registration)
    return registration


def checked(root=ROOT, require_review=True):
    registration = read(root / OUT / "registration.json")
    validate_roster(registration["models"])
    if require_review:
        review = read(root / OUT / "source_review.json")
        require(review["status"] == "passed" and review["registration_sha256"] == sha(root / OUT / "registration.json"),
                "Independent review does not bind this registration")
    for name, expected in registration["sources_sha256"].items():
        require(sha(relative(root, name)) == expected, f"Changed registered input: {name}")
    return registration


def original_function(root, path, name, globals_):
    tree = ast.parse((root / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1 and not nodes[0].decorator_list, "Missing plain frozen helper")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = dict(globals_)
    exec(compile(module, str(root / path), "exec"), namespace)
    return namespace[name]


def geometry_helper(root, physical_pairs, n):
    require(n == 16 and np.shape(physical_pairs) == (120,), "Expected fixed 16-goal / 120-pair geometry")
    return original_function(root, HELPER, "geometry", {"np": np, "spearmanr": spearmanr,
                              "tri": torch.triu_indices(n, n, offset=1), "physical_pairs": physical_pairs})


def drone(registry, root=ROOT):
    from shiftwm.extensions.checkpoint import load_package
    from shiftwm.extensions.geometry_revision import load_geometry_package
    features_list, actions_list, goals, positions, identities = [], [], [], [], []
    for ep in registry["selection"]["episodes"]:
        with np.load(root / "data/features/drone_v1" / ep["file"], allow_pickle=False) as z:
            features, actions = torch.from_numpy(z["features"].copy()).float(), torch.from_numpy(z["actions"].copy()).float()
        require(features.shape == (4, 41, 192) and actions.shape == (40, 10), "Unexpected cache dimensions")
        require(bool(torch.isfinite(features).all() and torch.isfinite(actions).all()), "Nonfinite cached input")
        with np.load(root / "data/extensions/drone_v1" / ep["audit_file"], allow_pickle=False) as z:
            state, recorded = z["simulator_states"].copy(), z["actions"].copy()
        require(np.array_equal(actions.numpy(), recorded.reshape(40, 10)), "Native command/cache mismatch")
        for start in (0, 16):
            features_list.append(features[1, start:start + 3])
            actions_list.append(actions[start:start + 2])
            identities.append({"trajectory_id": ep["trajectory_id"], "dynamics_id": ep["dynamics_id"], "start": start})
        if ep["dynamics_id"] == 1:
            goals.append(features[:, -1])
            positions.append(state[-1, :2])
    support, past = torch.stack(features_list), torch.stack(actions_list)
    goals, positions = torch.stack(goals), torch.tensor(np.asarray(positions))
    tri = torch.triu_indices(16, 16, offset=1)
    physical = ((positions[:, None] - positions[None]).square().sum(-1))[tri[0], tri[1]].numpy()
    geometry = geometry_helper(root, physical, len(goals))
    selected = [i for i, row in enumerate(identities) if row["dynamics_id"] == 1 and row["start"] == 0]
    require(len(selected) == 16, "Wrong fixed-context selection")
    baseline = {"canonical": geometry(goals[:, 0]), "warm_uncorrected": geometry(goals[:, 1])}
    original = read(root / PROBE)
    for label in baseline:
        for key, value in baseline[label].items():
            require(math.isclose(value, original["goal_geometry"][label][key], rel_tol=1e-6, abs_tol=1e-8), "Original goal geometry mismatch")
    rows = []
    for entry in registry["models"]:
        started = time.monotonic()
        loader = load_package if entry["family"] == "narrow" else load_geometry_package
        model, state = loader(root / entry["package"], device="cpu")
        require(model.provenance["cache_manifest_sha256"] == sha(root / "data/features/drone_v1/manifest.json"), "Model cache provenance mismatch")
        require(state["config"]["metadata"]["training_identity"] == entry["training_identity"], "Training identity mismatch")
        obs, _ = model.infer_context(support, past)
        contexts = []
        for index in selected:
            context = obs[index]
            mapped = model.correct_observations(goals[:, 1, None], context[None].expand(16, -1))[:, 0]
            require(bool(torch.isfinite(mapped).all()), "Nonfinite corrected goals")
            # Recover the adapter's exact per-coordinate gain using its actual
            # forward method on 0 and 1; no reimplementation of either formula.
            gain = (model.observation_adapter(torch.ones(1, 1, 192), context[None])
                    - model.observation_adapter(torch.zeros(1, 1, 192), context[None]))[0, 0]
            pairs = ((mapped[:, None] - mapped[None]).square().mean(-1))[tri[0], tri[1]]
            contexts.append({"support": identities[index], **geometry(mapped),
                             "goal_calibration_mse_vs_canonical": float((mapped - goals[:, 0]).square().mean()),
                             "gain_min": float(gain.min()), "gain_mean": float(gain.mean()), "gain_max": float(gain.max()),
                             "pairwise_latent_mse": pairs.tolist()})
        aggregate = {key: float(np.mean([c[key] for c in contexts])) for key in
                     ("pairwise_latent_mse_mean", "spearman_with_squared_xy_distance", "goal_calibration_mse_vs_canonical")}
        aggregate["fraction_of_canonical_pairwise_mse"] = aggregate["pairwise_latent_mse_mean"] / baseline["canonical"]["pairwise_latent_mse_mean"]
        aggregate["gain_min"] = min(c["gain_min"] for c in contexts)
        aggregate["gain_max"] = max(c["gain_max"] for c in contexts)
        if entry["family"] == "narrow" and entry["seed"] == 0:
            old = next(r for r in original["rows"] if r["run_id"] == entry["run_id"])
            for key, value in old["goal_geometry_mean_over_fixed_support_contexts"].items():
                require(math.isclose(aggregate[key], value, rel_tol=1e-6, abs_tol=1e-8), "Original narrow seed0 diagnostic mismatch")
        row = {**entry, "selected_epoch": state["epoch"], "mean_over_fixed_contexts": aggregate,
               "contexts": contexts, "elapsed_seconds": time.monotonic() - started}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "contexts"}), flush=True)
        del model, state, obs
        gc.collect()
    pairs = []
    for mode in MODES:
        for seed in range(3):
            narrow, wide = [next(r for r in rows if r["family"] == f and r["mode"] == mode and r["seed"] == seed)
                            for f in ("narrow", "wide")]
            a, b = narrow["mean_over_fixed_contexts"], wide["mean_over_fixed_contexts"]
            pairs.append({"mode": mode, "seed": seed,
                          "wide_minus_narrow_pairwise_mse": b["pairwise_latent_mse_mean"] - a["pairwise_latent_mse_mean"],
                          "wide_minus_narrow_spearman": b["spearman_with_squared_xy_distance"] - a["spearman_with_squared_xy_distance"],
                          "wide_minus_narrow_calibration_mse": b["goal_calibration_mse_vs_canonical"] - a["goal_calibration_mse_vs_canonical"]})
    return {"scope": "Descriptive repeated-goal development diagnostic; contexts and 120 pairs are dependent, no confidence interval or performance claim",
            "goal_count": 16, "pair_count": 120, "fixed_support_contexts": 16,
            "physical_xy_squared_distances": physical.tolist(), "pair_indices": tri.T.tolist(),
            "baselines": baseline, "models": rows, "paired_wide_minus_narrow": pairs}


def tissue_audit(registry, root=ROOT):
    valid_success = original_function(root, "src/shiftwm/extensions/evaluate.py", "valid_success", {})
    rows = []
    counts = Counter()
    stops = Counter()
    for entry in registry["tissue_runs"]:
        result = read(root / entry["path"])
        run_counts, run_stops, decisions = Counter(), Counter(), []
        for record in result["records"]:
            metrics = record["metrics_per_native_step"]
            hits = [i + 1 for i, m in enumerate(metrics) if valid_success(m)]
            require(bool(hits) == record["success"], "Saved tissue success mismatch")
            require((hits[0] if hits else None) == record["first_success_native_step"], "First success index mismatch")
            run_counts["tasks"] += 1
            run_counts["successes"] += bool(hits)
            run_counts["failures"] += not bool(hits)
            run_counts["native_steps"] += len(metrics)
            run_counts["invalid_native_steps"] += sum(not m.get("valid_action", True) for m in metrics)
            run_counts["unstable_native_steps"] += sum(not m.get("stable_deformation", True) for m in metrics)
            if not hits:
                run_stops[record["stop_reason"]] += 1
                run_counts["failed_never_enter_2mm"] += not any(m["distance_m"] <= .002 for m in metrics)
                run_counts["failed_any_invalid"] += any(not m.get("valid_action", True) for m in metrics)
                run_counts["failed_any_unstable"] += any(not m.get("stable_deformation", True) for m in metrics)
            for d in record["decisions"]:
                before, after = d["native_start"], d["native_end"]
                require(after - before == len(d["executed_commands"]), "Decision command count mismatch")
                require(abs(metrics[before - 1]["distance_m"] - d["physical_distance_before_m"]) < 1e-10
                        and abs(metrics[after - 1]["distance_m"] - d["physical_distance_after_m"]) < 1e-10,
                        "Decision metric alignment mismatch")
                if after - before != 5:
                    continue
                progress = d["physical_distance_before_m"] - d["physical_distance_after_m"]
                run_counts["complete_five_command_decisions"] += 1
                run_counts["positive_progress_decisions"] += progress > 0
                decisions.append({"trajectory_id": record["trajectory_id"], "native_start": before,
                                  "predicted_next_goal_mse": d["predicted_next_goal_mse"],
                                  "distance_after_m": d["physical_distance_after_m"], "distance_progress_m": progress})
        require(run_counts["successes"] == result["successes"], "Saved aggregate success mismatch")
        counts.update(run_counts)
        stops.update(run_stops)
        rows.append({**entry, "counts": dict(run_counts), "failure_stops": dict(run_stops), "decisions": decisions})
    require(counts["tasks"] == 144 and len(rows) == 18, "Incomplete saved tissue audit")
    return {"scope": "Existing 18 development records, 8 shared goals each; descriptive chosen-action traces, not candidate ranking or independent trials",
            "alignment": "Predicted next transition paired with exactly five native commands; partial blocks excluded only from decision metrics, retained in failures",
            "counts": dict(counts), "failure_stops": dict(stops), "runs": rows}


def run(root=ROOT):
    started = time.monotonic()
    require(not (root / OUT / "results.json").exists(), "Diagnostic already completed")
    registry = checked(root)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with torch.inference_mode():
        result = {"status": "completed_development_diagnostic", "registration_sha256": sha(root / OUT / "registration.json"),
                  "job_id": os.environ.get("SLURM_JOB_ID"), "torch_version": torch.__version__,
                  "numpy_version": np.__version__, "threads": torch.get_num_threads(),
                  "drone": drone(registry, root), "tissue": tissue_audit(registry, root)}
    checked(root)
    result["elapsed_seconds"] = time.monotonic() - started
    write_new(root / OUT / "results.json", result)
    write_new(root / OUT / "completion.json", {"status": result["status"], "registration_sha256": result["registration_sha256"],
              "results_sha256": sha(root / OUT / "results.json"), "all_registered_inputs_unchanged": True,
              "job_id": result["job_id"], "elapsed_seconds": result["elapsed_seconds"]})
    print(json.dumps({k: v for k, v in result.items() if k not in ("drone", "tissue")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run"))
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"models": len(prepare()["models"]), "registration_sha256": sha(ROOT / OUT / "registration.json")}))
    elif args.command == "check":
        print(json.dumps({"status": "passed", "sources": len(checked()["sources_sha256"])}))
    else:
        run()
