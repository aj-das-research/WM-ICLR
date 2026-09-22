#!/usr/bin/env python3
"""Freeze the complete checkpoint/data/metric roster before new DROID scoring."""
import argparse
from pathlib import Path

import numpy as np

from droid_common import (ROOT, REPORT, REG, METRICS, POLICY, atomic_json, local,
    module, now, population, read, relative, require, sha, verify_registration)

CAMPAIGNS = (
    ("internal", "configs/real_video_spatial/v1/registration.json", "scripts/real_video_spatial/train.py",
     "reports/real_video_spatial", "reports/real_video_spatial/finalization.json"),
    ("component", "configs/real_video_spatial_components/v1/registration.json", "scripts/real_video_spatial_components/train.py",
     "reports/real_video_spatial_components", "reports/real_video_spatial_components/finalization.json"),
    ("external_v1", "reports/external_dinowm_train_v1/registration.json", "scripts/external_dinowm_train_v1/train.py",
     "reports/external_dinowm_train_v1", "reports/external_dinowm_reporting_v1/finalization.json"),
    ("external_raw_v2", "reports/external_dinowm_raw_v2/registration.json", "scripts/external_dinowm_raw_v2/train.py",
     "reports/external_dinowm_raw_v2", "reports/external_dinowm_raw_reporting_v2/finalization.json"),
)


def register():
    require(not REG.exists(), "Registration already exists; do not alter after scoring")
    sources = {}
    def bind(path, expected=None):
        p = local(path); value = sha(p)
        require(expected is None or value == expected, "Upstream binding changed: " + str(path))
        sources[relative(p)] = value
        return p

    for path in sorted(Path(__file__).parent.glob("droid_*")):
        if path.is_file():
            bind(path)
    bind(REPORT / "protocol.md")
    bind("scripts/real_video_iws/evaluate.py")
    cache = ROOT / "data/features/droid_spatial_v1"
    manifest = read(bind(cache / "manifest.json"))
    stats = read(bind(cache / "training_statistics.json"))
    old_manifest = read(bind("data/features/droid_selected_v1/manifest.json"))
    bind("data/features/droid_selected_v1/training_statistics.json")
    require(stats["fit_split"] == "train" and stats["cache_manifest_sha256"] == sha(cache / "manifest.json"),
            "Feature normalization was not fitted on the registered training cache")
    keys = population(manifest, old_manifest)
    for ep in manifest["episodes"]:
        if ep["split"] == "val":
            payload = ep["cameras"]["exterior_image_1_left"]
            bind(cache / payload["file"], payload["sha256"])
    runs = []
    for family, registration, trainer_path, prior_dir, finalizer_path in CAMPAIGNS:
        registered = read(bind(registration)); finalized = read(bind(finalizer_path))
        require(finalized.get("status") == "passed", "Upstream full-campaign finalizer not complete")
        for path, expected in registered["dependencies"].items():
            bind(path, expected)
        trainer = module(trainer_path, "droid_completion_register_" + family)
        for oldrow in registered["runs"]:
            conf = read(bind(oldrow["config"], oldrow["sha256"]))
            require(conf["mode"] == oldrow["mode"] and conf["seed"] == oldrow["seed"], "Run/config mismatch")
            require(conf["cache_root"] == relative(cache), "Unmatched feature cache")
            directory = local(conf["output_dir"])
            summary = trainer.validate_completed(directory)
            package, state = trainer.read_package(directory / "best")
            selected_epoch = state["epoch"]
            require(summary["best_epoch"] == selected_epoch and summary["completed_epochs"] == conf["epochs"],
                    "Incomplete or reselected upstream model")
            del state
            for filename in ("training_summary.json", "training_config.json", "metrics.jsonl"):
                bind(directory / filename)
            package_manifest = read(bind(directory / "best/package_manifest.json"))
            for filename, expected in package_manifest["files"].items():
                bind(directory / "best" / filename, expected)
            prior_path = bind(Path(prior_dir) / (oldrow["name"] + "_validation.json"))
            prior = read(prior_path)
            require(prior.get("status") == "complete" and prior.get("scope") == "original_validation_development_only"
                    and prior.get("mode") == oldrow["mode"] and prior.get("seed") == oldrow["seed"]
                    and prior.get("checkpoint_sha256") == sha(directory / "best/model.pt")
                    and prior.get("selected_epoch") == selected_epoch, "Prior finalized ledger identity differs")
            oldkeys = [(w["episode_id"], w["session_id"], w["window_start"]) for w in prior["windows"]]
            require(sorted(oldkeys) == keys, "Prior ledger is missing/duplicating validation windows")
            # Reconstruct existing native-MSE summaries before registration; no new scoring.
            per_episode = {}
            for eid, session, start in oldkeys:
                per_episode.setdefault(eid, [])
            for w in prior["windows"]:
                values = np.asarray(w["native_mse"], dtype=np.float64)
                require(values.shape == (10,) and np.isfinite(values).all() and (values >= 0).all(), "Invalid prior MSE")
                per_episode[w["episode_id"]].append(values)
            means = {eid: np.mean(values, axis=0) for eid, values in per_episode.items()}
            require(len(prior["episodes"]) == len(means) == 141, "Prior episode denominator differs")
            for e in prior["episodes"]:
                np.testing.assert_allclose(means[e["episode_id"]], e["native_mse"], **POLICY["arithmetic_tolerance"])
            np.testing.assert_allclose(np.mean(list(means.values()), axis=0), prior["summary"]["native_mse"],
                                       **POLICY["arithmetic_tolerance"])
            runs.append({"name": oldrow["name"], "mode": oldrow["mode"], "seed": oldrow["seed"],
                "family": family, "config": oldrow["config"], "trainer": trainer_path,
                "checkpoint": relative(directory / "best"), "checkpoint_sha256": sha(directory / "best/model.pt"),
                "selected_epoch": selected_epoch, "completed_training_epochs": conf["epochs"],
                "batch_size": conf["batch_size"], "prior_ledger": relative(prior_path),
                "prior_finalization": finalizer_path})
            print("verified fixed checkpoint " + oldrow["name"], flush=True)
    require(len(runs) == 33 and len({r["name"] for r in runs}) == 33, "Full learned roster is incomplete")
    reference = next(r for r in runs if r["name"] == "autoregressive_s0")
    runs.append({"name": "persistence", "mode": "persistence", "seed": None, "family": "persistence",
                 "batch_size": 128, "prior_ledger": reference["prior_ledger"], "checkpoint_sha256": None})
    for index, row in enumerate(runs):
        row["index"] = index
    result = {"schema": "droid_complementary_metrics_v1_registration", "status": "registered",
        "created_utc": now(), "policy": POLICY, "metrics": list(METRICS), "runs": runs,
        "window_keys": keys, "source_dependencies": sources,
        "new_model_inference_performed_before_registration": False,
        "metric_choice_timing": "post hoc after original DROID and external MSE findings; secondary only"}
    # Ensure nothing changed during the full input audit.
    for path, expected in sources.items():
        require(sha(local(path)) == expected, "Source changed during registration: " + path)
    atomic_json(result, REG)
    print("Registration committed: " + str(REG) + " sha256=" + sha(REG), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "verify", "list"))
    args = parser.parse_args()
    if args.command == "register":
        register()
    else:
        reg = verify_registration()
        if args.command == "list":
            for row in reg["runs"]:
                print(row["index"], row["name"], row["family"])
        else:
            print("Verified " + str(len(reg["runs"])) + " registered evaluations")
