#!/usr/bin/env python3
"""Build a self-contained local research page and a source-verified status snapshot."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FILE_CACHE = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def snapshot(root=ROOT):
    global FILE_CACHE
    # Reuse the established evaluation task inventory and planner validator.
    specification = importlib.util.spec_from_file_location("shiftwm_evaluation_inventory", root / "scripts/run_evaluation_campaign.py")
    inventory = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(inventory)
    if FILE_CACHE is None: FILE_CACHE = inventory.FileDigestCache()
    evaluator_hash = FILE_CACHE.sha256(root / "src/shiftwm/evaluate.py")
    previous = Path.cwd()
    os.chdir(root)
    try:
        campaign_path = root / "configs/world/full_campaign.json"
        campaign = read(campaign_path)
        tasks = inventory.evaluation_tasks(campaign_path)
    finally:
        os.chdir(previous)
    invalid, sources, training = [], {}, []

    def source(path):
        sources[str(path.relative_to(root))] = digest(path)

    source(campaign_path)
    datasets = {}
    for task in campaign["tasks"]:
        config_path = root / task["config"]
        config = read(config_path)
        source(config_path)
        run = root / config["output_dir"]
        summary_path = run / "training_summary.json"
        metrics_path = run / "metrics.jsonl"
        summary = read(summary_path) if summary_path.is_file() else {}
        last_epoch = 0
        if metrics_path.is_file():
            # An actively written incomplete final JSONL line is not an epoch.
            metrics_raw = metrics_path.read_bytes()
            lines = metrics_raw.splitlines(keepends=True)
            for line in reversed(lines):
                if line.strip() and line.endswith(b"\n"):
                    last_epoch = int(json.loads(line)["epoch"]); break
            sources[str(metrics_path.relative_to(root))] = hashlib.sha256(metrics_raw).hexdigest()
        if summary_path.is_file(): source(summary_path)
        completed = (summary.get("status") == "completed"
                     and summary.get("completed_epochs") == config["epochs"]
                     and last_epoch == config["epochs"]
                     and (run / "last/model.pt").is_file())
        training.append({"id": task["id"], "mode": config["model"]["mode"], "seed": config["seed"],
                         "environment": Path(config["pretrained_dir"]).name,
                         "state": "completed" if completed else "in_progress" if last_epoch else "awaiting_records",
                         "last_logged_epoch": last_epoch, "planned_epochs": config["epochs"]})
        manifest_path = root / config["data_root"] / "manifest.json"
        if config["data_root"] not in datasets:
            manifest = read(manifest_path)
            if manifest["environment"] == "pusht" and manifest.get("action_interface") != "relative":
                raise ValueError("Obsolete absolute-control PushT corpus is ineligible")
            source(manifest_path)
            datasets[config["data_root"]] = {"environment": manifest["environment"], "episodes": len(manifest["episodes"]),
                "manifest_sha256": sources[str(manifest_path.relative_to(root))],
                "split_episode_counts": dict(Counter(ep["split"] for ep in manifest["episodes"])),
                "grouped_transitions": sum(ep["steps"] for ep in manifest["episodes"]),
                "train_combinations": manifest["train_combinations"],
                "development_combinations": manifest["development_combinations"],
                "heldout_test_combinations": manifest["heldout_test_combinations"]}

    counts = {kind: {"planned": 0, "completed": 0, "records": []} for kind in ("forecast", "planning", "controls")}
    for task in tasks:
        group = "controls" if task["policy"] != "world_model" else task["kind"]
        counts[group]["planned"] += 1
        path = root / task["output"]
        if not path.is_file(): continue
        try:
            result = read(path)
            if result.get("status") != "complete": continue
            if result.get("environment") != task["environment"]: raise ValueError("Environment mismatch")
            if not result.get("checkpoint_sha256") or not result.get("evaluator_sha256"): raise ValueError("Missing provenance")
            if result["evaluator_sha256"] != evaluator_hash: raise ValueError("Evaluator source hash mismatch")
            checkpoint_path = root / task["checkpoint"] / "model.pt"
            if not checkpoint_path.is_file(): raise ValueError("Checkpoint weights missing")
            checkpoint_hash = FILE_CACHE.sha256(checkpoint_path)
            if result["checkpoint_sha256"] != checkpoint_hash: raise ValueError("Actual checkpoint hash mismatch")
            sources[str(checkpoint_path.relative_to(root))] = checkpoint_hash
            expected_manifest = datasets[task["config"]["data_root"]]["manifest_sha256"]
            if result.get("data_manifest_sha256") != expected_manifest: raise ValueError("Data manifest mismatch")
            mode = "frozen" if task["training_summary"] is None else task["config"]["model"]["mode"]
            if result.get("model_mode") != mode: raise ValueError("Model mode mismatch")
            if result.get("training_seed") != (None if mode == "frozen" else task["config"]["seed"]):
                raise ValueError("Training seed mismatch")
            if task["kind"] == "planning": inventory.validate_completed_planning(result, task, 64)
            elif result["forecast"].get("split") != task["split"]: raise ValueError("Forecast split mismatch")
            source(path)
            counts[group]["completed"] += 1
            counts[group]["records"].append(str(path.relative_to(root)))
        except (ValueError, KeyError, RuntimeError) as error:
            invalid.append({"file": str(path.relative_to(root)), "reason": str(error)})
    states = Counter(r["state"] for r in training)
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "training": {"planned": len(training), "completed": states["completed"], "in_progress": states["in_progress"],
                         "awaiting_records": states["awaiting_records"], "runs": training},
            "evaluation": counts, "datasets": list(datasets.values()),
            "planner": inventory.OFFICIAL_PLANNER_BUDGET,
            "conclusions": "pending_matched_comparisons", "invalid_records": invalid, "source_sha256": sources,
            "interpretation": "Counts measure verified completion, not scientific superiority. Development and non-model controls are excluded from main forecast/planning counts. Missing logs do not imply a scheduler state."}


def build(output=HERE, status_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = snapshot()
    write_atomic(output / "status.json", json.dumps(status, indent=2, allow_nan=False) + "\n")
    if status_only: return status
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("index.html", "styles.css", "app.js"):
        if output != HERE: shutil.copy2(HERE / name, output / name)
    copies = {"paper/world_model_draft.pdf": "paper.pdf", "paper/figures/method.svg": "method.svg",
              "reports/world_data_and_evaluation_protocol.md": "protocol.md", "demo/README.md": "demo-readme.md",
              "references/world_upstream_revisions.json": "upstream-revisions.json"}
    for original, name in copies.items():
        temporary = assets / f".{name}.{os.getpid()}.tmp"
        shutil.copy2(ROOT / original, temporary)
        temporary.replace(assets / name)
    source_files = [ROOT / name for name in ("README.md", "pyproject.toml", "requirements.lock.txt", "LICENSE", "THIRD_PARTY_NOTICES.md")]
    for folder, pattern in (("src", "*"), ("configs/world", "*.json"), ("scripts", "*"), ("tests", "*")):
        source_files += [p for p in (ROOT / folder).rglob(pattern) if p.is_file() and "__pycache__" not in p.parts
                         and (p.suffix in {".py", ".json", ".sh", ".md", ".toml", ".txt"} or p.name == "LICENSE")]
    source_files += [ROOT / "references" / name for name in ("world_upstream_revisions.json", "world_artifact_sources.json")]
    source_files += [ROOT / "reports" / name for name in ("world_execution.md", "implementation_review.md", "small_world_models_proposal.md", "small_world_novelty.md")]
    source_files += [ROOT / p for p in copies if p.endswith((".md", ".json"))]
    source_files += [ROOT / p for p in ("demo/app.py", "demo/backend.py", "demo/requirements.txt", "demo/MODEL_CARD.md", "demo/prepare_bundle.py")]
    # Only source, configuration and provenance: no credentials, logs, weights or data episodes.
    zip_temporary = assets / f".shiftwm-source.{os.getpid()}.zip.tmp"
    with zipfile.ZipFile(zip_temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(source_files)):
            archive.write(path, "shiftwm/" + str(path.relative_to(ROOT)))
    zip_temporary.replace(assets / "shiftwm-source.zip")
    manifest = {"generated_at_utc": status["generated_at_utc"], "external_publication": False,
                "files": {str(p.relative_to(output)): digest(p) for p in sorted(assets.iterdir()) if p.is_file()},
                "paper_source": "paper/world_model_draft.pdf", "figure_source": "paper/figures/method.svg"}
    write_atomic(output / "artifact-manifest.json", json.dumps(manifest, indent=2) + "\n")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    status = build(args.output, args.status_only)
    print(json.dumps({"training_complete": status["training"]["completed"], "training_planned": status["training"]["planned"],
                      "forecast_complete": status["evaluation"]["forecast"]["completed"],
                      "planning_complete": status["evaluation"]["planning"]["completed"],
                      "invalid_records": len(status["invalid_records"]), "output": str(args.output.resolve())}))
