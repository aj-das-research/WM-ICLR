#!/usr/bin/env python3
"""Build a portable inference release from a completed run's validation-best model.

No optimizer/RNG state is copied. The CPU release check uses a real cached
validation history and recorded/zero future actions; it is not a benchmark run.
Frozen upstream exports require the explicit --frozen flag.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch

from shiftwm.checkpoint import load_package
from shiftwm.data import TrajectoryDataset
import shiftwm.upstream as upstream


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def inspect_source(run_dir, frozen=False, data_root=None, feature_cache=None):
    run_dir = Path(run_dir)
    source = run_dir if (run_dir / "model.pt").is_file() else run_dir / "best"
    if source.name != "best":
        raise ValueError("Release source must be the run's validation-best directory")
    run_dir = source.parent
    config = read_json(source / "config.json")
    mode = config["model_config"]["mode"]
    if frozen != (mode == "frozen"):
        raise ValueError("Frozen baselines require explicit --frozen; trained models must not use it")
    record = {"run_dir": run_dir, "source": source, "config": config,
              "frozen": frozen, "mode": mode, "copy_files": []}
    if frozen:
        export = read_json(source / "export_manifest.json")
        if export.get("training_status") != "released_upstream_weights_not_finetuned":
            raise ValueError("Frozen export lacks explicit unchanged-upstream provenance")
        environment = export["environment"]
        record.update(environment=environment, completed_epochs=0, best_prediction=None)
        record["copy_files"].append(source / "export_manifest.json")
        run_config = {}
    else:
        summary = read_json(run_dir / "training_summary.json")
        run_config = read_json(run_dir / "run_config.json")
        expected_epochs = int(run_config["epochs"])
        if summary.get("status") != "completed" or summary.get("completed_epochs") != expected_epochs:
            raise ValueError("Only fully completed training runs may produce a release")
        metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
        if set(range(1, expected_epochs + 1)) - {int(row["epoch"]) for row in metrics}:
            raise ValueError("Training metric history is missing a completed epoch")
        prediction_values = [float(row["val"]["prediction_loss"]) for row in metrics]
        if not all(math.isfinite(value) for value in prediction_values):
            raise ValueError("Nonfinite validation prediction metric")
        best = min(prediction_values)
        if not math.isclose(best, float(summary["best_validation_prediction_loss"]), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Training summary disagrees with minimum validation prediction loss")
        best_epochs = [int(row["epoch"]) for row in metrics if
                       math.isclose(float(row["val"]["prediction_loss"]), best, rel_tol=1e-10, abs_tol=1e-12)]
        record.update(completed_epochs=expected_epochs, best_prediction=best, best_epochs=best_epochs)
        record["copy_files"] += [run_dir / name for name in
                                 ("run_config.json", "training_summary.json", "metrics.jsonl")]
    if data_root is None:
        data_root = run_config.get("data_root")
    if feature_cache is None:
        feature_cache = run_config.get("dataset_kwargs", {}).get("feature_cache")
    if data_root is None or feature_cache is None:
        raise ValueError("Real cached verification data required: provide --data and --feature-cache for frozen export")
    data_root, feature_cache = resolve(data_root), resolve(feature_cache)
    manifest = read_json(data_root / "manifest.json")
    if frozen and manifest["environment"] != record["environment"]:
        raise ValueError("Frozen baseline and verification data environment differ")
    record.update(environment=manifest["environment"], data_root=data_root,
                  feature_cache=feature_cache, data_manifest=manifest)
    return record


def verify_portable_forward(package, record):
    # Force the same bundled-source fallback a wheel or Space installation uses.
    previous_root = upstream.ROOT
    for key in ("shiftwm_upstream_module", "shiftwm_upstream_jepa"):
        sys.modules.pop(key, None)
    try:
        upstream.ROOT = Path("/nonexistent-shiftwm-release-check")
        model, state = load_package(package, device="cpu")
    finally:
        upstream.ROOT = previous_root
    model.eval().requires_grad_(False)
    if state["config"] != record["config"]:
        raise ValueError("JSON configuration differs from embedded checkpoint configuration")
    if not record["frozen"]:
        if not math.isclose(float(state["best_metric"]), record["best_prediction"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Selected model is not the recorded validation-best checkpoint")
        if state["epoch"] < 1 or state["epoch"] > record["completed_epochs"]:
            raise ValueError("Selected checkpoint epoch is outside completed training history")
        if state["epoch"] not in record["best_epochs"]:
            raise ValueError("Checkpoint epoch did not attain minimum validation prediction loss")
    cache_manifest = read_json(record["feature_cache"] / "manifest.json")
    if cache_manifest["encoder_weights_sha256"] != model.provenance["weights_sha256"]:
        raise ValueError("Verification cache and model pretrained encoder differ")
    dataset = TrajectoryDataset(record["data_root"], split="val", sequence_length=8,
                                stride=5, feature_cache=record["feature_cache"])
    example = dataset[0]
    history = model.config.history_length
    support = example["features"][:history].unsqueeze(0)
    past = example["actions"][:history - 1].unsqueeze(0)
    future = example["actions"][history - 1:history + 1].unsqueeze(0)
    with torch.inference_mode():
        contexts = model.infer_context(support, past)
        recorded = model.rollout_features(support, past, future, contexts=contexts)
        zero = model.rollout_features(support, past, torch.zeros_like(future), contexts=contexts)
    expected = (1, future.shape[1], model.latent_dim)
    if recorded.shape != expected or zero.shape != expected:
        raise ValueError("Portable rollout returned incorrect dimensions")
    if not torch.isfinite(recorded).all() or not torch.isfinite(zero).all():
        raise FloatingPointError("Portable rollout has nonfinite values")
    result = {"status": "passed", "device": "cpu", "source": "bundled_pinned_lewm",
              "test": "weights_only_load_and_real_cached_support_recorded_and_zero_action_rollouts",
              "trajectory_id": example["trajectory_id"], "split": "val", "start": int(example["start"]),
              "support_images": history, "future_blocks": future.shape[1],
              "output_shape": list(expected), "all_outputs_finite": True,
              "recorded_and_zero_action_outputs_differ": not torch.equal(recorded, zero),
              "interpretation": "API/checkpoint verification only; no benchmark accuracy or planning success measured"}
    return state, result


def model_card(record, state, verification):
    config = record["config"]
    environment, mode = record["environment"], record["mode"]
    run_id = record["run_dir"].name
    if record["frozen"]:
        training = "Unchanged released upstream checkpoint; this project performed no fine-tuning for this baseline."
    else:
        training = (f"Training completed {record['completed_epochs']} full epochs. The selected checkpoint is from "
                    f"epoch {state['epoch']}, chosen by validation one-step prediction MSE "
                    f"{record['best_prediction']:.9g}. This is a validation-selection statistic, not a test result.")
    variants = {
        "frozen": "Unchanged pretrained encoder, action encoder, and predictor; no learned context correction.",
        "plain": "Fine-tuned predictor without observation correction. Shifted goal features remain unaligned; retain as a diagnostic, not a decisive factorization planning comparator.",
        "framewise": "History-independent per-image residual calibration; the same mapping is applied to observed states and goals. No inferred dynamics context.",
        "single": "One shared chronological context conditions both observation and action-embedding correction.",
        "factorized": "Separate observation and dynamics contexts with paired-render consistency objectives during training.",
        "factorized_unpaired": "Separate contexts with canonical alignment and prediction objectives; no context-pairing regularizers.",
        "observation": "Observation-context branch ablation; predictor remains fine-tuned.",
        "dynamics": "Dynamics-context branch ablation; predictor remains fine-tuned and observed/goal coordinates are unaligned.",
    }
    action = ("Relative PushT commands: target = current pusher position + 100 × action. "
              "Do not substitute absolute pixel targets." if environment == "pusht" else
              "Reacher normalized torque controls, with the upstream environment's action_repeat=2.")
    provenance = config["provenance"]
    return f"""# {run_id}

This is an inference checkpoint for the **{environment}** environment family,
method **{mode}**. {training}

{variants[mode]}

## Inputs and use

- RGB floats in `[0,1]`; ImageNet normalization and resizing to 224 are internal.
- Three observation frames and two executed action blocks infer a context.
- Each action block contains five chronological two-dimensional native commands,
  flattened to dimension 10. The checkpoint normalizes raw actions internally.
- {action}
- Goal images use the same observation mapping as the history. Forecast outputs
  are 192-dimensional latents, not generated RGB images.

```python
import torch
from shiftwm.checkpoint import load_package

model, metadata = load_package("path/to/this/release", device="cuda")
with torch.inference_mode():
    # images [B,3,3,H,W]; past_actions [B,2,10]; future_actions [B,K,10]
    features = model.encode_images(images)
    contexts = model.infer_context(features, past_actions)
    predicted = model.rollout_features(features, past_actions, future_actions,
                                       contexts=contexts)
    goal = model.goal_embedding(goal_images, contexts[0])  # goal_images [B,3,H,W]
    cost = (predicted[:, -1] - goal).square().mean(-1)
```

Install the accompanying source wheel or the project source plus its pinned
dependencies. The LeWM modules are bundled with notices; no external git checkout
is needed by the loader. PyTorch loads model.pt with `weights_only=True`.

## Training information and limitations

The fine-tuned variants freeze the visual encoder/projector and train the
predictor/action modules plus the active correction modules. Targets use canonical
paired simulator renders: **privileged training supervision**. Canonical target
images, simulator states, physics parameters, and intervention IDs are not supplied
to deployment context inference. The factorized variant additionally uses paired
appearance views and appearance IDs in its training-only losses.

The implemented prediction loss is teacher-forced one-step latent regression,
not recursive rollout training. Deployment does use recursive prediction. The
networks are functional conditioning modules; low context-consistency loss is not
proof of physical-variable identification or disentanglement.

The first campaign covers canonical/warm/cool photometric conditions, PushT damping
or Reacher density changes, and held-out combinations described in the attached
data_manifest.json. Appearance transforms are not different camera viewpoints.
Dim appearance and stronger dynamics are separate extrapolation conditions.
This package was verified using a real validation history from the same family;
successful API execution is not evidence of task success or cross-domain transfer.

No real-robot, medical, or clinical capability has been demonstrated. Predictive
error, context ambiguity, compounded rollout error, and goal-coordinate mismatch
for the unaligned plain baseline are relevant limitations. Report the independent
test/evaluation artifacts separately; no benchmark result is invented by packaging.

## Provenance and verification

- Base weights SHA256: `{provenance['weights_sha256']}`.
- Action statistics SHA256: `{provenance.get('action_stats_sha256', 'not recorded')}`.
- Selected checkpoint epoch: {state['epoch']}.
- Portable bundled-source CPU verification: **{verification['status']}** using
  `{verification['trajectory_id']}` from the validation split.
- Exact source-file hashes, code revision, preprocessing, and dataset/cache
  manifests are attached. No optimizer state is included in this inference release.

Base source: https://github.com/lucas-maes/le-wm . `LICENSE` and `NOTICE.json`
preserve upstream code attribution. These notices do not grant rights beyond
the underlying model/data licenses. Public upload and applicable redistribution
terms must be recorded separately.
"""


def build_release(run_dir, output=None, frozen=False, data_root=None, feature_cache=None, source_wheel=None):
    torch.set_num_threads(2)
    record = inspect_source(resolve(run_dir), frozen, data_root, feature_cache)
    output = resolve(output) if output else ROOT / "artifacts" / "releases" / record["run_dir"].name
    source_model = record["source"] / "model.pt"
    source_hash = sha256(source_model)
    if output.exists():
        existing = output / "release_manifest.json"
        if existing.is_file():
            prior = read_json(existing)
            if prior.get("status") == "ready" and prior.get("model_sha256") == source_hash and prior.get("package_files") and all(
                    (output / name).is_file() and sha256(output / name) == digest
                    for name, digest in prior.get("package_files", {}).items()):
                print(json.dumps({"status": "already_packaged", "output": str(output)}), flush=True)
                return prior
        raise FileExistsError(f"Refusing to overwrite differing or incomplete release: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for name in ("model.pt", "config.json"):
            shutil.copy2(record["source"] / name, temporary / name)
        if sha256(temporary / "model.pt") != source_hash:
            raise RuntimeError("Source checkpoint changed while copying")
        for path in record["copy_files"]:
            shutil.copy2(path, temporary / path.name)
        shutil.copy2(record["data_root"] / "manifest.json", temporary / "data_manifest.json")
        shutil.copy2(record["feature_cache"] / "manifest.json", temporary / "feature_cache_manifest.json")
        for name in ("LICENSE", "NOTICE.json"):
            shutil.copy2(upstream.VENDOR / name, temporary / name)
        if source_wheel is not None:
            wheel = resolve(source_wheel)
            if not wheel.is_file() or wheel.suffix != ".whl":
                raise ValueError("--source-wheel must point to an existing wheel")
            shutil.copy2(wheel, temporary / wheel.name)
        state, verification = verify_portable_forward(temporary, record)
        (temporary / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        (temporary / "MODEL_CARD.md").write_text(model_card(record, state, verification))
        provenance = {"upstream": record["config"]["provenance"],
                      "selected_model_epoch": state["epoch"], "completed_epochs": record["completed_epochs"],
                      "selection": "unchanged_upstream" if frozen else "minimum_validation_prediction_loss",
                      "builder_sha256": sha256(Path(__file__)),
                      "source_run": str(record["run_dir"].resolve()),
                      "source_model_sha256": source_hash,
                      "model_config": record["config"]["model_config"],
                      "evaluation_status": "portable_API_verified_not_benchmark_evaluated_by_packager"}
        (temporary / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        manifest = {"format_version": 1, "status": "ready", "run_id": record["run_dir"].name,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "kind": "frozen_upstream_baseline" if frozen else "completed_training_validation_best",
                    "environment": record["environment"], "mode": record["mode"],
                    "model_sha256": source_hash, "optimizer_included": False,
                    "verification": verification,
                    "package_files": {p.name: sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()}}
        (temporary / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, output)
        print(json.dumps({"status": "ready", "output": str(output), "model_sha256": source_hash,
                          "portable_check": verification["status"]}), flush=True)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen", action="store_true", help="Explicitly permit unchanged upstream export")
    parser.add_argument("--data", type=Path, help="Verification data override; required for frozen exports")
    parser.add_argument("--feature-cache", type=Path, help="Verification feature-cache override")
    parser.add_argument("--source-wheel", type=Path, help="Optional already-built source wheel")
    args = parser.parse_args()
    build_release(args.run_dir, args.output, args.frozen, args.data, args.feature_cache, args.source_wheel)


if __name__ == "__main__":
    main()
