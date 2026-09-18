#!/usr/bin/env python3
"""Validate a complete real-DROID campaign, report all outcomes, export locally.

This postprocessor never trains, selects on test results, alters scientific
sources, downloads assets, or uploads a release. Its release is inference-only:
original run directories retain the optimizer/RNG continuation checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate
import train
from shiftwm.real_video.data import RealVideoDataset, sha256

METHOD_NAMES = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics",
                "factorized": "ShiftWM real-video variant (ours)", "action_free": "Action-free",
                "persistence": "Persistence", "constant_velocity": "Constant feature velocity"}
POPULATIONS = [(camera, horizon) for camera in evaluate.CAMERAS for horizon in (5, 10)]
OFFLINE_CHECK = r'''
import json, os, pathlib, socket, sys
release = pathlib.Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path[:0] = [str(release / "source/src"), str(release / "source/scripts/real_video")]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
def forbidden(*args, **kwargs):
    raise RuntimeError("Network access is forbidden during portable-package verification")
socket.create_connection = forbidden
import numpy as np
import torch
import train
import shiftwm.upstream as upstream
assert pathlib.Path(train.__file__).resolve().is_relative_to(release)
assert pathlib.Path(upstream.__file__).resolve().is_relative_to(release)
assert not (release / "source/external").exists()
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
sample = np.load(release / "verification/sample.npz", allow_pickle=False)
features = torch.from_numpy(sample["features"])[None].float()
actions = torch.from_numpy(sample["actions"])[None].float()
records = []
with torch.inference_mode():
    for path in sorted((release / "models").iterdir()):
        model, state = train.load_package(path, "cpu")
        reference = np.load(path / "verification_reference.npy", allow_pickle=False)
        predicted = model.predict(features[:, :3], actions[:, :2], actions[:, 2:])
        if not torch.isfinite(predicted).all() or tuple(predicted.shape) != tuple(reference.shape):
            raise ValueError("Portable forecast is nonfinite or has wrong dimensions")
        np.testing.assert_allclose(predicted.numpy(), reference, rtol=1e-6, atol=1e-6)
        records.append({"run": path.name, "status": "passed", "shape": list(predicted.shape),
                        "epoch": state["epoch"], "source": "bundled pinned LeWM",
                        "maximum_absolute_difference": float(np.max(np.abs(predicted.numpy()-reference)))})
print(json.dumps({"status": "passed", "models": records,
                  "test": "CPU offline weights-only load and real validation-support forecast parity; not a benchmark"}))
'''


def resolved(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(Path(path).read_text())


def collect_campaign(campaign_path, results_root):
    """Fail before producing a public report if even one registered arm is absent."""
    campaign_path, results_root = resolved(campaign_path), resolved(results_root)
    campaign = read_json(campaign_path)
    rows = campaign.get("runs", [])
    expected = {(mode, seed) for mode in train.RealVideoWorldModel.MODES for seed in (0, 1, 2)}
    if (len(rows) != 12 or {(row["mode"], row["seed"]) for row in rows} != expected
            or len({row["name"] for row in rows}) != 12):
        raise ValueError("Campaign is not exactly the four registered modes times three seeds")
    missing = []
    for row in rows:
        config = read_json(resolved(row["config"]))
        run = resolved(config["output_dir"])
        for path in [run / "training_summary.json", run / "best/model.pt", run / "last/model.pt"] + [
                results_root / row["name"] / camera / f"h{horizon}" / "results.json"
                for camera, horizon in POPULATIONS]:
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise ValueError(f"Incomplete campaign: {len(missing)} required files absent; first: {missing[0]}")
    records, populations = [], {(camera, horizon): [] for camera, horizon in POPULATIONS}
    for row in rows:
        path = resolved(row["config"])
        config = read_json(path)
        if config.get("mode") != row["mode"] or config.get("seed") != row["seed"]:
            raise ValueError("Campaign row/config method or seed mismatch")
        run = resolved(config["output_dir"])
        summary = train.validate_completed(run)
        runtime_config = read_json(run / "training_config.json")
        if train.scientific_config(runtime_config) != train.scientific_config(config):
            raise ValueError("Completed training differs from registered configuration")
        manifest, stats, identity = train.audit_inputs(runtime_config)
        if train.digest(identity) != summary["training_identity"]:
            raise ValueError("Completed training and live audit differ")
        for camera, horizon in POPULATIONS:
            result_path = results_root / row["name"] / camera / f"h{horizon}" / "results.json"
            result = read_json(result_path)
            if (result.get("mode") != row["mode"] or result.get("seed") != row["seed"]
                    or result.get("camera") != camera or result.get("horizon") != horizon
                    or result.get("training_identity") != summary["training_identity"]
                    or Path(result.get("checkpoint_path", "")).absolute() != (run / "best").absolute()):
                raise ValueError("Evaluation path/registered method/selected checkpoint mismatch")
            populations[(camera, horizon)].append(result_path)
        records.append({"row": row, "config_path": path, "config": runtime_config,
                        "run": run, "summary": summary, "cache_manifest": manifest,
                        "statistics": stats, "identity": identity})
    first = records[0]
    for record in records[1:]:
        if (record["cache_manifest"] != first["cache_manifest"]
                or record["statistics"] != first["statistics"]):
            raise ValueError("Registered models do not share a common cache and normalization")
    return records, populations


def checked_copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    before = sha256(source)
    if expected is not None and before != expected:
        raise ValueError(f"Source differs before release copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(destination) != before or sha256(source) != before:
        raise ValueError(f"Source changed during release copy: {source}")
    return before


def source_bundle(stage, campaign_path, records):
    """Copy exact required sources; imported vendor code remains unmodified."""
    required = ["src/shiftwm/__init__.py", "src/shiftwm/model.py", "src/shiftwm/upstream.py",
                "src/shiftwm/checkpoint.py", "src/shiftwm/real_video/__init__.py",
                "src/shiftwm/real_video/model.py", "src/shiftwm/real_video/data.py",
                "src/shiftwm/real_video/features.py", "src/shiftwm/vendor/lewm/__init__.py",
                "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/jepa.py",
                "src/shiftwm/vendor/lewm/NOTICE.json", "src/shiftwm/vendor/lewm/LICENSE",
                "scripts/real_video/train.py", "scripts/real_video/evaluate.py",
                "scripts/real_video/finalize_campaign.py", "scripts/real_video/run_campaign.py",
                "scripts/real_video/prepare_droid.py", "scripts/real_video/fetch_droid.py",
                "pyproject.toml", "requirements.lock.txt", "LICENSE"]
    for name in required:
        expected = records[0]["identity"]["dependencies"].get(str(ROOT / name))
        checked_copy(ROOT / name, stage / "source" / name, expected)
    checked_copy(campaign_path, stage / "provenance/campaign.json")
    for record in records:
        checked_copy(record["config_path"], stage / "provenance/configs" / record["config_path"].name)
    first = records[0]
    cache = resolved(first["config"]["cache_root"])
    audit = resolved(first["config"]["metadata_audit"])
    copies = {resolved(first["config"]["protocol_path"]): "protocol.md",
              cache / "manifest.json": "feature_cache_manifest.json",
              cache / "training_statistics.json": "training_statistics.json",
              audit: "data_audit.json", audit.parent / "manifest.json": "dataset_manifest.json"}
    for source, destination in copies.items():
        checked_copy(source, stage / "provenance" / destination)
    for name in ("reports/evidence/real_video/droid_selected_inventory.json",
                 "data/real_video/droid_selected/raw/download_receipt.json"):
        if (ROOT / name).is_file():
            checked_copy(ROOT / name, stage / "provenance" / Path(name).name)
    runtime = {name: importlib.metadata.version(name) for name in
               ("torch", "numpy", "transformers", "einops", "safetensors", "Pillow")}
    runtime.update(python=sys.version, torch_cuda_build=torch.version.cuda)
    train.atomic_json(runtime, stage / "provenance/runtime.json")


def encoder_bundle(stage, encoder_root, expected_provenance):
    encoder_root = resolved(encoder_root)
    actual = read_json(encoder_root / "provenance.json")
    if actual != expected_provenance:
        raise ValueError("Release encoder differs from the frozen feature-cache encoder")
    for entry in actual["files"]:
        filename = Path(entry["file"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError("Unsafe encoder provenance filename")
        checked_copy(encoder_root / filename, stage / "encoder/dinov2-small" / filename, entry["sha256"])
    checked_copy(encoder_root / "provenance.json", stage / "encoder/dinov2-small/provenance.json")
    if not (stage / "encoder/dinov2-small/model.safetensors").is_file():
        raise ValueError("Frozen shared encoder weights were not bundled")
    checked_copy(ROOT / "LICENSE", stage / "licenses/SHIFTWM-MIT.txt")
    checked_copy(ROOT / "src/shiftwm/vendor/lewm/LICENSE", stage / "licenses/LEWM-MIT.txt")
    checked_copy("/usr/share/common-licenses/Apache-2.0", stage / "licenses/DINOV2-APACHE-2.0.txt")
    (stage / "licenses/THIRD_PARTY_NOTICES.md").write_text(
        "# Third-party notices\n\n"
        "DINOv2-small: Meta/Facebook, Apache-2.0. Exact repository, revision, hashes, "
        "and original model card are in encoder/dinov2-small.\n\n"
        "LeWorldModel: Lucas Maes and contributors, MIT; exact upstream revision "
        "and unmodified source hashes are in source/src/shiftwm/vendor/lewm/NOTICE.json.\n\n"
        "DROID: Khazatsky et al., DROID: A Large-Scale In-The-Wild Robot Manipulation "
        "Dataset (2024), https://droid-dataset.github.io/ . Official release: "
        "https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md . "
        "The dataset is released under CC-BY-4.0, "
        "https://creativecommons.org/licenses/by/4.0/legalcode . "
        "This local research bundle includes provenance and one derived validation "
        "feature window, not the source videos or raw dataset archives.\n\n"
        "The project MIT license applies to its code. It does not replace upstream "
        "encoder, third-party source, or dataset terms. This local bundle does not "
        "assert a separate license grant for learned weights.\n")


def validation_sample(record, stage):
    config = record["config"]
    data = RealVideoDataset(resolved(config["cache_root"]), "val", horizon=5, stride=5,
                            camera=evaluate.CAMERAS[0], verify=True)
    batch = data[0]
    episode = data.episodes[batch["episode_index"]]
    destination = stage / "verification"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "sample.npz", features=batch["features"].numpy(),
                        actions=batch["actions"].numpy())
    metadata = {"split": "val", "camera": evaluate.CAMERAS[0],
                "episode_id": episode["episode_id"], "session_id": episode["session_id"],
                "window_start": batch["window_start"],
                "native_frame_indices": episode["frame_indices"][batch["window_start"]:batch["window_start"]+8].tolist(),
                "source_payload": episode["cameras"][evaluate.CAMERAS[0]],
                "purpose": "offline API/weight parity only; not test selection or benchmark evidence"}
    train.atomic_json(metadata, destination / "sample_provenance.json")
    return batch


def model_card(record, state):
    mode, seed = record["row"]["mode"], record["row"]["seed"]
    return f"""# {record['row']['name']}

{METHOD_NAMES[mode]}, training seed {seed}. Trained for 30 complete epochs on
genuine recorded DROID video and recorded commanded actions. Selected epoch
{state['epoch']} minimizes complete validation all-five-query standardized MSE
({state['best_metric']:.10g}); this is a selection statistic, not a test score.

The package is a trained residual feature predictor, not a video decoder or an
official DINO-WM reproduction. Its frozen DINOv2-small encoder is shared once
at ../../encoder/dinov2-small. Output coordinates are 1,536-dimensional frozen
DINO patch features; outputs are not RGB images or physical robot states.

Inputs: three support features, two executed command blocks, and K future
command blocks. Each block contains five chronological 7-D native commands,
flattened to 35 dimensions: commanded Cartesian position/orientation components
plus gripper position, as checked by the dataset audit. Normalization is embedded
in the checkpoint. Query images, state/reward/success labels, and language are
not model inputs. No synthetic appearance pairs or simulator consistency losses
are used in this study. Constant dynamics keeps an inferred observation context
and learns a shared dynamics context from zero inputs; it is not identical in
effective capacity to an input-conditioned context.

Training uses exterior camera 1 only. Session-disjoint test camera 1, camera-2
transfer, and ten-block horizon extrapolation are reported separately. These
results cannot establish closed-loop robot success or causal action effects.
Reversed future action blocks are only an observational sensitivity diagnostic.

Total parameters: {record['summary']['parameter_counts']['total']}; trainable:
{record['summary']['parameter_counts']['trainable']}. The original training
directory retains optimizer/RNG state; this portable inference package omits it.
See ../../README.md for offline loading and ../../reports/results.md for all
positive, negative, and inconclusive outcomes. Data/encoder/source identities
are preserved in config.json and the shared provenance directory.
"""


def export_model(record, stage, sample):
    name = record["row"]["name"]
    destination = stage / "models" / name
    destination.mkdir(parents=True)
    source, state = train.read_package(record["run"] / "best")
    model, loaded = train.load_package(source, "cpu")
    if loaded["epoch"] != record["summary"]["best_epoch"]:
        raise ValueError("Export is not the completed run's validation minimum")
    files = {}
    for filename in ("model.pt", "config.json"):
        files[filename] = checked_copy(source / filename, destination / filename)
    train.atomic_json({"format_version": 1, "package_kind": train.PACKAGE_KIND, "files": files},
                      destination / "package_manifest.json")
    for filename in ("training_config.json", "training_summary.json", "metrics.jsonl"):
        checked_copy(record["run"] / filename, destination / filename)
    with torch.inference_mode():
        prediction = model.predict(sample["features"][None, :3], sample["actions"][None, :2],
                                   sample["actions"][None, 2:])
    if not torch.isfinite(prediction).all():
        raise ValueError("Source package has a nonfinite validation-support prediction")
    np.save(destination / "verification_reference.npy", prediction.numpy(), allow_pickle=False)
    (destination / "MODEL_CARD.md").write_text(model_card(record, state))
    return {"run": name, "mode": record["row"]["mode"], "seed": record["row"]["seed"],
            "checkpoint_epoch": state["epoch"], "completed_epochs": 30,
            "training_identity": record["summary"]["training_identity"],
            "checkpoint_sha256": files["model.pt"], "config_sha256": files["config.json"],
            "source_checkpoint": str(record["run"] / "best"),
            "validation_mse": state["best_metric"], "parameter_counts": record["summary"]["parameter_counts"],
            "path": str(destination.relative_to(stage)), "optimizer_included": False}


def verify_offline(stage):
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "OMP_NUM_THREADS": "2"}
    result = subprocess.run([sys.executable, "-I", "-c", OFFLINE_CHECK, str(stage.resolve())],
                            cwd=stage, env=environment, text=True, capture_output=True, check=True, timeout=600)
    verification = json.loads(result.stdout.strip().splitlines()[-1])
    if verification.get("status") != "passed" or len(verification.get("models", [])) != 12:
        raise ValueError("Not all twelve portable packages passed offline verification")
    train.atomic_json(verification, stage / "verification/offline_packages.json")
    return verification


def gain_rows(aggregate):
    result = []
    ours = aggregate["methods"]["factorized"]
    for comparison in aggregate["paired_comparisons"]:
        reference, metric = comparison["reference"], comparison["metric"]
        reference_value = (aggregate["methods"][reference][metric]["mean"]
                           if reference in aggregate["methods"] else aggregate["fixed_support_baselines"][reference][metric])
        value = ours[metric]["mean"]
        if not np.isclose(value - reference_value, comparison["mean_difference"], rtol=1e-10, atol=1e-12):
            raise ValueError("Point estimates differ from matched paired differences")
        lower, upper = comparison["ci95"]
        status = "interval includes zero" if lower <= 0 <= upper else (
            "interval below zero" if upper < 0 else "interval above zero")
        result.append({**comparison, "ours_value": value, "reference_value": reference_value,
                       "relative_mse_reduction_percent": 100*(reference_value-value)/reference_value if reference_value > 0 else None,
                       "interval_description": status})
    return result


def render_report(report):
    lines = ["# Real DROID forecasting results", "",
             f"Validated {report['completed_runs']}/12 full 30-epoch runs and {report['completed_evaluations']}/48 evaluations.", "",
             "These are recorded-video feature forecasts on a prespecified DROID subset, not closed-loop robot success or the full published DROID benchmark. "
             "Checkpoint selection used validation only. Each test error first averages windows within an episode, then weights episodes equally. "
             "Learned rows show the three-seed mean ± sample SD; fixed support baselines have no training-seed SD. "
             "Paired intervals resample recording sessions and training seeds together; they are unadjusted for multiple comparisons.", ""]
    for entry in report["populations"]:
        aggregate = entry["aggregate"]
        title = f"{aggregate['camera']}, horizon {aggregate['horizon']} — {aggregate['population']}"
        horizons = (1, 3, 5) if aggregate["horizon"] == 5 else (10,)
        metrics = ["mean_standardized_mse"] + [f"h{h}_standardized_mse" for h in horizons]
        lines += [f"## {title}", "",
                  "| Method | All-query standardized MSE | " + " | ".join(f"h{h} MSE" for h in horizons) + " |",
                  "|---|---:|" + "---:|" * len(horizons)]
        for mode, values in aggregate["methods"].items():
            cells = [f"{values[metric]['mean']:.6g} ± {values[metric]['seed_sd']:.3g}" for metric in metrics]
            lines.append("| " + METHOD_NAMES[mode] + " | " + " | ".join(cells) + " |")
        for mode, values in aggregate["fixed_support_baselines"].items():
            lines.append("| " + METHOD_NAMES[mode] + " | " + " | ".join(f"{values[metric]:.6g}" for metric in metrics) + " |")
        lines += ["", "Explicit comparisons for the same matched population:", "",
                  "| Reference | Metric | Ours − reference MSE [95% paired CI] | Relative MSE reduction | Interval |",
                  "|---|---|---:|---:|---|"]
        for row in entry["gains"]:
            change = row["relative_mse_reduction_percent"]
            percent = "undefined (zero reference)" if change is None else f"{change:+.2f}%"
            lines.append(f"| {METHOD_NAMES[row['reference']]} | {row['metric']} | "
                         f"{row['mean_difference']:+.6g} [{row['ci95'][0]:+.6g}, {row['ci95'][1]:+.6g}] | "
                         f"{percent} | {row['interval_description']} |")
        lines += ["", "Negative MSE differences favor ours; positive relative reductions indicate lower error. "
                  "The percentages are point-estimate ratios, not percentage points and not confidence intervals. "
                  "Intervals that include zero are inconclusive. All signs are retained. "
                  "Raw MSE, cosine error, per-seed values, and reversed-action diagnostics are preserved in the accompanying JSON.", ""]
    lines += ["## Scope and reuse", "",
              "No test result was used to choose a model or training epoch. Context conditioning on observed support is not evidence of identified physical factors. "
              "Reversing future commands is an observational diagnostic; the recordings do not reveal the outcome of unexecuted actions. "
              "Camera transfer and ten-block extrapolation remain separate from the primary camera-one/five-block population.", "",
              "The local release contains all twelve validation-selected predictors, a shared frozen DINOv2-small encoder, pinned source, statistics, "
              "protocol, model cards, and offline CPU verification. It contains feature predictors, not generated RGB-video models. "
              "No files were uploaded to GitHub, Hugging Face, or another service.", ""]
    return "\n".join(lines)


def release_readme():
    return """# Recorded real-DROID feature-forecasting release

Local inference bundle; twelve full-30-epoch predictors and one shared frozen
DINOv2-small encoder. See reports/results.md and reports/results.json for all
outcomes. Keep camera-transfer and horizon-extrapolation scores separate.

Install the exact dependencies recorded in provenance/runtime.json (the broader
project environment is source/requirements.lock.txt). Then, from this directory:

```python
import sys, torch
sys.path[:0] = ["source/src", "source/scripts/real_video"]
from train import load_package
model, state = load_package("models/droid_factorized_s0", device="cpu")
with torch.inference_mode():
    # features [B,3,1536]; past [B,2,35]; future [B,K,35]
    prediction = model.predict(features, past, future)
```

The model normalizes raw cached DINO features/actions using its embedded train
statistics. Outputs are raw DINO feature coordinates, not RGB images. For real
images, use the exact resize/ImageNet-normalization/2x2 pooling computation in
source/src/shiftwm/real_video/features.py and load encoder/dinov2-small with
`local_files_only=True`. Recorded encoding used BF16; CPU FP32 encoding is a
different numerical setting and does not reproduce the frozen cache bitwise.

Source/weights load offline without an external LeWM checkout. The verification
sample is one actual validation feature window, with exact recording/frame
provenance; it checks API/weight parity and is not a benchmark result. Every
package was loaded and run in an isolated CPU subprocess using only bundled
project source. Encoder files were checked against the cache's pinned SHA256s.

Inference packages intentionally omit optimizer/RNG files; original run folders
retain resumable best/last states. Training journals, configs, source hashes,
protocol, and normalization are retained here. Source videos are not included.
Consult licenses/THIRD_PARTY_NOTICES.md for separate source/encoder/data terms.
"""


def tex_number(value, precision=4):
    text = f"{float(value):.{precision}g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    return mantissa + r"\!\times\!10^{" + str(int(exponent)) + "}"


def render_paper_tables(report):
    """Result floats only: called after the complete campaign/release gate."""
    populations = {(entry["aggregate"]["camera"], entry["aggregate"]["horizon"]): entry for entry in report["populations"]}
    if report.get("completed_runs") != 12 or report.get("completed_evaluations") != 48 or set(populations) != set(POPULATIONS):
        raise ValueError("Paper tables require the complete registered campaign")
    primary = populations[(evaluate.CAMERAS[0], 5)]
    aggregate = primary["aggregate"]
    methods = ["framewise", "constant_dynamics", "factorized", "action_free", "persistence", "constant_velocity"]
    names = {**METHOD_NAMES, "factorized": r"\textbf{ShiftWM (ours)}", "constant_velocity": "Constant velocity"}

    def cell(entry, mode, horizon):
        metric = f"h{horizon}_standardized_mse"
        if mode in entry["aggregate"]["methods"]:
            value = entry["aggregate"]["methods"][mode][metric]
            return "$" + tex_number(value["mean"]) + r"\pm" + tex_number(value["seed_sd"], 2) + "$"
        return "$" + tex_number(entry["aggregate"]["fixed_support_baselines"][mode][metric]) + "$"

    reference = aggregate["methods"]["framewise"]["h5_standardized_mse"]["mean"]
    lines = ["% Generated only after 12 complete runs, 48 validated evaluations, and portable-release verification.",
             r"\begin{table}[t]\centering\small", r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular}{lrrrr}\toprule", r"Method & MSE@1 $\downarrow$ & MSE@3 $\downarrow$ & MSE@5 $\downarrow$ & Gain@5 (\%) $\uparrow$\\\midrule"]
    for mode in methods:
        if mode == "persistence":
            lines.append(r"\midrule")
        if mode == "factorized":
            lines.append(r"\rowcolor{orange!9}")
        value = (aggregate["methods"][mode]["h5_standardized_mse"]["mean"] if mode in aggregate["methods"]
                 else aggregate["fixed_support_baselines"][mode]["h5_standardized_mse"])
        gain = 100*(reference-value)/reference if reference > 0 else None
        text = "---" if gain is None else f"{gain:+.2f}"
        if mode == "framewise":
            text = "reference"
        elif mode == "factorized" and gain is not None and gain > 0:
            text = r"\positivegain{" + text + "}"
        lines.append(names[mode] + " & " + " & ".join(cell(primary, mode, h) for h in (1, 3, 5)) + " & " + text + r"\\")
    lines += [r"\bottomrule\end{tabular}"]
    intervals = []
    for comparator in ("framewise", "constant_dynamics"):
        rows = [row for row in primary["gains"] if row["reference"] == comparator and row["metric"] == "h5_standardized_mse"]
        if len(rows) != 1:
            raise ValueError("Primary h5 paired interval is missing or ambiguous")
        row = rows[0]
        interval = "$" + tex_number(row["mean_difference"]) + r"\;[" + tex_number(row["ci95"][0]) + "," + tex_number(row["ci95"][1]) + "]$"
        qualifier = "includes zero" if row["ci95"][0] <= 0 <= row["ci95"][1] else "excludes zero"
        intervals.append(METHOD_NAMES[comparator] + ": " + interval + " (" + qualifier + ")")
    lines += [r"\caption{\textbf{Recorded DROID: primary external-camera-1 forecasting.} "
              r"Train-standardized feature MSE at 1, 3 and 5 action blocks (5, 15 and 25 native transitions); lower is better. "
              r"Learned entries show three-seed mean $\pm$ sample SD; support-only baselines have no training-seed SD. "
              r"Gain@5 is $100(\mathrm{Framewise}-\mathrm{method})/\mathrm{Framewise}$, a point reduction, not percentage points. "
              r"\positivegain{Bold green} marks a positive point reduction, not significance. "
              r"At five blocks, ours minus the named reference in MSE with 95\% paired session/seed bootstrap intervals is "
              + "; ".join(intervals) + ". Intervals are unadjusted for multiple comparisons; zero-crossing intervals are inconclusive.}",
              r"\label{tab:real-video-primary}\end{table}", "",
              r"\begin{table}[t]\centering\footnotesize",
              r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}",
              r"\begin{tabular}{lrrrr}\toprule",
              r"& \multicolumn{2}{c}{External camera 1} & \multicolumn{2}{c}{External camera 2}\\",
              r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
              r"Method & 5 blocks & 10 blocks & 5 blocks & 10 blocks\\\midrule"]
    for mode in methods:
        if mode == "persistence":
            lines.append(r"\midrule")
        if mode == "factorized":
            lines.append(r"\rowcolor{orange!9}")
        lines.append(names[mode] + " & " + " & ".join(cell(populations[key], mode, key[1]) for key in POPULATIONS) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{\textbf{Separate recorded-video populations.} Final-horizon standardized MSE; lower is better. "
              r"External-camera-1/five-block is primary; camera 2 measures natural camera transfer, and ten blocks measure horizon extrapolation. "
              r"The same session-disjoint test recordings are considered, but complete-window eligibility can differ by horizon. "
              r"Three-seed means and sample SD are shown for learned methods. These feature forecasts do not measure closed-loop robot success. "
              r"Raw MSE, cosine error, all paired intervals and reversed-action diagnostics are retained in the source-linked results report.}",
              r"\label{tab:real-video-transfer}\end{table}", ""]
    return "\n".join(lines)


def file_inventory(stage):
    return {str(path.relative_to(stage)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(stage.rglob("*")) if path.is_file() and path != stage / "manifest.json"}


def retire_paper_publication(table_path, release_root):
    """Keep old evidence, but never leave its table as a new run's ready signal."""
    paths = [table_path, table_path.with_suffix(".sources.json")]
    if not any(path.exists() for path in paths):
        return
    archive = release_root.parent / ".real_droid_table_history" / uuid.uuid4().hex
    archive.mkdir(parents=True)
    # Remove the conditional-input signal first. If interrupted between moves,
    # an old ledger alone cannot publish an apparently complete result table.
    for path in paths:
        if path.exists():
            path.replace(archive / path.name)


def finalize(campaign_path, results_root, report_prefix, release_root, encoder_root, draws=10000,
             paper_table="paper/tables/real_video_results.tex"):
    campaign_path, release_root, report_prefix = map(resolved, (campaign_path, release_root, report_prefix))
    release_root.parent.mkdir(parents=True, exist_ok=True)
    with (release_root.parent / ".real_droid_finalize.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        table_path = resolved(paper_table)
        retire_paper_publication(table_path, release_root)
        records, populations = collect_campaign(campaign_path, results_root)
        generations = release_root.parent / ("." + release_root.name + ".generations")
        generations.mkdir(exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="build-", dir=generations))
        committed = False
        try:
            source_bundle(stage, campaign_path, records)
            encoder_bundle(stage, encoder_root, records[0]["cache_manifest"]["identity"]["encoder"])
            (stage / "reports").mkdir()
            entries = []
            for (camera, horizon), paths in populations.items():
                output = stage / "reports" / f"{camera}_h{horizon}.json"
                aggregate = evaluate.aggregate(paths, output, draws=draws)
                entries.append({"aggregate": aggregate, "gains": gain_rows(aggregate)})
                for path in paths:
                    destination = stage / "reports/raw" / path.relative_to(resolved(results_root))
                    checked_copy(path, destination, aggregate["sources"][str(path.resolve())])
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            sample = validation_sample(records[0], stage)
            models = [export_model(record, stage, sample) for record in records]
            verification = verify_offline(stage)
            report = {"status": "completed", "completed_runs": 12, "completed_evaluations": 48,
                      "created_utc": datetime.now(timezone.utc).isoformat(), "populations": entries,
                      "models": models, "release_path": str(release_root),
                      "campaign_sha256": sha256(campaign_path), "reporter_sha256": sha256(__file__),
                      "offline_verification": verification,
                      "scope": "prespecified recorded DROID subset; not closed-loop control or the complete published benchmark"}
            train.atomic_json(report, stage / "reports/results.json")
            markdown = render_report(report)
            (stage / "reports/results.md").write_text(markdown)
            table = render_paper_tables(report)
            (stage / "reports/real_video_results.tex").write_text(table)
            (stage / "README.md").write_text(release_readme())
            inventory = file_inventory(stage)
            manifest = {"format_version": 1, "status": "completed", "kind": "real_droid_inference_release_v1",
                        "models": models, "files": inventory, "offline_verification": verification,
                        "total_bytes": sum(row["bytes"] for row in inventory.values()),
                        "campaign_sha256": sha256(campaign_path), "reporter_sha256": sha256(__file__)}
            train.atomic_json(manifest, stage / "manifest.json")
            # Recheck the exact published bytes after all verification/copy work.
            for name, entry in manifest["files"].items():
                if sha256(stage / name) != entry["sha256"]:
                    raise ValueError("Release content changed after inventory")
            if release_root.exists() and not release_root.is_symlink():
                raise ValueError("Refusing to overwrite an unrelated release directory")
            pointer = release_root.with_name("." + release_root.name + "." + uuid.uuid4().hex)
            try:
                os.symlink(os.path.relpath(stage, release_root.parent), pointer)
                os.replace(pointer, release_root)
            finally:
                pointer.unlink(missing_ok=True)
            committed = True
            report_prefix.parent.mkdir(parents=True, exist_ok=True)
            train.atomic_json(report, report_prefix.with_suffix(".json"))
            temporary = report_prefix.with_suffix(".md." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(markdown)
            temporary.replace(report_prefix.with_suffix(".md"))
            table_path.parent.mkdir(parents=True, exist_ok=True)
            ledger = {"status": "completed", "completed_runs": 12, "completed_evaluations": 48,
                      "table_sha256": hashlib.sha256(table.encode()).hexdigest(),
                      "release_manifest": str(release_root / "manifest.json"),
                      "release_manifest_sha256": sha256(release_root / "manifest.json"),
                      "report_json": str(report_prefix.with_suffix(".json")),
                      "report_json_sha256": sha256(report_prefix.with_suffix(".json")),
                      "reporter_sha256": sha256(__file__),
                      "evaluation_sources": {path: value for entry in entries for path, value in entry["aggregate"]["sources"].items()}}
            train.atomic_json(ledger, table_path.with_suffix(".sources.json"))
            # The table is the final publication signal observed by the paper
            # watcher; source ledger and verified release are committed first.
            temporary = table_path.with_name(table_path.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(table)
            temporary.replace(table_path)
            return manifest
        finally:
            if not committed:
                shutil.rmtree(stage, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="configs/real_video/campaign.json")
    parser.add_argument("--results-root", default="results/real_video/droid_selected_v1")
    parser.add_argument("--report-prefix", default="reports/real_droid_results")
    parser.add_argument("--release-root", default="artifacts/releases/real_droid_v1")
    parser.add_argument("--encoder-root", default="data/pretrained/dinov2-small")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--paper-table", default="paper/tables/real_video_results.tex")
    args = parser.parse_args()
    manifest = finalize(args.campaign, args.results_root, args.report_prefix, args.release_root,
                        args.encoder_root, args.bootstrap_draws, args.paper_table)
    print(json.dumps({"status": manifest["status"], "models": len(manifest["models"]),
                      "bytes": manifest["total_bytes"], "release": str(resolved(args.release_root))}))


if __name__ == "__main__":
    main()
