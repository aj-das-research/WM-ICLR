#!/usr/bin/env python3
"""Fit one train-only shrinkage scalar per frozen model, evaluate validation.

Not launched automatically. The original test population is forbidden.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/real_video"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as training
from evaluate import evaluate_dataset
from diagnose import development_dataset
from shiftwm.real_video.data import atomic_json, sha256
from shiftwm.real_video_development import ResidualCalibratedWorldModel, residual_scale


def load_calibrated_package(path, device="cpu", base_checkpoint=None):
    package = json.loads(Path(path).read_text())
    if package.get("package_kind") != "development_train_only_residual_calibration" or package.get("format_version") != 1:
        raise ValueError("Unsupported calibration package")
    if package["fit"]["fit_split"] != "train":
        raise ValueError("Calibration did not use training-only moments")
    checkpoint = Path(base_checkpoint if base_checkpoint is not None else package["base_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path(path).resolve().parent / checkpoint
    if sha256(checkpoint / "model.pt") != package["base_checkpoint_sha256"]:
        raise ValueError("Base checkpoint changed")
    if sha256(ROOT / "src/shiftwm/real_video_development.py") != package["wrapper_sha256"]:
        raise ValueError("Calibration wrapper changed")
    expected = residual_scale(package["fit"]["displacement_energy"], package["fit"]["displacement_alignment"])
    if expected != package["fit"]["scale"]:
        raise ValueError("Calibration coefficient does not match training moments")
    base, _ = training.load_package(checkpoint, device)
    return ResidualCalibratedWorldModel(base, expected).to(device).eval()


def register(path):
    """Freeze checked source, protocol, checkpoint, and train/val payload bytes."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("A registered campaign is immutable")
    cache = ROOT / "data/features/droid_selected_v1"
    manifest = json.loads((cache / "manifest.json").read_text())
    dependencies = training.source_files()
    for relative in ("scripts/real_video_development/calibrate_residual.py",
                     "scripts/real_video_development/diagnose.py",
                     "src/shiftwm/real_video_development.py",
                     "reports/real_droid_residual_calibration_protocol.md"):
        source = ROOT / relative
        dependencies[str(source)] = sha256(source)
    for name in ("manifest.json", "training_statistics.json"):
        dependencies[str(cache / name)] = sha256(cache / name)
    payloads = {}
    counts = defaultdict(int)
    for row in manifest["episodes"]:
        if row["split"] not in ("train", "val"):
            continue
        record = row["cameras"]["exterior_image_1_left"]
        payload = cache / record["file"]
        actual = sha256(payload)
        if actual != record["sha256"]:
            raise ValueError("Training/validation payload hash does not match audited cache")
        payloads[str(payload)] = actual
        counts[row["split"]] += 1
    runs = []
    for mode in ("framewise", "constant_dynamics", "factorized", "action_free"):
        for seed in (0, 1, 2):
            directory = ROOT / f"runs/real_video/droid_{mode}_s{seed}"
            training.validate_completed(directory)
            config = json.loads((directory / "best/config.json").read_text())
            dependencies.update(config["metadata"]["identity"]["dependencies"])
            for filename in ("model.pt", "config.json", "package_manifest.json"):
                source = directory / "best" / filename
                dependencies[str(source)] = sha256(source)
            runs.append({"mode": mode, "seed": seed, "name": directory.name,
                         "base_checkpoint": str(directory / "best"), "model_sha256": sha256(directory / "best/model.pt")})
    training.verify_sources(dependencies)
    registration = {"status": "registered_before_execution", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": "fit full training, evaluate validation only; no test payload reads",
                    "precision": "FP32 inference; autocast off; TF32 off; FP64 scalar fitting moments",
                    "dependencies": dependencies, "payloads": payloads, "payload_episode_counts": dict(counts), "runs": runs}
    atomic_json(registration, path)
    return registration


def verify_registration(registration, verify_payloads=True):
    expected = {(mode, seed) for mode in ("framewise", "constant_dynamics", "factorized", "action_free") for seed in (0, 1, 2)}
    actual = {(row["mode"], row["seed"]) for row in registration["runs"]}
    if registration["status"] != "registered_before_execution" or actual != expected or len(registration["runs"]) != 12:
        raise ValueError("Require the complete registered 12-model matched campaign")
    training.verify_sources(registration["dependencies"])
    if verify_payloads:
        training.verify_sources(registration["payloads"])


@torch.inference_mode()
def fit_training_scale(model, dataset, device="cuda", batch_size=128):
    if any(row["split"] != "train" for row in dataset.episodes):
        raise ValueError("Residual scale must be fitted only on training recordings")
    model.to(device).eval()
    accumulators = defaultdict(lambda: [0., 0., 0])
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        features, actions = batch["features"].to(device), batch["actions"].to(device)
        support = features[:, :3]
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            prediction = model.predict(support, actions[:, :2], actions[:, 2:])
        predicted = (prediction - support[:, -1:]) / model.feature_std
        observed = (features[:, 3:] - support[:, -1:]) / model.feature_std
        energy = predicted.double().square().mean((1, 2)).cpu().numpy()
        alignment = (predicted.double() * observed.double()).mean((1, 2)).cpu().numpy()
        for index, episode in enumerate(batch["episode_index"].tolist()):
            accumulators[episode][0] += float(energy[index])
            accumulators[episode][1] += float(alignment[index])
            accumulators[episode][2] += 1
    energy = np.mean([value[0] / value[2] for value in accumulators.values()])
    alignment = np.mean([value[1] / value[2] for value in accumulators.values()])
    return {"fit_split": "train", "horizon": 10, "stride": 5,
            "aggregation": "mean all ten query steps and dimensions, windows within episode, then equal eligible episodes",
            "scale": residual_scale(energy, alignment), "displacement_energy": float(energy),
            "displacement_alignment": float(alignment), "eligible_episodes": len(accumulators),
            "windows": sum(value[2] for value in accumulators.values()),
            "train_payloads": {row["episode_id"]: row["cameras"]["exterior_image_1_left"]["sha256"] for row in dataset.episodes}}


def run(output, registration_path, device="cuda"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new development campaign directory")
    torch.set_num_threads(8)
    training.seed_everything(5192026)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run inside an allocated GPU job")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must be disabled")
    registration_path = Path(registration_path)
    registration = json.loads(registration_path.read_text())
    registration_hash = sha256(registration_path)
    verify_registration(registration)
    cache = ROOT / "data/features/droid_selected_v1"
    train_dataset = development_dataset(cache, "train", horizon=10, stride=5)
    validation = {horizon: development_dataset(cache, "val", horizon=horizon, stride=5) for horizon in (5, 10)}
    output.mkdir(parents=True)
    protocol = ROOT / "reports/real_droid_residual_calibration_protocol.md"
    rows = []
    for mode in ("framewise", "constant_dynamics", "factorized", "action_free"):
        for seed in (0, 1, 2):
            if sha256(registration_path) != registration_hash:
                raise ValueError("Campaign registration changed during execution")
            verify_registration(registration, verify_payloads=False)
            run_dir = ROOT / f"runs/real_video/droid_{mode}_s{seed}"
            training.validate_completed(run_dir)
            base, _ = training.load_package(run_dir / "best", device)
            fitted = fit_training_scale(base, train_dataset, device)
            package = {"package_kind": "development_train_only_residual_calibration", "format_version": 1,
                       "mode": mode, "seed": seed, "base_checkpoint": str(run_dir / "best"),
                       "base_checkpoint_sha256": sha256(run_dir / "best/model.pt"), "fit": fitted,
                       "source_sha256": sha256(__file__),
                       "wrapper_sha256": sha256(ROOT / "src/shiftwm/real_video_development.py"),
                       "protocol_sha256": sha256(protocol), "test_evaluated": False,
                       "cache_manifest_sha256": sha256(cache / "manifest.json"),
                       "registration_sha256": registration_hash,
                       "precision": registration["precision"]}
            model = ResidualCalibratedWorldModel(base, fitted["scale"]).to(device).eval()
            name = f"droid_{mode}_s{seed}"
            atomic_json(package, output / name / "calibration.json")
            result = {"mode": mode, "seed": seed, "scale": fitted["scale"], "split": "val", "horizons": {}}
            for horizon, dataset in validation.items():
                reference = evaluate_dataset(base, dataset, device)
                calibrated = evaluate_dataset(model, dataset, device)
                result["horizons"][str(horizon)] = {"base": reference, "calibrated": calibrated}
            # Offline CPU reconstruction from the calibration JSON, explicitly
            # exercising the override used when an immutable base is relocated.
            reloaded = load_calibrated_package(output / name / "calibration.json", device="cpu", base_checkpoint=run_dir / "best")
            model.to("cpu").eval()
            batch = next(iter(DataLoader(validation[10], batch_size=4, shuffle=False)))
            with torch.inference_mode(), torch.autocast("cpu", enabled=False):
                first = model.predict(batch["features"][:, :3], batch["actions"][:, :2], batch["actions"][:, 2:])
                second = reloaded.predict(batch["features"][:, :3], batch["actions"][:, :2], batch["actions"][:, 2:])
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            result["offline_reload"] = {"status": "passed", "device": "cpu", "max_abs_error": float((first - second).abs().max()),
                                        "base_path_override_exercised": True, "windows": len(first), "horizon": 10}
            verify_registration(registration, verify_payloads=False)
            atomic_json(result, output / name / "validation.json")
            rows.append({"mode": mode, "seed": seed, "scale": fitted["scale"], "h5": result["horizons"]["5"]["calibrated"]["summary"]["model"]["h5_standardized_mse"],
                         "h10": result["horizons"]["10"]["calibrated"]["summary"]["model"]["h10_standardized_mse"]})
            print(json.dumps(rows[-1]), flush=True)
            atomic_json({"status": "running", "test_evaluated": False, "runs": rows, "registration_sha256": registration_hash}, output / "campaign.partial.json")
            del base, model, reloaded
            if device == "cuda":
                torch.cuda.empty_cache()
    verify_registration(registration)
    if len(rows) != 12 or sha256(registration_path) != registration_hash:
        raise ValueError("Incomplete or changed matched campaign")
    atomic_json({"status": "completed", "test_evaluated": False, "runs": rows, "protocol_sha256": sha256(protocol),
                 "registration_sha256": registration_hash, "registration_path": str(registration_path),
                 "payloads_verified_before_and_after": True, "all_offline_cpu_reloads_exact": True,
                 "completed_utc": datetime.now(timezone.utc).isoformat()}, output / "campaign.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="runs/real_video_development/residual_calibration_v1")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--registration", default="configs/real_video_development/residual_calibration_v1.json")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        result = register(args.registration)
        print(json.dumps({"status": result["status"], "payload_episode_counts": result["payload_episode_counts"]}))
    else:
        run(args.output, args.registration, args.device)
