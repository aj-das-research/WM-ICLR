#!/usr/bin/env python3
"""Development diagnostics using train/validation payloads only.

This file never evaluates test recordings. Perturbed actions are observational
sensitivity probes, not physically executed counterfactual targets.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/real_video"))
import train as training
from shiftwm.real_video.data import RealVideoDataset, atomic_json, sha256

PERTURBATIONS = ("recorded", "reversed", "permuted_recording", "raw_zero", "train_mean", "hold_support_command")


def development_dataset(root, split, **kwargs):
    if split not in ("train", "val"):
        raise ValueError("Development diagnostics must never load test payloads")
    return RealVideoDataset(root, split, **kwargs)


def donor_indices(episode_indices, sessions):
    """Deterministic donor recording from a distinct session, independent of batch."""
    unique = sorted(set(episode_indices))
    first = {episode: episode_indices.index(episode) for episode in unique}
    donors = {}
    for pos, episode in enumerate(unique):
        candidates = unique[pos + 1:] + unique[:pos]
        donor = next((other for other in candidates if sessions[other] != sessions[episode]), None)
        if donor is None:
            raise ValueError("Need a donor from a different recording session")
        donors[episode] = first[donor]
    return np.asarray([donors[episode] for episode in episode_indices])


def perturb_future(name, future, support_actions, action_mean, donor=None):
    if name == "recorded":
        return future
    if name == "reversed":
        return future.flip(1)
    if name == "permuted_recording":
        if donor is None or donor.shape != future.shape:
            raise ValueError("Missing matched donor command sequence")
        return donor
    if name == "raw_zero":
        return torch.zeros_like(future)
    if name == "train_mean":
        return action_mean[None, None].expand_as(future)
    if name == "hold_support_command":
        # Each block stores five native 7D commands. Repeat the final observed
        # native command, not the complete (potentially changing) support block.
        return support_actions[:, -1, -7:].repeat(1, 5)[:, None].expand_as(future)
    raise ValueError(name)


def descriptors(features, actions, feature_std, action_std):
    support = features[:, :3]
    future = features[:, 3:]
    return {
        "support_motion": ((support[:, 1:] - support[:, :-1]) / feature_std).square().mean((1, 2)),
        "future_motion": ((future - support[:, -1:]) / feature_std).square().mean((1, 2)),
        "action_variation": ((actions[:, 3:] - actions[:, 2:-1]) / action_std).square().mean((1, 2)),
    }


def episode_mean(values, episode_ids, mask=None):
    values = np.asarray(values, dtype=np.float64)
    mask = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask)
    selected = sorted(set(np.asarray(episode_ids)[mask]))
    if not selected:
        return None
    return np.mean([values[mask & (np.asarray(episode_ids) == episode)].mean(0) for episode in selected], axis=0).tolist()


def aggregate_rows(arrays, ids, thresholds):
    strata = {"all": np.ones(len(ids), dtype=bool)}
    for name, boundaries in thresholds.items():
        bins = np.searchsorted(boundaries, arrays[name], side="right")
        for index in range(3):
            strata[f"{name}_tertile_{index + 1}"] = bins == index
    result = {}
    for name, mask in strata.items():
        result[name] = {"windows": int(mask.sum()), "episodes": len(set(np.asarray(ids)[mask])),
                        "metrics": {key: episode_mean(value, ids, mask) for key, value in arrays.items()}}
    return result


@torch.inference_mode()
def diagnose_model(model, dataset, device, thresholds, window_indices=None, batch_size=128, probes=True):
    model.to(device).eval()
    if window_indices is None:
        window_indices = list(range(len(dataset)))
    windows = [dataset.windows[index] for index in window_indices]
    episode_indices = [episode for episode, _ in windows]
    sessions = {i: row["session_id"] for i, row in enumerate(dataset.episodes)}
    donors = donor_indices(episode_indices, sessions)
    donor_actions = torch.stack([dataset[window_indices[int(index)]]["actions"][2:] for index in donors])
    loader = DataLoader(Subset(dataset, window_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    values = defaultdict(list)
    ids, rows = [], []
    offset = 0
    for batch in loader:
        features = batch["features"].to(device)
        actions = batch["actions"].to(device)
        support, target = features[:, :3], features[:, 3:]
        past, future = actions[:, :2], actions[:, 2:]
        baseline = support[:, -1:].expand_as(target)
        motion = descriptors(features, actions, model.feature_std, model.action_std)
        for key, value in motion.items():
            values[key].append(value.cpu().numpy())
        targets = (target - baseline) / model.feature_std
        values["persistence_mse"].append(targets.square().mean(-1).cpu().numpy())
        predictions = model.predict(support, past, future)
        residual = (predictions - baseline) / model.feature_std
        values["model_mse"].append(((predictions - target) / model.feature_std).square().mean(-1).cpu().numpy())
        values["predicted_displacement_mse"].append(residual.square().mean(-1).cpu().numpy())
        values["displacement_alignment"].append((residual * targets).mean(-1).cpu().numpy())
        values["prediction_step_mse"].append(torch.diff(torch.cat((support[:, -1:], predictions), 1), dim=1).div(model.feature_std).square().mean(-1).cpu().numpy())
        if probes:
            donor = donor_actions[offset:offset + len(features)].to(device)
            for name in PERTURBATIONS[1:]:
                changed = perturb_future(name, future, past, model.action_mean, donor)
                alternate = model.predict(support, past, changed)
                values[name + "_mse"].append(((alternate - target) / model.feature_std).square().mean(-1).cpu().numpy())
                values[name + "_prediction_change"].append(((alternate - predictions) / model.feature_std).square().mean(-1).cpu().numpy())
                values[name + "_command_change"].append(((changed - future) / model.action_std).square().mean((1, 2)).cpu().numpy())
        # Oracle refresh is a diagnosis, not an eligible deployment baseline:
        # it supplies the actual preceding observations at each prediction step.
        refreshed = torch.cat([model.predict(features[:, step:step+3], actions[:, step:step+2],
                                             actions[:, step+2:step+3])
                               for step in range(dataset.horizon)], 1)
        values["oracle_observation_refresh_mse"].append(((refreshed - target) / model.feature_std).square().mean(-1).cpu().numpy())
        ids.extend(batch["episode_index"].tolist())
        rows.extend({"episode_id": dataset.episodes[i]["episode_id"], "session_id": dataset.episodes[i]["session_id"],
                     "window_start": int(start)} for i, start in zip(batch["episode_index"].tolist(), batch["window_start"].tolist()))
        offset += len(features)
    arrays = {key: np.concatenate(value) for key, value in values.items()}
    result = {"aggregation": "mean windows within episode in each stratum, then equal episodes",
              "horizon": dataset.horizon, "windows": len(rows), "episodes": len(set(ids)),
              "sessions": len({row["session_id"] for row in rows}),
              "strata": aggregate_rows(arrays, ids, thresholds), "rows": rows,
              "window_arrays": {key: value.tolist() for key, value in arrays.items()}}
    return result


@torch.inference_mode()
def pooling_probe(device, limit=48):
    """Quantify information loss from pooling on deterministic TRAIN frame pairs.

    No object annotations exist here, so this cannot identify manipulated-object
    information or establish pooling as the causal source of forecasting error.
    """
    from transformers import Dinov2Model
    from torch.nn import functional as F
    root = ROOT / "data/real_video/droid_selected/processed"
    manifest = json.loads((root / "manifest.json").read_text())
    rows = sorted((row for row in manifest["episodes"] if row["split"] == "train"),
                  key=lambda row: hashlib.sha256(("pooling-diagnosis-v1:" + row["episode_id"]).encode()).hexdigest())
    encoder_root = ROOT / "data/pretrained/dinov2-small"
    encoder = Dinov2Model.from_pretrained(encoder_root, local_files_only=True).to(device).eval()
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    output = []
    for row in rows:
        record = row["cameras"]["exterior_image_1_left"]
        path = root / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Changed original training RGB payload")
        with np.load(path, allow_pickle=False) as data:
            if len(data["images"]) < 8:
                continue
            images = data["images"][[2, 7]]
        pixels = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255
        pixels = F.interpolate(pixels, (224, 224), mode="bilinear", align_corners=False, antialias=True)
        with torch.autocast(device_type=str(device).split(":")[0], enabled=str(device).startswith("cuda"), dtype=torch.bfloat16):
            patches = encoder(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:].float()
        delta = (patches[1] - patches[0]).T.reshape(1, 384, 16, 16)
        full_energy = float(delta.square().mean())
        pooled_energy = {str(size): float(F.adaptive_avg_pool2d(delta, (size, size)).square().mean()) for size in (2, 4, 8, 16)}
        rgb_delta = (images[1].astype(np.float32) - images[0].astype(np.float32)) / 255
        output.append({"episode_id": row["episode_id"], "session_id": row["session_id"], "split": "train",
                       "payload_sha256": record["sha256"], "stored_indices": [2, 7],
                       "raw_feature_delta_mse": full_energy, "pooled_feature_delta_mse": pooled_energy,
                       "retained_delta_energy_fraction": {k: v / max(full_energy, 1e-20) for k, v in pooled_energy.items()},
                       "rgb_delta_mse": float(np.mean(rgb_delta ** 2))})
        if len(output) == limit:
            break
    return {"split": "train", "pairs": output,
            "median_retained_fraction": {str(size): float(np.median([row["retained_delta_energy_fraction"][str(size)] for row in output])) for size in (2, 4, 8, 16)},
            "limitation": "Pooling contracts spatial feature changes by construction; no object masks, no evidence that contraction causes forecasting failures."}


def run(output, device="cuda"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite an existing diagnostic run")
    torch.set_num_threads(8)
    training.seed_everything(5192026)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("An allocated GPU is required")
    cache = ROOT / "data/features/droid_selected_v1"
    train_dataset = development_dataset(cache, "train", horizon=10, stride=5)
    val_dataset = development_dataset(cache, "val", horizon=10, stride=5)
    reference, _ = training.load_package(ROOT / "runs/real_video/droid_factorized_s0/best", device="cpu")
    populations = defaultdict(list)
    for batch in DataLoader(train_dataset, batch_size=256):
        for key, value in descriptors(batch["features"], batch["actions"], reference.feature_std, reference.action_std).items():
            populations[key].append(value.numpy())
    thresholds = {key: np.quantile(np.concatenate(value), [1 / 3, 2 / 3]).tolist() for key, value in populations.items()}
    result = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(), "split_policy": "train and validation only; test metadata is visible but no test payload read",
              "source_sha256": sha256(__file__), "cache_manifest_sha256": sha256(cache / "manifest.json"),
              "threshold_fit_split": "train", "train_tertile_thresholds": thresholds, "runs": [],
              "diagnostic_protocol": "reports/real_droid_development_diagnosis_protocol.md",
              "diagnostic_protocol_sha256": sha256(ROOT / "reports/real_droid_development_diagnosis_protocol.md"),
              "caveats": ["Development results are not confirmatory held-out evidence.", "All horizon steps use the common horizon-10-eligible validation population.", "Perturbed commands were not physically executed; original targets cannot label their true counterfactual outcome.", "Raw-zero absolute position commands are only an offline sensitivity stress test.", "Oracle observation refresh uses actual future observations and cannot be a deployment baseline."],
              "validation_payloads": {row["episode_id"]: row["cameras"]["exterior_image_1_left"]["sha256"] for row in val_dataset.episodes}}
    selected_train = sorted(range(len(train_dataset.episodes)), key=lambda i: hashlib.sha256(("train-diagnosis-v1:" + train_dataset.episodes[i]["episode_id"]).encode()).hexdigest())[:64]
    train_windows = [i for i, (episode, _) in enumerate(train_dataset.windows) if episode in selected_train]
    for mode in ("framewise", "constant_dynamics", "factorized", "action_free"):
        for seed in (0, 1, 2):
            directory = ROOT / f"runs/real_video/droid_{mode}_s{seed}"
            summary = training.validate_completed(directory)
            for which in (("best", "last") if mode == "factorized" else ("best",)):
                model, state = training.load_package(directory / which, device=device)
                diagnostic = diagnose_model(model, val_dataset, device, thresholds)
                entry = {"mode": mode, "seed": seed, "checkpoint": which, "epoch": state["epoch"], "checkpoint_sha256": sha256(directory / which / "model.pt"),
                         "summary": summary, "validation": diagnostic}
                if mode == "factorized" and seed == 0:
                    entry["train_subset"] = diagnose_model(model, train_dataset, device, thresholds, train_windows, probes=False)
                result["runs"].append(entry)
                atomic_json(result, output.with_suffix(".partial.json"))
                print(json.dumps({"event": "diagnosed", "mode": mode, "seed": seed, "checkpoint": which, "val_mse_h10": diagnostic["strata"]["all"]["metrics"]["model_mse"][-1]}), flush=True)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
    result["pooling"] = pooling_probe(device)
    result["status"] = "completed"
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(result, output)
    print(json.dumps({"event": "completed", "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="reports/real_droid_development_diagnosis.json")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    run(args.output, args.device)
