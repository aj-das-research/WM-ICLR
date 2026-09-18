"""Real checkpoint inference for the research demo; no simulated outputs."""
from __future__ import annotations
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))
from shiftwm.checkpoint import load_package

torch.set_num_threads(int(os.environ.get("SHIFTWM_DEMO_THREADS", "2")))
DEVICE = os.environ.get("SHIFTWM_DEMO_DEVICE", "cpu")
LOCK = threading.Lock()
MODEL_CACHE = {}


@lru_cache(maxsize=128)
def _file_hash(path, size, modified_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def samples():
    return {item["id"]: item for item in json.loads((HERE / "samples/manifest.json").read_text())["samples"]}


def checkpoint_catalog():
    """Only actual frozen exports or fully trained runs; never partial training."""
    catalog = {}
    project = Path(os.environ.get("SHIFTWM_PROJECT_ROOT", str(HERE.parent)))
    paths = [(p.parent, "release") for p in sorted((project / "artifacts/releases").glob("*/config.json"))]
    paths += [(p.parent, "bundled") for p in sorted((HERE / "checkpoints").glob("*/config.json"))]
    paths += [(p.parent, "local") for p in sorted((project / "runs/world").glob("*/best/config.json"))]
    hashes = set()
    for directory, location in paths:
        model_file = directory / "model.pt"
        if not model_file.is_file():
            continue
        config = json.loads((directory / "config.json").read_text())
        mode = config["model_config"]["mode"]
        metadata = config.get("metadata", {})
        environment = metadata.get("environment") or Path(metadata.get("config", {}).get("pretrained_dir", "")).name
        if environment not in {"pusht", "reacher"}:
            continue
        release_file = directory / "release_manifest.json"
        if release_file.exists():
            release = json.loads(release_file.read_text())
            if release.get("status") != "ready" or release.get("verification", {}).get("status") != "passed":
                continue
            if release["model_sha256"] != sha(model_file):
                raise ValueError(f"Release checkpoint hash mismatch: {directory}")
        elif location == "release":
            continue
        if mode == "frozen":
            state = "frozen upstream export; no new fine-tuning"
        else:
            summary_file = (directory if release_file.exists() else directory.parent) / "training_summary.json"
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text())
            planned = metadata.get("config", {}).get("epochs")
            if summary.get("status") != "completed" or summary.get("completed_epochs") != planned:
                continue
            if environment == "pusht" and "pusht_relative" not in metadata["config"]["data_root"]:
                continue
            state = f"training complete ({planned} epochs); best validation checkpoint"
        digest = sha(model_file)
        if digest in hashes:
            continue
        hashes.add(digest)
        name = directory.parent.name if location == "local" else directory.name
        catalog[f"{environment} · {name}"] = {"directory": str(directory), "environment": environment,
                  "mode": mode, "status": state, "sha256": digest, "config": config,
                  "bytes": model_file.stat().st_size, "location": location}
    return catalog


def available(environment):
    return [name for name, checkpoint in checkpoint_catalog().items() if checkpoint["environment"] == environment]


def load_clip(sample_id):
    item = samples()[sample_id]
    path = HERE / item["file"]
    if sha(path) != item["sha256"]:
        raise ValueError("Bundled sample checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        return item, data["images"].copy(), data["actions"].copy()


def observed_images(images, appearance, specification):
    tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255
    transform = specification[str(int(appearance))]
    gain = torch.tensor(transform["gain"])[None, :, None, None]
    bias = torch.tensor(transform["bias"])[None, :, None, None]
    return (tensor * gain + bias).clamp(0, 1)


def model_for(checkpoint):
    key = checkpoint["sha256"]
    if key not in MODEL_CACHE:
        MODEL_CACHE.clear()  # Bound resident model memory to one checkpoint.
        model, state = load_package(checkpoint["directory"], device=DEVICE)
        model.requires_grad_(False)
        MODEL_CACHE[key] = (model, state)
    return MODEL_CACHE[key]


@torch.inference_mode()
def forecast(checkpoint_name, sample_id, appearance=2):
    catalog = checkpoint_catalog()
    if checkpoint_name not in catalog:
        raise ValueError("Checkpoint is unavailable or its training is incomplete; refresh choices")
    checkpoint = catalog[checkpoint_name]
    item, images, actions = load_clip(sample_id)
    if checkpoint["environment"] != item["environment"]:
        raise ValueError("Checkpoint and sample environment differ")
    appearance = int(appearance)
    observed = observed_images(images, appearance, item["appearances"])
    canonical = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255
    with LOCK:
        model, state = model_for(checkpoint)
        history = model.config.history_length
        if history != 3:
            raise ValueError("Bundled clips currently validate the three-observation context interface")
        started = time.perf_counter()
        # No future observation is encoded into the context or candidate inputs.
        support = model.encode_images(observed[:history].to(DEVICE))[None]
        past_actions = torch.tensor(actions[:history - 1], device=DEVICE)[None]
        candidate = torch.tensor(actions[history - 1:], device=DEVICE)[None]
        contexts = model.infer_context(support, past_actions)
        recorded_prediction = model.rollout_features(support, past_actions, candidate, contexts=contexts)
        zero_prediction = model.rollout_features(support, past_actions, torch.zeros_like(candidate), contexts=contexts)
        # Privileged canonical future images enter ONLY offline scoring.
        target = model.encode_images(canonical[history:].to(DEVICE), reference=True)[None]
        persistence = model.correct_observations(support, contexts[0])[:, -1:]
        arrays = {"recorded_action_mse": (recorded_prediction - target).square().mean(-1)[0].cpu().numpy(),
                  "zero_action_mismatch": (zero_prediction - target).square().mean(-1)[0].cpu().numpy(),
                  "persistence_mse": (persistence - target).square().mean(-1)[0].cpu().numpy(),
                  "action_effect_mse": (recorded_prediction - zero_prediction).square().mean(-1)[0].cpu().numpy()}
        elapsed = time.perf_counter() - started
    for values in arrays.values():
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite predictions; refusing to display a numerical result")
    rows = [{"horizon": i + 1, **{name: float(values[i]) for name, values in arrays.items()}}
            for i in range(len(candidate[0]))]
    record = {"kind": "single_heldout_clip_forecast", "checkpoint": checkpoint_name,
              "checkpoint_sha256": checkpoint["sha256"], "checkpoint_status": checkpoint["status"],
              "checkpoint_epoch": int(state["epoch"]), "model_mode": checkpoint["mode"],
              "sample": item, "appearance_id": appearance, "support_observations": history,
              "candidate_actions": actions[history - 1:].tolist(), "device": DEVICE,
              "elapsed_seconds_excluding_load": elapsed, "rows": rows,
              "interpretation": "Zero-action mismatch is scored against the recorded-action path, not counterfactual zero-action ground truth. No future video is generated. This single clip is not a benchmark aggregate."}
    rendered = (observed.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    return record, rendered, checkpoint


def render_plot(record):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.0), layout="constrained")
    values = record["rows"]
    for key, label, color, style in [("recorded_action_mse", "Recorded-action forecast", "#0072B2", "-"),
                                     ("zero_action_mismatch", "Zero-action sensitivity control", "#B75B16", "--"),
                                     ("persistence_mse", "Last-state persistence", "#707070", ":")]:
        ax.plot([r["horizon"] for r in values], [r[key] for r in values], marker="o", markersize=3,
                label=label, color=color, linestyle=style, linewidth=1.5)
    ax.set(xlabel="Forecast horizon (grouped action steps)",
           ylabel="Fixed-coordinate MSE", xticks=[r["horizon"] for r in values])
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    plt.close(fig)  # Return the renderable object without retaining pyplot state.
    return fig
