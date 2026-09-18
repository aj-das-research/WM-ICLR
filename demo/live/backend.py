"""Bounded, genuine CPU inference using released DROID feature predictors."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RELEASE = Path(os.environ.get("SHIFTWM_LIVE_RELEASE", PROJECT / "artifacts/releases/real_droid_v1")).resolve()
SAMPLES = HERE / "samples"
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
LOCK = threading.Lock()
MODELS = {}
MANIFEST = json.loads((SAMPLES / "manifest.json").read_text())
SAMPLE_MAP = {x["id"]: x for x in MANIFEST["samples"]}
sys.path.insert(0, str(RELEASE / "source/src"))
spec = importlib.util.spec_from_file_location("live_release_train", RELEASE / "source/scripts/real_video/train.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sample(sample_id, camera, horizon):
    if sample_id not in SAMPLE_MAP or camera not in (1, 2) or horizon not in (5, 10):
        raise ValueError("Choose a listed sample, camera 1 or 2, and horizon 5 or 10")
    item = SAMPLE_MAP[sample_id]["cameras"][str(camera)]
    path = SAMPLES / item["file"]
    if sha(path) != item["sha256"]:
        raise ValueError("Bundled input hash differs")
    with np.load(path, allow_pickle=False) as source:
        features = torch.tensor(source["features"][:3+horizon], dtype=torch.float32)[None]
        actions = torch.tensor(source["actions"][:2+horizon], dtype=torch.float32)[None]
    return features, actions, item


def model_for(mode, seed):
    key = (mode, seed)
    if key not in MODELS:
        package = RELEASE / f"models/droid_{mode}_s{seed}"
        model, state = training.load_package(package, device="cpu")
        model.requires_grad_(False)
        MODELS[key] = (model, {"mode": mode, "seed": seed, "epoch": state["epoch"],
            "checkpoint_sha256": sha(package / "model.pt")})
    return MODELS[key]


@torch.inference_mode()
def forecast(sample_id, camera=1, seed=0, horizon=5):
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Choose training seed 0, 1, or 2")
    if not LOCK.acquire(blocking=False):
        raise BlockingIOError("Another forecast is running. Please retry shortly.")
    try:
        features, actions, item = load_sample(sample_id, camera, horizon)
        loaded = {mode: model_for(mode, seed) for mode in ("factorized", "framewise")}
        support, target = features[:, :3], features[:, 3:]
        started = time.perf_counter()
        predictions = {mode: model.predict(support, actions[:, :2], actions[:, 2:])
                       for mode, (model, _) in loaded.items()}
        predictions["persistence"] = support[:, -1:].expand(-1, horizon, -1)
        std = loaded["factorized"][0].feature_std
        if not torch.equal(std, loaded["framewise"][0].feature_std):
            raise ValueError("Comparison models have different normalization")
        result = {}
        for mode, prediction in predictions.items():
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError("Invalid model output")
            squared = ((prediction - target) / std).square()[0]
            curve = squared.mean(-1).tolist()
            # Feature cache: [2,2,384] pooling positions, then flatten. This is
            # forecast error by pooled grid cell, not object/attention saliency.
            patch = squared.reshape(horizon, 4, 384).mean(-1).tolist()
            result[mode] = {"mse": curve, "patch_mse": patch,
                "prediction_sha256": hashlib.sha256(prediction.contiguous().numpy().tobytes()).hexdigest()}
        elapsed = time.perf_counter() - started
        return {"request_id": uuid.uuid4().hex, "sample_id": sample_id,
            "camera": camera, "seed": seed, "horizon": horizon, "device": "cpu",
            "inference_seconds": elapsed, "fresh_inference": True,
            "models": {mode: metadata for mode, (_, metadata) in loaded.items()},
            "input_sha256": item["sha256"], "support_native_frames": [x["native_frame"] for x in item["frames"][:3]],
            "query_native_frames": [x["native_frame"] for x in item["frames"]][3:3+horizon],
            "methods": result, "feature_shape": [horizon, 1536],
            "gain_percent_vs_framewise": 100 * (1-result["factorized"]["mse"][-1]/result["framewise"]["mse"][-1]),
            "scope": "First window, selected seed; measured latent feature forecasts using cached DINOv2 inputs. Query images enter scoring only. Illustrative examples, not aggregate benchmark results."}
    finally:
        LOCK.release()
