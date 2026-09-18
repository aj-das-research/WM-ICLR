#!/usr/bin/env python3
"""Verify portable extension checkpoints using only bundled source and examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    release = args.release.resolve()
    source = release / "source/src"
    sys.path.insert(0, str(source))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    attempted = []

    def deny(*args, **kwargs):
        attempted.append("outgoing socket connection")
        raise RuntimeError("Network access is forbidden in portable inference verification")

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    import numpy as np
    import torch
    import shiftwm.extensions.model as module
    import shiftwm.upstream as upstream
    from shiftwm.extensions.checkpoint import load_package, read_package

    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError("The model was imported from outside the release")
    if (upstream.ROOT / "external/le-wm").exists():
        raise RuntimeError("Offline check must use bundled vendor code, not an external checkout")
    torch.set_num_threads(1)
    results = []
    for case in json.loads((release / "verification_cases.json").read_text()):
        started = time.monotonic()
        package = release / case["package"]
        if package.is_symlink() or any(path.is_symlink() for path in package.rglob("*")):
            raise RuntimeError("Release package contains symlinks")
        read_package(package, require_training=True)
        model, state = load_package(package, device="cpu")
        model.eval().requires_grad_(False)
        example_path = release / case["example"]
        metadata = json.loads(example_path.with_suffix(".json").read_text())
        if sha256(example_path) != metadata["example_sha256"]:
            raise ValueError("Example identity changed")
        with np.load(example_path, allow_pickle=False) as example:
            features = torch.from_numpy(example["features"].copy()).float()
            past = torch.from_numpy(example["past_actions"].copy()).float()
            future = torch.from_numpy(example["future_actions"].copy()).float()
            images = torch.from_numpy(example["support_rgb"].copy()).permute(0, 1, 4, 2, 3).float() / 255
        with torch.inference_mode():
            predicted = model.rollout_features(features, past, future)
            zero = model.rollout_features(features, past, torch.zeros_like(future))
            encoded = model.encode_images(images)
            rgb_prediction = model.rollout(images, past, future)
            repeat = model.rollout_features(features, past, future)
        expected = (1, 5, model.latent_dim)
        if any(value.shape != expected or not torch.isfinite(value).all()
               for value in (predicted, zero, rgb_prediction)):
            raise ValueError("Unexpected shape or nonfinite prediction")
        if not torch.equal(predicted, repeat):
            raise ValueError("Inference is not deterministic in evaluation mode")
        if torch.equal(predicted, zero):
            raise ValueError("The exported predictor did not respond to different candidate actions")
        if not torch.isfinite(encoded).all() or encoded.shape != features.shape:
            raise ValueError("Bundled visual encoder failed on actual observed RGB")
        # The cache is float32; CPU/GPU arithmetic can still differ. Report the
        # measured discrepancy and check a declared cross-device tolerance.
        difference = float((encoded - features).abs().max())
        torch.testing.assert_close(encoded, features, rtol=5e-3, atol=2e-3)
        results.append({"key": case["key"], "package": case["package"],
                        "status": "passed", "selected_epoch": state["epoch"],
                        "model_sha256": sha256(package / "model.pt"),
                        "example_sha256": sha256(example_path), "example_trajectory_id": metadata["trajectory_id"],
                        "output_shape": list(expected), "cached_and_rgb_outputs_finite": True,
                        "repeated_forward_exact": True, "candidate_action_sensitivity": True,
                        "cached_vs_new_cpu_encoding_max_abs": difference,
                        "cached_vs_new_cpu_encoding_tolerance": {"rtol": 5e-3, "atol": 2e-3},
                        "seconds": time.monotonic() - started})
    if attempted:
        raise RuntimeError("An inference dependency attempted network access")
    result = {"status": "passed", "verified_cases": len(results), "cases": results,
              "imported_model_source": str(Path(module.__file__).resolve().relative_to(release)),
              "upstream_source": "bundled pinned LeWM vendor", "network_attempts": attempted,
              "device": "cpu", "torch": torch.__version__,
              "interpretation": "strict actual-weight loading and API verification; not benchmark accuracy or control success"}
    (release / "offline_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
