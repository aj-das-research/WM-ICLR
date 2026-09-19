#!/usr/bin/env python3
"""Independently verify every simulator package using bundled files only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.release.resolve()
    sys.path.insert(0, str(root / "source/src"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    attempted = []

    def deny(*args, **kwargs):
        attempted.append("network connection")
        raise RuntimeError("Offline inference must not access the network")

    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    import numpy as np
    import torch
    import shiftwm.simulator_release as loader
    import shiftwm.upstream as upstream
    if not Path(loader.__file__).resolve().is_relative_to(root / "source/src"):
        raise ValueError("Inference source escaped the relocated release")
    if (upstream.ROOT / "external/le-wm").exists():
        raise ValueError("Independent verification must use vendored upstream source")
    torch.set_num_threads(1)
    records = json.loads((root / "models.json").read_text())["models"]
    if len(records) != 42 or len({r["id"] for r in records}) != 42:
        raise ValueError("All 42 unique models are mandatory")
    results = []
    for record in records:
        model, state = loader.load_package(root, record["id"])
        if tensor_digest(state["state_dict"]) != record["source_tensor_sha256"]:
            raise ValueError("Reassembled tensors differ from the original full checkpoint")
        example_path = root / record["example"]["path"]
        reference_path = root / record["reference"]["path"]
        if loader.sha256(example_path) != record["example"]["sha256"] or loader.sha256(reference_path) != record["reference"]["sha256"]:
            raise ValueError("Verification input or source forecast changed")
        with np.load(example_path, allow_pickle=False) as values:
            features = torch.from_numpy(values["features"].copy()).float()
            past = torch.from_numpy(values["past_actions"].copy()).float()
            future = torch.from_numpy(values["future_actions"].copy()).float()
            images = torch.from_numpy(values["support_rgb"].copy()).permute(0, 1, 4, 2, 3).float() / 255
        with torch.inference_mode():
            outputs = {"cached": model.rollout_features(features, past, future),
                       "rgb": model.rollout(images, past, future), "encoded": model.encode_images(images)}
            repeat = model.rollout_features(features, past, future)
            changed = model.rollout_features(features, past, torch.zeros_like(future))
        if not torch.equal(outputs["cached"], repeat) or torch.equal(outputs["cached"], changed):
            raise ValueError("Determinism or candidate-action sensitivity check failed")
        differences = {}
        with np.load(reference_path, allow_pickle=False) as reference:
            for key, value in outputs.items():
                expected = torch.from_numpy(reference[key].copy())
                if not torch.isfinite(value).all() or not torch.equal(value, expected):
                    raise ValueError("Original-versus-export exact CPU parity failed: " + record["id"] + "/" + key)
                differences[key] = float((value - expected).abs().max())
        torch.testing.assert_close(outputs["encoded"], features, rtol=5e-3, atol=2e-3)
        results.append({"id": record["id"], "status": "passed", "selected_epoch": state["epoch"],
            "source_tensor_sha256": record["source_tensor_sha256"], "all_tensors_exact": True,
            "source_export_max_abs": differences, "repeated_prediction_exact": True,
            "candidate_action_sensitivity": True, "shape": list(outputs["cached"].shape),
            "cached_vs_cpu_encoding_max_abs": float((outputs["encoded"] - features).abs().max()),
            "cached_vs_cpu_encoding_tolerance": {"rtol": 5e-3, "atol": 2e-3}})
        print(json.dumps({"verified": record["id"]}), flush=True)
    if attempted:
        raise ValueError("A dependency attempted network access")
    result = {"status": "passed", "models": results, "verified_models": 42, "device": "cpu",
        "torch": torch.__version__, "source": "relocated bundled source and pinned vendor code",
        "network_attempts": attempted,
        "scope": "One genuine validation support window per simulated domain; engineering portability only, not benchmark performance"}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
