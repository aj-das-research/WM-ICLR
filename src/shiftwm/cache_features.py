"""Cache frozen pretrained latents once for all subsequent model comparisons.

All four photometric conditions are cached, but train loaders independently
enforce allowed factor combinations. File existence never grants train access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import APPEARANCES, pixels_to_tensor, validate_manifest
from .model import ModelConfig, ShiftWorldModel
from .upstream import load_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Released upstream config.json + weights.pt directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--follow", action="store_true", help="Encode completed episodes while generation is still running")
    parser.add_argument("--idle-timeout", type=int, default=7200, help="Fail if generation makes no progress for this many seconds")
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset_manifest_path = args.data / "manifest.json"
    if not args.follow and not dataset_manifest_path.exists():
        raise FileNotFoundError(dataset_manifest_path)
    base, config, provenance = load_base(args.checkpoint)
    action_dim = config["action_encoder"]["input_dim"]
    model = ShiftWorldModel(base, config, ModelConfig(mode="frozen"),
                            np.zeros(action_dim), np.ones(action_dim), provenance).to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "dataset_root": str(args.data.resolve()),
                "encoder_weights_sha256": provenance["weights_sha256"],
                "preprocessing": "RGB uint8->float/255, appearance affine transform, ImageNet normalization, model resizing",
                "appearance_factors": APPEARANCES, "dtype": "float32", "latent_dim": model.latent_dim,
                "base_checkpoint": str(args.checkpoint.resolve())}
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata["cache_signature"] = signature
    metadata_path = args.output / "manifest.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["cache_signature"] != signature:
            raise ValueError("Refusing incompatible feature cache overwrite")
    (args.output / "pending_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start, frame_count, completed = time.time(), 0, set()
    last_progress = time.time()
    with torch.inference_mode():
        while True:
            if dataset_manifest_path.exists():
                raw_manifest = dataset_manifest_path.read_bytes()
                manifest = json.loads(raw_manifest)
                validate_manifest(manifest)
                episodes, finalized = manifest["episodes"], True
            else:
                episodes, finalized = [], False
                for path in sorted((args.data / "episodes").glob("*.json")):
                    try:
                        episodes.append(json.loads(path.read_text()))
                    except json.JSONDecodeError:
                        # Sidecar is still being written; read on next iteration.
                        continue
            for episode in episodes:
                uid = episode["trajectory_id"]
                if uid in completed:
                    continue
                target = args.output / episode["file"]
                signature_path = target.with_suffix(".sha256")
                episode_signature = signature + ":" + episode["sha256"]
                if target.exists() and signature_path.exists() and signature_path.read_text().strip() == episode_signature:
                    completed.add(uid)
                    continue
                with np.load(args.data / episode["file"], allow_pickle=False) as source:
                    images, actions = source["images"], source["actions"]
                # Pool all appearances to fill accelerator batches rather than
                # launching four short 65-frame inference passes per episode.
                views = torch.cat([pixels_to_tensor(images, appearance) for appearance in sorted(APPEARANCES)])
                encoded = []
                for pixels in views.split(args.batch_size):
                    encoded.append(model.encode_images(pixels.to(args.device), reference=True,
                                   chunk_size=args.batch_size).cpu().numpy().astype(np.float32))
                features = np.concatenate(encoded).reshape(len(APPEARANCES), len(images), -1)
                if not np.isfinite(features).all():
                    raise FloatingPointError(f"Nonfinite features {uid}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".npz.partial")
                with temporary.open("wb") as f:
                    np.savez(f, features=features, actions=actions)
                temporary.replace(target)
                signature_path.write_text(episode_signature + "\n")
                frame_count += len(images) * len(APPEARANCES)
                completed.add(uid)
                last_progress = time.time()
                if len(completed) % 10 == 0:
                    print(json.dumps({"episodes": len(completed), "available": len(episodes), "data_finalized": finalized,
                                      "frames_encoded": frame_count, "elapsed_seconds": time.time() - start}), flush=True)
            if finalized:
                metadata["dataset_manifest_sha256"] = hashlib.sha256(raw_manifest).hexdigest()
                metadata["episodes"] = len(episodes)
                break
            if time.time() - last_progress > args.idle_timeout:
                raise TimeoutError("Data generation made no progress within --idle-timeout")
            time.sleep(10)
    metadata["elapsed_seconds"] = time.time() - start
    (args.output / "pending_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "pending_manifest.json").replace(metadata_path)
    print(json.dumps({"feature_cache": str(metadata_path), "elapsed_seconds": time.time() - start}), flush=True)


if __name__ == "__main__":
    main()
