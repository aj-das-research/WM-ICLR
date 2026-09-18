"""Extract a training observation for the method's pairing illustration.

Pixels are copied losslessly, without crop, augmentation, or image synthesis.
The example is not selected using any model outcome.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/world/pusht_relative/episodes/train-s31002-d0.npz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = ROOT / "data/world/pusht_relative/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest["episodes"]
               if Path(entry["file"]).name == SOURCE.name]
    if len(entries) != 1 or entries[0]["split"] != "train" or entries[0]["dynamics_id"] != 0:
        raise ValueError("Illustrated source must be the specified canonical training episode")
    with np.load(SOURCE, allow_pickle=False) as episode:
        pixels = episode["images"][0].copy()
    if pixels.dtype != np.uint8 or pixels.shape != (224, 224, 3):
        raise ValueError("Expected the original 224-pixel RGB observation")
    target = OUTPUT / "method_train_other.png"
    Image.fromarray(pixels).save(target)
    if not np.array_equal(np.asarray(Image.open(target)), pixels):
        raise ValueError("Exported illustration changed the observed pixels")
    record = {
        "scope": "Actual training observation illustrating a permissible same-appearance batch pair",
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha(SOURCE),
        "dataset_manifest_sha256": sha(manifest_path),
        "trajectory_id": entries[0]["trajectory_id"], "split": "train",
        "frame_index": 0, "appearance_id": 0,
        "selection": "Next numbered seed after the existing split-figure training example; no model-outcome selection",
        "asset": str(target.relative_to(ROOT)), "sha256": sha(target),
        "operation": "Lossless uint8 RGB extraction, no crop, transformation, or synthesis",
        "generator_sha256": sha(Path(__file__)),
        "pairing_scope": "Different trajectories illustrate one allowed pair; the loss only requires distinct batch indices sharing appearance ID, not different trajectories or physics.",
        "companion": "paper/figures/split_assets/pusht_v0.png; provenance in split_assets/manifest.json",
        "limits": "Frame thumbnail represents a training example; context values and loss links are schematic, not measured activations or evidence of successful identification."
    }
    (OUTPUT / "method_training_assets.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"asset": record["asset"], "split": "train", "lossless": True}))


if __name__ == "__main__":
    main()
