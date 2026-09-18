"""Extract training examples and exact RGB variants; no new simulator rollout."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.data import APPEARANCES, appearance_transform

OUT = Path(__file__).resolve().parent
SOURCES = {
    "pusht": ROOT / "data/world/pusht_relative/episodes/train-s31001-d0.npz",
    "reacher": ROOT / "data/world/reacher/episodes/train-s31000-d0.npz",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = {"scope": "Observed canonical training frames; exact paired RGB transforms",
            "frame_index": 0, "assets": []}
for environment, source in SOURCES.items():
    with np.load(source, allow_pickle=False) as episode:
        pixels = episode["images"][0].copy()
    target = OUT / f"{environment}.png"
    Image.fromarray(pixels).save(target)
    manifest[environment] = {
        "source": str(source.relative_to(ROOT)), "source_sha256": sha(source),
        "selection": ("First lexicographic dynamics-0 training episode with the entire block "
                      "at least 8 pixels inside the frame; no outcome selection"
                      if environment == "pusht" else
                      "First lexicographic dynamics-0 training episode; no outcome selection"),
        "native_size": list(pixels.shape[:2]), "crop": "none", "frame_index": 0,
    }
    manifest["assets"].append({"file": target.name, "sha256": sha(target)})
    if environment == "pusht":
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        for appearance_id in range(4):
            transformed = appearance_transform(tensor, appearance_id)
            rgb = (transformed.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            target = OUT / f"pusht_v{appearance_id}.png"
            Image.fromarray(rgb).save(target)
            manifest["assets"].append({"file": target.name, "sha256": sha(target),
                                       "appearance": APPEARANCES[appearance_id],
                                       "quantization": "nearest uint8 for display"})
manifest["generator_sha256"] = sha(Path(__file__))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
