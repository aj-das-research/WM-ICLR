"""DINOv2-S extraction with the reviewed DROID spatial numeric recipe, copied explicitly.

The original extractor is immutable. Its source hash is bound by IWS registration.
IWS keeps native RGB rows, and does not reuse DROID sampling or action grouping.
"""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_json, require, sha

ENCODER_REPO = "facebook/dinov2-small"
ENCODER_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
PREPROCESSING = {
    "input": "unaltered sequential native RGB uint8 rows, converted from decoder BGR",
    "resize": [224, 224], "interpolation": "torch bilinear align_corners=False antialias=True",
    "scale": "float32 / 255", "mean": [.485, .456, .406], "std": [.229, .224, .225],
    "tokens": "remove CLS; 256 spatial tokens x384 channels",
    "pooling": "float32 adaptive_avg_pool2d from16x16 to4x4; channel-major flatten",
    "feature_dim": 6144, "grid": [4, 4], "channels": 384,
}


def verify_encoder(root, encoder_root):
    root, encoder_root = Path(root), Path(encoder_root)
    expected = read_json(root / "references/real_dinov2_sources.json")
    actual = read_json(encoder_root / "provenance.json")
    require(actual == expected and actual["repo"] == ENCODER_REPO and actual["revision"] == ENCODER_REVISION, "Unexpected encoder provenance/revision")
    require({r["file"] for r in actual["files"]} == {"config.json", "preprocessor_config.json", "model.safetensors", "README.md"}, "Unexpected encoder file inventory")
    for row in actual["files"]:
        path = encoder_root / row["file"]
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Frozen encoder file changed: " + row["file"])
    return actual


def preprocess(images, device):
    require(images.dtype == np.uint8 and images.ndim == 4 and images.shape[-1] == 3 and len(images) > 0, "Expected actual uint8 RGB frames")
    pixels = torch.from_numpy(np.ascontiguousarray(images)).to(device).permute(0, 3, 1, 2).float() / 255
    pixels = F.interpolate(pixels, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(PREPROCESSING["mean"], device=device)[None, :, None, None]
    std = torch.tensor(PREPROCESSING["std"], device=device)[None, :, None, None]
    return (pixels - mean) / std


def pool_tokens(encoded):
    require(tuple(encoded.shape[1:]) == (256, 384), "DINO patch layout changed")
    grid = encoded.float().transpose(1, 2).reshape(-1, 384, 16, 16)
    return F.adaptive_avg_pool2d(grid, (4, 4)).flatten(1)


class DinoSpatialEncoder:
    def __init__(self, root, encoder_root, device):
        from transformers import Dinov2Model
        verify_encoder(root, encoder_root)
        require(device in ("cpu", "cuda"), "Unsupported encoder device")
        if device == "cuda":
            require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Allocated BF16-capable GPU required")
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = Dinov2Model.from_pretrained(encoder_root, local_files_only=True, use_safetensors=True).to(device).eval().requires_grad_(False)
        require(self.model.config.hidden_size == 384 and self.model.config.patch_size == 14, "Wrong DINO-small architecture")

    def __call__(self, images, batch_size):
        require(isinstance(batch_size, int) and batch_size >= 1, "Invalid encoder batch size")
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                pixels = preprocess(images[start:start + batch_size], self.device)
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
                with amp:
                    encoded = self.model(pixel_values=pixels).last_hidden_state[:, 1:]
                pieces.append(pool_tokens(encoded).cpu().numpy())
        result = np.concatenate(pieces)
        require(result.dtype == np.float32 and result.shape == (len(images), 6144) and np.isfinite(result).all(), "Invalid extracted features")
        return result
