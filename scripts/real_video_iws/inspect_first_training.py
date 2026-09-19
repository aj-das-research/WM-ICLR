#!/usr/bin/env python3
"""Read-only compatibility probe on the first INTERNAL training recording only."""
from pathlib import Path
import hashlib
import json
import sys
import datetime
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws.data import InternalInventory, decode_native_rgb, load_commands, sha
from shiftwm.real_video_iws.features import DinoSpatialEncoder


def main():
    torch.set_num_threads(2)
    inventory = InternalInventory(ROOT)
    eid = inventory.selected("internal_train")[0]
    row, split, metadata, video = inventory.paths(eid, "internal_train")
    frames = decode_native_rgb(inventory, eid, sha(video), "internal_train")
    commands = load_commands(inventory, eid, "internal_train")
    destination = ROOT / "artifacts/development/iws_pusht_cache_preflight"
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[0]).save(destination / f"{eid}_frame0.png")
    indices = [0, len(frames) - 1]
    encoder = DinoSpatialEncoder(ROOT, ROOT / "data/pretrained/dinov2-small", "cpu")
    features = encoder(frames[indices], 32)
    # Independently spell out the immutable spatial extractor's CPU numerical recipe.
    pixels = torch.from_numpy(frames[indices]).permute(0, 3, 1, 2).float() / 255
    pixels = F.interpolate(pixels, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    with torch.inference_mode():
        tokens = encoder.model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:]
        reference = F.adaptive_avg_pool2d(tokens.float().transpose(1, 2).reshape(-1, 384, 16, 16), (4, 4)).flatten(1).numpy()
    if not np.array_equal(features, reference):
        raise ValueError("Copied spatial extraction recipe differs on actual recorded pixels")
    report = {"status": "actual_internal_training_decode_and_cpu_encoder_compatibility_passed", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "episode_id": eid, "split": split, "upstream_split": "train", "selection": "first sorted internal_train ID; fixed before viewing",
              "native_decoded_frames": len(frames), "audited_metadata_rows": row["shapes"]["target_qpos"][0], "command_shape": list(commands.shape),
              "decoded_shape": list(frames.shape), "frame_timebase": "native recorded row indices; no physical seconds inferred",
              "metadata_sha256": sha(metadata), "video_sha256": sha(video), "split_sha256": inventory.split_sha256,
              "command_values_sha256": hashlib.sha256(commands.tobytes()).hexdigest(),
              "rgb_frame0_pixels_sha256": hashlib.sha256(frames[0].tobytes()).hexdigest(),
              "preview_path": str((destination / f"{eid}_frame0.png").relative_to(ROOT)), "preview_png_sha256": sha(destination / f"{eid}_frame0.png"),
              "encoded_native_indices": indices, "feature_shape": list(features.shape), "spatial_recipe_max_abs_error": float(np.max(np.abs(features - reference))),
              "encoder_provenance_sha256": sha(ROOT / "data/pretrained/dinov2-small/provenance.json"),
              "source_sha256": {p: sha(ROOT / p) for p in ["src/shiftwm/real_video_iws/data.py", "src/shiftwm/real_video_iws/features.py", "src/shiftwm/real_video_spatial/features.py", "scripts/real_video_iws/inspect_first_training.py"]},
              "official_validation_payloads_opened": 0, "gpu_jobs_submitted": 0, "predictor_training_or_evaluation": False,
              "full_feature_cache_completed": False,
              "prior_exposure": "Earlier upstream-train000010 preview belongs to internal_development; it was not moved or used for this internal-training probe.",
              "visual_review": "Pending inspection of actual preview pixels; no image-content conclusion inferred from filenames."}
    path = ROOT / "reports/evidence/iws_pusht_cache_actual_preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "episode_id", "native_decoded_frames", "command_shape", "spatial_recipe_max_abs_error"]}))


if __name__ == "__main__":
    main()
