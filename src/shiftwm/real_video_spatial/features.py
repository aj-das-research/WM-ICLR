"""Train/validation-only 4x4 cache; DINO extraction reused from real_video.features."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from shiftwm.real_video.data import atomic_json, sha256, validate_manifest
from .data import training_statistics


def extract(data_root, output_root, encoder_root, batch_size=32, device="cuda", original_cache="data/features/droid_selected_v1"):
    import transformers
    from transformers import Dinov2Model
    data_root, output_root, encoder_root = map(Path, (data_root, output_root, encoder_root))
    source = json.loads((data_root / "manifest.json").read_text())
    validate_manifest(source)
    # Filter before opening any episode payload. Test metadata is never decoded.
    selected = [row for row in source["episodes"] if row["split"] in ("train", "val")]
    old_root=Path(original_cache)
    old_manifest=json.loads((old_root / "manifest.json").read_text())
    old_rows={row["episode_id"]:row for row in old_manifest["episodes"] if row["split"] in ("train","val")}
    audit = json.loads((data_root / "data_audit.json").read_text())
    if audit.get("status") != "passed" or audit.get("dataset_manifest_sha256") != sha256(data_root / "manifest.json"):
        raise ValueError("Missing or stale real-video data audit")
    provenance = json.loads((encoder_root / "provenance.json").read_text())
    for record in provenance["files"]:
        if sha256(encoder_root / record["file"]) != record["sha256"]:
            raise ValueError("Frozen pretrained encoder changed")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires an allocated GPU")
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {"dataset_manifest_sha256": sha256(data_root / "manifest.json"),
                "encoder": provenance, "extractor_sha256": sha256(__file__),
                "resize": [224,224], "interpolation": "bilinear_antialias",
                "pixel_normalization": "ImageNet mean/std", "pooling": "4x4 spatial mean of final patch tokens; channel-major",
                "feature_dim": 6144, "precision": "bfloat16 encoder; float32 pooling/storage" if device == "cuda" else "float32",
                "batch_size": batch_size,
                "data_audit_sha256": sha256(data_root / "data_audit.json"),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None,
                "camera_policy": "original train/val exterior1 only; test payloads forbidden",
                "original_2x2_cache_manifest_sha256": sha256(old_root / "manifest.json")}
    identity_path = output_root / "identity.json"
    if not identity_path.exists() and any(output_root.iterdir()):
        raise ValueError("Nonempty feature cache has no extraction identity")
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Cannot mix feature extraction identities")
    atomic_json(identity, identity_path)
    identity_hash = sha256(identity_path)
    encoder = Dinov2Model.from_pretrained(encoder_root, local_files_only=True).to(device).eval().requires_grad_(False)
    mean = torch.tensor([.485,.456,.406], device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
    episodes = []
    for episode in selected:
        row = {key: episode[key] for key in ("episode_id", "session_id", "split", "steps")}
        row["cameras"] = {}
        cameras = ["exterior_image_1_left"]
        for camera in cameras:
            record = episode["cameras"][camera]
            path = data_root / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Real video payload changed: {path}")
            destination = output_root / "episodes" / camera / (row["episode_id"] + ".npz")
            receipt_path = destination.with_suffix(".json")
            receipt = None
            if receipt_path.exists() or destination.exists():
                if not receipt_path.exists() or not destination.exists():
                    raise ValueError("Incomplete immutable feature payload/receipt pair")
                old = json.loads(receipt_path.read_text())
                if old.get("source_sha256") != record["sha256"] or old.get("identity_sha256") != identity_hash:
                    raise ValueError("Feature receipt belongs to a different extraction")
                if sha256(destination) != old.get("sha256"):
                    raise ValueError("Immutable feature payload is corrupt")
                receipt = old
            if receipt is None:
                with np.load(path, allow_pickle=False) as arrays:
                    images = arrays["images"]
                    actions = arrays["actions"].copy()
                    frame_indices = arrays["frame_indices"].copy()
                if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3 or len(images) < 1:
                    raise ValueError("Expected genuine uint8 RGB frames")
                if actions.shape != (len(images)-1,35) or not np.isfinite(actions).all():
                    raise ValueError("Invalid recorded action blocks")
                if (frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer)
                        or len(frame_indices) != len(images) or frame_indices[0] != 0
                        or not np.all(np.diff(frame_indices) == 5)):
                    raise ValueError("Invalid recorded frame sequence")
                pieces = []
                with torch.inference_mode():
                    for start in range(0, len(images), batch_size):
                        pixels = torch.from_numpy(images[start:start+batch_size]).to(device).permute(0,3,1,2).float()/255
                        pixels = F.interpolate(pixels, size=(224,224), mode="bilinear", align_corners=False, antialias=True)
                        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
                        with amp:
                            encoded = encoder(pixel_values=(pixels-mean)/std).last_hidden_state[:,1:]
                        if encoded.shape[1:] != (256,384):
                            raise ValueError("DINO patch layout changed")
                        grid = encoded.float().transpose(1,2).reshape(-1,384,16,16)
                        pieces.append(F.adaptive_avg_pool2d(grid,(4,4)).flatten(1).cpu().numpy())
                features = np.concatenate(pieces)
                if not np.isfinite(features).all():
                    raise ValueError("Nonfinite features")
                old_record=old_rows[row["episode_id"]]["cameras"][camera]
                old_path=old_root/old_record["file"]
                if sha256(old_path)!=old_record["sha256"]: raise ValueError("Original feature payload changed")
                with np.load(old_path,allow_pickle=False) as old:
                    pooled=F.avg_pool2d(torch.from_numpy(features).reshape(-1,384,4,4),2).flatten(1).numpy()
                    error=float(np.max(np.abs(pooled-old["features"])))
                    if not np.allclose(pooled,old["features"],atol=2e-5,rtol=1e-5):
                        raise ValueError(f"Original 2x2 coordinate parity failed: {error}")
                    if not np.array_equal(actions,old["actions"]) or not np.array_equal(frame_indices,old["frame_indices"]):
                        raise ValueError("Original action/frame coordinates differ")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, features=features, actions=actions, frame_indices=frame_indices)
                temporary.replace(destination)
                receipt = {"file": str(destination.relative_to(output_root)), "sha256": sha256(destination),
                           "source_sha256": record["sha256"], "identity_sha256": identity_hash,
                           "frames": len(features), "original_2x2_parity_max_abs": error}
                atomic_json(receipt, receipt_path)
            row["cameras"][camera] = receipt
        episodes.append(row)
        print(json.dumps({"event":"encoded", "episode":row["episode_id"], "count":len(episodes), "total":len(selected)}), flush=True)
    result = {"status":"complete", "dataset":"DROID selected real recordings", "identity": identity,
              "feature_dim":6144, "action_dim":35, "history_length":3,
              "episodes":episodes, "action_block":5}
    atomic_json(result, output_root / "manifest.json")
    statistics = training_statistics(output_root)
    print(json.dumps({"event":"complete", "episodes":len(episodes), "counts":statistics["counts"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="data/pretrained/dinov2-small")
    parser.add_argument("--original-cache", default="data/features/droid_selected_v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    args = parser.parse_args()
    extract(args.data,args.output,args.encoder,args.batch_size,args.device,args.original_cache)
