"""Native-resolution frozen-encoder feature extraction for ShiftWM-v2.

Reads either the DROID processed manifest (scripts/real_video/prepare_droid.py) or the
generic Stage-1 frame format (docs/v2_data_format.md) and writes Stage-2 caches:
features float16 [T, G, G, C] (all patch tokens, no pooling), actions, optional proprio.
Normalization statistics are computed on the train split only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ENCODERS = {
    # tag: (local dir, patch size, image size)
    "dinov2s": ("data/pretrained/dinov2-small", 14, 224),
    "dinov2b": ("data/pretrained/dinov2-base", 14, 224),
    "dinov3s": ("data/pretrained/dinov3-vits16", 16, 256),
    "dinov3b": ("data/pretrained/dinov3-vitb16", 16, 256),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(tag, device):
    from transformers import AutoModel
    root, patch, size = ENCODERS[tag]
    model = AutoModel.from_pretrained(root, local_files_only=True).to(device).eval().requires_grad_(False)
    return model, patch, size


def encode(model, images, patch, size, device, batch_size=64):
    """uint8 [T,H,W,3] -> float16 [T, size/patch, size/patch, C] patch tokens (registers/CLS dropped)."""
    grid = size // patch
    mean = torch.tensor(IMAGENET_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, device=device)[None, :, None, None]
    out = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start:start + batch_size]).to(device).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                tokens = model(pixel_values=(x - mean) / std).last_hidden_state
            tokens = tokens[:, -grid * grid:]  # patch tokens are last for DINOv2 (CLS) and DINOv3 (CLS+regs)
            out.append(tokens.float().reshape(len(x), grid, grid, -1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def droid_episodes(root, camera):
    manifest = json.loads((root / "manifest.json").read_text())
    for row in manifest["episodes"]:
        record = row["cameras"][camera]
        yield {"id": row["episode_id"], "split": row["split"], "session": row["session_id"],
               "task": "droid", "path": root / record["file"]}


def stage1_episodes(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for row in manifest["episodes"]:
        yield {"id": row["id"], "split": row["split"], "task": row.get("task", ""),
               "session": row.get("session", row["id"]), "path": root / row["file"]}


def extract(source, kind, output, tag, camera="exterior_image_1_left", device="cuda", limit=None):
    source, output = Path(source), Path(output)
    rows = list(droid_episodes(source, camera) if kind == "droid" else stage1_episodes(source))
    if limit:
        rows = rows[:limit]
    model, patch, size = load_encoder(tag, device)
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        dest = output / "episodes" / f"{row['id']}.npz"
        with np.load(row["path"], allow_pickle=False) as z:
            actions = z["actions"].astype(np.float32)
            proprio = z["proprio"].astype(np.float32) if "proprio" in z.files else None
            if not dest.exists():
                feats = encode(model, z["images"], patch, size, device)
                if not np.isfinite(feats.astype(np.float32)).all():
                    raise ValueError(f"nonfinite features {row['id']}")
                payload = {"features": feats, "actions": actions}
                if proprio is not None:
                    payload["proprio"] = proprio
                tmp = dest.with_suffix(".tmp.npz")
                np.savez(tmp, **payload)
                tmp.replace(dest)
            T = len(z["images"])
        if actions.shape[0] != T - 1:
            raise ValueError(f"action/frame mismatch {row['id']}")
        records.append({"id": row["id"], "split": row["split"], "task": row["task"], "session": row["session"],
                        "file": f"episodes/{row['id']}.npz", "T": int(T)})
        if i % 50 == 0:
            print(json.dumps({"event": "encoded", "n": i + 1, "total": len(rows)}), flush=True)
    grid = size // patch
    with np.load(output / records[0]["file"]) as z:
        channels, action_dim = z["features"].shape[-1], z["actions"].shape[-1]
        proprio_dim = z["proprio"].shape[-1] if "proprio" in z.files else 0
    stats = train_statistics(output, records)
    manifest = {"source": str(source), "kind": kind, "camera": camera if kind == "droid" else None,
                "encoder": tag, "encoder_dir": ENCODERS[tag][0],
                "encoder_weights_sha256": sha256(Path(ENCODERS[tag][0]) / "model.safetensors"),
                "image_size": size, "patch": patch, "grid": grid, "channels": int(channels),
                "action_dim": int(action_dim), "proprio_dim": int(proprio_dim), "episodes": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (output / "stats.json").write_text(json.dumps(stats))
    counts = {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}
    print(json.dumps({"event": "complete", "counts": counts, "grid": grid, "channels": int(channels)}), flush=True)


def train_statistics(output, records):
    """Per-channel feature mean/std shared across positions; per-dim action/proprio stats. Train only."""
    s1 = s2 = None; n = 0; acts = []; props = []
    for r in records:
        if r["split"] != "train":
            continue
        with np.load(output / r["file"]) as z:
            f = z["features"].astype(np.float64).reshape(-1, z["features"].shape[-1])
            s1 = f.sum(0) if s1 is None else s1 + f.sum(0)
            s2 = (f ** 2).sum(0) if s2 is None else s2 + (f ** 2).sum(0)
            n += len(f)
            acts.append(z["actions"])
            if "proprio" in z.files:
                props.append(z["proprio"])
    mean = s1 / n; std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    a = np.concatenate(acts)
    stats = {"feature_mean": mean.tolist(), "feature_std": std.tolist(),
             "action_mean": a.mean(0).tolist(), "action_std": np.maximum(a.std(0), 1e-6).tolist()}
    if props:
        p = np.concatenate(props)
        stats.update(proprio_mean=p.mean(0).tolist(), proprio_std=np.maximum(p.std(0), 1e-6).tolist())
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--kind", choices=("droid", "stage1"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", choices=tuple(ENCODERS), default="dinov2s")
    parser.add_argument("--camera", default="exterior_image_1_left")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    a = parser.parse_args()
    extract(a.source, a.kind, a.output, a.encoder, a.camera, a.device, a.limit)
