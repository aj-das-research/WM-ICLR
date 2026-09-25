"""Forecast one test window of a real episode with a trained ShiftWM checkpoint.

Expects a feature directory with ``stats.json`` (train mean/std of features and actions),
``manifest.json`` (episodes with ``split`` and ``file``) and per-episode ``.npz`` files holding
``features [T, G, G, C]`` (or ``[T, N, C]``) and ``actions [T-1, A]``.

    python examples/forecast_checkpoint.py --checkpoint best.pt --features features/droid/dinov2s
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from shiftwm import ShiftWM, forecast, skill, transport_field


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/v2s/droid/dinov2s/shiftwm/s0/best.pt")
    ap.add_argument("--features", default="data/v2/features/droid/dinov2s")
    ap.add_argument("--episode", type=int, default=0, help="index among test episodes")
    ap.add_argument("--start", type=int, default=0, help="first frame of the window")
    args = ap.parse_args()

    model = ShiftWM.from_checkpoint(args.checkpoint)
    c = model.config
    print(f"loaded {c.arm}: grid {c.grid}x{c.grid}, C={c.channels}, H={c.history}, K={c.horizon}, "
          f"{model.num_params() / 1e6:.2f}M params")

    root = Path(args.features)
    stats = json.loads((root / "stats.json").read_text())
    rows = [r for r in json.loads((root / "manifest.json").read_text())["episodes"] if r["split"] == "test"]
    with np.load(root / rows[args.episode]["file"]) as z:
        f = torch.from_numpy(z["features"].astype(np.float32))
        a = torch.from_numpy(z["actions"].astype(np.float32))
    f = (f.reshape(f.shape[0], -1, f.shape[-1]) - torch.tensor(stats["feature_mean"])) / torch.tensor(stats["feature_std"])
    a = (a - torch.tensor(stats["action_mean"])) / torch.tensor(stats["action_std"])
    s, H, K = args.start, c.history, c.horizon
    hist, target = f[s:s + H], f[s + H:s + H + K]
    past, future = a[s:s + H - 1], a[s + H - 1:s + H - 1 + K]
    print(f"test episode {args.episode} ({f.shape[0]} frames), window {s}..{s + H + K - 1}")

    pred, det = forecast(model, hist, past, future, return_details=True)
    print("pred", tuple(pred.shape), "target", tuple(target.shape))
    g = det["gate"]
    print(f"gate: mean {g.mean():.3f}  (per step: {' '.join(f'{v:.2f}' for v in g.mean((1, 2)).tolist())})")
    flow = transport_field(det)
    print(f"transport: mean |offset| {flow.norm(dim=-1).mean():.2f} patches, "
          f"max {flow.norm(dim=-1).max():.2f}")
    per_k = skill(pred, target, hist, per_horizon=True)
    print(f"skill vs persistence: {skill(pred, target, hist):+.3f}  "
          f"(k=1 {per_k[0]:+.3f}, k={K} {per_k[-1]:+.3f})")


if __name__ == "__main__":
    main()
