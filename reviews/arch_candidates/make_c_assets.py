"""Extra real-data assets for candidate C: PCA->RGB views of the observed grids Z_{-2}, Z_{-1}, Z_0 and of the
forecasts, using exactly the projection of arch_assets.head_images (so head_obs.png / head_hat_k*.png match).

Episode droid-c70170b3ca00f756c4bb35c8 (held-out DROID), checkpoint results/v2s/droid/dinov2s/shiftwm/s0/best.pt, k=10.
Usage (repo root): PYTHONPATH=src .venv/bin/python reviews/arch_candidates/make_c_assets.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shiftwm.v2.models import V2WorldModel

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "assets"
EP, K = "droid-c70170b3ca00f756c4bb35c8", 10

root = ROOT / "data/v2/features/droid/dinov2s"
stats = json.loads((root / "stats.json").read_text())
with np.load(root / f"episodes/{EP}.npz") as z:
    f, a = z["features"].astype(np.float32), z["actions"]
fm, fs = np.array(stats["feature_mean"]), np.array(stats["feature_std"])
am, ast = np.array(stats["action_mean"]), np.array(stats["action_std"])
fz = (f.reshape(len(f), -1, f.shape[-1]) - fm) / fs
an = (a - am) / ast
st = torch.load(ROOT / "results/v2s/droid/dinov2s/shiftwm/s0/best.pt", map_location="cpu")
m = V2WorldModel(st["config"]).eval(); m.load_state_dict(st["model"])
Tt = lambda x: torch.tensor(x, dtype=torch.float32)
with torch.no_grad():
    pred, det = m(Tt(fz[None, :3]), Tt(an[None, :2]), Tt(an[None, 2:2 + K]), return_details=True)
p = pred[0, K - 1].numpy(); zt = fz[2 + K]
fit = fz[:3 + K].reshape(-1, fz.shape[-1]); mu = fit.mean(0)
_, _, Vt = np.linalg.svd(fit - mu, full_matrices=False); P = Vt[:3].T
proj = lambda v: (v - mu) @ P
lo, hi = np.percentile(np.concatenate([proj(fz[2]), proj(zt), proj(p)]), [1, 99], axis=0)


def save(v, name):
    im = np.clip((proj(v) - lo) / (hi - lo), 0, 1).reshape(16, 16, 3)
    fig = plt.figure(figsize=(2, 2), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(im, interpolation="bicubic"); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)


for j, nm in enumerate(("obs_m2", "obs_m1", "obs_0")):
    save(fz[j], nm)
gate = det["gate"][0, K - 1, :, 0].numpy()
np.save(OUT / "gate_k10.npy", gate)
print("saved; gate max patch", int(gate.argmax()), "err persistence", float(((fz[2] - zt) ** 2).mean()),
      "err forecast", float(((p - zt) ** 2).mean()))
