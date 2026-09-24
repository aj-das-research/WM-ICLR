"""How close to a copy is ShiftWM's transport on static vs moving patches (held-out DROID, final seed-0 checkpoint)?

Per horizon k and region (moving = top 25% true change at k, static = rest; as in anatomy.py / region_eval.py):
  gate          : mean gate g
  id_weight     : mean transport weight on the identity source (same patch, last observed frame)
  moved_change  : mean ||g (T - Z0)||^2 / C, how far the gated transport moves the forecast from the observed grid
  true_change   : mean ||Z_k - Z0||^2 / C, the true change (reference scale)
Writes results/v2/analysis/anatomy/identity.json. Runs on CPU (--device cpu) or GPU.
"""
import argparse
import json
from pathlib import Path

import torch

from shiftwm.v2.models import V2WorldModel
from shiftwm.v2.train import FeatureSplit

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
p.add_argument("--ckpt", default="results/v2s/droid/dinov2s/shiftwm/s0/best.pt")
a = p.parse_args()
dev = a.device
root = ROOT / "data/v2/features/droid/dinov2s"
stats = json.loads((root / "stats.json").read_text())
data = FeatureSplit(root, "test", 3, 10, dev, stats, stride=2)
st = torch.load(ROOT / a.ckpt, map_location=dev)
model = V2WorldModel(st["config"]).to(dev).eval(); model.load_state_dict(st["model"])
c = model.config
s, w = min(c.sources, 3), c.window
ID = (s - 1) * w * w + (w * w) // 2
K = 10
acc = {r: {m: torch.zeros(K, dtype=torch.float64) for m in ("gate", "id_weight", "moved_change", "true_change")}
       for r in ("moving", "static")}
cnt = {r: torch.zeros(K, dtype=torch.float64) for r in ("moving", "static")}
with torch.no_grad():
    for i in range(0, len(data), 64):
        idx = torch.arange(i, min(i + 64, len(data)), device=dev)
        hist, past, fut, tgt = data.batch(idx)
        pred, det = model(hist, past, fut, return_details=True)
        z0 = hist[:, -1]
        g = det["gate"].float()[..., 0]                                    # [B,K,N]
        wid = det["weights"].float()[..., ID]                               # [B,K,N]
        mch = ((pred.float() - z0[:, None] - det["correction"].float()) ** 2).mean(-1)   # ||g (T - Z0)||^2 / C
        tch = ((tgt - z0[:, None]) ** 2).mean(-1)
        mv = tch >= tch.quantile(0.75, dim=-1, keepdim=True)
        for r, m in (("moving", mv), ("static", ~mv)):
            m = m.double()
            for name, v in (("gate", g), ("id_weight", wid), ("moved_change", mch), ("true_change", tch)):
                acc[r][name] += (v.double() * m).sum((0, 2)).cpu()
            cnt[r] += m.sum((0, 2)).cpu()
out = {"checkpoint": a.ckpt, "windows": len(data), "identity_index": ID,
       "regions": {r: {m: (v / cnt[r]).tolist() for m, v in d.items()} for r, d in acc.items()}}
f = ROOT / "results/v2/analysis/anatomy/identity.json"
f.write_text(json.dumps(out, indent=1))
for r, d in out["regions"].items():
    print(r, {m: [round(x, 3) for x in v[::3]] for m, v in d.items()})
