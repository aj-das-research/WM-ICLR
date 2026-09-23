"""Forecast sharpness: spatial variance of predicted feature grids relative to the true grid, per horizon.

Regression-style predictors regress toward the conditional mean and lose spatial contrast; transport copies
measured features. For each test window we compute var_patches(pred_k)/var_patches(true_k) (mean over channels)
and report per-arm, per-horizon means with episode-bootstrap 95% CIs.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from shiftwm.v2.models import V2WorldModel
from shiftwm.v2.train import FeatureSplit

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="droid"); p.add_argument("--encoder", default="dinov2s")
p.add_argument("--arms", nargs="*", default=["persistence", "ar_tf", "ar", "direct", "shiftwm"])
a = p.parse_args()
root = Path(f"data/v2/features/{a.dataset}/{a.encoder}"); stats = json.loads((root / "stats.json").read_text())
data = FeatureSplit(root, "test", 3, 10, "cuda", stats, stride=2)
res = {}
for arm in a.arms:
    runs = sorted(Path(f"results/v2/{a.dataset}/{a.encoder}/{arm}").glob("s*/best.pt"))
    if arm in ("persistence",):
        model = V2WorldModel({"arm": arm, "grid": 16, "channels": data.features.shape[-1], "horizon": 10}).cuda()
    elif not runs:
        continue
    else:
        st = torch.load(runs[0], map_location="cuda"); model = V2WorldModel(st["config"]).cuda().eval()
        model.load_state_dict(st["model"])
    ratios = torch.zeros(len(data.episodes), 10, device="cuda", dtype=torch.float64); cnt = torch.zeros(len(data.episodes), device="cuda", dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(data), 128):
            idx = torch.arange(i, min(i + 128, len(data)), device="cuda")
            h, pa, f, t = data.batch(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(h, pa, f).float()
            r = y.var(2).mean(-1) / t.var(2).mean(-1)                  # [B,K]
            ep = data.episode_of[idx]
            ratios.index_add_(0, ep, r.double()); cnt.index_add_(0, ep, torch.ones_like(ep, dtype=torch.float64))
    per_ep = (ratios[cnt > 0] / cnt[cnt > 0, None]).cpu().numpy()
    rng = np.random.default_rng(0)
    boots = np.stack([per_ep[rng.integers(0, len(per_ep), len(per_ep))].mean(0) for _ in range(2000)])
    res[arm] = {"mean": per_ep.mean(0).tolist(), "lo": np.percentile(boots, 2.5, 0).tolist(),
                "hi": np.percentile(boots, 97.5, 0).tolist(), "checkpoint": str(runs[0]) if runs else None}
    print(arm, [round(v, 3) for v in res[arm]["mean"]], flush=True)
out = Path(f"results/v2/analysis/sharpness/{a.dataset}_{a.encoder}.json"); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1))
