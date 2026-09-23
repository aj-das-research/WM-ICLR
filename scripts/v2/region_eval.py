"""Error on moving vs static regions + skill scores, for every finished checkpoint of a dataset.

Moving patches: per window and horizon, the patches whose true feature change from the last observed grid is
in the top 25% (a property of the data, identical for all methods). Writes per-episode arrays for bootstrap.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from shiftwm.v2.models import V2WorldModel
from shiftwm.v2.train import FeatureSplit

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="droid"); p.add_argument("--encoder", default="dinov2s")
p.add_argument("--horizon", type=int, default=10); p.add_argument("--single-image", action="store_true")
a = p.parse_args()
root = Path(f"data/v2/features/{a.dataset}/{a.encoder}"); stats = json.loads((root / "stats.json").read_text())
data = FeatureSplit(root, "test", 3, a.horizon, "cuda", stats, stride=2, single_image=a.single_image)
res = {}
for armdir in sorted(Path(f"results/v2/{a.dataset}/{a.encoder}").iterdir()):
    arm = armdir.name
    if arm in ("ablations",):
        continue
    for run in sorted(armdir.glob("s*")):
        if arm in ("persistence", "linear"):
            model = V2WorldModel({"arm": arm, "grid": 16, "channels": data.features.shape[-1], "horizon": a.horizon}).cuda()
        elif (run / "best.pt").exists() and (run / "summary.json").exists():
            st = torch.load(run / "best.pt", map_location="cuda"); model = V2WorldModel(st["config"]).cuda().eval()
            model.load_state_dict(st["model"])
        else:
            continue
        E = len(data.episodes); K = a.horizon
        acc = {m: torch.zeros(E, K, device="cuda", dtype=torch.float64) for m in ("all", "moving", "static")}
        cnt = torch.zeros(E, device="cuda", dtype=torch.float64)
        with torch.no_grad():
            for i in range(0, len(data), 128):
                idx = torch.arange(i, min(i + 128, len(data)), device="cuda")
                h, pa, f, t = data.batch(idx)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = model(h, pa, f).float()
                err = ((y - t) ** 2).mean(-1)                                   # [B,K,N]
                change = ((t - h[:, -1:]) ** 2).mean(-1)                         # true change per patch
                thr = change.quantile(0.75, dim=-1, keepdim=True)
                mv = (change >= thr).double()
                ep = data.episode_of[idx]
                acc["all"].index_add_(0, ep, err.mean(-1).double())
                acc["moving"].index_add_(0, ep, ((err * mv).sum(-1) / mv.sum(-1)).double())
                acc["static"].index_add_(0, ep, ((err * (1 - mv)).sum(-1) / (1 - mv).sum(-1)).double())
                cnt.index_add_(0, ep, torch.ones_like(ep, dtype=torch.float64))
        keep = cnt > 0
        res[f"{arm}/{run.name}"] = {m: (v[keep] / cnt[keep, None]).cpu().numpy().tolist() for m, v in acc.items()}
        print(arm, run.name, {m: round(float(np.mean(v)), 4) for m, v in res[f"{arm}/{run.name}"].items()}, flush=True)
        if arm in ("persistence", "linear"):
            break
out = Path(f"results/v2/analysis/regions/{a.dataset}_{a.encoder}_K{a.horizon}.json"); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res))
