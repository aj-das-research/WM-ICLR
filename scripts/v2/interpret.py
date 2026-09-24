"""Mechanistic analysis of ShiftWM on held-out DROID windows (causal interventions on the trained model).

  knockout : set the gate to 0 at inference (no transport; forecast = Z0 + r) and measure the error change per
             region -- moving patches (top 25% true change), static patches, and patches with a high learned gate.
  motion   : error of ShiftWM and Direct binned by deciles of true per-patch feature change.
  steering : replace the future actions with another window's actions and measure how much the transport field
             (expected source offset) changes on moving vs. static patches.
  examples : per-patch error maps of ShiftWM and Direct (k=10) for 4 test windows chosen by a fixed rule
             (largest, 75th-percentile, median and 25th-percentile true motion), with frames for plotting.
Writes results/v2/analysis/interpret/droid.npz and summary.json. Run on a GPU.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from shiftwm.v2.models import V2WorldModel
from shiftwm.v2.train import FeatureSplit

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / os.environ.get("SHIFTWM_RUNS", "results/v2s")
OUT = ROOT / "results/v2/analysis/interpret"
K = 10


def load(arm, dev):
    ck = sorted((RUNS / "droid/dinov2s" / arm).glob("s*/best.pt"))[0]
    st = torch.load(ck, map_location=dev)
    m = V2WorldModel(st["config"]).to(dev).eval(); m.load_state_dict(st["model"])
    return m, str(ck)


def offsets(model, weights):
    c = model.config; r = c.window // 2
    oy, ox = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
    oy = oy.flatten().repeat(c.sources).to(weights); ox = ox.flatten().repeat(c.sources).to(weights)
    return torch.stack((weights @ ox, weights @ oy), -1)                      # [B,K,N,2] expected source offset


@torch.no_grad()
def main():
    dev = "cuda"
    root = ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    data = FeatureSplit(root, "test", 3, K, dev, stats, stride=2)
    sw, ck_sw = load("shiftwm", dev); di, ck_di = load("direct", dev)
    acc = {k: [] for k in ("ko_moving", "ko_static", "ko_highgate", "base_moving", "base_static", "base_highgate",
                           "steer_moving", "steer_static")}
    bins = np.zeros((2, 10)); counts = np.zeros(10); change_all = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(0, len(data), 96):
        idx = torch.arange(i, min(i + 96, len(data)), device=dev)
        h, pa, f, t = data.batch(idx)
        p_sw, det = sw(h, pa, f, return_details=True)
        p_di = di(h, pa, f)
        # knockout: gate = 0  ->  Z0 + r
        ko = h[:, -1:].float() + det["correction"]
        e = lambda p: ((p[:, K - 1] - t[:, K - 1]) ** 2).mean(-1)            # [B,N] per-patch error at k=10
        e_sw, e_ko, e_di = e(p_sw), e(ko), e(p_di)
        chg = ((t[:, K - 1] - h[:, -1]) ** 2).mean(-1)
        mv = chg >= chg.quantile(0.75, dim=-1, keepdim=True)
        hg = det["gate"][:, K - 1, :, 0] >= 0.5
        for name, mask in (("moving", mv), ("static", ~mv), ("highgate", hg)):
            acc["ko_" + name].append(((e_ko * mask).sum() / mask.sum().clamp(min=1)).item())
            acc["base_" + name].append(((e_sw * mask).sum() / mask.sum().clamp(min=1)).item())
        # motion deciles (global thresholds computed afterwards from stored values)
        change_all.append(torch.stack((chg.flatten(), e_sw.flatten(), e_di.flatten()), 1).cpu())
        # steering: other window's actions
        perm = torch.randperm(len(idx), generator=gen).to(dev)
        perm = torch.where(perm == torch.arange(len(idx), device=dev), (perm + 1) % len(idx), perm)
        _, det2 = sw(h, pa, f[perm], return_details=True)
        d = (offsets(sw, det["weights"][:, K - 1]) - offsets(sw, det2["weights"][:, K - 1])).norm(dim=-1)  # [B,N]
        acc["steer_moving"].append(((d * mv).sum() / mv.sum()).item()); acc["steer_static"].append(((d * ~mv).sum() / (~mv).sum()).item())
    ca = torch.cat(change_all).numpy()
    qs = np.quantile(ca[:, 0], np.linspace(0, 1, 11))
    which = np.clip(np.searchsorted(qs, ca[:, 0], side="right") - 1, 0, 9)
    for b in range(10):
        sel = which == b; bins[0, b] = ca[sel, 1].mean(); bins[1, b] = ca[sel, 2].mean(); counts[b] = sel.sum()
    summ = {k: float(np.mean(v)) for k, v in acc.items()}
    summ.update(knockout_increase_moving=100 * (summ["ko_moving"] / summ["base_moving"] - 1),
                knockout_increase_static=100 * (summ["ko_static"] / summ["base_static"] - 1),
                knockout_increase_highgate=100 * (summ["ko_highgate"] / summ["base_highgate"] - 1),
                steer_ratio_moving_over_static=summ["steer_moving"] / summ["steer_static"],
                gain_vs_direct_by_decile=(100 * (1 - bins[0] / bins[1])).tolist(),
                checkpoints={"shiftwm": ck_sw, "direct": ck_di}, windows=len(data))
    # examples for plotting (rule stated in the caption): among windows in the top half of mean true change, the ones
    # with the largest relative k=10 advantage of ShiftWM over Direct, 1 - err_S / err_D, one per episode
    mean_chg, adv = [], []
    for i in range(0, len(data), 256):
        idx = torch.arange(i, min(i + 256, len(data)), device=dev)
        h, pa, f, t = data.batch(idx)
        mean_chg.append(((t[:, K - 1] - h[:, -1]) ** 2).mean((-1, -2)).cpu())
        e_s = ((sw(h, pa, f)[:, K - 1] - t[:, K - 1]) ** 2).mean((-1, -2))
        e_d = ((di(h, pa, f)[:, K - 1] - t[:, K - 1]) ** 2).mean((-1, -2))
        adv.append((1 - e_s / e_d).cpu())
    mean_chg = torch.cat(mean_chg).numpy(); adv = torch.cat(adv).numpy()
    ok = mean_chg >= np.median(mean_chg)
    ex_idx, seen = [], set()
    for j in np.argsort(-np.where(ok, adv, -np.inf)):
        e = int(data.episode_of[int(j)])
        if e not in seen:
            ex_idx.append(int(j)); seen.add(e)
        if len(ex_idx) == 4:
            break
    summ["example_rule"] = "top-half motion windows with the largest k=10 advantage over Direct, one per episode"
    summ["example_advantage"] = [float(adv[j]) for j in ex_idx]
    idx = torch.tensor(ex_idx, device=dev); h, pa, f, t = data.batch(idx)
    p_sw, det = sw(h, pa, f, return_details=True); p_di = di(h, pa, f)
    ex = {"err_sw": ((p_sw[:, K - 1] - t[:, K - 1]) ** 2).mean(-1).cpu().numpy(),
          "err_di": ((p_di[:, K - 1] - t[:, K - 1]) ** 2).mean(-1).cpu().numpy(),
          "gate": det["gate"][:, K - 1, :, 0].cpu().numpy(),
          "episode": np.array([data.episodes[int(data.episode_of[j])]["id"] for j in ex_idx]),
          "start": np.array([int(data.starts[j] - data.starts[data.episode_of == data.episode_of[j]].min()) for j in ex_idx])}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "droid.npz", bins=bins, counts=counts, **ex)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k != "checkpoints"}, indent=1))


if __name__ == "__main__":
    main()
