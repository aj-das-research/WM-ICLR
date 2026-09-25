"""Per-episode win map on held-out DROID: where does ShiftWM beat (or lose to) Direct and AR?

Source: the Table-1 test evaluations (train.evaluate, eval_stride 2): per-episode, per-horizon feature MSE
results/v2s/droid/dinov2s/<arm>/s<seed>/eval_test.npz (mean over the episode's windows, patches and channels).
Seed: s0, the checkpoints of every other analysis (geometry.py, anatomy.py); seeds s1, s2 are used only for the
robustness numbers (sign agreement across seeds).
  rel[e, k] = (err_ShiftWM - err_base) / err_base   (< 0: ShiftWM better), base in {Direct, AR}
  motion[e] = persistence error averaged over k = the episode's mean true feature change ||z_{t0+k} - z_{t0}||^2
Episodes are sorted by motion. An episode "loses" when its horizon-averaged rel is > 0.
Writes results/v2/analysis/winmap/{winmap.npz, summary.json}. CPU only.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/v2s/droid/dinov2s"
OUT = ROOT / "results/v2/analysis/winmap"
SEEDS = (0, 1, 2)


def ev(arm, seed):
    z = np.load(BASE / arm / f"s{seed}/eval_test.npz", allow_pickle=True)
    return z["mse"], z["episodes"], z["windows"]


def main():
    per, eps, win = {}, None, None
    for a in ("shiftwm", "direct", "ar"):
        for s in SEEDS:
            m, e, w = ev(a, s)
            if eps is None:
                eps, win = e, w
            assert (e == eps).all()
            per[(a, s)] = m
    pers, e, _ = ev("persistence", 0); assert (e == eps).all()
    motion = pers.mean(1)
    order = np.argsort(motion, kind="stable")
    out = {"episodes": eps[order], "windows": win[order], "motion": motion[order], "persistence": pers[order]}
    summ = {"episodes": int(len(eps)), "windows": int(win.sum()), "seed": 0, "rule": __doc__}
    for base in ("direct", "ar"):
        rel = {s: (per[("shiftwm", s)] - per[(base, s)]) / per[(base, s)] for s in SEEDS}
        r0 = rel[0][order]
        out[f"rel_{base}"] = r0
        avg = r0.mean(1)
        lose = avg > 0
        agree = np.mean([np.sign(rel[s].mean(1)) == np.sign(rel[0].mean(1)) for s in SEEDS[1:]], 0)[order]
        lose_all = np.all([rel[s].mean(1) > 0 for s in SEEDS], 0)[order]
        out[f"lose_{base}"] = lose; out[f"lose_allseeds_{base}"] = lose_all
        q = np.array_split(np.arange(len(order)), 4)               # motion quartiles (sorted order)
        summ[base] = {
            "win_frac_by_k": (r0 < 0).mean(0).tolist(),
            "win_frac_avg": float((avg < 0).mean()), "n_lose_avg": int(lose.sum()),
            "n_lose_all_seeds": int(lose_all.sum()), "sign_agree_other_seeds": float(agree.mean()),
            "median_rel_by_k": np.median(r0, 0).tolist(), "mean_rel_avg": float(avg.mean()),
            "cells_lose": float((r0 > 0).mean()),
            "win_frac_by_motion_quartile": [float((avg[i] < 0).mean()) for i in q],
            "median_rel_by_motion_quartile": [float(np.median(avg[i])) for i in q],
            "spearman_motion_vs_rel": list(map(float, spearmanr(motion[order], avg))),
            "losing_episodes": [{"episode": str(out["episodes"][i]), "rel_avg": float(avg[i]), "motion": float(out["motion"][i]),
                                 "windows": int(out["windows"][i]), "motion_rank": int(i)} for i in np.where(lose)[0]],
            "windows_in_losing": int(out["windows"][lose].sum()),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "winmap.npz", **out)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    for b in ("direct", "ar"):
        s = summ[b]
        print(b, {k: s[k] for k in ("win_frac_avg", "n_lose_avg", "n_lose_all_seeds", "sign_agree_other_seeds", "cells_lose",
                                    "win_frac_by_motion_quartile", "median_rel_by_motion_quartile", "spearman_motion_vs_rel",
                                    "windows_in_losing")})
        print("  win frac by k", np.round(s["win_frac_by_k"], 2).tolist())


if __name__ == "__main__":
    main()
