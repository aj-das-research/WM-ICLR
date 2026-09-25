"""CPU robustness check for the segmentation view: recompute the latent-stage summaries (IoU and placement px, all and
moving windows) restricted to windows whose reference mask at t touches the frame border (an arm / instrument shaft
always enters the frame; a mask that does not may be a detector error on tissue or objects).

Same statistics as segments.py: pooled window means with episode-cluster bootstrap CIs; paired ShiftWM-minus-baseline
differences as means of per-episode (horizon-averaged) differences with an episode bootstrap. "moving" keeps the
full-set definition (motion >= median over all kept windows). Border contact is read from the cached 16x16 coverage
of the reference mask at t (windows.npz, foreground = coverage >= the dataset threshold).

Output: results/v2/analysis/segments/<ds>/border_restricted.json
Usage:  PYTHONPATH=src python scripts/v2/segments_border_check.py [--datasets openh_hamlyn]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segments import DATASETS, OUT0, boot_ci, boot_ci_cluster, boot_ci_nan  # noqa: E402

ARMS = ("shiftwm", "direct", "ar")


def summarise(V, ep, sel, nan_aware):
    eps = np.unique(ep[sel])

    def mean_ep(a):
        if not nan_aware:
            return a.mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, 0) if np.isfinite(a).any() else np.full(a.shape[1], np.nan)
    per_ep = {n: np.stack([mean_ep(V[n][sel & (ep == e)]) for e in eps]) for n in ARMS}
    avg = {n: (np.nanmean(v, 1) if nan_aware else v.mean(1)) for n, v in per_ep.items()}
    bci = boot_ci_nan if nan_aware else boot_ci
    res = {"windows": int(sel.sum()), "episodes": int(len(eps))}
    for n in ARMS:
        lo, hi = boot_ci_cluster(V[n][sel], ep[sel], pool=True)
        res[n] = {"mean": float(np.nanmean(V[n][sel])), "lo": float(lo), "hi": float(hi)}
        if n != "shiftwm":
            da = (avg["shiftwm"] - avg[n])[:, None]; dlo, dhi = bci(da)
            res[n]["diff_sw_minus"] = {"mean": float(np.nanmean(da)), "lo": float(dlo[0]), "hi": float(dhi[0])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["openh_hamlyn"])
    args = ap.parse_args()
    for ds in args.datasets:
        out = OUT0 / ds
        W = np.load(out / "windows.npz"); P = np.load(out / "perwindow.npz")
        keep = W["reason"] == 0
        assert np.array_equal(W["widx"][keep], P["widx"])
        g0 = W["cov"][keep][:, 0] >= DATASETS[ds]["cov"]                       # [n,16,16] reference mask at t
        border = g0[:, 0].any(1) | g0[:, -1].any(1) | g0[:, :, 0].any(1) | g0[:, :, -1].any(1)
        names = [str(x) for x in P["names"]]; ep = P["episode"]; moving = P["moving"]
        iou = {n: P[f"iou_{n}"] for n in ARMS}
        place = {n: P[f"place_{n}"] for n in names}
        valid = np.all(np.stack([np.isfinite(place[n]) for n in names if n != "oracle"]), 0)
        pm = {n: np.where(valid, place[n], np.nan) for n in ARMS}
        R = {"dataset": ds, "rule": "reference mask at t (16x16 coverage >= threshold) touches the frame border",
             "windows_total": int(len(ep)), "windows_border": int(border.sum()),
             "moving_total": int(moving.sum()), "moving_border": int((moving & border).sum())}
        for tag, sel0 in (("full", np.ones(len(ep), bool)), ("border", border)):
            R[tag] = {sub: {"iou": summarise(iou, ep, sel0 & s, False), "place_px": summarise(pm, ep, sel0 & s, True)}
                      for sub, s in (("all", np.ones(len(ep), bool)), ("moving", moving))}
        S = json.loads((out / "summary.json").read_text())                    # full set must reproduce the summary
        for sub in ("all", "moving"):
            for n in ("direct", "ar"):
                for m, ref in (("iou", S["results"][S["primary_labeller"]][sub][n]["avg"]["diff_sw_minus"]),
                               ("place_px", S["placement"]["px"][sub][n]["avg"]["diff_sw_minus"])):
                    got = R["full"][sub][m][n]["diff_sw_minus"]
                    assert all(np.isclose(got[k], ref[k], rtol=0, atol=1e-12) for k in ("mean", "lo", "hi")), (sub, n, m)
        (out / "border_restricted.json").write_text(json.dumps(R, indent=1))
        print(f"[{ds}] border-touching windows {R['windows_border']}/{R['windows_total']} "
              f"(moving {R['moving_border']}/{R['moving_total']})")
        for tag in ("full", "border"):
            for sub in ("all", "moving"):
                for m, sc in (("iou", 100), ("place_px", 1)):
                    r = R[tag][sub][m]
                    print(f"  {tag:6s} {sub:6s} {m:8s} n={r['windows']:4d}",
                          " ".join(f"{n}={r[n]['mean']:.3f}" for n in ARMS),
                          " ".join(f"SW-{n}={sc * r[n]['diff_sw_minus']['mean']:+.2f}[{sc * r[n]['diff_sw_minus']['lo']:+.2f},"
                                   f"{sc * r[n]['diff_sw_minus']['hi']:+.2f}]" for n in ("direct", "ar")))


if __name__ == "__main__":
    main()
