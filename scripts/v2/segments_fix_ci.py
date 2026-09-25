"""CPU-only repair of the ABSOLUTE 95% CIs in results/v2/analysis/segments/<ds>/summary.json.

segments.py used to report absolute "mean" as the pooled mean over WINDOWS but "lo"/"hi" as a bootstrap of EPISODE
means (two different statistics; many means fell outside their own CI). This script recomputes only the absolute
lo/hi entries with the episode-cluster bootstrap of the pooled window mean (segments.boot_ci_cluster, n=2000, seed 0)
from the saved per-window arrays (perwindow.npz, decoded_perwindow.npz). Every "mean", "ep_mean", "diff_sw_minus" and
every other entry is left byte-identical (checked). Blocks whose per-window values were not saved are left unchanged
and listed.

Usage: PYTHONPATH=src python scripts/v2/segments_fix_ci.py [--datasets droid openh_hamlyn] [--dry-run]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segments import OUT0, PRIMARY, LABELLERS, boot_ci_cluster  # noqa: E402

ABS = ("lo", "hi")


def same(a, b):
    return json.dumps(a) == json.dumps(b)


def check_means(block, name, vals, per_horizon, where):
    """The saved per-window arrays must reproduce the published means exactly (else they are not the same run)."""
    m = block[name]
    if per_horizon:
        assert np.allclose(np.asarray(m["mean"], float), np.nanmean(vals, 0), rtol=0, atol=1e-12, equal_nan=True), where
        assert np.isclose(m["avg"]["mean"], float(np.nanmean(vals)), rtol=0, atol=1e-12), where
    else:
        assert np.isclose(m["mean"], float(np.nanmean(vals)), rtol=0, atol=1e-12), where


def fix_block(block, names, V, ep, sel, per_horizon, where, log):
    for n in names:
        check_means(block, n, V[n][sel], per_horizon, f"{where}/{n}")
        lo, hi = boot_ci_cluster(V[n][sel], ep[sel])
        if per_horizon:
            block[n]["lo"], block[n]["hi"] = lo.tolist(), hi.tolist()
            alo, ahi = boot_ci_cluster(V[n][sel], ep[sel], pool=True)
            block[n]["avg"]["lo"], block[n]["avg"]["hi"] = float(alo), float(ahi)
        else:
            block[n]["lo"], block[n]["hi"] = float(lo), float(hi)
    log.append(where)


def absolute_entries(S):
    """Yield (path, mean, lo, hi) for every absolute estimate (per-horizon arrays flattened; diffs excluded)."""
    def walk(d, path):
        if isinstance(d, dict):
            if {"mean", "lo", "hi"} <= d.keys():
                mean, lo, hi = (np.atleast_1d(np.asarray(d[k], float)) for k in ("mean", "lo", "hi"))
                yield path, mean, lo, hi
            for k, v in d.items():
                if k != "diff_sw_minus":
                    yield from walk(v, f"{path}/{k}")
    yield from walk({"results": S["results"], "placement": S["placement"], "decoded_segment": S["decoded_segment"]}, "")


def outside(S, prefix=""):
    bad, tot, bad_scalar, tot_scalar = [], 0, 0, 0
    for path, m, lo, hi in absolute_entries(S):
        ok = ~np.isfinite(m) | ((lo - 1e-12 <= m) & (m <= hi + 1e-12))
        tot += m.size; tot_scalar += m.size == 1
        if not ok.all():
            bad.append((prefix + path, int((~ok).sum())))
            bad_scalar += m.size == 1
    return bad, tot, bad_scalar, tot_scalar


def strip_abs(S):
    """Copy with every absolute lo/hi removed (what must stay byte-identical)."""
    S = copy.deepcopy(S)

    def walk(d, in_diff=False):
        if isinstance(d, dict):
            if not in_diff and {"mean", "lo", "hi"} <= d.keys():
                d.pop("lo"); d.pop("hi")
            for k, v in d.items():
                walk(v, in_diff or k == "diff_sw_minus")
    walk(S)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["droid", "openh_hamlyn"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for ds in args.datasets:
        out = OUT0 / ds
        txt = (out / "summary.json").read_text()
        S0 = json.loads(txt)
        assert json.dumps(S0, indent=1) == txt, "summary.json does not round-trip; refusing to rewrite"
        S = copy.deepcopy(S0)
        P = np.load(out / "perwindow.npz"); D = np.load(out / "decoded_perwindow.npz")
        names = [str(x) for x in P["names"]]; ep = P["episode"]; moving = P["moving"]
        assert len(ep) == S["windows_kept"]
        subsets = {"all": np.ones(len(ep), bool), "moving": moving}
        fixed, skipped = [], []
        # latent-stage IoU: only the primary labeller's per-window IoUs were saved
        for lab in LABELLERS:
            for sub, sel in subsets.items():
                if lab == PRIMARY:
                    V = {n: P[f"iou_{n}"] for n in names}
                    assert S["results"][lab][sub]["windows"] == int(sel.sum())
                    fix_block(S["results"][lab][sub], names, V, ep, sel, True, f"results/{lab}/{sub}", fixed)
                else:
                    skipped.append(f"results/{lab}/{sub}")
        # placement (display px): paired validity = every compared non-oracle method labels >= 1 cell at that k
        Vp = {n: P[f"place_{n}"] for n in names}
        valid = np.all(np.stack([np.isfinite(Vp[n]) for n in names if n != "oracle"]), 0)
        Pm = {n: np.where(valid, Vp[n], np.nan) for n in names}
        for sub, sel in subsets.items():
            fix_block(S["placement"]["px"][sub], names, Pm, ep, sel, True, f"placement/px/{sub}", fixed)
            skipped.append(f"placement/patches/{sub}")          # patch-unit distances were not saved per window
        # decoded segments (k = K)
        dnames = [str(x) for x in D["names"]]
        assert np.array_equal(D["episode"], ep) and np.array_equal(D["moving"], moving)
        both = np.all(np.stack([np.isfinite(D[f"place_{n}"]) for n in dnames if n != "decoded_truth"]), 0)
        for sub, sel in subsets.items():
            fix_block(S["decoded_segment"]["iou"][sub], dnames, {n: D[f"iou_{n}"] for n in dnames}, ep, sel, False,
                      f"decoded_segment/iou/{sub}", fixed)
            fix_block(S["decoded_segment"]["place_px"][sub], dnames, {n: D[f"place_{n}"] for n in dnames}, ep,
                      sel & both, False, f"decoded_segment/place_px/{sub}", fixed)
        # nothing but absolute lo/hi may change
        assert same(strip_abs(S0), strip_abs(S)), "a non-CI entry changed"
        b0, tot, bs0, ts = outside(S0); b1, _, bs1, _ = outside(S)
        fixed_bad = [p for p, _ in b1 if not any(p.startswith("/" + s) for s in skipped)]
        print(f"[{ds}] absolute estimates: {tot} values ({ts} scalar); outside own CI before: "
              f"{sum(c for _, c in b0)} values / {bs0} scalar; after: {sum(c for _, c in b1)} / {bs1}")
        print(f"[{ds}] recomputed: {fixed}")
        print(f"[{ds}] NOT reconstructible (left unchanged): {skipped}; still outside there: {b1}")
        assert not fixed_bad, f"mean outside its recomputed CI: {fixed_bad}"
        if not args.dry_run:
            (out / "summary.json").write_text(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
