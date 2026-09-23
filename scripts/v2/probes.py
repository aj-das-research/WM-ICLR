"""Linear (ridge) probes from frozen/predicted patch features to robot state (Table: tab:probes).

Probe input: standardised features -> [global mean-pool (C) ++ 4x4 average-pooled grid (16C)] = 17C dims
(6528 for DINOv2-S), each dimension z-scored with train statistics.
Targets (per frame t of the feature cache):
  DROID        commanded end-effector xyz = actions[t, 0:3] (first 7-D command of the 35-D block that
               starts at frame t; metres -> reported in cm). Frame T-1 has no action and is excluded.
  Open-H Hamlyn measured instrument-tip xyz of both PSMs = proprio[t, 0:3] (left) and proprio[t, 8:11]
               (right) (observation.state, metres in each PSM base frame, docs/v2_openh_hamlyn.md) -> mm.
Fit: closed-form ridge on TRUE train-split features (all frames, stride --train-stride); the ridge
strength is chosen on TRUE val-split features (min MAE) from a log grid; the probe is then frozen.
Test: FeatureSplit test windows (stride --stride, H=3); the target is the state at the future frame
t0+k (k=10). Rows: probe applied to the true future features (upper bound), and to each arm's k-step
forecast (persistence = observed features at t0). MAE = mean |error| over coordinates, averaged over
windows within an episode, then over episodes, then over seeds. Pearson r = per-coordinate correlation over
all test windows, averaged over coordinates (and seeds).

Outputs: results/v2/analysis/probes/<ds>/{probe.pt, <arm>/s<seed>.npz, true_features.npz},
results/v2/analysis/probes/summary.json, paper/submission_folder/tables/generated/probe_rows.tex.
Usage (GPU): python scripts/v2/probes.py        |   --rows-only to rebuild the LaTeX rows
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from shiftwm.v2.analysis import (ANALYSIS, ROOT, load_model, local_starts, predict, probe_features, read_cache,
                                 run_config, run_dirs, write_json)
from shiftwm.v2.train import FeatureSplit

DATASETS = ("droid", "openh_hamlyn")
ARMS = ("persistence", "ar_tf", "ar", "direct", "shiftwm")
ROWS = [("true", "True future features (upper bound)"), ("persistence", "Persistence"),
        ("ar_tf", "AR-TF (DINO-WM-style)"), ("ar", "AR"), ("direct", "Direct"), ("shiftwm", r"\ours{}")]
SPEC = {  # dataset: (array, columns, scale, unit, column names)
    "droid": ("actions", [0, 1, 2], 100.0, "cm", ["x", "y", "z"]),
    "openh_hamlyn": ("proprio", [0, 1, 2, 8, 9, 10], 1000.0, "mm", ["L.x", "L.y", "L.z", "R.x", "R.y", "R.z"]),
}
OUT = ANALYSIS / "probes"
GEN = ROOT / "paper/submission_folder/tables/generated"


def frame_targets(root, row, ds):
    """float32 [T, D] state per frame in report units; NaN where undefined."""
    key, cols, scale, _, _ = SPEC[ds]
    with np.load(root / row["file"]) as z:
        v = z[key][:, cols].astype(np.float64) * scale
    y = np.full((row["T"], len(cols)), np.nan)
    y[:len(v)] = v
    return y.astype(np.float32)


def load_frames(root, ds, split, stats, stride, device, max_episodes=None):
    manifest, _ = read_cache(root)
    rows = [r for r in manifest["episodes"] if r["split"] == split][:max_episodes]
    fm = torch.tensor(stats["feature_mean"], device=device)
    fs = torch.tensor(stats["feature_std"], device=device)
    X, Y = [], []
    for r in rows:
        with np.load(root / r["file"]) as z:
            f = torch.from_numpy(z["features"][::stride].astype(np.float32)).to(device)
        f = (f.reshape(f.shape[0], -1, f.shape[-1]) - fm) / fs
        y = torch.from_numpy(frame_targets(root, r, ds)[::stride]).to(device)
        ok = torch.isfinite(y).all(1)
        X.append(probe_features(f[ok], manifest["grid"])); Y.append(y[ok])
    return torch.cat(X), torch.cat(Y)


class Probe:
    def __init__(self, X, Y, Xv, Yv, alphas):
        X = X.double(); Y = Y.double()
        self.mu, self.sd = X.mean(0), X.std(0).clamp_min(1e-6)
        self.ymu = Y.mean(0)
        Xs = (X - self.mu) / self.sd
        evals, evecs = torch.linalg.eigh(Xs.T @ Xs)
        xty = evecs.T @ (Xs.T @ (Y - self.ymu))
        scale = evals.mean()
        self.val = {}
        best = None
        for a in alphas:
            W = evecs @ (xty / (evals + a * scale)[:, None])
            mae = float((self._apply(Xv, W) - Yv.double()).abs().mean())
            self.val[float(a)] = mae
            if best is None or mae < best[0]:
                best = (mae, float(a), W)
        self.val_mae, self.alpha, self.W = best

    def _apply(self, X, W):
        return ((X.double() - self.mu) / self.sd) @ W + self.ymu

    def __call__(self, X):
        return self._apply(X, self.W).float()

    def state(self):
        return {"mu": self.mu, "sd": self.sd, "ymu": self.ymu, "W": self.W, "alpha": self.alpha,
                "val_mae": self.val_mae, "val_grid": self.val}


def metrics(yhat, y, ep, n_ep):
    """Per-episode MAE [E] and pooled per-dimension Pearson r [D]."""
    err = (yhat - y).abs().mean(1).double()
    s = torch.zeros(n_ep, dtype=torch.float64, device=y.device).index_add_(0, ep, err)
    c = torch.zeros(n_ep, dtype=torch.float64, device=y.device).index_add_(0, ep, torch.ones_like(err))
    mae = (s / c.clamp_min(1)).cpu().numpy(); mae[c.cpu().numpy() == 0] = np.nan
    a, b = yhat.double() - yhat.double().mean(0), y.double() - y.double().mean(0)
    r = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)
    return mae, r.cpu().numpy()


def run_dataset(ds, a):
    root = ROOT / "data/v2/features" / ds / a.encoder
    manifest, stats = read_cache(root)
    key = SPEC[ds][0]
    if key == "proprio" and not manifest.get("proprio_dim"):
        print(json.dumps({"skip": ds, "reason": "no proprio in cache"})); return
    t0 = time.time()
    X, Y = load_frames(root, ds, "train", stats, a.train_stride, a.device, a.max_episodes)
    Xv, Yv = load_frames(root, ds, "val", stats, 2, a.device, a.max_episodes)
    probe = Probe(X, Y, Xv, Yv, np.logspace(-4, 3, 15))
    del X, Y, Xv, Yv
    (OUT / ds).mkdir(parents=True, exist_ok=True)
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in probe.state().items()}, OUT / ds / "probe.pt")
    print(json.dumps({"probe": ds, "alpha": probe.alpha, "val_mae": probe.val_mae, "sec": round(time.time() - t0, 1)}),
          flush=True)

    jobs = [(arm, run) for arm in a.arms for run in run_dirs(ds, arm, a.encoder, a.seeds)]
    H, K, _ = run_config(jobs[0][1]) if jobs else (3, 10, None)
    k = min(a.k, K)
    data = FeatureSplit(root, "test", H, K, a.device, stats, a.stride, None, a.max_episodes)
    rows = {r["id"]: r for r in manifest["episodes"]}
    ytab = [torch.from_numpy(frame_targets(root, rows[e["id"]], ds)).to(a.device) for e in data.episodes]
    t_target = (local_starts(data) + H - 1 + k).tolist()
    y_all = torch.stack([ytab[e][t] for e, t in zip(data.episode_of.tolist(), t_target)])
    valid = torch.isfinite(y_all).all(1)
    n_ep = len(data.episodes)
    names = SPEC[ds][4]

    def score(fn, dest):
        preds = []
        for i in range(0, len(data), a.batch_size):
            idx = torch.arange(i, min(i + a.batch_size, len(data)), device=data.features.device)
            preds.append(probe(probe_features(fn(idx), manifest["grid"])))
        yhat = torch.cat(preds)
        mae, r = metrics(yhat[valid], y_all[valid], data.episode_of[valid], n_ep)
        keep = np.isfinite(mae)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, mae=mae[keep], r=r, dims=np.array(names), unit=SPEC[ds][3], k=k,
                 episodes=np.array([e["id"] for e in data.episodes])[keep],
                 yhat=yhat[valid].cpu().numpy(), y=y_all[valid].cpu().numpy(),
                 episode_of=data.episode_of[valid].cpu().numpy())
        print(json.dumps({"done": str(dest), "mae": float(np.nanmean(mae)), "r": float(r.mean())}), flush=True)

    score(lambda idx: data.features[data.starts[idx] + H - 1 + k].float(), OUT / ds / "true_features.npz")
    for arm, run in jobs:
        model = load_model(arm, run, manifest, H, K, a.device)

        def fn(idx, model=model):
            hist, past, fut, _ = data.batch(idx)
            return predict(model, hist, past, fut)[:, k - 1]

        score(fn, OUT / ds / arm / f"{run.name}.npz")
        del model


def load_row(ds, key):
    files = [OUT / ds / "true_features.npz"] if key == "true" else sorted((OUT / ds / key).glob("s*.npz"))
    if key == "persistence":
        files = files[:1]
    files = [f for f in files if f.exists()]
    if not files:
        return None
    zs = [np.load(f) for f in files]
    out = {"mae": float(np.mean([np.nanmean(z["mae"]) for z in zs])), "r": float(np.mean([z["r"].mean() for z in zs])),
           "seeds": len(zs), "unit": str(zs[0]["unit"]),
           "r_per_dim": dict(zip(map(str, zs[0]["dims"]), np.mean([z["r"] for z in zs], 0).tolist()))}
    if ds == "openh_hamlyn":  # per-arm MAE (left dims 0-2, right 3-5)
        for side, sl in (("left", slice(0, 3)), ("right", slice(3, 6))):
            out[f"mae_{side}"] = float(np.mean([np.abs(z["yhat"][:, sl] - z["y"][:, sl]).mean() for z in zs]))
    return out


def write_rows(rows_file=None):
    summary, lines = {}, []
    for key, label in ROWS:
        cells = []
        for ds in DATASETS:
            r = load_row(ds, key)
            summary[f"{ds}/{key}"] = r
            cells += [r"\pend"] * 2 if r is None else [f"{r['mae']:.2f}", f"{r['r']:.3f}"]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    rows_file = Path(rows_file) if rows_file else GEN / "probe_rows.tex"
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    rows_file.write_text("% Generated by scripts/v2/probes.py (k=10; DROID cm, Hamlyn mm; mean over seeds)\n"
                         + "\n".join(lines) + "\n")
    write_json(OUT / "summary.json", summary)
    print("wrote", rows_file)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--encoder", default="dinov2s")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rows-only", action="store_true")
    p.add_argument("--out", help="override results/v2/analysis/probes (smoke tests)")
    p.add_argument("--rows-file", help="override tables/generated/probe_rows.tex (smoke tests)")
    a = p.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    if not a.rows_only:
        for ds in a.datasets:
            run_dataset(ds, a)
    write_rows(a.rows_file)


if __name__ == "__main__":
    main()
