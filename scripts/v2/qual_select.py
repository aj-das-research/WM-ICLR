"""Largest-advantage qualitative windows per forecasting benchmark (appendix figures qual_best_<ds>.pdf).

Selection rule (deterministic, stated in the paper):
  over ALL test windows (stride 2) compute, at forecast step k=10, the per-window MSE of ShiftWM, Direct and AR
  (mean over patches and channels of the standardised features) and the true change ||z_{t+10} - z_t||^2
  (same mean). Among the windows in the top 50% of true change, rank by the relative advantage
      adv = 1 - err_ShiftWM / min(err_Direct, err_AR)
  and keep the top 3 from distinct episodes. Every method is evaluated on the same window and frame.
Models: the lowest seed for which ShiftWM, Direct and AR all have a finished run (best.pt + summary.json), searched
under results/v2s/<ds>/dinov2s/<arm>/ first, then results/v2/<ds>/dinov2s/<arm>/. Datasets without all three are
recorded as "pending".

Outputs (per finished dataset)
  results/v2/analysis/qual_best/<ds>.npz  episode, t0, adv, err [3 x (shiftwm,direct,ar)], perpatch [3,3,G,G],
                                          frame_obs / frame_true (stored frames, uint8), decoded [3,4,224,224,3]
                                          (truth, shiftwm, direct, ar; only if a decoder exists), plus the
                                          per-window arrays over all test windows (err_all, change_all, ...)
  results/v2/analysis/qual_best/<ds>.json rule, checkpoints, chosen windows, advantages, dataset context
  results/v2/analysis/qual_best/summary.json  status per dataset (done / pending)
Usage (GPU): PYTHONPATH=src python scripts/v2/qual_select.py [--datasets droid ...] [--device cuda]
Rerun later (new datasets): sbatch -J qual_select --nice=20 --time=01:30:00 --mem=64G \
                              slurm_jebel/job.sbatch slurm_jebel/tasks/qual_select.sh
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from shiftwm.v2.analysis import (ANALYSIS, ROOT, FrameSource, cache_root, decode, decoder_path, load_decoder,
                                 load_model, local_starts, predict, read_cache, run_config, write_json)
from shiftwm.v2.train import FeatureSplit

DATASETS = ("droid", "openh_hamlyn", "bridge", "fractal", "language_table", "iws_pusht", "iws_box", "iws_rope")
ARMS = ("shiftwm", "direct", "ar")
BASES = (ROOT / "results/v2s", ROOT / "results/v2")
OUT = ANALYSIS / "qual_best"
K_SHOW, N_PICK, STRIDE = 10, 3, 2
RULE = ("Over all test windows (stride 2), per-window k=10 feature MSE of ShiftWM, Direct and AR and true change "
        "||z_{t+10}-z_t||^2 (both averaged over patches and channels). Among windows in the top 50% of true change, "
        "rank by relative advantage 1 - err_ShiftWM / min(err_Direct, err_AR); keep the top 3 from distinct episodes. "
        "All methods on the same window; lowest common finished seed.")


def find_runs(ds, encoder="dinov2s"):
    """{arm: run_dir} for the lowest seed where all ARMS have finished runs (first base that has them all)."""
    for base in BASES:
        seeds = None
        for arm in ARMS:
            s = {d.name for d in (base / ds / encoder / arm).glob("s*")
                 if (d / "best.pt").exists() and (d / "summary.json").exists()}
            seeds = s if seeds is None else seeds & s
        if seeds:
            seed = sorted(seeds, key=lambda n: int(n[1:]))[0]
            return {arm: base / ds / encoder / arm / seed for arm in ARMS}
    return None


def run_dataset(ds, a):
    runs = find_runs(ds, a.encoder)
    root = cache_root(ds, a.encoder)
    if runs is None or not (root / "manifest.json").exists():
        return {"status": "pending", "reason": "missing finished ShiftWM/Direct/AR runs" if runs is None else "no cache"}
    t_start = time.time()
    manifest, stats = read_cache(root)
    H, K, _ = run_config(runs["shiftwm"])
    for arm in ARMS[1:]:
        h2, k2, _ = run_config(runs[arm])
        assert (h2, k2) == (H, K), f"{ds}: history/horizon differ between arms"
    assert K >= K_SHOW, f"{ds}: horizon {K} < {K_SHOW}"
    data = FeatureSplit(root, "test", H, K, a.device, stats, STRIDE, None, a.max_episodes)
    models = {arm: load_model(arm, runs[arm], manifest, H, K, a.device) for arm in ARMS}
    j = K_SHOW - 1
    err = {arm: [] for arm in ARMS}
    change = []
    for i in range(0, len(data), a.batch_size):
        idx = torch.arange(i, min(i + a.batch_size, len(data)), device=data.starts.device)
        hist, past, fut, tgt = data.batch(idx)
        y = tgt[:, j]
        change.append(((y - hist[:, -1]) ** 2).mean((1, 2)).cpu())
        for arm, m in models.items():
            err[arm].append(((predict(m, hist, past, fut)[:, j] - y) ** 2).mean((1, 2)).cpu())
    err = {arm: torch.cat(v).double().numpy() for arm, v in err.items()}
    change = torch.cat(change).double().numpy()
    base = np.minimum(err["direct"], err["ar"])
    adv = 1 - err["shiftwm"] / np.maximum(base, 1e-12)
    thr = float(np.median(change))
    hi = change >= thr
    ep_of = data.episode_of.cpu().numpy()
    t0_all = (local_starts(data) + H - 1).cpu().numpy()
    order = np.where(hi)[0][np.argsort(-adv[hi], kind="stable")]
    picks, seen = [], set()
    for w in order:
        if ep_of[w] not in seen:
            picks.append(int(w)); seen.add(ep_of[w])
        if len(picks) == N_PICK:
            break
    # per-patch errors + decoded forecasts for the chosen windows
    idx = torch.tensor(picks, device=data.starts.device)
    hist, past, fut, tgt = data.batch(idx)
    y = tgt[:, j]
    preds = {arm: predict(m, hist, past, fut)[:, j] for arm, m in models.items()}
    g = manifest["grid"]
    perpatch = torch.stack([((preds[arm] - y) ** 2).mean(-1) for arm in ARMS], 1).reshape(len(picks), 3, g, g)
    frames = FrameSource(root)
    eps = [data.episodes[ep_of[w]]["id"] for w in picks]
    t0s = [int(t0_all[w]) for w in picks]
    obs = np.stack([frames.load_native(e, [t])[0] for e, t in zip(eps, t0s)])
    true = np.stack([frames.load_native(e, [t + K_SHOW])[0] for e, t in zip(eps, t0s)])
    extra = {}
    dpath = decoder_path(ds, a.encoder)
    if dpath.exists():
        dec = load_decoder(dpath, a.device)
        z = torch.stack([y] + [preds[arm] for arm in ARMS], 1)                 # [n, 4, N, C]
        img = decode(dec, z.reshape(-1, *z.shape[2:])).clamp(0, 1)
        extra["decoded"] = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8).reshape(
            len(picks), 4, *img.shape[2:], 3)
        extra["decoded_order"] = np.array(["truth", *ARMS])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{ds}.npz", episode=np.array(eps), t0=np.array(t0s), window=np.array(picks),
                        adv=adv[picks], err=np.stack([err[arm][picks] for arm in ARMS], 1), arms=np.array(ARMS),
                        change=change[picks], perpatch=perpatch.float().cpu().numpy(), frame_obs=obs,
                        frame_true=true, k=K_SHOW, err_all=np.stack([err[arm] for arm in ARMS], 1),
                        change_all=change, adv_all=adv, episode_of_all=ep_of, t0_all=t0_all, **extra)
    info = {
        "status": "done", "rule": RULE, "k": K_SHOW, "stride": STRIDE, "history": H, "horizon": K,
        "checkpoints": {arm: str((runs[arm] / "best.pt").relative_to(ROOT)) for arm in ARMS},
        "decoder": str(dpath.relative_to(ROOT)) if dpath.exists() else None,
        "n_windows": int(len(data)), "n_episodes": len(data.episodes), "change_median": thr,
        "picks": [{"episode": e, "t0": t, "adv": float(adv[w]), "change": float(change[w]),
                   **{f"err_{arm}": float(err[arm][w]) for arm in ARMS}} for e, t, w in zip(eps, t0s, picks)],
        # dataset context, so the selected examples are not mistaken for typical behaviour
        "context_high_change": {
            "frac_windows_shiftwm_best": float((adv[hi] > 0).mean()),
            "median_adv": float(np.median(adv[hi])),
            "mean_err": {arm: float(err[arm][hi].mean()) for arm in ARMS}},
        "context_all": {"mean_err": {arm: float(err[arm].mean()) for arm in ARMS},
                        "frac_windows_shiftwm_best": float((adv > 0).mean())},
        "sec": round(time.time() - t_start, 1),
    }
    write_json(OUT / f"{ds}.json", info)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--encoder", default="dinov2s")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-episodes", type=int, help="timing/smoke tests only")
    p.add_argument("--out", help="override results/v2/analysis/qual_best (tests)")
    a = p.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    summ_path = OUT / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    for ds in a.datasets:
        t = time.time()
        info = run_dataset(ds, a)
        summary[ds] = {k: info[k] for k in ("status", "reason") if k in info}
        if info["status"] == "done":
            summary[ds]["adv"] = [round(p_["adv"], 4) for p_ in info["picks"]]
        print(json.dumps({"dataset": ds, **summary[ds], "sec": round(time.time() - t, 1)}), flush=True)
        write_json(summ_path, summary)


if __name__ == "__main__":
    main()
