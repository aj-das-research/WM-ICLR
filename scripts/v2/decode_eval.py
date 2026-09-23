"""Decoded-pixel metrics (Table: tab:pixel) for every forecasting arm through one shared decoder.

For each dataset (DROID cam 1, Open-H Hamlyn) and each test window (stride --stride, same windows as
train.evaluate at stride 2), the forecast at k in {1,5,10} is decoded to RGB with the dataset's decoder
(scripts/v2/train_decoder.py, trained on true train features) and compared with the REAL future frame
(resized to 224x224 exactly like the encoder input): PSNR, SSIM (Gaussian 11x11) and LPIPS (AlexNet,
lpips v0.1). "Decoder on true features" decodes the true future features: the decoder's ceiling.
Per-episode means are saved per arm/seed; table cells average episodes, then seeds.

Outputs
  results/v2/analysis/pixel/<ds>/<arm>/s<seed>.npz   psnr/ssim/lpips [E, len(ks)], episodes, ks
  results/v2/analysis/pixel/<ds>/true_features.npz   upper bound
  results/v2/analysis/pixel/summary.json
  paper/submission_folder/tables/generated/pixel_rows.tex   (k=10 rows; \\pend where missing)
Usage (GPU): python scripts/v2/decode_eval.py            # everything available
             python scripts/v2/decode_eval.py --rows-only  # rebuild the LaTeX rows from saved npz
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from shiftwm.v2.analysis import (ANALYSIS, ROOT, FrameSource, decode, decoder_path, load_decoder, load_model,
                                 local_starts, predict, psnr, read_cache, run_config, run_dirs, ssim, write_json)
from shiftwm.v2.train import FeatureSplit

DATASETS = ("droid", "openh_hamlyn")
ARMS = ("persistence", "ar_tf", "ar", "direct", "shiftwm")
ROWS = [("true", "Decoder on true features (upper bound)"), ("persistence", "Persistence"),
        ("ar_tf", "AR-TF (DINO-WM-style)"), ("ar", "AR"), ("direct", "Direct"), ("shiftwm", r"\ours{}")]
OUT = ANALYSIS / "pixel"
GEN = ROOT / "paper/submission_folder/tables/generated"


class Scorer:
    def __init__(self, device):
        import lpips
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x, y):
        """x,y float [B,3,224,224] in [0,1] -> dict of [B] tensors."""
        return {"psnr": psnr(x, y), "ssim": ssim(x, y), "lpips": self.lp(x * 2 - 1, y * 2 - 1).flatten()}


def evaluate_dataset(ds, a, scorer):
    root = ROOT / "data/v2/features" / ds / a.encoder
    dpath = Path(a.decoder_root) / ds / a.encoder / "best.pt" if a.decoder_root else decoder_path(ds, a.encoder)
    if not dpath.exists():
        print(json.dumps({"skip": ds, "reason": f"missing decoder {dpath}"}), flush=True)
        return
    manifest, stats = read_cache(root)
    dec = load_decoder(dpath, a.device)
    frames = FrameSource(root)
    jobs = [(arm, run) for arm in a.arms for run in run_dirs(ds, arm, a.encoder, a.seeds)]
    H, K, _ = run_config(jobs[0][1]) if jobs else (3, 10, None)
    data = FeatureSplit(root, "test", H, K, a.device, stats, a.stride, None, a.max_episodes)
    ks = [k for k in a.ks if k <= K]
    ls = local_starts(data)
    t0s = ls + H - 1
    ep_frames = {}
    for e, info in enumerate(data.episodes):
        ep_frames[e] = torch.from_numpy(frames.load(info["id"])).to(a.device)
    n_ep = len(data.episodes)
    episodes = [e["id"] for e in data.episodes]

    def targets(idx):
        e, t = data.episode_of[idx].tolist(), t0s[idx].tolist()
        return [torch.stack([ep_frames[ei][ti + k] for ei, ti in zip(e, t)]).permute(0, 3, 1, 2).float() / 255
                for k in ks]

    def score(source, dest):
        if dest.exists() and not a.overwrite:
            print(json.dumps({"skip": str(dest), "reason": "done"}), flush=True)
            return
        t_start = time.time()
        sums = {m: torch.zeros(n_ep, len(ks), dtype=torch.float64, device=a.device) for m in ("psnr", "ssim", "lpips")}
        cnt = torch.zeros(n_ep, dtype=torch.float64, device=a.device)
        for i in range(0, len(data), a.batch_size):
            idx = torch.arange(i, min(i + a.batch_size, len(data)), device=data.features.device)
            z = source(idx)                                          # [B, len(ks), N, C]
            ys = targets(idx)
            ep = data.episode_of[idx]
            for j in range(len(ks)):
                x = decode(dec, z[:, j])
                for m, v in scorer(x, ys[j]).items():
                    sums[m][:, j].index_add_(0, ep, v.double())
            cnt.index_add_(0, ep, torch.ones_like(ep, dtype=torch.float64))
        keep = (cnt > 0).cpu().numpy()
        res = {m: (s / cnt.clamp_min(1)[:, None]).cpu().numpy()[keep] for m, s in sums.items()}
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, **res, episodes=np.array(episodes)[keep], ks=np.array(ks), windows=cnt.cpu().numpy()[keep],
                 decoder=str(dpath), stride=a.stride)
        print(json.dumps({"done": str(dest), "sec": round(time.time() - t_start, 1),
                          **{m: [round(float(x), 4) for x in res[m].mean(0)] for m in res}}), flush=True)

    kk = torch.tensor(ks, device=a.device) - 1

    def true_source(idx):
        s = data.starts[idx]
        t = s[:, None] + H - 1 + kk[None] + 1
        return data.features[t].float()

    score(true_source, OUT / ds / "true_features.npz")
    for arm, run in jobs:
        model = load_model(arm, run, manifest, H, K, a.device)

        def model_source(idx, model=model):
            hist, past, fut, _ = data.batch(idx)
            return predict(model, hist, past, fut)[:, kk]

        score(model_source, OUT / ds / arm / f"{run.name}.npz")
        del model


def load_metric(ds, key):
    if key == "true":
        files = [OUT / ds / "true_features.npz"]
    else:
        files = sorted((OUT / ds / key).glob("s*.npz"))
        if key == "persistence":
            files = files[:1]
    files = [f for f in files if f.exists()]
    if not files:
        return None
    zs = [np.load(f) for f in files]
    return {"ks": list(zs[0]["ks"]), "seeds": len(zs),
            **{m: float(np.mean([z[m].mean(0) for z in zs], 0)[list(zs[0]["ks"]).index(10)])
               if 10 in list(zs[0]["ks"]) else None for m in ("psnr", "ssim", "lpips")},
            "per_k": {m: np.mean([z[m].mean(0) for z in zs], 0).tolist() for m in ("psnr", "ssim", "lpips")}}


def write_rows(rows_file=None):
    summary, lines = {}, []
    for key, label in ROWS:
        cells = []
        for ds in DATASETS:
            r = load_metric(ds, key)
            summary[f"{ds}/{key}"] = r
            if r is None or r["psnr"] is None:
                cells += [r"\pend"] * 3
            else:
                cells += [f"{r['psnr']:.2f}", f"{r['ssim']:.3f}", f"{r['lpips']:.3f}"]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    rows_file = Path(rows_file) if rows_file else GEN / "pixel_rows.tex"
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    rows_file.write_text("% Generated by scripts/v2/decode_eval.py (k=10; mean over seeds)\n"
                                        + "\n".join(lines) + "\n")
    write_json(OUT / "summary.json", summary)
    print("wrote", rows_file)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--encoder", default="dinov2s")
    p.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rows-only", action="store_true")
    p.add_argument("--out", help="override results/v2/analysis/pixel (smoke tests)")
    p.add_argument("--decoder-root", help="override results/v2/analysis/decoder (smoke tests)")
    p.add_argument("--rows-file", help="override tables/generated/pixel_rows.tex (smoke tests)")
    a = p.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    if not a.rows_only:
        scorer = Scorer(a.device)
        for ds in a.datasets:
            evaluate_dataset(ds, a, scorer)
    write_rows(a.rows_file)


if __name__ == "__main__":
    main()
