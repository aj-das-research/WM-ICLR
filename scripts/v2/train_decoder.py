"""Train the shared feature -> RGB decoder used for all decoded-pixel metrics and visuals.

One decoder per dataset, trained only on TRUE train-split features (standardised with the cache's
train stats) -> the matching real frame resized to the encoder input (224x224, full-frame resize, no
crop). Loss: L1 + lambda * LPIPS(VGG) (lpips package, v0.1 linear heads). Model selection on the val
split (same loss); LPIPS(AlexNet) and PSNR are logged for reference. The decoder never sees predicted
features, so every forecasting arm is decoded by the same function.

Architecture (shiftwm.v2.analysis.FeatureDecoder): 16x16xC -> conv -> bilinear 28x28 -> 3 x
[nearest x2 upsample + conv + 2 residual blocks] -> 224x224x3, sigmoid (~18M params).

Output: results/v2/analysis/decoder/<dataset>/<enc>/{best.pt,last.pt,log.jsonl,config.json}; resumable.
Usage (GPU): python scripts/v2/train_decoder.py --dataset droid
CPU smoke:   python scripts/v2/train_decoder.py --dataset droid --device cpu --max-episodes 2 --steps 3 \
               --batch-size 2 --eval-every 3 --out /tmp/dec
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from shiftwm.v2.analysis import ANALYSIS, ROOT, FeatureDecoder, FrameSource, psnr, read_cache


def load_split(root, frames, split, stats, stride, device, max_episodes=None, workers=8):
    manifest, _ = read_cache(root)
    rows = [r for r in manifest["episodes"] if r["split"] == split and frames.exists(r["id"])]
    if max_episodes:
        rows = rows[:max_episodes]
    fm = torch.tensor(stats["feature_mean"], dtype=torch.float32)
    fs = torch.tensor(stats["feature_std"], dtype=torch.float32)

    def one(r):
        with np.load(root / r["file"]) as z:
            f = torch.from_numpy(z["features"][::stride].astype(np.float32))
        f = ((f.reshape(f.shape[0], -1, f.shape[-1]) - fm) / fs).half()
        im = torch.from_numpy(frames.load(r["id"], np.arange(0, r["T"], stride)))
        assert len(im) == len(f), r["id"]
        return f, im

    feats, ims = [], []
    with ThreadPoolExecutor(workers) as ex:
        for i, (f, im) in enumerate(ex.map(one, rows)):
            feats.append(f.to(device)); ims.append(im.to(device))
            if i % 100 == 0:
                print(json.dumps({"event": "load", "split": split, "n": i + 1, "of": len(rows)}), flush=True)
    return torch.cat(feats), torch.cat(ims)


def to_float(im):
    return im.permute(0, 3, 1, 2).float() / 255


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--encoder", default="dinov2s")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lpips-weight", type=float, default=1.0)
    p.add_argument("--frame-stride", type=int, default=1, help="use every n-th train frame")
    p.add_argument("--val-episodes", type=int, default=48)
    p.add_argument("--val-stride", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    import lpips

    torch.manual_seed(a.seed)
    dev = a.device
    root = ROOT / "data/v2/features" / a.dataset / a.encoder
    out = Path(a.out) if a.out else ANALYSIS / "decoder" / a.dataset / a.encoder
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    manifest, stats = read_cache(root)
    frames = FrameSource(root)
    t0 = time.time()
    trf, tri = load_split(root, frames, "train", stats, a.frame_stride, dev, a.max_episodes)
    vaf, vai = load_split(root, frames, "val", stats, a.val_stride, dev, a.max_episodes or a.val_episodes)
    log = open(out / "log.jsonl", "a")
    rec = {"event": "start", "train_frames": len(trf), "val_frames": len(vaf), "load_sec": round(time.time() - t0, 1)}
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n")

    arch = {"channels": manifest["channels"], "grid": manifest["grid"], "size": 224}
    dec = FeatureDecoder(**arch).to(dev)
    perc = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)
    perc_alex = lpips.LPIPS(net="alex", verbose=False).to(dev).eval().requires_grad_(False)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step, best = 0, float("inf")
    if (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        dec.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best = st["step"], st["best"]
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

    def losses(z, im):
        with torch.autocast(**amp):
            x = dec(z.float())
            y = to_float(im)
            l1 = (x - y).abs().mean()
            lp = perc(x * 2 - 1, y * 2 - 1).mean()
        return x, l1, lp

    @torch.no_grad()
    def validate():
        dec.eval()
        tot = {"l1": 0.0, "lpips_vgg": 0.0, "lpips_alex": 0.0, "psnr": 0.0}
        n = 0
        for i in range(0, len(vaf), 64):
            z, im = vaf[i:i + 64], vai[i:i + 64]
            x, l1, lp = losses(z, im)
            y = to_float(im)
            with torch.autocast(**amp):
                la = perc_alex(x * 2 - 1, y * 2 - 1).mean()
            b = len(z)
            tot["l1"] += float(l1) * b; tot["lpips_vgg"] += float(lp) * b; tot["lpips_alex"] += float(la) * b
            tot["psnr"] += float(psnr(x.float(), y).sum()); n += b
        dec.train()
        return {k: v / n for k, v in tot.items()}

    dec.train()
    t0 = time.time()
    while step < a.steps:
        idx = torch.randint(0, len(trf), (a.batch_size,), device=dev)
        _, l1, lp = losses(trf[idx], tri[idx])
        loss = l1 + a.lpips_weight * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            r = {"step": step, "loss": float(loss), "l1": float(l1), "lpips": float(lp), "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(r) + "\n"); log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            v = validate()
            score = v["l1"] + a.lpips_weight * v["lpips_vgg"]
            r = {"event": "val", "step": step, **v, "score": score, "sec": round(time.time() - t0, 1)}
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
            if score < best:
                best = score
                torch.save({"model": dec.state_dict(), "arch": arch, "step": step, "val": v,
                            "dataset": a.dataset, "encoder": a.encoder, "cache": str(root.relative_to(ROOT)),
                            "normalisation": "cache stats.json (train split)", "lpips_weight": a.lpips_weight},
                           out / "best.pt")
            torch.save({"model": dec.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best}, out / "last.pt")
    print(json.dumps({"event": "done", "step": step, "best": best}), flush=True)


if __name__ == "__main__":
    main()
