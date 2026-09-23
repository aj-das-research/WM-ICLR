"""Zero-shot transfer evaluation: a model trained on one feature cache, scored on another.

Default use (Table 1, "DROID cam 2 (0-shot)"): checkpoints trained on DROID exterior camera 1
(results/v2/droid/dinov2s/<arm>/s<seed>/best.pt) are evaluated on the camera-2 cache
(data/v2/features/droid_cam2/dinov2s, same episodes/splits) *using the training cache's normalisation
statistics* (features and actions standardised with droid/dinov2s/stats.json; the target cache's own
stats are never read). Output: results/v2/droid_cam2/dinov2s/<arm>/s<seed>/eval_test.npz in exactly the
format of train.evaluate(), plus summary.json and transfer.json (provenance), so scripts/v2/make_tables.py
picks the runs up without changes.

Usage (GPU):  python scripts/v2/eval_transfer.py [--arms ...] [--seeds 0 1 2]
CPU smoke:    python scripts/v2/eval_transfer.py --device cpu --max-episodes 2 --out /tmp/x --arms persistence
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from shiftwm.v2.analysis import ROOT, TEST_ARMS, load_model, read_cache, run_config, run_dirs, write_json
from shiftwm.v2.train import FeatureSplit, evaluate, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dataset", default="droid")
    p.add_argument("--target-dataset", default="droid_cam2")
    p.add_argument("--encoder", default="dinov2s")
    p.add_argument("--arms", nargs="*", default=list(TEST_ARMS))
    p.add_argument("--seeds", nargs="*", type=int, help="default: every seed with a best.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--eval-stride", type=int, help="default: training config eval_stride (2)")
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--out", help="default results/v2/<target>/<enc>")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    train_root = ROOT / "data/v2/features" / a.train_dataset / a.encoder
    target_root = ROOT / "data/v2/features" / a.target_dataset / a.encoder
    out_base = Path(a.out) if a.out else ROOT / "results/v2" / a.target_dataset / a.encoder
    t_man, t_stats = read_cache(train_root)
    g_man, _ = read_cache(target_root)
    for key in ("grid", "channels", "action_dim", "encoder"):
        if t_man[key] != g_man[key]:
            raise SystemExit(f"cache mismatch on {key}: {t_man[key]} vs {g_man[key]}")
    cache = {}
    for arm in a.arms:
        runs = run_dirs(a.train_dataset, arm, a.encoder, a.seeds)
        if not runs:
            print(json.dumps({"skip": arm, "reason": "no trained checkpoint"}), flush=True)
            continue
        for run in runs:
            dest = out_base / arm / run.name
            if (dest / f"eval_{a.split}.npz").exists() and not a.overwrite:
                print(json.dumps({"skip": f"{arm}/{run.name}", "reason": "done"}), flush=True)
                continue
            H, K, cfg = run_config(run)
            stride = a.eval_stride or cfg.get("eval_stride", 2)
            key = (H, K, stride, cfg.get("single_image", False))
            if key not in cache:
                cache.clear()
                torch.cuda.empty_cache() if a.device == "cuda" else None
                cache[key] = FeatureSplit(target_root, a.split, H, K, a.device, t_stats, stride, cfg.get("tasks"),
                                          a.max_episodes, single_image=key[3])
            data = cache[key]
            model = load_model(arm, run, t_man, H, K, a.device)
            t0 = time.time()
            ev = evaluate(model, data)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez(dest / f"eval_{a.split}.npz", **{k: np.asarray(v) for k, v in ev.items()})
            s = summary(ev)
            (dest / "summary.json").write_text(json.dumps({"event": "transfer", "results": {a.split: s}}, indent=1))
            ck = run / "best.pt"
            write_json(dest / "transfer.json", {
                "train_cache": str(train_root.relative_to(ROOT)), "target_cache": str(target_root.relative_to(ROOT)),
                "normalisation": str((train_root / "stats.json").relative_to(ROOT)),
                "checkpoint": str(ck.relative_to(ROOT)) if ck.exists() else None,
                "checkpoint_mtime": ck.stat().st_mtime if ck.exists() else None,
                "history": H, "horizon": K, "eval_stride": stride, "split": a.split,
                "max_episodes": a.max_episodes, "seconds": round(time.time() - t0, 1)})
            print(json.dumps({"done": f"{arm}/{run.name}", **s}), flush=True)
            del model


if __name__ == "__main__":
    main()
