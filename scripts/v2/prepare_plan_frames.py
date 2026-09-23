#!/usr/bin/env python
"""Convert a LeWM planning dataset (PushT / TwoRoom / Reacher HDF5) into v2 Stage-1 frames.

The layout follows the official LeWM planning protocol: one model step is one action block of
5 native env steps (frameskip 5). Frame j is dataset row ``phase + 5*j`` of an episode, and
``actions[j]`` is the 10-dim concatenation of the 5 raw 2-dim actions applied between frame j and
frame j+1. This is the same ordering the planner uses when it un-blocks
``(horizon, 10) -> (25, 2)``. Images are the stored 224x224 RGB renders, left unresized.

Episodes are subsampled (seeded) until ``--max-frames`` strided frames are collected, which keeps
the DINOv2-S 16x16x384 fp16 cache at about 196 KB/frame x max_frames. Each episode gets a random
phase in {0..4} so that all block alignments are covered. The split is episode-disjoint: seed 0,
``--val-frac`` of the selected episodes go to val, and there is no test split because evaluation
is planning.

The manifest also stores the planner's action StandardScaler (population std over all finite
dataset rows, as in le-wm eval.py). Planning wrappers use it to map planner-normalised actions
back to raw units.

    python scripts/v2/prepare_plan_frames.py --env pusht --out data/v2/frames/plan_pusht
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if (ROOT / ".cache/v2_planning_pydeps").is_dir():
    sys.path.append(str(ROOT / ".cache/v2_planning_pydeps"))

DATASETS = {
    "pusht": ("data/upstream/pusht/pusht_expert_train.h5", "state"),
    "tworoom": ("data/upstream/tworoom/tworoom.h5", "proprio"),
    "reacher": ("data/upstream/reacher/reacher.h5", "qpos"),
}
BLOCK = 5

_H5 = None


def _open(path):
    global _H5
    if _H5 is None:
        import hdf5plugin  # noqa: F401  (Blosc filter on pixels)
        import h5py
        _H5 = h5py.File(path, "r")
    return _H5


def planner_scaler(f):
    """Numerically identical to sklearn StandardScaler fit on finite rows (le-wm eval.py)."""
    a = f["action"][:].astype(np.float64)
    a = a[~np.isnan(a).any(axis=1)]
    mean, var = a.mean(0), a.var(0)
    scale = np.sqrt(var)
    scale[scale < 10 * np.finfo(np.float64).eps] = 1.0
    return mean, scale


def episode_blocks(path, offset, length, phase, proprio_key):
    f = _open(path)
    sl = slice(offset, offset + length)
    act = f["action"][sl].astype(np.float32)
    n_frames = (length - 1 - phase) // BLOCK + 1
    # drop trailing frames whose incoming action block is incomplete or non-finite
    while n_frames > 1:
        a = act[phase:phase + BLOCK * (n_frames - 1)]
        if len(a) == BLOCK * (n_frames - 1) and np.isfinite(a).all():
            break
        n_frames -= 1
    rows = phase + BLOCK * np.arange(n_frames)
    pix = f["pixels"][offset + rows[0]: offset + rows[-1] + 1][:: BLOCK]
    actions = act[phase:phase + BLOCK * (n_frames - 1)].reshape(n_frames - 1, BLOCK * act.shape[1])
    out = {"images": np.ascontiguousarray(pix, dtype=np.uint8), "actions": actions}
    if proprio_key and proprio_key in f:
        out["proprio"] = f[proprio_key][sl][rows].astype(np.float32)
    return out, rows


def _work(job):
    path, out_dir, ep, offset, length, phase, split, proprio_key = job
    dest = Path(out_dir) / "episodes" / f"ep{ep:06d}.npz"
    data, rows = episode_blocks(path, offset, length, phase, proprio_key)
    T = len(data["images"])
    if T < 2:
        return None
    if not dest.exists():
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, **data)
        tmp.replace(dest)
    return {"id": f"ep{ep:06d}", "task": "", "split": split, "file": f"episodes/ep{ep:06d}.npz", "T": int(T),
            "source_episode": int(ep), "phase": int(phase), "first_row": int(offset + rows[0])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=tuple(DATASETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--h5", default=None)
    ap.add_argument("--max-frames", type=int, default=72000, help="strided frames (~196 KB each as DINOv2-S fp16)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    import h5py
    import hdf5plugin  # noqa: F401
    path = str(ROOT / (a.h5 or DATASETS[a.env][0]))
    proprio_key = DATASETS[a.env][1]
    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        lengths, offsets = f["ep_len"][:].astype(np.int64), f["ep_offset"][:].astype(np.int64)
        mean, scale = planner_scaler(f)
        raw_dim = f["action"].shape[1]
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(lengths))
    phases = rng.integers(0, BLOCK, size=len(lengths))
    chosen, total = [], 0
    for ep in order:
        n = (lengths[ep] - 1 - phases[ep]) // BLOCK + 1
        if n < 2:
            continue
        chosen.append(int(ep)); total += n
        if total >= a.max_frames:
            break
    n_val = max(1, int(round(a.val_frac * len(chosen))))
    split_of = {ep: ("val" if i < n_val else "train") for i, ep in enumerate(chosen)}  # chosen is already random
    jobs = [(path, str(out), ep, int(offsets[ep]), int(lengths[ep]), int(phases[ep]), split_of[ep], proprio_key)
            for ep in sorted(chosen)]
    records = []
    with get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(_work, jobs, chunksize=4)):
            if r is not None:
                records.append(r)
            if i % 200 == 0:
                print(json.dumps({"event": "frames", "n": i + 1, "total": len(jobs)}), flush=True)
    manifest = {
        "dataset": f"plan_{a.env}", "version": 1, "source": path, "frame_stride": BLOCK, "action_block": BLOCK,
        "action_dim": BLOCK * raw_dim, "raw_action_dim": raw_dim,
        "proprio_key": proprio_key,
        "resize_policy": "none (dataset renders are already 224x224 RGB)",
        "action_semantics": f"actions[j] = concat of the {BLOCK} raw {raw_dim}-dim env actions applied between "
                            f"frame j and j+1 (row-major, time-major), i.e. the LeWM action_block layout",
        "planner_action_scaler": {"mean": mean.tolist(), "scale": scale.tolist(),
                                  "note": "StandardScaler over all finite dataset action rows (le-wm eval.py)"},
        "splits_policy": f"episode-disjoint random subsample, seed {a.seed}, val_frac {a.val_frac}; no test (eval = planning)",
        "selection": {"max_frames": a.max_frames, "n_source_episodes": int(len(lengths)), "n_selected": len(chosen)},
        "license": "MIT (LeWM datasets, quentinll/lewm-*)",
        "episodes": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    counts = {s: sum(r["split"] == s for r in records) for s in ("train", "val")}
    frames = {s: sum(r["T"] for r in records if r["split"] == s) for s in ("train", "val")}
    print(json.dumps({"event": "complete", "episodes": counts, "frames": frames}), flush=True)


if __name__ == "__main__":
    main()
