#!/usr/bin/env python
"""Convert IWS / RLA-WM real bimanual recordings (PushT, Box, Rope) into ShiftWM-v2 Stage-1.

Input : data/real_video/iws_public_v1/extracted/iws_converted/<src_task>/traj_<id>/
          camera_0_rgb.mp4 (640x480 @ 30 fps, h264), metadata.h5 (target_qpos, qpos, ...)
        data/real_video/iws_public_v1/download/eval_handles/iws/handles.<short>.json
Output: data/v2/frames/iws_<short>/{manifest.json, episodes/*.npz}   (docs/v2_data_format.md)

Splits (per task):
  train/val : the upstream *train* recordings (split_ranges.json "train"), episode-disjoint,
              val = 15% chosen by sha256(episode id) (deterministic); full recordings,
              kept frames 0, s, 2s, ...
  test      : exactly the 200 official RLA-WM validation handles (10 upstream *val*
              recordings). One Stage-1 "episode" per handle: kept frames
              start, start+s, ..., start+60 (T = 60/s + 1) and the 60 native command rows
              start..start+59 grouped into 60/s blocks.
  test_recordings : the same 10 reserved recordings in full (grid from frame 0), for
              diagnostics with real history only; never used for official numbers.

Model step: s = 5 native rows (1/6 s at 30 fps) -> the 60-row official horizon = K = 12 steps.
actions[k] = concat(target_qpos[k*s + j] for j in 0..s-1)  (native command rows; command row i
drives frame i -> i+1, exactly as v1 IWSWindowDataset). proprio = qpos (14-D) per kept frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SRC = Path("data/real_video/iws_public_v1/extracted/iws_converted")
HANDLES = Path("data/real_video/iws_public_v1/download/eval_handles/iws")
TASKS = {"pusht": "pusht", "box": "bimanual_box", "rope": "bimanual_rope"}  # short -> source dir
WIDTHS = {"pusht": 4, "bimanual_box": 14, "bimanual_rope": 8}
IMG = 224
STRIDE = 5
HANDLE_ROWS = 60
VAL_FRAC = 0.15
FPS = 30
RESIZE_POLICY = ("full-frame resize 640x480 -> 224x224 (no crop; aspect ratio not preserved), "
                 "PIL Image.resize BILINEAR (antialiased when downsampling), PyAV rgb24 decode")


# --------------------------------------------------------------------------- pure helpers
def split_hash(eid: str) -> str:
    return hashlib.sha256(eid.encode()).hexdigest()


def assign_train_val(ids: list[str], val_frac: float = VAL_FRAC) -> dict[str, str]:
    order = sorted(set(ids), key=lambda e: (split_hash(e), e))
    n_val = int(round(val_frac * len(order)))
    return {e: ("val" if i < n_val else "train") for i, e in enumerate(order)}


def grid_indices(n_rows: int, stride: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Kept frame rows start, start+s, ... <= last (stop exclusive, default n_rows)."""
    stop = n_rows if stop is None else min(stop, n_rows)
    return np.arange(start, stop, stride, dtype=np.int64)


def blocks(commands: np.ndarray, idx: np.ndarray, stride: int) -> np.ndarray:
    """commands [N, w]; kept rows idx [T] -> [T-1, stride*w], block k = rows idx[k]..idx[k]+s-1."""
    w = commands.shape[1]
    out = np.empty((max(len(idx) - 1, 0), stride * w), dtype=np.float32)
    for k in range(len(idx) - 1):
        assert idx[k + 1] - idx[k] == stride
        out[k] = commands[idx[k]: idx[k] + stride].reshape(-1)
    return out


def handle_window(start: int, n_rows: int, stride: int = STRIDE, rows: int = HANDLE_ROWS):
    """Official handle (start, 60 command rows) -> kept frame rows start..start+60 step s."""
    assert rows % stride == 0 and start + rows < n_rows, (start, rows, n_rows)
    return np.arange(start, start + rows + 1, stride, dtype=np.int64)


def resize_frame(rgb: np.ndarray, size: int = IMG) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR), dtype=np.uint8)


# --------------------------------------------------------------------------- IO
def load_h5(path: Path):
    import h5py
    with h5py.File(path, "r") as f:
        cmd = f["target_qpos"][...].astype(np.float32)
        qpos = f["qpos"][...].astype(np.float32)
        attrs = {k: (v.item() if hasattr(v, "item") else str(v)) for k, v in f.attrs.items()}
    return cmd, qpos, attrs


def decode(video: Path, keep: set[int]) -> tuple[dict[int, np.ndarray], int]:
    import av
    out, n = {}, 0
    last = max(keep) if keep else -1
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            n = i + 1
            if i in keep:
                out[i] = resize_frame(fr.to_ndarray(format="rgb24"))
            if i >= last:
                break
    return out, n


def save(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def process_recording(job: dict) -> list[dict]:
    """One source recording -> one or more Stage-1 episodes (full grid and/or handle windows)."""
    d = Path(job["dir"])
    cmd, qpos, attrs = load_h5(d / "metadata.h5")
    n = len(cmd)
    assert cmd.shape[1] == job["width"] and len(qpos) == n and np.isfinite(cmd).all(), d
    outs = []  # (record, idx)
    for spec in job["outputs"]:
        if spec["kind"] == "full":
            idx = grid_indices(n, job["stride"], spec.get("phase", 0))
        else:
            idx = handle_window(spec["start"], n, job["stride"])
        outs.append((spec, idx))
    keep = set(int(i) for _, idx in outs for i in idx)
    frames, n_dec = decode(d / "camera_0_rgb.mp4", keep)
    recs = []
    for spec, idx in outs:
        rec = {k: spec[k] for k in ("id", "split", "file", "recording", "kind")}
        rec.update(task=job["task"], session=f"{job['task']}__{job['episode_id']}",
                   n_raw=int(n), rows=[int(idx[0]), int(idx[-1])], T=int(len(idx)),
                   n_video_decoded=int(n_dec), success=bool(attrs.get("success", True)))
        if "start" in spec:
            rec.update(handle_index=spec["handle_index"], handle_start=spec["start"],
                       history_frames_available=int(spec["start"] // job["stride"]))
        missing = [int(i) for i in idx if int(i) not in frames]
        if missing:
            rec["skipped"] = f"video lacks rows {missing[:3]} (decoded {n_dec}, h5 {n})"
            recs.append(rec)
            continue
        imgs = np.stack([frames[int(i)] for i in idx])
        acts = blocks(cmd, idx, job["stride"])
        prop = qpos[idx]
        save(Path(job["out"]) / spec["file"], images=imgs, actions=acts, proprio=prop)
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- driver
def read_handles(short: str, task: str):
    p = HANDLES / f"handles.{short}.json"
    d = json.loads(p.read_text())
    assert d["meta"]["horizons"] == [HANDLE_ROWS] and d["meta"]["total_handles"] == 200, d["meta"]
    hs = d["handles_by_horizon"][str(HANDLE_ROWS)]
    for h in hs:
        assert h["group_name"] == task and h["sampled_horizon"] == HANDLE_ROWS
        assert h["camera_keys"] == ["camera_0"] and h["rgb_variant"] == "base"
    return hs, hashlib.sha256(p.read_bytes()).hexdigest(), str(p)


def plan_task(short: str, stride: int, out: Path, train_phases: int = 1):
    task = TASKS[short]
    root = SRC / task
    ranges = json.loads((root / "split_ranges.json").read_text())
    ids = sorted(p.name[5:] for p in root.glob("traj_*") if p.is_dir())
    up_val = [e for e in ids if ranges["val"]["start"] <= int(e) <= ranges["val"]["end"]]
    up_train = [e for e in ids if ranges["train"]["start"] <= int(e) <= ranges["train"]["end"]]
    assert len(up_val) + len(up_train) == len(ids)
    handles, hsha, hpath = read_handles(short, task)
    h_ids = sorted({h["traj_id"] for h in handles})
    assert h_ids == up_val, (h_ids, up_val)  # test recordings == upstream val == handle trajs
    assert len({(h["traj_id"], h["frame_id"]) for h in handles}) == 200
    tv = assign_train_val([f"{short}__{e}" for e in up_train])
    jobs = {}
    def job(eid):
        return jobs.setdefault(eid, dict(dir=str(root / f"traj_{eid}"), task=short, episode_id=eid,
                                         width=WIDTHS[task], stride=stride, out=str(out),
                                         outputs=[]))
    for e in up_train:
        split = tv[f"{short}__{e}"]
        for ph in range(train_phases if split == "train" else 1):
            eid = f"{short}__{e}" + (f"__p{ph}" if train_phases > 1 and split == "train" else "")
            job(e)["outputs"].append(dict(kind="full", phase=ph, id=eid, split=split,
                                          recording=e, file=f"episodes/{eid}.npz"))
    for e in up_val:
        eid = f"{short}__{e}__full"
        job(e)["outputs"].append(dict(kind="full", phase=0, id=eid, split="test_recordings",
                                      recording=e, file=f"episodes/{eid}.npz"))
    for i, h in enumerate(handles):
        eid = f"{short}__h{i:03d}__{h['traj_id']}_f{h['frame_id']:03d}"
        job(h["traj_id"])["outputs"].append(dict(kind="handle", start=int(h["frame_id"]),
                                                 handle_index=i, id=eid, split="test",
                                                 recording=h["traj_id"],
                                                 file=f"episodes/{eid}.npz"))
    meta = dict(source_task=task, source_dir=str(root), upstream_split_ranges=ranges,
                n_upstream_train=len(up_train), n_upstream_val=len(up_val),
                handles_path=hpath, handles_sha256=hsha, n_handles=len(handles),
                command_width=WIDTHS[task], task_description=task)
    return list(jobs.values()), meta


def validate(out: Path, m: dict) -> list[str]:
    errs, A, P = [], m["action_dim"], m["proprio_dim"]
    for e in m["episodes"]:
        with np.load(out / e["file"]) as z:
            im, ac, pr = z["images"], z["actions"], z["proprio"]
        T = e["T"]
        if im.shape != (T, IMG, IMG, 3) or im.dtype != np.uint8 or im.std() < 1:
            errs.append(f"{e['id']} images {im.shape}")
        if ac.shape != (T - 1, A) or ac.dtype != np.float32 or not np.isfinite(ac).all():
            errs.append(f"{e['id']} actions {ac.shape}")
        if pr.shape != (T, P) or pr.dtype != np.float32 or not np.isfinite(pr).all():
            errs.append(f"{e['id']} proprio {pr.shape}")
        if e["split"] == "test" and T != HANDLE_ROWS // m["frame_stride"] + 1:
            errs.append(f"{e['id']} test T={T}")
    ids = [e["id"] for e in m["episodes"]]
    if len(ids) != len(set(ids)):
        errs.append("duplicate ids")
    rec = {}
    for e in m["episodes"]:
        rec.setdefault(e["recording"], set()).add(e["split"])
    for r, s in rec.items():
        if ("train" in s or "val" in s) and len(s) > 1:
            errs.append(f"recording {r} crosses splits {s}")
    if sum(e["split"] == "test" for e in m["episodes"]) != 200:
        errs.append("test handle count != 200")
    return errs


def summarize(m):
    c = {}
    for e in m["episodes"]:
        d = c.setdefault(e["split"], [])
        d.append(e["T"])
    return {s: dict(n=len(T), T_min=int(min(T)), T_median=float(np.median(T)),
                    T_max=int(max(T)), frames=int(sum(T))) for s, T in c.items()}


def run_task(short: str, args) -> int:
    out = Path(args.out_root) / f"iws_{short}"
    out.mkdir(parents=True, exist_ok=True)
    jobs, meta = plan_task(short, args.stride, out, args.train_phases)
    if args.limit:
        jobs = jobs[: args.limit]
    todo, recs = [], []
    for j in jobs:
        if not args.overwrite and all((out / o["file"]).exists() for o in j["outputs"]):
            for o in j["outputs"]:
                with np.load(out / o["file"]) as z:
                    T = int(z["images"].shape[0])
                recs.append({k: o[k] for k in ("id", "split", "file", "recording", "kind")}
                            | dict(task=short, session=f"{short}__{j['episode_id']}", T=T,
                                   cached=True) | ({"handle_index": o["handle_index"],
                                                    "handle_start": o["start"]}
                                                   if "start" in o else {}))
        else:
            todo.append(j)
    print(f"[{short}] {len(jobs)} recordings, {len(todo)} to process", flush=True)
    if args.workers <= 1:
        it = (process_recording(j) for j in todo)
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers)
        it = (f.result() for f in as_completed([ex.submit(process_recording, j) for j in todo]))
    for n, r in enumerate(it, 1):
        recs += r
        if n % 100 == 0 or n == len(todo):
            print(f"[{short}] {n}/{len(todo)}", flush=True)
    if args.workers > 1:
        ex.shutdown()
    kept = sorted([r for r in recs if "skipped" not in r], key=lambda r: r["id"])
    skipped = [r for r in recs if "skipped" in r]
    s = args.stride
    w = meta["command_width"]
    m = {
        "dataset": f"iws_{short}", "version": 1, "frame_stride": s, "fps_source": FPS,
        "step_seconds": s / FPS, "action_dim": s * w, "raw_action_dim": w, "proprio_dim": 14,
        "image_size": IMG, "resize_policy": RESIZE_POLICY,
        "action_semantics": (
            f"actions[k] = concat_(j=0..{s - 1}) target_qpos[row_k + j] ({s} native command rows "
            f"x {w}-D, block dim {s * w}); command row i drives native frame i -> i+1 (as v1 "
            "IWSWindowDataset / official handles). target_qpos is the recorded command vector "
            "(units unverified; pusht 4-D, box 14-D bimanual joint targets, rope 8-D)."),
        "proprio_semantics": "qpos (14-D recorded joint positions) at each kept frame",
        "license": "see HF xyzhang368/RLA-WM (IWS subset); research use",
        "source": "HF dataset xyzhang368/RLA-WM iws_converted.tar + eval_handles/iws "
                  "(revision in configs/real_video_development/iws_acquisition_v1.json)",
        "splits_policy": (
            "train/val: upstream train recordings, episode-disjoint, val = first "
            "round(0.15 n) by sha256('<task>__<traj_id>'); test: one episode per official "
            "RLA-WM validation handle (200, horizon 60 native rows) from the 10 upstream val "
            "recordings; test_recordings: those 10 recordings in full (diagnostic only). "
            "Frozen before any v2 training."),
        "test_protocol": {
            "official": "handle (traj_id, frame_id=s): single RGB frame s, 60 command rows "
                        "s..s+59, targets frames s+1..s+59 (59 offsets)",
            "ours": f"K={HANDLE_ROWS // s} model steps of {s} rows: context frame s, action "
                    f"blocks rows s+{s}k..s+{s}k+{s - 1}, targets frames s+{s}..s+{HANDLE_ROWS} "
                    f"(offsets {s},{2 * s},..,{HANDLE_ROWS}); offsets {s}..{HANDLE_ROWS - s} are "
                    f"official targets, offset {HANDLE_ROWS} (step K) is one row beyond the "
                    "official 59 but exists by the upstream s+60<N rule and uses only the "
                    "handle's own 60 command rows. Prefix horizons 15/30/45/60 rows = "
                    f"steps {15 // s}/{30 // s}/{45 // s}/{60 // s}.",
            "single_image": "official protocol gives one observation: at eval replicate "
                            "frame 0 into the H-frame history with neutral past actions "
                            "(suggested trainer flag single_image=True); "
                            "history_frames_available records how much real history exists.",
        },
        "tasks": {short: meta},
        "train_phases": args.train_phases,
        "skipped": skipped,
        "episodes": [{k: r.get(k) for k in ("id", "task", "split", "file", "T", "session",
                                             "recording", "kind")}
                     | ({"handle_index": r["handle_index"], "handle_start": r["handle_start"]}
                        if "handle_start" in r else {}) for r in kept],
    }
    m["counts"] = summarize(m)
    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    errs = validate(out, m)
    print(f"[{short}] action_dim={m['action_dim']} proprio_dim=14 stride={s} "
          f"skipped={len(skipped)}")
    for sp, c in sorted(m["counts"].items()):
        print(f"  {sp:16s} n={c['n']:4d} T min/med/max={c['T_min']}/{c['T_median']:.0f}/"
              f"{c['T_max']} frames={c['frames']}")
    if errs:
        print(f"[{short}] VALIDATION ERRORS ({len(errs)}):", *errs[:20], sep="\n  ")
        return 1
    print(f"[{short}] validate OK ({len(m['episodes'])} episodes)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--out-root", default="data/v2/frames")
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--train-phases", type=int, default=1,
                    help="store train recordings at this many grid phases (0..n-1)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="debug: max recordings per task")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    assert HANDLE_ROWS % args.stride == 0 and 1 <= args.train_phases <= args.stride
    rc = 0
    for short in args.tasks:
        rc |= run_task(short, args)
    return rc


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"done in {time.time() - t0:.0f}s")
    sys.exit(rc)
