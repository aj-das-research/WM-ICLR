#!/usr/bin/env python
"""Convert the Open-H-Embodiment Hamlyn dVRK subset into ShiftWM-v2 Stage-1 frames.

Input : data/medical/open_h/Surgical/hamlyn/<task>/  (LeRobot v2.1: parquet + MP4)
Output: data/v2/frames/openh_hamlyn/{manifest.json, episodes/<task>__<idx:06d>.npz}

See docs/v2_data_format.md (format) and docs/v2_openh_hamlyn.md (dataset-specific choices).

Per episode (T kept frames, stride s = round(fps/3), raw action dim A_raw=16, K = sub-commands
per step = round(action_hz/3); default action_hz=15 -> K=5, like DROID's 5 commands @ 15 Hz):
  images  uint8   [T, 224, 224, 3]  primary camera (modality "endoscope" =
                                    observation.images.color), full-frame antialiased resize
  actions float32 [T-1, K*16]       raw absolute-Cartesian actions resampled to action_hz inside
                                    each stride, concatenated (mirrors DROID's 5x7=35 block)
  proprio float32 [T, 30]           observation.state (16) ++ left joints (7) ++ right joints (7)

Decoding uses PyAV (system ffmpeg is absent on the login node).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DATASET = "openh_hamlyn"
TASKS = ["knot_tying", "needle_grasp_and_handover", "peg_transfer", "suturing_1",
         "suturing_2", "tissue_lifting", "tissue_retraction"]
IMG = 224
TARGET_STEP_S = 1.0 / 3.0  # DROID: 5 frames @ 15 Hz
MIN_T = 14
SPLIT_FRACS = (0.70, 0.15, 0.15)
ACTION_KEY = "action"  # == action.cartesian_absolute (verified identical)
PROPRIO_KEYS = ["observation.state", "observation.state.left_arm_joint",
                "observation.state.right_arm_joint"]
CAMERA_CANDIDATES = ["observation.images.color", "observation.images.endoscope_left",
                     "observation.images.left", "observation.images.endoscope"]
RESIZE_POLICY = ("full-frame resize (no crop; aspect ratio NOT preserved, e.g. 848x480 -> "
                 "224x224) with PIL Image.resize BILINEAR (antialiased/area-aware when "
                 "downsampling), from PyAV rgb24 decode")
ACTION_SEMANTICS = (
    "actions[k] = concat_{j=0..K-1} action[k*s + ceil((j+1)*fps/action_hz) - 1] (K=5 at the "
    "default action_hz=15: 30 Hz tasks keep raw rows k*10+1,3,5,7,9; the 15 Hz tissue_lifting "
    "task keeps rows k*5+0..4; i.e. the set-point at the END of each 1/15 s sub-interval) where "
    "action is the per-frame "
    "LeRobot 'action' (= 'action.cartesian_absolute'): absolute Cartesian set-point per PSM in "
    "its own PSM base frame, layout per arm [x,y,z (m), qx,qy,qz,qw, gripper (1 open, 0 "
    "closed)], left arm dims 0-7, right arm dims 8-15. action[i] ~= observation.state[i+1] "
    "(commanded next pose), so the block drives kept frame k*s to frame (k+1)*s. Absolute, not "
    "deltas; quaternion sign not canonicalised. block dim = K*16 = 80. (tissue_lifting is "
    "recorded at 15 Hz and there action[i] is closer to state[i] than state[i+1].)")
PROPRIO_SEMANTICS = (
    "proprio[k] at raw frame k*s = observation.state (16: same layout as action, measured) ++ "
    "observation.state.left_arm_joint (7: j1-j6 rad, gripper) ++ "
    "observation.state.right_arm_joint (7).")


# --------------------------------------------------------------------------- pure helpers
def choose_stride(fps: float, target_s: float = TARGET_STEP_S) -> int:
    return max(1, int(round(fps * target_s)))


def split_hash(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()


def assign_splits(episode_ids: list[str], fracs=SPLIT_FRACS) -> dict[str, str]:
    """Deterministic episode-disjoint split for ONE task: sort by sha256(id), cut 70/15/15."""
    ids = sorted(set(episode_ids), key=lambda e: (split_hash(e), e))
    n = len(ids)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    if n >= 3:  # guarantee non-empty val/test when possible
        n_val = max(n_val, 1)
        n_train = min(n_train, n - n_val - 1)
    out = {}
    for i, e in enumerate(ids):
        out[e] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def kept_indices(n_raw: int, stride: int) -> np.ndarray:
    return np.arange(0, n_raw, stride, dtype=np.int64)


def sub_offsets(fps: float, stride: int, action_hz: float | None) -> np.ndarray:
    """Raw-row offsets (within a stride) of the K sub-commands kept per step.

    action_hz=None keeps every raw row (K = stride). Otherwise K = round(stride/fps*action_hz)
    and sub-command j is the raw row at the end of the j-th 1/action_hz interval
    (ceil((j+1)*fps/action_hz) - 1); action_hz > fps repeats rows (zero-order hold)."""
    if action_hz is None:
        return np.arange(stride, dtype=np.int64)
    K = int(round(stride / fps * action_hz))
    off = np.array([math.ceil((j + 1) * fps / action_hz - 1e-9) - 1 for j in range(K)],
                   dtype=np.int64)
    assert off.min() >= 0 and off.max() < stride, off
    return off


def build_blocks(raw_actions: np.ndarray, raw_proprio: np.ndarray, stride: int,
                 offsets: np.ndarray | None = None):
    """raw_actions [N, A], raw_proprio [N, P] -> actions [T-1, K*A], proprio [T, P]."""
    if offsets is None:
        offsets = np.arange(stride, dtype=np.int64)
    n = raw_actions.shape[0]
    idx = kept_indices(n, stride)
    T = len(idx)
    A = raw_actions.shape[1]
    acts = np.empty((max(T - 1, 0), len(offsets) * A), dtype=np.float32)
    for k in range(T - 1):
        acts[k] = raw_actions[idx[k] + offsets].reshape(-1)
    return idx, acts, raw_proprio[idx].astype(np.float32)


def resize_frame(rgb: np.ndarray, size: int = IMG) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR), dtype=np.uint8)


# --------------------------------------------------------------------------- IO
def read_json(p: Path):
    return json.loads(Path(p).read_text())


def read_jsonl(p: Path):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def camera_key(info: dict, modality: dict | None) -> str:
    feats = info["features"]
    if modality:
        k = modality.get("video", {}).get("endoscope", {}).get("original_key")
        if k in feats:
            return k
    for k in CAMERA_CANDIDATES:
        if k in feats and feats[k]["dtype"] == "video":
            return k
    raise KeyError(f"no primary camera among {[k for k in feats if 'images' in k]}")


def load_parquet(path: Path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=[ACTION_KEY, *PROPRIO_KEYS, "frame_index"])
    col = lambda k: np.asarray(t.column(k).to_pylist(), dtype=np.float64)
    acts = col(ACTION_KEY)
    prop = np.concatenate([col(k) for k in PROPRIO_KEYS], axis=1)
    fidx = np.asarray(t.column("frame_index").to_pylist(), dtype=np.int64)
    return acts, prop, fidx


def decode_kept(video: Path, keep: set[int], max_idx: int):
    import av
    out, codec, n_dec = {}, None, 0
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        codec = s.codec_context.name
        for i, fr in enumerate(c.decode(s)):
            n_dec = i + 1
            if i in keep:
                out[i] = resize_frame(fr.to_ndarray(format="rgb24"))
            if i >= max_idx:
                break
    return out, codec, n_dec


def process_episode(job: dict) -> dict:
    t0 = time.time()
    out_path = Path(job["out_path"])
    rec = {k: job[k] for k in ("id", "task", "episode_index", "split", "instruction")}
    rec["file"] = f"episodes/{out_path.name}"
    try:
        acts, prop, fidx = load_parquet(Path(job["parquet"]))
        if not np.array_equal(fidx, np.arange(len(fidx))):
            order = np.argsort(fidx)
            acts, prop, fidx = acts[order], prop[order], fidx[order]
        n_rows = len(acts)
        stride = job["stride"]
        idx_all = kept_indices(n_rows, stride)
        frames, codec, n_dec = decode_kept(Path(job["video"]), set(idx_all.tolist()),
                                           int(idx_all[-1]) if len(idx_all) else 0)
        n_raw = n_rows
        if len(frames) < len(idx_all):  # video shorter than parquet: truncate
            n_vid = max(frames) + 1 if frames else 0
            n_raw = min(n_rows, n_vid)
        idx, a_blk, p_kept = build_blocks(acts[:n_raw], prop[:n_raw], stride,
                                          np.asarray(job["offsets"], dtype=np.int64))
        # last action block must be complete (it is, since idx[k]+stride <= idx[k+1] <= n_raw-1)
        T = len(idx)
        rec.update(T=int(T), n_raw=int(n_rows), n_video_decoded=int(n_dec), codec=codec)
        if T < MIN_T:
            rec["skipped"] = f"T={T}<{MIN_T}"
            return rec
        imgs = np.stack([frames[int(i)] for i in idx])
        if not (np.isfinite(a_blk).all() and np.isfinite(p_kept).all()):
            rec["skipped"] = "non-finite action/proprio"
            return rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez(tmp, images=imgs, actions=a_blk, proprio=p_kept)
        os.replace(tmp, out_path)
        rec["secs"] = round(time.time() - t0, 2)
    except Exception as e:  # noqa: BLE001
        rec["skipped"] = f"error: {type(e).__name__}: {e}"
    return rec


# --------------------------------------------------------------------------- driver
def scan_task(task_dir: Path, task: str, out_dir: Path, action_hz: float | None = 15.0):
    info = read_json(task_dir / "meta/info.json")
    mod_p = task_dir / "meta/modality.json"
    modality = read_json(mod_p) if mod_p.exists() else None
    eps = read_jsonl(task_dir / "meta/episodes.jsonl")
    tasks_txt = [r["task"] for r in read_jsonl(task_dir / "meta/tasks.jsonl")]
    cam = camera_key(info, modality)
    fps = float(info["fps"])
    stride = choose_stride(fps)
    offsets = sub_offsets(fps, stride, action_hz).tolist()
    feats = info["features"]
    assert feats[ACTION_KEY]["shape"] == [16], feats[ACTION_KEY]
    chunk_size = info.get("chunks_size", 1000)
    jobs, missing = [], []
    for e in eps:
        ei = int(e["episode_index"])
        ch = ei // chunk_size
        pq_p = task_dir / info["data_path"].format(episode_chunk=ch, episode_index=ei)
        vid_p = task_dir / info["video_path"].format(episode_chunk=ch, episode_index=ei,
                                                     video_key=cam)
        eid = f"{task}__{ei:06d}"
        if not (pq_p.exists() and vid_p.exists()):
            missing.append(eid)
            continue
        jobs.append(dict(id=eid, task=task, episode_index=ei, parquet=str(pq_p),
                         video=str(vid_p), stride=stride, offsets=offsets, length=int(e["length"]),
                         instruction=(e.get("tasks") or [""])[0],
                         out_path=str(out_dir / "episodes" / f"{eid}.npz")))
    # eligibility from metadata length (deterministic, before split assignment)
    elig = [j for j in jobs if len(kept_indices(j["length"], stride)) >= MIN_T]
    splits = assign_splits([j["id"] for j in elig])
    for j in jobs:
        j["split"] = splits.get(j["id"], "excluded")
    meta = dict(task=task, fps_source=fps, frame_stride=stride, step_seconds=stride / fps,
                action_raw_offsets=offsets, camera_key=cam,
                camera_shape=feats[cam]["shape"], video_codec=feats[cam].get("info", {}).get(
                    "video.codec"), n_episodes_meta=len(eps), n_missing_files=len(missing),
                missing=missing, task_texts=tasks_txt, robot_type=info.get("robot_type"),
                source_splits_ignored=info.get("splits"),
                video_keys=[k for k, v in feats.items() if v["dtype"] == "video"],
                parquet_columns=list(feats))
    return jobs, meta


def validate(out_dir: Path, manifest: dict) -> list[str]:
    errs = []
    A, P = manifest["action_dim"], manifest["proprio_dim"]
    for ep in manifest["episodes"]:
        z = np.load(out_dir / ep["file"])
        im, ac, pr = z["images"], z["actions"], z["proprio"]
        T = ep["T"]
        if im.shape != (T, IMG, IMG, 3) or im.dtype != np.uint8:
            errs.append(f"{ep['id']} images {im.shape} {im.dtype}")
        if ac.shape != (T - 1, A) or ac.dtype != np.float32 or not np.isfinite(ac).all():
            errs.append(f"{ep['id']} actions {ac.shape} {ac.dtype}")
        if pr.shape != (T, P) or pr.dtype != np.float32 or not np.isfinite(pr).all():
            errs.append(f"{ep['id']} proprio {pr.shape}")
        if T < MIN_T:
            errs.append(f"{ep['id']} T={T}")
        if im.std() < 1.0:
            errs.append(f"{ep['id']} images look constant")
    # episode-disjointness
    ids = [e["id"] for e in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        errs.append("duplicate episode ids")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="data/medical/open_h/Surgical/hamlyn")
    ap.add_argument("--out", default=f"data/v2/frames/{DATASET}")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="max episodes per task (debug)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--action-hz", type=float, default=15.0,
                    help="sub-command rate inside a step (15 -> K=5, DROID-like); "
                         "<=0 keeps every raw row (only valid if all tasks share one fps)")
    args = ap.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_jobs, task_meta = [], {}
    for task in args.tasks:
        td = src / task
        if not (td / "meta/info.json").exists():
            print(f"[warn] task {task} missing at {td}", file=sys.stderr)
            continue
        jobs, meta = scan_task(td, task, out, args.action_hz if args.action_hz > 0 else None)
        if args.limit:
            jobs = jobs[: args.limit]
        task_meta[task] = meta
        all_jobs += jobs
        print(f"[scan] {task}: {len(jobs)} eps, fps={meta['fps_source']} "
              f"stride={meta['frame_stride']} cam={meta['camera_key']} "
              f"missing={meta['n_missing_files']}", flush=True)
    steps = {round(m["step_seconds"], 6) for m in task_meta.values()}
    Ks = {len(m["action_raw_offsets"]) for m in task_meta.values()}
    assert len(steps) == 1 and len(Ks) == 1, (steps, Ks)
    K = Ks.pop()
    fps_by_task = {t: m["fps_source"] for t, m in task_meta.items()}
    fps = max(set(fps_by_task.values()), key=list(fps_by_task.values()).count)  # majority
    stride = choose_stride(fps)

    todo = [j for j in all_jobs if j["split"] != "excluded"]
    recs, pending = [], []
    for j in todo:
        if not args.overwrite and Path(j["out_path"]).exists():
            try:
                with np.load(j["out_path"]) as z:
                    T = int(z["images"].shape[0])
                recs.append({k: j[k] for k in ("id", "task", "episode_index", "split",
                                             "instruction")}
                            | dict(file=f"episodes/{Path(j['out_path']).name}", T=T,
                                   n_raw=j["length"], cached=True))
                continue
            except Exception:  # noqa: BLE001
                pass
        pending.append(j)
    if args.workers <= 1:
        results = (process_episode(j) for j in pending)
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers)
        results = (f.result() for f in as_completed([ex.submit(process_episode, j)
                                                     for j in pending]))
    for n, r in enumerate(results, 1):
        recs.append(r)
        if n % 50 == 0 or n == len(pending):
            print(f"[run] {n}/{len(pending)} done", flush=True)
    if args.workers > 1:
        ex.shutdown()

    kept = sorted([r for r in recs if "skipped" not in r], key=lambda r: r["id"])
    skipped = sorted([r for r in recs if "skipped" in r], key=lambda r: r["id"])
    skipped += [dict(id=j["id"], task=j["task"], skipped=f"length {j['length']} -> T<{MIN_T}")
                for j in all_jobs if j["split"] == "excluded"]
    manifest = {
        "dataset": DATASET, "version": 1, "frame_stride": stride, "fps_source": fps,
        "fps_note": "majority value; per-task fps_source/frame_stride in tasks[*] "
                    "(tissue_lifting is 15 Hz -> stride 5); every task has the same "
                    "step_seconds and action_dim",
        "frame_stride_per_task": {t: m["frame_stride"] for t, m in task_meta.items()},
        "fps_source_per_task": fps_by_task,
        "step_seconds": stride / fps, "action_hz": args.action_hz if args.action_hz > 0
        else fps, "action_subcommands_per_step": K,
        "action_dim": K * 16, "raw_action_dim": 16,
        "proprio_dim": 30, "image_size": IMG, "min_T": MIN_T,
        "resize_policy": RESIZE_POLICY, "action_semantics": ACTION_SEMANTICS,
        "proprio_semantics": PROPRIO_SEMANTICS,
        "camera": "primary camera = modality.json video.endoscope -> "
                  "observation.images.color (Intel RealSense D405 RGB, 848x480@30, h264); "
                  "the Hamlyn subset has no stereo endoscope pair, so this is the single "
                  "primary scene view. Wrist cams (INSKAM, 640x480) and depth are ignored.",
        "decoder": "PyAV (av) rgb24",
        "license": "CC-BY-4.0",
        "source": "nvidia/PhysicalAI-Robotics-Open-H-Embodiment (HF dataset), "
                  "Surgical/hamlyn/<task>, LeRobot v2.1; Hamlyn Centre, Imperial College "
                  "London (Su, Deng, Hu, Rodriguez y Baena, Giannarou 2026)",
        "splits_policy": "episode-disjoint, per task: eligible episodes (T>=14 from metadata "
                         "length) sorted by sha256(episode_id), first round(0.70n) train, "
                         "next round(0.15n) val, rest test; source info.json splits ignored; "
                         "frozen before any training",
        "tasks": task_meta,
        "skipped": skipped,
        "episodes": [{k: r[k] for k in ("id", "task", "split", "file", "T")} | {
            "episode_index": r["episode_index"], "n_raw": r.get("n_raw"),
            "instruction": r.get("instruction", "")} for r in kept],
    }
    manifest["counts"] = summarize(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    errs = validate(out, manifest)
    print_summary(manifest, skipped)
    if errs:
        print(f"[validate] {len(errs)} ERRORS:", *errs[:20], sep="\n  ")
        return 1
    print(f"[validate] OK: {len(manifest['episodes'])} episodes, shapes/finiteness/counts pass")
    return 0


def summarize(manifest):
    c = {}
    for e in manifest["episodes"]:
        d = c.setdefault(e["task"], {"train": 0, "val": 0, "test": 0, "T": []})
        d[e["split"]] += 1
        d["T"].append(e["T"])
    for t, d in c.items():
        T = np.array(d.pop("T"))
        d.update(n=int(len(T)), T_min=int(T.min()), T_median=float(np.median(T)),
                 T_mean=round(float(T.mean()), 1), T_max=int(T.max()), frames=int(T.sum()))
    return c


def print_summary(manifest, skipped):
    c = manifest["counts"]
    hdr = f"{'task':28s} {'train':>5s} {'val':>4s} {'test':>4s} {'n':>4s} {'Tmin':>5s} " \
          f"{'Tmed':>6s} {'Tmean':>6s} {'Tmax':>5s} {'frames':>7s}"
    print(hdr)
    print("-" * len(hdr))
    tot = dict(train=0, val=0, test=0, n=0, frames=0)
    for t, d in sorted(c.items()):
        print(f"{t:28s} {d['train']:5d} {d['val']:4d} {d['test']:4d} {d['n']:4d} "
              f"{d['T_min']:5d} {d['T_median']:6.1f} {d['T_mean']:6.1f} {d['T_max']:5d} "
              f"{d['frames']:7d}")
        for k in tot:
            tot[k] += d[k]
    print(f"{'TOTAL':28s} {tot['train']:5d} {tot['val']:4d} {tot['test']:4d} {tot['n']:4d} "
          f"{'':5s} {'':6s} {'':6s} {'':5s} {tot['frames']:7d}")
    print(f"stride={manifest['frame_stride']} fps={manifest['fps_source']} "
          f"action_dim={manifest['action_dim']} proprio_dim={manifest['proprio_dim']} "
          f"skipped={len(skipped)}")
    for s in skipped[:30]:
        print(f"  skipped {s['id']}: {s['skipped']}")


if __name__ == "__main__":
    sys.exit(main())
