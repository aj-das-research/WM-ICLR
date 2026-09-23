#!/usr/bin/env python
"""Convert Open-X RLDS shards (BridgeData V2, RT-1/fractal) into ShiftWM-v2 Stage-1 frames.

Input : data/real_video/<ds>/raw/  (TFDS RLDS TFRecord shards fetched by
        scripts/v2/fetch_openx_subset.py + features.json/dataset_info.json)
Output: data/v2/frames/<ds>/{manifest.json, episodes/<id>.npz, _shards/<shard>.json}

See docs/v2_data_format.md (format). Per episode (T kept frames, stride s):
  images  uint8   [T, 224, 224, 3]  primary camera 'observation/image', full-frame
                                    antialiased resize (PIL BILINEAR, no crop)
  actions float32 [T-1, s*A_raw]    raw per-step actions of the s RLDS steps inside each
                                    model step, concatenated (DROID-style 5x7 block)
  proprio float32 [T, P]            robot state at each kept frame

  bridge : 5 Hz, s=2 (0.4 s/step), A_raw=7 [world_vector(3) EE delta xyz, rotation_delta(3)
           EE delta roll/pitch/yaw, open_gripper(1) {0,1}] -> action_dim 14; proprio =
           observation/state (7: EE xyz, rpy, gripper).
  fractal: 3 Hz, s=1 (0.333 s/step), A_raw=7 [world_vector(3), rotation_delta(3),
           gripper_closedness_action(1)] -> action_dim 7; the base_displacement_* dims (3) are
           checked to be identically zero (mobile base unused) and dropped; terminate_episode
           (discrete one-hot) dropped. proprio = base_pose_tool_reached(7: xyz + quaternion)
           ++ gripper_closed(1) = 8.

Parsing: raw tf.data.TFRecordDataset + tf.train.Example (the flattened RLDS layout
'steps/<key>'), no TFDS builder (only a shard subset is present). Resumable: a shard with a
_shards/<shard>.json record is skipped; an episode whose npz already exists is not re-decoded.

Run it inside an existing allocation (login node has a 5 GB RAM cgroup), e.g.
  srun --jobid=<RUNNING> --overlap -n1 -c8 bash -c 'CUDA_VISIBLE_DEVICES= \\
     environments/real_video/.venv/bin/python scripts/v2/prepare_openx.py --dataset bridge'
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

IMG = 224
MIN_T = 14
VAL_FRAC = 0.10
TEST_FRAC = 0.10  # fractal only (no official held-out split)
SALT = "shiftwm-v2-openx-split-v1"
RESIZE_POLICY = ("full-frame resize (no crop; aspect ratio NOT preserved, 640x480 or 320x256 "
                 "-> 224x224) with PIL Image.resize BILINEAR (antialiased when downsampling), "
                 "decoded from the RLDS PNG/JPEG bytes with PIL")

SPECS = {
    "bridge": dict(
        tfds_name="bridge", fps=5.0, stride=2,
        action_keys=[("steps/action/world_vector", 3), ("steps/action/rotation_delta", 3),
                     ("steps/action/open_gripper", 1)],
        zero_keys=[],
        proprio_keys=[("steps/observation/state", 7)],
        image_key="steps/observation/image",
        heldout_split="test",
        license="CC-BY-4.0 (BridgeData V2)",
        source="Open X-Embodiment RLDS gs://gresearch/robotics/bridge/0.1.0 (BridgeData V2, "
               "Walke et al. 2023, WidowX 250 toy kitchens); deterministic shard subset, see "
               "reports/evidence/v2/bridge_inventory.json",
        camera="'observation/image' = BridgeData V2 fixed over-the-shoulder camera image_0 "
               "(the OXE 0.1.0 release only ships this one view), 640x480 PNG",
        action_semantics=(
            "actions[k] = concat_{j=0,1} a[2k+j], a[i] = [world_vector (3): commanded EE "
            "translation delta xyz (m) for RLDS step i, rotation_delta (3): EE roll/pitch/yaw "
            "delta (rad), open_gripper (1): 1=open 0=close (bool->float)]; i.e. the two 5 Hz "
            "commands executed between kept frame 2k and 2k+2. terminate_episode dropped. "
            "RLDS step i pairs observation i with the action taken at it; in this release "
            "a[0] is ~all-zero. dim = 2*7 = 14."),
        proprio_semantics="proprio[k] = observation/state at raw step 2k: EE [x,y,z (m), roll, "
                          "pitch, yaw (rad), gripper opening] (7).",
        splits_policy=("episode-disjoint: every episode of the official OXE held-out split "
                       "'test' (= BridgeData V2 val) in the fetched subset -> test; official "
                       "'train' episodes -> val if sha256(SALT:id) / 2^256 < 0.10 else train; "
                       "episodes with T<14 kept frames dropped; frozen before any training"),
    ),
    "fractal": dict(
        tfds_name="fractal20220817_data", fps=3.0, stride=1,
        action_keys=[("steps/action/world_vector", 3), ("steps/action/rotation_delta", 3),
                     ("steps/action/gripper_closedness_action", 1)],
        zero_keys=[("steps/action/base_displacement_vector", 2),
                   ("steps/action/base_displacement_vertical_rotation", 1)],
        proprio_keys=[("steps/observation/base_pose_tool_reached", 7),
                      ("steps/observation/gripper_closed", 1)],
        image_key="steps/observation/image",
        heldout_split=None,
        license="CC-BY-4.0 (Open X-Embodiment / RT-1)",
        source="Open X-Embodiment RLDS gs://gresearch/robotics/fractal20220817_data/0.1.0 "
               "(RT-1, Brohan et al. 2022, Everyday Robots mobile manipulator); deterministic "
               "shard subset of the only split 'train', see "
               "reports/evidence/v2/fractal_inventory.json",
        camera="'observation/image' = the robot head camera, 320x256 JPEG (only view)",
        action_semantics=(
            "actions[k] = a[k] (stride 1 at 3 Hz), a = [world_vector (3): commanded EE "
            "translation delta for the step in the robot base frame (units as released; "
            "observed per-dim range in manifest raw_action_min/max), rotation_delta (3): EE rotation delta roll/pitch/yaw (rad), "
            "gripper_closedness_action (1): +1 close, -1 open, 0 no change]. "
            "base_displacement_vector (2) and base_displacement_vertical_rotation (1) are "
            "verified to be identically 0 in every converted episode and dropped; "
            "terminate_episode (int one-hot 3) dropped. dim = 7."),
        proprio_semantics="proprio[k] = base_pose_tool_reached (7: tool xyz + quaternion "
                          "xyzw in base frame) ++ gripper_closed (1) at step k (8).",
        splits_policy=("episode-disjoint by hash (the source has only 'train'): "
                       "test if sha256(SALT:test:id)/2^256 < 0.10, else val if "
                       "sha256(SALT:val:id)/2^256 < 0.10, else train (~10/9/81 %); episodes "
                       "with T<14 dropped; frozen before any training"),
    ),
}


# --------------------------------------------------------------------------- pure helpers
def hash_unit(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) / 2.0 ** 256


def assign_split(dataset: str, episode_id: str, source_split: str) -> str:
    heldout = SPECS[dataset]["heldout_split"]
    if heldout is not None:
        if source_split == heldout:
            return "test"
        return "val" if hash_unit(f"{SALT}:{episode_id}") < VAL_FRAC else "train"
    if hash_unit(f"{SALT}:test:{episode_id}") < TEST_FRAC:
        return "test"
    return "val" if hash_unit(f"{SALT}:val:{episode_id}") < VAL_FRAC else "train"


def parse_shard_name(path: Path):
    m = re.fullmatch(r"(.+)-(\w+)\.tfrecord-(\d{5})-of-(\d{5})", path.name)
    if not m:
        raise ValueError(f"not a TFDS shard: {path.name}")
    return m.group(2), int(m.group(3)), int(m.group(4))


def episode_id(dataset: str, source_split: str, shard: int, index: int) -> str:
    """Position in a pinned-generation shard (the RLDS records carry no episode id)."""
    return f"{dataset}__{source_split}{shard:05d}_{index:03d}"


def build_blocks(raw_actions: np.ndarray, raw_proprio: np.ndarray, stride: int):
    """raw_actions [N, A], raw_proprio [N, P] -> kept idx [T], actions [T-1, s*A], proprio [T, P].

    Kept frames 0, s, 2s, ...; block k = raw actions idx[k] .. idx[k]+s-1 (the commands that
    move kept frame k to k+1)."""
    n = raw_actions.shape[0]
    idx = np.arange(0, n, stride, dtype=np.int64)
    T = len(idx)
    acts = np.stack([raw_actions[i:i + stride].reshape(-1) for i in idx[:-1]]) if T > 1 else \
        np.zeros((0, stride * raw_actions.shape[1]))
    return idx, acts.astype(np.float32), raw_proprio[idx].astype(np.float32)


def resize_frame(rgb: np.ndarray, size: int = IMG) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def decode_resize(encoded: bytes) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(encoded)) as im:
        return resize_frame(np.asarray(im.convert("RGB")))


def feature_array(feature, width: int, n_steps: int) -> np.ndarray:
    kind = feature.WhichOneof("kind")
    arr = np.asarray(getattr(feature, kind).value, dtype=np.float64)
    if arr.size != n_steps * width:
        raise ValueError(f"size {arr.size} != {n_steps}x{width}")
    return arr.reshape(n_steps, width)


def parse_episode(features, spec) -> dict:
    """Pure: tf.train.Example feature map (or dict of Feature-like) -> raw arrays."""
    n = len(features["steps/is_first"].int64_list.value)
    acts = np.concatenate([feature_array(features[k], w, n) for k, w in spec["action_keys"]], 1)
    prop = np.concatenate([feature_array(features[k], w, n) for k, w in spec["proprio_keys"]], 1)
    zero_max = max([float(np.abs(feature_array(features[k], w, n)).max()) if n else 0.0
                    for k, w in spec["zero_keys"]] or [0.0])
    instr = {v.decode("utf-8") for v in
             features["steps/observation/natural_language_instruction"].bytes_list.value}
    images = features[spec["image_key"]].bytes_list.value
    if len(images) != n:
        raise ValueError("image count != steps")
    return dict(n=n, actions=acts, proprio=prop, zero_max=zero_max, images=images,
                instructions=sorted(instr))


def save_npz(path: Path, arrays: dict):
    """Standard NPZ, deflate level 1 (as prepare_droid.py), atomic."""
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, array in arrays.items():
            with z.open(name + ".npy", "w", force_zip64=True) as entry:
                np.lib.format.write_array(entry, np.asarray(array), allow_pickle=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- per-shard worker
def process_shard(job: dict) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    ds, spec = job["dataset"], SPECS[job["dataset"]]
    shard_path, out = Path(job["path"]), Path(job["out"])
    src_split, shard, _ = parse_shard_name(shard_path)
    t0 = time.time()
    recs = []
    for index, raw in enumerate(tf.data.TFRecordDataset(str(shard_path)).as_numpy_iterator()):
        eid = episode_id(ds, src_split, shard, index)
        rec = dict(id=eid, source_split=src_split, shard=shard, index=index,
                   split=assign_split(ds, eid, src_split),
                   example_sha256=hashlib.sha256(raw).hexdigest())
        try:
            ex = parse_episode(tf.train.Example.FromString(raw).features.feature, spec)
            idx, a_blk, p_kept = build_blocks(ex["actions"], ex["proprio"], spec["stride"])
            T = len(idx)
            rec.update(T=int(T), n_raw=int(ex["n"]), instruction=" | ".join(ex["instructions"]),
                       zero_dims_absmax=ex["zero_max"],
                       a_min=ex["actions"].min(0).tolist() if ex["n"] else None,
                       a_max=ex["actions"].max(0).tolist() if ex["n"] else None)
            if T < MIN_T:
                rec["skipped"] = f"T={T}<{MIN_T} (n_raw={ex['n']})"
            elif not (np.isfinite(a_blk).all() and np.isfinite(p_kept).all()):
                rec["skipped"] = "non-finite action/proprio"
            elif ex["zero_max"] != 0.0:
                rec["skipped"] = f"nonzero dropped dims (absmax {ex['zero_max']})"
            else:
                dest = out / "episodes" / f"{eid}.npz"
                if not dest.exists():
                    imgs = np.stack([decode_resize(ex["images"][int(i)]) for i in idx])
                    save_npz(dest, dict(images=imgs, actions=a_blk, proprio=p_kept))
                rec["file"] = f"episodes/{eid}.npz"
        except Exception as e:  # noqa: BLE001
            rec["skipped"] = f"error: {type(e).__name__}: {e}"
        recs.append(rec)
    result = dict(shard=shard_path.name, source_split=src_split, n_records=len(recs),
                  secs=round(time.time() - t0, 1), episodes=recs)
    tmp = out / "_shards" / (shard_path.name + ".json.tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(out / "_shards" / (shard_path.name + ".json"))
    return result


# --------------------------------------------------------------------------- driver
def validate(out_dir: Path, manifest: dict, full: bool = True) -> list[str]:
    errs = []
    A, P = manifest["action_dim"], manifest["proprio_dim"]
    for ep in manifest["episodes"]:
        with np.load(out_dir / ep["file"]) as z:
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
    ids = [e["id"] for e in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        errs.append("duplicate episode ids")
    return errs


def summarize(episodes):
    c = {}
    for s in ("train", "val", "test"):
        T = np.array([e["T"] for e in episodes if e["split"] == s])
        if len(T):
            c[s] = dict(n=int(len(T)), T_min=int(T.min()), T_p10=float(np.percentile(T, 10)),
                        T_median=float(np.median(T)), T_mean=round(float(T.mean()), 1),
                        T_p90=float(np.percentile(T, 90)), T_max=int(T.max()),
                        frames=int(T.sum()))
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=tuple(SPECS), required=True)
    ap.add_argument("--raw")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-shards", type=int, default=0, help="debug")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)
    ds, spec = args.dataset, SPECS[args.dataset]
    raw = Path(args.raw or f"data/real_video/{ds}/raw")
    out = Path(args.out or f"data/v2/frames/{ds}")
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "_shards").mkdir(parents=True, exist_ok=True)

    receipt = json.loads((raw / "download_receipt.json").read_text())
    shards = sorted(Path(f["local_path"]).name for f in receipt["files"] if ".tfrecord-" in
                    f["local_path"])
    shards = [raw / s for s in shards]
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    done = {p.name[: -len(".json")] for p in (out / "_shards").glob("*.json")}
    todo = [s for s in shards if s.name not in done]
    print(f"[scan] {ds}: {len(shards)} shards, {len(done & {s.name for s in shards})} done, "
          f"{len(todo)} to do", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(process_shard, dict(dataset=ds, path=str(s), out=str(out))) for s in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            n_ok = sum("skipped" not in e for e in r["episodes"])
            print(f"[run] {n}/{len(todo)} {r['shard']}: {n_ok}/{r['n_records']} eps "
                  f"{r['secs']}s", flush=True)

    recs = []
    for s in shards:
        recs += json.loads((out / "_shards" / (s.name + ".json")).read_text())["episodes"]
    kept = sorted([r for r in recs if "skipped" not in r], key=lambda r: r["id"])
    skipped = sorted([r for r in recs if "skipped" in r], key=lambda r: r["id"])
    n_short = sum(r["skipped"].startswith("T=") for r in skipped)
    A = spec["stride"] * sum(w for _, w in spec["action_keys"])
    P = sum(w for _, w in spec["proprio_keys"])
    manifest = {
        "dataset": ds, "version": 1, "frame_stride": spec["stride"], "fps_source": spec["fps"],
        "step_seconds": spec["stride"] / spec["fps"],
        "action_dim": A, "raw_action_dim": A // spec["stride"],
        "action_keys": [k for k, _ in spec["action_keys"]],
        "dropped_zero_action_keys": [k for k, _ in spec["zero_keys"]],
        "proprio_dim": P, "proprio_keys": [k for k, _ in spec["proprio_keys"]],
        "image_size": IMG, "min_T": MIN_T,
        "resize_policy": RESIZE_POLICY, "camera": spec["camera"],
        "action_semantics": spec["action_semantics"],
        "proprio_semantics": spec["proprio_semantics"],
        "license": spec["license"], "source": spec["source"],
        "tfds_name": spec["tfds_name"],
        "episode_id_policy": "<dataset>__<source split><shard:05d>_<index in shard:03d> of "
                             "the generation-pinned TFRecord shard (RLDS has no episode id); "
                             "example_sha256 in _shards/*.json identifies content",
        "splits_policy": spec["splits_policy"], "split_salt": SALT,
        "raw_receipt_sha256": hashlib.sha256((raw / "download_receipt.json").read_bytes()).hexdigest(),
        "n_shards": len(shards), "n_source_episodes": len(recs),
        "n_skipped": len(skipped), "n_skipped_short": n_short,
        "skipped": [{k: r.get(k) for k in ("id", "skipped")} for r in skipped],
        "episodes": [{"id": r["id"], "task": ds, "split": r["split"], "file": r["file"],
                      "T": r["T"], "n_raw": r["n_raw"], "source_split": r["source_split"],
                      "instruction": r["instruction"]} for r in kept],
    }
    manifest["raw_action_min"] = np.min([r["a_min"] for r in kept], 0).round(5).tolist()
    manifest["raw_action_max"] = np.max([r["a_max"] for r in kept], 0).round(5).tolist()
    manifest["n_multi_instruction"] = sum(" | " in r["instruction"] for r in kept)
    manifest["n_empty_instruction"] = sum(not r["instruction"].strip() for r in kept)
    manifest["counts"] = summarize(manifest["episodes"])
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(out / "manifest.json")
    print(json.dumps({"dataset": ds, "counts": manifest["counts"], "action_dim": A,
                      "proprio_dim": P, "source_episodes": len(recs), "skipped": len(skipped),
                      "skipped_short": n_short}, indent=1))
    for r in skipped:
        if not r["skipped"].startswith("T="):
            print("  skipped", r["id"], r["skipped"])
    if args.no_validate:
        return 0
    errs = validate(out, manifest)
    if errs:
        print(f"[validate] {len(errs)} ERRORS:", *errs[:20], sep="\n  ")
        return 1
    print(f"[validate] OK: {len(manifest['episodes'])} episodes, shapes/finiteness pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
