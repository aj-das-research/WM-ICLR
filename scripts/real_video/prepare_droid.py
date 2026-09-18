#!/usr/bin/env python3
"""Audit and extract real DROID RLDS episodes, preserving session separation.

The authoritative raw JPEG TFRecords remain unchanged. Each camera gets every
fifth observed frame and exactly the five intervening recorded 7-D commands.
No artificial shift, simulation, generated action, or inferred clock is added.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
SALT = "shiftwm-real-split-v1"
ACTION_SEMANTICS = "commanded_cartesian_position_6_plus_gripper_position_1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, arrays):
    """Lossless standard NPZ, compression level1 to bound CPU extraction cost."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, array in arrays.items():
            with z.open(name + ".npy", "w", force_zip64=True) as entry:
                np.lib.format.write_array(entry, np.asarray(array), allow_pickle=False)
    temporary.replace(path)


def session_from_metadata(file_path, recording_path):
    """Site/calendar-day grouping, deliberately coarser than an episode."""
    def extract(value):
        parts = Path(value).parts
        candidates = [(i, part) for i, part in enumerate(parts)
                      if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part)]
        if len(candidates) != 1:
            raise ValueError("Cannot identify one recording date; no episode-level split fallback")
        i, day = candidates[0]
        if i < 2 or parts[i-1] not in {"success", "failure", "failures"}:
            raise ValueError("Unrecognized site/status/date layout; must review metadata before splitting")
        site = parts[i-2]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", site):
            raise ValueError("Invalid collection-site key")
        return site + "/" + day
    a, b = extract(file_path), extract(recording_path)
    if a != b:
        raise ValueError("Episode and camera-recording session identities disagree")
    return a


def split_for(session):
    digest = hashlib.sha256((SALT + ":" + session).encode()).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if score < .70 else "val" if score < .85 else "test"


def group_actions(native):
    if native.ndim != 2 or native.shape[1] != 7 or len(native) < 1 or not np.isfinite(native).all():
        raise ValueError("Invalid native recorded actions")
    blocks = (len(native)-1)//5
    indices = np.arange(blocks+1, dtype=np.int64)*5
    return native[:blocks*5].reshape(blocks, 35).astype(np.float32), indices


def verify_episode(row, output):
    output = Path(output)
    audit_path = output / row["audit"]["file"]
    if sha256(audit_path) != row["audit"]["sha256"]:
        raise ValueError("Changed native audit payload")
    with np.load(audit_path, allow_pickle=False) as z:
        native = z["action"].copy()
        known = np.concatenate([z["action_dict__cartesian_position"], z["action_dict__gripper_position"]], axis=1)
        if not np.array_equal(native, known):
            raise ValueError("Recorded action does not equal commanded Cartesian pose+gripper")
        expected, frame_indices = group_actions(native)
        if len(native) != row["native_steps"] or len(expected) != row["steps"]:
            raise ValueError("Episode lengths changed")
    for camera in CAMERAS:
        entry = row["cameras"][camera]
        path = output / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Changed camera payload")
        with np.load(path, allow_pickle=False) as z:
            if z["images"].dtype != np.uint8 or z["images"].shape != (len(expected)+1, 180, 320, 3):
                raise ValueError("Camera image dtype/shape changed")
            if z["actions"].dtype != np.float32 or not np.array_equal(z["actions"], expected):
                raise ValueError("Five-command grouping changed")
            if not np.array_equal(z["frame_indices"], frame_indices):
                raise ValueError("Future target/frame alignment changed")


def extract_example(serialized, shard, episode_index, output, identity):
    import tensorflow as tf
    f = tf.train.Example.FromString(serialized).features.feature
    expected = {"episode_metadata/file_path", "episode_metadata/recording_folderpath"}
    scalar_names = ("is_first", "is_last", "is_terminal", "reward", "discount")
    languages = ("language_instruction", "language_instruction_2", "language_instruction_3")
    numeric_widths = {"action": 7,
        **{"action_dict/"+k: w for k,w in {"cartesian_position":6,"cartesian_velocity":6,"joint_position":7,"joint_velocity":7,"gripper_position":1,"gripper_velocity":1}.items()},
        **{"observation/"+k: w for k,w in {"cartesian_position":6,"joint_position":7,"gripper_position":1}.items()}}
    expected.update("steps/"+k for k in (*scalar_names, *languages, *numeric_widths))
    expected.update("steps/observation/"+k for k in CAMERAS)
    if set(f) != expected:
        raise ValueError(f"Unexpected RLDS fields: {set(f)^expected}")
    metadata = {}
    for name in ("file_path", "recording_folderpath"):
        values = f["episode_metadata/"+name].bytes_list.value
        if len(values) != 1:
            raise ValueError("Episode metadata must be scalar")
        metadata[name] = values[0].decode("utf-8")
    session = session_from_metadata(metadata["file_path"], metadata["recording_folderpath"])
    episode_id = "droid-" + hashlib.sha256(metadata["file_path"].encode()).hexdigest()[:24]
    output = Path(output)
    sidecar = output / "episodes" / (episode_id+".json")
    serialized_sha = hashlib.sha256(serialized).hexdigest()
    if sidecar.exists():
        row = json.loads(sidecar.read_text())
        if row["extraction_identity"] != identity or row["serialized_example_sha256"] != serialized_sha:
            raise ValueError("Existing episode belongs to a different extraction/source")
        verify_episode(row, output)
        return row
    count = len(f["steps/is_first"].int64_list.value)
    if count < 1:
        raise ValueError("Empty episode")
    arrays, text = {}, {}
    for name in scalar_names:
        if name.startswith("is_"):
            value = np.asarray(f["steps/"+name].int64_list.value, dtype=np.int64)
            if not np.isin(value, [0,1]).all():
                raise ValueError("Nonboolean RLDS flag")
            value = value.astype(bool)
        else:
            value = np.asarray(f["steps/"+name].float_list.value, dtype=np.float32)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError("Scalar native field length/nonfinite mismatch")
        arrays[name] = value
    if np.flatnonzero(arrays["is_first"]).tolist() != [0] or np.flatnonzero(arrays["is_last"]).tolist() != [count-1]:
        raise ValueError("Invalid RLDS episode boundary flags")
    if arrays["is_terminal"][:-1].any():
        raise ValueError("Terminal transition inside episode")
    for name,width in numeric_widths.items():
        value = np.asarray(f["steps/"+name].float_list.value, dtype=np.float64)
        if value.size != count*width or not np.isfinite(value).all():
            raise ValueError("Numeric native field shape/nonfinite mismatch: "+name)
        arrays[name.replace("/", "__")] = value.reshape(count,width)
    action = arrays["action"]
    known = np.concatenate([arrays["action_dict__cartesian_position"], arrays["action_dict__gripper_position"]], axis=1)
    if not np.array_equal(action, known):
        raise ValueError("Action semantics differ from verified Cartesian pose+gripper contract")
    for name in languages:
        values = f["steps/"+name].bytes_list.value
        if len(values) != count:
            raise ValueError("Language length mismatch")
        text[name] = [v.decode("utf-8") for v in values]
    actions, frame_indices = group_actions(action)
    arrays["selected_frame_indices"] = frame_indices
    arrays["omitted_tail_action_indices"] = np.arange(frame_indices[-1], count, dtype=np.int64)
    audit_file = Path("audit") / (episode_id+".npz")
    save_npz(output/audit_file, arrays)
    cameras = {}
    for camera in CAMERAS:
        encoded = f["steps/observation/"+camera].bytes_list.value
        if len(encoded) != count or any(not value.startswith(b"\xff\xd8") for value in encoded):
            raise ValueError("Camera does not supply one JPEG per native step")
        images = []
        for i in frame_indices:
            with Image.open(io.BytesIO(encoded[int(i)])) as im:
                if im.format != "JPEG" or im.mode != "RGB" or im.size != (320,180):
                    raise ValueError("Unexpected native JPEG format or size")
                images.append(np.asarray(im).copy())
        file = Path("cameras") / camera / (episode_id+".npz")
        save_npz(output/file, dict(images=np.stack(images), actions=actions, frame_indices=frame_indices))
        cameras[camera] = dict(file=str(file), sha256=sha256(output/file), frames=len(images), shape=[len(images),180,320,3])
    row = dict(episode_id=episode_id, session_id=session, split=split_for(session),
        steps=len(actions), native_steps=count, cameras=cameras,
        audit=dict(file=str(audit_file),sha256=sha256(output/audit_file)),
        metadata=metadata, language_instructions=text, action_semantics=ACTION_SEMANTICS,
        action_field_mapping="steps/action == concat(steps/action_dict/cartesian_position, steps/action_dict/gripper_position), exact equality over every step",
        source_shard=shard["filename"],source_shard_sha256=shard["sha256"],source_episode_index=episode_index,
        serialized_example_sha256=serialized_sha,extraction_identity=identity,
        omitted_tail_action_count=count-int(frame_indices[-1]),
        complete_eight_frame_windows=max(0,len(frame_indices)-7),
        temporal_contract="images at0,5,...; each35D action concatenates the5 commands from frame index t through t+4; no terminal action beyond the last observed target is used",
        timestamps_available=False,synthetic_corruptions=False,simulated_images=False,synthetic_actions=False,session_fallback_used=False)
    verify_episode(row, output)
    atomic_json(row, sidecar)
    return row


def process_shard(spec):
    os.environ.update(CUDA_VISIBLE_DEVICES="",TF_CPP_MIN_LOG_LEVEL="2",TF_NUM_INTRAOP_THREADS="1",TF_NUM_INTEROP_THREADS="1",OMP_NUM_THREADS="1")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    path = Path(spec["path"])
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError("Downloaded shard checksum changed: "+str(path))
    rows=[]
    # The compatibility iterator validates TFRecord CRCs and does not prefetch
    # multiple serialized episodes or build an unnecessary image-decoding graph.
    for index, serialized in enumerate(tf.compat.v1.io.tf_record_iterator(str(path))):
        if spec.get("limit") is not None and index >= spec["limit"]:
            break
        rows.append(extract_example(serialized,spec,index,spec["output"],spec["identity"]))
    if spec.get("limit") is None and len(rows) != spec["expected_episodes"]:
        raise ValueError("Shard record count differs from official dataset_info")
    return rows


def preview(output, row):
    import imageio.v2 as imageio
    root = Path(output)/"preview"
    root.mkdir(exist_ok=True)
    views = {}
    for camera in CAMERAS:
        with np.load(Path(output)/row["cameras"][camera]["file"],allow_pickle=False) as z:
            views[camera] = z["images"].copy()
            indices = z["frame_indices"].copy()
    video = root/"recorded_three_camera_episode.mp4"
    with imageio.get_writer(video,fps=6,codec="libx264",quality=8,macro_block_size=1) as writer:
        for t in range(len(indices)):
            writer.append_data(np.concatenate([views[c][t] for c in CAMERAS],axis=1))
    positions = np.unique(np.linspace(0,len(indices)-1,min(5,len(indices))).round().astype(int))
    sheet = Image.new("RGB",(len(positions)*320,3*210),(255,255,255))
    draw=ImageDraw.Draw(sheet)
    for r,camera in enumerate(CAMERAS):
        for col,t in enumerate(positions):
            sheet.paste(Image.fromarray(views[camera][t]),(col*320,r*210+30))
            draw.text((col*320+5,r*210+8),f"{camera} | native index {indices[t]}",fill=(20,20,20))
    sheet.save(root/"recorded_three_camera_contact_sheet.png")
    atomic_json(dict(episode_id=row["episode_id"],session_id=row["session_id"],split=row["split"],
        source_camera_payloads=row["cameras"],frame_indices=indices.tolist(),
        video_sha256=sha256(video),contact_sheet_sha256=sha256(root/"recorded_three_camera_contact_sheet.png"),
        playback_frames_per_second=6,clock_claim="Chosen display playback only; source per-frame timestamps unavailable. Not a real-time speed claim.",
        footage="Actual released face-blurred robot-camera RGB; no generated imagery or artificial shifts.",
        camera_order=list(CAMERAS)),root/"provenance.json")


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--expected-episodes",type=int,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--schema-audit",action="store_true",help="Inspect one episode only; never emit a trainable complete manifest")
    args=p.parse_args()
    started=time.monotonic()
    raw=args.raw.resolve();directory=raw/"1.0.0" if (raw/"1.0.0").is_dir() else raw
    receipt_path=raw/"download_receipt.json" if (raw/"download_receipt.json").exists() else raw.parent/"download_receipt.json"
    receipt=json.loads(receipt_path.read_text());inventory=json.loads(args.inventory.read_text())
    if receipt["status"]!="verified" or receipt["inventory_sha256"]!=sha256(args.inventory):
        raise ValueError("Missing/mismatched complete download receipt")
    if inventory.get("selection",{}).get("expected_episodes",args.expected_episodes)!=args.expected_episodes:
        raise ValueError("Prespecified inventory episode count changed")
    sources={Path(r["local_path"]).name:r for r in receipt["files"]}
    for name in ("dataset_info.json","features.json"):
        if sha256(directory/name)!=sources[name]["sha256"]:
            raise ValueError("RLDS schema metadata checksum mismatch")
    info=json.loads((directory/"dataset_info.json").read_text())
    shard_lengths=next(s for s in info["splits"] if s["name"]=="train")["shardLengths"]
    filenames=sorted(n for n in sources if ".tfrecord-" in n)
    expected=sum(int(shard_lengths[int(re.search(r'tfrecord-(\d+)-of-',n).group(1))]) for n in filenames)
    if expected!=args.expected_episodes:
        raise ValueError("Selected official shard lengths differ from expected episode count")
    identity=dict(extractor_sha256=sha256(__file__),requirements_sha256=sha256(ROOT/"environments/real_video/requirements.lock.txt"),
        receipt_sha256=sha256(receipt_path),inventory_sha256=sha256(args.inventory),
        features_sha256=sha256(directory/"features.json"),dataset_info_sha256=sha256(directory/"dataset_info.json"),
        split_salt=SALT,split_probabilities=[.70,.15,.15],action_block=5,action_semantics=ACTION_SEMANTICS,
        schema_audit_only=args.schema_audit,primary_camera=CAMERAS[0],source_scope=inventory.get("selection","author100-episode debugging subset"))
    args.output.mkdir(parents=True,exist_ok=True)
    identity_path=args.output/"extraction_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError("Extraction identity changed; preserve existing artifacts and use a new output directory")
    atomic_json(identity,identity_path)
    specs=[]
    for name in filenames[:1] if args.schema_audit else filenames:
        source=sources[name]
        specs.append(dict(path=str(directory/name),filename=name,bytes=source["bytes"],sha256=source["sha256"],
            output=str(args.output.resolve()),identity=identity,limit=1 if args.schema_audit else None,
            expected_episodes=int(shard_lengths[int(re.search(r'tfrecord-(\d+)-of-',name).group(1))])))
    rows=[]
    if args.schema_audit or args.workers==1:
        for spec in specs:rows.extend(process_shard(spec))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            futures={pool.submit(process_shard,s):s for s in specs}
            for future in as_completed(futures):
                added=future.result();rows.extend(added)
                print(json.dumps(dict(shard=futures[future]["filename"],episodes=len(added),completed_episodes=len(rows))),flush=True)
    rows.sort(key=lambda r:r["episode_id"])
    if len({r["episode_id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate original episode identity among selected shards")
    if not args.schema_audit and len(rows)!=args.expected_episodes:
        raise ValueError("Incomplete extraction")
    sessions={}
    for row in rows:
        if row["session_id"] in sessions and sessions[row["session_id"]]!=row["split"]:
            raise ValueError("Session leakage")
        sessions[row["session_id"]]=row["split"]
    if not args.schema_audit and set(sessions.values())!={"train","val","test"}:
        raise ValueError("Missing a complete session-disjoint split")
    manifest=dict(status="schema_audit_only" if args.schema_audit else "complete",schema_version=1,
        dataset="DROID recorded real robot video",primary_camera=CAMERAS[0],cameras=list(CAMERAS),
        action_block=5,action_dim=35,native_action_dim=7,action_semantics=ACTION_SEMANTICS,image_shape=[180,320,3],
        synthetic_corruptions=False,simulated_images=False,synthetic_actions=False,session_fallback_used=False,
        timestamps_available=False,source_identity=identity,source_files=sources,episodes=rows,
        expected_source_episodes=args.expected_episodes,extracted_episodes=len(rows),
        native_steps=sum(r["native_steps"] for r in rows),grouped_steps=sum(r["steps"] for r in rows),
        split_counts=dict(Counter(r["split"] for r in rows)),session_split_counts=dict(Counter(sessions.values())),
        session_split_mapping=sessions,source_record_float_storage="TF Example float_list; decoded to declared float64 audit arrays; model actions stored float32",
        caution="A deterministically selected real DROID subset, not the complete benchmark. No nominal frame clock is inferred.")
    atomic_json(manifest,args.output/"manifest.json")
    preview(args.output,rows[0])
    audit=dict(status="schema_audit_passed" if args.schema_audit else "passed",dataset_manifest_sha256=sha256(args.output/"manifest.json"),
        episodes=len(rows),expected_source_episodes=args.expected_episodes,native_steps=manifest["native_steps"],grouped_steps=manifest["grouped_steps"],
        split_counts=manifest["split_counts"],session_split_counts=manifest["session_split_counts"],sessions_disjoint=True,
        exact_action_field_equality_checked_all_native_steps=True,action_semantics=ACTION_SEMANTICS,
        model_action_target_alignment_checked=True,all_exported_payload_hashes_verified=True,
        raw_shards_verified=len(specs),all_raw_shards_verified=not args.schema_audit,
        all_exported_cameras_decoded_at_native_resolution=True,rl_ds_episode_boundaries_checked=True,
        short_episodes_without_eight_frame_window=sum(r["complete_eight_frame_windows"]==0 for r in rows),
        omitted_tail_action_count=sum(r["omitted_tail_action_count"] for r in rows),
        skipped_due_to_outcome_or_motion=0,synthetic_corruptions=False,generated_frames=False,
        seconds=time.monotonic()-started,source_identity=identity,
        timestamp_statement="Released feature schema contains no per-step timestamp; frame_indices represent ordinal native observations only.")
    atomic_json(audit,args.output/"data_audit.json")
    print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":main()
