#!/usr/bin/env python3
"""Download one pinned public Open-H endoscopy video/actions sample and audit it.

No authentication, model training, or synthetic video. Original MP4/Parquet are
preserved, and derived JPEG frames are explicitly inspection artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
REPO = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
SUBSET = "Endoscopy/cuhk/openh_dataset_full/find_greater_curvature"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def get_json(session, url, **params):
    response = session.get(url, params=params, timeout=(15, 90))
    response.raise_for_status()
    return response.json(), response


def fetch(session, revision, entry, output):
    relative = entry["path"][len(SUBSET) + 1:]
    local = output / "original" / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/" + quote(entry["path"], safe="/")
    start = time.perf_counter()
    if not local.exists():
        response = session.get(url, stream=True, timeout=(15, 120))
        response.raise_for_status()
        temporary = local.with_suffix(local.suffix + ".partial")
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
        temporary.replace(local)
    if local.stat().st_size != entry["size"]:
        raise ValueError(f"Downloaded length mismatch {relative}")
    checksum = digest(local)
    if entry.get("lfs", {}).get("oid") and checksum != entry["lfs"]["oid"]:
        raise ValueError(f"Remote LFS SHA256 mismatch {relative}")
    return {"path": str(local.relative_to(output)), "source_url": url, "bytes": local.stat().st_size,
            "sha256": checksum, "remote_lfs_sha256": entry.get("lfs", {}).get("oid"),
            "download_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/real_video/openh_sample_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/real_openh_video_audit.json")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    # Bypass ~/.netrc and environment tokens: these URLs are genuinely public.
    session.trust_env = False
    start = time.perf_counter()
    metadata, _ = get_json(session, f"https://huggingface.co/api/datasets/{REPO}",
                          **{"expand[]": ["sha", "gated", "private", "cardData"]})
    if metadata["private"] or metadata["gated"]:
        raise ValueError("Expected public ungated source; no authentication will be attempted")
    revision = metadata["sha"]
    pinned_file = output / "source_identity.json"
    if pinned_file.exists():
        previous = json.loads(pinned_file.read_text())
        if previous["revision"] != revision:
            raise ValueError("Remote main changed; use a new output directory for a new revision")
    identity = {"repository": REPO, "subset": SUBSET, "revision": revision,
                "metadata": metadata, "no_credentials_used": True}
    save_json(pinned_file, identity)
    tree = []
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/" + quote(SUBSET, safe="/")
    params = {"recursive": "true", "expand": "false", "limit": 1000}
    while url:
        entries, response = get_json(session, url, **params)
        tree.extend(entries)
        url = response.links.get("next", {}).get("url")
        params = {}
    files = [row for row in tree if row["type"] == "file"]
    save_json(output / "remote_inventory.json", files)
    chosen = [row for row in files if row["path"].endswith("/README.md") or "/meta/" in row["path"]]
    parquet = sorted(row["path"] for row in files if row["path"].endswith(".parquet"))[0]
    episode_name = Path(parquet).stem
    videos = [row for row in files if row["path"].endswith("/" + episode_name + ".mp4")]
    chosen += [row for row in files if row["path"] == parquet] + videos
    downloads = []
    for entry in chosen:
        record = fetch(session, revision, entry, output)
        downloads.append(record)
        print(json.dumps({"downloaded": record["path"], "bytes": record["bytes"]}), flush=True)
    save_json(output / "downloads.json", downloads)
    info = json.loads((output / "original/meta/info.json").read_text())
    parquet_path = output / "original" / parquet[len(SUBSET) + 1:]
    episodes = [json.loads(line) for line in (output / "original/meta/episodes.jsonl").read_text().splitlines()]
    if len(episodes) != info["total_episodes"] or sum(row["length"] for row in episodes) != info["total_frames"]:
        raise ValueError("Episode metadata totals disagree with info.json")
    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    timestamps = np.asarray(data["timestamp"], np.float64)
    frame_indices = np.asarray(data["frame_index"]).reshape(-1)
    actions = np.asarray(data["action"], np.float64)
    states = np.asarray(data["observation.state"], np.float64)
    if not all(np.isfinite(values).all() for values in (timestamps, actions, states)):
        raise ValueError("Nonfinite sample values")
    expected_time = np.arange(len(timestamps)) / float(info["fps"])
    probes = []
    for entry in videos:
        path = output / "original" / entry["path"][len(SUBSET) + 1:]
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
                   "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:frame=best_effort_timestamp_time,pkt_duration_time",
                   "-of", "json", str(path)]
        probe = json.loads(subprocess.check_output(command, text=True))
        save_json(output / "derived/ffprobe.json", probe)
        stream = probe["streams"][0]
        presentation_times = np.asarray([float(row["best_effort_timestamp_time"]) for row in probe["frames"]])
        video_name = path.parent.name
        decode_start = time.perf_counter()
        raw_rgb = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
            "-vf", "scale=128:128", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        decode_seconds = time.perf_counter() - decode_start
        decoded = np.frombuffer(raw_rgb, np.uint8).reshape(-1, 128, 128, 3)
        if len(decoded) != len(presentation_times):
            raise ValueError("Full RGB decoding and ffprobe frame counts disagree")
        derived = output / "derived"
        derived.mkdir(parents=True, exist_ok=True)
        preview = derived / (episode_name + "_original.mp4")
        if not preview.exists():
            preview.symlink_to(Path("../original") / "videos/chunk-000" / video_name / path.name)
        if digest(preview) != digest(path):
            raise ValueError("Original-video convenience link changed bytes")
        thumbs = output / "derived/frames"
        thumbs.mkdir(parents=True, exist_ok=True)
        indexes = np.unique(np.rint(np.linspace(0, len(presentation_times) - 1, 8)).astype(int))
        # Decode actual original MP4 frames. No inferred/generated frame is used.
        select = "+".join(f"eq(n,{int(index)})" for index in indexes)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1", "-i", str(path),
                        "-vf", f"select='{select}',scale=320:240", "-vsync", "0", "-q:v", "2",
                        str(thumbs / "frame_%02d.jpg")], check=True)
        canvas = Image.new("RGB", (4 * 328, 2 * 276 + 55), (248, 248, 246))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), "Open-H / CUHK: real captured stomach-phantom endoscopy, episode 0", fill=(20, 20, 20))
        draw.text((12, 28), "Actual source video frames; physical phantom, not patient surgery or generated imagery", fill=(65, 65, 65))
        for slot, (frame, native_index) in enumerate(zip(sorted(thumbs.glob("*.jpg")), indexes)):
            x, y = (slot % 4) * 328, 55 + (slot // 4) * 276
            canvas.paste(Image.open(frame), (x, y))
            label = f"frame {native_index} | {presentation_times[native_index]:.2f}s"
            if native_index < len(actions):
                label += f" | a=({actions[native_index, 0]:.1f}, {actions[native_index, 1]:.1f})"
            draw.text((x + 3, y + 244), label, fill=(25, 25, 25))
        contact = output / "derived/episode_000000_contact_sheet.jpg"
        canvas.save(contact, quality=93)
        aligned_length = min(len(timestamps), len(presentation_times))
        probes.append({"path": str(path), "camera_key": video_name, "stream": stream,
                       "decoded_frames": len(presentation_times), "first_pts_s": float(presentation_times[0]),
                       "last_pts_s": float(presentation_times[-1]),
                       "equal_frame_and_row_count": len(presentation_times) == len(timestamps),
                       "max_abs_pts_minus_parquet_timestamp_s": float(np.max(np.abs(presentation_times[:aligned_length] - timestamps[:aligned_length]))),
                       "contact_sheet": str(contact), "contact_sheet_sha256": digest(contact),
                       "selected_frame_indices": indexes.tolist(),
                       "original_video_convenience_link": str(preview),
                       "full_decode_128": {"frames": len(decoded), "seconds": decode_seconds,
                         "fps_including_ffmpeg_startup": len(decoded) / decode_seconds,
                         "raw_rgb_sha256": hashlib.sha256(raw_rgb).hexdigest(),
                         "mean_consecutive_rgb_absolute_difference": float(np.abs(np.diff(decoded.astype(np.float32), axis=0)).mean())},
                       "caution": "PTS/row timestamp consistency does not prove sensor acquisition synchronization or causal actuation latency"})
    kind_counts = Counter(Path(row["path"]).suffix for row in files)
    kind_bytes = {suffix: sum(row["size"] for row in files if Path(row["path"]).suffix == suffix) for suffix in kind_counts}
    report = {"status": "public_sample_downloaded_and_decoded_no_training", "source": identity,
              "source_urls": {"dataset_card": f"https://huggingface.co/datasets/{REPO}",
                              "subset_readme": f"https://huggingface.co/datasets/{REPO}/blob/{revision}/{SUBSET}/README.md"},
              "subset_inventory": {"total_bytes": sum(row["size"] for row in files), "files": len(files),
                                   "by_suffix_count": dict(kind_counts), "by_suffix_bytes": kind_bytes,
                                   "sha256": digest(output / "remote_inventory.json")},
              "downloads": downloads, "downloaded_bytes": sum(row["bytes"] for row in downloads),
              "info": info, "episode_metadata": {"episode_count": len(episodes),
                 "frame_count_sum": sum(row["length"] for row in episodes),
                 "episode_length_range": [min(row["length"] for row in episodes), max(row["length"] for row in episodes)],
                 "fields": list(episodes[0]), "operator_or_session_identifier_found_in_inspected_metadata": False}, "sample": {"episode": episode_name, "rows": len(timestamps),
                  "parquet_schema": str(table.schema), "columns": table.column_names,
                  "episode_index_unique": np.unique(np.asarray(data["episode_index"]).reshape(-1)).tolist(),
                  "task_index_unique": np.unique(np.asarray(data["task_index"]).reshape(-1)).tolist(),
                  "language_instruction_unique": np.unique(np.asarray(data["language_instruction"]).reshape(-1)).tolist(),
                  "schema_note": "Episode/frame/index/task IDs are Arrow fixed_size_list[1]; flattened only for index verification; timestamps are scalar float32",
                  "frame_index_is_contiguous_from_zero": bool(np.array_equal(frame_indices, np.arange(len(timestamps)))),
                  "timestamp_start_s": float(timestamps[0]), "timestamp_end_s": float(timestamps[-1]),
                  "timestamp_diff_min_max_s": [float(np.diff(timestamps).min()), float(np.diff(timestamps).max())],
                  "max_abs_timestamp_vs_frame_over_fps_s": float(np.abs(timestamps - expected_time).max()),
                  "action_shape": list(actions.shape), "action_min": actions.min(0).tolist(), "action_max": actions.max(0).tolist(),
                  "nonzero_action_rows": int(np.any(actions != 0, axis=1).sum()),
                  "action_unique_values": [np.unique(actions[:, index]).tolist() for index in range(actions.shape[1])],
                  "light_val_unique_values": np.unique(states[:, -1]).tolist(),
                  "consecutive_action_changes": int(np.any(np.diff(actions, axis=0) != 0, axis=1).sum()),
                  "state_shape": list(states.shape), "state_min": states.min(0).tolist(), "state_max": states.max(0).tolist(),
                  "all_values_finite": True, "videos": probes},
              "license": metadata["cardData"].get("license"), "elapsed_seconds": time.perf_counter() - start,
              "collector_script_sha256": digest(Path(__file__)),
              "runtime_lock_sha256": digest(ROOT / "environments/real_video/requirements.lock.txt"),
              "interpretation": "Real captured physical-phantom endoscopy, not patient surgery; offline action/video sample, not online control evidence",
              "open_questions": ["README acquisition/synchronization limitations must be reviewed before a dynamics claim",
                                 "Episode/session/operator grouping for held-out splits",
                                 "Action units, normalization, timing convention and physical motor response delay",
                                 "One episode does not establish whole-subset integrity or domain-generalization performance"]}
    save_json(args.report, report)
    print(json.dumps({"report": str(args.report), "subset_bytes": report["subset_inventory"]["total_bytes"],
                      "downloaded_bytes": report["downloaded_bytes"], "sample_rows": len(timestamps), "videos": probes}, indent=2))


if __name__ == "__main__":
    main()
