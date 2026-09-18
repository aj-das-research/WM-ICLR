#!/usr/bin/env python3
"""Metadata-only audit of pinned IWS timing/action/index contracts.

No video library, neural model, image decoder, or validation command values are
used. Three upstream indexing methods execute on a recording stub rather than a
real dataset, allowing exact index verification without opening observations.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external/rla-wm"
DATA = ROOT / "data/real_video/iws_public_v1"
PINNED = "6f19048758699bf9a152eaed5ac6dbf1caa07c18"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def index_stub():
    source = UPSTREAM / "src/datasets/trajectory_dataset.py"
    module = ast.parse(source.read_text())
    cls = next(item for item in module.body if isinstance(item, ast.ClassDef) and item.name == "TrajectoryDataset")
    names = {"_compute_sampled_frame_indices", "_feasible_horizon_for_start", "frame_index_to_trajectory_data"}
    methods = [item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name in names]
    if len(methods) != len(names):
        raise ValueError("Pinned indexing method structure changed")
    stub_class = ast.ClassDef(name="IndexOnly", bases=[], keywords=[], body=methods, decorator_list=[])
    header = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[header, stub_class], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(tree, str(source), "exec"), namespace)
    instance = namespace["IndexOnly"]()
    calls = []

    def read_trajectory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(video_streams={}, metadata={})

    instance.datasets = [SimpleNamespace(read_trajectory=read_trajectory)]
    instance.num_frames = 2
    instance.horizon = 15
    instance.dataset_horizon_strides = [1]
    instance.interaction_only = False
    instance._get_rgb_mode_for_dataset = lambda unused: "base"
    instance.optional_video_suffixes = []
    instance.read_eef_pose = False
    instance.read_object_poses = False
    instance.cfg = {"img_size": 512}
    instance.bg_mask_top = 0.0
    return instance, calls


def main():
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED or subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip():
        raise ValueError("Require clean pinned upstream checkout")
    acquisition_path = ROOT / "reports/evidence/iws_acquisition.json"
    registry_path = ROOT / "configs/real_video_development/iws_acquisition_v1.json"
    acquisition = json.loads(acquisition_path.read_text())
    if (acquisition["status"] != "download_verified_and_extracted_no_evaluation"
            or acquisition["registry_sha256"] != sha(registry_path)):
        raise ValueError("Require completed immutable acquisition")
    source_paths = [Path(__file__), acquisition_path, registry_path]
    source_paths += [UPSTREAM / name for name in (
        "src/datasets/trajectory_dataset.py", "datalib/dataset.py",
        "src/trainers/rla_wm_trainer.py", "src/models/rla_wm.py",
        "eval/predictors/rla_wm_predictor_iws.py", "eval/run_eval_iws.sh",
        "eval/eval_wrapper.py", "configs/rla_wm/iws_pusht.yaml",
        "configs/rla_wm/iws_box.yaml", "configs/rla_wm/iws_rope.yaml")]
    frozen_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    base = DATA / "extracted/iws_converted"
    summary = json.loads((base / "summary.json").read_text())
    source_paths += [base / "summary.json"]
    episodes, tasks, populations, frame_counts = [], [], {}, {}
    summary_discrepancies = []
    timestamp_names = []
    expected_keys = {"camera_0_extrinsics", "camera_0_intrinsics", "qpos", "root_poses", "target_qpos"}
    for task, conversion in sorted(summary["subdatasets"].items()):
        ranges = json.loads((base / task / "split_ranges.json").read_text())
        index = json.loads((base / task / "index.json").read_text())["trajectories"]
        source_paths += [base / task / "split_ranges.json", base / task / "index.json"]
        split_ids = {split: set(range(value["start"], value["end"] + 1)) for split, value in ranges.items()}
        if set(split_ids) != {"train", "val"} or split_ids["train"] & split_ids["val"]:
            raise ValueError("Invalid original split")
        if {int(key) for key in index} != split_ids["train"] | split_ids["val"]:
            raise ValueError("Split/index population mismatch")
        populations[task] = split_ids
        task_rows = []
        for eid in sorted(index):
            split = "train" if int(eid) in split_ids["train"] else "val"
            path = base / task / ("traj_" + eid) / "metadata.h5"
            with h5py.File(path, "r") as metadata:
                if set(metadata) != expected_keys:
                    raise ValueError("Unexpected metadata keys: " + str(path))
                keys = list(metadata) + list(metadata.attrs)
                timestamp_names.extend([str(path.relative_to(ROOT)) + ":" + key for key in keys
                                        if any(token in key.lower() for token in ("time", "fps", "rate", "stamp"))])
                shapes = {key: list(metadata[key].shape) for key in sorted(metadata)}
                dtypes = {key: str(metadata[key].dtype) for key in sorted(metadata)}
                count = shapes["target_qpos"][0]
                if (any(shape[0] != count for shape in shapes.values())
                        or len(shapes["qpos"]) != 2 or len(shapes["target_qpos"]) != 2
                        or shapes["target_qpos"][1] not in conversion[split]["action_dims"]
                        or shapes["qpos"][1] not in conversion[split]["joint_pos_dims"]):
                    raise ValueError("Metadata alignment/dimensions differ: " + str(path))
                # Validation payload VALUES are not read; shapes/names suffice
                # to verify the official handle bounds. No success values read.
                if split == "train":
                    for key in sorted(metadata):
                        if not np.isfinite(metadata[key][...]).all():
                            raise ValueError("Nonfinite training metadata: " + str(path) + ":" + key)
                row = {"task": task, "episode_id": eid, "split": split,
                       "file": str(path.relative_to(ROOT)), "sha256": sha(path),
                       "shapes": shapes, "dtypes": dtypes, "attribute_names": sorted(metadata.attrs),
                       "all_metadata_finite": True if split == "train" else "values_not_read"}
            episodes.append(row)
            task_rows.append(row)
            frame_counts[(task, eid)] = count
        task_summary = {"task": task, "splits": {}}
        for split in ("train", "val"):
            rows = [row for row in task_rows if row["split"] == split]
            counts = [row["shapes"]["target_qpos"][0] for row in rows]
            published = conversion[split]
            if len(rows) != published["converted"] or published["failed"] != 0:
                raise ValueError("Published conversion population mismatch")
            if (min(counts) != published["frame_count_min"] or max(counts) != published["frame_count_max"]
                    or not np.isclose(np.mean(counts), published["frame_count_mean"], rtol=0, atol=1e-10)):
                summary_discrepancies.append({"task": task, "split": split,
                    "actual_metadata_frame_count_min_max_mean": [min(counts), max(counts), float(np.mean(counts))],
                    "published_frame_count_min_max_mean": [published["frame_count_min"], published["frame_count_max"], published["frame_count_mean"]],
                    "interpretation": "Published summary differs from downloaded metadata. No records removed or changed."})
            task_summary["splits"][split] = {"episodes": len(rows), "stored_frames": sum(counts),
                "frame_counts": dict(sorted(Counter(counts).items())),
                "command_dimensions": sorted({r["shapes"]["target_qpos"][1] for r in rows}),
                "qpos_dimensions": sorted({r["shapes"]["qpos"][1] for r in rows}),
                "finite_metadata_checked": split == "train"}
        tasks.append(task_summary)
        print(json.dumps({"task": task, "metadata_episodes_checked": len(task_rows)}), flush=True)

    stub, calls = index_stub()
    handles_report = []
    video_headers = {}
    names = {"pusht": "pusht", "box": "bimanual_box", "rope": "bimanual_rope"}
    for scene, task in sorted(names.items()):
        config_path = UPSTREAM / f"configs/rla_wm/iws_{scene}.yaml"
        config = yaml.safe_load(config_path.read_text())
        if (config["vars"]["num_frames"] != 2 or config["vars"]["horizon"] != 15
                or config["trainer"]["args"]["directly_use_target_qpos"] is not True
                or config["dataset"]["args"]["configs"][task].get("horizon_stride", 1) != 1
                or config["val_dataset"]["args"]["configs"][task].get("horizon_stride", 1) != 1):
            raise ValueError("Unexpected IWS training/evaluation temporal configuration")
        path = DATA / f"download/eval_handles/iws/handles.{scene}.json"
        source_paths.append(path)
        handle_file = json.loads(path.read_text())
        if set(handle_file["handles_by_horizon"]) != {"60"}:
            raise ValueError("Unexpected official handle horizon")
        handles = handle_file["handles_by_horizon"]["60"]
        keys, starts, ends, tails, feasibility = [], [], [], [], []
        for row in handles:
            task_name, eid, start = row["group_name"], row["traj_id"], row["frame_id"]
            metadata_count = frame_counts[(task_name, eid)]
            headers = []
            for suffix in ("rgb", "foreground_mask"):
                video_path = base / task_name / ("traj_" + eid) / ("camera_0_" + suffix + ".mp4")
                relative = str(video_path.relative_to(ROOT))
                if relative not in video_headers:
                    # Container metadata only: no -count_frames/-show_frames,
                    # no decoder or image output is requested.
                    header = json.loads(subprocess.check_output([
                        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_frames,r_frame_rate,avg_frame_rate,width,height,duration",
                        "-of", "json", str(video_path)], text=True))["streams"]
                    if len(header) != 1 or not header[0].get("nb_frames", "").isdigit():
                        raise ValueError("Container has no authoritative sample count: " + relative)
                    video_headers[relative] = header[0]
                headers.append(video_headers[relative])
            count = int(headers[0]["nb_frames"])
            if any(int(header["nb_frames"]) != count for header in headers) or metadata_count != count:
                raise ValueError("Official validation video/metadata lengths disagree: " + str((task_name, eid)))
            if (task_name != task or int(eid) not in populations[task]["val"] or type(start) is not int
                    or start < 0 or row["sampled_horizon"] != 60 or row["dataset_idx"] != 0
                    or row["camera_keys"] != ["camera_0"] or row["interaction_frame_indices"] is not None
                    or row["rgb_variant"] != "base" or row["dataset_root"] != "data/iws_converted/" + task):
                raise ValueError("Unexpected official handle identity")
            feasible = stub._feasible_horizon_for_start({"dataset_idx": 0, "num_frames": count}, start)
            if row["feasible_horizon"] != feasible or 60 > feasible:
                raise ValueError("Official feasible horizon disagrees with pinned indexing code")
            _, video_indices = stub.frame_index_to_trajectory_data(row)
            metadata_indices = calls[-1]["metadata_frame_indices"]
            if video_indices != [start, start + 59] or metadata_indices != list(range(start, start + 60)):
                raise ValueError("Pinned indexing code differs from inclusive horizon contract")
            if not all(0 <= idx < count - 1 for idx in metadata_indices):
                raise ValueError("Official handle violates metadata bound/tail reservation")
            keys.append((eid, start)); starts.append(start); ends.append(start + 59)
            tails.append(count - (start + 60)); feasibility.append(feasible)
        handles_report.append({"scene": scene, "file": str(path.relative_to(ROOT)), "sha256": sha(path),
            "windows": len(handles), "unique_windows": len(set(keys)), "episodes": len({eid for eid, _ in keys}),
            "start_min_max": [min(starts), max(starts)], "endpoint_min_max": [min(ends), max(ends)],
            "reserved_tail_min_max": [min(tails), max(tails)],
            "all_bounds_passed": True, "all_legacy_feasible_horizons_match": True,
            "video_indices": "[start, start + 59]", "metadata_slots": 60,
            "config_path_stored_in_handle": handle_file["meta"]["config_path"],
            "stored_config_exists_in_pinned_repo": (UPSTREAM / handle_file["meta"]["config_path"]).exists()})

    if stub._compute_sampled_frame_indices(2,15).tolist() != [0,14]:
        raise ValueError("Unexpected upstream training endpoint convention")
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    for name, expected in frozen_hashes.items():
        if source_hashes[name] != expected:
            raise ValueError("Input changed while auditing: " + name)
    manifest_path = ROOT / "reports/evidence/iws_temporal_metadata_manifest.json"
    write_json(manifest_path, {"status": "metadata_only_audit_passed", "episodes": episodes})
    report = {"status": "metadata_and_index_semantics_verified_no_model_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(), "upstream_commit": PINNED,
        "source_sha256": source_hashes, "tasks": tasks, "official_handles": handles_report,
        "total_train_episodes": sum(row["splits"]["train"]["episodes"] for row in tasks),
        "total_val_metadata_shapes": sum(row["splits"]["val"]["episodes"] for row in tasks),
        "training_metadata_all_finite": True, "timestamp_or_rate_metadata_names": timestamp_names,
        "published_summary_discrepancies": summary_discrepancies,
        "official_validation_video_container_headers": video_headers,
        "images_decoded": 0, "validation_command_values_read": 0, "models_loaded_or_evaluated": 0,
        "metadata_manifest": str(manifest_path.relative_to(ROOT)), "metadata_manifest_sha256": sha(manifest_path),
        "indexing_trace": {"training_horizon_15_rgb_offsets": [0,14], "training_command_offsets_inclusive": [0,14],
            "official_horizon_60_rgb_offsets": [0,59], "official_command_offsets_inclusive": [0,59],
            "official_rollout_command_chunks_inclusive": [[0,14],[15,29],[30,44],[45,59]],
            "nominal_sum_of_four_training_endpoint_spans": 56, "official_endpoint_span": 59,
            "interpretation": "Chunking arithmetic does not establish physical timestamps. Preserve official evaluator; report endpoint/command convention and unresolved boundary alignment."},
        "unresolved": ["No physical timestamps, command units, or conversion frame-rate mapping established.",
            "Published conversion frame-count summary differs from downloaded metadata for some splits; all discrepancies are retained.",
            "No observation/command lead-lag alignment established from metadata alone.",
            "Four disjoint15-slot chunks do not overlap the endpoint conditioning row used by the preceding trained14-interval map.",
            "Handle-stored historical configuration paths are absent from this pinned repo.",
            "No session/person/scene independence established; official trajectory split only."]}
    output = ROOT / "reports/evidence/iws_temporal_semantics_audit.json"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "train_episodes": report["total_train_episodes"],
        "official_handles": sum(row["windows"] for row in handles_report), "report": str(output.relative_to(ROOT))}), flush=True)


if __name__ == "__main__":
    main()
