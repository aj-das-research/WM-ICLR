#!/usr/bin/env python3
"""Register native-row Box/Rope cache inputs, then build only that reviewed identity."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws_tasks.data import (InternalInventory, SPLITS, SPLIT_RELATIVE, METADATA_RELATIVE,
                                       canonical_hash, read_json, require, sha, task_width, TASK_WIDTHS)
from shiftwm.real_video_iws.features import PREPROCESSING, DinoSpatialEncoder, verify_encoder
from shiftwm.real_video_iws_tasks.cache import atomic_json, build_cache, IWSFeatureCache

SOURCES = [
    "src/shiftwm/real_video_iws_tasks/__init__.py", "src/shiftwm/real_video_iws_tasks/data.py",
    "src/shiftwm/real_video_iws_tasks/cache.py", "scripts/real_video_iws_tasks/prepare_cache.py",
    "scripts/real_video_iws_tasks/cache_gpu.slurm", "scripts/real_video_iws_tasks/inspect_training.py",
    "tests/test_real_video_iws_tasks.py", "reports/real_video_development/iws_box_rope_cache_protocol.md",
    "src/shiftwm/real_video_iws/__init__.py", "src/shiftwm/real_video_iws/data.py",
    "src/shiftwm/real_video_iws/features.py", "src/shiftwm/real_video_iws/cache.py",
    "scripts/real_video_iws/prepare_cache.py", "scripts/real_video_iws/prepare_split.py",
    "src/shiftwm/real_video_spatial/features.py",
]


def task_paths(task):
    task_width(task)
    return (f"configs/real_video_iws/{task}_cache_v1.json",
            f"reports/evidence/iws_{task}_cache_inputs_v1.json",
            f"configs/real_video_iws/{task}_cache_registration_v1.json")


def immutable_json(value, path):
    path = Path(path)
    if path.exists():
        require(read_json(path) == value, "Refusing to overwrite a different registration/input manifest")
    else:
        atomic_json(value, path)


def validate_config(config, inventory):
    require(config.get("schema") == "shiftwm_iws_task_cache_config_v1", "Wrong cache config")
    require(config["task"] == inventory.task and config["command_width"] == inventory.command_width and config["splits"] == list(SPLITS), "This preparation uses exact Box/Rope internal partitions only")
    require(config["expected_episodes"] == {"internal_train": 481, "internal_development": 121}, "Population configuration changed")
    require(config["split_sha256"] == inventory.split_sha256, "Frozen split identity changed")
    require(config["preprocessing"] == PREPROCESSING, "Preprocessing changed")
    require(config["native_frame_policy"] == "all sequential stored frames; no resampling or row removal", "Frame policy changed")
    require(config["command_policy"] == f"all target_qpos rows; native N rows x{inventory.command_width} coordinates", "Command policy changed")
    require(config["batch_size"] == 32, "Encoder batch size changed")
    require(config["minimum_free_cuda_bytes"] == 4 * 1024**3, "GPU memory preflight policy changed")
    for split in SPLITS:
        require(len(inventory.partitions[split]) == config["expected_episodes"][split], "Internal population changed")


def prepare(task, root=ROOT):
    """Hash allowed source bytes only; no RGB decoding, HDF5 values, or model load."""
    CONFIG, INPUTS, REGISTRATION = task_paths(task)
    root = Path(root); inventory = InternalInventory(root, task); config = read_json(root / CONFIG)
    validate_config(config, inventory)
    require(sha(root / METADATA_RELATIVE) == inventory.document["source_files"][str(METADATA_RELATIVE)], "Metadata audit changed")
    require(sha(root / "scripts/real_video_iws/prepare_split.py") == inventory.document["source_sha256"], "Frozen split preparer changed")
    encoder_root = root / config["encoder_root"]
    encoder = verify_encoder(root, encoder_root)
    records = []
    for eid in inventory.selected():
        row, split, metadata, video = inventory.paths(eid)
        require(sha(metadata) == row["sha256"], "Training metadata differs from audited source")
        records.append({"task": task, "command_width": inventory.command_width, "episode_id": eid, "split": split, "frames": row["shapes"]["target_qpos"][0],
                        "metadata_path": str(metadata.relative_to(root)), "metadata_sha256": row["sha256"],
                        "video_path": str(video.relative_to(root)), "video_sha256": sha(video), "video_bytes": video.stat().st_size})
    inputs = {"schema": "shiftwm_iws_task_cache_inputs_v1", "task": task, "status": "allowed_input_bytes_verified_no_video_decoding",
              "split_sha256": inventory.split_sha256, "metadata_audit_sha256": sha(root / METADATA_RELATIVE),
              "dataset_revision": inventory.document["dataset_revision"], "upstream_code_revision": inventory.document["upstream_code_revision"],
              "episodes": records, "counts": {s: sum(r["split"] == s for r in records) for s in SPLITS},
              "native_frame_rows": sum(r["frames"] for r in records), "command_width": inventory.command_width,
              "official_validation_payloads_opened_by_registration": 0, "videos_decoded_by_registration": 0,
              "previous_visual_exposure": {"source": "reports/evidence/iws_box_rope_cache_actual_preflight.json",
                  "scope": "Only prespecified internal-training records for shape/decoder compatibility, not method selection."}}

    immutable_json(inputs, root / INPUTS)
    dependencies = {p: sha(root / p) for p in SOURCES + [CONFIG, INPUTS, str(SPLIT_RELATIVE), str(METADATA_RELATIVE),
                    "reports/evidence/iws_temporal_semantics_audit.json", "references/real_dinov2_sources.json",
                    "data/pretrained/dinov2-small/provenance.json", "configs/real_video_development/iws_acquisition_v1.json"]}
    for file in encoder["files"]:
        path = str(Path(config["encoder_root"]) / file["file"]); dependencies[path] = sha(root / path)
    registration = {"schema": "shiftwm_iws_task_cache_registration_v1", "status": "registered_before_full_cache_extraction",
                    "task": task, "command_width": inventory.command_width,
                    "scope": "Input/cache preparation only, not a model/training/evaluation protocol", "dependencies": dependencies,
                    "input_records_sha256": canonical_hash(records), "expected_episodes": len(records), "encoder": encoder,
                    "preserved_temporal_ambiguity": "No windows defined. Official s to s+59 with60 command rows remains unchanged and reserved.",
                    "resource_policy": "At most two cache GPUs; one per task,8CPUs/24GiB, actual allocated free-memory preflight >=4GiB. No model training."}
    immutable_json(registration, root / REGISTRATION)
    return {"status": registration["status"], "registration_sha256": sha(root / REGISTRATION),
            "input_manifest_sha256": sha(root / INPUTS), "episodes": len(records), "native_frames": inputs["native_frame_rows"]}


def checked_registration(task, root=ROOT):
    CONFIG, INPUTS, REGISTRATION = task_paths(task)
    root = Path(root); registry = read_json(root / REGISTRATION)
    require(registry.get("schema") == "shiftwm_iws_task_cache_registration_v1" and registry.get("status") == "registered_before_full_cache_extraction", "Missing preparation registration")
    for relative, expected in registry["dependencies"].items():
        require(sha(root / relative) == expected, "Registered cache source/config/encoder changed: " + relative)
    inventory = InternalInventory(root, task); config = read_json(root / CONFIG); validate_config(config, inventory)
    inputs = read_json(root / INPUTS)
    require(inputs["split_sha256"] == inventory.split_sha256 and len(inputs["episodes"]) == registry["expected_episodes"] == 602, "Wrong registered population")
    require(inputs["task"] == registry["task"] == task and inputs["command_width"] == registry["command_width"] == inventory.command_width, "Registered task/width changed")
    require(canonical_hash(inputs["episodes"]) == registry["input_records_sha256"], "Registered input rows changed")
    return inventory, config, inputs, registry


def static_identity(root, inventory, config, inputs):
    """Expected current registration fields, safe to verify a CUDA cache on CPU."""
    CONFIG, INPUTS, REGISTRATION = task_paths(inventory.task)
    return {"schema": "shiftwm_iws_task_extraction_identity_v1", "task": inventory.task, "command_width": inventory.command_width, "registration_sha256": sha(Path(root) / REGISTRATION),
            "split_sha256": inventory.split_sha256, "input_manifest_sha256": sha(Path(root) / INPUTS),
            "input_records_sha256": canonical_hash(inputs["episodes"]), "config_sha256": sha(Path(root) / CONFIG),
            "preprocessing": PREPROCESSING, "pooling_and_storage_precision": "float32", "tf32": False,
            "batch_size": config["batch_size"],
            "encoder_provenance_sha256": sha(Path(root) / config["encoder_root"] / "provenance.json"),
            "official_validation_payloads_allowed": False}


def runtime_identity(root, inventory, config, inputs, device):
    import cv2
    import torch
    import transformers
    require(device in ("cpu", "cuda"), "Unsupported extraction device")
    if device == "cuda":
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Allocated BF16-capable GPU required")
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
        require(free_bytes >= config["minimum_free_cuda_bytes"], "Allocated GPU has insufficient actually free memory; no fallback")
        print(json.dumps({"event": "allocated_device_preflight", "free_bytes": free_bytes, "total_bytes": total_bytes,
                          "minimum_free_bytes": config["minimum_free_cuda_bytes"], "device": torch.cuda.get_device_name()}), flush=True)
    return {**static_identity(root, inventory, config, inputs), "device": device,
            "encoder_precision": "bfloat16" if device == "cuda" else "float32",
            "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": importlib.metadata.version("numpy"),
                        "h5py": importlib.metadata.version("h5py"), "opencv": cv2.__version__,
                        "opencv_build_sha256": __import__('hashlib').sha256(cv2.getBuildInformation().encode()).hexdigest(),
                        "cuda_version": torch.version.cuda if device == "cuda" else None,
                        "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None}}


def build(task, device, output=None, root=ROOT):
    root = Path(root); inventory, config, inputs, registry = checked_registration(task, root)
    output = Path(output) if output else root / (config["output_root"] + ("_cpu" if device == "cpu" else ""))
    identity = runtime_identity(root, inventory, config, inputs, device)
    model = None
    def encoder(images, batch_size):
        nonlocal model
        if model is None:
            model = DinoSpatialEncoder(root, root / config["encoder_root"], device)
        return model(images, batch_size)
    result = build_cache(inventory, inputs["episodes"], output, identity, encoder, config["batch_size"],
                         before_complete=lambda: checked_registration(task, root))
    checked_registration(task, root)
    return result


def validate(task, output, root=ROOT):
    inventory, config, inputs, registry = checked_registration(task, root)
    cache = IWSFeatureCache(output, inventory, inputs["episodes"], static_identity(root, inventory, config, inputs))
    for eid in inventory.selected():
        cache.episode(eid, inventory.assignment[eid])
    return {"status": "complete_cache_all_packages_verified", "episodes": len(inventory.assignment),
            "manifest_sha256": sha(Path(output) / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command", choices=("register", "build", "validate"))
    parser.add_argument("--task", required=True, choices=tuple(TASK_WIDTHS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "register":
        result = prepare(args.task)
    elif args.command == "build":
        result = build(args.task, args.device, args.output)
    else:
        require(args.output is not None, "Validation requires explicit --output")
        result = validate(args.task, args.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
