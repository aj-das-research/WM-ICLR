"""Pre-access identities for the new, exact-handle IWS reserved evaluation.

Only identity JSON and already trained checkpoint files are read here. Original
video, HDF5 values and reserved feature arrays belong to the separately reviewed
cache implementation. An independent, source-bound receipt opens that boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/real_video_iws_reserved_v1/protocol.json")
REGISTRATION_PATH = Path("configs/real_video_iws_reserved_v1/registration.json")
REVIEW_PATH = Path("reports/real_video_iws_reserved_v1/preaccess_review.json")
TASK_WIDTHS = {"pusht": 4, "bimanual_box": 14, "bimanual_rope": 8}
MODES = ("autoregressive", "anchored_additive", "bounded_spatial_mix", "unbounded_spatial_mix")
SCHEMA = "shiftwm_iws_reserved_registration_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def unique(pairs):
        out = dict(pairs)
        if len(out) != len(pairs):
            raise ValueError("Duplicate JSON keys")
        return out
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local(root, relative):
    root = Path(root).resolve()
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError("Expected a confined relative path")
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError("Path leaves the registered workspace")
    return path


def immutable_json(value, path):
    path = Path(path)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Refusing to overwrite different evidence: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        # link, unlike replace, also rejects a concurrent writer.
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def expected_grid():
    return {(task, mode, seed) for task in TASK_WIDTHS for mode in MODES for seed in (0, 1, 2)}


def reserved_payload_prefixes(registry):
    return ["data/features/iws_reserved_v1/"] + [
        f"data/real_video/iws_public_v1/extracted/iws_converted/{task}/traj_{eid}/"
        for task, row in registry["tasks"].items() for eid in row["episode_ids"]]


def validate_registration(registry):
    if registry.get("schema") != SCHEMA or registry.get("status") != "registered_before_reserved_access":
        raise ValueError("A new reserved registration is required")
    if registry.get("expected_runs") != 36 or registry.get("payloads_read_at_registration") != 0:
        raise ValueError("Incomplete roster or pre-access identity violation")
    runs = registry.get("runs", [])
    grid = [(r["task"], r["mode"], r["seed"]) for r in runs]
    if len(grid) != 36 or set(grid) != expected_grid() or len({r["name"] for r in runs}) != 36:
        raise ValueError("All 36 fixed model identities are required")
    if set(registry.get("tasks", {})) != set(TASK_WIDTHS):
        raise ValueError("Missing reserved task")
    for task, width in TASK_WIDTHS.items():
        row = registry["tasks"][task]
        if (row["command_width"] != width or row["expected_handles"] != 200
                or row["expected_trajectories"] != 10
                or len(row["episode_ids"]) != 10 or len(set(row["episode_ids"])) != 10):
            raise ValueError("Reserved task denominator/layout differs")
        if len(row["episode_ids"]) != 10:
            raise ValueError("Ten trajectories per task are required")
    evaluation = registry["evaluation"]
    expected = {"device": "cpu", "dtype": "float32", "threads": 8, "batch_size": 64,
                "horizon": 60, "prefix_horizons": [15, 30, 45], "bootstrap_draws": 10000,
                "bootstrap_seed": 173, "primary_method": "bounded_spatial_mix",
                "primary_reference": "anchored_additive", "primary_metric": "standardized_mse"}
    if any(evaluation.get(k) != v for k, v in expected.items()):
        raise ValueError("The fixed inference or primary-comparison contract differs")
    dependencies = registry.get("dependencies", {})
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("Registration omits its immutable dependencies")
    payload_prefixes = reserved_payload_prefixes(registry)
    for name, digest in dependencies.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Dependency path is not confined")
        if any((str(path) + "/").startswith(prefix) for prefix in payload_prefixes):
            raise ValueError("Reserved payloads cannot be pre-access dependencies")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid dependency SHA256")
    for run in runs:
        if run["name"] != f"{run['task']}_{run['mode']}_s{run['seed']}":
            raise ValueError("Run name does not identify its task/method/seed")
        for filename, field in (("model.pt", "checkpoint_sha256"), ("config.json", "config_sha256"),
                                ("package_manifest.json", "package_manifest_sha256")):
            if dependencies.get(str(Path(run["package_path"]) / filename)) != run[field]:
                raise ValueError("Selected package is not bound by the registration")
    return registry


def checked_registration(root=ROOT, registration_path=None, require_review=True):
    root = Path(root).resolve()
    path = Path(registration_path) if registration_path is not None else root / REGISTRATION_PATH
    if not path.is_absolute():
        path = root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError("Registration leaves the workspace")
    registry = validate_registration(read_json(path))
    if require_review:
        review = read_json(root / REVIEW_PATH)
        if (review.get("status") != "passed" or review.get("registration_sha256") != sha(path)
                or review.get("reserved_payloads_read") != 0
                or review.get("source_sha256") != registry["dependencies"]):
            raise ValueError("Independent pre-access review is missing or does not bind this exact study")
    # Check the review before opening dependency payloads: an unreviewed
    # registry must never turn its own dependency list into an access grant.
    for relative, expected in registry["dependencies"].items():
        source = local(root, relative)
        resolved_name = str(source.resolve().relative_to(root)) + "/"
        if any(resolved_name.startswith(prefix) for prefix in reserved_payload_prefixes(registry)):
            raise ValueError("Reserved payload alias cannot be a pre-access dependency")
        if sha(source) != expected:
            raise ValueError("Frozen reserved-study dependency changed: " + relative)
    return registry


def prepare_registration(root=ROOT):
    """Construct a candidate by reading metadata and existing trained artifacts."""
    root = Path(root).resolve()
    config = read_json(root / CONFIG_PATH)
    split_rel = "configs/real_video_iws/split_v1.json"
    split = read_json(root / split_rel)
    original_rel = "reports/real_video_iws/development_finalization.json"
    followup_rel = "reports/real_video_iws_unbounded/development_finalization.json"
    original, followup = read_json(root / original_rel), read_json(root / followup_rel)
    if (original.get("status") != "passed" or original.get("completed_runs") != 27
            or followup.get("status") != "passed" or followup.get("completed_new_runs") != 9
            or followup.get("completed_v1_comparator_runs") != 27
            or followup["baseline_finalization_sha256"] != sha(root / original_rel)):
        raise ValueError("Both complete, mutually bound development finalizers are required")
    for report in (original, followup):
        if report.get("official_validation_payloads_read") != 0:
            raise ValueError("Development finalizer includes reserved data")
    dependencies = {}
    forbidden = reserved_payload_prefixes({"tasks": {
        task: {"episode_ids": split["partitions"][task]["reserved_official_validation"]}
        for task in TASK_WIDTHS}})
    def bind(relative, expected=None):
        relative = str(relative)
        path = local(root, relative)
        resolved = str(path.resolve().relative_to(root))
        if any((name + "/").startswith(prefix) for name in (relative, resolved) for prefix in forbidden):
            raise ValueError("Reserved payloads cannot be read while preparing a registration")
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError("Existing frozen identity changed: " + relative)
        if relative in dependencies and dependencies[relative] != actual:
            raise ValueError("Conflicting dependency: " + relative)
        dependencies[relative] = actual
        return actual
    for name in (CONFIG_PATH, split_rel, original_rel, followup_rel):
        bind(name)
    for reg_rel in ("configs/real_video_iws/training_registration_v1.json",
                    "configs/real_video_iws_unbounded/registration_v1.json",
                    "reports/real_video_iws/recovery/common_cpu_v1/registration.json"):
        prior = read_json(root / reg_rel); bind(reg_rel)
        for field in ("dependencies", "frozen_scientific_dependencies", "operational_dependencies"):
            for relative, expected in prior.get(field, {}).items():
                bind(relative, expected)
    # Newly implemented science plus its focused tests are fixed before access.
    for folder in ("src/shiftwm/real_video_iws_reserved", "scripts/real_video_iws_reserved_v1"):
        for path in sorted((root / folder).glob("*")):
            if path.is_file() and path.suffix in (".py", ".slurm"):
                bind(path.relative_to(root))
    for path in sorted((root / "tests").glob("test_iws_reserved*.py")):
        bind(path.relative_to(root))
    for relative in config["required_source_files"]:
        bind(relative)
    tasks = {}
    training_config = read_json(root / "configs/real_video_iws/training_v1.json")
    for task, width in TASK_WIDTHS.items():
        handles = split["reserved_handles"][task]
        bind(handles["path"], handles["sha256"])
        ids = split["partitions"][task]["reserved_official_validation"]
        records = [r for r in split["metadata"][task] if r["episode_id"] in ids]
        if any(r["split"] != "val" for r in records) or len(records) != 10:
            raise ValueError("Reserved metadata population mismatch")
        stats_rel = str(Path(training_config["tasks"][task]["cache_root"]) / "training_statistics.json")
        stats_sha = bind(stats_rel)
        tasks[task] = {"command_width": width, "cache_root": "data/features/iws_reserved_v1/" + task,
                       "training_statistics_path": stats_rel, "training_statistics_sha256": stats_sha,
                       "handles_path": handles["path"], "handles_sha256": handles["sha256"],
                       "episode_ids": ids, "expected_handles": 200, "expected_trajectories": 10,
                       "native_frames": sum(r["shapes"]["target_qpos"][0] for r in records)}
    rows = {r["name"]: r for r in followup["per_run"]}
    if len(rows) != 36:
        raise ValueError("Follow-up does not bind the complete fixed model roster")
    runs = []
    for task in TASK_WIDTHS:
        for mode in MODES:
            for seed in (0, 1, 2):
                name = f"{task}_{mode}_s{seed}"; result = rows[name]
                namespace = "real_video_iws_unbounded" if mode == "unbounded_spatial_mix" else "real_video_iws"
                directory = Path("runs") / namespace / "v1" / name
                package = (root / directory / "best").resolve()
                if not package.is_relative_to((root / directory).resolve()):
                    raise ValueError("Selected package escapes the original run")
                relative = package.relative_to(root)
                cfg = read_json(package / "config.json")
                stats = read_json(root / tasks[task]["training_statistics_path"])
                # Statistics JSON retains double precision; packages store the
                # exact FP32 buffers used at training/inference. Compare those
                # bytes, without fitting or relaxing numerical equality.
                def fp32_bytes(values):
                    return struct.pack("<" + "f" * len(values), *values)
                if stats.get("fit_split") != "internal_train" or any(
                        fp32_bytes(cfg[key]) != fp32_bytes(stats[key])
                        for key in ("feature_mean", "feature_std", "command_mean", "command_std")):
                    raise ValueError("Selected predictor does not use the fixed training-only statistics")
                recipe = cfg["metadata"]["identity"]["scientific_config"]
                if (recipe["task"], recipe["mode"], recipe["seed"]) != (task, mode, seed):
                    raise ValueError("Selected package task/method/seed differs")
                if result["completed_epochs"] != 30:
                    raise ValueError("Training did not complete its full budget")
                checkpoint = bind(relative / "model.pt", result["selected_checkpoint_sha256"])
                config_sha = bind(relative / "config.json")
                manifest_sha = bind(relative / "package_manifest.json")
                bind(result["evaluation_path"], result["evaluation_sha256"])
                for key in ("training_config.json", "training_summary.json", "metrics.jsonl"):
                    if (root / directory / key).is_file(): bind(directory / key)
                runs.append({"name": name, "task": task, "mode": mode, "seed": seed,
                             "output": str(directory), "package_path": str(relative),
                             "package_kind": cfg["package_kind"], "checkpoint_sha256": checkpoint,
                             "config_sha256": config_sha, "package_manifest_sha256": manifest_sha,
                             "selected_epoch": result["selected_epoch"], "completed_epochs": 30})
    from shiftwm.real_video_iws.features import verify_encoder, PREPROCESSING
    encoder = verify_encoder(root, root / config["extraction"]["encoder_root"])
    for row in encoder["files"]:
        bind(Path(config["extraction"]["encoder_root"]) / row["file"], row["sha256"])
    extraction = dict(config["extraction"], preprocessing=PREPROCESSING)
    value = {"schema": SCHEMA, "status": "registered_before_reserved_access", "created_utc": now(),
             "scope": config["scope"], "authorization": config["authorization"],
             "expected_runs": 36, "tasks": tasks, "runs": runs,
             "evaluation": config["evaluation"], "extraction": extraction,
             "qualitative_selection": config["qualitative_selection"],
             "payloads_read_at_registration": 0, "dependencies": dict(sorted(dependencies.items())),
             "split_sha256": sha(root / split_rel), "development_original_sha256": sha(root / original_rel),
             "development_followup_sha256": sha(root / followup_rel),
             "no_test_based_model_selection": True}
    return validate_registration(value)
