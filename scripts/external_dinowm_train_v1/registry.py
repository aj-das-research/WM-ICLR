"""Freeze all six external-predictor arms before training; metadata-only gate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import datetime

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / "reports/external_dinowm_train_v1"
REG = REPORT / "registration.json"
REVIEW = REPORT / "source_review.json"
MODES = ("official_one_step_shifted", "matched_recursive_h10")
PROFILE = ROOT / "reports/external_dinowm_profile_v1/job_201998/profile.json"
EVALUATION = {"split": "val", "stride": 5, "windows": 1631, "episodes": 141,
              "sessions": 59, "horizons": list(range(1, 11)),
              "metrics": ["native_mse", "native_persistence_mse", "original_2x2_mse", "original_2x2_persistence_mse"]}
RESOURCE_POLICY = {"gpu_per_job": 1, "allocation_seconds": 7200,
                   "epoch_boundary_requeue_after_seconds": 5400, "max_restarts": 5,
                   "allowed_partitions": ["ws-ia", "gpu"], "account": "students",
                   "maximum_simultaneous_ws_ia_jobs": 2, "maximum_simultaneous_gpu_partition_jobs": 1,
                   "no_implicit_model_batch_or_precision_fallback": True}


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream: stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def model_module():
    spec = importlib.util.spec_from_file_location("external_dinowm_registry_model", HERE / "model.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def expected_config(mode, seed):
    if mode not in MODES or type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Unknown mode/seed")
    return {"mode": mode, "seed": seed, "model_config": model_module().ARCHITECTURE,
            "epochs": 30, "train_horizon": 10, "validation_horizon": 10,
            "lr": 1e-4, "min_lr": 1e-6, "weight_decay": .01, "batch_size": 128,
            "grad_clip": 1., "bf16": True, "stride": 2, "validation_stride": 5,
            "cpu_threads": 8, "num_workers": 0, "device": "cuda",
            "cache_root": "data/features/droid_spatial_v1",
            "original_cache": "data/features/droid_selected_v1",
            "metadata_audit": "data/real_video/droid_selected/processed/data_audit.json",
            "protocol_path": "scripts/external_dinowm_train_v1/README.md",
            "output_dir": f"runs/external_dinowm_train_v1/{mode}_s{seed}"}


def validate_config(config):
    expected = expected_config(config.get("mode"), config.get("seed"))
    scientific = {k: v for k, v in config.items() if k not in ("resume_if_present", "max_runtime_seconds")}
    if scientific != expected: raise ValueError("Training configuration differs from complete registered contract")


def sources():
    names = ["model.py", "train.py", "evaluate.py", "registry.py", "campaign.py", "run.slurm", "test_contract.py", "README.md"]
    paths = [HERE / name for name in names]
    original = ["scripts/real_video/train.py", "scripts/real_video_spatial/train.py",
                "scripts/real_video_spatial/evaluate.py", "scripts/real_video_spatial/validate_ledger.py",
                "src/shiftwm/real_video/data.py", "src/shiftwm/real_video/model.py",
                "src/shiftwm/real_video_spatial/data.py", "src/shiftwm/real_video_spatial/model.py",
                "src/shiftwm/model.py", "src/shiftwm/upstream.py", "src/shiftwm/checkpoint.py",
                "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/NOTICE.json",
                "data/features/droid_spatial_v1/manifest.json", "data/features/droid_spatial_v1/training_statistics.json",
                "data/features/droid_selected_v1/manifest.json", "data/features/droid_selected_v1/training_statistics.json",
                "data/real_video/droid_selected/processed/manifest.json", "data/real_video/droid_selected/processed/data_audit.json",
                "data/pretrained/dinov2-small/provenance.json",
                "reports/external_dinowm_profile_v1/registration.json", "reports/external_dinowm_profile_v1/source_review.json"]
    paths += [ROOT / name for name in original] + [PROFILE]
    profile = read(ROOT / "reports/external_dinowm_profile_v1/registration.json")
    paths += [ROOT / name for name in profile["source_sha256"]]
    reporting = ROOT / "scripts/external_dinowm_reporting_v1"
    paths += [reporting / name for name in ("finalize.py", "test_finalize.py", "README.md")]
    paths += [HERE / "configs" / f"{mode}_s{seed}.json" for seed in (0, 1, 2) for mode in MODES]
    return sorted(set(paths))


def profile_gate():
    result = read(PROFILE)
    registration_path = ROOT / "reports/external_dinowm_profile_v1/registration.json"
    registration = read(registration_path)
    review_path = ROOT / "reports/external_dinowm_profile_v1/source_review.json"
    review = read(review_path)
    if (result.get("status") != "complete" or result.get("registration_sha256") != sha(registration_path)
            or result.get("source_review_sha256") != sha(review_path)
            or review.get("status") != "passed" or review.get("source_sha256") != registration["source_sha256"]
            or result.get("source_sha256") != registration["source_sha256"]):
        raise ValueError("Independent full-shape capacity evidence absent or stale")
    for name, value in registration["source_sha256"].items():
        if sha(ROOT / name) != value: raise ValueError("Profile dependency changed")
    if {r["objective"] for r in result["rows"]} != set(MODES) or len(result["rows"]) != 2:
        raise ValueError("Both objective profiles are required")
    for row in result["rows"]:
        if (row["status"] != "measured" or row["batch_size"] != 128
                or row["parameters_total"] != 19412420 or row["parameters_trainable"] != 19412420
                or row["projection"]["estimated_30epoch_hours_one_seed_without_io"] > 1.5):
            raise ValueError("Full-size external predictor exceeds measured scheduling gate")
    return result


def register():
    if REG.exists(): raise ValueError("Existing six-run registration is immutable")
    profile_gate()
    runs = []
    for seed in (0, 1, 2):
        for mode in MODES:
            name = f"{mode}_s{seed}"; path = HERE / "configs" / (name + ".json")
            config = expected_config(mode, seed)
            if path.exists():
                if read(path) != config: raise ValueError("Candidate configuration differs")
            else: write_new(path, config)
            runs.append({"name": name, "mode": mode, "seed": seed,
                         "config": str(path.relative_to(ROOT)), "sha256": sha(path)})
    manifests = read(ROOT / "data/features/droid_spatial_v1/manifest.json")
    if {r["split"] for r in manifests["episodes"]} != {"train", "val"}:
        raise ValueError("Spatial cache must contain train/development only")
    value = {"schema": "adapted_official_dinowm_droid_registration_v1", "status": "registered",
             "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "expected_runs": 6, "expected_epochs_per_run": 30, "modes": list(MODES), "runs": runs,
             "scope": "Original DROID train/development only. No reserved or test model evaluation.",
             "selection": "window_mean_all10_shared_channel_standardized_mse",
             "native_loss_grid_indices": [1, 2, 3], "recursive_loss_grid_indices": list(range(3, 13)),
             "evaluation": EVALUATION, "resource_policy": RESOURCE_POLICY,
             "dependencies": {str(p.relative_to(ROOT)): sha(p) for p in sources()}}
    write_new(REG, value)
    print(json.dumps({"registration": str(REG.relative_to(ROOT)), "sha256": sha(REG), "runs": 6}))
    return value


def verify(require_review=True):
    record = read(REG)
    if require_review:
        review = read(REVIEW)
        if (review.get("status") != "passed" or review.get("registration_sha256") != sha(REG)
                or review.get("source_sha256") != record["dependencies"]):
            raise ValueError("Six-run independent source review missing or stale")
    expected = {(mode, seed) for mode in MODES for seed in (0, 1, 2)}
    runs = record.get("runs", [])
    if (record.get("schema") != "adapted_official_dinowm_droid_registration_v1"
            or record.get("status") != "registered" or record.get("expected_epochs_per_run") != 30
            or record.get("modes") != list(MODES)
            or record.get("evaluation") != EVALUATION or record.get("resource_policy") != RESOURCE_POLICY
            or record.get("native_loss_grid_indices") != [1, 2, 3]
            or record.get("recursive_loss_grid_indices") != list(range(3, 13))
            or record.get("expected_runs") != 6 or len(runs) != 6
            or {(r["mode"], r["seed"]) for r in runs} != expected
            or len({r["name"] for r in runs}) != 6
            or record.get("selection") != "window_mean_all10_shared_channel_standardized_mse"):
        raise ValueError("Complete mode/seed grid or selection differs")
    if set(record["dependencies"]) != {str(p.relative_to(ROOT)) for p in sources()}:
        raise ValueError("Registered source closure differs")
    for name, expected_sha in record["dependencies"].items():
        if sha(ROOT / name) != expected_sha: raise ValueError("Frozen dependency changed: " + name)
    for row in runs:
        expected_path = str((HERE / "configs" / f"{row['mode']}_s{row['seed']}.json").relative_to(ROOT))
        if (type(row["seed"]) is not int or row["name"] != f"{row['mode']}_s{row['seed']}"
                or row["config"] != expected_path or sha(ROOT / row["config"]) != row["sha256"]):
            raise ValueError("Registered run identity/configuration differs")
        validate_config(read(ROOT / row["config"]))
    profile_gate()
    return record
