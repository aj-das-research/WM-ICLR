#!/usr/bin/env python3
"""Build a new synthetic-only inference release without reading dataset payloads."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / "reports/real_video_iws_release_v2"
REG = REPORT / "registration.json"
REVIEW = REPORT / "source_review.json"
DEFAULT = ROOT / "artifacts/releases/iws_single_observation_rowwise_local_v2"
RECOVERY = ROOT / "configs/real_video_iws_reserved_recovery_v2/registration.json"
RESULT = ROOT / "reports/real_video_iws_reserved_recovery_v2/finalization.json"
RESULT_REVIEW = ROOT / "reports/real_video_iws_reserved_recovery_v2/independent_result_review.json"
DEVELOPMENT = ROOT / "reports/real_video_iws_unbounded/development_finalization.json"
OLD = [ROOT / "artifacts/releases/iws_single_observation_local_v1",
       ROOT / "artifacts/releases/iws_unbounded_single_observation_local_v1"]
OLD_EXPORTS = [ROOT / "reports/real_video_iws/release/local_inference_export.json",
               ROOT / "reports/real_video_iws_unbounded/release/local_inference_export.json"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value); return value


runtime = module("iws_new_release_runtime", HERE / "runtime.py")
sha, read = runtime.sha, runtime.read


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle: handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def copy_exact(source, target):
    source = Path(source); target = Path(target); before = sha(source)
    if source.is_symlink(): raise ValueError("Source package must be an independent regular file")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    if sha(source) != before or sha(target) != before: raise ValueError("Copy changed source bytes")


def authorized_models():
    """Metadata and selected predictor files only; never open old real fixtures."""
    registration = read(RECOVERY); result = read(RESULT); review = read(RESULT_REVIEW); dev = read(DEVELOPMENT)
    if (registration.get("expected_runs") != 36 or result.get("status") != "passed"
            or result.get("completed_runs") != 36 or result.get("registration_sha256") != sha(RECOVERY)
            or review.get("status") != "passed" or review.get("completed_runs") != 36
            or review.get("registration_sha256") != sha(RECOVERY) or review.get("finalization_sha256") != sha(RESULT)
            or dev.get("status") != "passed" or dev.get("completed_new_runs") != 9
            or dev.get("completed_v1_comparator_runs") != 27):
        raise ValueError("Complete reviewed selected-checkpoint evidence required")
    fixed = {r["name"]: r for r in registration["runs"]}
    completed = {r["name"]: r for r in result["per_run"]}
    if set(fixed) != runtime.expected_names() or set(completed) != set(fixed):
        raise ValueError("Recovery selected model grid differs")
    for name in ("src/shiftwm/real_video_iws/model.py", "src/shiftwm/real_video_iws_unbounded/model.py",
                 "src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py",
                 "src/shiftwm/upstream.py", "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/NOTICE.json"):
        if registration["dependencies"].get(name) != sha(ROOT / name):
            raise ValueError("Recovery model/backend source changed: " + name)
    records = []
    for bundle, export_path, count in zip(OLD, OLD_EXPORTS, (27, 9)):
        manifest_path = bundle / "manifest.json"
        export = read(export_path)
        if (export.get("manifest_sha256") != sha(manifest_path) or export.get("models") != count
                or export.get("bundle") != str(bundle.relative_to(ROOT))
                or export.get("status") != "passed_local_inference_export"):
            raise ValueError("Old bundle manifest differs from completed export receipt")
        pinned_manifest = registration["dependencies"].get(str(manifest_path.relative_to(ROOT)))
        if pinned_manifest is not None and pinned_manifest != sha(manifest_path):
            raise ValueError("Old bundle manifest differs from recorded recovery dependency")
        manifest = read(manifest_path)
        for row in manifest["models"]:
            pinned = fixed[row["name"]]; selected = completed[row["name"]]
            if (any(row[k] != pinned[k] for k in ("name", "task", "mode", "seed", "selected_epoch", "completed_epochs"))
                    or selected["selected_epoch"] != row["selected_epoch"]):
                raise ValueError("Old export differs from selected recovery model")
            package = bundle / row["directory"]
            for filename, expected in (("model.pt", pinned["checkpoint_sha256"]), ("config.json", pinned["config_sha256"])):
                if sha(package / filename) != expected or manifest["files"][row["directory"] + "/" + filename] != expected:
                    raise ValueError("Selected model/config bytes differ")
            package_manifest = read(package / "package_manifest.json")
            if (package_manifest.get("package_kind") != pinned["package_kind"]
                    or set(package_manifest.get("files", {})) != {"model.pt", "config.json"}
                    or manifest["files"][row["directory"] + "/package_manifest.json"] != sha(package / "package_manifest.json")):
                raise ValueError("Old exported inference package is not complete")
            for filename, expected in package_manifest["files"].items():
                if sha(package / filename) != expected: raise ValueError("Old package changed")
            records.append({**{k: row[k] for k in ("name", "task", "mode", "seed", "selected_epoch", "completed_epochs", "training_identity")},
                            "directory": "models/" + row["name"], "package_kind": pinned["package_kind"],
                            "source_package": str(package.relative_to(ROOT)), "source_bundle_manifest": str(manifest_path.relative_to(ROOT)),
                            "checkpoint_sha256": pinned["checkpoint_sha256"], "config_sha256": pinned["config_sha256"]})
    if len(records) != 36 or {r["name"] for r in records} != runtime.expected_names():
        raise ValueError("Incomplete source export roster")
    return sorted(records, key=lambda r: r["name"])


def sources(rows):
    paths = [HERE / n for n in ("build.py", "runtime.py", "example.py", "README.md", "MODEL_CARD.md", "MODEL_LICENSE.md", "test_runtime.py", "run.slurm")]
    paths += [ROOT / n for n in runtime.SOURCE_FILES] + [ROOT / "LICENSE", RECOVERY, RESULT, RESULT_REVIEW, DEVELOPMENT]
    paths += [ROOT / "configs/real_video_iws" / f"{task}_cache_v1.json" for task in runtime.TASKS]
    paths += [ROOT / "data/pretrained/dinov2-small/provenance.json",
              ROOT / "reports/real_video_iws/release/public_source_metadata.json"]
    paths += [p / "manifest.json" for p in OLD]
    paths += OLD_EXPORTS
    paths += [ROOT / row["source_package"] / name for row in rows for name in ("model.pt", "config.json", "package_manifest.json")]
    return sorted(set(paths))


def register():
    if REG.exists(): raise ValueError("Release registration already frozen")
    rows = authorized_models()
    value = {"schema": "iws_rowwise_release_registration_v2", "status": "registered_before_synthetic_inference",
             "backend": runtime.BACKEND, "models": rows, "batches": [1, 64, 8], "horizon": 60,
             "prefix_horizons": [15, 30, 45], "device": "cpu", "precision": "float32", "threads": 8, "interop_threads": 1,
             "fixture_scope": "deterministic synthetic inputs; no dataset examples or targets",
             "dependencies": {str(p.relative_to(ROOT)): sha(p) for p in sources(rows)}}
    write(REG, value); return {"registration_sha256": sha(REG), "models": 36, "dependencies": len(value["dependencies"])}


def verify():
    record = read(REG); review = read(REVIEW)
    if (review.get("status") != "passed" or review.get("registration_sha256") != sha(REG)
            or review.get("source_sha256") != record["dependencies"]):
        raise ValueError("Independent release source review missing or stale")
    if (record.get("backend") != runtime.BACKEND or record.get("batches") != [1, 64, 8]
            or record.get("horizon") != 60 or record.get("prefix_horizons") != [15, 30, 45]
            or record.get("device") != "cpu" or record.get("precision") != "float32"
            or record.get("threads") != 8 or record.get("interop_threads") != 1
            or record.get("fixture_scope") != "deterministic synthetic inputs; no dataset examples or targets"):
        raise ValueError("Release contract differs")
    if set(record["dependencies"]) != {str(p.relative_to(ROOT)) for p in sources(record["models"])}:
        raise ValueError("Source closure differs")
    for name, value in record["dependencies"].items():
        if sha(ROOT / name) != value: raise ValueError("Registered source changed: " + name)
    if authorized_models() != record["models"]: raise ValueError("Selected source roster changed")
    return record


def destination_guard(destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink(): raise ValueError("Refusing to replace any existing bundle")
    target = destination.resolve(); allowed = (ROOT / "artifacts/releases").resolve()
    if not target.is_relative_to(allowed) or target == allowed: raise ValueError("Bundle must stay under ignored artifacts/releases")
    archive = target.with_suffix(".tar.gz")
    if archive.exists() or archive.is_symlink(): raise ValueError("Archive already exists")
    return target


def selected_model(row):
    import torch
    import importlib
    sys.path.insert(0, str(ROOT / "src"))
    namespace = "real_video_iws_unbounded" if row["mode"] == "unbounded_spatial_mix" else "real_video_iws"
    source = importlib.import_module("shiftwm." + namespace + ".model")
    if Path(source.__file__).resolve() != ROOT / "src/shiftwm" / namespace / "model.py":
        raise ValueError("Wrong source model module")
    path = ROOT / row["source_package"]; state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    config = read(path / "config.json")
    if state["config"] != config or state["epoch"] != row["selected_epoch"]: raise ValueError("Source package identity differs")
    model = source.from_config(config); model.load_state_dict(state["state_dict"], strict=True)
    if any(config.get(k) != v for k, v in model.package_config.items()): raise ValueError("Source normalization reconstruction differs")
    if model.parameter_counts != config["metadata"]["parameter_counts"]: raise ValueError("Source parameter count differs")
    return model.cpu().eval(), config


def model_card(row, config):
    return (f"# {row['name']}\n\nTask: {row['task']}; method: {row['mode']}; training seed: {row['seed']}.\n\n"
            f"Completed epochs: 30. Selected epoch: {row['selected_epoch']}. Selection: {runtime.SELECTION}.\n\n"
            f"Package kind: `{row['package_kind']}`. Training identity: `{row['training_identity']}`.\n\n"
            f"Model SHA256: `{row['checkpoint_sha256']}`. Config SHA256: `{row['config_sha256']}`.\n\n"
            f"Parameter counts: `{json.dumps(config['metadata']['parameter_counts'], sort_keys=True)}`.\n\n"
            f"Input: one raw DINOv2 feature vector [B,6144] and native target_qpos commands [B,H,{runtime.TASKS[row['task']]}]. "
            "Coordinates and normalization follow the unchanged package config and preprocessing.json; command physical units are unverified. "
            "Output: [B,H-1,6144] feature forecasts. CPU FP32, 8 threads, interop1; explicit row-wise backend.\n\n"
            + ("This is a post-development no-tanh secondary ablation.\n\n" if row["mode"] == "unbounded_spatial_mix" else
               "This is an original registered internal-study arm.\n\n")
            + "See ../MODEL_CARD.md for complete training, evaluation, negative-results, license and reuse limitations. "
              "All development and reserved comparisons are retained in ../evidence/. This card makes no per-seed superiority claim. "
              "No RGB decoder, encoder weights, policy, optimizer or dataset example is included.\n")


def build(destination):
    registration = verify(); destination = destination_guard(destination)
    import numpy as np
    import torch
    runtime.configure_runtime()
    backend = module("iws_release_original_rowwise_backend", ROOT / "src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py")
    began = time.monotonic(); destination.parent.mkdir(parents=True, exist_ok=True)
    with (REPORT / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary = Path(tempfile.mkdtemp(prefix=".iws-rowwise-pending-", dir=destination.parent))
        for relative in runtime.SOURCE_FILES | {"LICENSE"}: copy_exact(ROOT / relative, temporary / relative)
        for name in ("runtime.py", "example.py", "README.md", "MODEL_CARD.md", "MODEL_LICENSE.md"):
            copy_exact(HERE / name, temporary / name)
        (temporary / "requirements.txt").write_text(f"torch=={str(torch.__version__).split('+')[0]}\nnumpy=={np.__version__}\neinops=={__import__('einops').__version__}\n")
        preprocessing = {"encoder": read(ROOT / "data/pretrained/dinov2-small/provenance.json"),
            "image_preprocessing": read(ROOT / "configs/real_video_iws/pusht_cache_v1.json")["preprocessing"],
            "extraction_precision": "CUDA BF16 encoder; FP32 pooling/storage; TF32 disabled",
            "encoder_weights_included": False, "cpu_reencoding_cache_identity_claimed": False,
            "coordinate_layout": "channel_major_384x4x4_shared_channel_normalization",
            "normalization": "frozen training-only feature/command means and sample standard deviations, std floor1e-5; embedded in selected model",
            "command_source": "native stored target_qpos rows, all columns in source order; no grouping or resampling",
            "command_dimensions": runtime.TASKS, "command_units": "unverified recorded coordinates; no physical interpretation",
            "timebase": "native stored row indices, not seconds", "offset_contract": "offset k>=1 consumes native command rows0..k; H rows yield H-1 forecasts"}
        write(temporary / "preprocessing.json", preprocessing)
        for name, path in (("development_summary.json", DEVELOPMENT), ("reserved_recovery_summary.json", RESULT)):
            value = read(path)
            write(temporary / "evidence" / name, {"original_path": str(path.relative_to(ROOT)), "original_sha256": sha(path),
                **{k: v for k, v in value.items() if k not in ("source_dependencies", "cache_payload_sha256_from_metadata")}})
        references = {}; rows = []
        for row in registration["models"]:
            for filename in ("model.pt", "config.json", "package_manifest.json"):
                copy_exact(ROOT / row["source_package"] / filename, temporary / row["directory"] / filename)
            model, config = selected_model(row)
            initial, commands = runtime.synthetic_inputs(row["task"], 1)
            with torch.inference_mode(): native = model.predict(torch.from_numpy(initial), torch.from_numpy(commands))
            if not torch.isfinite(native).all(): raise ValueError("Native synthetic reference nonfinite")
            backend.install_rowwise_gru(model)
            predictor = runtime.Predictor(model, {"backend": runtime.BACKEND})
            reference = {}
            for batch_size in (1, 64, 8):
                initial, commands = runtime.synthetic_inputs(row["task"], batch_size)
                x, u = torch.from_numpy(initial), torch.from_numpy(commands)
                prediction = predictor.predict(x, u)
                if list(prediction.shape) != [batch_size, 59, 6144] or not torch.isfinite(prediction).all():
                    raise ValueError("Invalid full-shape synthetic prediction")
                reference[str(batch_size)] = {"shape": list(prediction.shape), "prediction_sha256": runtime.array_sha(prediction.numpy()),
                    "initial_sha256": runtime.array_sha(initial), "commands_sha256": runtime.array_sha(commands),
                    "prefix_checks": runtime.prefix_checks(predictor, x, u, prediction)}
                if batch_size == 1:
                    (temporary / "fixtures").mkdir(exist_ok=True)
                    np.savez_compressed(temporary / "fixtures" / (row["name"] + "_b1_prediction.npz"), predictions=prediction.numpy())
                    reference["native_vs_rowwise_b1"] = {"maximum_absolute_difference": float((native - prediction).abs().max()),
                        "bitwise_equal": bool(torch.equal(native, prediction)), "scope": "one fixed synthetic example; not general backend equivalence"}
            references[row["name"]] = reference; rows.append(row)
            card = temporary / "model_cards" / (row["name"] + ".md"); card.parent.mkdir(exist_ok=True); card.write_text(model_card(row, config))
            print(row["name"] + ": selected bytes copied, synthetic references/prefix checks complete", flush=True)
        write(temporary / "fixtures/references.json", references)
        write(temporary / "provenance.json", {"registration_sha256": sha(REG), "source_review_sha256": sha(REVIEW),
              "source_dependencies": registration["dependencies"], "backend": runtime.BACKEND,
              "selected_weights_changed": False, "old_bundles_overwritten": False, "dataset_payloads_read": 0,
              "fixture_recipe": "PCG64 seed20260920+taskindex; independent raw features U[-.5,.5], commands U[-.25,.25]",
              "results_recomputed": False, "encoder_weights_included": False, "optimizer_or_rng_included": False})
        manifest = {"schema": runtime.SCHEMA, "status": "local_prerelease", "backend": runtime.BACKEND,
                    "device": "cpu", "precision": "float32", "threads": 8, "interop_threads": 1,
                    "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                    "fixture_scope": registration["fixture_scope"], "models": rows,
                    "files": {str(p.relative_to(temporary)): sha(p) for p in sorted(temporary.rglob("*")) if p.is_file()}}
        write(temporary / "manifest.json", manifest)
        (temporary / "SHA256SUMS").write_text("".join(f"{sha(temporary / name)}  {name}\n" for name in sorted(set(manifest["files"]) | {"manifest.json"})))
        runtime.verify_bundle(temporary); verify()
        pending_archive = temporary.with_name(temporary.name + ".tar.gz")
        with tarfile.open(pending_archive, "w:gz", compresslevel=6) as archive:
            archive.add(temporary, arcname=destination.name, recursive=True)
        with tempfile.TemporaryDirectory(prefix="iws-rowwise-relocated-") as relocation:
            relocation = Path(relocation)
            with tarfile.open(pending_archive, "r:gz") as archive:
                for member in archive.getmembers():
                    p = Path(member.name)
                    if p.is_absolute() or ".." in p.parts or not (member.isfile() or member.isdir()):
                        raise ValueError("Unsafe archive inventory")
                archive.extractall(relocation, filter="data")
            relocated = relocation / destination.name; proof = relocation / "proof.json"
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
            env.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            process = subprocess.run([sys.executable, "-I", str(relocated / "runtime.py"), "--output", str(proof), "--forbid-root", str(ROOT)],
                                     cwd=relocation, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1200)
            (REPORT / "relocated_parity.log").write_text(process.stdout)
            if process.returncode: raise RuntimeError("Archive relocation proof failed; pending artifacts retained")
            result = read(proof)
            if (result.get("status") != "passed" or len(result.get("models", [])) != 36
                    or not result.get("isolated_python") or not result.get("original_workspace_reads_denied")
                    or result.get("network_attempts") != 0 or result.get("original_workspace_read_attempts") != 0
                    or result.get("manifest_sha256") != sha(temporary / "manifest.json")):
                raise ValueError("Incomplete relocated archive proof")
            write(REPORT / "relocated_cpu_parity.json", result)
        verify(); runtime.verify_bundle(temporary)
        os.rename(temporary, destination); archive = destination.with_suffix(".tar.gz"); os.rename(pending_archive, archive)
        receipt = {"schema": "iws_rowwise_release_readiness_v2", "status": "passed_local_release", "published": False,
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - began,
            "registration_sha256": sha(REG), "source_review_sha256": sha(REVIEW), "models": 36,
            "bundle": str(destination.relative_to(ROOT)), "manifest_sha256": sha(destination / "manifest.json"),
            "archive": str(archive.relative_to(ROOT)), "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
            "proof_sha256": sha(REPORT / "relocated_cpu_parity.json"), "dataset_payloads_read": 0,
            "accuracy_evaluation_performed": False, "old_bundle_manifests_unchanged": True,
            "backend": runtime.BACKEND, "threads": 8, "interop_threads": 1}
        write(REPORT / "readiness.json", receipt); print(json.dumps(receipt), flush=True); return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command", choices=("register", "verify", "build"))
    parser.add_argument("--output", type=Path, default=DEFAULT); args = parser.parse_args()
    if args.command == "register": print(json.dumps(register()))
    elif args.command == "verify": print(json.dumps({"status": "passed", "models": len(verify()["models"])}))
    else: build(args.output)
