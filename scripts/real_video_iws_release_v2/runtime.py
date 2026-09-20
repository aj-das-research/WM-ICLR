#!/usr/bin/env python3
"""Portable, explicit row-wise CPU FP32 inference for all 36 IWS predictors."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import socket
import sys

sys.dont_write_bytecode = True
BACKEND = "one_command_row_gru_cpu_fp32_v1"
SCHEMA = "iws_single_observation_rowwise_inference_v2"
TASKS = {"pusht": 4, "bimanual_box": 14, "bimanual_rope": 8}
MODES = ("autoregressive", "anchored_additive", "bounded_spatial_mix", "unbounded_spatial_mix")
KINDS = {False: "shiftwm_iws_single_observation_v1", True: "shiftwm_iws_single_observation_unbounded_v1"}
SELECTION = "internal_dev_equal_trajectory_endpoint_H60_standardized_mse"
PRECISION = "float32 validation; autocast disabled; CUDA TF32 disabled"
SOURCE_FILES = {"src/shiftwm/__init__.py", "src/shiftwm/upstream.py",
    "src/shiftwm/real_video_iws/__init__.py", "src/shiftwm/real_video_iws/model.py",
    "src/shiftwm/real_video_iws_unbounded/__init__.py", "src/shiftwm/real_video_iws_unbounded/model.py",
    "src/shiftwm/real_video_iws_reserved_recovery/__init__.py",
    "src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py",
    "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/jepa.py",
    "src/shiftwm/vendor/lewm/NOTICE.json", "src/shiftwm/vendor/lewm/LICENSE"}
RUNTIME_FILES = SOURCE_FILES | {"runtime.py", "example.py", "README.md", "MODEL_CARD.md",
    "MODEL_LICENSE.md", "LICENSE", "requirements.txt", "preprocessing.json", "provenance.json",
    "evidence/development_summary.json", "evidence/reserved_recovery_summary.json", "fixtures/references.json"}


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path): return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def regular(root, name):
    if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
            or "\\" in name or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("Unsafe bundle path")
    root = Path(root).resolve(); path = root / name
    if (not path.is_file() or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in [path, *path.parents] if p != root and p.is_relative_to(root))):
        raise ValueError("Bundle requires independent regular files: " + name)
    return path


def expected_names():
    return {f"{task}_{mode}_s{seed}" for task in TASKS for mode in MODES for seed in range(3)}


def verify_bundle(root):
    root = Path(root).resolve(); manifest = read(regular(root, "manifest.json"))
    if (manifest.get("schema") != SCHEMA or manifest.get("backend") != BACKEND
            or manifest.get("precision") != "float32" or manifest.get("device") != "cpu"
            or manifest.get("threads") != 8 or manifest.get("interop_threads") != 1
            or manifest.get("fixture_scope") != "deterministic synthetic inputs; no dataset examples or targets"):
        raise ValueError("Wrong bundle runtime or fixture contract")
    rows = manifest.get("models", [])
    if len(rows) != 36 or {r.get("name") for r in rows} != expected_names():
        raise ValueError("Complete 36-model task/mode/seed grid required")
    required = set(RUNTIME_FILES)
    for row in rows:
        if (row["name"] != f"{row['task']}_{row['mode']}_s{row['seed']}"
                or row["directory"] != "models/" + row["name"]
                or type(row["seed"]) is not int or row["seed"] not in range(3)
                or row["package_kind"] != KINDS[row["mode"] == "unbounded_spatial_mix"]
                or row.get("completed_epochs") != 30):
            raise ValueError("Model task/objective/seed/package identity differs")
        required.update(row["directory"] + "/" + name for name in ("model.pt", "config.json", "package_manifest.json"))
        required.add("model_cards/" + row["name"] + ".md")
        required.add("fixtures/" + row["name"] + "_b1_prediction.npz")
    if set(manifest.get("files", {})) != required:
        raise ValueError("Incomplete runtime/model/source/reference inventory")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != required | {"manifest.json", "SHA256SUMS"}:
        raise ValueError("Missing, unlisted, optimizer or symlink payload")
    for name, value in manifest["files"].items():
        if sha(regular(root, name)) != value: raise ValueError("Bundle digest differs: " + name)
    checksums = "".join(f"{sha(regular(root, name))}  {name}\n" for name in sorted(required | {"manifest.json"}))
    if regular(root, "SHA256SUMS").read_text() != checksums:
        raise ValueError("Public checksum inventory differs")
    return manifest


def configure_runtime():
    import torch
    torch.set_num_threads(8)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify_import_locations(root):
    root = Path(root).resolve()
    for name, loaded in list(sys.modules.items()):
        if (name == "shiftwm" or name.startswith("shiftwm.") or name.startswith("shiftwm_upstream")) and getattr(loaded, "__file__", None):
            if not Path(loaded.__file__).resolve().is_relative_to(root / "src"):
                raise ValueError("Use a fresh Python process: cached source module outside this bundle: " + name)


class Predictor:
    """Feature-only inference with explicit CPU/FP32/eight-thread enforcement."""
    def __init__(self, model, metadata): self.model, self.metadata = model, metadata

    def predict(self, initial_features, commands):
        import torch
        if (torch.get_num_threads() != 8 or torch.get_num_interop_threads() != 1
                or self.model.training or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in self.model.parameters())
                or initial_features.device.type != "cpu" or commands.device.type != "cpu"
                or initial_features.dtype != torch.float32 or commands.dtype != torch.float32):
            raise ValueError("This release is verified only for eval-mode CPU FP32, 8 threads, interop 1")
        with torch.inference_mode(): return self.model.predict(initial_features, commands)


def load_model(name, bundle_root=None):
    """Return (Predictor, metadata); inputs [B,6144] and [B,H,A], output [B,H-1,6144]."""
    root = Path(bundle_root or Path(__file__).parent).resolve()
    manifest = verify_bundle(root)
    if manifest.get("backend") != BACKEND: raise ValueError("Wrong numerical backend")
    row = next((r for r in manifest["models"] if r["name"] == name), None)
    if row is None: raise ValueError("Unknown released model")
    path = root / row["directory"]
    if {p.name for p in path.iterdir()} != {"model.pt", "config.json", "package_manifest.json"}:
        raise ValueError("Inference package must exclude training state")
    package = read(regular(root, row["directory"] + "/package_manifest.json"))
    if (package.get("package_kind") != row["package_kind"] or package.get("format_version") != 1
            or set(package.get("files", {})) != {"model.pt", "config.json"}):
        raise ValueError("Wrong weights-only package")
    for filename, value in package["files"].items():
        if sha(regular(root, row["directory"] + "/" + filename)) != value:
            raise ValueError("Selected package digest differs")
    if sha(path / "model.pt") != row["checkpoint_sha256"] or sha(path / "config.json") != row["config_sha256"]:
        raise ValueError("Selected recovery checkpoint/config identity differs")
    configure_runtime()
    verify_import_locations(root)
    sys.path.insert(0, str(root / "src"))
    import torch
    namespace = "real_video_iws_unbounded" if row["mode"] == "unbounded_spatial_mix" else "real_video_iws"
    module = importlib.import_module("shiftwm." + namespace + ".model")
    backend = importlib.import_module("shiftwm.real_video_iws_reserved_recovery.prefix_backend")
    for current, relative in ((module, f"src/shiftwm/{namespace}/model.py"),
                             (backend, "src/shiftwm/real_video_iws_reserved_recovery/prefix_backend.py")):
        if Path(current.__file__).resolve() != root / relative:
            raise ValueError("Use a fresh Python process: imported model/backend is outside this bundle")
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    config = read(path / "config.json"); metadata = config.get("metadata", {})
    if (state.get("config") != config or config.get("package_kind") != row["package_kind"]
            or type(state.get("epoch")) is not int or state["epoch"] != row["selected_epoch"]
            or not 1 <= state["epoch"] <= 30 or metadata.get("selection") != SELECTION
            or metadata.get("validation_precision") != PRECISION
            or metadata.get("training_identity") != digest(metadata.get("identity"))
            or metadata.get("training_identity") != row["training_identity"]):
        raise ValueError("Selected package identity differs")
    recipe = metadata["identity"]["scientific_config"]
    if (any(recipe.get(k) != row[k] for k in ("task", "mode", "seed"))
            or config["model_config"]["mode"] != row["mode"]
            or config["model_config"]["action_dim"] != TASKS[row["task"]]):
        raise ValueError("Selected weights belong to another task/mode/seed")
    model = module.from_config(config); model.load_state_dict(state["state_dict"], strict=True)
    verify_import_locations(root)
    if any(config.get(k) != v for k, v in model.package_config.items()):
        raise ValueError("Reconstructed configuration or normalization differs")
    if model.parameter_counts != metadata.get("parameter_counts"):
        raise ValueError("Parameter counts differ")
    backend.install_rowwise_gru(model.cpu().eval())
    info = {"name": name, "epoch": state["epoch"], "checkpoint_sha256": sha(path / "model.pt"),
            "backend": BACKEND, "device": "cpu", "precision": "float32", "threads": 8, "interop_threads": 1,
            "torch_version": str(torch.__version__), "selected_package_kind": row["package_kind"]}
    return Predictor(model, info), info


def synthetic_inputs(task, batch_size):
    """Fixed public synthetic values, independent of every dataset and statistic."""
    import numpy as np
    if task not in TASKS or batch_size not in (1, 8, 64): raise ValueError("Unregistered synthetic fixture")
    rng = np.random.Generator(np.random.PCG64(20260920 + list(TASKS).index(task)))
    initial = rng.uniform(-0.5, 0.5, (64, 6144)).astype(np.float32)
    commands = rng.uniform(-0.25, 0.25, (64, 60, TASKS[task])).astype(np.float32)
    return initial[:batch_size].copy(), commands[:batch_size].copy()


def array_sha(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prefix_checks(predictor, initial, commands, full):
    import torch
    checks = []
    for horizon in (15, 30, 45):
        short = predictor.predict(initial, commands[:, :horizon])
        reference = full[:, :horizon - 1]
        gap = (short - reference).abs(); tolerance = 2e-5 + 1e-5 * reference.abs()
        passed = bool(torch.allclose(short, reference, rtol=1e-5, atol=2e-5))
        if not passed or not torch.isfinite(gap).all(): raise ValueError("Synthetic prefix gate failed")
        checks.append({"horizon": horizon, "allclose": passed, "maximum_absolute_difference": float(gap.max()),
                       "maximum_tolerance_ratio": float((gap / tolerance).max()),
                       "bitwise_equal": bool(torch.equal(short, reference))})
    return checks


def verify_offline(root, output, forbid_root=None):
    attempts, denied_reads = [], []
    def denied(*args, **kwargs):
        attempts.append("network"); raise RuntimeError("Network forbidden in relocation proof")
    socket.socket.connect = denied; socket.socket.connect_ex = denied
    socket.create_connection = denied; socket.getaddrinfo = denied
    if forbid_root:
        forbidden = Path(forbid_root).resolve(); environment = Path(sys.prefix).resolve()
        def audit(event, args):
            if event == "open" and isinstance(args[0], (str, bytes)):
                path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
                if path.is_relative_to(forbidden) and not path.is_relative_to(environment):
                    denied_reads.append(str(path)); raise RuntimeError("Original workspace read forbidden")
        sys.addaudithook(audit)
    import numpy as np
    import torch
    configure_runtime(); root = Path(root).resolve()
    if Path(output).resolve().is_relative_to(root): raise ValueError("Proof must stay outside immutable bundle")
    manifest = verify_bundle(root); references = read(regular(root, "fixtures/references.json"))
    results = []
    for row in manifest["models"]:
        model, identity = load_model(row["name"], root)
        cases = []
        for batch_size in (1, 64, 8):
            initial, commands = synthetic_inputs(row["task"], batch_size)
            x, u = torch.from_numpy(initial), torch.from_numpy(commands)
            actual = model.predict(x, u)
            expected = references[row["name"]][str(batch_size)]
            if (list(actual.shape) != [batch_size, 59, 6144] or not torch.isfinite(actual).all()
                    or array_sha(actual.numpy()) != expected["prediction_sha256"]
                    or array_sha(initial) != expected["initial_sha256"]
                    or array_sha(commands) != expected["commands_sha256"]):
                raise ValueError("Relocated output/input bytes differ: " + row["name"])
            if batch_size == 1:
                with np.load(regular(root, "fixtures/" + row["name"] + "_b1_prediction.npz"), allow_pickle=False) as f:
                    if set(f.files) != {"predictions"} or not np.array_equal(actual.numpy(), f["predictions"]):
                        raise ValueError("B1 numerical reference differs")
            cases.append({"batch_size": batch_size, "shape": list(actual.shape),
                          "prediction_sha256": expected["prediction_sha256"],
                          "maximum_absolute_difference": 0., "byte_identity_verified": True,
                          "prefix_checks": prefix_checks(model, x, u, actual)})
        results.append({**identity, "status": "passed", "cases": cases})
        print(row["name"] + ": exact synthetic relocation, batches1/64/8 and all prefixes", flush=True)
    verify_bundle(root)
    if attempts or denied_reads: raise ValueError("Network or original workspace access attempted")
    result = {"schema": "iws_rowwise_relocation_proof_v2", "status": "passed", "models": results,
              "manifest_sha256": sha(root / "manifest.json"), "backend": BACKEND,
              "isolated_python": bool(sys.flags.isolated), "network_attempts": len(attempts),
              "original_workspace_reads_denied": bool(forbid_root), "original_workspace_read_attempts": len(denied_reads),
              "fixture_scope": manifest["fixture_scope"], "dataset_payloads_read": 0,
              "accuracy_evaluation_performed": False, "device": "cpu", "precision": "float32",
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbid-root", type=Path)
    args = parser.parse_args(); verify_offline(args.bundle, args.output, args.forbid_root)
