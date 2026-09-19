#!/usr/bin/env python3
"""Strict offline feature-input inference for the separate IWS model packages.

This file is copied into the inference bundle. Frozen model/vendor sources are
copied byte-for-byte; no training or dataset module is imported by this runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import socket
import sys

sys.dont_write_bytecode = True  # Immutable bundles must not acquire unlisted code.

KIND = "shiftwm_iws_single_observation_v1"
SELECTION = "internal_dev_equal_trajectory_endpoint_H60_standardized_mse"
PRECISION = "float32 validation; autocast disabled; CUDA TF32 disabled"
RUNTIME_FILES = {"runtime.py", "README.md", "requirements.txt", "LICENSE",
                 "src/shiftwm/__init__.py", "src/shiftwm/upstream.py",
                 "src/shiftwm/real_video_iws/__init__.py", "src/shiftwm/real_video_iws/model.py",
                 "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/jepa.py",
                 "src/shiftwm/vendor/lewm/NOTICE.json", "src/shiftwm/vendor/lewm/LICENSE"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def regular(root, name):
    parts = PurePosixPath(name)
    if (not isinstance(name, str) or not name or parts.is_absolute()
            or "\\" in name or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("Unsafe bundle path")
    root = Path(root).resolve()
    target = root / name
    if (not target.is_file() or not target.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in [target, *target.parents] if p != root and p.is_relative_to(root))):
        raise ValueError("Bundle needs independent regular files: " + name)
    return target


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = read(regular(root, "manifest.json"))
    if (manifest.get("schema") != "iws_local_inference_bundle_v1"
            or manifest.get("package_kind") != KIND
            or manifest.get("benchmark_finalization_claimed") is not False):
        raise ValueError("Wrong inference bundle schema/scope")
    expected = {f"{t}_{m}_s{s}" for t in ("pusht", "bimanual_box", "bimanual_rope")
                for m in ("autoregressive", "anchored_additive", "bounded_spatial_mix") for s in range(3)}
    rows = manifest.get("models", [])
    if len(rows) != 27 or {r.get("name") for r in rows} != expected:
        raise ValueError("All 27 registered models are required")
    required_files = set(RUNTIME_FILES)
    for row in rows:
        if (row.get("name") != f"{row.get('task')}_{row.get('mode')}_s{row.get('seed')}"
                or row.get("directory") != "models/" + row["name"]
                or row.get("fixture") != "fixtures/" + row["task"] + ".npz"
                or row.get("expected_prediction") != "fixtures/" + row["name"] + "_prediction.npz"):
            raise ValueError("Task/arm/seed label differs from package path")
        required_files.update({row["directory"] + "/" + name
                               for name in ("model.pt", "config.json", "package_manifest.json")})
        required_files.update({row["fixture"], row["expected_prediction"]})
    if set(manifest.get("files", {})) != required_files:
        raise ValueError("Manifest must cover every runtime/package/input/reference file")
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual_files != required_files | {"manifest.json"}:
        raise ValueError("Unlisted or missing bundle payload")
    for name, value in manifest["files"].items():
        if sha(regular(root, name)) != value:
            raise ValueError("Bundle hash mismatch: " + name)
    if any(p.name == "training_state.pt" or p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Optimizer/RNG state or links in inference bundle")
    return manifest


def load_model(directory, bundle_root=None):
    """Return (eval-mode model, metadata); CPU FP32 is the verified backend.

    model.predict(initial_features[B,6144], commands[B,H,A]) -> [B,H-1,6144]
    accepts raw frozen-DINO features and native task command rows. There is no
    target, history, RGB-decoder, online adaptation, or simulator interface.
    """
    root = Path(bundle_root or Path(__file__).parent).resolve()
    manifest = verify_bundle(root)
    path = Path(directory)
    path = (root / path).absolute() if not path.is_absolute() else path.absolute()
    row = next((r for r in manifest["models"] if root / r["directory"] == path), None)
    if row is None:
        raise ValueError("Package is not in this complete inference bundle")
    # Authorize the canonical in-bundle path before inspecting package contents.
    regular(root, row["directory"] + "/package_manifest.json")
    path = root / row["directory"]
    if {p.name for p in path.iterdir()} != {"model.pt", "config.json", "package_manifest.json"}:
        raise ValueError("Inference package has unexpected files")
    package = read(path / "package_manifest.json")
    if (package.get("package_kind") != KIND or package.get("format_version") != 1
            or set(package.get("files", {})) != {"model.pt", "config.json"}):
        raise ValueError("Wrong strict inference package")
    for name, value in package["files"].items():
        if sha(regular(root, str((path / name).relative_to(root)))) != value:
            raise ValueError("Package digest differs")
    sys.path.insert(0, str(root / "src"))
    import torch
    from shiftwm.real_video_iws import model as module
    if Path(module.__file__).resolve() != root / "src/shiftwm/real_video_iws/model.py":
        raise ValueError("Model was imported outside relocated bundle")
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    config = read(path / "config.json")
    metadata = config.get("metadata", {})
    if (state.get("config") != config or config.get("package_kind") != KIND
            or type(state.get("epoch")) is not int or state["epoch"] != row["selected_epoch"]
            or not 1 <= state["epoch"] <= 30
            or metadata.get("selection") != SELECTION
            or metadata.get("validation_precision") != PRECISION
            or metadata.get("training_identity") != digest(metadata.get("identity"))
            or metadata.get("training_identity") != row["training_identity"]):
        raise ValueError("Selected IWS package identity differs")
    recipe = metadata["identity"]["scientific_config"]
    if (any(recipe.get(k) != row[k] for k in ("task", "mode", "seed"))
            or config["model_config"]["mode"] != row["mode"]
            or config["model_config"]["action_dim"] != {"pusht":4, "bimanual_box":14, "bimanual_rope":8}[row["task"]]):
        raise ValueError("Weights belong to a different task/arm/seed")
    model = module.from_config(config)
    model.load_state_dict(state["state_dict"], strict=True)
    if any(config.get(k) != v for k, v in model.package_config.items()):
        raise ValueError("Reconstructed model recipe/statistics differ")
    if model.parameter_counts != metadata.get("parameter_counts"):
        raise ValueError("Parameter counts differ")
    return model.cpu().eval(), {"epoch": state["epoch"], "name": row["name"],
                               "checkpoint_sha256": sha(path / "model.pt")}


def verify_offline(root, output):
    """Run in a fresh `python -I` process from a copied directory."""
    attempts = []
    def denied(*args, **kwargs):
        attempts.append("network")
        raise RuntimeError("Network forbidden during relocated inference proof")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    socket.getaddrinfo = denied
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(root).resolve()
    manifest = verify_bundle(root)
    results = []
    for row in manifest["models"]:
        model, identity = load_model(row["directory"], root)
        with np.load(regular(root, row["fixture"]), allow_pickle=False) as fixture:
            if set(fixture.files) != {"initial_features", "commands"}:
                raise ValueError("Parity fixtures must contain inputs only")
            initial = torch.from_numpy(fixture["initial_features"].copy())
            commands = torch.from_numpy(fixture["commands"].copy())
        with torch.inference_mode():
            actual = model.predict(initial, commands).numpy()
        with np.load(regular(root, row["expected_prediction"]), allow_pickle=False) as f:
            expected = f["predictions"]
        if (actual.shape != (1, 59, 6144) or not np.isfinite(actual).all()
                or not np.array_equal(actual, expected)):
            raise ValueError("Relocated H60 CPU prediction differs: " + row["name"])
        results.append({**identity, "status": "passed", "shape": list(actual.shape),
                        "maximum_absolute_difference": float(np.max(np.abs(actual-expected)))})
        print(row["name"] + ": exact CPU parity", flush=True)
    if attempts:
        raise ValueError("A network request was attempted")
    imported = {name: str(Path(mod.__file__).resolve().relative_to(root))
                for name, mod in list(sys.modules.items())
                if (name == "shiftwm" or name.startswith("shiftwm.") or name.startswith("shiftwm_upstream"))
                and getattr(mod, "__file__", None)}
    result = {"status": "passed", "schema": "iws_relocated_cpu_parity_v1",
              "models": results, "device": "cpu", "precision": "float32", "threads": 2,
              "isolated_python": bool(sys.flags.isolated), "network_attempts": len(attempts),
              "bundle_directory": str(root), "manifest_sha256": sha(root / "manifest.json"),
              "input_scope": "one fixed internal-training example per task; no targets",
              "official_validation_payloads_read": 0, "benchmark_accuracy_computed": False,
              "model_imports": imported, "torch_version": str(torch.__version__),
              "numpy_version": np.__version__}
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_offline(args.bundle, args.output)
