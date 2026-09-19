"""Strict loader for losslessly deduplicated simulator inference releases.

The original training package remains immutable. This separate format contains
inference tensors only and restores shared tensors before the established model
constructor is called. Historical paths in the configuration are never opened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

KIND = "shiftwm_simulator_shared_inference_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe release-relative file")
    candidate = root / relative
    if any(p.is_symlink() for p in (candidate, *candidate.parents) if p != root.parent):
        raise ValueError("Symlinks are not permitted in inference packages")
    if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError("Missing or escaping release file")
    return candidate


def merge_tensors(common, delta):
    if not isinstance(common, dict) or not isinstance(delta, dict) or set(common) & set(delta):
        raise ValueError("Common and model-specific tensors must be disjoint mappings")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
           for mapping in (common, delta) for k, v in mapping.items()):
        raise ValueError("State dictionaries may contain named tensors only")
    return {**common, **delta}


def read_package(root, model_id):
    root = Path(root).resolve()
    registry = json.loads(contained_file(root, "models.json").read_text())
    if registry.get("kind") != KIND or registry.get("format_version") != 1:
        raise ValueError("Unsupported simulator release contract")
    records = registry.get("models", [])
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids) or model_id not in ids:
        raise ValueError("Unknown or duplicated model identity")
    record = records[ids.index(model_id)]
    paths = {}
    for key, specification in (("common", registry["shared"]), ("delta", record["delta"]),
                               ("config", record["config"])):
        path = contained_file(root, specification["path"])
        if sha256(path) != specification["sha256"]:
            raise ValueError("Inference artifact checksum mismatch: " + key)
        paths[key] = path
    common = torch.load(paths["common"], map_location="cpu", weights_only=True)
    payload = torch.load(paths["delta"], map_location="cpu", weights_only=True)
    if set(payload) != {"state_dict", "config", "epoch", "step", "best_metric"}:
        raise ValueError("Unexpected inference payload fields")
    config = json.loads(paths["config"].read_text())
    if payload["config"] != config or payload["epoch"] != record["selected_epoch"]:
        raise ValueError("Embedded configuration/selection differs from manifest")
    if type(payload["epoch"]) is not int or not 1 <= payload["epoch"] <= 30:
        raise ValueError("Invalid selected epoch")
    merged = merge_tensors(common, payload["state_dict"])
    if len(merged) != record["state_tensor_count"] or len(common) != registry["shared"]["tensor_count"]:
        raise ValueError("State tensor inventory mismatch")
    payload["state_dict"] = merged
    return payload, record


def load_package(root, model_id, device="cpu"):
    payload, record = read_package(root, model_id)
    config = payload["config"]
    if config.get("extension_format_version") == 1:
        from .extensions.model import build_from_config
        expected_family = "original"
    elif config.get("extension_format_version") == 2:
        from .extensions.geometry_revision import build_from_config
        expected_family = "geometry"
    else:
        raise ValueError("Unsupported underlying simulator model")
    if record["family"] != expected_family:
        raise ValueError("Model family differs from its architecture configuration")
    model = build_from_config(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    for key, value in model.package_config.items():
        if config.get(key) != value:
            raise ValueError("Reconstructed model contract differs: " + key)
    return model.to(device).eval(), payload
