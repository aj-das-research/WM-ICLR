"""Offline extension packages with atomic model/optimizer generation commits.

``best`` and ``last`` are relative symlinks to immutable package generations.
Readers resolve the link once, then validate every listed file before loading.
A copied real package directory is equally valid; network access is never used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import torch

from shiftwm.checkpoint import atomic_torch_save, rng_state, restore_rng


PACKAGE_KIND = "shiftwm_domain_predictor_extension_v1"
FORMAT_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def package_configuration(model, metadata=None):
    return {**deepcopy(model.package_config), "format_version": FORMAT_VERSION,
            "package_kind": PACKAGE_KIND, "metadata": deepcopy(metadata or {})}


def save_package(model, directory, *, optimizer=None, scheduler=None, epoch=0, step=0,
                 best_metric=float("inf"), metadata=None, loader_generator=None, progress=None):
    """Commit all package files together, retaining the previous generation.

    Completed packages contain plain tensor state dictionaries, not pickled
    model objects. A crash before pointer replacement leaves the prior complete
    package usable. A lone unreferenced temporary generation is never loaded.
    """
    directory = Path(directory).absolute()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists() and not directory.is_symlink():
        raise ValueError("Refusing to replace an existing non-generation package directory")
    previous = directory.resolve() if directory.is_symlink() else None
    generations = directory.parent / ("." + directory.name + ".generations")
    generations.mkdir(exist_ok=True)
    generation = generations / uuid.uuid4().hex
    generation.mkdir()
    pointer = directory.with_name("." + directory.name + "." + uuid.uuid4().hex + ".tmp")
    committed = False
    try:
        config = package_configuration(model, metadata)
        state = {"config": config, "state_dict": model.state_dict(), "epoch": epoch,
                 "step": step, "best_metric": best_metric, "progress": progress or {}}
        atomic_torch_save(state, generation / "model.pt")
        atomic_json(config, generation / "config.json")
        if optimizer is not None:
            resume = {"optimizer": optimizer.state_dict(), "rng": rng_state(),
                      "scheduler": scheduler.state_dict() if scheduler is not None else None,
                      "loader_generator": loader_generator.get_state() if loader_generator is not None else None,
                      "epoch": epoch, "step": step,
                      "training_identity": config["metadata"].get("training_identity")}
            atomic_torch_save(resume, generation / "training_state.pt")
        atomic_json({"format_version": FORMAT_VERSION, "package_kind": PACKAGE_KIND,
                     "files": {p.name: file_sha256(p) for p in sorted(generation.iterdir())}},
                    generation / "package_manifest.json")
        os.symlink(os.path.relpath(generation, directory.parent), pointer)
        os.replace(pointer, directory)
        committed = True
        # A pinned reader of the preceding generation remains valid. Older
        # generations are obsolete snapshots, not independent trained models.
        for old in generations.iterdir():
            if old.is_dir() and old not in {generation, previous}:
                shutil.rmtree(old)
        return state
    finally:
        pointer.unlink(missing_ok=True)
        if not committed:
            shutil.rmtree(generation, ignore_errors=True)


def read_package(directory, *, require_training=False):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / "package_manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("package_kind") != PACKAGE_KIND:
        raise ValueError("Unsupported extension package manifest")
    names = set(manifest.get("files", {}))
    if not {"model.pt", "config.json"}.issubset(names) or names - {"model.pt", "config.json", "training_state.pt"}:
        raise ValueError("Extension package manifest has missing or unexpected files")
    if require_training and "training_state.pt" not in names:
        raise ValueError("Extension package lacks optimizer/RNG continuation state")
    for name, expected in manifest["files"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"Extension package file hash differs: {name}")
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    config = json.loads((directory / "config.json").read_text())
    if (state.get("config") != config or config.get("format_version") != FORMAT_VERSION
            or config.get("package_kind") != PACKAGE_KIND):
        raise ValueError("Embedded/external extension configurations differ")
    if any(type(state.get(name)) is not int or state[name] < 0 for name in ("epoch", "step")):
        raise ValueError("Invalid extension checkpoint epoch/step")
    if require_training:
        training = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
        if (any(training.get(name) != state[name] for name in ("epoch", "step"))
                or training.get("training_identity") != config["metadata"].get("training_identity")):
            raise ValueError("Extension model and continuation state disagree")
    return directory, state


def load_package(directory, device="cpu"):
    from .model import build_from_config

    _, state = read_package(directory)
    model = build_from_config(state["config"])
    model.load_state_dict(state["state_dict"], strict=True)
    # Reject aliases silently normalized by a constructor, including changed
    # method mode, action buffers, objective or predictor initialization.
    for key, value in model.package_config.items():
        if state["config"].get(key) != value:
            raise ValueError(f"Reconstructed extension configuration differs: {key}")
    return model.to(device).eval(), state


def resume_training(directory, model, optimizer, scheduler=None, loader_generator=None):
    path, state = read_package(directory, require_training=True)
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    if (any(state[name] != training.get(name) for name in ("epoch", "step"))
            or training.get("training_identity") != state["config"]["metadata"].get("training_identity")):
        raise ValueError("Extension model and continuation state disagree")
    for key, value in model.package_config.items():
        if state["config"].get(key) != value:
            raise ValueError(f"Resume model configuration differs: {key}")
    model.load_state_dict(state["state_dict"], strict=True)
    optimizer.load_state_dict(training["optimizer"])
    if scheduler is not None:
        if training["scheduler"] is None:
            raise ValueError("Resume lacks scheduler state")
        scheduler.load_state_dict(training["scheduler"])
    if loader_generator is not None:
        if training["loader_generator"] is None:
            raise ValueError("Resume lacks sampler state")
        loader_generator.set_state(training["loader_generator"])
    restore_rng(training["rng"])
    return state
