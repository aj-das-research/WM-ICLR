"""Atomic portable model packages and resumable weights-only training state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], torch.from_numpy(state[1].copy().astype(np.int64)),
                                                  state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    arr = state["numpy"]
    np.random.set_state((arr[0], arr[1].cpu().numpy().astype(np.uint32), arr[2], arr[3], arr[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_package(model, directory, *, optimizer=None, scheduler=None, epoch=0, step=0,
                 best_metric=float("inf"), metadata=None, loader_generator=None, progress=None):
    """Save tensors/config, never a pickled Python model instance."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {"format_version": 1, **model.export_config(), "metadata": metadata or {}}
    state = {"config": config, "state_dict": model.state_dict(), "epoch": epoch, "step": step,
             "best_metric": best_metric, "progress": progress or {}}
    atomic_torch_save(state, directory / "model.pt")
    if optimizer is not None:
        resume = {"optimizer": optimizer.state_dict(), "rng": rng_state(),
                  "scheduler": scheduler.state_dict() if scheduler else None,
                  "loader_generator": loader_generator.get_state() if loader_generator else None,
                  "epoch": epoch, "step": step}
        atomic_torch_save(resume, directory / "training_state.pt")
    tmp = directory / "config.json.tmp"
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(tmp, directory / "config.json")


def load_package(directory, device="cpu"):
    from .model import ShiftWorldModel
    from .upstream import create_base
    path = Path(directory)
    model_file = path if path.is_file() else path / "model.pt"
    state = torch.load(model_file, map_location="cpu", weights_only=True)
    config = state["config"]
    if config["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    base = create_base(config["base_config"])
    model = ShiftWorldModel(base, config["base_config"], config["model_config"],
                            config["action_mean"], config["action_std"], config["provenance"])
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), state


def resume_training(directory, model, optimizer, scheduler=None, loader_generator=None):
    directory = Path(directory)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    training = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    if state["epoch"] != training["epoch"] or state["step"] != training["step"]:
        raise RuntimeError("Checkpoint pair interrupted during saving: model/training steps differ")
    model.load_state_dict(state["state_dict"], strict=True)
    optimizer.load_state_dict(training["optimizer"])
    if scheduler and training["scheduler"] is not None:
        scheduler.load_state_dict(training["scheduler"])
    if loader_generator and training["loader_generator"] is not None:
        loader_generator.set_state(training["loader_generator"])
    restore_rng(training["rng"])
    return state
