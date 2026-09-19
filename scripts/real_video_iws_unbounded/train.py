#!/usr/bin/env python3
"""Train/profile the new unbounded-innovation ablation with frozen v1 utilities."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws_unbounded.model import SingleObservationWorldModel
from shiftwm.real_video_iws_unbounded import training


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


legacy = _load(ROOT / "scripts/real_video_iws/train.py", "_unbounded_private_frozen_runner")
_original_module = legacy._module


def _module(name, relative):
    if relative == "scripts/real_video_iws/campaign.py":
        return _load(Path(__file__).with_name("campaign.py"), "_unbounded_current_campaign")
    return _original_module(name, relative)


def source_files():
    campaign = _module("unused", "scripts/real_video_iws/campaign.py")
    return campaign.source_dependencies()


legacy.SingleObservationWorldModel = SingleObservationWorldModel
legacy.engine = training.engine
legacy.PACKAGE_KIND = training.PACKAGE_KIND
legacy._module = _module
legacy.source_files = source_files
for _name in ("read_package", "load_package", "validate_completed", "export_inference_package"):
    setattr(legacy, _name, getattr(training, _name))

_original_train = legacy.train
_original_profile = legacy.profile


def train(config_path, task, mode, seed, output, resume=False, max_runtime_seconds=None):
    campaign = _module("unused", "scripts/real_video_iws/campaign.py")
    registration = campaign.check_registration(config_path)
    row = next((r for r in registration["runs"]
                if (r["task"], r["mode"], r["seed"]) == (task, mode, seed)), None)
    if row is None or Path(output).resolve() != ROOT / row["output"]:
        raise ValueError("Training output must equal the registered new-namespace run path")
    return _original_train(config_path, task, mode, seed, output, resume, max_runtime_seconds)


def profile(config_path, task, mode, microbatch, output, updates=2):
    if not Path(output).resolve().is_relative_to(ROOT / "reports/real_video_iws_unbounded"):
        raise ValueError("Resource profiles must stay in the new ablation report namespace")
    return _original_profile(config_path, task, mode, microbatch, output, updates)


legacy.train, legacy.profile = train, profile

PACKAGE_KIND, SELECTION = training.PACKAGE_KIND, training.SELECTION
read_package, load_package = legacy.read_package, legacy.load_package
validate_completed, export_inference_package = legacy.validate_completed, legacy.export_inference_package
open_cache = legacy.open_cache
allocated_device = legacy.allocated_device

if __name__ == "__main__":
    raise SystemExit(legacy.main())
