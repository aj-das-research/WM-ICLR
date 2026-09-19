#!/usr/bin/env python3
"""Private reuse of the frozen spatial full-30 atomic/resumable trainer."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video.data import sha256
from shiftwm.real_video_spatial_components.model import ComponentWorldModel, from_config, PACKAGE_KIND, MODES


def private_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


legacy = private_module(ROOT / "scripts/real_video_spatial/train.py", "component_private_spatial_train")
original_source_files = legacy.source_files
base = legacy.base
legacy.SpatialWorldModel = ComponentWorldModel
base.RealVideoWorldModel = ComponentWorldModel
base.from_config = from_config
base.PACKAGE_KIND = PACKAGE_KIND


def source_files():
    paths = [ROOT / p for p in (
        "src/shiftwm/real_video_spatial_components/__init__.py",
        "src/shiftwm/real_video_spatial_components/model.py",
        "scripts/real_video_spatial_components/train.py",
        "scripts/real_video_spatial_components/evaluate.py",
        "scripts/real_video_spatial_components/campaign.py",
        "scripts/real_video_spatial_components/analysis.py",
        "scripts/real_video_spatial_components/inference.py",
        "scripts/real_video_spatial_components/train.slurm",
        "scripts/real_video_spatial_components/finalize.slurm",
        "reports/real_video_development/spatial_components_protocol.md",
        "tests/test_real_video_spatial_components.py",
    )]
    registry = ROOT / "configs/real_video_spatial_components/v1/registration.json"
    if registry.exists():
        paths.append(registry)
    return {**original_source_files(), **{str(p): sha256(p) for p in paths}}


base.source_files = source_files


def train(config):
    if config.get("mode") not in MODES or config.get("component_schema") != "observed_anchor_mixing_x_innovation_bound_v1":
        raise ValueError("Not a registered component training configuration")
    return legacy.train(config)


epoch_pass = legacy.epoch_pass
load_package = base.load_package
read_package = base.read_package
save_package = base.save_package
validate_completed = base.validate_completed
atomic_json = base.atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-if-present", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config["resume_if_present"] = args.resume_if_present
    result = train(config)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "completed" else 75)
