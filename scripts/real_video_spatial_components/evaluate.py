#!/usr/bin/env python3
"""Reuse exact frozen spatial scoring, including original cached 2x2 targets."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video.data import sha256
from shiftwm.real_video_spatial_components.model import PACKAGE_KIND, MODES


def private(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


train = private(Path(__file__).with_name("train.py"), "components_evaluation_private_train")
core = private(ROOT / "scripts/real_video_spatial/evaluate.py", "components_private_evaluator")
core.train = train
ledger = private(ROOT / "scripts/real_video_spatial/validate_ledger.py", "components_private_ledger")
ledger.MODES = ledger.MODES + MODES


def validate(result, config, selected_state):
    if (result.get("component_package_kind") != PACKAGE_KIND
            or result.get("component_wrapper_sha256") != sha256(__file__)
            or result.get("component_registration_sha256") != sha256(ROOT / "configs/real_video_spatial_components/v1/registration.json")
            or selected_state["config"].get("package_kind") != PACKAGE_KIND):
        raise ValueError("Component evaluation identity differs")
    return ledger.validate_ledger(result, config, ROOT, selected_state)


def evaluate(config, output):
    result = core.evaluate(config, output)
    result.update(component_package_kind=PACKAGE_KIND, component_wrapper_sha256=sha256(__file__),
                  component_registration_sha256=sha256(ROOT / "configs/real_video_spatial_components/v1/registration.json"))
    _, selected = train.read_package(ROOT / config["output_dir"] / "best")
    result["summary"] = validate(result, config, selected)
    train.atomic_json(result, output)
    return result
