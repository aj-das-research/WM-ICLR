#!/usr/bin/env python3
"""Reuse the exact spatial ledger computation with external checkpoint loading."""
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def local(name):
    spec = importlib.util.spec_from_file_location("external_dinowm_eval_" + name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def evaluate(config, output):
    registry = local("registry"); registry.verify(); registry.validate_config(config)
    train = local("train")
    train.base.seed_everything(config["seed"])
    train.validate_completed(config["output_dir"])
    upstream_path = ROOT / "scripts/real_video_spatial/evaluate.py"
    spec = importlib.util.spec_from_file_location("external_dinowm_reused_spatial_evaluator", upstream_path)
    upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    upstream.train = train  # Only package/model loading differs; metric arithmetic is unchanged.
    with tempfile.TemporaryDirectory(prefix="external-dinowm-evaluation-") as temporary:
        result = upstream.evaluate(config, Path(temporary) / "validation.json")
    _, state = train.read_package(Path(config["output_dir"]) / "best")
    result.update(schema="adapted_official_dinowm_raw_droid_validation_v2",
                  run_name=f"{config['mode']}_s{config['seed']}",
                  registration_sha256=registry.sha(registry.REG),
                  config_sha256=registry.sha(HERE / "configs" / f"{config['mode']}_s{config['seed']}.json"),
                  training_identity=state["config"]["metadata"]["training_identity"],
                  normalization_sha256=registry.sha(ROOT / config["cache_root"] / "training_statistics.json"),
                  upstream_evaluation_source_sha256=registry.sha(upstream_path),
                  source_sha256=registry.sha(Path(__file__)),
                  training_objective=config["mode"],
                  selection_metric=train.base.SELECTION,
                  evaluation_precision=train.base.PRECISION,
                  model_family="adapted_official_dinowm_raw_v2")
    registry.verify()
    train.atomic_json(result, output)
    return result
