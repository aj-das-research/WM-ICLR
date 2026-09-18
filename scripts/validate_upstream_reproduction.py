#!/usr/bin/env python3
"""Validate historical imports and released-model CPU costs, without planning."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/upstream_planning_reproduction"
sys.path[:0] = [str(BUNDLE / "stable-worldmodel"), str(BUNDLE / "le-wm")]
os.environ["STABLEWM_HOME"] = str(BUNDLE / "cache")

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import sklearn.preprocessing
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    torch.set_num_threads(2)
    assert Path(swm.__file__).resolve().is_relative_to(BUNDLE / "stable-worldmodel")
    from stable_worldmodel.solver import CEMSolver
    import inspect
    assert "model" in inspect.signature(CEMSolver).parameters
    preflight = json.loads((ROOT / "reports/evidence/upstream_planning_preflight.json").read_text())
    result = {"device": "cpu", "gpu_evaluation_started": False, "simulator_execution_started": False,
              "stable_worldmodel_import": swm.__file__, "stable_pretraining_import": spt.__file__,
              "environments": {}, "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    transform = transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size=224)])
    for env, item in preflight["environments"].items():
        with initialize_config_dir(version_base=None, config_dir=str(BUNDLE / "le-wm/config/eval")):
            cfg = compose(config_name=env, overrides=[f"policy={env}/lewm"])
        model = swm.wm.utils.load_pretrained(cfg.policy).eval().requires_grad_(False)
        assert next(model.parameters()).device.type == "cpu"
        first = item["official_data"]["selected_tasks"][0]
        with h5py.File(ROOT / item["official_data"]["file"], "r") as data:
            frames = data["pixels"][[first["row"], first["goal_row"]]]
            image_pair = torch.stack([transform(frame) for frame in frames])
            action = data["action"][:]
            action = action[~np.isnan(action).any(axis=1)]
            scaler = sklearn.preprocessing.StandardScaler().fit(action)
            stats = json.loads((ROOT / f"data/upstream/{env}/action_stats.json").read_text())
            np.testing.assert_allclose(scaler.mean_, stats["action"]["mean"], rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(scaler.scale_, stats["population_std"], rtol=1e-7, atol=1e-9)
        info = {"pixels": image_pair[0][None,None,None].expand(1,2,1,-1,-1,-1),
                "goal": image_pair[1][None,None,None].expand(1,2,1,-1,-1,-1),
                "action": torch.zeros(1,2,1,10)}
        candidates = torch.zeros(1,2,5,10)
        candidates[:,1] = 0.1
        with torch.inference_mode():
            costs = model.get_cost(info, candidates)
        assert costs.shape == (1,2) and torch.isfinite(costs).all()
        result["environments"][env] = {
            "strict_released_checkpoint_load": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "cpu_goal_cost_shape": list(costs.shape), "cpu_goal_costs_finite": True,
            "cpu_goal_cost_sha256": hashlib.sha256(costs.numpy().tobytes()).hexdigest(),
            "action_population_statistics_match_released_archive": True,
            "action_mean": scaler.mean_.tolist(), "action_std": scaler.scale_.tolist(),
            "config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        }
    output = ROOT / "reports/evidence/upstream_runtime_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "cpu_import_model_cost_validation_passed", "output": str(output),
                      "environments": list(result['environments'])}))


if __name__ == "__main__": main()
