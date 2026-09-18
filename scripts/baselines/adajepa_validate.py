#!/usr/bin/env python3
"""Validate the actual released AdaJEPA model, observed replay, and update reset."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external/adajepa"
sys.path.insert(0, str(UPSTREAM))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TORCH_HOME", str(ROOT / "environments/adajepa/torch_cache"))

import numpy as np
import torch
from omegaconf import OmegaConf

from plan import load_model
from datasets.img_transforms import default_transform
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD, PROPRIO_MEAN, PROPRIO_STD
from env.pusht.pusht_wrapper import PushTWrapper
from planning.adajepa import AdaJEPATrainer
from preprocessor import Preprocessor


def tensors_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    start = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(100)
    checkpoint_root = ROOT / "data/baselines/adajepa/release/pusht_visual_shift"
    checkpoint = checkpoint_root / "checkpoints/model_latest.pth"
    config = OmegaConf.load(checkpoint_root / "hydra.yaml")
    model = load_model(checkpoint, config, config.num_action_repeat, torch.device("cpu"))
    model.eval()
    with (ROOT / "data/baselines/adajepa/release/pushobj_eval/val_T/plan_targets.pkl").open("rb") as handle:
        targets = pickle.load(handle)
    chosen = random.Random(100).sample(targets["segments"], min(50, len(targets["segments"])))[0]
    initial = np.asarray(chosen["states"][0], dtype=np.float32)
    if initial.shape == (5,):
        initial = np.pad(initial, (0, 2))
    commands = np.asarray(chosen["actions"][:25], dtype=np.float32) / 100
    env = PushTWrapper(with_velocity=True, with_target=True)
    env.update_env({"shape": chosen.get("shape", "T")})
    try:
        observations, states = env.rollout(100, initial, commands)
        repeated, repeated_states = env.rollout(100, initial, commands)
        for key in observations:
            np.testing.assert_array_equal(observations[key], repeated[key])
        np.testing.assert_array_equal(states, repeated_states)
    finally:
        env.close()
    preprocessor = Preprocessor(ACTION_MEAN, ACTION_STD, STATE_MEAN[:5], STATE_STD[:5],
                                PROPRIO_MEAN[:4], PROPRIO_STD[:4], default_transform())
    obs = preprocessor.transform_obs({key: np.expand_dims(value[::5], 0)
                                      for key, value in observations.items()})
    actions = ((torch.tensor(commands) - ACTION_MEAN) / ACTION_STD).reshape(1, -1, 10)
    trainer = AdaJEPATrainer(model, lr=5e-4, steps=1, finetune_encoder=True,
                            last_layer_only=True, encoder_lr=1e-5, encoder_last_layer_only=True)
    before = tensors_digest(model)
    losses = trainer.finetune([obs], [actions])
    after = tensors_digest(model)
    assert len(losses) == 1 and np.isfinite(losses).all()
    assert before != after, "The official adaptation step made no parameter/buffer change"
    trainer.reset()
    restored = tensors_digest(model)
    assert before == restored, "Official per-episode reset failed to restore exact weights/buffers"
    with torch.no_grad():
        observation_latents = model.encode_obs(obs)
    assert all(torch.isfinite(value).all() for value in observation_latents.values())
    result = {"status": "passed", "device": "cpu", "python": sys.version,
              "torch": torch.__version__, "parameter_count": sum(p.numel() for p in model.parameters()),
              "encoder": type(model.encoder).__name__, "predictor": type(model.predictor).__name__,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "released_segments": len(targets["segments"]), "observed_native_actions": len(commands),
              "encoded_frames": int(obs["visual"].shape[1]), "latent_shapes": {
                  key: list(value.shape) for key, value in observation_latents.items()},
              "model_initial_state_sha256": before, "adapted_state_sha256": after,
              "reset_state_sha256": restored, "adaptation_training_loss": losses,
              "tests": ["official checkpoint loads", "actual released actions replay exactly",
                        "all latent outputs finite", "official online update changes model",
                        "official reset restores every parameter and buffer"],
              "interpretation": "engineering validation only; not a planning comparison or benchmark result",
              "elapsed_seconds": time.time() - start}
    destination = ROOT / "reports/evidence/adajepa_cpu_validation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
