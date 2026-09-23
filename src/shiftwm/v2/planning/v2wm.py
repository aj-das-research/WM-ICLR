"""Trained ShiftWM-v2 / matched-baseline predictors as pluggable planning predictors.

``build(env, ckpt)`` loads ``results/v2/plan/<env>/<arm>/s<seed>/best.pt`` together with the
training ``config.json`` saved next to it. That config gives the feature cache, whose
``stats.json`` holds the feature and action standardisation and whose ``manifest.json`` names the
encoder. The Stage-1 manifest gives the planner's action scaler. Latents are frozen-encoder patch
grids (DINOv2-S: 16x16x384) standardised with the cache statistics, exactly as in training. The
cost is the MSE between the predicted final grid and the goal grid.

    python scripts/v2/planning_eval.py --env pusht --predictor shiftwm.v2.planning.v2wm:build \
        --ckpt results/v2/plan/pusht/shiftwm/s0/best.pt --model v2_shiftwm
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..extract import ENCODERS
from ..models import V2WorldModel
from .interface import PlanningPredictor
from .lewm import sha256

ROOT = Path(__file__).resolve().parents[4]


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class V2PlanningPredictor(PlanningPredictor):
    def __init__(self, model, encoder, grid, feature_mean, feature_std, act_offset, act_scale, chunk=150):
        super().__init__()
        self.model, self.encoder, self.grid = model, encoder, grid
        self.name = f"v2_{model.config.arm}"
        self.register_buffer("fm", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("fs", torch.as_tensor(feature_std, dtype=torch.float32))
        # model_action = planner_action * act_scale + act_offset  (planner z-score -> raw -> train z-score)
        self.register_buffer("a_scale", torch.as_tensor(act_scale, dtype=torch.float32))
        self.register_buffer("a_off", torch.as_tensor(act_offset, dtype=torch.float32))
        self.chunk = chunk

    @torch.no_grad()
    def encode(self, pixels):
        """pixels (N,T,3,224,224) ImageNet-normalised (same normalisation as shiftwm.v2.extract.encode)."""
        n, t = pixels.shape[:2]
        x = pixels.reshape(n * t, *pixels.shape[2:]).float()
        g = self.grid
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            tok = self.encoder(pixel_values=x).last_hidden_state[:, -g * g:]
        z = (tok.float() - self.fm) / self.fs
        return z.reshape(n, t, g * g, -1)

    def _act(self, a):
        return a.float() * self.a_scale + self.a_off

    @torch.no_grad()
    def rollout(self, latent_hist, action_hist, actions):
        n = latent_hist.shape[0]
        past = (self._act(action_hist) if action_hist is not None
                else actions.new_zeros(n, 0, actions.shape[-1]).float())
        fut = self._act(actions)
        outs = []
        for i in range(0, n, self.chunk):
            sl = slice(i, i + self.chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=latent_hist.is_cuda):
                outs.append(self.model(latent_hist[sl].float(), past[sl], fut[sl]).float())
        return torch.cat(outs)

    def cost(self, pred_latents, goal_latent):
        return (pred_latents[:, -1].float() - goal_latent.float()).pow(2).mean(dim=(1, 2))


def build(env: str, ckpt: str, device: str = "cuda", **_):
    ckpt = _abs(ckpt)
    if ckpt.is_dir():
        ckpt = ckpt / "best.pt"
    state = torch.load(ckpt, map_location="cpu")
    train_cfg = json.loads((ckpt.parent / "config.json").read_text())
    feat_root = _abs(train_cfg["data"])
    stats = json.loads((feat_root / "stats.json").read_text())
    fman = json.loads((feat_root / "manifest.json").read_text())
    s1man = json.loads((_abs(fman["source"]) / "manifest.json").read_text())

    model = V2WorldModel(state["config"])
    model.load_state_dict(state["model"])
    model = model.to(device).eval().requires_grad_(False)

    from transformers import AutoModel
    enc_dir = _abs(ENCODERS[fman["encoder"]][0])
    encoder = AutoModel.from_pretrained(enc_dir, local_files_only=True).to(device).eval().requires_grad_(False)
    if fman["image_size"] != 224:
        raise ValueError("planning renders are 224 px; encoders needing resize are not wired in yet")

    block = s1man["action_block"]
    pm = np.tile(np.asarray(s1man["planner_action_scaler"]["mean"]), block)
    ps = np.tile(np.asarray(s1man["planner_action_scaler"]["scale"]), block)
    am, ast = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])
    pred = V2PlanningPredictor(model, encoder, fman["grid"], stats["feature_mean"], stats["feature_std"],
                               act_offset=(pm - am) / ast, act_scale=ps / ast).to(device)
    meta = {"ckpt": str(ckpt), "ckpt_sha256": sha256(ckpt), "step": state.get("step"), "arm": model.config.arm,
            "n_params": model.num_params(), "train_config": train_cfg, "features": str(feat_root),
            "encoder": fman["encoder"], "encoder_sha256": fman.get("encoder_weights_sha256"),
            "history": model.config.history, "horizon": model.config.horizon}
    return pred, meta
