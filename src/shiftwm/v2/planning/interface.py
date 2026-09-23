"""Pluggable predictor interface for the official-protocol planning harness.

A world model is plugged into planning by implementing three methods on a
``torch.nn.Module`` subclass of :class:`PlanningPredictor`:

``encode(pixels) -> latents``
    pixels: ``(N, T, 3, 224, 224)`` float, ImageNet-normalised (exactly the
    tensors the official LeWM evaluator feeds its encoder). Returns latents of
    shape ``(N, T, *latent_shape)`` (e.g. ``(N, T, 192)`` for LeWM's CLS token
    or ``(N, T, 256, 384)`` for a 16x16 DINO patch grid).

``rollout(latent_hist, action_hist, actions) -> latents``
    latent_hist: ``(N, H, *latent_shape)`` encoded context frames (H = number of
    context frames, 1 under the released protocol).
    action_hist: ``(N, H-1, A)`` executed action blocks between context frames,
    or ``None`` when H == 1.
    actions: ``(N, T, A)`` candidate future action blocks. ``A`` is the *blocked*
    action dim (``env_action_dim * action_block``; PushT 2*5=10) and actions are
    z-scored with the dataset action StandardScaler (as in LeWM ``eval.py``).
    ``actions[:, 0]`` is the block executed from the current frame.
    Returns the ``T`` predicted future latents ``(N, T, *latent_shape)``; entry
    ``t`` is the state after executing ``actions[:, :t+1]``.

``cost(pred_latents, goal_latent) -> (N,)``
    pred_latents: ``(N, T, *latent_shape)`` from ``rollout``; goal_latent:
    ``(N, *latent_shape)``. Default: LeWM's last-step summed squared error.

:class:`PredictorDynamics` adapts such a predictor to the stable-worldmodel
``Dynamics``/``Objective`` seam so the *unmodified* upstream CEM solver,
``WorldModelPolicy`` MPC loop and dataset-driven ``World.evaluate`` are reused.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class PlanningPredictor(nn.Module):
    """Base class: override ``encode`` and ``rollout``; optionally ``cost``."""

    #: optional descriptive name recorded in result JSON
    name: str = "predictor"

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def rollout(
        self,
        latent_hist: torch.Tensor,
        action_hist: torch.Tensor | None,
        actions: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def cost(self, pred_latents: torch.Tensor, goal_latent: torch.Tensor) -> torch.Tensor:
        diff = pred_latents[:, -1].float() - goal_latent.float()
        return diff.pow(2).flatten(1).sum(dim=1)


class PredictorDynamics(nn.Module):
    """Expose a :class:`PlanningPredictor` as swm ``Dynamics`` (encode/rollout on info dicts)."""

    CTX_KEY = "_ctx_latent"

    def __init__(self, predictor: PlanningPredictor):
        super().__init__()
        self.predictor = predictor

    # swm Dynamics.encode: info['pixels'] (B, T, C, H, W) -> info['emb'] (B, T, ...)
    def encode(self, info: dict) -> dict:
        info["emb"] = self.predictor.encode(info["pixels"])
        return info

    def rollout(self, info: dict, action_candidates: torch.Tensor) -> dict:
        B, S, T = action_candidates.shape[:3]
        # context pixels are identical across the S candidates -> encode once per env
        # and cache for the remaining CEM iterations (the solver reuses this dict).
        if self.CTX_KEY not in info:
            with torch.no_grad():
                info[self.CTX_KEY] = self.predictor.encode(info["pixels"][:, 0])  # (B, H, ...)
        ctx = info[self.CTX_KEY]
        H = ctx.shape[1]
        ctx_bs = ctx.unsqueeze(1).expand(B, S, *ctx.shape[1:])
        hist = info.get("action_history")
        if hist is not None and hist.shape[2] == 0:
            hist = None
        if H > 1:
            assert hist is not None and hist.shape[2] == H - 1, "need H-1 executed action blocks"
            hist = rearrange(hist, "b s ... -> (b s) ...")
        else:
            hist = None
        preds = self.predictor.rollout(
            rearrange(ctx_bs, "b s ... -> (b s) ..."),
            hist,
            rearrange(action_candidates, "b s ... -> (b s) ..."),
        )
        assert preds.shape[1] == T, f"rollout must return T={T} future latents, got {tuple(preds.shape)}"
        preds = rearrange(preds, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = torch.cat([ctx_bs.to(preds.dtype), preds], dim=2)  # (B, S, H+T, ...)
        info["_n_ctx"] = H
        return info


class PredictorObjective(nn.Module):
    """swm ``Objective`` that delegates to ``predictor.cost`` on the future part of the rollout."""

    def __init__(self, predictor: PlanningPredictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, info: dict) -> torch.Tensor:
        pred = info["predicted_emb"]  # (B, S, H+T, ...)
        H = info.get("_n_ctx", 0)
        goal = info["goal_emb"][:, -1]  # (B, ...)
        B, S = pred.shape[:2]
        fut = rearrange(pred[:, :, H:], "b s ... -> (b s) ...")
        g = goal.unsqueeze(1).expand(B, S, *goal.shape[1:])
        c = self.predictor.cost(fut, rearrange(g, "b s ... -> (b s) ..."))
        return c.view(B, S)


def predictor_goal_encode(model: PredictorDynamics, info: dict) -> torch.Tensor:
    """Goal encoder for :class:`PredictorDynamics` (goal image -> ``(B, 1, ...)``)."""
    goal = info["goal"][:, 0]  # (B, T_goal, C, H, W) -- identical across candidates
    with torch.no_grad():
        return model.predictor.encode(goal)[:, -1:]


def build_cost(predictor: PlanningPredictor):
    """Return a swm ``Costable`` (ShootingCostEvaluator) for a pluggable predictor."""
    import stable_worldmodel as swm

    dyn = PredictorDynamics(predictor)
    return swm.planning.ShootingCostEvaluator(
        dyn, PredictorObjective(predictor), encode_goal=predictor_goal_encode
    )
