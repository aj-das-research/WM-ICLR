"""Recursive cross-domain models with two predictor families and matched controls.

The transformer, encoder, action embedding and projection layers reuse the pinned
LeWM implementation and released initialization. The alternative predictor is an
explicit in-house GRU control, not a reproduction of Dreamer or another system.
Comparisons are made within predictor families; parameter counts differ between
families. All methods predict in the same immutable visual coordinates.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from shiftwm.model import ModelConfig, ShiftWorldModel
from shiftwm.upstream import create_base, load_base


ARCHITECTURES = ("transformer", "gru")
MODES = ("framewise", "factorized", "constant_dynamics")


class GRUPredictor(nn.Module):
    """Causal sequence predictor using the upstream projected state/action inputs."""

    def __init__(self, dimension: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dimension)
        self.gru = nn.GRU(2 * dimension, hidden, num_layers=layers, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dimension))

    def forward(self, states, actions):
        if states.shape != actions.shape:
            raise ValueError("GRU needs aligned projected state/action sequences")
        output, _ = self.gru(self.norm(torch.cat((states, actions), dim=-1)))
        return self.output(output)


def configure_predictor(base, base_config, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown predictor architecture {architecture}")
    if architecture == "gru":
        dimension = int(base_config["predictor"]["input_dim"])
        base.predictor = GRUPredictor(dimension)
    return base


class ExtensionWorldModel(ShiftWorldModel):
    """Five-step recursive objective, including the first unobserved target.

Only the first three observations and preceding two action blocks infer context.
No simulator state, gain, desired coordinates or future image enters inference.
Canonical future images are privileged training/evaluation targets only.
"""

    def __init__(self, base, base_config, mode, action_mean, action_std,
                 architecture="transformer", provenance=None, model_config=None):
        if mode not in MODES:
            raise ValueError(f"Unknown extension method {mode}")
        self.extension_mode = mode
        self.architecture = architecture
        kwargs = dict(model_config or {})
        kwargs["mode"] = "framewise" if mode == "framewise" else "factorized"
        kwargs["freeze_visual"] = True
        kwargs["train_predictor"] = True
        super().__init__(base, base_config, ModelConfig(**kwargs), action_mean,
                         action_std, provenance)

    def infer_context(self, support_features, support_actions):
        if self.extension_mode != "constant_dynamics":
            return super().infer_context(support_features, support_actions)
        if support_actions.shape[1] != support_features.shape[1] - 1:
            raise ValueError("Support actions must align with observed transitions")
        # Keep observation inference unchanged; remove episode information only
        # from dynamics inference. Biases learn a common context. Nominal module
        # size matches factorized, but effective input-dependent capacity differs.
        style = torch.cat((support_features.mean(1),
                           support_features.var(1, unbiased=False)), dim=-1)
        obs = self.observation_context(style)
        dyn = self.dynamics_context(torch.zeros_like(support_features),
                                    torch.zeros_like(self.normalize_actions(support_actions)))
        return obs, dyn

    def forward(self, batch):
        features = self._features(batch)
        with torch.no_grad():
            target = self._features(batch, "reference_")
        actions = batch["actions"]
        h = self.config.history_length
        if features.shape[1] <= h or actions.shape[1] != features.shape[1] - 1:
            raise ValueError("Need chronological support, aligned actions and query targets")
        obs, dyn = self.infer_context(features[:, :h], actions[:, :h - 1])
        predictions = self.rollout_features(features[:, :h], actions[:, :h - 1],
                                             actions[:, h - 1:], contexts=(obs, dyn))
        query_target = target[:, h:]
        prediction = F.mse_loss(predictions, query_target)
        corrected = self.correct_observations(features, obs)
        alignment = F.mse_loss(corrected[:, h:], query_target)
        dynamics_consistency = prediction.new_zeros(())
        observation_consistency = prediction.new_zeros(())
        if self.extension_mode != "framewise":
            if "paired_features" in batch or "paired_images" in batch:
                paired = self._features(batch, "paired_")
                _, paired_dyn = self.infer_context(paired[:, :h], actions[:, :h - 1])
                dynamics_consistency = F.mse_loss(dyn, paired_dyn)
            if "observation_id" in batch:
                ids = batch["observation_id"].reshape(-1)
                same = ids[:, None].eq(ids[None, :])
                same.fill_diagonal_(False)
                if same.any():
                    observation_consistency = (obs[:, None] - obs[None, :]).square().mean(-1)[same].mean()
        loss = (prediction + self.config.alignment_weight * alignment
                + self.config.dynamics_consistency_weight * dynamics_consistency
                + self.config.observation_consistency_weight * observation_consistency)
        return {"loss": loss, "prediction_loss": prediction, "alignment_loss": alignment,
                "dynamics_consistency_loss": dynamics_consistency,
                "observation_consistency_loss": observation_consistency,
                "predictions": predictions, "targets": query_target,
                "observation_context": obs, "dynamics_context": dyn}

    @property
    def history_length(self):
        return self.config.history_length

    def recursive_validation(self, batch):
        """Return the common all-query recursive metric; caller selects FP32/eval."""
        return self.forward(batch)

    @property
    def package_config(self):
        return {"extension_format_version": 1,
                "upstream_base_config": deepcopy(self.base_config),
                "architecture": self.architecture, "mode": self.extension_mode,
                "model_config": asdict(self.config),
                "action_mean": self.action_mean.detach().cpu().tolist(),
                "action_std": self.action_std.detach().cpu().tolist(),
                "provenance": deepcopy(self.provenance),
                "objective": "recursive_mse_all_query_steps",
                "predictor_initialization": ("released_upstream" if self.architecture == "transformer"
                                               else "new_gru_with_released_action_and_projection_layers")}


def build_model(pretrained_dir, architecture, mode, action_mean, action_std, seed=0):
    """Initialize all methods within a family from the same predictor weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        base, config, provenance = load_base(pretrained_dir)
        base = configure_predictor(base, config, architecture)
        return ExtensionWorldModel(base, config, mode, action_mean, action_std,
                                   architecture, provenance)


def build_from_config(config):
    """Construct a package without downloads; load its complete state afterwards."""
    if config.get("extension_format_version") != 1:
        raise ValueError("Unsupported extension checkpoint format")
    base_config = config["upstream_base_config"]
    base = configure_predictor(create_base(base_config), base_config, config["architecture"])
    return ExtensionWorldModel(base, base_config, config["mode"],
                               config["action_mean"], config["action_std"],
                               config["architecture"], config.get("provenance", {}),
                               config["model_config"])
