"""Action-conditioned world models with temporally inferred shift contexts.

All target latents inhabit the immutable pretrained encoder coordinates. The
context for a query is inferred solely from preceding support observations and
executed actions. Intervention IDs are used only by optional training losses.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    mode: str = "factorized"
    context_dim: int = 32
    context_hidden: int = 128
    history_length: int = 3
    freeze_visual: bool = True
    train_predictor: bool = True
    alignment_weight: float = 1.0
    dynamics_consistency_weight: float = 0.1
    observation_consistency_weight: float = 0.01


class ResidualFiLM(nn.Module):
    def __init__(self, latent_dim, context_dim):
        super().__init__()
        self.affine = nn.Linear(context_dim, 2 * latent_dim)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, value, context):
        scale, shift = self.affine(context).chunk(2, dim=-1)
        return value * (1 + 0.1 * torch.tanh(scale[:, None])) + shift[:, None]


class TransitionContext(nn.Module):
    def __init__(self, dim, action_dim, hidden, context_dim):
        super().__init__()
        self.gru = nn.GRU(2 * dim + action_dim, hidden, batch_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, context_dim))

    def forward(self, observations, actions):
        if observations.shape[1] < 2:
            return observations.new_zeros(observations.shape[0], self.readout[-1].out_features)
        transitions = torch.cat((observations[:, :-1],
                                 observations[:, 1:] - observations[:, :-1], actions), -1)
        _, final = self.gru(transitions)
        return self.readout(final[-1])


class ShiftWorldModel(nn.Module):
    """Loadable full world model; action inputs are raw executed action blocks."""
    MODES = {"frozen", "plain", "framewise", "single", "factorized", "factorized_unpaired",
             "observation", "dynamics"}

    def __init__(self, base, base_config: dict, config: ModelConfig | dict,
                 action_mean, action_std, provenance: dict | None = None):
        super().__init__()
        self.config = ModelConfig(**config) if isinstance(config, dict) else config
        if self.config.mode not in self.MODES:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        self.base_config = deepcopy(base_config)
        self.provenance = provenance or {}
        self.base = base
        self.reference_encoder = deepcopy(base.encoder).requires_grad_(False).eval()
        self.reference_projector = deepcopy(base.projector).requires_grad_(False).eval()
        dim = int(base_config["predictor"]["input_dim"])
        act_dim = int(base_config["action_encoder"]["input_dim"])
        self.latent_dim, self.action_dim = dim, act_dim
        if self.config.history_length > base_config["predictor"]["num_frames"]:
            raise ValueError("Support history exceeds upstream predictor positional capacity")
        c, h = self.config.context_dim, self.config.context_hidden
        self.observation_context = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, h),
                                                nn.GELU(), nn.Linear(h, c))
        self.dynamics_context = TransitionContext(dim, act_dim, h, c)
        # Match the *active context network* parameter budget, not just total
        # backbone size. The nearest integer GRU width differs by <1% here.
        factor_budget = sum(p.numel() for p in self.observation_context.parameters()) + sum(
            p.numel() for p in self.dynamics_context.parameters())
        def single_size(width):
            return 3 * width * (2 * dim + act_dim) + 3 * width * width + 8 * width + width * c + c
        single_h = min(range(h, 3 * h + 1), key=lambda width: abs(single_size(width) - factor_budget))
        self.single_context = TransitionContext(dim, act_dim, single_h, c)
        self.observation_adapter = ResidualFiLM(dim, c)
        self.dynamics_adapter = ResidualFiLM(dim, c)
        if self.config.mode == "framewise":
            # Additional control only: existing modes instantiate exactly the
            # same modules in the same order, preserving RNG and checkpoint keys.
            self.framewise_adapter = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h),
                                                   nn.GELU(), nn.Linear(h, dim))
            nn.init.zeros_(self.framewise_adapter[-1].weight)
            nn.init.zeros_(self.framewise_adapter[-1].bias)
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32).reshape(-1))
        self.register_buffer("action_std", torch.as_tensor(action_std, dtype=torch.float32).reshape(-1))
        if self.action_mean.numel() != act_dim or self.action_std.numel() != act_dim:
            raise ValueError(f"Action statistics must contain {act_dim} grouped entries")
        if not torch.isfinite(self.action_mean).all():
            raise ValueError("Action means must be finite")
        if not torch.isfinite(self.action_std).all() or (self.action_std <= 0).any():
            raise ValueError("Action standard deviations must be finite and positive")
        self.register_buffer("pixel_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self._set_trainability()
        self.train(False)

    def _set_trainability(self):
        self.base.requires_grad_(self.config.train_predictor and self.config.mode != "frozen")
        if self.config.freeze_visual or self.config.mode == "frozen":
            self.base.encoder.requires_grad_(False)
            self.base.projector.requires_grad_(False)
        active = self.config.mode
        self.observation_context.requires_grad_(active in {"factorized", "factorized_unpaired", "observation"})
        self.dynamics_context.requires_grad_(active in {"factorized", "factorized_unpaired", "dynamics"})
        self.single_context.requires_grad_(active == "single")
        self.observation_adapter.requires_grad_(active in {"factorized", "factorized_unpaired", "observation", "single"})
        self.dynamics_adapter.requires_grad_(active in {"factorized", "factorized_unpaired", "dynamics", "single"})
        if active == "framewise":
            self.framewise_adapter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.reference_encoder.eval()
        self.reference_projector.eval()
        if self.config.freeze_visual or self.config.mode == "frozen":
            self.base.encoder.eval()
            self.base.projector.eval()
        if self.config.mode == "frozen":
            self.base.eval()
        # Retain upstream projection BN coordinates/statistics even for B=1 MPC.
        # Affine BN weights remain trainable when predictor training is enabled.
        for projection in (self.base.projector, self.base.pred_proj):
            for module in projection.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    # Never mix query/future statistics into support features,
                    # including the optional end-to-end visual fine-tune path.
                    module.eval()
        return self

    def normalize_actions(self, actions):
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected grouped action dimension {self.action_dim}")
        return (actions.float() - self.action_mean) / self.action_std

    def encode_images(self, images, reference=False, chunk_size=128):
        """Encode [...,3,H,W] RGB floats in [0,1]; normalizes once internally."""
        shape = images.shape[:-3]
        pixels = images.reshape(-1, *images.shape[-3:]).float()
        size = self.base_config["encoder"]["image_size"]
        encoder = self.reference_encoder if reference else self.base.encoder
        projector = self.reference_projector if reference else self.base.projector
        pieces = []
        for pixels_chunk in pixels.split(chunk_size):
            # Upstream normalizes then resizes with bilinear antialiasing.
            x = (pixels_chunk - self.pixel_mean) / self.pixel_std
            if x.shape[-2:] != (size, size):
                x = F.interpolate(x, (size, size), mode="bilinear", align_corners=False, antialias=True)
            enc = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            pieces.append(projector(enc))
        return torch.cat(pieces).reshape(*shape, self.latent_dim)

    def infer_context(self, support_features, support_actions):
        """No future frames/actions or intervention IDs accepted by this API."""
        if support_actions.shape[1] != support_features.shape[1] - 1:
            raise ValueError("Support requires exactly one executed action block per observed transition")
        act = self.normalize_actions(support_actions)
        mode = self.config.mode
        zero = support_features.new_zeros(support_features.shape[0], self.config.context_dim)
        if mode == "single":
            shared = self.single_context(support_features, act)
            return shared, shared
        if mode in {"plain", "frozen", "framewise"}:
            return zero, zero
        style = torch.cat((support_features.mean(1), support_features.var(1, unbiased=False)), -1)
        obs = self.observation_context(style) if mode != "dynamics" else zero
        corrected = self.correct_observations(support_features, obs)
        dyn = self.dynamics_context(corrected, act) if mode != "observation" else zero
        return obs, dyn

    def correct_observations(self, features, context):
        if self.config.mode == "framewise":
            return features + self.framewise_adapter(features)
        if self.config.mode in {"plain", "frozen", "dynamics"}:
            return features
        return self.observation_adapter(features, context)

    def predict_features(self, features, actions, dynamics_context):
        act = self.base.action_encoder(self.normalize_actions(actions))
        if self.config.mode not in {"plain", "frozen", "observation", "framewise"}:
            act = self.dynamics_adapter(act, dynamics_context)
        return self.base.predict(features, act)

    def _features(self, batch, prefix=""):
        key = prefix + "features"
        if key in batch:
            if not self.config.freeze_visual:
                raise ValueError("Cached features require freeze_visual=True")
            return batch[key].float()
        return self.encode_images(batch[prefix + "images"], reference=prefix == "reference_")

    def forward(self, batch):
        features = self._features(batch)
        with torch.no_grad():
            target = self._features(batch, "reference_")
        actions = batch["actions"]
        h = self.config.history_length
        if features.shape[1] < h + 2 or actions.shape[1] != features.shape[1] - 1:
            raise ValueError("Training needs H support frames plus at least two query frames and T-1 actions")
        obs, dyn = self.infer_context(features[:, :h], actions[:, :h - 1])
        corrected = self.correct_observations(features, obs)
        # Query starts at index H. Context sees [0,H); no query targets enter it.
        positions = range(h, features.shape[1] - 1)
        states = torch.stack([corrected[:, t - h + 1:t + 1] for t in positions], 1)
        action_windows = torch.stack([actions[:, t - h + 1:t + 1] for t in positions], 1)
        b, q = states.shape[:2]
        query_dyn = dyn[:, None].expand(-1, q, -1).reshape(b * q, -1)
        pred = self.predict_features(states.reshape(b * q, h, -1),
                                     action_windows.reshape(b * q, h, -1), query_dyn)[:, -1]
        pred = pred.reshape(b, q, -1)
        prediction_loss = F.mse_loss(pred, target[:, h + 1:])
        alignment_loss = F.mse_loss(corrected[:, h:], target[:, h:])
        dyn_loss, obs_loss = pred.new_zeros(()), pred.new_zeros(())
        if self.config.mode == "factorized":
            if "paired_features" in batch or "paired_images" in batch:
                paired = self._features(batch, "paired_")
                _, paired_dyn = self.infer_context(paired[:, :h], actions[:, :h - 1])
                dyn_loss = F.mse_loss(dyn, paired_dyn)
            if "observation_id" in batch:
                ids = batch["observation_id"].reshape(-1)
                same = ids[:, None].eq(ids[None, :])
                same.fill_diagonal_(False)
                if same.any():
                    obs_loss = (obs[:, None] - obs[None, :]).square().mean(-1)[same].mean()
        loss = (prediction_loss + self.config.alignment_weight * alignment_loss
                + self.config.dynamics_consistency_weight * dyn_loss
                + self.config.observation_consistency_weight * obs_loss)
        return {"loss": loss, "prediction_loss": prediction_loss, "alignment_loss": alignment_loss,
                "dynamics_consistency_loss": dyn_loss, "observation_consistency_loss": obs_loss,
                "observation_context_std": obs.float().std(0, unbiased=False).mean(),
                "dynamics_context_std": dyn.float().std(0, unbiased=False).mean(),
                "predictions": pred, "targets": target[:, h + 1:],
                "observation_context": obs, "dynamics_context": dyn}

    def rollout_features(self, history_features, past_actions, future_actions, contexts=None):
        """Strictly future actions; output [B,K,D] excludes observed history."""
        obs, dyn = contexts or self.infer_context(history_features, past_actions)
        states = self.correct_observations(history_features, obs)
        all_actions = torch.cat((past_actions, future_actions), 1)
        predictions = []
        h = self.config.history_length
        for step in range(future_actions.shape[1]):
            end = states.shape[1]
            lo = max(0, end - h)
            pred = self.predict_features(states[:, lo:end], all_actions[:, lo:end], dyn)[:, -1:]
            states = torch.cat((states, pred), 1)
            predictions.append(pred)
        if not predictions:
            return history_features.new_empty(history_features.shape[0], 0, self.latent_dim)
        return torch.cat(predictions, 1)

    def rollout(self, images, past_actions, future_actions):
        return self.rollout_features(self.encode_images(images), past_actions, future_actions)

    def goal_embedding(self, goal_images, observation_context):
        features = self.encode_images(goal_images)
        return self.correct_observations(features[:, None], observation_context)[:, 0]

    def export_config(self):
        return {"model_config": asdict(self.config), "base_config": self.base_config,
                "provenance": self.provenance, "action_mean": self.action_mean.tolist(),
                "action_std": self.action_std.tolist()}
