"""Residual DINO-feature forecasting with the pinned LeWM causal predictor.

This is a real-video adaptation of our architecture, not a reproduction of the
published DINO-WM or a claim that a new backbone establishes novelty. All target
coordinates and normalization statistics are frozen. Only observed support
frames and recorded support actions infer contexts; query images are targets.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from shiftwm.model import ResidualFiLM, TransitionContext
from shiftwm.upstream import _source_module


@dataclass
class RealVideoConfig:
    feature_dim: int = 1536
    action_dim: int = 35
    hidden_dim: int = 192
    history_length: int = 3
    context_dim: int = 32
    context_hidden: int = 128
    depth: int = 4
    mode: str = "factorized"


class RealVideoWorldModel(nn.Module):
    MODES = ("framewise", "constant_dynamics", "factorized", "action_free")

    def __init__(self, config, feature_mean, feature_std, action_mean, action_std):
        super().__init__()
        self.config = RealVideoConfig(**config) if isinstance(config, dict) else config
        c = self.config
        if c.mode not in self.MODES or c.history_length < 2:
            raise ValueError("Unsupported model configuration")
        for name, values, dimension in (
            ("feature_mean", feature_mean, c.feature_dim),
            ("feature_std", feature_std, c.feature_dim),
            ("action_mean", action_mean, c.action_dim),
            ("action_std", action_std, c.action_dim),
        ):
            value = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
            if len(value) != dimension or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name}")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"Nonpositive {name}")
            self.register_buffer(name, value)
        modules = _source_module("shiftwm_upstream_module", "module.py")
        d = c.hidden_dim
        self.input_projection = nn.Sequential(nn.LayerNorm(c.feature_dim), nn.Linear(c.feature_dim, d))
        self.action_projection = modules.Embedder(input_dim=c.action_dim, emb_dim=d)
        self.predictor = modules.ARPredictor(num_frames=c.history_length, input_dim=d,
            hidden_dim=d, output_dim=d, depth=c.depth, heads=6, mlp_dim=4*d,
            dim_head=32, dropout=0.1, emb_dropout=0.0)
        self.output_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c.feature_dim))
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)
        self.observation_context = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, c.context_hidden),
            nn.GELU(), nn.Linear(c.context_hidden, c.context_dim))
        self.dynamics_context = TransitionContext(d, c.action_dim, c.context_hidden, c.context_dim)
        self.observation_adapter = ResidualFiLM(d, c.context_dim)
        self.dynamics_adapter = ResidualFiLM(d, c.context_dim)
        self.framewise_adapter = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c.context_hidden),
            nn.GELU(), nn.Linear(c.context_hidden, d))
        nn.init.zeros_(self.framewise_adapter[-1].weight)
        nn.init.zeros_(self.framewise_adapter[-1].bias)
        # Keep initialization of the common predictor identical across modes.
        context_active = c.mode in ("factorized", "constant_dynamics")
        for module in (self.observation_context, self.dynamics_context,
                       self.observation_adapter, self.dynamics_adapter):
            module.requires_grad_(context_active)
        self.framewise_adapter.requires_grad_(not context_active)
        if c.mode == "action_free":
            self.action_projection.requires_grad_(False)

    def normalize_features(self, values):
        return (values.float() - self.feature_mean) / self.feature_std

    def normalize_actions(self, values):
        if self.config.mode == "action_free":
            return torch.zeros_like(values, dtype=torch.float32)
        return (values.float() - self.action_mean) / self.action_std

    def _context(self, support, actions):
        encoded = self.input_projection(support)
        if self.config.mode in ("framewise", "action_free"):
            return None
        style = torch.cat((encoded.mean(1), encoded.var(1, unbiased=False)), -1)
        obs = self.observation_context(style)
        corrected = self.observation_adapter(encoded, obs)
        if self.config.mode == "constant_dynamics":
            corrected, actions = torch.zeros_like(corrected), torch.zeros_like(actions)
        dyn = self.dynamics_context(corrected, actions)
        return obs, dyn

    def _predict_normalized(self, support, support_actions, future_actions):
        h = self.config.history_length
        if support.shape[1] != h or support_actions.shape[1] != h - 1:
            raise ValueError("Support must contain the fixed causal history")
        if support.shape[-1] != self.config.feature_dim:
            raise ValueError("Wrong frozen feature coordinates")
        if future_actions.shape[1] < 1:
            raise ValueError("No future actions")
        if support_actions.shape[-1] != self.config.action_dim or future_actions.shape[-1] != self.config.action_dim:
            raise ValueError("Wrong action dimension")
        past = self.normalize_actions(support_actions)
        future = self.normalize_actions(future_actions)
        contexts = self._context(support, past)
        observations = support
        actions = past
        predictions = []
        for t in range(future.shape[1]):
            current_actions = torch.cat((actions, future[:, t:t+1]), 1)[:, -h:]
            encoded = self.input_projection(observations[:, -h:])
            if contexts is None:
                encoded = encoded + self.framewise_adapter(encoded)
            else:
                encoded = self.observation_adapter(encoded, contexts[0])
            action_embeddings = self.action_projection(current_actions)
            if self.config.mode == "action_free":
                action_embeddings = torch.zeros_like(action_embeddings)
            predicted = self.predictor(encoded, action_embeddings)
            if contexts is not None:
                predicted = self.dynamics_adapter(predicted, contexts[1])
            # Persistence is the shared initialization, never a moving target.
            next_features = observations[:, -1] + self.output_projection(predicted[:, -1])
            predictions.append(next_features)
            observations = torch.cat((observations, next_features[:, None]), 1)[:, -h:]
            actions = current_actions[:, 1:]
        return torch.stack(predictions, 1)

    def predict(self, support_features, support_actions, future_actions):
        predictions = self._predict_normalized(self.normalize_features(support_features),
                                                support_actions, future_actions)
        return predictions * self.feature_std + self.feature_mean

    def forward(self, batch):
        features, actions = batch["features"], batch["actions"]
        h = self.config.history_length
        if features.shape[1] <= h or actions.shape[1] != features.shape[1] - 1:
            raise ValueError("Chronological features and actions must align")
        standardized = self.normalize_features(features)
        predictions = self._predict_normalized(standardized[:, :h], actions[:, :h-1], actions[:, h-1:])
        loss = F.mse_loss(predictions, standardized[:, h:])
        return {"loss": loss, "predictions": predictions*self.feature_std+self.feature_mean,
                "targets": features[:, h:], "standardized_predictions": predictions,
                "standardized_targets": standardized[:, h:]}

    @property
    def package_config(self):
        return {"format_version": 1, "model_config": asdict(self.config),
                **{key: getattr(self, key).detach().cpu().tolist() for key in
                   ("feature_mean", "feature_std", "action_mean", "action_std")}}


def from_config(config):
    if config.get("format_version") != 1:
        raise ValueError("Unsupported real-video package")
    return RealVideoWorldModel(config["model_config"], **{key: config[key] for key in
        ("feature_mean", "feature_std", "action_mean", "action_std")})
