"""Exploratory IWS bound-removal ablation of the frozen single-image decoder.

The constructor and all shared operators are copied from the immutable v1 model
whose SHA is SOURCE_BASE_SHA256 below. Only one public mode is accepted. The
fixed-source mixing path and matched initialization remain unchanged; innovation
is the raw residual projection, without tanh. This is not an AR/additive fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from shiftwm.upstream import _source_module

PACKAGE_KIND = "shiftwm_iws_single_observation_unbounded_v1"
SOURCE_BASE_SHA256 = "f10c5f3855d510d982e80001da19d040cb280eaa747fdb01f8fcffb5af4a1f1a"
COORDINATE_LAYOUT = "channel_major_384x4x4_shared_channel_normalization"
INPUT_CONTRACT = "one_observation_H_native_command_rows_predict_offsets_1_through_Hminus1"


@dataclass
class IWSConfig:
    mode: str = "unbounded_spatial_mix"
    feature_dim: int = 6144
    channels: int = 384
    grid_size: int = 4
    action_dim: int = 4
    hidden_dim: int = 96
    depth: int = 4
    temporal_slots: int = 3
    innovation_bound: float = 1.0
    identity_bias: float = 4.0
    initial_gate_logit: float = -3.0


class SingleObservationWorldModel(nn.Module):
    """Fixed-source mixer with only the innovation bound removed; no FiLM/context.

    Normalization statistics are fitted by the frozen internal-training cache.
    Targets are used only by ``forward`` for loss calculation; ``predict`` accepts
    one initial feature vector and native commands, never a target tensor.
    """
    MODES = ("unbounded_spatial_mix",)

    def __init__(self, config, feature_mean, feature_std, command_mean, command_std):
        super().__init__()
        self.config = IWSConfig(**config) if isinstance(config, dict) else config
        if not isinstance(self.config, IWSConfig):
            raise ValueError("Expected the distinct IWS model configuration")
        c = self.config
        if (c.mode not in self.MODES or c.feature_dim != 6144 or c.channels != 384
                or c.grid_size != 4 or c.temporal_slots != 3 or c.action_dim not in (4, 8, 14)
                or not isinstance(c.hidden_dim, int) or c.hidden_dim < 6 or c.hidden_dim % 6
                or not isinstance(c.depth, int) or c.depth < 1
                or not math.isfinite(c.innovation_bound) or c.innovation_bound != 1.0
                or not math.isfinite(c.identity_bias) or not math.isfinite(c.initial_gate_logit)):
            raise ValueError("Invalid unbounded IWS single-observation configuration")
        n, d = c.grid_size ** 2, c.hidden_dim
        for key, value, dim in (("feature_mean", feature_mean, c.feature_dim),
                                ("feature_std", feature_std, c.feature_dim),
                                ("command_mean", command_mean, c.action_dim),
                                ("command_std", command_std, c.action_dim)):
            value = torch.as_tensor(value, dtype=torch.float32).detach().clone().reshape(-1)
            if len(value) != dim or not torch.isfinite(value).all() or (key.endswith("std") and (value <= 0).any()):
                raise ValueError("Invalid IWS normalization: " + key)
            if key.startswith("feature"):
                grid = value.view(c.channels, n)
                if not torch.equal(grid, grid[:, :1].expand(-1, n)):
                    raise ValueError("IWS mixing requires shared per-channel normalization")
            self.register_buffer(key, value)

        # Exact v1 construction order preserves seeded parameter/active-head parity.
        # innovation_bound=1 is retained as recipe metadata only; it is not applied.
        self.input_projection = nn.Sequential(nn.LayerNorm(c.channels), nn.Linear(c.channels, d))
        self.position = nn.Parameter(torch.zeros(1, n, d))
        nn.init.normal_(self.position, std=.01)
        self.spatial_norm = nn.LayerNorm(d)
        self.spatial_attention = nn.MultiheadAttention(d, 6, dropout=0.0, batch_first=True)
        self.spatial_ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.action_prefix = nn.GRU(c.action_dim, d, batch_first=True)
        modules = _source_module("shiftwm_upstream_module", "module.py")
        self.predictor = modules.ARPredictor(num_frames=c.temporal_slots, input_dim=d,
            hidden_dim=d, output_dim=d, depth=c.depth, heads=6, mlp_dim=4*d,
            dim_head=16, dropout=.1, emb_dropout=0.)
        self.output_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c.channels))
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)
        self.transport_query = nn.Linear(d, d, bias=False)
        self.transport_key = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, c.initial_gate_logit)
        self.register_buffer("identity_transport", torch.eye(n))
        # Preserve the v1 head construction and flags. Every mixing head is
        # active in this namespace's sole supported ablation mode.
        mixing = c.mode == "unbounded_spatial_mix"
        for head in (self.transport_query, self.transport_key, self.gate):
            head.requires_grad_(mixing)

    def tokens(self, flat):
        if flat.shape[-1] != self.config.feature_dim:
            raise ValueError("Expected channel-major 384x4x4 features")
        return flat.reshape(*flat.shape[:-1], self.config.channels, 16).transpose(-1, -2)

    def flatten(self, value):
        return value.transpose(-1, -2).reshape(*value.shape[:-2], self.config.feature_dim)

    def normalize_features(self, value):
        return (value.float() - self.feature_mean) / self.feature_std

    def normalize_commands(self, value):
        return (value.float() - self.command_mean) / self.command_std

    def _encode(self, value):
        encoded = self.input_projection(value) + self.position
        shape = encoded.shape
        encoded = encoded.reshape(-1, shape[-2], shape[-1])
        normalized = self.spatial_norm(encoded)
        encoded = encoded + self.spatial_attention(normalized, normalized, normalized, need_weights=False)[0]
        return (encoded + self.spatial_ff(encoded)).reshape(shape)

    def _predict_normalized(self, initial, native_commands, return_details=False):
        """Initial is standardized; commands are raw native rows.

        Output index k-1 refers to stored offset k. In particular, prediction0
        uses rows0 and1, and prediction58 at H60 uses all60 supplied rows.
        """
        c = self.config
        if c.mode != "unbounded_spatial_mix" or c.innovation_bound != 1.0:
            raise ValueError("Only the fixed-source innovation-bound removal is supported")
        if (initial.ndim != 2 or initial.shape[1] != c.feature_dim or len(initial) < 1
                or native_commands.ndim != 3 or native_commands.shape[0] != len(initial)
                or native_commands.shape[1] < 2 or native_commands.shape[2] != c.action_dim):
            raise ValueError("Expected one initial feature[B,6144] and commands[B,H>=2,A]")
        if not torch.isfinite(initial).all() or not torch.isfinite(native_commands).all():
            raise ValueError("Nonfinite IWS model input")
        b, n = len(initial), 16
        # Builtin GRU is unidirectional; state[k] is independent of rows after k.
        action_states = self.action_prefix(self.normalize_commands(native_commands))[0]
        anchor = self.tokens(initial)
        register = anchor[:, None].expand(-1, c.temporal_slots, -1, -1)
        encoded_anchor_register = self._encode(register)
        anchor_encoded = encoded_anchor_register[:, -1]
        predictions, details = [], []
        for k in range(1, native_commands.shape[1]):
            encoded = encoded_anchor_register
            temporal = encoded.transpose(1, 2).reshape(b*n, c.temporal_slots, c.hidden_dim)
            # The same actual command-prefix state conditions all three
            # architectural slots, not invented past commands or timestamps.
            condition = action_states[:, k, None, None, :].expand(-1, n, c.temporal_slots, -1)
            condition = condition.reshape(b*n, c.temporal_slots, c.hidden_dim)
            hidden = self.predictor(temporal, condition)[:, -1].reshape(b, n, c.hidden_dim)
            residual = self.output_projection(hidden).float()
            scores = (self.transport_query(hidden) @ self.transport_key(anchor_encoded).transpose(-1, -2)
                      / math.sqrt(c.hidden_dim))
            scores = scores.float() + c.identity_bias * self.identity_transport
            transport = torch.softmax(scores, dim=-1)
            gate = torch.sigmoid(self.gate(hidden).float())
            innovation = residual  # The only mathematical intervention: no tanh.
            with torch.autocast(device_type=anchor.device.type, enabled=False):
                value = (1-gate)*anchor + gate*(transport @ anchor) + innovation
            diagnostic = ({"transport": transport, "gate": gate, "innovation": innovation}
                          if return_details else {})
            predictions.append(value)
            if return_details:
                details.append({**diagnostic, "prediction": value})
        result = self.flatten(torch.stack(predictions, dim=1))
        return (result, details) if return_details else result

    def predict(self, initial_features, native_commands):
        normalized = self._predict_normalized(self.normalize_features(initial_features), native_commands)
        return normalized*self.feature_std + self.feature_mean

    def forward(self, batch):
        prediction = self._predict_normalized(self.normalize_features(batch["initial_features"]), batch["commands"])
        target = batch["targets"]
        if target.shape != prediction.shape:
            raise ValueError("Targets must contain exactly H-1 future native feature rows")
        normalized_target = self.normalize_features(target)
        return {"loss": F.mse_loss(prediction, normalized_target),
                "standardized_predictions": prediction, "standardized_targets": normalized_target,
                "predictions": prediction*self.feature_std + self.feature_mean, "targets": target}

    @property
    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    @property
    def package_config(self):
        return {"package_kind": PACKAGE_KIND, "format_version": 1, "coordinate_layout": COORDINATE_LAYOUT,
                "input_contract": INPUT_CONTRACT, "model_config": asdict(self.config),
                **{key: getattr(self, key).detach().cpu().tolist()
                   for key in ("feature_mean", "feature_std", "command_mean", "command_std")}}


def from_config(config):
    if (config.get("package_kind") != PACKAGE_KIND or config.get("format_version") != 1
            or config.get("coordinate_layout") != COORDINATE_LAYOUT or config.get("input_contract") != INPUT_CONTRACT):
        raise ValueError("Unsupported unbounded IWS package; no v1/DROID/history compatibility")
    return SingleObservationWorldModel(config["model_config"],
        **{key: config[key] for key in ("feature_mean", "feature_std", "command_mean", "command_std")})
