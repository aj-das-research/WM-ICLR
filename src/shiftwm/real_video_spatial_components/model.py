"""Complete the observed-anchor mixing × innovation-bound factorial design.

Only the two missing cells are implemented. The constructor instantiates the
exact corresponding frozen control before changing its descriptive mode name;
initial tensors, RNG consumption, and active parameter masks are preserved.
"""
from dataclasses import asdict
import math

import torch
from shiftwm.real_video_spatial.model import SpatialWorldModel, SpatialConfig

PACKAGE_KIND = "shiftwm_real_video_spatial_components_v1"
SCHEMA = "observed_anchor_mixing_x_innovation_bound_v1"
MODES = ("bounded_additive", "unbounded_transport")
CONTROL_FOR = {"bounded_additive": "anchored_additive", "unbounded_transport": "transport"}


class ComponentWorldModel(SpatialWorldModel):
    # The frozen constructor validates its corresponding original control mode.
    MODES = SpatialWorldModel.MODES + MODES

    def __init__(self, config, feature_mean, feature_std, action_mean, action_std):
        config = dict(config) if isinstance(config, dict) else asdict(config)
        mode = config.get("mode")
        if mode not in MODES:
            raise ValueError("Only the two registered component arms are supported")
        super().__init__({**config, "mode": CONTROL_FOR[mode]}, feature_mean, feature_std, action_mean, action_std)
        self.config.mode = mode

    def _predict_normalized(self, support, past_actions, future_actions, return_details=False):
        c = self.config
        b, n = support.shape[0], c.grid_size ** 2
        if (support.ndim != 3 or support.shape[1:] != (3, c.feature_dim)
                or past_actions.shape != (b, 2, c.action_dim) or future_actions.ndim != 3
                or future_actions.shape[0] != b or future_actions.shape[1] < 1
                or future_actions.shape[2] != c.action_dim):
            raise ValueError("Expected three observed frames, two past actions, and a nonempty future action prefix")
        past = self.normalize_actions(past_actions)
        future = self.normalize_actions(future_actions)
        action_states = self.action_prefix(torch.cat((past, future), 1))[0]
        observed = self.tokens(support)
        anchor = observed[:, -1]
        encoded_support = self._encode(observed)
        context = self.context(encoded_support.mean(2), past)
        anchor_encoded = encoded_support[:, -1]
        predictions, details = [], []
        for t in range(future.shape[1]):
            encoded = self.context_adapter(encoded_support.reshape(b, 3*n, c.hidden_dim), context).reshape(b, 3, n, c.hidden_dim)
            action_history = torch.cat((action_states[:, :2], action_states[:, t+2:t+3]), 1)
            temporal = encoded.transpose(1, 2).reshape(b*n, 3, c.hidden_dim)
            actions = action_history[:, None].expand(b, n, 3, c.hidden_dim).reshape(b*n, 3, c.hidden_dim)
            hidden = self.predictor(temporal, actions)[:, -1].reshape(b, n, c.hidden_dim)
            residual = self.output_projection(hidden).float()
            if c.mode == "bounded_additive":
                innovation = c.innovation_bound * torch.tanh(residual)
                value = anchor + innovation
                detail = {"innovation": innovation}
            else:
                scores = self.transport_query(hidden) @ self.transport_key(anchor_encoded).transpose(-1, -2) / math.sqrt(c.hidden_dim)
                scores = scores.float() + c.identity_bias * self.identity_transport
                transport = torch.softmax(scores, -1)
                gate = torch.sigmoid(self.gate(hidden).float())
                innovation = residual
                with torch.autocast(device_type=anchor.device.type, enabled=False):
                    value = (1-gate)*anchor + gate*(transport @ anchor) + innovation
                detail = {"transport": transport, "gate": gate, "innovation": innovation}
            predictions.append(value)
            if return_details:
                details.append(detail)
        result = self.flatten(torch.stack(predictions, 1))
        return (result, details) if return_details else result

    @property
    def package_config(self):
        return {**super().package_config, "component_schema": SCHEMA}


def from_config(config):
    if (config.get("format_version") != 1 or config.get("component_schema") != SCHEMA
            or config.get("coordinate_layout") != "channel_major_384x4x4_shared_channel_normalization"
            or config.get("package_kind", PACKAGE_KIND) != PACKAGE_KIND):
        raise ValueError("Unsupported spatial component package identity")
    return ComponentWorldModel(config["model_config"], **{key: config[key] for key in ("feature_mean", "feature_std", "action_mean", "action_std")})
