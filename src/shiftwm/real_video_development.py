"""Development-only residual calibration; the registered model is unchanged.

This simple calibration is a control for excessive predicted motion. It is not
a novel world-model architecture and has no physical safety guarantee.
"""
import math

import torch
from torch import nn


def residual_scale(displacement_energy, displacement_alignment):
    """Least-squares coefficient constrained to [0, 1], from TRAIN moments."""
    energy, alignment = float(displacement_energy), float(displacement_alignment)
    if not math.isfinite(energy) or not math.isfinite(alignment) or energy < 0:
        raise ValueError("Invalid calibration sufficient statistics")
    if energy <= 1e-20:
        return 0.0
    return max(0.0, min(1.0, alignment / energy))


class ResidualCalibratedWorldModel(nn.Module):
    """Contract a completed rollout toward the final observed feature vector.

    The base trajectory is computed unchanged. This does not feed contracted
    predictions back into the autoregressor and does not use query images.
    """
    def __init__(self, base, scale):
        super().__init__()
        if not math.isfinite(float(scale)) or not 0 <= float(scale) <= 1:
            raise ValueError("Residual scale must be finite and within [0,1]")
        self.base = base
        self.register_buffer("residual_scale", torch.tensor(float(scale), dtype=torch.float64))

    @property
    def feature_std(self):
        return self.base.feature_std

    @property
    def action_mean(self):
        return self.base.action_mean

    @property
    def action_std(self):
        return self.base.action_std

    def predict(self, support_features, support_actions, future_actions):
        prediction = self.base.predict(support_features, support_actions, future_actions)
        anchor = support_features[:, -1:]
        return anchor + self.residual_scale.to(prediction.dtype) * (prediction - anchor)
