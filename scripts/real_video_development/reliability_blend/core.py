"""Causal support-reliability control; no query images in the prediction API."""
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn


def gate(energy, alignment, prior, regularization):
    energy, alignment = np.asarray(energy, dtype=np.float64), np.asarray(alignment, dtype=np.float64)
    if (not np.isfinite(energy).all() or not np.isfinite(alignment).all()
            or (energy < 0).any() or not np.isfinite(prior) or not 0 <= prior <= 1
            or not np.isfinite(regularization) or regularization < 0):
        raise ValueError("Invalid gate moments/prior")
    denominator = energy + regularization
    result = np.full(np.broadcast_shapes(energy.shape, alignment.shape), prior, dtype=np.float64)
    np.divide(alignment + regularization * prior, denominator, out=result, where=denominator > 0)
    return np.clip(result, 0, 1)


def validate_prefix(prefix, prefix_actions, future_actions=None):
    if prefix.ndim != 3 or prefix.shape[1] != 13 or prefix_actions.ndim != 3 or prefix_actions.shape[:2] != (len(prefix), 12):
        raise ValueError("Require exactly thirteen observed frames and twelve connecting action blocks")
    for value in (prefix, prefix_actions, future_actions):
        if value is not None and not torch.isfinite(value).all():
            raise ValueError("Nonfinite causal inputs")
    if future_actions is not None and (future_actions.ndim != 3 or len(future_actions) != len(prefix)
            or future_actions.shape[1] not in (5, 10) or future_actions.shape[-1] != prefix_actions.shape[-1]):
        raise ValueError("Require five or ten recorded future action blocks")


def prefix_forecast(model, prefix, prefix_actions, one_step=False):
    validate_prefix(prefix, prefix_actions)
    if not one_step:
        return model.predict(prefix[:, :3], prefix_actions[:, :2], prefix_actions[:, 2:])
    # Each prediction receives only its own strictly earlier three observations.
    b, _, d = prefix.shape
    support = torch.stack([prefix[:, h:h+3] for h in range(10)], 1).reshape(b*10, 3, d)
    actions = torch.stack([prefix_actions[:, h:h+3] for h in range(10)], 1).reshape(b*10, 3, -1)
    return model.predict(support, actions[:, :2], actions[:, 2:]).reshape(b, 10, d)


def moments(framewise, factorized, target, std):
    difference = (factorized.double() - framewise.double()) / std.double()
    residual = (target.double() - framewise.double()) / std.double()
    return (difference.square().mean(-1), (difference * residual).mean(-1), residual.square().mean(-1))


class ReliabilityBlend(nn.Module):
    def __init__(self, framewise, factorized, prior, regularization, one_step=False):
        super().__init__()
        self.framewise, self.factorized = framewise, factorized
        self.prior, self.regularization, self.one_step = float(prior), float(regularization), bool(one_step)
        gate(0., 0., self.prior, self.regularization)
        if not torch.equal(framewise.feature_std, factorized.feature_std):
            raise ValueError("Donors must share frozen feature normalization")

    @property
    def feature_std(self):
        return self.framewise.feature_std

    def infer_gate(self, prefix, prefix_actions):
        first = prefix_forecast(self.framewise, prefix, prefix_actions, self.one_step)
        second = prefix_forecast(self.factorized, prefix, prefix_actions, self.one_step)
        a, b, _ = moments(first, second, prefix[:, 3:], self.feature_std)
        alpha = gate(a.mean(1).detach().cpu().numpy(), b.mean(1).detach().cpu().numpy(), self.prior, self.regularization)
        return torch.as_tensor(alpha, dtype=prefix.dtype, device=prefix.device)

    def predict(self, prefix, prefix_actions, future_actions):
        validate_prefix(prefix, prefix_actions, future_actions)
        alpha = self.infer_gate(prefix, prefix_actions)
        first = self.framewise.predict(prefix[:, -3:], prefix_actions[:, -2:], future_actions)
        second = self.factorized.predict(prefix[:, -3:], prefix_actions[:, -2:], future_actions)
        return first + alpha[:, None, None] * (second - first)


def load_gate(path, device="cpu"):
    # Imported lazily so causality/unit tests require no checkpoint I/O.
    from calibrate_residual import load_calibrated_package
    from shiftwm.real_video.data import sha256
    path = Path(path)
    record = json.loads(path.read_text())
    if record.get("format_version") != 1 or record.get("kind") != "development_causal_reliability_blend":
        raise ValueError("Unknown gate package")
    if record["fit_split"] != "train" or sha256(__file__) != record["inference_source_sha256"]:
        raise ValueError("Gate provenance differs")
    donors = []
    for key in ("framewise", "factorized"):
        item = record["donors"][key]
        target = path.parent / item["calibration"]
        if sha256(target) != item["sha256"]:
            raise ValueError("Calibration identity differs")
        donors.append(load_calibrated_package(target, device=device))
    return ReliabilityBlend(*donors, record["parameters"]["prior"], record["parameters"]["regularization"]).to(device).eval()
