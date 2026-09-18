"""Controlled rollout objectives over an immutable framewise world-model donor.

Every training window has eight frames: context uses frames 0:3 and executed
actions 0:2; both objectives predict canonical targets 3:8. The constant-context
control retains the same modules and trainable flags but supplies zero inputs to
the context network. It controls access to episode information, not effective
capacity: zero inputs necessarily leave some input weights without gradients.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .dynamics_revision import DynamicsRevision, RevisionConfig
from .model import ModelConfig, ShiftWorldModel


PACKAGE_KIND = "frozen_framewise_with_rollout_residual_v1"


@dataclass
class RolloutRevisionConfig:
    context_dim: int = 32
    context_hidden: int = 128
    history_length: int = 3
    mode: str = "framewise_rollout_revision"
    context_mode: str = "inferred"
    objective: str = "recursive"
    freeze_visual: bool = True
    alignment_weight: float = 0.0
    dynamics_consistency_weight: float = 0.0
    observation_consistency_weight: float = 0.0

    def __post_init__(self):
        if any(type(value) is not int or value < 1
               for value in (self.context_dim, self.context_hidden)):
            raise ValueError("Context dimensions must be positive integers")
        if type(self.history_length) is not int or self.history_length != 3:
            raise ValueError("Rollout revision requires exactly three support frames")
        if self.context_mode not in {"inferred", "constant"}:
            raise ValueError("context_mode must be inferred or constant")
        if self.objective not in {"teacher_forced", "recursive"}:
            raise ValueError("objective must be teacher_forced or recursive")
        if (self.mode != "framewise_rollout_revision" or self.freeze_visual is not True
                or any((self.alignment_weight, self.dynamics_consistency_weight,
                        self.observation_consistency_weight))):
            raise ValueError("Rollout revision fixes the donor and optimizes prediction loss only")


class RolloutRevision(DynamicsRevision):
    """Frozen donor inference with support-only context and explicit objectives.

    Donor visual/goal calibration, action encoder, predictor, dropout and batch
    statistics remain fixed. The inherited predictor call intentionally permits
    autograd through its frozen weights into the new action residual.
    """

    def __init__(self, donor: ShiftWorldModel, config=None, provenance=None):
        rollout_config = (config if isinstance(config, RolloutRevisionConfig)
                          else RolloutRevisionConfig(**(config or {})))
        # Reuse the established module construction and frozen-donor inference
        # contract without changing the source-pinned first revision.
        inherited = RevisionConfig(context_dim=rollout_config.context_dim,
                                   context_hidden=rollout_config.context_hidden,
                                   history_length=rollout_config.history_length)
        super().__init__(donor, inherited, provenance)
        self.config = rollout_config

    def infer_context(self, support_features, support_actions):
        if support_features.ndim != 3 or support_features.shape[-1] != self.latent_dim:
            raise ValueError("Support features must have shape [B,H,D]")
        if support_actions.shape != (support_features.shape[0], support_features.shape[1] - 1,
                                     self.action_dim):
            raise ValueError("Exactly one executed action block per observed transition is required")
        obs = support_features.new_zeros(support_features.shape[0], self.config.context_dim)
        corrected = self.correct_observations(support_features, obs)
        actions = self.normalize_actions(support_actions)
        if self.config.context_mode == "constant":
            # Zero *normalized* inputs, not raw zero actions (which could become
            # nonzero after normalization). Network biases remain learnable.
            corrected, actions = torch.zeros_like(corrected), torch.zeros_like(actions)
        return obs, self.dynamics_context(corrected, actions)

    def _objective(self, batch, objective):
        features = self._features(batch)
        with torch.no_grad():
            target = self._features(batch, "reference_").detach()
        actions = batch["actions"]
        if features.ndim != 3 or features.shape[1:] != (8, self.latent_dim):
            raise ValueError("Rollout training requires eight frames with the donor latent dimension")
        if target.shape != features.shape:
            raise ValueError("Canonical targets must match the eight-frame feature shape")
        if actions.shape != (features.shape[0], 7, self.action_dim):
            raise ValueError("Rollout training requires seven chronological action blocks")
        h = self.config.history_length
        obs, dyn = self.infer_context(features[:, :h], actions[:, :h - 1])
        corrected = self.correct_observations(features, obs)
        if objective == "teacher_forced":
            # The first window ends at frame2 and includes candidate action2;
            # its target is frame3. Subsequent windows end at frames3..6.
            positions = range(h - 1, features.shape[1] - 1)
            states = torch.stack([corrected[:, t - h + 1:t + 1] for t in positions], 1)
            action_windows = torch.stack([actions[:, t - h + 1:t + 1] for t in positions], 1)
            b, q = states.shape[:2]
            query_dyn = dyn[:, None].expand(-1, q, -1).reshape(b * q, -1)
            pred = self.predict_features(states.reshape(b * q, h, -1),
                                         action_windows.reshape(b * q, h, -1), query_dyn)[:, -1]
            pred = pred.reshape(b, q, self.latent_dim)
        elif objective == "recursive":
            # Existing rollout appends predictions without detaching or applying
            # visual calibration again. No observed query frames enter it.
            pred = self.rollout_features(features[:, :h], actions[:, :h - 1],
                                         actions[:, h - 1:], contexts=(obs, dyn))
        else:
            raise ValueError("Unknown rollout objective")
        targets = target[:, h:]
        prediction_loss = F.mse_loss(pred, targets)
        zero = pred.new_zeros(())
        return {"loss": prediction_loss, "prediction_loss": prediction_loss,
                "alignment_loss": F.mse_loss(corrected[:, h:], targets),
                "dynamics_consistency_loss": zero, "observation_consistency_loss": zero,
                "observation_context_std": obs.float().std(0, unbiased=False).mean(),
                "dynamics_context_std": dyn.float().std(0, unbiased=False).mean(),
                "predictions": pred, "targets": targets,
                "observation_context": obs, "dynamics_context": dyn}

    def forward(self, batch):
        return self._objective(batch, self.config.objective)

    def recursive_validation(self, batch):
        """Common five-step recursive MSE, regardless of training objective.

        The caller controls inference_mode/autocast; leaving this differentiable
        also permits checking recursion gradients in correctness fixtures.
        """
        return self._objective(batch, "recursive")

    def export_config(self):
        config = asdict(self.config)
        return {"package_kind": PACKAGE_KIND, "revision_config": config,
                "donor_config": self.donor.export_config(), "provenance": self.provenance,
                "model_config": dict(config)}


def load_rollout_package(directory, device="cpu"):
    """Strict weights-only reconstruction; the full donor is embedded locally."""
    from .upstream import create_base

    path = Path(directory)
    state = torch.load(path if path.is_file() else path / "model.pt",
                       map_location="cpu", weights_only=True)
    config = state["config"]
    if config.get("format_version") != 1 or config.get("package_kind") != PACKAGE_KIND:
        raise ValueError("Not a supported rollout-revision package")
    if config.get("model_config") != config.get("revision_config"):
        raise ValueError("Embedded rollout configuration aliases disagree")
    revision = RolloutRevisionConfig(**config["revision_config"])
    donor_config = config["donor_config"]
    donor = ShiftWorldModel(create_base(donor_config["base_config"]), donor_config["base_config"],
                            ModelConfig(**donor_config["model_config"]), donor_config["action_mean"],
                            donor_config["action_std"], donor_config["provenance"])
    model = RolloutRevision(donor, revision, config["provenance"])
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), state
