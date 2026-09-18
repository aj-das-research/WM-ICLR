"""Development revision: frozen framewise world model, learned dynamics residual.

This module is separate from the original campaign. Its package embeds the full
donor state and loads without the donor checkpoint or the upstream download.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from .model import ModelConfig, ResidualFiLM, ShiftWorldModel, TransitionContext


PACKAGE_KIND = "frozen_framewise_with_dynamics_residual_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RevisionConfig:
    context_dim: int = 32
    context_hidden: int = 128
    history_length: int = 3
    mode: str = "framewise_dynamics_revision"
    freeze_visual: bool = True
    # Frozen alignment is logged, but contributes no trainable gradient.
    alignment_weight: float = 0.0
    dynamics_consistency_weight: float = 0.0
    observation_consistency_weight: float = 0.0

    def __post_init__(self):
        if self.context_dim < 1 or self.context_hidden < 1 or self.history_length < 2:
            raise ValueError("Positive context dimensions and at least two support frames required")
        if (self.mode != "framewise_dynamics_revision" or not self.freeze_visual
                or any((self.alignment_weight, self.dynamics_consistency_weight,
                        self.observation_consistency_weight))):
            raise ValueError("Revision fixes visual weights and optimizes prediction loss only")


class DynamicsRevision(nn.Module):
    """Same inference API as ShiftWorldModel; only new dynamics modules train."""
    def __init__(self, donor: ShiftWorldModel, config=None, provenance=None):
        super().__init__()
        if donor.config.mode != "framewise" or not donor.config.freeze_visual:
            raise ValueError("Revision requires a frozen-visual framewise donor")
        self.config = config if isinstance(config, RevisionConfig) else RevisionConfig(**(config or {}))
        if self.config.history_length != donor.config.history_length:
            raise ValueError("Revision and donor support lengths must match")
        self.donor = donor.requires_grad_(False).eval()
        self.latent_dim, self.action_dim = donor.latent_dim, donor.action_dim
        self.base_config = deepcopy(donor.base_config)
        self.provenance = deepcopy(provenance if provenance is not None else donor.provenance)
        self.dynamics_context = TransitionContext(self.latent_dim, self.action_dim,
                                                 self.config.context_hidden, self.config.context_dim)
        self.dynamics_adapter = ResidualFiLM(self.latent_dim, self.config.context_dim)
        self.train(False)

    @property
    def action_mean(self):
        return self.donor.action_mean

    @property
    def action_std(self):
        return self.donor.action_std

    def train(self, mode=True):
        super().train(mode)
        # A frozen parameter flag alone would not disable predictor dropout/BN.
        self.donor.eval()
        return self

    def encode_images(self, *args, **kwargs):
        return self.donor.encode_images(*args, **kwargs)

    def normalize_actions(self, actions):
        return self.donor.normalize_actions(actions)

    def correct_observations(self, features, observation_context):
        return self.donor.correct_observations(features, observation_context)

    def infer_context(self, support_features, support_actions):
        if support_features.ndim != 3 or support_features.shape[-1] != self.latent_dim:
            raise ValueError("Support features must have shape [B,H,D]")
        if support_actions.shape != (support_features.shape[0], support_features.shape[1] - 1,
                                     self.action_dim):
            raise ValueError("Exactly one executed action block per observed transition is required")
        obs = support_features.new_zeros(support_features.shape[0], self.config.context_dim)
        corrected = self.correct_observations(support_features, obs)
        return obs, self.dynamics_context(corrected, self.normalize_actions(support_actions))

    def predict_features(self, features, actions, dynamics_context):
        embedded = self.donor.base.action_encoder(self.normalize_actions(actions))
        embedded = self.dynamics_adapter(embedded, dynamics_context)
        # Do not use no_grad here: derivatives must flow through the frozen
        # predictor to the new action adapter, without updating donor weights.
        return self.donor.base.predict(features, embedded)

    def goal_embedding(self, *args, **kwargs):
        return self.donor.goal_embedding(*args, **kwargs)

    # Retain the original exact feature, query, target and recursive indexing.
    _features = ShiftWorldModel._features
    forward = ShiftWorldModel.forward
    rollout_features = ShiftWorldModel.rollout_features
    rollout = ShiftWorldModel.rollout

    def export_config(self):
        return {"package_kind": PACKAGE_KIND, "revision_config": asdict(self.config),
                "donor_config": self.donor.export_config(), "provenance": self.provenance,
                "model_config": asdict(self.config)}


def load_revision_package(directory, device="cpu"):
    """Load an embedded full model with weights_only=True and strict state keys."""
    from .upstream import create_base
    path = Path(directory)
    state = torch.load(path if path.is_file() else path / "model.pt",
                       map_location="cpu", weights_only=True)
    config = state["config"]
    if config.get("format_version") != 1 or config.get("package_kind") != PACKAGE_KIND:
        raise ValueError("Not a supported dynamics-revision package")
    donor_config = config["donor_config"]
    donor = ShiftWorldModel(create_base(donor_config["base_config"]), donor_config["base_config"],
                            ModelConfig(**donor_config["model_config"]), donor_config["action_mean"],
                            donor_config["action_std"], donor_config["provenance"])
    model = DynamicsRevision(donor, config["revision_config"], config["provenance"])
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), state


def completed_framewise_source(directory):
    """Validate completed seed0 donor metadata and record the immutable sources."""
    package = Path(directory)
    if package.name != "best":
        raise ValueError("Require a completed framewise validation-best donor package")
    read = lambda path: json.loads(Path(path).read_text())
    config = read(package / "config.json")
    run = read(package.parent / "run_config.json")
    summary = read(package.parent / "training_summary.json")
    if config["model_config"]["mode"] != "framewise" or int(run["seed"]) != 0:
        raise ValueError("Donor must be framewise training seed0")
    if (summary.get("status") != "completed" or run["epochs"] != 30
            or summary.get("completed_epochs") != 30):
        raise ValueError("Donor must have completed all 30 epochs")
    rows = [json.loads(line) for line in (package.parent / "metrics.jsonl").read_text().splitlines()
            if line.strip()]
    if sorted(row["epoch"] for row in rows) != list(range(1, 31)):
        raise ValueError("Donor epoch history must be complete and unique")
    values = [float(row["val"]["prediction_loss"]) for row in rows]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Donor validation metrics must be finite")
    best = min(values)
    if not math.isclose(best, summary["best_validation_prediction_loss"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Donor validation summary disagrees with epoch history")
    paths = {"model": package / "model.pt", "config": package / "config.json",
             "run_config": package.parent / "run_config.json",
             "training_summary": package.parent / "training_summary.json",
             "metrics": package.parent / "metrics.jsonl"}
    return {"path": str(package.resolve()), "config": config, "run_config": run,
            "best_metric": best,
            "best_epochs": [r["epoch"] for r, value in zip(rows, values)
                            if math.isclose(value, best, rel_tol=1e-10, abs_tol=1e-12)],
            "hashes": {name + "_sha256": file_sha256(path) for name, path in paths.items()}}


def validate_donor_state(state, source):
    if state["config"] != source["config"]:
        raise ValueError("Donor embedded and external configurations differ")
    if state["epoch"] not in source["best_epochs"] or not math.isclose(
            state["best_metric"], source["best_metric"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Donor state is not the recorded validation-best epoch")
