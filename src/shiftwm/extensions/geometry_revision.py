"""Isolated observation-gain capacity ablation; original sources stay frozen.

Only the observation FiLM gain changes. Its identity value, derivative at zero,
translation and parameter shapes match the original; higher-order curvature and
attainable range differ. No auxiliary loss or action-adapter change is added.

The existing transactional package writer handles tensor/RNG persistence.
Extension model-format version2 requires THIS constructor: the original generic
loader rejects it rather than silently reconstructing the narrow-gain adapter.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from shiftwm.model import ResidualFiLM
from . import checkpoint as storage
from . import model as original
from . import train as training


GEOMETRY_KIND = "observation_gain_capacity_v1"
GEOMETRY_CONFIG = {
    "kind": GEOMETRY_KIND,
    "gain_formula": "exp(log(4)*tanh(0.1*scale/log(4)))",
    "gain_bounds": [0.25, 4.0],
    "gain_at_zero": 1.0,
    "scale_derivative_at_zero": 0.1,
    "translation": "unchanged",
    "dynamics_adapter": "unchanged_narrow_residual_film",
    "additional_loss": None,
}
MODES = ("factorized", "constant_dynamics")


class WideObservationFiLM(ResidualFiLM):
    """Same affine parameters as ResidualFiLM; wider, derivative-matched gain."""

    def forward(self, value, context):
        scale, shift = self.affine(context).chunk(2, dim=-1)
        log_range = math.log(4.)
        gain = torch.exp(log_range * torch.tanh(0.1 * scale / log_range))
        return value * gain[:, None] + shift[:, None]


class GeometryRevision(original.ExtensionWorldModel):
    def __init__(self, base, base_config, mode, action_mean, action_std,
                 architecture="transformer", provenance=None, model_config=None):
        if mode not in MODES:
            raise ValueError("Geometry ablation requires factorized or constant-dynamics context")
        super().__init__(base, base_config, mode, action_mean, action_std,
                         architecture, provenance, model_config)
        previous = self.observation_adapter
        self.observation_adapter = WideObservationFiLM(self.latent_dim, self.config.context_dim)
        self.observation_adapter.load_state_dict(previous.state_dict(), strict=True)
        self.observation_adapter.requires_grad_(True)
        self.train(False)

    @property
    def package_config(self):
        config = super().package_config
        config["extension_format_version"] = 2
        config["geometry_revision"] = deepcopy(GEOMETRY_CONFIG)
        return config


def from_extension(model):
    """Copy an initialized original model without changing RNG or parameter values.

    On nonzero trained observation weights this deliberately changes the
    function; exact functional equivalence is guaranteed only at identity
    initialization. Registered experiments start from released initialization.
    """
    if isinstance(model, GeometryRevision):
        raise ValueError("Model already uses the geometry revision")
    config = model.package_config
    provenance = deepcopy(model.provenance)
    provenance["geometry_revision"] = {"kind": GEOMETRY_KIND,
                                       "implementation_sha256": storage.file_sha256(__file__)}
    with torch.random.fork_rng(devices=[]):
        revised = GeometryRevision(model.base, model.base_config, model.extension_mode,
                                   config["action_mean"], config["action_std"],
                                   model.architecture, provenance, config["model_config"])
        revised.load_state_dict(model.state_dict(), strict=True)
    return revised.to(next(model.parameters()).device).eval()


def build_model(pretrained_dir, architecture, mode, action_mean, action_std, seed=0):
    return from_extension(original.build_model(pretrained_dir, architecture, mode,
                                               action_mean, action_std, seed))


def build_from_config(config):
    if config.get("extension_format_version") != 2 or config.get("geometry_revision") != GEOMETRY_CONFIG:
        raise ValueError("Unsupported observation-geometry package contract")
    if config.get("architecture") not in original.ARCHITECTURES or config.get("mode") not in MODES:
        raise ValueError("Unsupported observation-geometry architecture or mode")
    base_config = config["upstream_base_config"]
    base = original.configure_predictor(original.create_base(base_config), base_config, config["architecture"])
    return GeometryRevision(base, base_config, config["mode"], config["action_mean"], config["action_std"],
                            config["architecture"], config["provenance"], config["model_config"])


def load_geometry_package(directory, device="cpu"):
    _, state = storage.read_package(directory)
    model = build_from_config(state["config"])
    model.load_state_dict(state["state_dict"], strict=True)
    for key, value in model.package_config.items():
        if state["config"].get(key) != value:
            raise ValueError(f"Reconstructed geometry configuration differs: {key}")
    return model.to(device).eval(), state


# Explicit public aliases for the new revision's caller. No original loader or
# trainer global is replaced; storage only serializes and validates tensors.
load_package = load_geometry_package
save_package = storage.save_package
resume_training = storage.resume_training


def initialize(config):
    if (config.get("geometry_revision") != GEOMETRY_KIND or config.get("architecture") != "transformer"
            or config.get("mode") not in MODES or config.get("seed") not in (0, 1, 2)):
        raise ValueError("Declare one of the six registered transformer geometry conditions")
    root = Path(__file__).resolve().parents[3]
    destination = Path(config["output_dir"]).resolve()
    original_runs = root / "runs/extensions"
    if destination == original_runs or original_runs in destination.parents:
        raise ValueError("Use a new namespace; original extension runs are immutable")
    model = training.initialize(config)
    if model.provenance.get("extension_environment") != "drone":
        raise ValueError("Initial geometry study is restricted to the registered drone domain")
    return from_extension(model)


def validate_completed(output, identity):
    """Same completion gates as the frozen trainer, with an explicit v2 loader.

    The original completion helper calls its v1 constructor internally; this
    narrow adapter preserves its gates without modifying or monkeypatching it.
    """
    output = Path(output)
    summary = json.loads((output / "training_summary.json").read_text())
    run = json.loads((output / "run_config.json").read_text())
    training.validate_config(run)
    if (run.get("geometry_revision") != GEOMETRY_KIND or run.get("mode") not in MODES
            or run.get("architecture") != "transformer" or run.get("seed") not in (0, 1, 2)
            or summary.get("status") != "completed" or summary.get("completed_epochs") != 30
            or summary.get("training_identity") != identity
            or summary.get("validation_metric") != "recursive_mse_all_query_steps"):
        raise ValueError("Geometry revision is not complete under the declared identity/criterion")
    rows = training.metric_rows(output)
    if any(type(r.get("epoch")) is not int for r in rows) or [r["epoch"] for r in rows] != list(range(1, 31)):
        raise ValueError("Expected all 30 unique, chronological metric epochs")
    values = [float(r["val"]["prediction_loss"]) for r in rows]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Nonfinite or negative recursive validation metrics")
    best = min(values)
    if (not math.isclose(best, summary["best_validation_prediction_loss"], rel_tol=1e-10, abs_tol=1e-12)
            or summary.get("step") != rows[-1].get("step")):
        raise ValueError("Geometry summary differs from completed validation history")
    for name in ("best", "last"):
        _, state = storage.read_package(output / name, require_training=True)
        config, metadata = state["config"], state["config"]["metadata"]
        if (config.get("extension_format_version") != 2 or config.get("geometry_revision") != GEOMETRY_CONFIG
                or metadata.get("training_identity") != identity
                or training.scientific_config(metadata.get("config", {})) != training.scientific_config(run)
                or metadata.get("selection") != training.SELECTION
                or metadata.get("validation_protocol") != training.VALIDATION_PROTOCOL
                or metadata.get("validation_precision") != "float32"
                or config.get("architecture") != run["architecture"] or config.get("mode") != run["mode"]
                or config.get("objective") != "recursive_mse_all_query_steps"):
            raise ValueError("Geometry package differs from its model/training protocol")
        identity_config = {k: v for k, v in config.items() if k not in {"format_version", "package_kind", "metadata"}}
        expected = hashlib.sha256(json.dumps({"config": training.scientific_config(run),
                                              "package_config": identity_config},
                                             sort_keys=True, allow_nan=False).encode()).hexdigest()
        if expected != identity or not math.isclose(state["best_metric"], best, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Geometry model identity or best metric differs")
        if name == "last" and (state["epoch"] != 30 or state["step"] != summary["step"] or state["progress"]):
            raise ValueError("Last geometry package does not finish all 30 epochs")
        if name == "best" and not any(r["epoch"] == state["epoch"] and math.isclose(
                r["val"]["prediction_loss"], best, rel_tol=1e-10, abs_tol=1e-12) for r in rows):
            raise ValueError("Best geometry package is not selected by recursive validation")
    load_geometry_package(output / "best", device="cpu")
    return summary


def train(config, resume_if_present=False):
    training.validate_config(config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    if str(config.get("device", "cuda")).startswith("cuda"):
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
    with training.training_lock(config["output_dir"]):
        model = initialize(config)
        output = Path(config["output_dir"])
        identity = training.training_identity(config, model)
        summary_path = output / "training_summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "completed":
            return validate_completed(output, identity)
        if resume_if_present and (output / "last/model.pt").exists():
            config = {**config, "resume": str(output / "last")}
        return training.fit(model, config, training.make_dataset(config, "train"), training.make_dataset(config, "val"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-if-present", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_runtime_seconds is not None:
        if not math.isfinite(args.max_runtime_seconds) or args.max_runtime_seconds <= 0:
            parser.error("max-runtime-seconds must be finite and positive")
        config["max_runtime_seconds"] = args.max_runtime_seconds
    result = train(config, args.resume_if_present)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "completed" else 75


if __name__ == "__main__":
    raise SystemExit(main())
