"""Package interface around the frozen, reviewed official DINO-WM adapter."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "scripts/external_dinowm_profile_v1/adapter.py"
spec = importlib.util.spec_from_file_location("external_dinowm_training_frozen_adapter", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

ARCHITECTURE = {"history_length": 3, "feature_dim": 6144, "action_dim": 35,
                "num_patches": 16, "visual_channels": 384,
                "action_embedding_channels": 10, "predictor_dim": 394,
                "depth": 6, "heads": 16, "dim_head": 64, "mlp_dim": 2048,
                "dropout": .1, "emb_dropout": 0.}
COORDINATES = "channel_major_384x4x4_shared_channel_normalization"


class ExternalDinoWM(adapter.ExternalDinoWM):
    MODES = adapter.OBJECTIVES

    def __init__(self, config, **normalization):
        config = dict(config)
        if config.get("mode") not in self.MODES or {k: v for k, v in config.items() if k != "mode"} != ARCHITECTURE:
            raise ValueError("External DINO-WM architecture/objective differs from fixed study")
        super().__init__(**normalization)
        self.config = SimpleNamespace(**config)

    def forward(self, batch, objective=None):
        return super().forward(batch, self.config.mode if objective is None else objective)

    @property
    def package_config(self):
        return {"format_version": 1, "model_family": "adapted_official_dinowm_v1",
                "upstream_revision": adapter.REVISION,
                "coordinate_layout": COORDINATES,
                "model_config": vars(self.config).copy(),
                **{k: getattr(self, k).detach().cpu().tolist() for k in
                   ("feature_mean", "feature_std", "action_mean", "action_std")}}


def from_config(config):
    if (config.get("format_version") != 1 or config.get("model_family") != "adapted_official_dinowm_v1"
            or config.get("coordinate_layout") != COORDINATES
            or config.get("upstream_revision") != adapter.REVISION):
        raise ValueError("Unsupported external DINO-WM package")
    return ExternalDinoWM(config["model_config"], **{k: config[k] for k in
                          ("feature_mean", "feature_std", "action_mean", "action_std")})
