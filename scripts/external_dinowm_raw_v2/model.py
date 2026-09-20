"""Package interface around the frozen, reviewed official DINO-WM adapter."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F

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
COORDINATES = "raw_channel_major_384x4x4_dino_features_normalized_actions"


class ExternalDinoWM(adapter.ExternalDinoWM):
    MODES = ("official_raw_one_step", "official_raw_recursive_h10")

    def __init__(self, config, **normalization):
        config = dict(config)
        if config.get("mode") not in self.MODES or {k: v for k, v in config.items() if k != "mode"} != ARCHITECTURE:
            raise ValueError("External DINO-WM architecture/objective differs from fixed study")
        super().__init__(**normalization)
        self.config = SimpleNamespace(**config)

    def forward(self, batch, objective=None):
        objective = self.config.mode if objective is None else objective
        features, actions = batch["features"].float(), batch["actions"].float()
        b = len(features)
        if features.shape != (b, 13, 6144) or actions.shape != (b, 12, 35):
            raise ValueError("Expected thirteen raw grids and twelve action blocks")
        if objective == "official_raw_one_step":
            prediction = self.predict_three_slots(features[:, :3], self.normalized_actions(actions[:, :3]))
            target = features[:, 1:4].detach()
        elif objective == "official_raw_recursive_h10":
            prediction = self.predict(features[:, :3], actions[:, :2], actions[:, 2:])
            target = features[:, 3:13].detach()
        else:
            raise ValueError("Unknown raw-coordinate objective")
        return {"loss": F.mse_loss(prediction, target), "raw_predictions": prediction,
                "raw_targets": target, "standardized_predictions": self.normalized_features(prediction),
                "standardized_targets": self.normalized_features(target)}

    def predict(self, support, past_actions, future_actions):
        b = len(support)
        if (support.shape != (b, 3, 6144) or past_actions.shape != (b, 2, 35)
                or future_actions.ndim != 3 or future_actions.shape[0] != b
                or future_actions.shape[2] != 35 or not 1 <= future_actions.shape[1] <= 10):
            raise ValueError("Expected three observations, two past blocks and 1–10 future blocks")
        observations = support.float()
        actions = self.normalized_actions(torch.cat((past_actions, future_actions), dim=1))
        forecasts = []
        for step in range(future_actions.shape[1]):
            predicted = self.predict_three_slots(observations, actions[:, step:step + 3])[:, -1:]
            forecasts.append(predicted)
            observations = torch.cat((observations[:, 1:], predicted), dim=1)
        return torch.cat(forecasts, dim=1)

    def predict_normalized(self, support, past_actions, future_actions):
        return self.normalized_features(self.predict(support, past_actions, future_actions))

    @property
    def package_config(self):
        return {"format_version": 1, "model_family": "adapted_official_dinowm_raw_v2",
                "upstream_revision": adapter.REVISION,
                "coordinate_layout": COORDINATES,
                "model_config": vars(self.config).copy(),
                **{k: getattr(self, k).detach().cpu().tolist() for k in
                   ("feature_mean", "feature_std", "action_mean", "action_std")}}


def from_config(config):
    if (config.get("format_version") != 1 or config.get("model_family") != "adapted_official_dinowm_raw_v2"
            or config.get("coordinate_layout") != COORDINATES
            or config.get("upstream_revision") != adapter.REVISION):
        raise ValueError("Unsupported external DINO-WM package")
    return ExternalDinoWM(config["model_config"], **{k: config[k] for k in
                          ("feature_mean", "feature_std", "action_mean", "action_std")})
