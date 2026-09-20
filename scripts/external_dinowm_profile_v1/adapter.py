"""Pinned official DINO-WM computation on the declared DROID feature interface.

This is an explicitly adapted external predictor, not an unchanged reproduction
of the paper's images, proprioception, training recipe or benchmark. No data
loader or checkpoint loader is provided in this synthetic-profile namespace.
"""
from pathlib import Path
import hashlib
import types

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external/official-dino-wm"
REVISION = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
HASHES = {
    "models/vit.py": "60c77bdb03f3b2d565bfdb1695c9c193a802236763d33a03e79ed382b51f0670",
    "models/proprio.py": "b0f7c761c45a6a889dc51f6d9cf4539480d1a0bd2bd200ad44c4b9373a573c72",
}
MASK_ORIGINAL = "self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')"
MASK_PORTABLE = 'self.register_buffer("bias", generate_mask_matrix(NUM_PATCHES, NUM_FRAMES), persistent=False)'
OBJECTIVES = ("official_one_step_shifted", "matched_recursive_h10")


def load_source(relative, portable_mask=False):
    path = UPSTREAM / relative
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != HASHES[relative]:
        raise ValueError("Pinned upstream source differs: " + relative)
    text = source.decode()
    if portable_mask:
        if text.count(MASK_ORIGINAL) != 1:
            raise ValueError("Expected exactly one upstream CUDA mask construction")
        text = text.replace(MASK_ORIGINAL, MASK_PORTABLE)
    module = types.ModuleType("external_dinowm_profile_" + Path(relative).stem)
    module.__file__ = str(path)
    exec(compile(text, str(path), "exec"), module.__dict__)
    return module


def tokens(flat):
    if flat.shape[-1] != 6144:
        raise ValueError("Expected channel-major 384x4x4 features")
    return flat.reshape(*flat.shape[:-1], 384, 16).transpose(-1, -2)


def flatten(value):
    if value.shape[-2:] != (16, 384):
        raise ValueError("Expected sixteen 384-channel tokens")
    return value.transpose(-1, -2).reshape(*value.shape[:-2], 6144)


class ExternalDinoWM(nn.Module):
    """Official transformer and kernel-one action embedding, no proprioception.

    Inputs are raw cached coordinates. All normalization values must come from
    training only in a future study; the profiler supplies synthetic zero/one
    fixtures. The constructor intentionally fixes the full upstream width/depth.
    """
    def __init__(self, feature_mean, feature_std, action_mean, action_std):
        super().__init__()
        for name, value, width in (("feature_mean", feature_mean, 6144),
                                   ("feature_std", feature_std, 6144),
                                   ("action_mean", action_mean, 35),
                                   ("action_std", action_std, 35)):
            value = torch.as_tensor(value, dtype=torch.float32).flatten()
            if (len(value) != width or not torch.isfinite(value).all()
                    or (name.endswith("std") and (value <= 0).any())):
                raise ValueError("Invalid normalization: " + name)
            if name.startswith("feature"):
                grid = value.reshape(384, 16)
                if not torch.equal(grid, grid[:, :1].expand_as(grid)):
                    raise ValueError("Normalization must be shared per channel")
            self.register_buffer(name, value.clone())
        vit = load_source("models/vit.py", portable_mask=True)
        proprio = load_source("models/proprio.py")
        self.action_encoder = proprio.ProprioceptiveEmbedding(
            num_frames=1, tubelet_size=1, in_chans=35, emb_dim=10,
            use_3d_pos=False)
        self.predictor = vit.ViTPredictor(
            num_patches=16, num_frames=3, dim=394, depth=6, heads=16,
            mlp_dim=2048, pool="mean", dim_head=64, dropout=.1,
            emb_dropout=0.)

    def normalized_features(self, value):
        return (value.float() - self.feature_mean) / self.feature_std

    def normalized_actions(self, value):
        return (value.float() - self.action_mean) / self.action_std

    def predict_three_slots(self, normalized_history, normalized_actions):
        """Each slot predicts its next grid with full within-frame attention."""
        b = len(normalized_history)
        if normalized_history.shape != (b, 3, 6144) or normalized_actions.shape != (b, 3, 35):
            raise ValueError("Expected three grids and their three outgoing blocks")
        visual = tokens(normalized_history)
        action = self.action_encoder(normalized_actions)
        tiled = action.unsqueeze(2).expand(-1, -1, 16, -1)
        embedded = torch.cat((visual, tiled), dim=-1)
        predicted = self.predictor(embedded.reshape(b, 48, 394))
        return flatten(predicted.reshape(b, 3, 16, 394)[..., :384]).float()

    def predict_normalized(self, support, past_actions, future_actions):
        b = len(support)
        if (support.shape != (b, 3, 6144) or past_actions.shape != (b, 2, 35)
                or future_actions.ndim != 3 or future_actions.shape[0] != b
                or future_actions.shape[2] != 35 or not 1 <= future_actions.shape[1] <= 10):
            raise ValueError("Expected three observations, two past blocks and 1–10 future blocks")
        observations = self.normalized_features(support)
        actions = self.normalized_actions(torch.cat((past_actions, future_actions), dim=1))
        forecasts = []
        for step in range(future_actions.shape[1]):
            predicted = self.predict_three_slots(observations, actions[:, step:step + 3])[:, -1:]
            forecasts.append(predicted)
            # Full BPTT: predicted features remain in the next history's graph.
            observations = torch.cat((observations[:, 1:], predicted), dim=1)
        return torch.cat(forecasts, dim=1)

    def predict(self, support, past_actions, future_actions):
        value = self.predict_normalized(support, past_actions, future_actions)
        return value * self.feature_std + self.feature_mean

    def forward(self, batch, objective="matched_recursive_h10"):
        features, actions = batch["features"], batch["actions"]
        b = len(features)
        if features.shape != (b, 13, 6144) or actions.shape != (b, 12, 35):
            raise ValueError("Full profile window is thirteen grids and twelve action blocks")
        if objective == "official_one_step_shifted":
            prediction = self.predict_three_slots(
                self.normalized_features(features[:, :3]),
                self.normalized_actions(actions[:, :3]))
            # Official source applies one-step shifted loss to all three slots:
            # two observed next states and the first unobserved next state.
            target = self.normalized_features(features[:, 1:4]).detach()
        elif objective == "matched_recursive_h10":
            prediction = self.predict_normalized(features[:, :3], actions[:, :2], actions[:, 2:])
            target = self.normalized_features(features[:, 3:13]).detach()
        else:
            raise ValueError("Unknown objective")
        return {"loss": F.mse_loss(prediction, target),
                "standardized_predictions": prediction,
                "standardized_targets": target}


def synthetic_model():
    return ExternalDinoWM(torch.zeros(6144), torch.ones(6144),
                          torch.zeros(35), torch.ones(35))


def synthetic_batch(batch_size=128, seed=173):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {"features": torch.randn(batch_size, 13, 6144, generator=generator),
            "actions": torch.randn(batch_size, 12, 35, generator=generator)}
