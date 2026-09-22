"""Post-hoc IWS RGB decoder: immutable recipe and official decoder adapter.

This auxiliary fit does not change any feature predictor or primary endpoint.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
REPORT = ROOT / 'reports/metrics_completion_v1/rgb'
REG = REPORT / 'registration.json'
TASKS = ('pusht', 'bimanual_box', 'bimanual_rope')
UPSTREAM = 'external/official-dino-wm/models/decoder/transposed_conv.py'
UPSTREAM_SHA = 'c1327257b0f964edb60c66ff528d6c9d24ae5d43322a13ef72eacd716374b0ac'
RECIPE = {
    'schema': 'iws_shared_rgb_decoder_recipe_v1', 'tasks': list(TASKS), 'seed': 173,
    'epochs': 30, 'batch_size': 32, 'lr': 0.0001, 'optimizer': 'Adam',
    'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0,
    'frame_stride': 5, 'frame_start': 0, 'resolution': [224, 224],
    'target': 'native RGB /255; CPU torch bilinear align_corners=False antialias=True; *255 round clamp uint8; /255 for loss',
    'input': 'raw frozen DINOv2-small 4x4 channel-major 6144, standardized by existing train-only per-channel statistics',
    'upstream_commit': '0a9492fa12044b852ae9e001cc74604b79c8bb0c',
    'upstream_source': UPSTREAM, 'upstream_sha256': UPSTREAM_SHA,
    'architecture': {'emb_dim': 6144, 'depth': 32, 'kernel_size': 5, 'stride': 3, 'observation_shape': [3, 224, 224]},
    'train_loss': 'unclipped RGB MSE, all selected native frames equally weighted',
    'selection': 'minimum FP32 unclipped RGB MSE: equal native selected frames within trajectory, equal development trajectories; earliest strict minimum',
    'evaluation_pixels': 'decoder output clamped to [0,1], identical actual target preprocessing',
    'training_precision': 'float32; autocast disabled; TF32 disabled', 'cpu_threads': 8,
    'selection_split': 'internal_development', 'fit_split': 'internal_train',
    'shared_across': 'all four frozen predictors, persistence, all three predictor seeds; decoder seed 173 fixed per task',
    'uncertainty': 'conditional on one shared decoder fit per task; not decoder-seed uncertainty',
    'scope': 'post-hoc reconstruction/readout added after feature-space development and reserved results; not an original preregistered RGB endpoint',
    'reserved_payload_access': False, 'resource_profile_updates': 20,
    'resource_profile_train_frames': 256, 'epoch_boundary_seconds': 25200,
    'maximum_restarts': 8,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def internal_cache(task):
    require(task in TASKS, 'Unknown task')
    runner = module(ROOT / 'scripts/real_video_iws/train.py', '_rgb_frozen_iws_runner')
    return runner.open_cache(read(ROOT / 'configs/real_video_iws/training_v1.json'), task)


def selected_indices(frames):
    require(isinstance(frames, int) and frames > 0, 'Invalid native frame count')
    return np.arange(0, frames, RECIPE['frame_stride'], dtype=np.int64)


def target_pixels(rgb):
    require(rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[-1] == 3, 'Expected real RGB uint8')
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
    return (x * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous().numpy()


class RGBDecoder(nn.Module):
    """Official unchanged CNN; constructor adapts the complete 4x4 input vector."""
    def __init__(self, statistics):
        super().__init__()
        require(sha(ROOT / UPSTREAM) == UPSTREAM_SHA, 'Official decoder source changed')
        mean = torch.tensor(statistics['feature_mean'], dtype=torch.float32)
        std = torch.tensor(statistics['feature_std'], dtype=torch.float32)
        require(mean.shape == std.shape == (6144,) and torch.isfinite(mean).all() and torch.isfinite(std).all() and (std >= 1e-5).all(), 'Bad training statistics')
        self.register_buffer('feature_mean', mean); self.register_buffer('feature_std', std)
        upstream = module(ROOT / UPSTREAM, '_rgb_official_dino_decoder')
        self.decoder = upstream.TransposedConvDecoder(**RECIPE['architecture'])

    def forward(self, raw_features):
        require(raw_features.ndim == 2 and raw_features.shape[1] == 6144, 'Expected channel-major 4x4 raw features')
        normalized = (raw_features.float() - self.feature_mean) / self.feature_std
        # The official forward requires a horizon axis. No feature encoder is fit.
        prediction, _ = self.decoder(normalized[:, None, :])
        return prediction


def configure_cpu():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def checked_registration(require_review=True):
    reg = read(REG)
    require(reg.get('schema') == 'iws_shared_rgb_decoder_registration_v1' and reg.get('status') == 'frozen_before_decoder_training', 'Invalid RGB registration')
    require(reg['recipe'] == RECIPE, 'RGB recipe changed')
    if require_review:
        review = read(REPORT / 'source_review.json')
        require(review.get('status') == 'passed' and review.get('registration_sha256') == sha(REG), 'Independent RGB source review missing/stale')
    import cv2, einops
    environment={'torch':str(torch.__version__),'numpy':np.__version__,'opencv':cv2.__version__,'einops':einops.__version__,'cuda_build':torch.version.cuda}
    require(reg['environment']==environment,'RGB numerical software environment changed')
    for relative, expected in reg['dependencies'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts and not str(path).startswith(('data/features/iws_reserved', 'data/real_video/')), 'Unsafe source dependency')
        require(sha(ROOT / path) == expected, 'RGB source dependency changed: ' + relative)
    return reg
