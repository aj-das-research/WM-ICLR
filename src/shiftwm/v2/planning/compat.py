"""Small shims so the pinned stable-worldmodel planning stack runs in the project venv.

The project ``.venv`` deliberately does not carry ``stable_pretraining``,
``scikit-learn`` or ``hdf5plugin``. The official LeWM evaluation touches each of
them in exactly one narrow place, which we reproduce here verbatim:

* ``stable_pretraining.backbone.utils.vit_hf`` (0.1.8) -- the ViT-Tiny encoder
  builder referenced by the released ``config.json`` files.
* ``sklearn.preprocessing.StandardScaler`` -- used by ``eval.py`` to z-score
  actions (population std, near-zero variance -> scale 1).
* ``stable_pretraining.data.dataset_stats.ImageNet`` -- normalisation constants.
* ``hdf5plugin`` -- imported (for side effects only) by the swm HDF5 reader. The
  released LeWM HDF5 files are checked for non-builtin filters at load time.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np

IMAGENET_STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_VIT_SIZES = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
    "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
}


def vit_hf(size="tiny", patch_size=16, image_size=224, pretrained=False, use_mask_token=True, **kwargs):
    """Verbatim re-implementation of ``stable_pretraining.backbone.utils.vit_hf`` (v0.1.8)."""
    from transformers import ViTConfig, ViTModel

    if pretrained:
        raise ValueError("pretrained=True is not supported offline; released LeWM configs use False")
    params = dict(_VIT_SIZES[size])
    params["intermediate_size"] = params["hidden_size"] * 4
    params["image_size"] = image_size
    params["patch_size"] = patch_size
    params.update(kwargs)
    model = ViTModel(ViTConfig(**params), add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model


class StandardScaler:
    """Numerically identical to ``sklearn.preprocessing.StandardScaler`` (with_mean/with_std)."""

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.var_ = x.var(axis=0)  # population variance, as sklearn
        scale = np.sqrt(self.var_)
        # sklearn._handle_zeros_in_scale: constant features get scale 1
        eps = 10 * np.finfo(scale.dtype).eps
        scale[scale < eps] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x):
        x = np.asarray(x)
        out = (x - self.mean_) / self.scale_
        return out.astype(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64)

    def inverse_transform(self, x):
        x = np.asarray(x)
        out = x * self.scale_ + self.mean_
        return out.astype(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64)


def install_import_shims() -> None:
    """Register lightweight stand-ins for modules missing from the project venv."""
    try:
        importlib.import_module("hdf5plugin")
    except ImportError:
        sys.modules["hdf5plugin"] = types.ModuleType("hdf5plugin")

    try:
        importlib.import_module("stable_pretraining")
    except ImportError:
        spt = types.ModuleType("stable_pretraining")
        backbone = types.ModuleType("stable_pretraining.backbone")
        utils = types.ModuleType("stable_pretraining.backbone.utils")
        utils.vit_hf = vit_hf
        data = types.ModuleType("stable_pretraining.data")
        stats = types.ModuleType("stable_pretraining.data.dataset_stats")
        stats.ImageNet = IMAGENET_STATS
        spt.backbone, backbone.utils, spt.data, data.dataset_stats = backbone, utils, data, stats
        for name, mod in {
            "stable_pretraining": spt,
            "stable_pretraining.backbone": backbone,
            "stable_pretraining.backbone.utils": utils,
            "stable_pretraining.data": data,
            "stable_pretraining.data.dataset_stats": stats,
        }.items():
            sys.modules[name] = mod
