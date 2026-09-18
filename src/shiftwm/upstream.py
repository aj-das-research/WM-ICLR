"""Strict, weights-only construction of the pinned upstream LeWorldModel.

The JEPA and transformer implementations are imported without modification from
the pinned checkout or byte-identical bundled source. The Hugging Face ViT
constructor reproduces stable_pretraining's vit_hf configuration without importing
its training framework. Model packages therefore remain loadable without a
separate git checkout of upstream code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(__file__).resolve().parent / "vendor" / "lewm"


def _source_module(name: str, filename: str):
    manifest = json.loads((VENDOR / "NOTICE.json").read_text())
    external = ROOT / "external" / "le-wm" / filename
    path = external if external.is_file() else VENDOR / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["files"][filename]:
        raise RuntimeError(f"Upstream source differs from pinned revision {manifest['revision']}: {path}")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def create_base(config: dict) -> nn.Module:
    from transformers import ViTConfig, ViTModel

    modules = _source_module("shiftwm_upstream_module", "module.py")
    jepa = _source_module("shiftwm_upstream_jepa", "jepa.py")
    sizes = {"tiny": (192, 12, 3), "small": (384, 12, 6), "base": (768, 12, 12)}
    enc = config["encoder"]
    width, layers, heads = sizes[enc["size"]]
    vit = ViTModel(ViTConfig(
        hidden_size=width, num_hidden_layers=layers, num_attention_heads=heads,
        intermediate_size=4 * width, image_size=enc["image_size"],
        patch_size=enc["patch_size"],
    ), add_pooling_layer=False, use_mask_token=enc.get("use_mask_token", False))
    clean = lambda cfg: {k: v for k, v in cfg.items() if not k.startswith("_")}
    def projection(key):
        args = clean(config[key])
        args["norm_fn"] = nn.BatchNorm1d
        return modules.MLP(**args)
    return jepa.JEPA(
        encoder=vit,
        predictor=modules.ARPredictor(**clean(config["predictor"])),
        action_encoder=modules.Embedder(**clean(config["action_encoder"])),
        projector=projection("projector"), pred_proj=projection("pred_proj"),
    )


def load_base(directory: str | Path):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model = create_base(config)
    weights = directory / "weights.pt"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    provenance = {"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                  "path": str(directory.resolve())}
    source = directory / "provenance.json"
    if source.exists():
        provenance["download"] = json.loads(source.read_text())
    return model.eval(), config, provenance
