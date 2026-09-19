"""Private reuse of the immutable IWS optimizer/selector/checkpoint engine.

Only this private module instance changes package construction/kind. The v1
module and all registered files remain unmodified. Epoch weighting, AdamW,
cosine schedule, FP32 selector and exact epoch-boundary continuation are reused.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

from .model import PACKAGE_KIND, from_config

ROOT = Path(__file__).resolve().parents[3]
_name = __package__ + "._frozen_training"
_spec = importlib.util.spec_from_file_location(_name, ROOT / "src/shiftwm/real_video_iws/training.py")
engine = importlib.util.module_from_spec(_spec)
sys.modules[_name] = engine
_spec.loader.exec_module(engine)
engine.PACKAGE_KIND = PACKAGE_KIND
engine.base.PACKAGE_KIND = PACKAGE_KIND
engine.base.from_config = from_config

SELECTION, PRECISION = engine.SELECTION, engine.PRECISION
atomic_json, digest, seed_everything = engine.atomic_json, engine.digest, engine.seed_everything
epoch_pass, fit = engine.epoch_pass, engine.fit
read_package, load_package = engine.read_package, engine.load_package
validate_completed, export_inference_package = engine.validate_completed, engine.export_inference_package
