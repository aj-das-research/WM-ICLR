"""Released LeWM checkpoints (HF ``weights.pt`` + ``config.json``) for the planning harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .compat import install_import_shims
from .interface import PlanningPredictor

ROOT = Path(__file__).resolve().parents[4]
PRETRAINED = {
    "pusht": ROOT / "data/pretrained/pusht",
    "reacher": ROOT / "data/pretrained/reacher",
    "tworoom": ROOT / "data/pretrained/tworooms",
    "cube": ROOT / "data/pretrained/cube",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lewm(ckpt_dir: str | Path) -> tuple[torch.nn.Module, dict]:
    """Instantiate ``stable_worldmodel.wm.lewm.LeWM`` from an HF mirror folder.

    This is what ``swm.wm.utils.load_pretrained(<folder>)`` does (the path used
    by the pinned ``eval_wm.py``); we do it explicitly to record hashes and to
    avoid the implicit ``$STABLEWM_HOME/checkpoints`` resolution.
    """
    install_import_shims()
    from hydra.utils import instantiate

    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    model = instantiate(cfg)
    sd = torch.load(ckpt_dir / "weights.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    meta = {
        "ckpt_dir": str(ckpt_dir),
        "weights_sha256": sha256(ckpt_dir / "weights.pt"),
        "config_sha256": sha256(ckpt_dir / "config.json"),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "config": cfg,
    }
    return model, meta


class LeWMPredictor(PlanningPredictor):
    """LeWM expressed through the pluggable interface (reference implementation).

    Numerically mirrors ``LeWM.rollout`` + ``GoalMSE`` (checked by
    ``scripts/v2/planning_eval.py --check-adapter``).
    """

    name = "lewm"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.history_size = getattr(model.predictor, "num_frames", 3)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model.encode({"pixels": pixels})["emb"]

    def rollout(self, latent_hist, action_hist, actions):
        H, T = latent_hist.shape[1], actions.shape[1]
        acts = actions if action_hist is None else torch.cat([action_hist, actions], dim=1)
        act_emb = self.model.action_encoder(acts)  # index k = block leaving frame k
        embs = list(latent_hist.unbind(dim=1))
        for t in range(T):
            lo = max(0, H + t - self.history_size)
            e = torch.stack(embs[lo:], dim=1)
            a = act_emb[:, lo : H + t]
            embs.append(self.model.predict(e, a)[:, -1])
        return torch.stack(embs[H:], dim=1)


def build(env: str, ckpt: str | None = None, device: str = "cuda", **_) -> tuple[LeWMPredictor, dict]:
    """Factory used by ``planning_eval.py --predictor shiftwm.v2.planning.lewm:build``."""
    model, meta = load_lewm(ckpt or PRETRAINED[env])
    model = model.to(device).eval().requires_grad_(False)
    return LeWMPredictor(model), meta
