"""Functional helpers: :func:`forecast`, :func:`transport_field`, :func:`skill`."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from .model import WorldModel

__all__ = ["forecast", "transport_field", "skill"]

Details = Dict[str, Union[torch.Tensor, int]]


@torch.no_grad()
def forecast(model: WorldModel, history_feats: torch.Tensor, past_actions: torch.Tensor,
             future_actions: torch.Tensor, return_details: bool = False
             ) -> Union[torch.Tensor, Tuple[torch.Tensor, Details]]:
    """Forecast future patch features with a (ShiftWM) world model, without gradients.

    Args:
        model: a :class:`~shiftwm.WorldModel`.
        history_feats: observed features ``[B, H, N, C]`` (or ``[H, N, C]`` for a single window).
        past_actions: ``[B, H-1, A]`` (or ``[H-1, A]``).
        future_actions: ``[B, K, A]`` (or ``[K, A]``).
        return_details: also return the transport details of the ``shiftwm`` arm:
            ``weights [B,K,N,M]``, ``gate [B,K,N,1]``, ``correction [B,K,N,C]``, plus the
            integers ``grid``, ``window`` and ``sources`` needed by :func:`transport_field`.

    Returns:
        predictions ``[B, K, N, C]`` (``[K, N, C]`` for unbatched input), optionally with details.
    """
    single = history_feats.dim() == 3
    if single:
        history_feats, past_actions, future_actions = history_feats[None], past_actions[None], future_actions[None]
    was_training = model.training
    model.eval()
    try:
        if return_details:
            if model.config.arm != "shiftwm":
                raise ValueError(f"return_details needs arm='shiftwm', got {model.config.arm!r}")
            pred, det = model(history_feats, past_actions, future_actions, return_details=True)
            c = model.config
            det = dict(det, grid=c.grid, window=c.window, sources=min(c.sources, history_feats.shape[1]))
        else:
            pred, det = model(history_feats, past_actions, future_actions), None
    finally:
        model.train(was_training)
    if single:
        pred = pred[0]
        if det is not None:
            det = {k: (v[0] if torch.is_tensor(v) else v) for k, v in det.items()}
    return (pred, det) if return_details else pred


def transport_field(details: Details, grid: Optional[int] = None, window: Optional[int] = None) -> torch.Tensor:
    """Expected source offset of every predicted patch, in patch units.

    Args:
        details: the dict returned by ``forecast(..., return_details=True)`` (or any dict with a
            ``"weights"`` tensor ``[..., N, M]``, then pass ``grid`` and ``window``).

    Returns:
        ``[..., N, 2]`` tensor of ``(dx, dy)``: the transport-weighted mean displacement from the
        predicted patch to the patches it copies from (``(0, 0)`` = stays in place). Weights are
        summed over source frames.
    """
    w8 = details["weights"]
    g = int(grid if grid is not None else details["grid"])
    win = int(window if window is not None else details["window"])
    n, m = w8.shape[-2], w8.shape[-1]
    if n != g * g:
        raise ValueError(f"weights have N={n} tokens, grid={g}")
    dev = w8.device
    if win and win < 2 * g:
        if m % (win * win):
            raise ValueError(f"last dim {m} is not a multiple of window**2={win * win} (dilated windows unsupported)")
        r = torch.arange(win, device=dev) - win // 2
        dy, dx = torch.meshgrid(r, r, indexing="ij")
        off = torch.stack((dx.flatten(), dy.flatten()), -1).float()               # [w*w, 2]
        off = off.repeat(m // (win * win), 1)                                       # [M, 2]
        return w8.float() @ off
    ys, xs = torch.meshgrid(torch.arange(g, device=dev), torch.arange(g, device=dev), indexing="ij")
    pos = torch.stack((xs.flatten(), ys.flatten()), -1).float()                   # [N, 2]
    src = pos.repeat(m // n, 1)                                                     # [M, 2]
    return w8.float() @ src - pos


def skill(pred: torch.Tensor, target: torch.Tensor, history: torch.Tensor, per_horizon: bool = False) -> torch.Tensor:
    """Skill over persistence: ``1 - MSE(pred, target) / MSE(last observed frame, target)``.

    Args:
        pred, target: ``[B, K, N, C]`` (or ``[K, N, C]``).
        history: observed features ``[B, H, N, C]``; the last frame is the persistence forecast.
        per_horizon: return one value per step ``[K]`` instead of a scalar.

    ``> 0`` beats persistence, ``1`` is perfect.
    """
    if pred.dim() == 3:
        pred, target, history = pred[None], target[None], history[None]
    last = history[:, -1:].float()
    dims = (0, 2, 3) if per_horizon else (0, 1, 2, 3)
    mse = ((pred.float() - target.float()) ** 2).mean(dims)
    mse_p = ((last - target.float()) ** 2).mean(dims)
    return 1 - mse / mse_p
