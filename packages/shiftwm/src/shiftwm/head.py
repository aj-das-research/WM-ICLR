"""Backbone-agnostic ShiftWM plug-in head.

Given any predictor that outputs patch-token forecasts ``P`` (and, optionally, a hidden state per
token), the head re-writes each forecast as

    Z_hat = (1 - g) * P + g * T,

where ``T`` is a local soft transport of *measured* tokens of the last ``sources`` observed frames
(identity-biased softmax over a ``window x window`` neighbourhood) and ``g = sigmoid(w . h + b)`` is a
per-token gate. The gate weight is zero and its bias is ``gate_bias`` (default -4) at initialisation,
so ``g = 0.018`` everywhere and the wrapped model starts numerically close to the backbone
(``gate_bias=float("-inf")`` makes it exactly equal).

The maths is the one used for the V-JEPA 2-AC and DINO-WM plug-in studies of the paper.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["local_transport", "ShiftHead", "ShiftWrapped"]


def local_transport(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, grid: int, window: int = 7,
                    identity_bias: Union[float, torch.Tensor] = 4.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identity-biased local soft transport on a ``grid x grid`` patch lattice.

    Args:
        query: ``[B, K, N, dk]`` one query per predicted token.
        keys: ``[B, S, N, dk]`` keys of the ``S`` source frames (oldest first; the last is the most recent).
        values: ``[B, S, N, C]`` measured tokens of the source frames.
        grid: side of the patch grid (``N = grid**2``).
        window: odd side of the local window (``0`` = global attention over all ``S*N`` tokens).
        identity_bias: logit bonus of the same patch in the most recent source frame.

    Returns:
        ``(moved [B, K, N, C], weights [B, K, N, M])`` with ``M = S*window**2`` (or ``S*N`` if global).
        Source slot ``m`` of the local layout is ``(s, dy, dx)`` in row-major order, ``dy, dx`` in
        ``[-window//2, window//2]``.
    """
    b, k, n, _ = query.shape
    s = keys.shape[1]
    q = query.float()
    if window and window < 2 * grid:
        w, r = window, window // 2

        def unfold(t: torch.Tensor) -> torch.Tensor:  # [B,S,N,D] -> [B,N,S*w*w,D]
            bb, ss, _, dd = t.shape
            t = t.permute(0, 1, 3, 2).reshape(bb * ss, dd, grid, grid)
            u = F.unfold(t, w, padding=r).reshape(bb, ss, dd, w * w, n)
            return u.permute(0, 4, 1, 3, 2).reshape(bb, n, ss * w * w, dd)

        ku, vu = unfold(keys.float()), unfold(values.float())
        valid = F.unfold(torch.ones(1, 1, grid, grid, device=q.device), w, padding=r)[0].T.bool().repeat(1, s)
        center = torch.zeros(s * w * w, device=q.device)
        center[(s - 1) * w * w + (w * w) // 2] = 1.0
    else:
        ku = keys.float().reshape(b, 1, s * n, -1).expand(b, n, s * n, keys.shape[-1])
        vu = values.float().reshape(b, 1, s * n, -1).expand(b, n, s * n, values.shape[-1])
        valid = torch.ones(n, s * n, dtype=torch.bool, device=q.device)
        center = torch.zeros(n, s * n, device=q.device)
        center[torch.arange(n), (s - 1) * n + torch.arange(n)] = 1.0
    logits = torch.einsum("bknd,bnmd->bknm", q, ku) / math.sqrt(q.shape[-1])
    logits = logits + identity_bias * center
    logits = logits.masked_fill(~valid, float("-inf"))
    weights = torch.softmax(logits, -1)
    moved = torch.einsum("bknm,bnmc->bknc", weights, vu)
    return moved, weights


class ShiftHead(nn.Module):
    """Plug-in transport head: ``Z_hat = (1 - g) * P + g * T``.

    Args:
        token_dim: channels ``C`` of the forecast / observed tokens.
        grid: side of the square patch grid (``N = grid**2``).
        hidden_dim: channels of the backbone hidden state used for queries and gate
            (defaults to ``token_dim``, i.e. the forecast ``P`` itself).
        sources: number of most recent observed frames the transport may copy from.
        window: odd local window side (``0`` = global).
        key_dim: query/key width.
        identity_bias: initial logit bonus of "stay in place".
        gate_bias: initial gate logit (``-4`` -> ``g = 0.018``; ``-inf`` -> exactly the backbone).
    """

    def __init__(self, token_dim: int, grid: int, hidden_dim: Optional[int] = None, sources: int = 1,
                 window: int = 7, key_dim: int = 64, identity_bias: float = 4.0, gate_bias: float = -4.0) -> None:
        super().__init__()
        hidden_dim = token_dim if hidden_dim is None else hidden_dim
        self.token_dim, self.grid, self.sources, self.window = token_dim, grid, sources, window
        self.tq = nn.Linear(hidden_dim, key_dim, bias=False)
        self.tk = nn.Linear(token_dim, key_dim, bias=False)
        self.src_emb = nn.Parameter(torch.zeros(sources, key_dim))     # per-source-slot key offset
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.id_bias = nn.Parameter(torch.tensor(float(identity_bias)))

    def forward(self, pred: torch.Tensor, observed: torch.Tensor, hidden: Optional[torch.Tensor] = None,
                return_details: bool = False):
        """Correct a backbone forecast with transported observations.

        Args:
            pred: backbone forecast ``P`` of shape ``[B, K, N, C]``.
            observed: measured tokens ``[B, H, N, C]``; the last ``sources`` frames are read.
            hidden: backbone hidden state ``[B, K, N, dh]`` for queries and gate (default: ``pred``).
            return_details: also return ``{"weights", "gate", "moved"}``.

        Returns:
            ``Z_hat [B, K, N, C]`` (dtype of ``pred``), or ``(Z_hat, details)``.
        """
        h = pred if hidden is None else hidden
        src = observed[:, -self.sources:]
        with torch.autocast(pred.device.type, enabled=False):
            q = self.tq(h.float())
            k = self.tk(src.float()) + self.src_emb[-src.shape[1]:][None, :, None]
            moved, weights = local_transport(q, k, src, self.grid, self.window, self.id_bias)
            gate = torch.sigmoid(self.gate(h.float()))
            out = ((1 - gate) * pred.float() + gate * moved).to(pred.dtype)
        if return_details:
            return out, {"weights": weights, "gate": gate, "moved": moved}
        return out

    def wrap(self, backbone: Callable[..., object]) -> "ShiftWrapped":
        """Return ``backbone`` with this head applied to its output (see :class:`ShiftWrapped`)."""
        return ShiftWrapped(backbone, self)


class ShiftWrapped(nn.Module):
    """``backbone`` followed by a :class:`ShiftHead`.

    ``backbone(observed, *args, **kwargs)`` must return the forecast ``P [B, K, N, C]`` or a tuple
    ``(P, hidden)``; ``observed [B, H, N, C]`` is passed to both the backbone and the head.
    """

    def __init__(self, backbone: Callable[..., object], head: ShiftHead) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, observed: torch.Tensor, *args, return_details: bool = False, **kwargs):
        out = self.backbone(observed, *args, **kwargs)
        pred, hidden = out if isinstance(out, tuple) else (out, None)
        return self.head(pred, observed, hidden, return_details=return_details)
