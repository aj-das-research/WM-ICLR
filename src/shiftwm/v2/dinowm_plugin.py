"""ShiftWM transport head as a plug-in for the official DINO-WM predictor.

Official code: github.com/gaoyuezhou/dino_wm @ 0a9492fa12044b852ae9e001cc74604b79c8bb0c
(checked out, unmodified, at external/dino_wm).

DINO-WM's `ViTPredictor` maps the num_hist input tokens [b, T*N, D] (frame-causal attention) to
predicted tokens P [b, T*N, D]; output slot t is the prediction of frame t+1 and is trained against
the encoder tokens of frame t+1 (`VWorldModel.forward`, teacher forcing).  With concat_dim=1 each
token is [DINOv2 patch (384) | proprio emb (10*rep) | action emb (10*rep)].

`ShiftViTPredictor` subclasses it and changes only the output:

    Z_hat_vis[t] = (1 - g) * P_vis[t] + g * Transport(x_vis[t])
    Z_hat_rest[t] = P_rest[t]                      (proprio / action channels untouched)

  * x[t]  = the predictor's input token at slot t, i.e. the most recent frame the prediction may see
            (the encoded observation during training and at the first rollout step; DINO-WM's own
            earlier prediction later in an autoregressive rollout, exactly as in `VWorldModel.rollout`).
  * Transport = local 7x7 soft transport on the 14x14 patch grid (identity-biased softmax), the
            sources=1 / no-dilation path of `shiftwm.v2.models.V2WorldModel.transport`:
            queries = Linear(P[t]) (predictor output token), keys = Linear(x[t]) (observed token),
            values = x_vis[t] (observed DINOv2 patch features), logit(centre) += identity_bias.
  * g = sigmoid(Linear(P[t])) per token, weight 0 and bias -4 at init (g = 0.018), so the model starts
            (numerically almost) equal to DINO-WM; `gate_bias=-inf` makes it exactly equal.

Causality is preserved: slot t reads P[t] (sees inputs <= t) and x[t] only.
The extra parameters are drawn from a private generator, so the global RNG stream (and hence the
base predictor, decoder initialisation, dropout and data order) is identical to the baseline arm.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

try:  # available when external/dino_wm is on sys.path (training / planning)
    from models.vit import ViTPredictor
except Exception:  # pragma: no cover - allows importing the pure transport function alone
    ViTPredictor = nn.Module


def local_transport(q, k, v, grid, window=7, identity_bias=4.0):
    """Local soft transport. q,k [B,N,dk]; v [B,N,C] on a grid x grid lattice -> (moved [B,N,C], weights [B,N,w*w]).

    Same arithmetic as V2WorldModel.transport with sources=1 and no extra dilations:
    softmax_j( <q_i, k_j>/sqrt(dk) + b_id * [j == i] ), j in the w x w window around i (out-of-grid masked).
    """
    b, n, _ = q.shape
    w, r = window, window // 2

    def unfold(t):  # [B,N,D] -> [B,N,w*w,D]
        d = t.shape[-1]
        u = F.unfold(t.transpose(1, 2).reshape(b, d, grid, grid), w, padding=r)  # [B, D*w*w, N]
        return u.reshape(b, d, w * w, n).permute(0, 3, 2, 1)

    ku, vu = unfold(k.float()), unfold(v.float())
    valid = F.unfold(torch.ones(1, 1, grid, grid, device=q.device), w, padding=r)[0].T.bool()  # [N,w*w]
    center = torch.zeros(w * w, device=q.device)
    center[(w * w) // 2] = 1.0
    logits = torch.einsum("bnd,bnmd->bnm", q.float(), ku) / math.sqrt(q.shape[-1])
    logits = logits + identity_bias * center
    logits = logits.masked_fill(~valid, float("-inf"))
    weights = torch.softmax(logits, -1)
    return torch.einsum("bnm,bnmc->bnc", weights, vu), weights


class ShiftWMHead(nn.Module):
    def __init__(self, dim, visual_dim=384, key_dim=64, window=7, identity_bias=4.0, gate_bias=-4.0, seed=12345):
        super().__init__()
        self.visual_dim, self.window = visual_dim, window
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)  # private init stream: keeps the global RNG identical to the baseline arm
        try:
            self.tq = nn.Linear(dim, key_dim, bias=False)
            self.tk = nn.Linear(dim, key_dim, bias=False)
            self.gate = nn.Linear(dim, 1)
        finally:
            torch.random.set_rng_state(state)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.id_bias = nn.Parameter(torch.tensor(float(identity_bias)))

    def forward(self, pred, src, grid, return_details=False):
        """pred, src: [B,T,N,D] (predictor output / predictor input). Returns [B,T,N,D]."""
        b, t, n, d = pred.shape
        vd = self.visual_dim
        p2, s2 = pred.reshape(b * t, n, d), src.reshape(b * t, n, d)
        moved, weights = local_transport(self.tq(p2), self.tk(s2), s2[..., :vd], grid, self.window, self.id_bias)
        gate = torch.sigmoid(self.gate(p2).float())
        vis = ((1 - gate) * p2[..., :vd].float() + gate * moved).to(pred.dtype)
        out = torch.cat((vis, p2[..., vd:]), -1).reshape(b, t, n, d)
        if return_details:
            return out, {"gate": gate.reshape(b, t, n), "weights": weights.reshape(b, t, n, -1)}
        return out


class ShiftViTPredictor(ViTPredictor):
    """Drop-in replacement for models.vit.ViTPredictor (hydra: predictor._target_=shiftwm.v2.dinowm_plugin.ShiftViTPredictor)."""

    def __init__(self, *, num_patches, num_frames, dim, visual_dim=384, key_dim=64, window=7,
                 identity_bias=4.0, gate_bias=-4.0, head_seed=12345, **kw):
        super().__init__(num_patches=num_patches, num_frames=num_frames, dim=dim, **kw)
        grid = int(round(math.sqrt(num_patches)))
        if grid * grid != num_patches:
            raise ValueError(f"ShiftWM head needs a square patch grid (concat_dim=1); got num_patches={num_patches}")
        self.num_patches, self.grid = num_patches, grid
        self.shift_head = ShiftWMHead(dim, visual_dim, key_dim, window, identity_bias, gate_bias, head_seed)

    def forward(self, x, return_details=False):  # x: (b, T*N, dim), same contract as ViTPredictor.forward
        b, tn, d = x.shape
        t = tn // self.num_patches
        pred = super().forward(x)
        out = self.shift_head(pred.reshape(b, t, self.num_patches, d), x.reshape(b, t, self.num_patches, d),
                              self.grid, return_details=return_details)
        if return_details:
            out, det = out
            return out.reshape(b, tn, d), det
        return out.reshape(b, tn, d)
