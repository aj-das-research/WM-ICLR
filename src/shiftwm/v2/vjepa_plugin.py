"""ShiftWM transport head as a plug-in for V-JEPA 2-AC (Assran et al. 2025) -- arms B / C of the plug-in study.

Everything lives in V-JEPA 2-AC's own feature space: F.layer_norm(ViT-g/16 encoder tokens), [.., N=256, C=1408].

  B  (finetune)          : the released AC predictor, fine-tuned with its own objective (app/vjepa_droid/train.py:
                           teacher-forced L1 + auto_steps rollout L1, normalize_reps=True).
  C  (finetune_shiftwm)  : same predictor + ShiftWM head. For every predicted frame
                               Z_hat = (1 - g) * P + g * T
                           P = layer_norm(predictor output block)               (= B's prediction)
                           T = local soft transport (window x window, identity bias on the same patch of the most
                               recent observed frame) of MEASURED encoder tokens of the last `sources` observed frames;
                               queries from the predictor's hidden state of that output block (predictor_norm output,
                               1024-d), keys from the observed tokens (+ a learned per-source-slot key offset)
                           g = sigmoid(w . hidden + b), w = 0, b = gate_bias (-4) at init  ->  C ~= B at init.
                           The predictor consumes Z_hat as its own prediction during rollouts; T never reads predictions.

"Observed" frames per output block (src_idx rows, -1 = unavailable):
  * teacher forcing (all context frames measured): block t predicts frame t+1 and reads frames t-S+1..t;
  * their training rollout (context = [z_0, predictions...]): only z_0 is measured -> [-1, .., -1, 0];
  * evaluation rollout from H observed frames: every step reads frames H-S..H-1.

The predictor object is the vendored `src.models.ac_predictor.VisionTransformerPredictorAC` (external/vjepa2).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def window_mask(grid, window):
    """[N,N] bool: key patch within the (window x window) neighbourhood of the query patch (0 = global)."""
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    y, x = yy.flatten(), xx.flatten()
    if not window:
        return torch.ones(grid * grid, grid * grid, dtype=torch.bool)
    r = window // 2
    return ((y[:, None] - y[None]).abs() <= r) & ((x[:, None] - x[None]).abs() <= r)


class TransportHead(nn.Module):
    """Local soft transport + gate. ~0.16M params at (hidden 1024, tokens 1408, key_dim 64)."""

    def __init__(self, hidden_dim, token_dim, grid, sources=3, window=7, key_dim=64, identity_bias=4.0,
                 gate_bias=-4.0):
        super().__init__()
        self.grid, self.sources, self.window, self.key_dim = grid, sources, window, key_dim
        self.tq = nn.Linear(hidden_dim, key_dim, bias=False)
        self.tk = nn.Linear(token_dim, key_dim, bias=False)
        self.src_emb = nn.Parameter(torch.zeros(sources, key_dim))
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.id_bias = nn.Parameter(torch.tensor(float(identity_bias)))
        n = grid * grid
        self.register_buffer("win", window_mask(grid, window).repeat(1, sources), persistent=False)   # [N,S*N]
        center = torch.zeros(n, sources * n)
        center[torch.arange(n), (sources - 1) * n + torch.arange(n)] = 1.0
        self.register_buffer("center", center, persistent=False)

    def forward(self, hidden, frames, src_idx):
        """hidden [B,M,N,dh]: predictor hidden of M output blocks; frames [B,F,N,C]: measured tokens;
        src_idx LongTensor [M,S]: frame index (into F) of source slot s for block m, -1 = unavailable; the last
        slot (most recent observed frame, carries the identity bias) must be valid.
        Returns moved [B,M,N,C] fp32, gate [B,M,N,1] fp32, weights [B,M,N,S*N]."""
        b, m, n, _ = hidden.shape
        s = self.sources
        assert src_idx.shape == (m, s) and bool((src_idx[:, -1] >= 0).all()), src_idx
        src_idx = src_idx.to(frames.device)
        valid = (src_idx >= 0)                                                   # [M,S]
        idx = src_idx.clamp(min=0)
        with torch.autocast(frames.device.type, enabled=False):
            q = self.tq(hidden.float())                                          # [B,M,N,dk]
            k = self.tk(frames.float())                                          # [B,F,N,dk]
            kg = (k[:, idx] + self.src_emb[None, None, :, None]).reshape(b, m, s * n, -1)
            logits = torch.einsum("bmnd,bmkd->bmnk", q, kg) / math.sqrt(self.key_dim)
            logits = logits + self.id_bias * self.center
            ok = self.win[None] & valid.repeat_interleave(n, 1)[:, None, :]      # [M,N,S*N]
            logits = logits.masked_fill(~ok[None], float("-inf"))
            weights = torch.softmax(logits, -1)
            vals = frames[:, idx].float().reshape(b, m, s * n, -1)               # [B,M,S*N,C]
            moved = weights @ vals
            gate = torch.sigmoid(self.gate(hidden.float()))
        return moved, gate, weights


def make_head(predictor, **kw):
    return TransportHead(predictor.predictor_norm.normalized_shape[0], predictor.predictor_proj.out_features,
                         predictor.grid_height, **kw)


class PluginWM(nn.Module):
    """AC predictor (+ optional TransportHead). head=None reproduces the released model exactly."""

    def __init__(self, predictor, head=None, normalize=True):
        super().__init__()
        self.predictor, self.head, self.normalize = predictor, head, normalize
        self.tokens = predictor.grid_height * predictor.grid_width
        self._hidden = None
        if head is not None:
            predictor.predictor_norm.register_forward_hook(self._grab)

    def _grab(self, module, inp, out):
        self._hidden = out

    @property
    def sources(self):
        return self.head.sources if self.head is not None else 1

    def step(self, z, actions, states, frames=None, src_idx=None, last=None):
        """One predictor call. z [B,L,N,C] context (measured or predicted); actions/states [B,L,7].
        Output block t predicts frame t+1; returns the last `last` (default all L) blocks [B,M,N,C] fp32 and the
        gate (or None). frames/src_idx ([L,S] or [M,S]) feed the transport head."""
        b, L, n, c = z.shape
        m = L if last is None else last
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
            out = self.predictor(z.flatten(1, 2), actions, states)
        p = out.view(b, L, n, c)[:, L - m:].float()
        if self.normalize:
            p = F.layer_norm(p, (c,))
        if self.head is None:
            return p, None
        hid = self._hidden.view(b, L, n, -1)[:, L - m:]
        self._hidden = None
        if src_idx.shape[0] != m:
            src_idx = src_idx[-m:]
        moved, gate, _ = self.head(hid, frames, src_idx)
        return (1 - gate) * p + gate * moved, gate

    # ---------------------------------------------------------------- their training objective
    def tf_sources(self, T):
        s = self.sources
        return torch.tensor([[t - s + 1 + i if t - s + 1 + i >= 0 else -1 for i in range(s)] for t in range(T)])

    def first_frame_sources(self, M):
        return torch.tensor([[-1] * (self.sources - 1) + [0]] * M)

    def train_forward(self, h, actions, states, auto_steps=2):
        """== app/vjepa_droid/train.py::forward_predictions on per-frame tokens.
        h [B,T,N,C] layer-normed target tokens of the clip; actions [B,T-1,7]; states [B,T,7].
        Returns z_tf [B,T-1,N,C] (frames 1..T-1), z_ar [B,auto_steps,N,C] (frames 1..auto_steps), mean gate."""
        T = h.shape[1]
        z_tf, g_tf = self.step(h[:, :-1], actions, states[:, :-1], h, self.tf_sources(T - 1))
        seq = [h[:, :1], z_tf[:, :1]]
        gates = [g_tf]
        for k in range(1, auto_steps):
            ctx = torch.cat(seq, 1)
            nxt, g = self.step(ctx, actions[:, :k + 1], states[:, :k + 1], h[:, :1], self.first_frame_sources(1),
                               last=1)
            seq.append(nxt); gates.append(g)
        z_ar = torch.cat(seq[1:], 1)
        gate = None if self.head is None else torch.cat([g.flatten() for g in gates]).mean()
        return z_tf, z_ar, gate

    @staticmethod
    def loss(z_tf, z_ar, h, loss_exp=1.0):
        """jloss + sloss of their loss_fn: mean |z - h_{1..}|^p / p."""
        def fn(z):
            return torch.mean(torch.abs(z - h[:, 1:1 + z.shape[1]].float()) ** loss_exp) / loss_exp
        jl, sl = fn(z_tf), fn(z_ar)
        return jl + sl, jl, sl

    # ---------------------------------------------------------------- evaluation rollout
    @torch.no_grad()
    def rollout(self, z_hist, states, actions, horizon, max_context=8, return_gate=False):
        """== scripts/v2/eval_vjepa2ac.py::rollout (+ head). z_hist [B,H,N,C] measured; states/actions
        [B,H+K-1,7]. Transport always reads the last S of the H measured frames."""
        b, H, n, c = z_hist.shape
        seq = [z_hist[:, t] for t in range(H)]
        src = torch.tensor([[H - self.sources + i if H - self.sources + i >= 0 else -1
                             for i in range(self.sources)]])
        preds, gates = [], []
        for k in range(horizon):
            L = H + k
            lo = 0 if not max_context else max(0, L - max_context)
            z = torch.stack(seq[lo:L], 1)
            nxt, g = self.step(z, actions[:, lo:L], states[:, lo:L], z_hist, src, last=1)
            nxt = nxt[:, 0]
            seq.append(nxt); preds.append(nxt)
            if g is not None:
                gates.append(g[:, 0, :, 0])
        preds = torch.stack(preds, 1)
        if return_gate:
            return preds, (torch.stack(gates, 1) if gates else None)
        return preds
