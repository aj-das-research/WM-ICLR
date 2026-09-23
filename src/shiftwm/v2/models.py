"""ShiftWM-v2 and matched baselines on native-resolution frozen patch features.

Shared backbone (identical capacity for all learned arms):
  MemoryEncoder  -- self-attention over the H observed frames' patch tokens (+frame, position,
                    past-action embeddings).
  QueryDecoder   -- N query tokens per future step (initialised from the last observed grid),
                    self-attention + cross-attention to memory, AdaLN-zero conditioning on the
                    causal action-prefix state (GRU) and horizon embedding.

Output heads ("arms"):
  direct     : Z0 + W h                              (anchored additive, cross-patch capacity control)
  shiftwm    : (1-g) Z0 + g * LocalTransport(Z_{-S+1..0}) + s * W h   (ours)
  ar         : residual one-step predictor rolled out recursively (trained through the rollout)
  ar_tf      : DINO-WM-style teacher-forced one-step predictor of absolute features, rolled out
  persistence, linear : parameter-free references
All tensors are in train-standardised feature space: [B, frames, N=G*G, C].
"""
from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class V2Config:
    arm: str = "shiftwm"
    channels: int = 384
    grid: int = 16
    action_dim: int = 35
    history: int = 3
    horizon: int = 10
    dim: int = 384
    heads: int = 6
    enc_depth: int = 4
    dec_depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # transport
    window: int = 7            # local window side (odd); 0 = global
    sources: int = 3           # how many observed frames the transport may copy from (1 = last only)
    key_dim: int = 64
    identity_bias: float = 4.0
    initial_gate_logit: float = -3.0
    correction: str = "scaled"  # scaled | tanh | none
    tanh_bound: float = 1.0
    action_free: bool = False
    tf_residual: bool = False   # ar_tf variant: teacher-forced, but predicts a residual to the last frame
    extra: dict = field(default_factory=dict)


ARMS = ("persistence", "linear", "direct", "shiftwm", "ar", "ar_tf")


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Block(nn.Module):
    """Pre-norm transformer block with optional cross-attention and AdaLN-zero conditioning."""

    def __init__(self, dim, heads, mlp_ratio, cross=False, cond=False, dropout=0.0):
        super().__init__()
        self.heads, self.cross, self.cond = heads, cross, cond
        self.n1 = nn.LayerNorm(dim, elementwise_affine=not cond)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        if cross:
            self.nc = nn.LayerNorm(dim)
            self.q = nn.Linear(dim, dim)
            self.kv = nn.Linear(dim, 2 * dim)
            self.cproj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=not cond)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.drop = dropout
        if cond:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def _attn(self, q, k, v):
        b, n, d = q.shape
        h = self.heads
        q, k, v = (t.reshape(t.shape[0], t.shape[1], h, d // h).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return o.transpose(1, 2).reshape(b, n, d)

    def forward(self, x, memory=None, c=None):
        if self.cond:
            s1, g1, sh2, sc2, g2, sc1 = self.ada(c).unsqueeze(1).chunk(6, -1)
            y = modulate(self.n1(x), s1, sc1)
        else:
            y = self.n1(x)
        q, k, v = self.qkv(y).chunk(3, -1)
        a = self.proj(self._attn(q, k, v))
        x = x + (g1 * a if self.cond else a)
        if self.cross:
            y = self.nc(x)
            k, v = self.kv(memory).chunk(2, -1)
            x = x + self.cproj(self._attn(self.q(y), k, v))
        y = modulate(self.n2(x), sh2, sc2) if self.cond else self.n2(x)
        m = self.mlp(y)
        return x + (g2 * m if self.cond else m)


class V2WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = V2Config(**config) if isinstance(config, dict) else config
        if c.arm not in ARMS:
            raise ValueError(c.arm)
        n = c.grid * c.grid
        self.learned = c.arm not in ("persistence", "linear")
        if not self.learned:
            self.register_buffer("_dummy", torch.zeros(1))
            return
        d = c.dim
        self.inp = nn.Sequential(nn.LayerNorm(c.channels), nn.Linear(c.channels, d))
        self.pos = nn.Parameter(torch.randn(1, 1, n, d) * 0.02)
        self.frame = nn.Parameter(torch.randn(1, c.history, 1, d) * 0.02)
        self.act_in = nn.Sequential(nn.Linear(c.action_dim, d), nn.SiLU(), nn.Linear(d, d))
        r = c.extra.get("cost_volume_radius", 0)
        if r:
            self.cost_proj = nn.Sequential(nn.Linear((2 * r + 1) ** 2, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.cost_proj[-1].weight); nn.init.zeros_(self.cost_proj[-1].bias)
        self.prefix = nn.GRU(d, d, batch_first=True)  # causal: step k sees actions <= k only
        self.horizon_emb = nn.Embedding(max(c.horizon, 1) + 1, d)
        self.encoder = nn.ModuleList(Block(d, c.heads, c.mlp_ratio, dropout=c.dropout) for _ in range(c.enc_depth))
        self.enc_norm = nn.LayerNorm(d)
        self.q_inp = nn.Linear(d, d)
        self.decoder = nn.ModuleList(Block(d, c.heads, c.mlp_ratio, cross=True, cond=True, dropout=c.dropout)
                                     for _ in range(c.dec_depth))
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, c.channels)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        if c.arm == "shiftwm":
            self.tq = nn.Linear(d, c.key_dim, bias=False)
            self.tk = nn.Linear(d, c.key_dim, bias=False)
            self.gate = nn.Linear(d, 1)
            nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, c.initial_gate_logit)
            self.id_bias = nn.Parameter(torch.tensor(float(c.identity_bias)))
            self.corr_scale = nn.Parameter(torch.ones(c.channels))
            if c.extra.get("transport_iters", 1) > 1:
                self.refine = nn.Sequential(nn.LayerNorm(2 * c.channels + d), nn.Linear(2 * c.channels + d, d), nn.GELU(),
                                            nn.Linear(d, d))
                nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)

    # ------------------------------------------------------------------ helpers
    def _actions(self, a):
        return torch.zeros_like(a) if self.config.action_free else a

    def cost_volume(self, hist):
        """Cosine similarity of each patch at frame f with a (2r+1)^2 neighbourhood at frame f-1 (zeros for f=0)."""
        r = self.config.extra.get("cost_volume_radius", 0)
        b, h, n, ch = hist.shape
        g = self.config.grid
        z = F.normalize(hist.float(), dim=-1)
        cur = z[:, 1:].reshape(b * (h - 1), g, g, ch)
        prev = z[:, :-1].reshape(b * (h - 1), g, g, ch).permute(0, 3, 1, 2)
        w = 2 * r + 1
        nb = F.unfold(prev, w, padding=r).reshape(b * (h - 1), ch, w * w, n)          # [B',C,w*w,N]
        cv = torch.einsum("bnc,bcwn->bnw", cur.reshape(b * (h - 1), n, ch), nb)        # [B',N,w*w]
        cv = cv.reshape(b, h - 1, n, w * w)
        return torch.cat((torch.zeros_like(cv[:, :1]), cv), 1)

    def encode_memory(self, hist, past_actions):
        """hist [B,H,N,C], past_actions [B,H-1,A] -> memory tokens [B,H,N,d]."""
        c = self.config
        b, h, n, _ = hist.shape
        x = self.inp(hist) + self.pos + self.frame[:, -h:]
        if c.extra.get("cost_volume_radius", 0) and h > 1:
            x = x + self.cost_proj(self.cost_volume(hist).to(x.dtype))
        a = self.act_in(self._actions(past_actions))                  # action taken after frame i
        x = x + F.pad(a, (0, 0, 0, 1))[:, :, None]                     # frame i gets action i (last: none)
        x = x.reshape(b, h * n, -1)
        for blk in self.encoder:
            x = blk(x)
        return self.enc_norm(x).reshape(b, h, n, -1)

    def prefix_states(self, past_actions, future_actions):
        """Causal action-prefix state for each future step: [B,K,d]."""
        a = self.act_in(self._actions(torch.cat((past_actions, future_actions), 1)))
        s = self.prefix(a)[0]
        return s[:, past_actions.shape[1]:]

    def decode(self, memory, anchor_tokens, cond):
        """memory [B,H,N,d], anchor_tokens [B,N,d], cond [B,K,d] -> hidden [B,K,N,d]."""
        b, h, n, d = memory.shape
        k = cond.shape[1]
        mem = memory.reshape(b, 1, h * n, d).expand(b, k, h * n, d).reshape(b * k, h * n, d)
        x = self.q_inp(anchor_tokens)[:, None].expand(b, k, n, d).reshape(b * k, n, d) + self.pos[0]
        cc = cond.reshape(b * k, d)
        for blk in self.decoder:
            x = blk(x, mem, cc)
        return self.out_norm(x).reshape(b, k, n, d)

    def transport(self, hidden, memory, hist):
        """Local soft transport of observed features. hidden [B,K,N,d]; memory [B,H,N,d]; hist [B,H,N,C]."""
        c = self.config
        b, k, n, _ = hidden.shape
        g = c.grid
        s = min(c.sources, hist.shape[1])
        mem, val = memory[:, -s:], hist[:, -s:]
        q = self.tq(hidden).float()                                          # [B,K,N,dk]
        key = self.tk(mem).float()                                           # [B,S,N,dk]
        if c.window and c.window < 2 * g:
            w, r = c.window, c.window // 2
            def unfold(t):  # [B,S,N,D] -> [B,N,S*w*w,D]
                bb, ss, _, dd = t.shape
                t = t.permute(0, 1, 3, 2).reshape(bb * ss, dd, g, g)
                u = F.unfold(t, w, padding=r).reshape(bb, ss, dd, w * w, n)
                return u.permute(0, 4, 1, 3, 2).reshape(bb, n, ss * w * w, dd)
            ku, vu = unfold(key), unfold(val.float())
            valid = F.unfold(torch.ones(1, 1, g, g, device=q.device), w, padding=r)[0].T.bool()  # [N,w*w]
            valid = valid.repeat(1, s)                                        # [N,S*w*w]
            center = torch.zeros(s * w * w, device=q.device); center[(s - 1) * w * w + (w * w) // 2] = 1.0
        else:
            ku = key.reshape(b, 1, s * n, -1).expand(b, n, s * n, key.shape[-1])
            vu = val.float().reshape(b, 1, s * n, -1).expand(b, n, s * n, val.shape[-1])
            valid = torch.ones(n, s * n, dtype=torch.bool, device=q.device)
            center = torch.zeros(n, s * n, device=q.device)
            center[torch.arange(n), (s - 1) * n + torch.arange(n)] = 1.0
        logits = torch.einsum("bknd,bnmd->bknm", q, ku) / math.sqrt(q.shape[-1])
        logits = logits + self.id_bias * center
        logits = logits.masked_fill(~valid, float("-inf"))
        weights = torch.softmax(logits, -1)
        moved = torch.einsum("bknm,bnmc->bknc", weights, vu)
        return moved, weights

    # ------------------------------------------------------------------ forward
    def forward(self, hist, past_actions, future_actions, return_details=False, teacher=None):
        """hist [B,H,N,C] standardised; past_actions [B,H-1,A]; future_actions [B,K,A] -> [B,K,N,C]."""
        c = self.config
        k = future_actions.shape[1]
        z0 = hist[:, -1]
        if c.arm == "persistence":
            return z0[:, None].expand(-1, k, -1, -1)
        if c.arm == "linear":
            vel = hist[:, -1] - hist[:, -2]
            steps = torch.arange(1, k + 1, device=hist.device, dtype=hist.dtype)[None, :, None, None]
            return z0[:, None] + steps * vel[:, None]
        if c.arm in ("ar", "ar_tf"):
            return self._rollout(hist, past_actions, future_actions, teacher)
        memory = self.encode_memory(hist, past_actions)
        p_drop = self.config.extra.get("mem_token_drop", 0.0)
        if self.training and p_drop > 0:            # drop whole memory tokens (regularises the decoder's reading)
            keep = (torch.rand(memory.shape[:3], device=memory.device) > p_drop).to(memory.dtype)[..., None]
            memory = memory * keep
        cond = self.prefix_states(past_actions, future_actions) + self.horizon_emb.weight[1:k + 1][None]
        hidden = self.decode(memory, memory[:, -1], cond)
        corr = self.out(hidden).float()
        if c.arm == "direct":
            return z0[:, None].float() + corr
        if c.correction == "tanh":
            corr = c.tanh_bound * torch.tanh(corr)
        elif c.correction == "none":
            corr = torch.zeros_like(corr)
        else:
            corr = corr * self.corr_scale
        moved, weights = self.transport(hidden, memory, hist)
        for _ in range(c.extra.get("transport_iters", 1) - 1):
            # RAFT-style update: the query sees what it moved and what it would keep, then re-selects sources
            z0k = z0[:, None].expand_as(moved)
            hidden = hidden + self.refine(torch.cat((hidden.float(), moved, z0k.float()), -1).to(hidden.dtype))
            moved, weights = self.transport(hidden, memory, hist)
            corr = self.out(hidden).float()
            corr = c.tanh_bound * torch.tanh(corr) if c.correction == "tanh" else (torch.zeros_like(corr) if c.correction == "none" else corr * self.corr_scale)
        gate = torch.sigmoid(self.gate(hidden).float())
        pred = (1 - gate) * z0[:, None].float() + gate * moved + corr
        if return_details:
            return pred, {"weights": weights, "gate": gate, "correction": corr}
        return pred

    def _rollout(self, hist, past_actions, future_actions, teacher=None):
        """Recursive one-step prediction. `teacher` [B,K,N,C] enables teacher forcing (ar_tf training)."""
        c = self.config
        frames = hist
        acts = past_actions
        preds = []
        for t in range(future_actions.shape[1]):
            a_t = future_actions[:, t:t + 1]
            memory = self.encode_memory(frames, acts)
            p_drop = c.extra.get("mem_token_drop", 0.0)
            if self.training and p_drop > 0:
                memory = memory * (torch.rand(memory.shape[:3], device=memory.device) > p_drop).to(memory.dtype)[..., None]
            cond = self.prefix_states(acts, a_t) + self.horizon_emb.weight[1][None, None]
            hidden = self.decode(memory, memory[:, -1], cond)[:, 0]
            delta = self.out(hidden).float()
            absolute = c.arm == "ar_tf" and not c.tf_residual
            nxt = delta if absolute else frames[:, -1].float() + delta
            preds.append(nxt)
            fed = teacher[:, t] if teacher is not None else nxt
            frames = torch.cat((frames[:, 1:], fed[:, None].to(frames.dtype)), 1)
            acts = torch.cat((acts, a_t), 1)[:, 1:]  # keep H-1 past actions (identical for H>=2; fixes H=1)
        return torch.stack(preds, 1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def package_config(self):
        return asdict(self.config)
