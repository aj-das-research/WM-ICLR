"""V-JEPA 2-AC plug-in study on our DROID split, from the cached frozen ViT-g tokens (scripts/v2/vjepa_cache.py).

Arms (results/v2/external/vjepa2ac_plugin/<arm>/[s<seed>/]):
  zeroshot          (A) released AC predictor, no training.
  finetune          (B) AC predictor fine-tuned on our train split, encoder frozen (tokens cached).
  finetune_shiftwm  (C) B + ShiftWM transport head (shiftwm.v2.vjepa_plugin), identical budget/objective/schedule.

Training objective = app/vjepa_droid/train.py (droid-256px-8f.yaml): clips of 8 consecutive frames (one of our 3 Hz
steps = one predictor step, as in the evaluator), per-frame layer-normed target tokens, loss = teacher-forced L1 over
frames 1..7 + L1 of an auto_steps=2 rollout from frame 0; AdamW(0.9, 0.999), wd 0.04 (bias/1-d excluded), their WSD
schedule (warm-up 15/315 of the steps from start_lr, constant, linear anneal over the last 15/315 to 0), LR and
start LR scaled linearly with batch (their global batch 256 = 8 x 32 GPUs), bf16 autocast, activation checkpointing,
no grad clipping. Clip sampling = one episode uniformly, then a uniform start (their per-trajectory loader).
The new head's parameters use lr_scale = --head-lr-mult (their per-group lr_scale mechanism); everything else is
identical between B and C. Checkpoint selection: val MSE (their space, mean over horizons 1..10) of the evaluator's
rollout protocol on val windows (stride --val-stride), every --eval-every steps.

Evaluation = scripts/v2/eval_vjepa2ac.py protocol from cached tokens: windows stride 2, history 3, horizon 10, AR
rollout with a sliding 8-frame context, per-window mse/l1/cos/mse_pool4 + moving/static MSE (top-25% true change
patches as scripts/v2/region_eval.py), persistence + linear references, shuffled-action check, per-episode means.

Resumable: latest.pt every eval; the process exits with code 75 before --deadline (unix time) and a re-run continues.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import eval_vjepa2ac as ev  # noqa: E402
from shiftwm.v2.vjepa_plugin import PluginWM, make_head  # noqa: E402

ARMS = ("zeroshot", "finetune", "finetune_shiftwm")
EXIT_RESUME = 75
THEIR_BATCH, THEIR_LR, THEIR_START_LR = 256, 4.25e-4, 7.5e-5


# ------------------------------------------------------------------------------------------------ data
class CachedSplit:
    """Tokens [F,N,C] fp16, states/actions [F,7] of one split, resident on `device`."""

    def __init__(self, root, split, device, limit_episodes=None, chunk=2048):
        index = json.loads((root / "index.json").read_text())
        rows = [r for r in index["episodes"] if r["split"] == split]
        if limit_episodes:
            rows = rows[:limit_episodes]
        if not rows:
            raise SystemExit(f"no {split} episodes in {root}")
        lo, hi = rows[0]["offset"], rows[-1]["offset"] + rows[-1]["T"]
        assert hi - lo == sum(r["T"] for r in rows), "split not contiguous in the cache"
        tok = np.load(root / "tokens.npy", mmap_mode="r")
        self.Z = torch.empty((hi - lo,) + tok.shape[1:], dtype=torch.float16, device=device)
        for i in range(lo, hi, chunk):
            j = min(hi, i + chunk)
            self.Z[i - lo:j - lo] = torch.from_numpy(np.array(tok[i:j])).to(device)
        self.S = torch.from_numpy(np.load(root / "states.npy")[lo:hi]).float().to(device)
        self.A = torch.from_numpy(np.load(root / "actions.npy")[lo:hi]).float().to(device)
        self.eps = [dict(r, local=r["offset"] - lo) for r in rows]
        self.split, self.device = split, device

    def windows(self, history, horizon, stride):
        return [(e, r["local"] + st, st) for e, r in enumerate(self.eps)
                for st in ev.window_starts(r["T"], history, horizon, stride)]

    def clip_sampler(self, length):
        ok = [r for r in self.eps if r["T"] >= length]
        self.c_off = torch.tensor([r["local"] for r in ok], device=self.device)
        self.c_len = torch.tensor([r["T"] - length + 1 for r in ok], device=self.device)
        self.c_ar = torch.arange(length, device=self.device)
        return len(ok)

    def sample(self, batch, gen):
        e = torch.randint(len(self.c_off), (batch,), device=self.device, generator=gen)
        u = torch.rand(batch, device=self.device, generator=gen)
        st = self.c_off[e] + (u * self.c_len[e]).long().clamp(max=self.c_len[e] - 1)
        fr = st[:, None] + self.c_ar[None]
        return self.Z[fr].float(), self.A[fr[:, :-1]], self.S[fr]


# ------------------------------------------------------------------------------------------------ model
def load_predictor(repo, ckpt, device):
    ev.add_repo(repo)
    from src.hub.backbones import _clean_backbone_key
    from src.models.ac_predictor import vit_ac_predictor
    # == _make_vjepa2_ac_model(vit_ac_giant, 256): predictor kwargs of the hub (embed_dim = ViT-g 1408)
    pred = vit_ac_predictor(img_size=(ev.CROP, ev.CROP), patch_size=ev.PATCH, num_frames=64, tubelet_size=2,
                            embed_dim=1408)
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    pred.load_state_dict(_clean_backbone_key(dict(sd["predictor"])), strict=True)
    del sd
    return pred.to(device)


def build(a, device):
    pred = load_predictor(a.repo, a.checkpoint, device)
    head = None
    if a.arm == "finetune_shiftwm":
        torch.manual_seed(a.seed)
        head = make_head(pred, sources=a.sources, window=a.window, key_dim=a.key_dim,
                         identity_bias=a.identity_bias, gate_bias=a.gate_bias).to(device)
    return PluginWM(pred, head).to(device)


def param_groups(wm, head_lr_mult, wd):
    """Their init_opt grouping (bias / 1-d params without weight decay) + lr_scale for the new head."""
    groups = []
    for mod, scale in ((wm.predictor, 1.0), (wm.head, head_lr_mult)):
        if mod is None:
            continue
        named = [(n, p) for n, p in mod.named_parameters() if p.requires_grad]
        dec = lambda n, p: "bias" not in n and p.ndim >= 2 and n != "src_emb"   # == their rule for the predictor
        decay = [p for n, p in named if dec(n, p)]
        other = [p for n, p in named if not dec(n, p)]
        groups += [{"params": decay, "weight_decay": wd, "lr_scale": scale},
                   {"params": other, "weight_decay": 0.0, "lr_scale": scale}]
    return [g for g in groups if g["params"]]


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(wm, data, a, stride, shuffled, max_windows=None):
    """Per-window [W,K] metrics of model / persistence / linear (/ shuffled) + moving/static MSE."""
    H, K = a.history, a.horizon
    win = data.windows(H, K, stride)
    if max_windows and len(win) > max_windows:
        win = [win[i] for i in np.linspace(0, len(win) - 1, max_windows).round().astype(int)]
    W, dev = len(win), data.device
    w_ep = np.array([w[0] for w in win]); w_glob = torch.tensor([w[1] for w in win], device=dev)
    perm = torch.from_numpy(ev.derangement(W, a.eval_seed)).to(dev)
    names = ["model", "persistence", "linear"] + (["shuffled"] if shuffled else [])
    per_win = {f"{n}_{m}": np.zeros((W, K)) for n in names
               for m in ("mse", "cos", "l1", "mse_pool4", "mse_moving", "mse_static")}
    if shuffled:
        per_win["sens"] = np.zeros((W, K))
    if wm.head is not None:
        per_win["gate_mean"] = np.zeros((W, K)); per_win["gate_moving"] = np.zeros((W, K))
    ar = torch.arange(H + K, device=dev)
    wm.eval(); wm.predictor.use_activation_checkpointing = False
    for b0 in range(0, W, a.eval_batch):
        bi = torch.arange(b0, min(b0 + a.eval_batch, W), device=dev)
        t = w_glob[bi, None] + ar[None]
        z_true = data.Z[t].float()
        s_in, a_in = data.S[t[:, :-1]], data.A[t[:, :-1]]
        target, last = z_true[:, H:], z_true[:, H - 1:H]
        pred, gate = wm.rollout(z_true[:, :H], s_in, a_in, K, a.max_context, return_gate=True)
        steps = torch.arange(1, K + 1, device=dev, dtype=torch.float32)[None, :, None, None]
        outs = {"model": pred, "persistence": last.expand_as(target),
                "linear": last + steps * (last - z_true[:, H - 2:H - 1])}
        if shuffled:
            ta = w_glob[perm[bi], None] + ar[None]
            a_fut = data.A[ta[:, H - 1:H + K - 1]]
            s_fut = ev.integrate_states(s_in[:, H - 1].cpu().numpy(), a_fut[:, :-1].cpu().numpy())
            s_sh = torch.cat([s_in[:, :H - 1], torch.from_numpy(s_fut).float().to(dev)], 1)
            a_sh = torch.cat([a_in[:, :H - 1], a_fut], 1)
            outs["shuffled"] = wm.rollout(z_true[:, :H], s_sh, a_sh, K, a.max_context)
            per_win["sens"][b0:b0 + len(bi)] = ((outs["shuffled"] - pred) ** 2).mean((-1, -2)).cpu().numpy()
        change = ((target - last) ** 2).mean(-1)                                  # [B,K,N] true change
        mv = (change >= change.quantile(0.75, dim=-1, keepdim=True)).float()
        for n, y in outs.items():
            r = ev.metrics(y, target)
            err = ((y - target) ** 2).mean(-1)
            r["mse_moving"] = (err * mv).sum(-1) / mv.sum(-1)
            r["mse_static"] = (err * (1 - mv)).sum(-1) / (1 - mv).sum(-1)
            for m, v in r.items():
                per_win[f"{n}_{m}"][b0:b0 + len(bi)] = v.double().cpu().numpy()
        if gate is not None:
            per_win["gate_mean"][b0:b0 + len(bi)] = gate.mean(-1).double().cpu().numpy()
            per_win["gate_moving"][b0:b0 + len(bi)] = ((gate * mv).sum(-1) / mv.sum(-1)).double().cpu().numpy()
    E = len(data.eps)
    have = np.bincount(w_ep, minlength=E) > 0
    per_ep = {k: np.stack([v[w_ep == e].mean(0) for e in range(E) if have[e]]) for k, v in per_win.items()}
    if shuffled:
        rk = (per_win["model_mse"] < per_win["shuffled_mse"]).astype(float)
        per_win["rank_ok"] = rk
        per_ep["rank_ok"] = np.stack([rk[w_ep == e].mean(0) for e in range(E) if have[e]])
    eps = [r for e, r in enumerate(data.eps) if have[e]]
    return {"per_win": per_win, "per_ep": per_ep, "w_ep": w_ep, "win": win, "eps": eps}


def summarize(res, arm, split, extra):
    per_ep, eps = res["per_ep"], res["eps"]
    sessions = np.array([r["session"] for r in eps])
    s = {"model": f"V-JEPA 2-AC (vjepa2-ac-vitg) {arm}", "arm": arm, "split": split, "episodes": len(eps),
         "sessions": int(len(np.unique(sessions))), "windows": len(res["win"]),
         "per_horizon": {k: v.mean(0).tolist() for k, v in per_ep.items()},
         "mean_over_horizons": {k: float(v.mean()) for k, v in per_ep.items()}}
    g = {}
    for m in ("mse", "l1", "mse_pool4", "mse_moving", "mse_static"):
        for ref in ("persistence", "linear"):
            mm, pp = per_ep[f"model_{m}"], per_ep[f"{ref}_{m}"]
            g.setdefault(ref, {})[m] = {
                "per_horizon": (1 - mm.mean(0) / pp.mean(0)).tolist(),
                "mean_over_horizons": float(1 - mm.mean() / pp.mean()),
                "mean_over_horizons_ci95_session_bootstrap": ev.bootstrap_gain(mm.mean(1), pp.mean(1), sessions),
                "h1": float(1 - mm[:, 0].mean() / pp[:, 0].mean()),
                "h_end": float(1 - mm[:, -1].mean() / pp[:, -1].mean())}
    s["relative_gain_vs_persistence"] = g["persistence"]
    s["relative_gain_vs_linear"] = g["linear"]
    s["cos_delta_vs_persistence_per_horizon"] = (per_ep["model_cos"].mean(0) - per_ep["persistence_cos"].mean(0)).tolist()
    if "rank_ok" in per_ep:
        s["action_rank_acc"] = float(per_ep["rank_ok"].mean())
    s.update(extra)
    return s


def save_eval(res, out_npz, summary):
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, episodes=np.array([r["id"] for r in res["eps"]]),
                        sessions=np.array([r["session"] for r in res["eps"]]),
                        win_episode=res["w_ep"], win_start=np.array([w[2] for w in res["win"]]),
                        **res["per_ep"], **{f"win_{k}": v.astype(np.float32) for k, v in res["per_win"].items()})
    Path(str(out_npz).replace(".npz", "_summary.json")).write_text(json.dumps(summary, indent=1))


def protocol(a):
    return {"history": a.history, "horizon": a.horizon, "max_context_frames": a.max_context,
            "feature_space": "layer-normed ViT-g/16 tokens (cache data/v2/features/droid/vjepa2g, == eval_vjepa2ac.py)",
            "moving": "per window/horizon: patches in the top 25% of true change ||z_t+k - z_t||^2 (region_eval.py)",
            "metrics": "per-window mean over tokens/channels; per-episode mean over windows; mean over episodes"}


# ------------------------------------------------------------------------------------------------ training
def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def val_score(res):
    return float(res["per_ep"]["model_mse"].mean())


def train(a, wm, tr, va, run_dir, out_dir):
    ev.add_repo(a.repo)
    from src.utils.schedulers import WSDSchedule
    scale = a.batch_size / THEIR_BATCH
    lr = a.lr if a.lr else THEIR_LR * scale
    start_lr = a.start_lr if a.start_lr is not None else THEIR_START_LR * scale
    warm, anneal = round(a.steps * a.warmup_frac), round(a.steps * a.anneal_frac)
    opt = torch.optim.AdamW(param_groups(wm, a.head_lr_mult, a.weight_decay), betas=(0.9, 0.999), eps=1e-8)
    sched = WSDSchedule(opt, warmup_steps=warm, anneal_steps=anneal, T_max=a.steps, start_lr=start_lr, ref_lr=lr,
                        final_lr=0.0)
    n_clip_eps = tr.clip_sampler(a.clip_frames)
    state = {"step": 0, "curve": [], "best": None, "log": []}
    latest = run_dir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=a.device, weights_only=False)
        wm.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        state = ck["state"]
        print(f"[resume] step {state['step']}", flush=True)
    sched._step = float(state["step"])
    cfg = {"lr": lr, "start_lr": start_lr, "warmup_steps": warm, "anneal_steps": anneal, "steps": a.steps,
           "batch_size": a.batch_size, "clip_frames": a.clip_frames, "auto_steps": a.auto_steps,
           "weight_decay": a.weight_decay, "head_lr_mult": a.head_lr_mult, "train_episodes_with_clip": n_clip_eps,
           "trainable_params": {"predictor": sum(p.numel() for p in wm.predictor.parameters()),
                                "head": sum(p.numel() for p in wm.head.parameters()) if wm.head is not None else 0}}
    print("[train]", json.dumps(cfg), flush=True)

    def do_eval(step):
        t = time.time()
        res = evaluate(wm, va, a, a.val_stride, False, a.val_max_windows)
        rec = {"step": step, "val_mse": val_score(res),
               "val_mse_per_h": res["per_ep"]["model_mse"].mean(0).tolist(),
               "val_pers_mse": float(res["per_ep"]["persistence_mse"].mean()),
               "val_mse_moving": float(res["per_ep"]["model_mse_moving"].mean()), "eval_s": time.time() - t}
        if wm.head is not None:
            rec["val_gate_mean"] = float(res["per_ep"]["gate_mean"].mean())
            rec["id_bias"] = float(wm.head.id_bias)
        state["curve"].append(rec)
        print("[val]", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()
                                   if k != "val_mse_per_h"}), flush=True)
        if step > 0 or a.select_include_init:
            if state["best"] is None or rec["val_mse"] < state["best"]["val_mse"]:
                state["best"] = rec
                save_atomic({"model": wm.state_dict(), "step": step, "val": rec, "config": vars(a)}, run_dir / "best.pt")
        save_atomic({"model": wm.state_dict(), "opt": opt.state_dict(), "state": state, "config": vars(a)}, latest)
        (out_dir / "curve.json").write_text(json.dumps({"config": cfg, "curve": state["curve"], "best": state["best"],
                                                        "log": state["log"]}, indent=1))

    if state["step"] == 0 and not state["curve"]:
        do_eval(0)
    gen = torch.Generator(device=a.device)
    t_last, tl = time.time(), []
    while state["step"] < a.steps:
        if a.deadline and time.time() > a.deadline - a.reserve_min * 60:
            print(f"[deadline] stopping at step {state['step']} (resumable)", flush=True)
            save_atomic({"model": wm.state_dict(), "opt": opt.state_dict(), "state": state, "config": vars(a)}, latest)
            sys.exit(EXIT_RESUME)
        step = state["step"]
        cur_lr = sched.step()
        gen.manual_seed(a.seed * 1_000_003 + step)
        wm.train(); wm.predictor.use_activation_checkpointing = True
        h, act, st = tr.sample(a.batch_size, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=h.is_cuda):
            z_tf, z_ar, gate = wm.train_forward(h, act, st, a.auto_steps)
            loss, jl, sl = wm.loss(z_tf, z_ar, h)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        state["step"] = step + 1
        tl.append([float(loss), float(jl), float(sl)] + ([float(gate)] if gate is not None else []))
        if state["step"] % a.log_every == 0 or state["step"] == 1:
            dt = (time.time() - t_last) / max(1, len(tl))
            rec = {"step": state["step"], "loss": float(np.mean([x[0] for x in tl])),
                   "jloss": float(np.mean([x[1] for x in tl])), "sloss": float(np.mean([x[2] for x in tl])),
                   "lr": cur_lr, "s_per_step": dt,
                   "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if h.is_cuda else 0}
            if gate is not None:
                rec["gate"] = float(np.mean([x[3] for x in tl])); rec["id_bias"] = float(wm.head.id_bias)
            state["log"].append(rec)
            print("[step]", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}),
                  flush=True)
            tl, t_last = [], time.time()
            if not math.isfinite(rec["loss"]):
                raise SystemExit("non-finite loss")
        if state["step"] % a.eval_every == 0 or state["step"] == a.steps:
            do_eval(state["step"])
    return state


# ------------------------------------------------------------------------------------------------ compare
def compare(a):
    base = Path(a.results)
    def load(arm):
        fs = sorted((base / arm).glob("s*/test.npz")) if arm != "zeroshot" else [base / "zeroshot/test.npz"]
        return {f.parent.name if arm != "zeroshot" else "zs": np.load(f) for f in fs if f.exists()}
    runs = {arm: load(arm) for arm in ARMS}
    out = {}
    for cand, ref in (("finetune_shiftwm", "finetune"), ("finetune", "zeroshot"), ("finetune_shiftwm", "zeroshot")):
        for sc, zc in runs[cand].items():
            for sr, zr in runs[ref].items():
                if cand != "zeroshot" and ref != "zeroshot" and sc != sr:
                    continue
                assert np.array_equal(zc["episodes"], zr["episodes"])
                d = {}
                for m in ("mse", "l1", "mse_moving", "mse_static"):
                    mc, mr = zc[f"model_{m}"], zr[f"model_{m}"]
                    d[m] = {"rel_gain_mean_h": float(1 - mc.mean() / mr.mean()),
                            "ci95_session_bootstrap": ev.bootstrap_gain(mc.mean(1), mr.mean(1), zc["sessions"]),
                            "rel_gain_per_h": (1 - mc.mean(0) / mr.mean(0)).tolist(),
                            "rel_gain_h_end": float(1 - mc[:, -1].mean() / mr[:, -1].mean()),
                            "ci95_h_end": ev.bootstrap_gain(mc[:, -1], mr[:, -1], zc["sessions"])}
                out[f"{cand}/{sc} vs {ref}/{sr}"] = d
                print(f"{cand}/{sc} vs {ref}/{sr}: mse {d['mse']['rel_gain_mean_h']:+.4f} "
                      f"{np.round(d['mse']['ci95_session_bootstrap'], 4).tolist()}  moving "
                      f"{d['mse_moving']['rel_gain_mean_h']:+.4f}", flush=True)
    (base / "comparison.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ main
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=str(ROOT / "data/v2/features/droid/vjepa2g"))
    p.add_argument("--repo", default=str(ROOT / "external/vjepa2"))
    p.add_argument("--checkpoint", default=str(ROOT / "data/pretrained/vjepa2_ac/vjepa2-ac-vitg.pt"))
    p.add_argument("--results", default=str(ROOT / "results/v2/external/vjepa2ac_plugin"))
    p.add_argument("--runs", default=str(ROOT / "data/v2/runs/vjepa2ac_plugin"), help="checkpoints")
    # optimisation (B == C)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="default 4.25e-4 * batch/256 (their ref LR, linear scaling)")
    p.add_argument("--start-lr", type=float, default=None, help="default 7.5e-5 * batch/256")
    p.add_argument("--warmup-frac", type=float, default=15 / 315)
    p.add_argument("--anneal-frac", type=float, default=15 / 315)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--clip-frames", type=int, default=8)
    p.add_argument("--auto-steps", type=int, default=2)
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    # head (C)
    p.add_argument("--sources", type=int, default=3)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--identity-bias", type=float, default=4.0)
    p.add_argument("--gate-bias", type=float, default=-4.0)
    # evaluation
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--max-context", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--val-stride", type=int, default=4)
    p.add_argument("--val-max-windows", type=int, default=None)
    p.add_argument("--select-include-init", action="store_true")
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--no-shuffled", dest="shuffled", action="store_false")
    p.add_argument("--log-every", type=int, default=25)
    # plumbing
    p.add_argument("--deadline", type=float, default=0, help="unix time; exit 75 (resumable) before it")
    p.add_argument("--reserve-min", type=float, default=12)
    p.add_argument("--limit-episodes", type=int, default=None, help="smoke tests")
    p.add_argument("--tag", default="", help="suffix for results/run dirs (smoke tests)")
    p.add_argument("--compare", action="store_true", help="paired A/B/C comparison of finished test runs")
    return p


def main():
    a = build_parser().parse_args()
    if a.compare:
        return compare(a)
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    sub = a.arm if a.arm == "zeroshot" else f"{a.arm}/s{a.seed}"
    out_dir = Path(a.results + a.tag) / sub
    run_dir = Path(a.runs + a.tag) / sub
    out_dir.mkdir(parents=True, exist_ok=True); run_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "test_summary.json").exists():
        print(f"[skip] {out_dir}/test_summary.json exists", flush=True)
        return
    cache = Path(a.cache)
    idx = json.loads((cache / "index.json").read_text())
    if not idx.get("complete") and not a.limit_episodes:
        raise SystemExit(f"cache {cache} incomplete")
    wm = build(a, a.device)
    print(f"[model] predictor + head loaded {time.time() - t0:.0f}s", flush=True)
    va = CachedSplit(cache, "val", a.device, a.limit_episodes)
    extra = {"protocol": protocol(a), "cache_sanity": idx.get("sanity"), "checkpoint": str(a.checkpoint)}
    if a.arm != "zeroshot":
        tr = CachedSplit(cache, "train", a.device, a.limit_episodes)
        print(f"[data] train {tr.Z.shape[0]} val {va.Z.shape[0]} frames on {a.device}; {time.time() - t0:.0f}s", flush=True)
        state = train(a, wm, tr, va, run_dir, out_dir)
        del tr
        torch.cuda.empty_cache()
        best = torch.load(run_dir / "best.pt", map_location=a.device, weights_only=False)
        wm.load_state_dict(best["model"])
        extra["selected"] = {"step": best["step"], "val": best["val"]}
        extra["curve"] = state["curve"]
    te = CachedSplit(cache, "test", a.device, a.limit_episodes)
    for split, data, sh in (("val", va, False), ("test", te, a.shuffled)):
        t = time.time()
        res = evaluate(wm, data, a, 2, sh)
        extra["eval_s"] = time.time() - t
        summ = summarize(res, a.arm, split, extra)
        save_eval(res, out_dir / f"{split}.npz", summ)
        g = summ["relative_gain_vs_persistence"]["mse"]
        print(f"[{split}] mse {summ['mean_over_horizons']['model_mse']:.5f} pers "
              f"{summ['mean_over_horizons']['persistence_mse']:.5f} gain {g['mean_over_horizons']:+.4f} "
              f"{np.round(g['mean_over_horizons_ci95_session_bootstrap'], 4).tolist()} ({extra['eval_s']:.0f}s)",
              flush=True)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
