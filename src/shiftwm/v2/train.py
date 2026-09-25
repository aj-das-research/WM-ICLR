"""Train/evaluate one ShiftWM-v2 arm on a Stage-2 feature cache (docs/v2_data_format.md).

The whole cache split lives on the GPU in float16; windows are gathered by index, so the
loop is compute-bound. Selection uses the validation split only; the test split is scored
once with the selected checkpoint. Per-episode, per-horizon metrics are written for
paired bootstrap analysis.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .models import V2Config, V2WorldModel


class FeatureSplit:
    """GPU-resident windows: features [F,N,C] (standardised, fp16), actions [F,A] (standardised)."""

    def __init__(self, root, split, history, horizon, device, stats, stride=1, tasks=None, max_episodes=None,
                 single_image=False):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        rows = [r for r in manifest["episodes"] if r["split"] == split and (not tasks or r["task"] in tasks)]
        self.single_image = single_image
        need = 1 + horizon if single_image else history + horizon
        rows = [r for r in rows if r["T"] >= need]
        if max_episodes:
            rows = rows[:max_episodes]
        fm = torch.tensor(stats["feature_mean"], dtype=torch.float32)
        fs = torch.tensor(stats["feature_std"], dtype=torch.float32)
        am = torch.tensor(stats["action_mean"], dtype=torch.float32)
        ast = torch.tensor(stats["action_std"], dtype=torch.float32)
        feats, acts, starts, episode_of, self.episodes, self.proprio = [], [], [], [], [], []
        offset = 0
        for e, r in enumerate(rows):
            with np.load(root / r["file"]) as z:
                f = torch.from_numpy(z["features"].astype(np.float32))
                f = ((f.reshape(f.shape[0], -1, f.shape[-1]) - fm) / fs).half()
                a = (torch.from_numpy(z["actions"]) - am) / ast
                a = torch.cat((a, torch.zeros(1, a.shape[1])), 0)  # pad so action index == frame index
            T = f.shape[0]
            feats.append(f); acts.append(a)
            s = torch.arange(0, T - need + 1, stride)
            starts.append(s + offset); episode_of.append(torch.full_like(s, e))
            self.episodes.append({"id": r["id"], "task": r.get("task", ""), "session": r.get("session", "")})
            offset += T
        # Features stay on the GPU when it is large (H200); otherwise in pinned host RAM, gathered per batch.
        self.device = torch.device(device)
        host = features_on_host(device)
        self.features = _cat(feats, pin=host)
        if not host:
            self.features = self.features.to(device)
        self.actions = torch.cat(acts).to(device)
        self.starts = torch.cat(starts).to(device)
        self.episode_of = torch.cat(episode_of).to(device)
        self.history, self.horizon = history, horizon
        self.grid_channels = self.features.shape[1:]

    def __len__(self):
        return len(self.starts)

    def batch(self, idx):
        s = self.starts[idx]
        h, k = self.history, self.horizon
        if self.single_image:
            # One observed image: replicate it as the history; past actions at the train mean (0).
            t = s[:, None] + torch.arange(1 + k, device=s.device)[None]
            f = self._gather(t)
            hist = f[:, :1].expand(-1, h, -1, -1)
            past = torch.zeros(len(s), h - 1, self.actions.shape[1], device=s.device)
            return hist, past, self.actions[t[:, :-1]], f[:, 1:]
        t = s[:, None] + torch.arange(h + k, device=s.device)[None]
        f = self._gather(t)
        a = self.actions[t[:, :-1]]
        return f[:, :h], a[:, :h - 1], a[:, h - 1:h - 1 + k], f[:, h:]


    def _gather(self, t):
        if self.features.device == t.device:
            return self.features[t].float()
        return self.features[t.cpu()].to(t.device, non_blocking=True).float()


def gpu_gb(device="cuda"):
    return torch.cuda.get_device_properties(torch.device(device)).total_memory / 1e9


def features_on_host(device):
    """SHIFTWM_FEATURES=gpu|host overrides; default: host on GPUs under 100 GB (A100), GPU otherwise (H200)."""
    mode = os.environ.get("SHIFTWM_FEATURES", "auto")
    if mode != "auto":
        return mode == "host"
    return torch.device(device).type == "cuda" and gpu_gb(device) < 100


def _cat(chunks, pin=False):
    out = torch.empty((sum(len(c) for c in chunks),) + tuple(chunks[0].shape[1:]), dtype=chunks[0].dtype,
                      pin_memory=pin)
    i = 0
    for c in chunks:
        out[i:i + len(c)] = c; i += len(c)
    return out


def micro_batches(batch_size, device="cuda"):
    """Equal gradient-accumulation chunks so a step fits smaller GPUs; the summed gradient equals the
    full-batch gradient (MSE is a batch mean). SHIFTWM_MICRO overrides the chunk count."""
    n = int(os.environ.get("SHIFTWM_MICRO", 0))
    if not n:
        gb = gpu_gb(device)
        n = 1 if gb >= 100 else (2 if gb >= 60 else 4)
    while batch_size % n:
        n += 1
    return n


def build(cfg, grid, channels, action_dim):
    c = dict(cfg["model"])
    c.update(grid=grid, channels=channels, action_dim=action_dim, horizon=cfg["horizon"], history=cfg["history"])
    return V2WorldModel(c)


def loss_fn(model, batch, cfg):
    hist, past, fut, target = batch
    arm = model.config.arm
    if arm == "ar_tf":
        pred = model(hist, past, fut, teacher=target)
    else:
        pred = model(hist, past, fut)
    loss = F.mse_loss(pred, target)
    logs = {"mse": loss.detach()}
    w = cfg.get("contrastive_weight", 0.0)
    if w > 0 and arm not in ("ar", "ar_tf"):
        perm = torch.roll(torch.arange(len(fut), device=fut.device), 1)
        wrong = model(hist, past, fut[perm])
        err_true = ((pred - target) ** 2).mean((1, 2, 3))
        err_wrong = ((wrong - target) ** 2).mean((1, 2, 3))
        margin = cfg.get("contrastive_margin", 0.05)
        ctr = F.relu(margin - (err_wrong - err_true.detach())).mean()
        loss = loss + w * ctr
        logs["ctr"] = ctr.detach()
    return loss, logs


def _chunked(model, hist, past, fut):
    """Forward an eval batch in chunks on GPUs under 100 GB (identical outputs; batch composition, and hence the
    shuffled-action pairing, is unchanged). Avoids OOM when two runs share a 40 GB card."""
    n = len(hist) if hist.device.type != "cuda" or gpu_gb(hist.device) >= 100 else 32
    return torch.cat([model(hist[i:i + n], past[i:i + n], fut[i:i + n]).float() for i in range(0, len(hist), n)])


@torch.no_grad()
def evaluate(model, data, batch_size=128, details=False, shuffled=True):
    """Per-window, per-horizon metrics aggregated to episodes. Returns dict of numpy arrays."""
    model.eval()
    n_ep = len(data.episodes)
    k = data.horizon
    dev = data.starts.device
    sums = {m: torch.zeros(n_ep, k, device=dev, dtype=torch.float64)
            for m in ("mse", "cos", "mse_pool4", "mse_shuf", "rank_ok", "sens")}
    counts = torch.zeros(n_ep, device=dev, dtype=torch.float64)
    g = int(math.isqrt(data.grid_channels[0]))
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(0, len(data), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(data)), device=dev)
        hist, past, fut, target = data.batch(idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = _chunked(model, hist, past, fut)
        ep = data.episode_of[idx]
        err = ((pred - target) ** 2).mean(-1).mean(-1)                             # [B,K]
        cos = F.cosine_similarity(pred, target, dim=-1).mean(-1)                    # standardised space
        def pool(x):
            b, kk, n, c = x.shape
            return F.adaptive_avg_pool2d(x.reshape(b * kk, g, g, c).permute(0, 3, 1, 2), 4).reshape(b, kk, -1)
        errp = ((pool(pred) - pool(target)) ** 2).mean(-1)
        vals = {"mse": err, "cos": cos, "mse_pool4": errp}
        if shuffled and model.config.arm not in ("persistence", "linear"):
            perm = torch.randperm(len(idx), generator=gen).to(idx.device)
            if len(idx) > 1:
                perm = torch.where(perm == torch.arange(len(idx), device=idx.device), (perm + 1) % len(idx), perm)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                wrong = _chunked(model, hist, past, fut[perm])
            errw = ((wrong - target) ** 2).mean(-1).mean(-1)
            vals.update(mse_shuf=errw, rank_ok=(err < errw).double(),
                        sens=((wrong - pred) ** 2).mean(-1).mean(-1))
        for m, v in vals.items():
            sums[m].index_add_(0, ep, v.double())
        counts.index_add_(0, ep, torch.ones_like(ep, dtype=torch.float64))
    keep = counts > 0
    out = {m: (s[keep] / counts[keep, None]).cpu().numpy() for m, s in sums.items()}
    out["episodes"] = [e["id"] for e, kk in zip(data.episodes, keep.cpu().tolist()) if kk]
    out["tasks"] = [e["task"] for e, kk in zip(data.episodes, keep.cpu().tolist()) if kk]
    out["windows"] = counts[keep].cpu().numpy()
    model.train()
    return out


def summary(ev):
    return {"mse_mean_h": float(ev["mse"].mean()), "mse_h_end": float(ev["mse"][:, -1].mean()),
            "cos_h_end": float(ev["cos"][:, -1].mean()), "mse_pool4_h_end": float(ev["mse_pool4"][:, -1].mean()),
            "rank_acc": float(ev["rank_ok"].mean()) if ev["rank_ok"].any() else None,
            "episodes": len(ev["episodes"])}


def run(cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    root = Path(cfg["data"])
    manifest = json.loads((root / "manifest.json").read_text())
    stats = json.loads((root / "stats.json").read_text())
    dev = "cuda"
    H, K = cfg["history"], cfg["horizon"]
    tasks = cfg.get("tasks")
    single = cfg.get("single_image", False)
    train = FeatureSplit(root, "train", H, K, dev, stats, 1, tasks, cfg.get("max_train_episodes"), single_image=single)
    for extra in cfg.get("extra_train_roots", []):
        more = FeatureSplit(Path(extra), "train", H, K, dev, stats, 1, tasks, single_image=single)
        offset = len(train.features)
        train.features = _cat([train.features, more.features], pin=train.features.is_pinned()).to(
            train.features.device); train.actions = torch.cat((train.actions, more.actions))
        train.starts = torch.cat((train.starts, more.starts + offset))
        del more
    val = FeatureSplit(root, "val", H, K, dev, stats, cfg.get("eval_stride", 2), tasks, cfg.get("max_val_episodes"),
                       single_image=single)
    model = build(cfg, manifest["grid"], manifest["channels"], manifest["action_dim"]).to(dev)
    eval_bs = 128   # fixed: shuffled-action metrics pair windows within an eval batch
    log = open(out / "log.jsonl", "a")
    record = {"event": "start", "params": model.num_params() if model.learned else 0,
              "train_windows": len(train), "val_windows": len(val)}
    print(json.dumps(record), flush=True); log.write(json.dumps(record) + "\n")
    best_path, state_path = out / "best.pt", out / "last.pt"
    step, best = 0, float("inf")
    ema = None
    if model.learned and cfg.get("ema"):
        import copy
        ema = copy.deepcopy(model).eval().requires_grad_(False)
    if model.learned:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.05),
                                betas=(0.9, 0.95))
        total = cfg["steps"]; warm = cfg.get("warmup", 1000)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        if state_path.exists():
            st = torch.load(state_path, map_location=dev)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            if ema is not None and "ema" in st:
                ema.load_state_dict(st["ema"])
            step, best = st["step"], st["best"]
            torch.set_rng_state(st["rng"].cpu())
        t0 = time.time()
        n_micro = micro_batches(cfg["batch_size"], dev)
        print(json.dumps({"event": "memory_plan", "gpu_gb": round(gpu_gb(dev)),
                          "features_on_host": not train.features.is_cuda, "micro_batches": n_micro}), flush=True)
        while step < total:
            idx = torch.randint(0, len(train), (cfg["batch_size"],), device=dev)
            opt.zero_grad(set_to_none=True)
            loss, logs = 0.0, {}
            for sub in idx.chunk(n_micro):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l, lg = loss_fn(model, train.batch(sub), cfg)
                (l / n_micro).backward()
                loss = loss + l.detach() / n_micro
                for k_, v_ in lg.items():
                    logs[k_] = logs.get(k_, 0.0) + v_ / n_micro
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
            opt.step(); sched.step(); step += 1
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - cfg["ema"])
            if step % 100 == 0:
                rec = {"step": step, "loss": float(loss), **{k: float(v) for k, v in logs.items()},
                       "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n"); log.flush()
            if step % cfg["eval_every"] == 0 or step == total:
                ev = evaluate(ema if ema is not None else model, val, batch_size=eval_bs, shuffled=False)
                score = float(ev["mse"].mean())
                rec = {"event": "val", "step": step, "val_mse_mean_h": score, "val_mse_h_end": float(ev["mse"][:, -1].mean()),
                       "sec": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                if score < best:
                    best = score
                    torch.save({"model": (ema if ema is not None else model).state_dict(), "config": model.package_config,
                                "step": step}, best_path)
                torch.save({"model": model.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                            "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "best": best, "rng": torch.get_rng_state()}, state_path)
        model.load_state_dict(torch.load(best_path, map_location=dev)["model"])
    del train
    torch.cuda.empty_cache()
    results = {}
    for split in cfg.get("report_splits", ["val", "test"]):
        data = val if split == "val" else FeatureSplit(root, split, H, K, dev, stats, cfg.get("eval_stride", 2), tasks,
                                                        single_image=single)
        ev = evaluate(model, data, batch_size=eval_bs)
        np.savez(out / f"eval_{split}.npz", **{k: np.asarray(v) for k, v in ev.items()})
        results[split] = summary(ev)
    final = {"event": "done", "step": step, "best_val": best,
             "params": model.num_params() if model.learned else 0, "results": results}
    (out / "summary.json").write_text(json.dumps(final, indent=1))
    print(json.dumps(final), flush=True)
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="JSON file with training config")
    p.add_argument("--set", nargs="*", default=[], help="overrides key=json_value (model.x for model keys)")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    for kv in a.set:
        key, value = kv.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    run(cfg)
