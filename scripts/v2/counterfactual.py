"""Counterfactual actions on held-out DROID: same observed history, different (real) future command sequences.

DROID actions are absolute commanded end-effector poses (per model step: 5 native commands x [xyz, rpy, gripper],
robot base frame). A command sequence copied verbatim from another window would teleport the arm, so every
alternative plan is RE-ANCHORED: its 10-step sequence is expressed relative to its own last past command
(xyz and gripper differences; rpy differences wrapped to (-pi, pi]) and added to the last past command of the
history being forecast. The motion profile is therefore real (a recorded test trajectory); only its start is moved.
Net displacement below = commanded xyz at the last native command of step k=10 minus the anchor (metres).

(1) Example (fixed rule). History: the teaser window (results/v2/analysis/qualitative/teaser_pick.json, local start 0;
    also used by Figs. 2 and 11). Plans: the true one, plus three alternatives from test windows of OTHER episodes whose
    gripper command changes by < 0.05 over the 10 steps. A window's direction is the axis/sign of its net displacement
    in the robot frame (+x forward, +y left, +z up) when that axis dominates (|d_axis| >= 2 max of the others).
      opposite: direction opposite to the true plan's largest axis (here: true = left, so right)
      up      : +z (or +y if the true plan is vertical)
      (for both: among dominant windows of that direction, the one whose |d_axis| is closest to the 90th percentile)
      static  : the smallest commanded path length (ties: lowest window index)
    For each plan: ShiftWM forecasts for k=1..10, gate, expected transport source offsets (analysis.expected_offsets),
    and the Direct / AR forecasts (for the change-vs-true-plan numbers).
(2) All test windows (stride 2): alternative plan for window i = window j = i + W/2 (mod W), advanced until it comes
    from a different episode, re-anchored as above. Per arm (ShiftWM, Direct, AR), per k and region (moving = top 25%
    true change at k=10 within the window, as geometry.py; static = rest):
      change  = mean_c (f(a') - f(a))^2   (how much the forecast moves when the plan is swapped)
      err_true, err_alt = mean_c (f(.) - z_k)^2
    plus the un-anchored in-batch swap (train.evaluate's `sens` protocol) for reference. Episode means; 95% episode
    bootstrap CIs in summary.json.
Writes results/v2/analysis/counterfactual/{example.npz, per_episode.npz, summary.json}. Run on a GPU
(--example-only redoes only (1), cheap enough for a CPU, and updates the example entry of summary.json).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from geometry import load  # noqa: E402
from shiftwm.v2.analysis import FrameSource, bootstrap_ci, expected_offsets, local_starts  # noqa: E402
from shiftwm.v2.train import FeatureSplit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2/analysis/counterfactual"
H, K, B = 3, 10, 32
ARMS = ("shiftwm", "direct", "ar")


def wrap(x):
    return torch.remainder(x + np.pi, 2 * np.pi) - np.pi


class Plans:
    """Raw (unstandardised) past/future commands of every window and re-anchoring."""

    def __init__(self, data, stats, dev):
        self.am = torch.tensor(stats["action_mean"], dtype=torch.float32, device=dev)
        self.ast = torch.tensor(stats["action_std"], dtype=torch.float32, device=dev)
        t = data.starts[:, None] + torch.arange(H + K - 1, device=dev)[None]
        a = data.actions[t].float() * self.ast + self.am                          # [W, H+K-1, 35] raw
        self.past, self.fut = a[:, :H - 1], a[:, H - 1:]
        self.anchor = self.past[:, -1].reshape(-1, 5, 7)[:, -1]                   # [W,7] last past command
        f = self.fut.reshape(len(a), K, 5, 7)
        self.disp = f[:, :, -1, :3] - self.anchor[:, None, :3]                    # [W,K,3] per-step end position
        step = torch.diff(torch.cat((torch.zeros_like(self.disp[:, :1]), self.disp), 1), dim=1)
        self.path = step.norm(dim=-1).sum(-1)
        self.dgrip = (f[:, :, :, 6].reshape(len(a), -1) - self.anchor[:, None, 6]).abs().amax(-1)

    def reanchored(self, alt, cur):
        """Future commands of windows `alt`, re-anchored at the last past command of windows `cur` (standardised)."""
        f = self.fut[alt].reshape(len(alt), K, 5, 7)
        d = f - self.anchor[alt][:, None, None]
        d[..., 3:6] = wrap(d[..., 3:6])
        new = self.anchor[cur][:, None, None] + d
        new[..., 3:6] = wrap(new[..., 3:6]); new[..., 6] = new[..., 6].clamp(0, 1)
        return (new.reshape(len(alt), K, 35) - self.am) / self.ast


class LightSplit(FeatureSplit):
    """FeatureSplit with identical window indexing and actions, but features read lazily per episode (CPU, --example-only)."""

    def __init__(self, root, split, history, horizon, device, stats, stride=1):
        root = Path(root)
        rows = [r for r in json.loads((root / "manifest.json").read_text())["episodes"]
                if r["split"] == split and r["T"] >= history + horizon]
        self.fm = torch.tensor(stats["feature_mean"], dtype=torch.float32)
        self.fs = torch.tensor(stats["feature_std"], dtype=torch.float32)
        am = torch.tensor(stats["action_mean"], dtype=torch.float32)
        ast = torch.tensor(stats["action_std"], dtype=torch.float32)
        acts, starts, episode_of, self.episodes, self.offsets, self.files = [], [], [], [], [], []
        offset = 0
        for e, r in enumerate(rows):
            with np.load(root / r["file"]) as z:
                a = (torch.from_numpy(z["actions"]) - am) / ast
            a = torch.cat((a, torch.zeros(1, a.shape[1])), 0)
            T = a.shape[0]
            acts.append(a); s = torch.arange(0, T - history - horizon + 1, stride)
            starts.append(s + offset); episode_of.append(torch.full_like(s, e))
            self.episodes.append({"id": r["id"], "task": r.get("task", ""), "session": r.get("session", "")})
            self.offsets.append(offset); self.files.append(root / r["file"]); offset += T
        self.device = torch.device(device); self.single_image = False
        self.actions = torch.cat(acts).to(device); self.starts = torch.cat(starts).to(device)
        self.episode_of = torch.cat(episode_of).to(device); self.history, self.horizon = history, horizon
        self._cache = {}

    def _gather(self, t):
        out = []
        for row in t.cpu():
            e = int(np.searchsorted(self.offsets, int(row[0]), side="right") - 1)
            if e not in self._cache:
                with np.load(self.files[e]) as z:
                    f = torch.from_numpy(z["features"].astype(np.float32))
                self._cache[e] = ((f.reshape(f.shape[0], -1, f.shape[-1]) - self.fm) / self.fs).half()
            out.append(self._cache[e][row - self.offsets[e]])
        return torch.stack(out).to(self.device).float()


@torch.no_grad()
def run(models, hist, past, fut):
    out, det = {}, None
    for a, m in models.items():
        if a == "shiftwm":
            p, det = m(hist, past, fut, return_details=True)
        else:
            p = m(hist, past, fut)
        out[a] = p.float()
    return out, det


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--example-only", action="store_true")
    args = p.parse_args()
    dev = args.device
    root = ROOT / "data/v2/features/droid/dinov2s"
    stats = json.loads((root / "stats.json").read_text())
    data = (LightSplit if args.example_only else FeatureSplit)(root, "test", H, K, dev, stats, stride=2)
    models, ckpts = {}, {}
    for a in ARMS:
        models[a], ckpts[a] = load(a, dev)
    cfg = models["shiftwm"].config
    P = Plans(data, stats, dev)
    W = len(data); ep = data.episode_of; ls = local_starts(data)
    eps_np = ep.cpu().numpy(); U = len(data.episodes)

    # ------------------------------------------------------------------ (1) example
    teaser = json.loads((ROOT / "results/v2/analysis/qualitative/teaser_pick.json").read_text())["episode"]
    e0 = next(i for i, e in enumerate(data.episodes) if e["id"] == teaser)
    w0 = int(((ep == e0) & (ls == 0)).nonzero()[0, 0])
    net = P.disp[:, -1].cpu().numpy(); path = P.path.cpu().numpy(); dg = P.dgrip.cpu().numpy()
    other = (eps_np != e0) & (dg < 0.05)
    AX = {0: ("back", "forward"), 1: ("right", "left"), 2: ("down", "up")}

    def dominant(ax_, sgn):
        o = np.delete(np.abs(net), ax_, 1).max(1)
        return other & (sgn * net[:, ax_] > 0) & (np.abs(net[:, ax_]) >= 2 * o)

    ta = int(np.argmax(np.abs(net[w0]))); ts = int(np.sign(net[w0, ta]))
    dirs = [(ta, -ts), (2, 1) if ta != 2 else (1, 1)]
    pick = {}
    for ax_, sgn in dirs:
        c = np.where(dominant(ax_, sgn))[0]
        tgt = np.percentile(np.abs(net[c, ax_]), 90)
        pick[AX[ax_][(sgn + 1) // 2]] = int(c[np.argmin(np.abs(np.abs(net[c, ax_]) - tgt))])
    c = np.where(other)[0]
    pick["static"] = int(c[np.lexsort((c, path[c]))[0]])
    names = ["true", *pick.keys()]
    idx = torch.tensor([w0] * 4, device=dev)
    hist, past, fut, tgt = data.batch(idx)
    alt = torch.tensor([w0, *pick.values()], device=dev)
    fut_plans = P.reanchored(alt, idx)
    fut_plans[0] = fut[0]                                          # true plan exactly as stored
    preds, det = run(models, hist.float(), past, fut_plans)
    dx, dy, _ = expected_offsets(det["weights"], cfg)              # [4,K,N]
    frames = FrameSource(root).load(teaser, [int(ls[w0]) + H - 1, int(ls[w0]) + H - 1 + K])
    chg = ((tgt[0, K - 1].float() - hist[0, -1].float()) ** 2).mean(-1)
    mov = chg >= chg.quantile(0.75)
    ex = {"names": np.array(names), "window": w0, "episode": teaser, "t0": int(ls[w0]) + H - 1,
          "alt_windows": alt.cpu().numpy(), "alt_episodes": np.array([data.episodes[int(eps_np[j])]["id"] for j in alt.tolist()]),
          "alt_t0": (ls[alt] + H - 1).cpu().numpy(), "disp": P.disp[alt].cpu().numpy(), "path": path[alt.cpu().numpy()],
          "z0": hist[0, -1].float().cpu().numpy(), "true": tgt[0].float().cpu().numpy(),
          "shiftwm": preds["shiftwm"].cpu().numpy(), "gate": det["gate"][..., 0].float().cpu().numpy(),
          "dx": dx.cpu().numpy(), "dy": dy.cpu().numpy(), "frame_obs": frames[0], "frame_true": frames[1],
          "moving": mov.cpu().numpy()}
    for a in ARMS:                                                  # change vs the true plan, per plan, k=10
        d = ((preds[a][:, K - 1] - preds[a][:1, K - 1]) ** 2).mean(-1)                  # [4,N]
        ex[f"change_{a}_moving"] = d[:, mov].mean(-1).cpu().numpy()
        ex[f"change_{a}_static"] = d[:, ~mov].mean(-1).cpu().numpy()
        ex[f"err_{a}"] = ((preds[a][0] - tgt[0].float()) ** 2).mean((-1, -2)).cpu().numpy()   # true plan, per k
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "example.npz", **ex)
    print(json.dumps({"example": teaser, "window": w0, "alt": {n: int(v) for n, v in pick.items()},
                      "net_disp_m": {n: np.round(ex["disp"][i, -1], 3).tolist() for i, n in enumerate(names)},
                      "change_moving": {a: np.round(ex[f"change_{a}_moving"], 3).tolist() for a in ARMS}}), flush=True)

    ex_summ = {"episode": teaser, "window": w0, "alternatives": pick, "true_direction": AX[ta][(ts + 1) // 2],
               "net_disp_m": {n: ex["disp"][i, -1].tolist() for i, n in enumerate(names)},
               "path_m": {n: float(ex["path"][i]) for i, n in enumerate(names)}}
    if args.example_only:
        summ = json.loads((OUT / "summary.json").read_text()); summ["example"] = ex_summ
        summ["rule_example"] = __doc__.split("(2)")[0]
        (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
        return
    # ------------------------------------------------------------------ (2) all windows
    alt_all = (np.arange(W) + W // 2) % W
    for i in range(W):
        while eps_np[alt_all[i]] == eps_np[i]:
            alt_all[i] = (alt_all[i] + 1) % W
    alt_all = torch.tensor(alt_all, device=dev)
    keys = [f"{m}_{a}_{r}" for m in ("change", "err_true", "err_alt") for a in ARMS for r in ("moving", "static")]
    keys += [f"sensraw_{a}" for a in ARMS]
    acc = {k: torch.zeros(U, K, dtype=torch.float64, device=dev) for k in keys}
    cnt = torch.zeros(U, dtype=torch.float64, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i in range(0, W, B):
        idx = torch.arange(i, min(i + B, W), device=dev)
        hist, past, fut, tgt = data.batch(idx); hist, tgt = hist.float(), tgt.float()
        f_alt = P.reanchored(alt_all[idx], idx)
        p_true, _ = run(models, hist, past, fut)
        p_alt, _ = run(models, hist, past, f_alt)
        perm = torch.randperm(len(idx), generator=gen).to(dev)
        if len(idx) > 1:
            perm = torch.where(perm == torch.arange(len(idx), device=dev), (perm + 1) % len(idx), perm)
        p_raw, _ = run(models, hist, past, fut[perm])
        chg = ((tgt[:, K - 1] - hist[:, -1]) ** 2).mean(-1)
        mov = (chg >= chg.quantile(0.75, dim=-1, keepdim=True)).double()[:, None]          # [B,1,N]
        e = ep[idx]
        for a in ARMS:
            vals = {"change": ((p_alt[a] - p_true[a]) ** 2).mean(-1), "err_true": ((p_true[a] - tgt) ** 2).mean(-1),
                    "err_alt": ((p_alt[a] - tgt) ** 2).mean(-1)}
            for m, v in vals.items():
                v = v.double()
                acc[f"{m}_{a}_moving"].index_add_(0, e, (v * mov).sum(-1) / mov.sum(-1))
                acc[f"{m}_{a}_static"].index_add_(0, e, (v * (1 - mov)).sum(-1) / (1 - mov).sum(-1))
            acc[f"sensraw_{a}"].index_add_(0, e, ((p_raw[a] - p_true[a]) ** 2).mean((-1, -2)).double())
        cnt.index_add_(0, e, torch.ones(len(idx), dtype=torch.float64, device=dev))
    keep = cnt > 0
    per = {k: (v[keep] / cnt[keep, None]).cpu().numpy() for k, v in acc.items()}
    alt_np = alt_all.cpu().numpy()
    np.savez(OUT / "per_episode.npz", **per, windows=cnt[keep].cpu().numpy(),
             episodes=np.array([e["id"] for e, k_ in zip(data.episodes, keep.tolist()) if k_]),
             alt_net_disp=net[alt_np], net_disp=net, alt_path=path[alt_np], path=path)
    summ = {"windows": W, "episodes": int(keep.sum()), "checkpoints": ckpts, "rule_example": __doc__.split("(2)")[0],
            "example": ex_summ,
            "median_path_m": {"all": float(np.median(path)), "alternatives": float(np.median(path[alt_np]))}}
    for k, v in per.items():
        summ[k] = {"by_k": v.mean(0).tolist(), "k10": float(v[:, -1].mean()), "k10_ci": list(bootstrap_ci(v[:, -1])),
                   "avg": float(v.mean()), "avg_ci": list(bootstrap_ci(v.mean(1)))}
    for a in ARMS:                                                  # paired: error increase on moving patches, k=10
        d = per[f"err_alt_{a}_moving"][:, -1] - per[f"err_true_{a}_moving"][:, -1]
        summ[f"err_increase_{a}_moving_k10"] = [float(d.mean()), *bootstrap_ci(d)]
        r = per[f"change_{a}_moving"][:, -1] / per[f"change_{a}_static"][:, -1]
        summ[f"selectivity_{a}_k10"] = [float(r.mean()), *bootstrap_ci(r)]
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: round(summ[k]["k10"], 4) for k in keys}), flush=True)
    print(json.dumps({k: v for k, v in summ.items() if k.startswith(("err_increase", "selectivity"))}), flush=True)


if __name__ == "__main__":
    main()
