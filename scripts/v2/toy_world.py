"""Controlled toy world for the ShiftWM mechanism walkthrough (appendix).

64x64 RGB frames on a static, per-episode random striped background with
  * a shaded SQUARE (12x12 px) whose velocity is exactly the 2-D action (integer px/step, |v_i| <= 1),
  * a shaded DISC (radius 6) drifting at a constant, action-independent velocity (bounces off walls),
  * an OCCLUDER (16x16 slate block) that slides away at 1 px/step and progressively reveals a
    yellow CROSS hidden underneath -- an event transport cannot explain (the cross is never observed).
Features Z are the raw 4x4 patch pixels: a 16x16 grid of 48-D tokens, so predictions decode to RGB exactly.
Ground truth (positions, velocities, per-pixel object labels) is stored for evaluating the transport.

Usage: PYTHONPATH=src python scripts/v2/toy_world.py --out data/toy
"""
import argparse
import json
from pathlib import Path

import numpy as np

S, P, T = 64, 4, 16                     # frame size, patch size, episode length
SQ, DR, OC, CR = 12, 6, 16, 12          # square side, disc radius, occluder side, cross side
LABELS = {"bg": 0, "square": 1, "disc": 2, "occluder": 3, "cross": 4}
YY, XX = np.mgrid[0:S, 0:S].astype(np.float32)


def background(rng):
    th = rng.uniform(0, np.pi)
    f = rng.uniform(0.08, 0.16)
    ph = rng.uniform(0, 2 * np.pi)
    s = 0.5 + 0.5 * np.sin(f * (np.cos(th) * XX + np.sin(th) * YY) + ph)
    c0 = rng.uniform(0.55, 0.85, 3)
    c1 = rng.uniform(0.55, 0.85, 3)
    return (c0[None, None] * s[..., None] + c1[None, None] * (1 - s[..., None])).astype(np.float32)


def episode(rng):
    bg = background(rng)
    frames = np.zeros((T, S, S, 3), np.float32)
    labels = np.zeros((T, S, S), np.uint8)
    col_sq = np.array([0.85, 0.10, 0.15]); col_dc = np.array([0.10, 0.30, 0.85])
    col_oc = np.array([0.22, 0.25, 0.30]); col_cr = np.array([0.98, 0.80, 0.05])
    # cross (static, hidden) and occluder centred on it
    cx, cy = rng.integers(14, S - 14, 2)
    ov = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])[rng.integers(4)]
    op = np.array([cx - OC // 2, cy - OC // 2], np.int64)
    sp = rng.integers(0, S - SQ, 2).astype(np.int64)
    dp = rng.uniform(DR, S - DR, 2)
    dv = rng.choice([-1.0, 1.0], 2) * rng.choice([0.5, 1.0], 2)
    act = rng.integers(-1, 2, 2)
    actions, pos = [], []
    for t in range(T):
        img = bg.copy(); lab = np.zeros((S, S), np.uint8)
        m = (np.abs(XX - cx) < 2) & (np.abs(YY - cy) < CR / 2) | (np.abs(YY - cy) < 2) & (np.abs(XX - cx) < CR / 2)
        img[m] = col_cr; lab[m] = 4
        m = (XX >= op[0]) & (XX < op[0] + OC) & (YY >= op[1]) & (YY < op[1] + OC)
        img[m] = col_oc; lab[m] = 3
        d = np.hypot(XX - dp[0], YY - dp[1])
        m = d <= DR
        img[m] = (col_dc[None] * (1.15 - 0.5 * d[m, None] / DR)).clip(0, 1); lab[m] = 2
        m = (XX >= sp[0]) & (XX < sp[0] + SQ) & (YY >= sp[1]) & (YY < sp[1] + SQ)
        shade = 0.7 + 0.3 * ((XX - sp[0]) + (YY - sp[1])) / (2 * SQ)
        img[m] = (col_sq[None] * shade[m, None] + 0.1).clip(0, 1); lab[m] = 1
        frames[t], labels[t] = img, lab
        pos.append([*sp, *dp, *op])
        # action for the transition t -> t+1: persistent, resampled w.p. 0.25, kept inside the frame
        if rng.random() < 0.25:
            act = rng.integers(-1, 2, 2)
        while np.any(sp + act < 0) or np.any(sp + act > S - SQ):
            act = rng.integers(-1, 2, 2)
        actions.append(act.copy())
        sp = sp + act
        dp = dp + dv
        for i in range(2):
            if dp[i] < DR or dp[i] > S - DR:
                dv[i] = -dv[i]; dp[i] = np.clip(dp[i], DR, S - DR)
        op = op + ov
    return (np.round(frames * 255).astype(np.uint8), np.array(actions[:-1], np.float32), labels,
            np.array(pos, np.float32))


def patchify(frames):
    """[..., 64, 64, 3] in [0,1] -> [..., 256, 48] (row-major 16x16 grid, each token = 4x4x3 pixels)."""
    lead = frames.shape[:-3]
    g = S // P
    x = frames.reshape(*lead, g, P, g, P, 3)
    x = np.moveaxis(x, -4, -3) if isinstance(x, np.ndarray) else x.transpose(-4, -3)
    return x.reshape(*lead, g * g, P * P * 3)


def unpatchify(z):
    lead = z.shape[:-2]
    g = S // P
    x = z.reshape(*lead, g, g, P, P, 3)
    x = np.moveaxis(x, -3, -4) if isinstance(x, np.ndarray) else x.transpose(-3, -4)
    return x.reshape(*lead, S, S, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/toy")
    p.add_argument("--train", type=int, default=3000)
    p.add_argument("--val", type=int, default=150)
    p.add_argument("--test", type=int, default=300)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", a.train, 0), ("val", a.val, 1), ("test", a.test, 2)):
        rng = np.random.default_rng(seed)
        eps = [episode(rng) for _ in range(n)]
        np.savez_compressed(out / f"{split}.npz", frames=np.stack([e[0] for e in eps]),
                            actions=np.stack([e[1] for e in eps]), labels=np.stack([e[2] for e in eps]),
                            pos=np.stack([e[3] for e in eps]))
        if split == "train":   # uncompressed copy so the trainer can memory-map it (login-node RAM cap)
            np.save(out / "train_frames.npy", np.stack([e[0] for e in eps]))
        del eps
        print(split, n, flush=True)
    f = np.load(out / "train.npz")["frames"][:200].astype(np.float32) / 255
    z = patchify(f)
    stats = {"feature_mean": z.mean((0, 1, 2)).tolist(), "feature_std": (z.std((0, 1, 2)) + 1e-3).tolist()}
    (out / "stats.json").write_text(json.dumps(stats))


if __name__ == "__main__":
    main()
