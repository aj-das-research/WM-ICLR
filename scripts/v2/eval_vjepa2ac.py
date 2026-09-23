"""Zero-shot V-JEPA 2-AC (Assran et al. 2025, arXiv 2506.09985) forecasting on our DROID test split.

External reference for ICLR2027_PLAN.md §3 E2. Everything is measured in V-JEPA 2-AC's OWN feature space
(layer-normalised ViT-g/16 patch tokens, 16x16x1408 per frame), against the encodings of the true future
frames, next to a persistence (copy last encoded context frame) and a linear-extrapolation reference in
the same space, so that a relative gain vs persistence can be put next to ours.

Faithfulness to the released code (external/vjepa2, commit recorded in references/vjepa2_ac_sources.json):
  * model: src.hub.backbones._make_vjepa2_ac_model (vit_giant_xformers encoder, 256 px, patch 16,
    tubelet 2, RoPE; vit_ac_predictor depth 24 / dim 1024 / 16 heads, frame-causal, action_embed_dim 7,
    no extrinsics), weights = official vjepa2-ac-vitg.pt ("encoder" + "predictor" keys, the hub's own
    key cleaning; predictor loaded strict).
  * per-frame encoding exactly as app/vjepa_droid/train.py::forward_target and
    notebooks/utils/world_model_wrapper.py::encode: every frame is duplicated into a 2-frame tubelet,
    encoded alone (256 tokens), then F.layer_norm over channels (normalize_reps=True).
  * preprocessing = their DROID training transform (configs/train/vitg16/droid-256px-8f.yaml:
    random_resize_scale=(1.777,1.777), aspect=(0.75,1.35), crop 256, no flip). With scale>1 the Inception
    sampler can never fit and deterministically falls back to a central crop of aspect 1.35
    (for 180x320: rows 0..180, cols 38..281), bilinearly resized to 256x256 (align_corners=False),
    ImageNet mean/std on 0..255 values. --crop square gives the notebook's inference transform instead.
  * action/state convention = app/vjepa_droid/droid.py: state = [x,y,z, euler-xyz(3), gripper] (7-D,
    robot base frame), action_t = poses_to_diffs(state_t, state_{t+1}) = [dxyz, euler_xyz(R_{t+1} R_t^T),
    d gripper]. The predictor gets (frame tokens, action, state) per frame; output block t = frame t+1.
  * AR rollout as in their training loop / WorldModel.step_predictor: predictor(z_ctx, a, s)[:, -256:],
    layer-normed, appended to the context.

Protocol mapping (documented in the summary JSON too):
  * Their DROID clips: fps=4 -> fstp=ceil(15/4)=4 native 15 Hz frames per step (3.75 Hz), 8 frames per
    clip (tubelet_size=1 at the loader, i.e. one latent step per sampled frame), states from
    robot_state at those frames. Ours: 3 Hz frames = every 5 native frames. We map ONE of our steps to ONE
    predictor step (dt 0.333 s vs their 0.267 s, ratio 1.25; the closest faithful mapping that keeps our
    10-step horizon on the same frames as our models). Actions are the true pose deltas over our step,
    NOT rescaled.
  * Our npz stores the 15 Hz COMMANDED cartesian_position+gripper (block t = native commands 5t..5t+4),
    not the measured robot_state. State proxy at frame t (native 5t): the last command issued before it,
    cmd[5t-1] = actions[t-1, 28:35] (t>=1); t=0 uses cmd[0]. Hence action_t only uses commands issued
    during interval t (causal, same information our models get).
  * Windows: identical to ours (shiftwm.v2.train.FeatureSplit): episodes with T >= history+horizon,
    starts range(0, T-13+1, 2), history 3 encoded GT frames (with their 3 states and the 2 in-history
    actions), horizon 10 autoregressive steps with the recorded actions. The predictor was trained on
    8-frame clips; the context is a sliding window of the last --max-context (default 8) frames so that
    no RoPE temporal offset exceeds the trained range (--max-context 0 = keep everything, 12 frames max).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CROP = 256
PATCH = 16
GRID = CROP // PATCH
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_SCALE = (1.777, 1.777)      # droid-256px-8f.yaml data_aug.random_resize_scale
TRAIN_RATIO = (0.75, 1.35)        # droid-256px-8f.yaml data_aug.random_resize_aspect_ratio
NATIVE_PER_STEP = 5               # our processed DROID: images at native 0,5,10,... (15 Hz source)
THEIR_NATIVE_PER_STEP = math.ceil(15 / 4)  # droid.py: fstp = ceil(vfps / fps), fps=4


# ----------------------------------------------------------------------------------------------------
# vendored-code plumbing
# ----------------------------------------------------------------------------------------------------
def _install_timm_shim():
    """src/models/utils/modules.py imports timm only for drop_path (identity at eval, rate 0 here)."""
    try:
        import timm.models.layers  # noqa: F401
        return
    except ImportError:
        pass

    def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):  # == timm.layers.drop_path
        if drop_prob == 0.0 or not training:
            return x
        keep = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        if keep > 0.0 and scale_by_keep:
            mask.div_(keep)
        return x * mask

    timm = types.ModuleType("timm"); models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers"); layers.drop_path = drop_path
    timm.models = models; models.layers = layers
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})


def add_repo(repo):
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    _install_timm_shim()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model(repo, ckpt, device):
    add_repo(repo)
    from src.hub.backbones import _clean_backbone_key, _make_vjepa2_ac_model
    encoder, predictor = _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=CROP, pretrained=False)
    try:
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    except Exception as e:  # legacy/non-zip or non-tensor payloads
        print(f"[load] mmap/weights_only load failed ({e!r}); plain torch.load", flush=True)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc = _clean_backbone_key(dict(sd["encoder"]))
    missing, unexpected = encoder.load_state_dict(enc, strict=False)  # hub: strict=False (pos_embed vs RoPE)
    if missing:
        raise RuntimeError(f"encoder missing keys: {missing[:10]}")
    predictor.load_state_dict(_clean_backbone_key(dict(sd["predictor"])), strict=True)
    info = {"encoder_unexpected_keys": list(unexpected), "checkpoint_keys": sorted(sd.keys()),
            "epoch": int(sd["epoch"]) if "epoch" in sd and sd["epoch"] is not None else None}
    del sd, enc
    encoder.to(device).eval().requires_grad_(False)
    predictor.to(device).eval().requires_grad_(False)
    return encoder, predictor, info


# ----------------------------------------------------------------------------------------------------
# data / action conversion (pure, CPU-testable)
# ----------------------------------------------------------------------------------------------------
def crop_box(height, width, mode="train"):
    """Deterministic crop of their transform; mirrors _get_param_spatial_crop's fallback branch."""
    scale, ratio = (TRAIN_SCALE, TRAIN_RATIO) if mode == "train" else ((1.0, 1.0), (1.0, 1.0))
    # the random branch requires w<=W and h<=H with w*h ~= scale*W*H; impossible for scale>=1 unless the
    # aspect matches exactly -> assert we are in the deterministic fallback (true for 16:9 DROID frames).
    for ar in ratio:
        w = int(round(math.sqrt(scale[0] * height * width * ar)))
        h = int(round(math.sqrt(scale[0] * height * width / ar)))
        assert not (0 < w <= width and 0 < h <= height), "crop would be random; unsupported input aspect"
    in_ratio = width / height
    if in_ratio < min(ratio):
        w = width; h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height; w = int(round(h * max(ratio)))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


def preprocess(images, mode="train", device="cpu"):
    """uint8 [N,H,W,3] -> float [N,3,256,256], identical math to their VideoTransform (no aug)."""
    n, H, W, _ = images.shape
    i, j, h, w = crop_box(H, W, mode)
    x = torch.as_tensor(images).to(device).permute(0, 3, 1, 2).float()[:, :, i:i + h, j:j + w]
    x = F.interpolate(x, size=(CROP, CROP), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=device)[None, :, None, None] * 255.0
    std = torch.tensor(IMAGENET_STD, device=device)[None, :, None, None] * 255.0
    return (x - mean) / std


def frame_states(actions35):
    """[T-1,35] (5 commanded 7-D poses per step) -> [T,7] state proxy per frame (see module doc)."""
    cmd = np.asarray(actions35, dtype=np.float64).reshape(len(actions35), NATIVE_PER_STEP, 7)
    return np.concatenate([cmd[:1, 0], cmd[:, -1]], 0)


def poses_to_diffs(states):
    """[T,7] -> [T-1,7]; vectorised app/vjepa_droid/droid.py::DROIDVideoDataset.poses_to_diffs."""
    from scipy.spatial.transform import Rotation
    s = np.asarray(states, dtype=np.float64)
    r = Rotation.from_euler("xyz", s[:, 3:6], degrees=False).as_matrix()
    rel = np.einsum("tij,tkj->tik", r[1:], r[:-1])            # R_{t+1} @ R_t^T
    dtheta = Rotation.from_matrix(rel).as_euler("xyz", degrees=False)
    return np.concatenate([s[1:, :3] - s[:-1, :3], dtheta, s[1:, 6:] - s[:-1, 6:]], 1)


def integrate_states(s0, actions):
    """compute_new_pose (notebooks/utils/mpc_utils.py) applied recursively. s0 [B,7], actions [B,K,7]."""
    from scipy.spatial.transform import Rotation
    s0 = np.asarray(s0, np.float64); a = np.asarray(actions, np.float64)
    out = [s0]
    for k in range(a.shape[1]):
        s, d = out[-1], a[:, k]
        R = Rotation.from_euler("xyz", d[:, 3:6]).as_matrix() @ Rotation.from_euler("xyz", s[:, 3:6]).as_matrix()
        out.append(np.concatenate([s[:, :3] + d[:, :3], Rotation.from_matrix(R).as_euler("xyz"),
                                   np.clip(s[:, 6:] + d[:, 6:], 0, 1)], 1))
    return np.stack(out, 1)                                    # [B,K+1,7]


def window_starts(T, history, horizon, stride):
    """== shiftwm.v2.train.FeatureSplit: arange(0, T-need+1, stride), need = history+horizon."""
    need = history + horizon
    return list(range(0, T - need + 1, stride)) if T >= need else []


def derangement(n, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    fix = np.nonzero(p == np.arange(n))[0]
    for i in fix:                                               # swap fixed points with a neighbour
        j = (i + 1) % n
        p[i], p[j] = p[j], p[i]
    assert n < 2 or not np.any(p == np.arange(n))
    return p


# ----------------------------------------------------------------------------------------------------
# model-side helpers
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_frames(encoder, x, batch=64, normalize=True):
    """x [N,3,256,256] -> [N,256,D]; single frame duplicated into a 2-frame tubelet (their forward_target)."""
    out = []
    for i in range(0, len(x), batch):
        c = x[i:i + batch].unsqueeze(2).repeat(1, 1, 2, 1, 1)      # [B,C,2,H,W]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            h = encoder(c)
        h = h.float()
        if normalize:
            h = F.layer_norm(h, (h.size(-1),))
        out.append(h.half())
    return torch.cat(out)


@torch.no_grad()
def rollout(predictor, z_hist, states, actions, horizon, max_context=8, normalize=True):
    """z_hist [B,h,N,D]; states [B,h+K-1,7] (frames s..s+h+K-2); actions [B,h+K-1,7] (transition j->j+1).
    Returns predictions [B,K,N,D] for frames s+h .. s+h+K-1."""
    B, h, N, D = z_hist.shape
    seq = [z_hist[:, t] for t in range(h)]
    preds = []
    for k in range(horizon):
        L = h + k
        lo = 0 if not max_context else max(0, L - max_context)
        z = torch.stack(seq[lo:L], 1).flatten(1, 2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z.is_cuda):
            nxt = predictor(z, actions[:, lo:L], states[:, lo:L])[:, -N:]
        nxt = nxt.float()
        if normalize:
            nxt = F.layer_norm(nxt, (D,))
        seq.append(nxt)
        preds.append(nxt)
    return torch.stack(preds, 1)


def metrics(pred, target):
    """pred/target [B,K,N,D] float -> dict of [B,K] (same definitions as shiftwm.v2.train.evaluate)."""
    B, K, N, D = pred.shape
    g = int(math.isqrt(N))
    def pool(x):
        return F.adaptive_avg_pool2d(x.reshape(B * K, g, g, D).permute(0, 3, 1, 2), 4).reshape(B, K, -1)
    d = pred - target
    return {"mse": (d ** 2).mean((-1, -2)), "l1": d.abs().mean((-1, -2)),
            "cos": F.cosine_similarity(pred, target, dim=-1).mean(-1),
            "mse_pool4": ((pool(pred) - pool(target)) ** 2).mean(-1)}


# ----------------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------------
def load_episodes(data, split, camera, history, horizon, limit=None):
    manifest = json.loads((data / "manifest.json").read_text())
    rows = [r for r in manifest["episodes"] if r["split"] == split]
    eps = []
    for r in rows:
        rec = r["cameras"][camera]
        if rec["frames"] < history + horizon:
            continue
        eps.append({"id": r["episode_id"], "session": r["session_id"], "file": data / rec["file"],
                    "T": rec["frames"]})
    return eps[:limit] if limit else eps


def bootstrap_gain(model_ep, pers_ep, groups, n=10000, seed=0):
    """95% CI of 1 - sum(model)/sum(pers) resampling groups (sessions); arrays [E] per-episode values."""
    uniq, inv = np.unique(groups, return_inverse=True)
    m = np.bincount(inv, model_ep); p = np.bincount(inv, pers_ep); c = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (n, len(uniq)))
    # episode-weighted means within resample: sum over chosen groups / count
    gm = m[idx].sum(1) / c[idx].sum(1); gp = p[idx].sum(1) / c[idx].sum(1)
    return np.percentile(1 - gm / gp, [2.5, 97.5]).tolist()


def run(a):
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(a.data)
    eps = load_episodes(data, a.split, a.camera, a.history, a.horizon, a.limit_episodes)
    H, K = a.history, a.horizon
    print(f"[data] {len(eps)} {a.split} episodes (T >= {H + K}), device {dev}", flush=True)
    encoder, predictor, load_info = load_model(a.repo, a.checkpoint, dev)
    print(f"[model] loaded in {time.time() - t0:.0f}s; encoder dim {encoder.embed_dim}", flush=True)

    # 1) encode every frame once; build states/actions per episode
    feats, states, acts, win = [], [], [], []
    offset = 0
    for e_i, e in enumerate(eps):
        with np.load(e["file"]) as z:
            imgs, act, fidx = z["images"], z["actions"], z["frame_indices"]
        T = len(imgs)
        assert act.shape == (T - 1, 35), act.shape
        assert np.array_equal(fidx - fidx[0], NATIVE_PER_STEP * np.arange(T)), "unexpected frame clock"
        x = preprocess(imgs, a.crop, dev)
        feats.append(encode_frames(encoder, x, a.encode_batch))
        s = frame_states(act)
        states.append(torch.from_numpy(s).float()); acts.append(torch.from_numpy(poses_to_diffs(s)).float())
        for st in window_starts(T, H, K, a.stride):
            win.append((e_i, offset + st, st))
        offset += T
        if e_i % 10 == 0:
            print(f"[encode] {e_i + 1}/{len(eps)} episodes, {offset} frames, {time.time() - t0:.0f}s", flush=True)
    Z = torch.cat(feats)                                     # [F,N,D] fp16 on device
    S = torch.cat(states).to(dev)                            # [F,7]
    A = torch.cat([torch.cat([x, torch.zeros(1, 7)]) for x in acts]).to(dev)  # pad: action idx == frame idx
    N, D = Z.shape[1:]
    W = len(win)
    w_ep = np.array([w[0] for w in win]); w_glob = torch.tensor([w[1] for w in win], device=dev)
    perm = torch.from_numpy(derangement(W, a.seed)).to(dev)
    print(f"[windows] {W} windows; features {tuple(Z.shape)} ({Z.numel() * 2 / 2**30:.1f} GiB)", flush=True)

    names = ["model", "persistence", "linear"] + (["shuffled"] if a.shuffled else [])
    per_win = {f"{n}_{m}": np.zeros((W, K)) for n in names for m in ("mse", "cos", "l1", "mse_pool4")}
    per_win["sens"] = np.zeros((W, K))
    ar = torch.arange(H + K, device=dev)
    t1 = time.time()
    for b0 in range(0, W, a.batch_size):
        bi = torch.arange(b0, min(b0 + a.batch_size, W), device=dev)
        t = w_glob[bi, None] + ar[None]                      # [B,H+K] global frame index
        z_true = Z[t].float()                                # [B,H+K,N,D]
        s_in = S[t[:, :-1]]                                  # states of frames s..s+H+K-2
        a_in = A[t[:, :-1]]                                  # actions j->j+1
        target = z_true[:, H:]
        pred = rollout(predictor, z_true[:, :H], s_in, a_in, K, a.max_context)
        res = {"model": metrics(pred, target)}
        last = z_true[:, H - 1:H]
        res["persistence"] = metrics(last.expand_as(target), target)
        steps = torch.arange(1, K + 1, device=dev, dtype=torch.float32)[None, :, None, None]
        res["linear"] = metrics(last + steps * (last - z_true[:, H - 2:H - 1]), target)
        if a.shuffled:
            # future actions of another window; future states re-integrated with compute_new_pose
            ta = w_glob[perm[bi], None] + ar[None]
            a_fut = A[ta[:, H - 1:H + K - 1]]
            s_fut = integrate_states(s_in[:, H - 1].cpu().numpy(), a_fut[:, :-1].cpu().numpy())
            s_sh = torch.cat([s_in[:, :H - 1], torch.from_numpy(s_fut).float().to(dev)], 1)
            a_sh = torch.cat([a_in[:, :H - 1], a_fut], 1)
            wrong = rollout(predictor, z_true[:, :H], s_sh, a_sh, K, a.max_context)
            res["shuffled"] = metrics(wrong, target)
            per_win["sens"][b0:b0 + len(bi)] = ((wrong - pred) ** 2).mean((-1, -2)).cpu().numpy()
        for n, r in res.items():
            for m, v in r.items():
                per_win[f"{n}_{m}"][b0:b0 + len(bi)] = v.double().cpu().numpy()
        if (b0 // a.batch_size) % 10 == 0:
            done = b0 + len(bi)
            print(f"[rollout] {done}/{W} windows, {(time.time() - t1) / done * (W - done) / 60:.1f} min left; "
                  f"running mse model {per_win['model_mse'][:done].mean():.4f} "
                  f"pers {per_win['persistence_mse'][:done].mean():.4f}", flush=True)

    # 2) aggregate per episode exactly like shiftwm.v2.train.evaluate (mean over the episode's windows)
    E = len(eps)
    counts = np.bincount(w_ep, minlength=E).astype(float)
    per_ep = {k: np.stack([v[w_ep == e].mean(0) for e in range(E)]) for k, v in per_win.items()}
    if a.shuffled:
        per_win["rank_ok"] = (per_win["model_mse"] < per_win["shuffled_mse"]).astype(float)
        per_ep["rank_ok"] = np.stack([per_win["rank_ok"][w_ep == e].mean(0) for e in range(E)])
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, episodes=np.array([e["id"] for e in eps]), sessions=np.array([e["session"] for e in eps]),
                        windows=counts, win_episode=w_ep, win_start=np.array([w[2] for w in win]),
                        **per_ep, **{f"win_{k}": v.astype(np.float32) for k, v in per_win.items()})

    sessions = np.array([e["session"] for e in eps])
    mean_h = lambda k: per_ep[k].mean(0).tolist()
    summary = {"model": "V-JEPA 2-AC (vjepa2-ac-vitg), zero-shot", "split": a.split, "camera": a.camera,
               "episodes": E, "sessions": int(len(np.unique(sessions))), "windows": W,
               "per_horizon": {k: mean_h(k) for k in per_ep}}
    g = {}
    for m in ("mse", "l1", "mse_pool4"):
        mm, pp = per_ep[f"model_{m}"], per_ep[f"persistence_{m}"]
        g[m] = {"per_horizon": (1 - mm.mean(0) / pp.mean(0)).tolist(),
                "mean_over_horizons": float(1 - mm.mean() / pp.mean()),
                "mean_over_horizons_ci95_session_bootstrap": bootstrap_gain(mm.mean(1), pp.mean(1), sessions),
                "h1": float(1 - mm[:, 0].mean() / pp[:, 0].mean()),
                "h_end": float(1 - mm[:, -1].mean() / pp[:, -1].mean()),
                "h_end_ci95_session_bootstrap": bootstrap_gain(mm[:, -1], pp[:, -1], sessions)}
    summary["relative_gain_vs_persistence"] = g
    summary["cos_delta_vs_persistence_per_horizon"] = (per_ep["model_cos"].mean(0)
                                                       - per_ep["persistence_cos"].mean(0)).tolist()
    summary["model_beats_persistence_fraction_of_episodes_mse_mean_h"] = float(
        np.mean(per_ep["model_mse"].mean(1) < per_ep["persistence_mse"].mean(1)))
    if a.shuffled:
        summary["action_rank_acc"] = float(per_ep["rank_ok"].mean())
    summary["protocol"] = {
        "history": H, "horizon": K, "window_stride": a.stride, "max_context_frames": a.max_context,
        "crop": a.crop, "crop_box_ijhw_for_180x320": list(crop_box(180, 320, a.crop)),
        "native_frames_per_our_step": NATIVE_PER_STEP, "native_frames_per_their_step": THEIR_NATIVE_PER_STEP,
        "step_mapping": "1 of our 3 Hz steps (5 native @15 Hz, 0.333 s) -> 1 predictor step (trained at "
                        "4 native, 0.267 s); actions = true pose deltas over our step, not rescaled",
        "state_proxy": "state_t = last commanded cartesian_position+gripper before native frame 5t "
                       "(actions[t-1,28:35]); state_0 = actions[0,0:7]",
        "action": "poses_to_diffs(state_t, state_t+1): dxyz, euler_xyz(R_t+1 R_t^T), dgripper",
        "feature_space": "F.layer_norm(ViT-g encoder tokens) per frame (duplicated 2-frame tubelet), "
                         "256 tokens x 1408; predictions layer-normed (normalize_reps=True)",
        "metrics": "mse/l1 mean over tokens and channels; cos per token then mean; mse_pool4 = 4x4 avg-pooled "
                   "grid; per-episode mean over windows, then mean over episodes (as ours)",
        "shuffled": "future actions from a deranged other window (seed %d); states re-integrated with "
                    "compute_new_pose from the true last context state" % a.seed,
        "linear": "z_last + k (z_last - z_prev) in layer-normed space (not renormalised)"}
    summary["load_info"] = load_info
    summary["checkpoint"] = {"path": str(a.checkpoint), "sha256": a.checkpoint_sha256 or None}
    summary["runtime_s"] = time.time() - t0
    summary["peak_gpu_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None
    Path(a.summary or str(out).replace(".npz", "_summary.json")).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("episodes", "windows", "runtime_s")}), flush=True)
    print(json.dumps(summary["relative_gain_vs_persistence"]["mse"]), flush=True)


# ----------------------------------------------------------------------------------------------------
# CPU self-test on tiny inputs (no ViT-g)
# ----------------------------------------------------------------------------------------------------
def selftest(a):
    add_repo(a.repo)
    rng = np.random.default_rng(0)
    # (a) state proxy / action conversion vs their reference implementations
    T = 7
    cmd = np.zeros((T - 1, 5, 7))
    base = np.array([0.4, 0.0, 0.3, 3.1, -0.05, 0.02, 0.0])
    walk = np.cumsum(rng.normal(0, [0.004, 0.004, 0.004, 0.01, 0.01, 0.01, 0.05], (5 * (T - 1), 7)), 0) + base
    walk[:, 6] = np.clip(walk[:, 6], 0, 1)
    cmd[:] = walk.reshape(T - 1, 5, 7)
    s = frame_states(cmd.reshape(T - 1, 35))
    assert s.shape == (T, 7)
    assert np.allclose(s[0], walk[0]) and all(np.allclose(s[t], walk[5 * t - 1]) for t in range(1, T))
    act = poses_to_diffs(s)
    sys.path.insert(0, str(Path(a.repo) / "notebooks"))
    from utils.mpc_utils import compute_new_pose, poses_to_diff  # their notebook helpers
    ref = np.stack([poses_to_diff(s[t], s[t + 1]).numpy() for t in range(T - 1)])
    assert np.allclose(act, ref, atol=1e-10), np.abs(act - ref).max()
    # droid.py::poses_to_diffs (class method; pure numpy) -- exec its source without h5py/decord deps
    import ast
    src = (Path(a.repo) / "app/vjepa_droid/droid.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "poses_to_diffs")
    ns = {"np": np}
    from scipy.spatial.transform import Rotation
    ns["Rotation"] = Rotation
    exec(compile(ast.Module([fn], []), "droid_poses_to_diffs", "exec"), ns)
    ref2 = ns["poses_to_diffs"](None, s)
    assert np.allclose(act, ref2, atol=1e-10)
    # integration of true actions reproduces the states (their compute_new_pose, recursively)
    integ = integrate_states(s[None, 0], act[None])[0]
    assert np.allclose(integ, s, atol=1e-8), np.abs(integ - s).max()
    st = torch.tensor(s[None, :1]); at = torch.tensor(act[None, :1])
    assert np.allclose(compute_new_pose(st, at).numpy()[0, 0], integ[1], atol=1e-8)
    print("[selftest] states/actions: OK (matches droid.py poses_to_diffs, mpc_utils poses_to_diff/compute_new_pose)")

    # (b) preprocessing vs their training transform (make_transforms with the droid config values)
    from app.vjepa_droid.transforms import make_transforms
    imgs = rng.integers(0, 256, (3, 180, 320, 3), dtype=np.uint8)
    for mode, sc, ra in (("train", TRAIN_SCALE, TRAIN_RATIO), ("square", (1.0, 1.0), (1.0, 1.0))):
        tf = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=ra, random_resize_scale=sc,
                             reprob=0.0, auto_augment=False, motion_shift=False, crop_size=CROP)
        theirs = tf(imgs).permute(1, 0, 2, 3)          # C T H W -> T C H W
        ours = preprocess(imgs, mode)
        assert ours.shape == theirs.shape == (3, 3, CROP, CROP)
        assert torch.allclose(ours, theirs, atol=1e-4), (ours - theirs).abs().max()
    assert crop_box(180, 320, "train") == (0, 38, 180, 243)
    print("[selftest] preprocessing: OK (== their VideoTransform; train crop box (0,38,180,243))")

    # (c) windows == our FeatureSplit protocol on the real manifest (metadata only)
    data = Path(a.data)
    eps = load_episodes(data, "test", a.camera, 3, 10)
    nwin = sum(len(window_starts(e["T"], 3, 10, 2)) for e in eps)
    print(f"[selftest] test episodes {len(eps)}, windows {nwin} (ours: 130 episodes / 3923 windows)")
    assert window_starts(13, 3, 10, 2) == [0] and window_starts(12, 3, 10, 2) == []
    assert window_starts(16, 3, 10, 2) == [0, 2]

    # (d) rollout plumbing with a tiny random predictor (their class), grid 2x2
    from src.models.ac_predictor import vit_ac_predictor
    torch.manual_seed(0)
    pred = vit_ac_predictor(img_size=(32, 32), patch_size=16, num_frames=64, tubelet_size=2, embed_dim=24,
                            predictor_embed_dim=32, depth=2, num_heads=2).eval()
    B, Hh, Kk, N, D = 3, 3, 4, 4, 24
    z = F.layer_norm(torch.randn(B, Hh + Kk, N, D), (D,))
    ss = torch.randn(B, Hh + Kk - 1, 7); aa = torch.randn(B, Hh + Kk - 1, 7)
    p_full = rollout(pred, z[:, :Hh], ss, aa, Kk, max_context=0)
    p_win = rollout(pred, z[:, :Hh], ss, aa, Kk, max_context=5)
    assert p_full.shape == (B, Kk, N, D)
    # step 1 == teacher-forced predictor output on the history (frame-causal)
    tf1 = F.layer_norm(pred(z[:, :Hh].flatten(1, 2), aa[:, :Hh], ss[:, :Hh])[:, -N:], (D,))
    assert torch.allclose(p_full[:, 0], tf1, atol=1e-5)
    # frame-causal mask: last block of a Hh-frame call equals block Hh-1 of a longer call
    long = pred(z[:, :Hh + 1].flatten(1, 2), aa[:, :Hh + 1], ss[:, :Hh + 1]).view(B, Hh + 1, N, D)
    assert torch.allclose(F.layer_norm(long[:, Hh - 1], (D,)), tf1, atol=1e-5)
    # sliding window only changes steps whose context exceeds max_context (h+k > 5 -> k >= 3)
    assert torch.allclose(p_full[:, :3], p_win[:, :3], atol=1e-5) and not torch.allclose(p_full[:, 3], p_win[:, 3])
    m = metrics(z[:, Hh - 1:Hh].expand(B, Kk, N, D), z[:, Hh:])
    assert all(v.shape == (B, Kk) for v in m.values())
    mz = metrics(z[:, Hh:], z[:, Hh:])
    assert torch.allclose(mz["mse"], torch.zeros(B, Kk)) and torch.allclose(mz["cos"], torch.ones(B, Kk))
    p = derangement(7); assert sorted(p.tolist()) == list(range(7)) and not np.any(p == np.arange(7))
    # (e) one real episode: state/action statistics and conversion sanity
    e = eps[0]
    with np.load(e["file"]) as zz:
        act35, fidx = zz["actions"], zz["frame_indices"]
    assert np.array_equal(fidx - fidx[0], 5 * np.arange(len(fidx)))
    sr = frame_states(act35); ar_ = poses_to_diffs(sr)
    print(f"[selftest] real episode {e['id']}: T={len(sr)}, |dxyz| mean {np.linalg.norm(ar_[:, :3], axis=1).mean():.4f} m/step, "
          f"|dtheta| mean {np.linalg.norm(ar_[:, 3:6], axis=1).mean():.4f} rad/step, gripper range "
          f"[{sr[:, 6].min():.2f},{sr[:, 6].max():.2f}]")
    print("[selftest] rollout/metrics plumbing: OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data/real_video/droid_selected/processed"))
    p.add_argument("--split", default="test")
    p.add_argument("--camera", default="exterior_image_1_left")
    p.add_argument("--repo", default=str(ROOT / "external/vjepa2"))
    p.add_argument("--checkpoint", default=str(ROOT / "data/pretrained/vjepa2_ac/vjepa2-ac-vitg.pt"))
    p.add_argument("--checkpoint-sha256", default=None, help="recorded into the summary (not re-hashed)")
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-context", type=int, default=8, help="sliding predictor context (frames); 0 = all")
    p.add_argument("--crop", choices=("train", "square"), default="train")
    p.add_argument("--batch-size", type=int, default=32, help="windows per predictor batch")
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--no-shuffled", dest="shuffled", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-episodes", type=int, default=None)
    p.add_argument("--output", default=str(ROOT / "results/v2/external/vjepa2ac/droid_test.npz"))
    p.add_argument("--summary", default=None)
    p.add_argument("--selftest", action="store_true", help="CPU checks on tiny inputs (no ViT-g)")
    a = p.parse_args()
    if a.selftest:
        torch.set_num_threads(2)
        selftest(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
