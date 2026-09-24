"""Open-loop prediction evaluation of a trained DINO-WM (official or +ShiftWM head) with DINO-WM's own code/metrics.

Loads the checkpoint with the official `plan.load_model`, rebuilds the official val split (same seed as training),
and reports, using the official functions (`VWorldModel.forward/rollout/decode_obs`, `Trainer.err_eval` MSE,
`metrics.image_metrics.eval_images` = l1/l2/ssim/mse/psnr/lpips(vgg) on decoded frames in [-1,1]):

  teacher_forced : the official validation objective over the first `--tf_batches` batches of the val slice
                   loader (upstream only scores the first batch per epoch for images): z_{visual,proprio}_err on the
                   predicted slot (frame num_hist) + decoded-image metrics of that slot; persistence reference.
  rollout_H{k}   : upstream `openloop_rollout` protocol with FIXED horizons: `--n_rollouts` val trajectories/starts
                   drawn with the training seed, num_hist observed frames (frameskip-subsampled) and H future
                   actions; per-step latent MSE and decoded-image metrics against the encoded/true future frames.

    python scripts/external_dinowm_plugin/eval_openloop.py --run_dir runs/dinowm_plugin/outputs/pusht_dinowm \
        --out results/v2/external/dinowm_plugin/pusht/dinowm/openloop.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DWM = ROOT / "external" / "dino_wm"
sys.path[:0] = [str(DWM), str(ROOT / "src"), str(Path(__file__).resolve().parent)]
os.environ.setdefault("WANDB_MODE", "offline")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import hydra  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import shims  # noqa: E402

shims.install_cpu_mask_shim()
shims.install_pointmaze_stub()
if os.environ.get("DINOWM_FAST_SLICES") == "1":
    shims.install_fast_slices()

from metrics.image_metrics import eval_images  # noqa: E402
from plan import load_model  # noqa: E402
from utils import seed, slice_trajdict_with_t  # noqa: E402


def _acc(store, key, val, n=1):
    s, c = store.get(key, (0.0, 0))
    store[key] = (s + float(val) * n, c + n)


def _img(prefix, store, pred, tgt, n):
    for k, v in eval_images(pred, tgt).items():
        _acc(store, f"{prefix}img_{k}", v.item() if torch.is_tensor(v) else v, n)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tf_batches", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_rollouts", type=int, default=200)
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--n_rollout_data", type=int, default=None, help="tiny smoke tests only")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(run / "hydra.yaml")
    os.chdir(DWM)  # official code expects to run from its repo root
    if args.n_rollout_data:
        cfg.env.dataset.n_rollout = args.n_rollout_data
    seed(cfg.training.seed)  # same permutation of val slices as during training
    datasets, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
                                      frameskip=cfg.frameskip)
    model = load_model(run / "checkpoints" / f"model_{args.epoch}.pth", cfg, cfg.num_action_repeat, dev)
    model.eval()
    H, fs = cfg.num_hist, cfg.frameskip
    res = {"run_dir": str(run), "epoch": args.epoch, "predictor": type(model.predictor).__name__,
           "cfg": {"env": cfg.env.name, "frameskip": fs, "num_hist": H, "img_size": cfg.img_size}}

    # ---------------- teacher-forced one-step (official val objective) ----------------
    tf = {}
    dl = torch.utils.data.DataLoader(datasets["valid"], batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)
    nb = 0
    for obs, act, state in dl:
        if nb >= args.tf_batches:
            break
        obs = {k: v.to(dev) for k, v in obs.items()}
        act = act.to(dev)
        b = act.shape[0]
        z_pred, visual_pred, _, loss, comps = model(obs, act)
        for k, v in comps.items():
            _acc(tf, k, v.item(), b)
        z_obs_pred, _ = model.separate_emb(z_pred)
        z_gt = model.encode_obs(obs)
        for k in z_obs_pred:  # predicted slot = last (frame H); persistence = copy frame H-1
            _acc(tf, f"z_{k}_err_pred", model.emb_criterion(z_obs_pred[k][:, -1], z_gt[k][:, H]).item(), b)
            _acc(tf, f"z_{k}_err_persistence", model.emb_criterion(z_gt[k][:, H - 1], z_gt[k][:, H]).item(), b)
        if visual_pred is not None:
            _img("pred_", tf, visual_pred[:, -1], obs["visual"][:, H], b)
            rec = model.decode_obs({"visual": z_gt["visual"][:, H:H + 1], "proprio": z_gt["proprio"][:, H:H + 1]})[0]
            _img("recon_", tf, rec["visual"][:, 0], obs["visual"][:, H], b)
        nb += 1
    res["teacher_forced"] = {k: s / c for k, (s, c) in tf.items()}
    res["teacher_forced"]["n_slices"] = tf["loss"][1] if "loss" in tf else 0

    # ---------------- open-loop rollouts (official openloop_rollout protocol, fixed horizons) ----------------
    dset = traj["valid"]
    for hz in [int(x) for x in args.horizons.split(",")]:
        rng = np.random.RandomState(cfg.training.seed)
        ro, n_done, tries = {}, 0, 0
        while n_done < args.n_rollouts and tries < 50 * args.n_rollouts:
            tries += 1
            i = rng.randint(0, len(dset))
            T = dset.get_seq_length(i)
            span = (H - 1 + hz) * fs  # frames H-1+hz after the first context frame
            if T <= span:
                continue
            start = rng.randint(0, T - span)
            obs, act, _, _ = dset[i]
            sel = slice(start, start + span + 1, fs)
            o = {k: v[sel].unsqueeze(0).to(dev) for k, v in obs.items()}  # (1, H+hz, ...)
            a = rearrange(act[start:start + span], "(h f) d -> h (f d)", f=fs).unsqueeze(0).to(dev)  # (1, H-1+hz, A)
            z_obses, _ = model.rollout({k: v[:, :H] for k, v in o.items()}, a)
            z_true = model.encode_obs(o)
            for step in range(1, hz + 1):
                t = H - 1 + step
                for k in z_obses:
                    _acc(ro, f"z_{k}_err_step{step}", model.emb_criterion(z_obses[k][:, t], z_true[k][:, t]).item())
                    if step == hz:
                        _acc(ro, f"z_{k}_err_persistence_last",
                             model.emb_criterion(z_true[k][:, H - 1], z_true[k][:, t]).item())
            if model.decoder is not None:
                vis = model.decode_obs(slice_trajdict_with_t(z_obses, start_idx=H, end_idx=H + hz))[0]["visual"][0]
                for step in range(1, hz + 1):
                    _img(f"step{step}_", ro, vis[step - 1:step], o["visual"][0, H - 1 + step:H + step], 1)
            n_done += 1
        out = {k: s / c for k, (s, c) in ro.items()}
        out["n_rollouts"] = n_done
        res[f"rollout_H{hz}"] = out
    res["seconds"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cfg"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
