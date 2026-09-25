"""Largest-advantage (and largest-disadvantage) qualitative rollouts of DINO-WM with vs without the ShiftWM head.

Same start states and actions for both models: the upstream `openloop_rollout` protocol of eval_openloop.py (rollout_H5,
`--n_rollouts` val trajectories/starts drawn with the training seed, num_hist observed frames, 5 open-loop steps of
frameskip 5), i.e. exactly the rollouts behind openloop.json. For each rollout and model we keep the step-k latent
error (official emb_criterion = MSE of the DINOv2 patch features), the per-patch error, and the frames decoded by each
model's own decoder.

Selection rule (as scripts/v2/qual_select.py): among rollouts in the top 50% of true change ||z_{t+k}-z_t||^2
(persistence error), rank by gain = 1 - err_{+ShiftWM} / err_{DINO-WM}; keep the top 3 from distinct trajectories
("best"); the bottom 3 by the same rule are stored too ("worst"), for honest display where the head regresses (Wall).

Output: results/v2/analysis/qual_best/dinowm_<env>.{npz,json}
Run with the DINO-WM venv (slurm_jebel/tasks/dinowm_common.sh):
  python scripts/external_dinowm_plugin/qual_best.py --env pusht
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

from plan import load_model  # noqa: E402
from utils import seed, slice_trajdict_with_t  # noqa: E402

RUNS = ROOT / "runs/dinowm_plugin/outputs"
RES = ROOT / "results/v2/external/dinowm_plugin"
OUT = ROOT / "results/v2/analysis/qual_best"
ARMS = ("dinowm", "dinowm_shiftwm")
N_PICK = 3


def to_u8(x):
    """[3,H,W] in [-1,1] -> uint8 [H,W,3]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def pick(order, traj, n):
    out, seen = [], set()
    for w in order:
        if traj[w] not in seen:
            out.append(int(w)); seen.add(traj[w])
        if len(out) == n:
            break
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n_rollouts", type=int, default=200)
    ap.add_argument("--out", default=str(OUT), help="override (smoke tests)")
    a = ap.parse_args()
    out = Path(a.out)
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {arm: RUNS / f"{a.env}_{arm}" for arm in ARMS}
    epochs = {arm: json.loads((RES / a.env / arm / "openloop.json").read_text())["epoch"] for arm in ARMS}
    cfgs = {arm: OmegaConf.load(runs[arm] / "hydra.yaml") for arm in ARMS}
    cfg = cfgs["dinowm"]
    assert all(cfgs[x].training.seed == cfg.training.seed and cfgs[x].num_hist == cfg.num_hist
               and cfgs[x].frameskip == cfg.frameskip for x in ARMS)
    os.chdir(DWM)
    seed(cfg.training.seed)
    datasets, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
                                      frameskip=cfg.frameskip)
    models = {}
    for arm in ARMS:
        m = load_model(runs[arm] / "checkpoints" / f"model_{epochs[arm]}.pth", cfgs[arm], cfgs[arm].num_action_repeat, dev)
        m.eval(); models[arm] = m
    H, fs, hz = cfg.num_hist, cfg.frameskip, a.horizon
    dset = traj["valid"]
    rng = np.random.RandomState(cfg.training.seed)       # == eval_openloop.py rollout_H{hz}
    rec = {"err": [], "err_steps": [], "change": [], "traj": [], "start": [], "pix": []}
    store = []                                            # per rollout: images + per-patch errors (small)
    n_done, tries = 0, 0
    while n_done < a.n_rollouts and tries < 50 * a.n_rollouts:
        tries += 1
        i = rng.randint(0, len(dset))
        T = dset.get_seq_length(i)
        span = (H - 1 + hz) * fs
        if T <= span:
            continue
        start = rng.randint(0, T - span)
        obs, act, _, _ = dset[i]
        sel = slice(start, start + span + 1, fs)
        o = {k: v[sel].unsqueeze(0).to(dev) for k, v in obs.items()}
        ac = rearrange(act[start:start + span], "(h f) d -> h (f d)", f=fs).unsqueeze(0).to(dev)
        tl = H - 1 + hz
        z_true = models["dinowm"].encode_obs(o)["visual"]
        errs, steps, pp, dec, pix = [], [], [], [], []
        for arm in ARMS:
            m = models[arm]
            z_obses, _ = m.rollout({k: v[:, :H] for k, v in o.items()}, ac)
            zt = m.encode_obs(o)["visual"]
            d = (z_obses["visual"][0, H:H + hz] - zt[0, H:H + hz]) ** 2          # [hz, N, D]
            steps.append(d.mean((1, 2)).double().cpu().numpy())
            errs.append(float(d[-1].mean()))
            pp.append(d[-1].mean(-1).float().cpu().numpy())
            vis = m.decode_obs(slice_trajdict_with_t(z_obses, start_idx=H, end_idx=H + hz))[0]["visual"][0, hz - 1]
            dec.append(to_u8(vis))
            pix.append(float((((vis.clamp(-1, 1) + 1) / 2 - (o["visual"][0, tl].clamp(-1, 1) + 1) / 2) ** 2).mean()))
        rec["err"].append(errs); rec["err_steps"].append(np.stack(steps)); rec["pix"].append(pix)
        rec["change"].append(float(((z_true[0, tl] - z_true[0, H - 1]) ** 2).mean()))
        rec["traj"].append(i); rec["start"].append(start)
        store.append({"obs": to_u8(o["visual"][0, H - 1]), "true": to_u8(o["visual"][0, tl]),
                      "dec": np.stack(dec), "pp": np.stack(pp)})
        n_done += 1
    err = np.array(rec["err"]); change = np.array(rec["change"]); tr = np.array(rec["traj"])
    adv = 1 - err[:, 1] / np.maximum(err[:, 0], 1e-12)
    hi = change >= np.median(change)
    idx_hi = np.where(hi)[0]
    best = pick(idx_hi[np.argsort(-adv[hi], kind="stable")], tr, N_PICK)
    worst = pick(idx_hi[np.argsort(adv[hi], kind="stable")], tr, N_PICK)
    sel = best + worst
    g = int(round(store[0]["pp"].shape[-1] ** 0.5))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"dinowm_{a.env}.npz", kind=np.array(["best"] * len(best) + ["worst"] * len(worst)),
        rollout=np.array(sel), traj=tr[sel], start=np.array(rec["start"])[sel], adv=adv[sel], err=err[sel],
        pix=np.array(rec["pix"])[sel], arms=np.array(ARMS), change=change[sel], k=hz,
        perpatch=np.stack([store[w]["pp"] for w in sel]).reshape(len(sel), 2, g, g),
        frame_obs=np.stack([store[w]["obs"] for w in sel]), frame_true=np.stack([store[w]["true"] for w in sel]),
        decoded=np.stack([store[w]["dec"] for w in sel]),
        err_all=err, err_steps_all=np.stack(rec["err_steps"]), change_all=change, adv_all=adv, traj_all=tr)
    info = {
        "status": "done", "env": a.env, "k": hz, "frameskip": fs, "num_hist": H, "n_rollouts": n_done,
        "rule": ("Rollouts of eval_openloop.py rollout_H5 (same starts/actions for both models). Among rollouts in "
                 "the top 50% of true change ||z_{t+k}-z_t||^2, rank by gain 1 - err_{+ShiftWM}/err_{DINO-WM} at step "
                 f"k={hz}; best = top 3, worst = bottom 3, distinct trajectories."),
        "checkpoints": {arm: str((runs[arm] / "checkpoints" / f"model_{epochs[arm]}.pth").relative_to(ROOT))
                        for arm in ARMS},
        "mean_err_step_k": dict(zip(ARMS, err.mean(0).tolist())),
        "mean_err_per_step": {arm: np.stack(rec["err_steps"])[:, j].mean(0).tolist() for j, arm in enumerate(ARMS)},
        "context_all": {"frac_rollouts_shiftwm_better": float((adv > 0).mean()), "median_adv": float(np.median(adv)),
                        "mean_gain": float(1 - err[:, 1].mean() / err[:, 0].mean())},
        "context_high_change": {"frac_rollouts_shiftwm_better": float((adv[hi] > 0).mean()),
                                "median_adv": float(np.median(adv[hi]))},
        "picks": [{"kind": k, "traj": int(tr[w]), "start": int(rec["start"][w]), "adv": float(adv[w]),
                   "err_dinowm": float(err[w, 0]), "err_dinowm_shiftwm": float(err[w, 1]),
                   "pix_dinowm": rec["pix"][w][0], "pix_dinowm_shiftwm": rec["pix"][w][1]}
                  for k, w in zip(["best"] * len(best) + ["worst"] * len(worst), sel)],
        "sec": round(time.time() - t0, 1)}
    (out / f"dinowm_{a.env}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "picks"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
