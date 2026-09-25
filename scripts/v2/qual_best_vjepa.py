"""Largest-advantage qualitative windows for the V-JEPA 2-AC plug-in on DROID (appendix qual_best_vjepa2ac.pdf).

Selection rule (same as scripts/v2/qual_select.py, with the plug-in's own baseline): over ALL test windows (stride 2)
of results/v2/external/vjepa2ac_plugin/{finetune,finetune_shiftwm}/s0/test.npz (the stored per-window MSE of the
evaluator, V-JEPA's own layer-normed ViT-g token space), at k=10 compute the gain 1 - err_{+head} / err_{no head}; among
windows in the top 50% of true change ||z_{t+10}-z_t||^2 (= stored persistence MSE) keep the top 3 from distinct
episodes. Both checkpoints (data/v2/runs/vjepa2ac_plugin/<arm>/s0/best.pt) are then re-run on those windows with the
evaluator's rollout to obtain the token forecasts (the recomputed MSE is checked against the stored one).

Output: results/v2/analysis/qual_best/vjepa2ac_droid.{npz,json}
  tokens [n, 4, 256, 1408] fp16 (observed z_t, true z_{t+10}, no-head forecast, +head forecast), perpatch [n,2,16,16],
  err [n,2] (finetune, finetune_shiftwm), frame_obs / frame_true (native camera frames, uint8, full 180x320),
  plus the per-window arrays over all test windows (err_all, change_all, adv_all).
Usage (GPU): PYTHONPATH=src python scripts/v2/qual_best_vjepa.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import vjepa_finetune as vf  # noqa: E402
from shiftwm.v2.analysis import ANALYSIS, FrameSource, cache_root, write_json  # noqa: E402

RES = ROOT / "results/v2/external/vjepa2ac_plugin"
CKPT = ROOT / "data/v2/runs/vjepa2ac_plugin"
CACHE = ROOT / "data/v2/features/droid/vjepa2g"
OUT = ANALYSIS / "qual_best"
ARMS = ("finetune", "finetune_shiftwm")
K_SHOW, N_PICK, SEED, H, K = 10, 3, 0, 3, 10
RULE = ("All test windows (stride 2) of the V-JEPA 2-AC evaluator; per-window k=10 token MSE of the fine-tuned "
        "predictor with and without the ShiftWM head (seed 0) and true change ||z_{t+10}-z_t||^2. Among windows in the "
        "top 50% of true change, rank by gain 1 - err_{+head}/err_{no head}; keep the top 3 from distinct episodes.")


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = {arm: np.load(RES / arm / f"s{SEED}/test.npz") for arm in ARMS}
    for key in ("win_episode", "win_start"):
        assert np.array_equal(z[ARMS[0]][key], z[ARMS[1]][key])
    j = K_SHOW - 1
    err = np.stack([z[arm]["win_model_mse"][:, j] for arm in ARMS], 1).astype(np.float64)
    change = z[ARMS[0]]["win_persistence_mse"][:, j].astype(np.float64)
    adv = 1 - err[:, 1] / np.maximum(err[:, 0], 1e-12)
    w_ep, w_st = z[ARMS[0]]["win_episode"], z[ARMS[0]]["win_start"]
    hi = change >= np.median(change)
    order = np.where(hi)[0][np.argsort(-adv[hi], kind="stable")]
    picks, seen = [], set()
    for w in order:
        if w_ep[w] not in seen:
            picks.append(int(w)); seen.add(w_ep[w])
        if len(picks) == N_PICK:
            break
    rows = [r for r in json.loads((CACHE / "index.json").read_text())["episodes"] if r["split"] == "test"]
    tok = np.load(CACHE / "tokens.npy", mmap_mode="r")
    S_all, A_all = np.load(CACHE / "states.npy", mmap_mode="r"), np.load(CACHE / "actions.npy", mmap_mode="r")
    glob = np.array([[rows[w_ep[w]]["offset"] + w_st[w] + t for t in range(H + K)] for w in picks])
    Z = torch.from_numpy(np.array(tok[glob.ravel()])).float().reshape(len(picks), H + K, *tok.shape[1:]).to(dev)
    S = torch.from_numpy(np.array(S_all[glob[:, :-1].ravel()])).float().reshape(len(picks), H + K - 1, -1).to(dev)
    A = torch.from_numpy(np.array(A_all[glob[:, :-1].ravel()])).float().reshape(len(picks), H + K - 1, -1).to(dev)
    target = Z[:, H - 1 + K_SHOW]
    preds, perpatch, err_re = [], [], []
    for arm in ARMS:
        a = vf.build_parser().parse_args(["--arm", arm, "--seed", str(SEED)])
        wm = vf.build(a, dev)
        sd = torch.load(CKPT / arm / f"s{SEED}/best.pt", map_location=dev, weights_only=False)
        wm.load_state_dict(sd["model"]); wm.eval(); wm.predictor.use_activation_checkpointing = False
        with torch.no_grad():
            p = wm.rollout(Z[:, :H], S, A, K, a.max_context)[:, j]
        e = ((p - target) ** 2).mean(-1)
        preds.append(p); perpatch.append(e.reshape(len(picks), 16, 16)); err_re.append(e.mean(-1))
        del wm, sd
        torch.cuda.empty_cache()
    err_re = torch.stack(err_re, 1).double().cpu().numpy()
    print("stored err", err[picks].round(4).tolist(), "recomputed", err_re.round(4).tolist(), flush=True)
    frames = FrameSource(cache_root("droid"))
    eps = [rows[w_ep[w]]["id"] for w in picks]
    t_obs = [int(w_st[w] + H - 1) for w in picks]
    obs = np.stack([frames.load_native(e, [t])[0] for e, t in zip(eps, t_obs)])
    true = np.stack([frames.load_native(e, [t + K_SHOW])[0] for e, t in zip(eps, t_obs)])
    tokens = torch.stack([Z[:, H - 1], target, preds[0], preds[1]], 1).half().cpu().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "vjepa2ac_droid.npz", episode=np.array(eps), t0=np.array(t_obs), window=np.array(picks),
                        adv=adv[picks], err=err[picks], err_recomputed=err_re, arms=np.array(ARMS),
                        change=change[picks], perpatch=torch.stack(perpatch, 1).float().cpu().numpy(),
                        tokens=tokens, frame_obs=obs, frame_true=true, k=K_SHOW,
                        crop_box_ijhw=np.array([0, 38, 180, 243]), err_all=err, change_all=change, adv_all=adv)
    write_json(OUT / "vjepa2ac_droid.json", {
        "status": "done", "rule": RULE, "k": K_SHOW, "stride": 2, "history": H, "horizon": K,
        "checkpoints": {arm: str((CKPT / arm / f"s{SEED}/best.pt").relative_to(ROOT)) for arm in ARMS},
        "n_windows": int(len(err)), "change_median": float(np.median(change)),
        "picks": [{"episode": e, "t0": t, "adv": float(adv[w]), "change": float(change[w]),
                   "err_finetune": float(err[w, 0]), "err_finetune_shiftwm": float(err[w, 1])}
                  for e, t, w in zip(eps, t_obs, picks)],
        "context_high_change": {"frac_windows_shiftwm_best": float((adv[hi] > 0).mean()),
                                "median_adv": float(np.median(adv[hi])),
                                "mean_err": dict(zip(ARMS, err[hi].mean(0).tolist()))},
        "context_all": {"frac_windows_shiftwm_best": float((adv > 0).mean()), "median_adv": float(np.median(adv)),
                        "mean_err": dict(zip(ARMS, err.mean(0).tolist()))},
        "sec": round(time.time() - t0, 1)})
    print("done", OUT / "vjepa2ac_droid.npz", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
