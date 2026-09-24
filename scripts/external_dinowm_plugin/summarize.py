"""Summarise DINO-WM vs DINO-WM+ShiftWM results for one env into results/v2/external/dinowm_plugin/<env>/summary.json.

    python scripts/external_dinowm_plugin/summarize.py pusht
"""
import glob
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("dinowm", "dinowm_shiftwm")
OPENLOOP_KEYS = ("z_visual_err_pred", "z_proprio_err_pred", "z_visual_err_persistence", "pred_img_lpips",
                 "pred_img_ssim", "pred_img_psnr")


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main(env):
    base = ROOT / "results/v2/external/dinowm_plugin" / env
    out = {"env": env, "arms": {}}
    for arm in ARMS:
        a = {}
        ol = _load(base / arm / "openloop.json")
        if ol:
            a["teacher_forced"] = {k: ol["teacher_forced"].get(k) for k in OPENLOOP_KEYS}
            a["rollout"] = {k: v for k, v in ol.items() if k.startswith("rollout_H")}
        mj = base / arm / "metrics.jsonl"
        if mj.exists():
            rows = [json.loads(l) for l in open(mj) if l.strip()]
            if rows:
                last = rows[-1]
                a["last_epoch_official_val"] = {k: v for k, v in last.items() if k.startswith("val_z_")
                                                or k in ("epoch", "val_loss", "train_loss") or "img_" in k and k.startswith("val_")}
        plans = {}
        for d in sorted(glob.glob(str(base / arm / "plan_*_seed*"))):
            fin = _load(os.path.join(d, "final.json"))
            if not fin:
                continue
            planner, seed = os.path.basename(d)[5:].rsplit("_seed", 1)
            sr = fin.get("final_eval/success_rate")
            plans.setdefault(planner, {})[seed] = sr
        for planner, by_seed in plans.items():
            v = [x for x in by_seed.values() if x is not None]
            m = sum(v) / len(v) if v else None
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) if len(v) > 1 else None
            plans[planner] = {"per_seed": by_seed, "mean_success": m, "sd_over_seeds": sd, "n_seeds": len(v)}
        a["planning"] = plans
        out["arms"][arm] = a
    with open(base / "summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
