"""Evaluate an existing best.pt (e.g. a run stopped early) on val/test exactly like train.run's final scoring."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from shiftwm.v2.models import V2WorldModel
from shiftwm.v2.train import FeatureSplit, evaluate, summary

p = argparse.ArgumentParser(); p.add_argument("run_dirs", nargs="+"); a = p.parse_args()
for rd in map(Path, a.run_dirs):
    cfg = json.loads((rd / "config.json").read_text()); root = Path(cfg["data"])
    stats = json.loads((root / "stats.json").read_text())
    st = torch.load(rd / "best.pt", map_location="cuda"); model = V2WorldModel(st["config"]).cuda().eval()
    model.load_state_dict(st["model"])
    res = {}
    for split in ("val", "test"):
        data = FeatureSplit(root, split, cfg["history"], cfg["horizon"], "cuda", stats, cfg.get("eval_stride", 2),
                            cfg.get("tasks"), single_image=cfg.get("single_image", False))
        ev = evaluate(model, data); np.savez(rd / f"eval_{split}.npz", **{k: np.asarray(v) for k, v in ev.items()})
        res[split] = summary(ev)
    out = {"event": "done", "step": st.get("step"), "note": "evaluated from best.pt by eval_ckpt.py (run stopped early)",
           "params": model.num_params(), "results": res}
    (rd / "summary.json").write_text(json.dumps(out, indent=1)); print(rd, {k: round(v["mse_mean_h"], 4) for k, v in res.items()})
