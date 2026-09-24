"""Launch the OFFICIAL DINO-WM plan.py (external/dino_wm @ 0a9492fa, unmodified); Hydra overrides pass through.

    python scripts/external_dinowm_plugin/run_plan.py --config-name plan_pusht.yaml \
        ckpt_base_path=/abs/runs/dinowm_plugin model_name=pusht_dinowm hydra.run.dir=/abs/.../plan_seed99

Shims: CPU causal-mask shim (no-op on GPU); env.pointmaze is stubbed unless DINOWM_ENV=point_maze (the upstream
env package imports mujoco_py/d4rl unconditionally; PushT and Wall do not use them).  Results: logs.json in run dir.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DWM = ROOT / "external" / "dino_wm"
sys.path[:0] = [str(DWM), str(ROOT / "src"), str(Path(__file__).resolve().parent)]
os.environ.setdefault("WANDB_MODE", "offline")
os.chdir(DWM)  # env assets (e.g. wall) are resolved relative to the repo; hydra.run.dir must be absolute

import shims  # noqa: E402

shims.install_cpu_mask_shim()
shims.install_preprocessor_5d_shim()
shims.install_chunked_decode_shim()
if os.environ.get("DINOWM_ENV") != "point_maze":
    shims.install_pointmaze_stub()

import plan as dwm_plan  # noqa: E402

if __name__ == "__main__":
    import hydra
    from omegaconf import open_dict

    @hydra.main(config_path=str(DWM / "conf"), config_name="plan")
    def main(cfg):  # identical body to upstream plan.main (re-declared so hydra resolves the absolute conf dir)
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            dwm_plan.log.info(f"Planning result saved dir: {cfg['saved_folder']}")
        cfg_dict = dwm_plan.cfg_to_dict(cfg)
        cfg_dict["wandb_logging"] = True
        dwm_plan.planning_main(cfg_dict)

    main()
