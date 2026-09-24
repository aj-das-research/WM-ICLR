"""Launch the OFFICIAL DINO-WM train.py (external/dino_wm @ 0a9492fa, unmodified) with three wrapper-level shims.

Usage (from anywhere; Hydra overrides are passed through verbatim):
    python scripts/external_dinowm_plugin/run_train.py --config-name train.yaml env=pusht \
        ckpt_base_path=/abs/runs/dinowm_plugin hydra.run.dir=/abs/runs/dinowm_plugin/outputs/pusht_dinowm \
        training.epochs=E [predictor._target_=shiftwm.v2.dinowm_plugin.ShiftViTPredictor]

Shims (none changes the model, loss, optimiser, data or evaluation arithmetic):
  1. `training.epochs` is the ABSOLUTE number of epochs: after the official auto-resume from
     checkpoints/model_latest.pth, only the remaining epochs are run (upstream would run `epochs` more).
     Note (official behaviour, kept): on resume the optimisers are re-created, i.e. Adam moments reset.
  2. Every epoch's full log dict (everything upstream sends to wandb) is appended to metrics.jsonl in the run dir.
  3. CPU only: models.vit builds its causal mask with `.to('cuda')`; on hosts without CUDA it stays on CPU.
  Optional (DINOWM_FAST_SLICES=1): PushT slice loader decodes only the frames it returns (output-identical,
  see tests/test_dinowm_plugin.py::test_fast_slices_identical).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DWM = ROOT / "external" / "dino_wm"
sys.path[:0] = [str(DWM), str(ROOT / "src"), str(Path(__file__).resolve().parent)]
os.environ.setdefault("WANDB_MODE", "offline")

import torch  # noqa: E402

import shims  # noqa: E402

shims.install_cpu_mask_shim()
if os.environ.get("DINOWM_FAST_SLICES") == "1":
    shims.install_fast_slices()

import train as dwm_train  # noqa: E402  (official module)

_orig_init = dwm_train.Trainer.__init__
_orig_flash = dwm_train.Trainer.logs_flash


def _init(self, cfg):
    _orig_init(self, cfg)
    target = int(cfg.training.epochs)
    self.total_epochs = max(0, target - int(self.epoch))
    dwm_train.log.info(f"[wrapper] absolute epochs={target}; resumed at epoch {self.epoch}; running {self.total_epochs}")
    if self.accelerator.is_main_process:
        n_train = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        with open("wrapper_info.json", "w") as f:
            json.dump({"predictor_class": type(self.accelerator.unwrap_model(self.predictor)).__name__,
                       "predictor_trainable_params": n_train,
                       "train_slices": len(self.datasets["train"]), "val_slices": len(self.datasets["valid"]),
                       "resumed_epoch": int(self.epoch), "target_epochs": target}, f, indent=1)


def _flash(self, step):
    if self.accelerator.is_main_process:
        row = {k: float(s / c) for k, (c, s) in self.epoch_log.items()}
        row["epoch"] = step
        with open("metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    return _orig_flash(self, step)


dwm_train.Trainer.__init__ = _init
dwm_train.Trainer.logs_flash = _flash

if __name__ == "__main__":
    import hydra

    @hydra.main(config_path=str(DWM / "conf"), config_name="train")
    def main(cfg):  # identical body to upstream train.main (re-declared so hydra resolves the absolute conf dir)
        trainer = dwm_train.Trainer(cfg)
        trainer.run()

    main()
