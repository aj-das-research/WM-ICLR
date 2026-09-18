#!/usr/bin/env python3
"""Run an unmodified official AdaJEPA PushT evaluation with provenance.

This is an upstream reproduction: its pretrained representation, proprioceptive
inputs, released goal set, and planner budget differ from ShiftWM's experiments.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external/adajepa"
REVISION = "51d8665b7978824bd218decab9e05ddb6eb1f47b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["adaptive", "frozen"], required=True)
    parser.add_argument("--condition", choices=["clean", "blur", "dark"], required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        raise ValueError("This bounded reproduction accepts at most the official default of50episodes")
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError("Pinned upstream revision changed")
    checkpoint_root = ROOT / "data/baselines/adajepa/release/pusht_visual_shift"
    checkpoint = checkpoint_root / "checkpoints/model_latest.pth"
    goals = ROOT / "data/baselines/adajepa/release/pushobj_eval/val_T/plan_targets.pkl"
    output = (args.output or ROOT / "results/baselines/adajepa" /
              f"{args.condition}-{args.method}-seed{args.seed}").resolve()
    python = ROOT / "environments/adajepa/.venv/bin/python"
    command = [str(python), "plan.py", "--config-name", "adajepa_plan_cem_pushobj.yaml",
               f"ckpt_base_path={checkpoint_root}", f"eval_data_path={goals}",
               f"seed={args.seed}", f"n_evals={args.episodes}", "+wandb_logging=false",
               "decode_for_viz=false", f"hydra.run.dir={output}"]
    if args.method == "frozen":
        command.extend(["planner._target_=planning.mpc.MPCPlanner", "~planner.adapt"])
    if args.condition != "clean":
        command.append(f"ood_corruption={args.condition}")
    if args.print_command:
        print(json.dumps({"cwd": str(UPSTREAM), "command": command}, indent=2))
        return
    auxiliary = ROOT / "environments/adajepa/torch_cache/hub/dinov2_pinned_source.json"
    if not auxiliary.exists():
        raise RuntimeError("Run adajepa_pin_dinov2.py before reproduction")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Official full evaluation requires an allocated, working CUDA GPU")
    device = torch.cuda.get_device_properties(0)
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "reproduction_record.json"
    if status_file.exists():
        prior = json.loads(status_file.read_text())
        if prior.get("status") == "complete":
            raise RuntimeError("Refusing to overwrite a completed reproduction")
    record = {"status": "running", "method": args.method, "condition": args.condition,
              "seed": args.seed, "episodes": args.episodes, "official_default_episodes": 50,
              "upstream_revision": revision, "checkpoint_sha256": digest(checkpoint),
              "auxiliary_dinov2_source": json.loads(auxiliary.read_text()),
              "device": {"name": device.name, "total_memory_bytes": device.total_memory,
                         "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
              "launcher_sha256": digest(Path(__file__).resolve()),
              "requirements_sha256": digest(ROOT / "environments/adajepa/requirements.lock.txt"),
              "goals_sha256": digest(goals), "command": command, "cwd": str(UPSTREAM),
              "config_source_sha256": digest(UPSTREAM / "conf/adajepa_plan_cem_pushobj.yaml"),
              "model_input": "RGB plus agent proprioception", "planner": "official CEM",
              "planner_budget": {"horizon": 25, "samples": 200, "elites": 30,
                                 "optimization_iterations": 10, "max_mpc_replans": 20,
                                 "executed_grouped_actions_per_replan": 5, "frameskip": 5},
              "comparison_scope": "official upstream protocol; not input/budget/data matched to ShiftWM",
              "started_unix": time.time()}
    status_file.write_text(json.dumps(record, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update({"SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1",
                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1",
                        "TORCH_HOME": str(ROOT / "environments/adajepa/torch_cache")})
    with (output / "process.log").open("w") as log:
        result = subprocess.run(command, cwd=UPSTREAM, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record["return_code"] = result.returncode
    record["elapsed_seconds"] = time.time() - record["started_unix"]
    scientific_logs = output / "logs.json"
    final = None
    if scientific_logs.exists():
        for line in scientific_logs.read_text().splitlines():
            row = json.loads(line)
            if any(key.startswith("final_eval/") for key in row):
                final = row
    record["final_metrics"] = final
    record["status"] = "complete" if result.returncode == 0 and final is not None else "failed"
    status_file.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if record["status"] != "complete":
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
