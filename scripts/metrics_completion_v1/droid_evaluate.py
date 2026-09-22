#!/usr/bin/env python3
"""One full registered checkpoint evaluation, with fixed-source MSE parity gates."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from droid_common import (ROOT, REPORT, REG, METRICS, POLICY, atomic_json, atomic_npz,
    local, module, now, prior_mse_check, read, relative, require, sha,
    validate_arrays, validate_output, verify_registration)
from shiftwm.real_video_spatial.data import SpatialDataset


def evaluate(index, device="cuda"):
    reg = verify_registration()
    require(type(index) is int and 0 <= index < len(reg["runs"]), "Invalid registered index")
    row = reg["runs"][index]
    destination = REPORT / "evaluations" / (row["name"] + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / (row["name"] + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            validate_output(row, reg)
            print("Already complete, verified: " + row["name"], flush=True)
            return
        ledger_path = destination.with_suffix(".npz")
        require(not ledger_path.exists(), "Uncommitted metric ledger exists; preserve it and investigate before rerun")
        started = time.monotonic()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(device in ("cuda", "cpu"), "Unsupported scoring device")
        if device == "cuda":
            require(torch.cuda.is_available(), "CUDA requested but no allocated GPU is visible")
        config = None; trainer = None; model = None
        if row["family"] != "persistence":
            config = read(local(row["config"]))
            trainer = module(row["trainer"], "droid_completion_eval_" + row["family"])
            trainer.base.seed_everything(row["seed"])
            trainer.validate_completed(local(config["output_dir"]))
            model, selected = trainer.load_package(local(row["checkpoint"]), device)
            require(selected["epoch"] == row["selected_epoch"], "Selected epoch changed")
            del selected
            model.eval()
        cache = ROOT / "data/features/droid_spatial_v1"
        dataset = SpatialDataset(cache, "val", horizon=10, stride=5, verify=True)
        stats = read(cache / "training_statistics.json")
        feature_std = torch.tensor(stats["feature_std"], dtype=torch.float32, device=device)
        if model is not None:
            torch.testing.assert_close(model.feature_std, feature_std, rtol=0, atol=0)
        metric_helper = module("scripts/real_video_iws/evaluate.py", "droid_completion_frozen_metrics")
        require(tuple(metric_helper.METRICS) == METRICS, "Frozen metric interface changed")
        collected = {key: [] for key in ("episode_id", "session_id", "window_start", *METRICS)}
        roundtrip = {"npz": "exact", "checkpoint_prediction": "not_applicable_deterministic_persistence"}
        with torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
            for batch_index, batch in enumerate(DataLoader(dataset, batch_size=row["batch_size"], shuffle=False, num_workers=0)):
                x = batch["features"].to(device); actions = batch["actions"].to(device)
                target = x[:, 3:]
                prediction = (x[:, 2:3].expand_as(target) if model is None else
                              model.predict(x[:, :3], actions[:, :2], actions[:, 2:]))
                if model is not None and batch_index == 0:
                    # The full first batch is reloaded through the official package loader.
                    # No model selection or synthetic-only substitute for the complete run.
                    restored, restored_state = trainer.load_package(local(row["checkpoint"]), device)
                    replay = restored.predict(x[:, :3], actions[:, :2], actions[:, 2:])
                    torch.testing.assert_close(prediction, replay, rtol=0, atol=0)
                    roundtrip["checkpoint_prediction"] = "exact"
                    del restored, restored_state, replay
                require(prediction.dtype == target.dtype == torch.float32, "Scoring tensors are not FP32")
                errors = metric_helper.feature_errors(prediction, target, feature_std)
                for key in METRICS:
                    collected[key].append(errors[key].cpu().numpy().astype(np.float64))
                for episode_index, start in zip(batch["episode_index"].tolist(), batch["window_start"].tolist()):
                    ep = dataset.episodes[episode_index]
                    collected["episode_id"].append(ep["episode_id"])
                    collected["session_id"].append(ep["session_id"])
                    collected["window_start"].append(start)
                print(f"{row['name']} batch {batch_index + 1}: complete registered windows {len(collected['window_start'])}/1631", flush=True)
        arrays = {key: np.concatenate(collected[key]) for key in METRICS}
        arrays.update(episode_id=np.asarray(collected["episode_id"]), session_id=np.asarray(collected["session_id"]),
                      window_start=np.asarray(collected["window_start"], dtype=np.int64))
        order = np.lexsort((arrays["window_start"], arrays["session_id"], arrays["episode_id"]))
        arrays = {key: value[order] for key, value in arrays.items()}
        episodes, summary = validate_arrays(arrays, reg["window_keys"])
        parity = prior_mse_check(arrays, episodes, summary, read(local(row["prior_ledger"])), model is None)
        # Source closure is checked before and after every complete inference run.
        require(verify_registration() == reg, "Registration changed while scoring")
        atomic_npz(arrays, ledger_path)
        result = {"schema": "droid_complementary_metrics_v1_evaluation", "status": "passed",
            "completed_utc": now(), "name": row["name"], "mode": row["mode"], "seed": row["seed"],
            "family": row["family"], "registration_sha256": sha(REG), "scope": POLICY["scope"],
            "selected_checkpoint_sha256": row.get("checkpoint_sha256"), "selected_epoch": row.get("selected_epoch"),
            "ledger_path": relative(ledger_path), "ledger_sha256": sha(ledger_path),
            "episodes": episodes, "summary": summary, "prior_mse_parity": parity, "roundtrip_checks": roundtrip,
            "execution": {"device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "torch_version": str(torch.__version__), "numpy_version": np.__version__, "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "batch_size": row["batch_size"], "elapsed_seconds": time.monotonic() - started,
                "tf32": False, "autocast": False}}
        atomic_json(result, destination)
        validate_output(row, reg)
        print("Complete and verified: " + row["name"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    evaluate(args.index, args.device)
