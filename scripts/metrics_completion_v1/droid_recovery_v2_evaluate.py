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

from droid_recovery_v2_common import (ROOT, REPORT, REG, METRICS, POLICY, atomic_json, atomic_npz,
    local, module, now, prior_mse_check, read, relative, require, sha,
    validate_arrays, validate_output, verify_registration)
from shiftwm.real_video_spatial.data import SpatialDataset


def repeat_difference(first, second, target, feature_std):
    require(first.shape == second.shape == target.shape and torch.isfinite(first).all()
            and torch.isfinite(second).all(), "Invalid GPU reproducibility outputs")
    difference = (first - second).abs()
    first_mse = ((first-target)/feature_std).square().mean(-1)
    second_mse = ((second-target)/feature_std).square().mean(-1)
    return {"exact": bool(torch.equal(first, second)), "differing_elements": int(torch.count_nonzero(difference)),
            "max_absolute_prediction_difference": float(difference.max()),
            "max_absolute_window_mse_difference": float((first_mse-second_mse).abs().max())}


def exact_state_dict(first, second):
    require(set(first) == set(second), "Checkpoint state-dict keys differ")
    for key in first:
        torch.testing.assert_close(first[key].detach().cpu(), second[key].detach().cpu(), rtol=0, atol=0)


def check_reload(trainer, row, model, x, actions, prediction, target, feature_std):
    require(not any(m.training for m in model.modules()), "Scored model is not fully eval mode")
    repeated = model.predict(x[:, :3], actions[:, :2], actions[:, 2:])
    restored, restored_state = trainer.load_package(local(row["checkpoint"]), x.device)
    require(not any(m.training for m in restored.modules()), "Reloaded model is not fully eval mode")
    exact_state_dict(model.state_dict(), restored.state_dict())
    replay = restored.predict(x[:, :3], actions[:, :2], actions[:, 2:])
    diagnostic = {"equality_is_acceptance_gate": False, "prediction_elements": prediction.numel(),
                  "same_model": repeat_difference(prediction, repeated, target, feature_std),
                  "reloaded_model": repeat_difference(prediction, replay, target, feature_std),
                  "both_models_eval": True, "state_dict_equal": True}
    del restored, restored_state, repeated, replay
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first, first_state = trainer.load_package(local(row["checkpoint"]), "cpu")
        second, second_state = trainer.load_package(local(row["checkpoint"]), "cpu")
        require(not any(m.training for m in first.modules()) and not any(m.training for m in second.modules()),
                "CPU round-trip model is not fully eval mode")
        exact_state_dict(model.state_dict(), first.state_dict())
        exact_state_dict(first.state_dict(), second.state_dict())
        support = x[:2, :3].cpu().contiguous()
        past, future = actions[:2, :2].cpu().contiguous(), actions[:2, 2:].cpu().contiguous()
        expected = first.predict(support, past, future)
        actual = second.predict(support, past, future)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        del first, second, first_state, second_state, expected, actual
    finally:
        torch.set_num_threads(threads)
    return {"checkpoint_prediction": "exact", "checkpoint_prediction_device": "cpu",
            "checkpoint_prediction_windows": 2, "checkpoint_state_dict": "exact",
            "gpu_reproducibility_diagnostic": diagnostic}


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
                    roundtrip.update(check_reload(trainer, row, model, x, actions, prediction, target, feature_std))
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
        result = {"schema": "droid_complementary_metrics_recovery_v2_evaluation", "status": "passed",
            "recovery_original_registration_sha256": reg["recovery"]["original_registration_sha256"],
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
