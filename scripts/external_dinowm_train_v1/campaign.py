#!/usr/bin/env python3
"""Execute one registered full-budget arm under an exclusive run lock."""
import argparse
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import socket
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location("external_dinowm_campaign_" + name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def verify(require_review=True): return module("registry").verify(require_review)


def execution_identity(torch):
    """Runtime evidence only, captured within the actual scheduled allocation."""
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"visible_index": index, "name": props.name,
                            "total_memory_bytes": props.total_memory,
                            "compute_capability": [props.major, props.minor]})
    return {"hostname": socket.gethostname(), "torch_version": str(torch.__version__),
            "torch_cuda_build": torch.version.cuda, "visible_gpus": devices,
            "slurm": {key: os.environ.get(key) for key in
                      ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                       "SLURM_RESTART_COUNT", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
                       "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")}}


def run(name=None, index=None):
    registry = module("registry"); record = registry.verify()
    if index is not None:
        if type(index) is not int or not 0 <= index < 6: raise ValueError("Invalid run index")
        row = record["runs"][index]
    else:
        matches = [r for r in record["runs"] if r["name"] == name]
        if len(matches) != 1: raise ValueError("Unknown registered run")
        row = matches[0]
    name = row["name"]
    config = registry.read(ROOT / row["config"])
    config.update(resume_if_present=True, max_runtime_seconds=5400)
    directory = ROOT / config["output_dir"]; directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train = module("train")
        began = time.monotonic()
        execution = {"schema": "external_dinowm_training_execution_v1", "status": "started", "name": name,
                     "registration_sha256": registry.sha(registry.REG),
                     "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "identity": execution_identity(train.torch),
                     "scope": "Measured wall time includes input verification/loading and checkpoint IO; evaluation timed separately."}
        receipt = directory / ("execution_" + str(time.time_ns()) + ".json")
        train.atomic_json(execution, receipt)
        print(json.dumps({"execution_receipt": str(receipt.relative_to(ROOT)), **execution}), flush=True)
        try:
            result = train.train(config)
        except BaseException as error:
            execution.update(status="failed_training", ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             training_elapsed_seconds=time.monotonic() - began,
                             error_type=type(error).__name__, error=str(error))
            train.atomic_json(execution, receipt)
            raise
        execution["training_elapsed_seconds"] = time.monotonic() - began
        if result["status"] != "completed":
            execution.update(status="epoch_boundary_continuation", completed_epochs=result["completed_epochs"],
                             ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            train.atomic_json(execution, receipt)
            job = os.environ.get("SLURM_JOB_ID", "")
            if not job.isdigit() or int(os.environ.get("SLURM_RESTART_COUNT", "0")) >= 5:
                raise RuntimeError("Full training incomplete; continuation requires reviewed Slurm resumption")
            subprocess.run(["scontrol", "requeue", job], check=True)
            return result
        registry.verify()
        output = registry.REPORT / (name + "_validation.json")
        evaluation_began = time.monotonic()
        try:
            evaluation = module("evaluate").evaluate(config, output)
        except BaseException as error:
            execution.update(status="failed_evaluation", ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             evaluation_elapsed_seconds=time.monotonic() - evaluation_began,
                             error_type=type(error).__name__, error=str(error))
            train.atomic_json(execution, receipt)
            raise
        registry.verify()
        execution.update(status="completed", completed_epochs=30,
                         evaluation_elapsed_seconds=time.monotonic() - evaluation_began,
                         ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         total_elapsed_seconds=time.monotonic() - began)
        train.atomic_json(execution, receipt)
        completion = {"status": "completed", "name": name, "epochs": 30,
                      "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "registration_sha256": registry.sha(registry.REG),
                      "validation_sha256": registry.sha(output),
                      "training_summary_sha256": registry.sha(directory / "training_summary.json"),
                      "selected_epoch": evaluation["selected_epoch"],
                      "checkpoint_sha256": evaluation["checkpoint_sha256"],
                      "execution_receipt": str(receipt.relative_to(ROOT)),
                      "execution_receipt_sha256": registry.sha(receipt)}
        train.atomic_json(completion, registry.REPORT / (name + "_completed.json"))
        print(json.dumps(completion), flush=True)
        return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "verify", "run"))
    parser.add_argument("--name"); parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "register": module("registry").register()
    elif args.command == "verify": print(json.dumps({"status": "passed", "runs": len(verify()["runs"])}))
    else: run(args.name, args.index)
