#!/usr/bin/env python3
"""Separate ablation receipts using unchanged complete v1 CPU metric code."""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


frozen = load(ROOT / "scripts/real_video_iws/evaluate.py", "_unbounded_private_frozen_evaluator")
campaign = load(Path(__file__).with_name("campaign.py"), "_unbounded_evaluation_campaign")
frozen.campaign = campaign


def evaluate(config_path, name, output):
    output = Path(output).resolve()
    expected = campaign.REPORT / "evaluations" / (name + ".json")
    if output != expected:
        raise ValueError("Ablation evaluation must use its distinct canonical namespace")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry = campaign.check_registration(config_path)
        row = next((r for r in registry["runs"] if r["name"] == name),None)
        if row is None:raise ValueError("Unknown ablation run")
        attempt = output.with_suffix(".attempt.json")
        invocation = {"name":name,"device":"cpu","adapter_sha256":campaign.sha(__file__),
                      "frozen_evaluator_sha256":campaign.sha(frozen.__file__),
                      "config_sha256":campaign.sha(config_path),"registration_sha256":campaign.sha(campaign.REGISTRATION)}
        if attempt.exists():
            if campaign.read(attempt) != invocation:raise ValueError("Different previous evaluator invocation")
        elif output.exists():raise ValueError("Existing result has no registered CPU invocation proof")
        else:campaign.atomic_json(invocation,attempt)
        if not output.exists():frozen.evaluate(config_path,name,output,device="cpu")
        # Recover only a complete exact CPU result after interruption between the
        # frozen JSON commit and our adapter commit; never accept JSON alone.
        validator = load(ROOT/"scripts/real_video_iws/finalize.py", "_unbounded_cpu_result_validator")
        train = campaign.trainer(); directory = ROOT/row["output"]
        summary = train.validate_completed(directory); package,state = train.read_package(directory/"best")
        config = campaign.read(config_path)
        recipe = {"task":row["task"],"mode":row["mode"],"seed":row["seed"],"model":config["model"],
                  "training":config["training"],"task_config":config["tasks"][row["task"]],"study_config_sha256":campaign.sha(config_path)}
        if state["config"]["metadata"]["identity"]["scientific_config"] != recipe:
            raise ValueError("Selected package does not belong to this registered ablation")
        audit = validator.metadata_audit(train.open_cache(config,row["task"]))
        validator.validate_evaluation(output,row,summary,campaign.sha(package/"model.pt"),audit,
                                      campaign.sha(config_path),campaign.sha(campaign.REGISTRATION),{})
        adapter = {"status":"passed", "device":"cpu", "precision":"float32",
                   "scope":"exploratory_internal_development_after_v1",
                   "adapter_sha256":campaign.sha(__file__),"attempt_sha256":campaign.sha(attempt),
                   "registration_sha256":campaign.sha(campaign.REGISTRATION),
                   "evaluation_sha256":campaign.sha(output),"official_validation_payloads_read":0}
        target = output.with_suffix(".adapter.json")
        if target.exists():
            if campaign.read(target) != adapter:raise ValueError("Adapter evidence changed")
        else:campaign.atomic_json(adapter,target)
        return {"status":"passed","run":name,"device":"cpu","evaluation_sha256":campaign.sha(output)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, default=campaign.CONFIG)
    p.add_argument("--name", required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); print(campaign.json.dumps(evaluate(a.config, a.name, a.output), sort_keys=True))
