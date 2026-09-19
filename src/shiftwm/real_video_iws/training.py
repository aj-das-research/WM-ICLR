"""IWS full-horizon optimization and audited epoch-boundary continuation.

The existing DROID atomic package writer/reader and RNG recovery are imported
privately. Its data interface, objective, selector and fit loop are not reused.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_KIND = "shiftwm_iws_single_observation_v1"
SELECTION = "internal_dev_equal_trajectory_endpoint_H60_standardized_mse"
PRECISION = "float32 validation; autocast disabled; CUDA TF32 disabled"


def _private_trainer():
    spec = importlib.util.spec_from_file_location("_iws_atomic_droid_trainer", ROOT / "scripts/real_video/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PACKAGE_KIND = PACKAGE_KIND
    module.SELECTION = SELECTION
    def construct(config):
        from .model import from_config
        return from_config(config)
    module.from_config = construct
    return module


base = _private_trainer()
atomic_json = base.atomic_json
digest = base.digest
seed_everything = base.seed_everything
save_package = base.save_package


def read_package(directory, require_training=False):
    path, state = base.read_package(directory, require_training)
    metadata = state["config"].get("metadata", {})
    if metadata.get("selection") != SELECTION or metadata.get("validation_precision") != PRECISION:
        raise ValueError("Wrong IWS selection/precision contract")
    if metadata.get("training_identity") != digest(metadata.get("identity")):
        raise ValueError("IWS training identity is corrupt")
    return path, state


def load_package(directory, device="cpu"):
    read_package(directory)
    return base.load_package(directory, device)


def _microbatches(batch, size):
    count = len(batch["initial_features"])
    for start in range(0, count, size):
        yield {key: value[start:start + size] if torch.is_tensor(value) else value
               for key, value in batch.items()}


def epoch_pass(model, loader, device, *, optimizer=None, microbatch_size=16,
               bf16=True, grad_clip=1.0):
    """One optimizer update per loader batch, including its actual-size tail.

Training averages every window/offset/coordinate. Validation is FP32 and selects
only equal-trajectory endpoint error; all-offset and other horizons are named
diagnostics. Targets enter the loss only, never the prediction API.
"""
    if type(microbatch_size) is not int or microbatch_size < 1:
        raise ValueError("Invalid microbatch size")
    training = optimizer is not None
    model.train(training)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total, windows, batches, microbatches = 0.0, 0, 0, 0
    population = {}
    feature_dim = model.config.feature_dim
    with (nullcontext() if training else torch.inference_mode()):
        for batch in loader:
            n = len(batch["initial_features"])
            if n < 1 or batch["initial_features"].shape[1:] != (feature_dim,):
                raise ValueError("Wrong single-observation batch")
            if batch["commands"].shape[1:] != (60, model.config.action_dim) or batch["targets"].shape[1:] != (59, feature_dim):
                raise ValueError("IWS training always requires full H60 and all59 targets")
            if training:
                optimizer.zero_grad(set_to_none=True)
            for small in _microbatches(batch, microbatch_size):
                small = {k: v.to(device) if torch.is_tensor(v) else v for k, v in small.items()}
                m = len(small["initial_features"])
                with torch.autocast(device_type=device.type, enabled=training and bf16 and device.type == "cuda", dtype=torch.bfloat16):
                    predictions = model.predict(small["initial_features"], small["commands"])
                    if predictions.shape != small["targets"].shape:
                        raise ValueError("Predictor output shape differs from full-horizon targets")
                    errors = ((predictions.float() - small["targets"].float()) / model.feature_std.float()).square()
                    loss = errors.mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite IWS loss")
                if training:
                    # Actual effective-batch denominator, not nominal accumulation.
                    (loss * (m / n)).backward()
                else:
                    values = errors.mean(-1).double().cpu().numpy()
                    for eid, row in zip(small["episode_index"].tolist(), values):
                        accumulator, count = population.get(eid, (np.zeros(59, dtype=np.float64), 0))
                        population[eid] = (accumulator + row, count + 1)
                total += float(loss.detach()) * m
                windows += m
                microbatches += 1
            if training:
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite IWS gradients")
                optimizer.step()
            batches += 1
    if windows == 0:
        raise ValueError("Empty IWS population")
    result = {"standardized_mse": total / windows, "window_mean_all59_standardized_mse": total / windows,
              "elements": windows * 59 * feature_dim, "windows": windows, "batches": batches,
              "optimizer_updates": batches if training else 0, "microbatches": microbatches,
              "query_offsets": 59, "horizon": 60,
              "aggregation": "equal_window_all59" if training else SELECTION}
    if not training:
        episode_rows = [{"episode_index": int(eid), "windows": count,
                         "standardized_mse_by_offset": (summed / count).tolist()}
                        for eid, (summed, count) in sorted(population.items())]
        curve = np.array([row["standardized_mse_by_offset"] for row in episode_rows]).mean(0)
        result.update(standardized_mse=float(curve[-1]), episodes=len(episode_rows),
                      episode_metrics=episode_rows, equal_trajectory_mse_by_offset=curve.tolist(),
                      equal_trajectory_mean_all59=float(curve.mean()),
                      diagnostic_endpoints={str(h): float(curve[h-2]) for h in (15, 30, 45, 60)})
    return result


def _metadata(model, identity):
    return {"training_identity": digest(identity), "identity": deepcopy(identity),
            "selection": SELECTION, "validation_precision": PRECISION,
            "parameter_counts": {"total": sum(p.numel() for p in model.parameters()),
                                 "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}}


def verify_model_contract(package_config, identity, model=None):
    recipe = identity["scientific_config"]
    expected = {**recipe["model"], "mode": recipe["mode"],
                "action_dim": recipe["task_config"]["action_dim"]}
    if package_config.get("model_config") != expected:
        raise ValueError("Instantiated model differs from registered architecture")
    for key in ("feature_mean", "feature_std", "command_mean", "command_std"):
        value = torch.as_tensor(identity["normalization"][key], dtype=torch.float32)
        if not torch.equal(torch.as_tensor(package_config.get(key), dtype=torch.float32), value):
            raise ValueError("Package normalization differs from frozen training statistics: " + key)
        if model is not None and not torch.equal(getattr(model,key).detach().cpu(),value):
            raise ValueError("Executed normalization differs from frozen statistics: " + key)


def _repair_best_after_interruption(output, state):
    """A best swap may precede an interrupted last swap; last is authoritative."""
    best = output / "best"
    history = state["history"]
    if not history:
        if best.is_symlink():
            best.unlink()
        elif best.exists():
            raise ValueError("Unexpected non-generation best package")
        return
    selected = min(history, key=lambda row: row["val"]["standardized_mse"])
    candidates = [best] + sorted((output / ".best.generations").glob("*"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            _, value = read_package(candidate)
        except (ValueError, OSError):
            continue
        if (value["epoch"] == selected["epoch"] and value["config"] == state["config"]
                and value["best_metric"] == selected["val"]["standardized_mse"]
                and value["history"] == history[:selected["epoch"]]):
            if candidate != best:
                temporary = output / (".best.recovery-" + uuid.uuid4().hex)
                os.symlink(os.path.relpath(candidate.resolve(), output), temporary)
                os.replace(temporary, best)
            return
    raise ValueError("Authoritative last history has no recoverable selected generation")


def validate_completed(directory, expected_identity=None, verify_dependencies=True):
    directory = Path(directory)
    config = json.loads((directory / "training_config.json").read_text())
    summary = json.loads((directory / "training_summary.json").read_text())
    _, last = read_package(directory / "last", True)
    _, best = read_package(directory / "best")
    metadata = last["config"]["metadata"]
    identity = metadata["identity"]
    verify_model_contract(last["config"], identity)
    if identity["scientific_config"] != config["scientific_config"]:
        raise ValueError("Executed recipe differs from the frozen run recipe")
    if expected_identity is not None and digest(identity) != expected_identity:
        raise ValueError("Wrong completed training identity")
    if verify_dependencies:
        base.verify_sources(identity["dependencies"])
    rows = base.metric_rows(directory)
    if rows != last["history"] or [r["epoch"] for r in rows] != list(range(1, 31)):
        raise ValueError("Full30-epoch journal is incomplete or altered")
    if config["scientific_config"]["training"]["epochs"] != 30 or last["epoch"] != 30:
        raise ValueError("Full30 epochs required")
    training = config["scientific_config"]["training"]
    batch_size, microbatch = training["batch_size"], training["microbatch_size"]
    feature_dim = config["scientific_config"]["model"]["feature_dim"]
    expected_val = {r["episode_index"]: r["windows"] for r in identity["populations"]["val"]["records"] if r["windows"] > 0}
    expected_step = 0
    for row in rows:
        for part in ("train", "val"):
            metric = row[part]
            count = identity["populations"][part]["windows"]
            batches = math.ceil(count/batch_size)
            if (metric["windows"] != count or metric["query_offsets"] != 59 or metric["horizon"] != 60
                    or metric["batches"] != batches or metric["microbatches"] != math.ceil(count/microbatch)
                    or metric["optimizer_updates"] != (batches if part=="train" else 0)
                    or metric["elements"] != count*59*feature_dim
                    or metric["aggregation"] != ("equal_window_all59" if part=="train" else SELECTION)):
                raise ValueError("Incomplete population or horizon")
            if not math.isfinite(metric["standardized_mse"]) or metric["standardized_mse"] < 0:
                raise ValueError("Invalid IWS metric")
        val = row["val"]
        episodes = val["episode_metrics"]
        if (len({r["episode_index"] for r in episodes}) != len(episodes)
                or sum(r["windows"] for r in episodes) != val["windows"]
                or {r["episode_index"]:r["windows"] for r in episodes} != expected_val
                or val["episodes"] != len(expected_val)):
            raise ValueError("Incomplete validation episode ledger")
        curve = np.array([r["standardized_mse_by_offset"] for r in episodes])
        if curve.shape[1:] != (59,) or not np.isfinite(curve).all() or (curve < 0).any():
            raise ValueError("Invalid endpoint ledger")
        if not np.array_equal(curve.mean(0), val["equal_trajectory_mse_by_offset"]) or val["standardized_mse"] != float(curve.mean(0)[-1]):
            raise ValueError("Equal-trajectory selection arithmetic differs")
        expected_step += row["train"]["optimizer_updates"]
        if row["step"] != expected_step:
            raise ValueError("Optimizer-update journal differs")
    selected = min(rows, key=lambda r: r["val"]["standardized_mse"])
    steps = sum(r["train"]["optimizer_updates"] for r in rows)
    if (best["epoch"] != selected["epoch"] or best["best_metric"] != selected["val"]["standardized_mse"]
            or best["config"] != last["config"] or best["history"] != rows[:best["epoch"]]
            or last["best_metric"] != best["best_metric"] or last["step"] != steps
            or best["step"] != sum(r["train"]["optimizer_updates"] for r in rows[:best["epoch"]])):
        raise ValueError("Selected package is not the earliest declared minimum")
    expected = {"status": "completed", "completed_epochs": 30, "step": steps,
                "training_identity": digest(identity), "best_epoch": best["epoch"],
                "best_validation_mse": best["best_metric"], "validation_metric": SELECTION,
                "validation_precision": PRECISION, "parameter_counts": metadata["parameter_counts"]}
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError("Completion summary differs from authoritative packages")
    return summary


def export_inference_package(run_directory, destination):
    """Export only fully completed selected tensors/config, with no optimizer/RNG."""
    run_directory, destination = Path(run_directory), Path(destination)
    validate_completed(run_directory)
    source, state = read_package(run_directory / "best")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to replace an existing inference export")
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_name("."+destination.name+".pending-"+uuid.uuid4().hex)
    temporary.mkdir()
    try:
        for name in ("model.pt","config.json"):
            shutil.copyfile(source/name,temporary/name)
        atomic_json({"format_version":1,"package_kind":PACKAGE_KIND,
                     "files":{name:base.sha256(temporary/name) for name in ("model.pt","config.json")}},
                    temporary/"package_manifest.json")
        read_package(temporary)
        os.rename(temporary,destination)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return {"status":"inference_exported","package_kind":PACKAGE_KIND,"selected_epoch":state["epoch"],
            "manifest_sha256":base.sha256(destination/"package_manifest.json"),"optimizer_or_rng_included":False}


def fit(model, config, train_dataset, val_dataset, identity, verify_registration=lambda: None):
    """Epoch boundaries are atomic; interruption replays the uncommitted epoch."""
    recipe = config["scientific_config"]
    tr = recipe["training"]
    if recipe != identity["scientific_config"] or tr["epochs"] != 30 or recipe["seed"] not in (0,1,2):
        raise ValueError("Invalid registered IWS run")
    if model.config.mode != recipe["mode"]:
        raise ValueError("Model arm and recipe differ")
    verify_model_contract(model.package_config, identity, model)
    if tr["batch_size"] != tr["microbatch_size"] * tr["accumulation_steps"]:
        raise ValueError("Effective batch disagrees with accumulation")
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cuda"))
    seed_everything(recipe["seed"]); model.to(device)
    metadata = _metadata(model, identity)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tr["lr"], weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=tr["min_lr"])
    generator = torch.Generator().manual_seed(recipe["seed"])
    kw = dict(batch_size=tr["batch_size"], num_workers=0, pin_memory=device.type=="cuda", drop_last=False)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **kw)
    val_loader = DataLoader(val_dataset, shuffle=False, generator=torch.Generator().manual_seed(173), **kw)
    history, best, step, start = [], float("inf"), 0, 0
    if (output / "last").exists():
        if not config.get("resume_if_present"):
            raise ValueError("Existing run requires --resume-if-present")
        state = base.resume_training(output / "last", model, optimizer, scheduler, generator, metadata)
        _repair_best_after_interruption(output, state)
        history, best, step, start = state["history"], state["best_metric"], state["step"], state["epoch"]
        if [r["epoch"] for r in history] != list(range(1,start+1)):
            raise ValueError("Interrupted journal is inconsistent")
        base.write_history(history, output)
    elif any((output / p).exists() for p in ("best","training_summary.json","metrics.jsonl")):
        raise ValueError("Partial output lacks an authoritative initial checkpoint")
    else:
        save_package(model, output / "last", optimizer=optimizer, scheduler=scheduler, epoch=0,
                     step=0, best_metric=best, metadata=metadata, generator=generator, history=[])
    atomic_json(config, output / "training_config.json")
    began = time.monotonic()
    for epoch in range(start+1,31):
        verify_registration(); base.verify_sources(identity["dependencies"])
        train = epoch_pass(model, train_loader, device, optimizer=optimizer, microbatch_size=tr["microbatch_size"], bf16=tr["bf16"], grad_clip=tr["grad_clip"])
        val = epoch_pass(model, val_loader, device, microbatch_size=tr["microbatch_size"], bf16=False)
        step += train["optimizer_updates"]
        history.append({"epoch":epoch,"step":step,"lr":optimizer.param_groups[0]["lr"],"train":train,"val":val})
        scheduler.step()
        improved = val["standardized_mse"] < best
        best = min(best,val["standardized_mse"])
        args=dict(optimizer=optimizer,scheduler=scheduler,epoch=epoch,step=step,best_metric=best,metadata=metadata,generator=generator,history=history)
        if improved: save_package(model,output/"best",**args)
        save_package(model,output/"last",**args)
        base.write_history(history,output)
        summary={"status":"completed" if epoch==30 else "interrupted","completed_epochs":epoch,"step":step,
                 "training_identity":digest(identity),"best_epoch":min(history,key=lambda r:r["val"]["standardized_mse"])["epoch"],
                 "best_validation_mse":best,"validation_metric":SELECTION,"validation_precision":PRECISION,
                 "parameter_counts":metadata["parameter_counts"],"resume_semantics":"atomic completed epochs; partial epoch replays from last saved RNG/dataloader state"}
        atomic_json(summary,output/"training_summary.json")
        print(json.dumps({"epoch":epoch,"train_all59_mse":train["standardized_mse"],"dev_equal_trajectory_H60_mse":val["standardized_mse"]}),flush=True)
        if epoch<30 and time.monotonic()-began>=config.get("max_runtime_seconds",float("inf")):
            return summary
    if start==30:
        atomic_json({"status":"completed","completed_epochs":30,"step":step,"training_identity":digest(identity),
                     "best_epoch":min(history,key=lambda r:r["val"]["standardized_mse"])["epoch"],"best_validation_mse":best,
                     "validation_metric":SELECTION,"validation_precision":PRECISION,"parameter_counts":metadata["parameter_counts"]},output/"training_summary.json")
    verify_registration()
    return validate_completed(output,digest(identity))
