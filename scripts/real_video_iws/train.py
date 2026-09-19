#!/usr/bin/env python3
"""Train or resource-profile the separately registered single-observation IWS study."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws.model import SingleObservationWorldModel
from shiftwm.real_video_iws.windows import IWSWindowDataset
from shiftwm.real_video_iws import training as engine

PACKAGE_KIND, SELECTION = engine.PACKAGE_KIND, engine.SELECTION
load_package, read_package, validate_completed = engine.load_package, engine.read_package, engine.validate_completed
export_inference_package = engine.export_inference_package
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_files():
    paths = ["scripts/real_video_iws/train.py", "scripts/real_video_iws/train.slurm",
             "src/shiftwm/real_video_iws/training.py", "src/shiftwm/real_video_iws/model.py",
             "src/shiftwm/real_video_iws/windows.py", "src/shiftwm/real_video_iws/__init__.py",
             "scripts/real_video/train.py", "scripts/real_video_iws/prepare_cache.py",
             "scripts/real_video_iws_tasks/prepare_cache.py", "src/shiftwm/checkpoint.py",
             "src/shiftwm/real_video/model.py", "src/shiftwm/real_video/data.py",
             "src/shiftwm/real_video/features.py", "src/shiftwm/model.py", "src/shiftwm/upstream.py",
             "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/NOTICE.json"]
    paths += [f"src/shiftwm/{namespace}/{name}.py" for namespace in ("real_video_iws", "real_video_iws_tasks")
              for name in ("cache", "data")]
    return {p: sha(ROOT / p) for p in sorted(set(paths))}


def open_cache(config, task):
    """Check frozen cache identity; no episode payload (especially dev) opened here."""
    if task not in {"pusht":4,"bimanual_box":14,"bimanual_rope":8}:
        raise ValueError("Unknown IWS task")
    row = config["tasks"][task]
    if task == "pusht":
        helper = _module("_iws_frozen_pusht_cache", "scripts/real_video_iws/prepare_cache.py")
        inventory, frozen, inputs, registry = helper.checked_registration(ROOT)
        registration_path = ROOT / helper.REGISTRATION
    else:
        helper = _module("_iws_frozen_other_cache", "scripts/real_video_iws_tasks/prepare_cache.py")
        inventory, frozen, inputs, registry = helper.checked_registration(task, ROOT)
        registration_path = ROOT / helper.task_paths(task)[2]
    if registration_path.resolve() != (ROOT / row["cache_registration"]).resolve():
        raise ValueError("Unexpected cache registration path")
    if (ROOT / row["cache_root"]).resolve() != (ROOT / frozen["output_root"]).resolve():
        raise ValueError("Cannot replace registered cache output with another feature track")
    cache = helper.IWSFeatureCache(ROOT / row["cache_root"], inventory, inputs["episodes"],
                                   helper.static_identity(ROOT, inventory, frozen, inputs))
    stats = cache.statistics
    if (stats.get("ddof") != 1 or stats.get("std_floor") != 1e-5 or
            stats.get("normalization") != "shared_per_channel_over_all_internal_train_native_frames_and_16_patches"):
        raise ValueError("Wrong training-only normalization")
    if len(stats["command_mean"]) != row["action_dim"]:
        raise ValueError("Native task command width differs")
    return cache


def model_for(config, task, mode, stats):
    return SingleObservationWorldModel({**config["model"], "mode": mode,
                                       "action_dim": config["tasks"][task]["action_dim"]},
                 **{k: stats[k] for k in ("feature_mean", "feature_std", "command_mean", "command_std")})


def verify_training_statistics(dataset, cache):
    """Independently recompute once over native training rows, never windows/dev."""
    if dataset.split != "internal_train":
        raise ValueError("Statistics verification requires internal training")
    if dataset.task == "pusht":
        from shiftwm.real_video_iws.data import training_statistics
        actual = training_statistics(((cache.index[e["episode_id"]], {k:e[k] for k in ("features","commands")} | {
            "frame_indices":np.arange(e["frames"],dtype=np.int64),"command_row_indices":np.arange(e["frames"],dtype=np.int64)})
            for e in dataset.episodes), cache.inventory.partitions["internal_train"])
    else:
        from shiftwm.real_video_iws_tasks.data import training_statistics
        actual = training_statistics(((cache.index[e["episode_id"]], {k:e[k] for k in ("features","commands")} | {
            "frame_indices":np.arange(e["frames"],dtype=np.int64),"command_row_indices":np.arange(e["frames"],dtype=np.int64)})
            for e in dataset.episodes), cache.inventory.partitions["internal_train"], dataset.task)
    for key,value in actual.items():
        if value != cache.statistics.get(key):
            raise ValueError("Recomputed training normalization differs: " + key)


class ResourceUnavailable(RuntimeError):
    """Transient allocated-card collision; the scheduler may retry this run."""


def allocated_device():
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU training/profiling requires a scheduler allocation")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Allocated BF16-capable CUDA device required")
    free,total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise ResourceUnavailable("Allocated GPU has less than12GiB actually free; no device substitution")
    return {"slurm_job_id":os.environ["SLURM_JOB_ID"],"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name":torch.cuda.get_device_name(),"free_bytes_before_model":free,"total_bytes":total,
            "torch":str(torch.__version__),"cuda_build":torch.version.cuda}


def train(config_path, task, mode, seed, output, resume=False, max_runtime_seconds=None):
    path=Path(config_path).resolve();config=json.loads(path.read_text())
    campaign=_module("_iws_training_campaign", "scripts/real_video_iws/campaign.py")
    registration=campaign.check_registration(path)
    if task not in config["tasks"] or mode not in config["modes"] or seed not in config["seeds"]:
        raise ValueError("Run is outside the full registered grid")
    resource=allocated_device();torch.set_num_threads(config["training"]["cpu_threads"])
    if config["training"]["num_workers"] != 0:
        raise ValueError("Audited epoch-exact replay requires zero loader workers")
    cache=open_cache(config,task)
    train_data=IWSWindowDataset(cache,"internal_train",horizon=60,stride=5)
    val_data=IWSWindowDataset(cache,"internal_development",horizon=60,stride=5)
    if not len(train_data) or not len(val_data):raise ValueError("Empty eligible training/development population")
    verify_training_statistics(train_data,cache)
    recipe={"task":task,"mode":mode,"seed":seed,"training":config["training"],"model":config["model"],
            "task_config":config["tasks"][task],"study_config_sha256":sha(path)}
    dependencies={str((ROOT/p).resolve()):h for p,h in registration["dependencies"].items()}
    dependencies[str(path)]=sha(path)
    for name in ("manifest.json","identity.json","episode_index.json","training_statistics.json"):
        p=cache.output/name;dependencies[str(p.resolve())]=sha(p)
    identity={"scientific_config":recipe,"dependencies":dependencies,"normalization":cache.statistics,
              "populations":{"train":train_data.audit,"val":val_data.audit},
              "selection":SELECTION,"validation_precision":engine.PRECISION,
              "runtime":{"torch":str(torch.__version__),"numpy":np.__version__,"cuda_build":torch.version.cuda}}
    engine.seed_everything(seed);model=model_for(config,task,mode,cache.statistics)
    run={"scientific_config":recipe,"output_dir":str(Path(output).resolve()),"device":"cuda",
         "resume_if_present":resume,"max_runtime_seconds":max_runtime_seconds if max_runtime_seconds is not None else config["resource_policy"]["epoch_boundary_requeue_after_seconds"]}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/".training.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine.atomic_json(resource,output/"allocated_resource_latest.json")
        return engine.fit(model,run,train_data,val_data,identity,verify_registration=lambda:campaign.check_registration(path))


def profile(config_path, task, mode, microbatch, output, updates=2):
    """Resource-only fullH60 updates; never constructs a development dataset."""
    path=Path(config_path).resolve();config=json.loads(path.read_text())
    campaign=_module("_iws_profile_campaign", "scripts/real_video_iws/campaign.py")
    campaign.validate_config(config)
    if microbatch < 1 or 64 % microbatch or updates < 1:raise ValueError("Invalid profile batch/update count")
    modes=config["modes"] if mode=="all" else [mode]
    if task not in config["tasks"] or any(m not in config["modes"] for m in modes):raise ValueError("Unknown profile arm/task")
    device=allocated_device();torch.set_num_threads(config["training"]["cpu_threads"])
    before=source_files();cache=open_cache(config,task)
    data=IWSWindowDataset(cache,"internal_train",horizon=60,stride=5)
    if len(data)<64:raise ValueError("Insufficient complete training windows for full-sized profile")
    # Fixed real training windows; same batch for every arm. No target-dependent selection.
    loader=DataLoader(Subset(data,list(range(64))),batch_size=64,shuffle=False,num_workers=0)
    batch=next(iter(loader));rows=[]
    result={"status":"running","task":task,"source_hashes":before,"config_sha256":sha(path),
            "config_path":str(path),"resource":device,"horizon":60,"predicted_offsets":59,
            "microbatch_size":microbatch,"batch_size":64,"accumulation_steps":64//microbatch,
            "actual_training_windows":len(data),"training_population":data.audit,
            "development_payloads_opened":0,"official_validation_payloads_opened":0,
            "timing_data":"One fixed complete batch of real internal-training windows for both training and FP32 evaluation timing; no metrics retained.","modes":rows}
    try:
        for arm in modes:
            engine.seed_everything(0);model=model_for(config,task,arm,cache.statistics).cuda()
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config["training"]["lr"],weight_decay=config["training"]["weight_decay"])
            engine.epoch_pass(model,[batch],torch.device("cuda"),optimizer=opt,microbatch_size=microbatch,bf16=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            for _ in range(updates):engine.epoch_pass(model,[batch],torch.device("cuda"),optimizer=opt,microbatch_size=microbatch,bf16=True)
            torch.cuda.synchronize();train_seconds=(time.perf_counter()-started)/updates
            train_peak=torch.cuda.max_memory_allocated();train_reserved=torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            engine.epoch_pass(model,[batch],torch.device("cuda"),microbatch_size=microbatch,bf16=False)
            torch.cuda.synchronize();val_seconds=time.perf_counter()-started
            rows.append({"mode":arm,"status":"passed","optimizer_batch_samples":64,"microbatch_size":microbatch,
                         "timed_optimizer_updates":updates,"warmup_updates":1,"train_seconds_per_update":train_seconds,
                         "train_seconds_per_window":train_seconds/64,"fp32_evaluation_seconds_per_window":val_seconds/64,
                         "fp32_evaluation_timing_split":"internal_train","train_peak_allocated_bytes":train_peak,
                         "train_peak_reserved_bytes":train_reserved,"fp32_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                         "train_epoch_seconds_estimate":math.ceil(len(data)/64)*train_seconds,
                         "parameter_counts":model.parameter_counts})
            engine.atomic_json(result,output);del model,opt;gc.collect();torch.cuda.empty_cache()
        result["source_hashes_after"]=source_files()
        if before!=result["source_hashes_after"] or result["config_sha256"]!=sha(path):raise ValueError("Sources/config changed during profile")
        result["status"]="passed" if modes==config["modes"] else "partial_profile"
    except Exception as error:
        result["status"]="failed";result["failure"]={"type":type(error).__name__,"message":str(error)}
        engine.atomic_json(result,output);raise
    engine.atomic_json(result,output);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    for name in ("train","profile"):
        p=sub.add_parser(name);p.add_argument("--config",required=True);p.add_argument("--task",required=True)
        p.add_argument("--mode",required=True);p.add_argument("--output",required=True)
        if name=="train":
            p.add_argument("--seed",type=int,required=True);p.add_argument("--resume-if-present",action="store_true");p.add_argument("--max-runtime-seconds",type=float)
        else:p.add_argument("--microbatch",type=int,default=16);p.add_argument("--updates",type=int,default=2)
    export=sub.add_parser("export");export.add_argument("--run",required=True);export.add_argument("--output",required=True)
    args=parser.parse_args()
    try:
        if args.command=="train":result=train(args.config,args.task,args.mode,args.seed,args.output,args.resume_if_present,args.max_runtime_seconds)
        elif args.command=="export":result=export_inference_package(args.run,args.output)
        else:result=profile(args.config,args.task,args.mode,args.microbatch,args.output,args.updates)
    except ResourceUnavailable as error:
        result={"status":"allocated_resource_unavailable","reason":str(error),"slurm_job_id":os.environ.get("SLURM_JOB_ID")}
        target=Path(args.output)/"allocated_resource_retry.json" if args.command=="train" else Path(args.output)
        engine.atomic_json(result,target);print(json.dumps(result),flush=True);return 75
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if result["status"] in ("completed","passed","inference_exported") else 75


if __name__=="__main__":raise SystemExit(main())
