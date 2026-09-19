#!/usr/bin/env python3
"""Complete-nine exploratory report against all27 immutable common-CPU v1 runs.

Only hash-bound packages, metadata and existing primitive metric ledgers are read.
No dataset payload/model inference/paper edit occurs in this finalizer. Results
are committed last after all36 runs and all paired populations pass validation.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


campaign = module(Path(__file__).with_name("campaign.py"), "_unbounded_report_campaign")
frozen = module(ROOT / "scripts/real_video_iws/finalize.py", "_unbounded_frozen_result_validator")
FINAL = campaign.REPORT / "development_finalization.json"
METRICS = tuple(frozen.evaluation.METRICS)
TASKS = tuple(campaign.TASKS)
OURS = campaign.MODE
COMPARATORS = ("bounded_spatial_mix", "anchored_additive", "autoregressive", "persistence")
MODES = (OURS, *COMPARATORS)
HORIZONS = (15,30,45,60)


def check_run_population(rows):
    expected = {(t,m,s) for t in TASKS for m in MODES if m != "persistence" for s in range(3)}
    keys = [(r.get("task"), r.get("mode"), r.get("seed")) for r in rows]
    if len(keys) != 36 or set(keys) != expected:
        raise ValueError("Reporting requires all9 new and all27 baseline runs, without duplicates")


def arrays_from_receipts(rows, audits):
    check_run_population(rows)
    arrays = {task:{mode:np.empty((3,audits[task]["eligible_episodes"],59,len(METRICS)),dtype=np.float64)
                    for mode in MODES} for task in TASKS}
    fixed_persistence = {}
    for row in rows:
        task,mode,seed = row["task"], row["mode"], row["seed"]
        expected = [(r["episode_id"], r["windows"]) for r in audits[task]["records"] if r["windows"] > 0]
        episodes = row["receipt"]["episodes"]
        if [(e["episode_id"],e["windows"]) for e in episodes] != expected:
            raise ValueError("Matched task/trajectory population differs")
        predicted = np.stack([np.stack([e[m+"_by_offset"] for m in METRICS],axis=-1) for e in episodes])
        persistence = np.stack([np.stack([e["persistence_"+m+"_by_offset"] for m in METRICS],axis=-1) for e in episodes])
        shape = (len(expected),59,len(METRICS))
        if (predicted.shape != shape or persistence.shape != shape or not np.isfinite(predicted).all()
                or not np.isfinite(persistence).all() or (predicted < 0).any() or (persistence < 0).any()):
            raise ValueError("Incomplete or invalid full-horizon metric arrays")
        if task in fixed_persistence and not np.array_equal(persistence, fixed_persistence[task]):
            raise ValueError("Common-CPU persistence differs across matched models/seeds")
        fixed_persistence[task] = persistence
        arrays[task][mode][seed] = predicted
        arrays[task]["persistence"][seed] = persistence
    return arrays


def primitive_persistence_equal(new_path, baseline_path):
    keys = ("episode_index","window_start",*("persistence_"+m for m in METRICS))
    with np.load(new_path,allow_pickle=False) as new, np.load(baseline_path,allow_pickle=False) as reference:
        for key in keys:
            if (new[key].dtype != reference[key].dtype or not np.array_equal(new[key],reference[key])):
                raise ValueError("Primitive common-CPU persistence/window identity differs: " + key)


def crossed_intervals(arrays, draws=10000, seed=173):
    """Same matched seed draw across tasks; separate paired trajectories per task.

    Four metrics and four comparators share each draw. Only H60 is inferred;
    all59 curves and H15/30/45 summaries remain descriptive.
    """
    rng = np.random.default_rng(seed)
    sampled = {task:{"difference":np.empty((draws,4,4)),"gain":np.empty((draws,4,4))} for task in TASKS}
    stacked = {t:np.stack([arrays[t][m][:,:,-1,:] for m in MODES],axis=2) for t in TASKS}
    macro = np.empty((draws,4,4))
    for index in range(draws):
        selected_seeds = rng.integers(0,3,size=3); gains = []
        for task in TASKS:
            n = stacked[task].shape[1]; trajectories = rng.integers(0,n,size=n)
            means = stacked[task][selected_seeds][:,trajectories].mean(axis=(0,1))
            difference = means[0][None,:] - means[1:]
            gain = np.divide(-100*difference, means[1:], out=np.full((4,4),np.nan), where=means[1:]!=0)
            sampled[task]["difference"][index] = difference
            sampled[task]["gain"][index] = gain; gains.append(gain)
        macro[index] = np.mean(gains,axis=0)
    def interval(values):
        return None if not np.isfinite(values).all() else np.quantile(values,[.025,.975]).tolist()
    return {"task":{t:{m:{c:{key:interval(sampled[t][key][:,ci,mi]) for key in ("difference","gain")}
                           for ci,c in enumerate(COMPARATORS)} for mi,m in enumerate(METRICS)} for t in TASKS},
            "macro_relative_gain":{m:{c:interval(macro[:,ci,mi]) for ci,c in enumerate(COMPARATORS)}
                                   for mi,m in enumerate(METRICS)},
            "draws":draws,"seed":seed,"scope":"H60 only, unadjusted exploratory 95% percentile intervals",
            "resampling":"same matched seed draw across tasks; independent trajectory draws within task; paired methods/metrics"}


def numerical_report(arrays):
    intervals = crossed_intervals(arrays)
    results = {}; macros = {}
    for task in TASKS:
        results[task] = {}
        for mi,metric in enumerate(METRICS):
            curves = {mode:arrays[task][mode][:,:,:,mi].mean(axis=(0,1)) for mode in MODES}
            effects = {}
            for mode in COMPARATORS:
                a,b = float(curves[OURS][-1]),float(curves[mode][-1])
                effects[mode] = {"method_mean":a,"comparator_mean":b,"method_minus_comparator":a-b,
                    "relative_error_reduction_percent":None if b==0 else 100*(b-a)/b,
                    "paired95":intervals["task"][task][metric][mode]}
            results[task][metric] = {"means_all59_offsets":{m:v.tolist() for m,v in curves.items()},
                "horizon_means":{str(h):{m:float(v[h-2]) for m,v in curves.items()} for h in HORIZONS},
                "per_seed_horizon_means":{str(h):{m:arrays[task][m][:,:,h-2,mi].mean(1).tolist() for m in MODES} for h in HORIZONS},
                "h60_comparisons":effects}
    for metric in METRICS:
        macros[metric] = {}
        for mode in COMPARATORS:
            gains = [results[t][metric]["h60_comparisons"][mode]["relative_error_reduction_percent"] for t in TASKS]
            macros[metric][mode] = {"equal_task_relative_error_reduction_percent":None if any(v is None for v in gains) else float(np.mean(gains)),
                                    "paired95":intervals["macro_relative_gain"][metric][mode]}
    return {"task_results":results,"macro_h60":macros,"bootstrap":{k:v for k,v in intervals.items() if k not in ("task","macro_relative_gain")},
            "primary":"H60 standardized MSE: unbounded versus bounded mixing; equal-task macro of task-relative reductions",
            "scope":"Exploratory after v1 development; no reserved evaluation or new algorithm claim"}


def collect(config_path, registry):
    config = campaign.read(config_path); sources = dict(registry["dependencies"])
    baseline_path = ROOT / config["evaluation"]["baseline_finalization"]
    baseline = campaign.read(baseline_path)
    frozen.verify_existing(baseline, campaign.base.check_registration())
    for path,digest in baseline["source_dependencies"].items(): frozen.bind(ROOT/path,sources,digest)
    frozen.bind(baseline_path,sources)
    trainer = campaign.trainer()
    audits = {t:frozen.metadata_audit(trainer.open_cache(config,t)) for t in TASKS}
    rows = []
    baseline_primitive = {}
    for row in baseline["per_run"]:
        receipt = campaign.read(ROOT / row["evaluation_path"])
        frozen.bind(ROOT / row["evaluation_path"],sources,row["evaluation_sha256"])
        rows.append({**row,"receipt":receipt})
        if row["mode"] == "autoregressive" and row["seed"] == 0:
            baseline_primitive[row["task"]] = ROOT / row["window_ledger_path"]
    config_sha = frozen.bind(config_path,sources)
    registration_sha = frozen.bind(campaign.REGISTRATION,sources)
    for row in registry["runs"]:
        directory = ROOT / row["output"]
        summary = trainer.validate_completed(directory)
        best,state = trainer.read_package(directory / "best")
        last,_ = trainer.read_package(directory / "last",True)
        identity = state["config"]["metadata"]["identity"]
        expected = {"task":row["task"],"mode":row["mode"],"seed":row["seed"],"model":config["model"],
                    "training":config["training"],"task_config":config["tasks"][row["task"]],"study_config_sha256":config_sha}
        if identity["scientific_config"] != expected or identity["populations"]["val"] != audits[row["task"]]:
            raise ValueError("New selected package recipe or development population changed")
        for name in ("training_config.json","training_summary.json","metrics.jsonl"):
            frozen.bind(directory/name,sources)
        for package in (best,last):
            frozen.bind(package/"package_manifest.json",sources)
            for name,digest in campaign.read(package/"package_manifest.json")["files"].items():
                frozen.bind(package/name,sources,digest)
        checkpoint = frozen.bind(best/"model.pt",sources)
        path = campaign.REPORT / "evaluations" / (row["name"]+".json")
        receipt = frozen.validate_evaluation(path,row,summary,checkpoint,audits[row["task"]],config_sha,registration_sha,sources)
        primitive_persistence_equal(path.with_suffix(".npz"),baseline_primitive[row["task"]])
        adapter_path = path.with_suffix(".adapter.json"); adapter = campaign.read(adapter_path)
        attempt_path = path.with_suffix(".attempt.json")
        completion_path = campaign.REPORT / "completions" / (row["name"]+".json"); completion = campaign.read(completion_path)
        if (adapter.get("status") != "passed" or adapter.get("device") != "cpu"
                or adapter.get("evaluation_sha256") != campaign.sha(path)
                or adapter.get("adapter_sha256") != campaign.sha(Path(__file__).with_name("evaluate.py"))
                or adapter.get("registration_sha256") != registration_sha
                or adapter.get("attempt_sha256") != campaign.sha(attempt_path)
                or adapter.get("official_validation_payloads_read") != 0
                or completion.get("status") != "training_and_cpu_development_evaluation_completed"
                or completion.get("evaluation_sha256") != campaign.sha(path)
                or completion.get("summary_sha256") != campaign.sha(directory/"training_summary.json")):
            raise ValueError("Missing or mismatched CPU adapter/completion provenance")
        frozen.bind(adapter_path,sources);frozen.bind(attempt_path,sources);frozen.bind(completion_path,sources)
        rows.append({"name":row["name"],"task":row["task"],"mode":row["mode"],"seed":row["seed"],
                     "evaluation_path":str(path.relative_to(ROOT)),"evaluation_sha256":campaign.sha(path),
                     "selected_epoch":summary["best_epoch"],"selected_checkpoint_sha256":checkpoint,"completed_epochs":30,
                     "window_ledger_path":receipt["window_ledger_path"],"window_ledger_sha256":receipt["window_ledger_sha256"],"receipt":receipt})
    arrays = arrays_from_receipts(rows,audits)
    report = numerical_report(arrays)
    return {"schema":"iws_unbounded_complete_comparison_v1","status":"passed","completed_utc":campaign.now(),
            "completed_new_runs":9,"completed_v1_comparator_runs":27,"scope":"exploratory_internal_development_after_v1",
            "registration_sha256":registration_sha,"baseline_finalization_sha256":campaign.sha(baseline_path),
            "per_run":[{k:v for k,v in r.items() if k!="receipt"} for r in rows],
            "populations":audits,"results":report,"source_dependencies":sources,"official_validation_payloads_read":0,
            "new_model_inference_performed":False,"paper_ingestion":"Manual source/number/layout review still required"}


def table_text(result):
    lines = ["# IWS unbounded-innovation ablation — complete exploratory development report", "",
        "All nine new runs and all 27 frozen common-CPU v1 comparison runs passed. No reserved data were used. Lower errors are better; positive relative gains favor the unbounded ablation. This is a component study after development, not a new algorithm or a confirmatory result.", "",
        "Means average windows within trajectory, trajectories and three seeds equally. H labels count command rows; the target is stored offset H−1. Only H60 has paired 10,000-draw seed×trajectory intervals (seed173), unadjusted exploratory95% intervals. Prefix summaries are descriptive.", "",
        "| Task | Metric | H | Unbounded | Bounded | Additive | AR | Persistence |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for metric in METRICS:
            for h in HORIZONS:
                means = result["results"]["task_results"][task][metric]["horizon_means"][str(h)]
                lines.append("| "+" | ".join([task,metric,str(h)]+[f"{means[m]:.9g}" for m in MODES])+" |")
    lines += ["", "| Task | Metric at H60 | Comparator | Unbounded−comparator | Relative gain (%) | Paired95% gain interval (%) |", "|---|---|---|---:|---:|---|"]
    for task in TASKS:
        for metric in METRICS:
            for mode,effect in result["results"]["task_results"][task][metric]["h60_comparisons"].items():
                gain=effect["relative_error_reduction_percent"]; ci=effect["paired95"]["gain"]
                lines.append(f"| {task} | {metric} | {mode} | {effect['method_minus_comparator']:+.9g} | "+
                    ("undefined" if gain is None else f"{gain:+.6f}")+" | "+("undefined" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]")+" |")
    lines += ["", "| Equal-task macro, H60 metric | Comparator | Mean relative gain (%) | Paired95% interval (%) |", "|---|---|---:|---|"]
    for metric in METRICS:
        for mode,effect in result["results"]["macro_h60"][metric].items():
            gain=effect["equal_task_relative_error_reduction_percent"];ci=effect["paired95"]
            lines.append(f"| {metric} | {mode} | "+("undefined" if gain is None else f"{gain:+.6f}")+" | "+("undefined" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]")+" |")
    return "\n".join(lines)+"\n"


def finalize(config_path=campaign.CONFIG, if_ready=False):
    registry = campaign.check_registration(config_path)
    needed = [campaign.REPORT/"completions"/(r["name"]+".json") for r in registry["runs"]]
    if not all(p.exists() for p in needed):
        if if_ready:return {"status":"pending","completed_new_runs":sum(p.exists() for p in needed),"outputs_written":False}
        raise ValueError("All nine completed new evaluations are required")
    campaign.REPORT.mkdir(parents=True,exist_ok=True)
    with (campaign.REPORT/".finalizer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if FINAL.exists():
            old = campaign.read(FINAL)
            if (old.get("status") != "passed" or old.get("completed_new_runs") != 9
                    or old.get("schema") != "iws_unbounded_complete_comparison_v1"
                    or old.get("completed_v1_comparator_runs") != 27
                    or old.get("official_validation_payloads_read") != 0
                    or old.get("registration_sha256") != campaign.sha(campaign.REGISTRATION)):
                raise ValueError("Existing comparison completion differs")
            check_run_population(old["per_run"])
            for path,digest in old["source_dependencies"].items():
                if campaign.sha(ROOT/path) != digest:raise ValueError("Final evidence changed: "+path)
            return {"status":"unchanged_verified","completion_sha256":campaign.sha(FINAL)}
        value = collect(config_path,registry)
        for path,digest in value["source_dependencies"].items():
            if campaign.sha(ROOT/path) != digest:raise ValueError("Evidence changed before commit")
        report = campaign.REPORT/"comparison_report.md"
        report.write_text(table_text(value))
        value["source_dependencies"][str(report.relative_to(ROOT))] = campaign.sha(report)
        campaign.check_registration(config_path)
        for path,digest in value["source_dependencies"].items():
            if campaign.sha(ROOT/path) != digest:raise ValueError("Evidence changed before completion commit")
        campaign.atomic_json(value,FINAL)
        return {"status":"passed","completed_new_runs":9,"completed_v1_comparator_runs":27,"completion_sha256":campaign.sha(FINAL)}


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--config",type=Path,default=campaign.CONFIG);p.add_argument("--if-ready",action="store_true")
    a=p.parse_args();print(json.dumps(finalize(a.config,a.if_ready),sort_keys=True))
