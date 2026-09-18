#!/usr/bin/env python3
"""Registered original-train/val-only support reliability diagnostic."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for path in (ROOT / "src", ROOT / "scripts/real_video", HERE.parent, HERE):
    sys.path.insert(0, str(path))
import train as training
from calibrate_residual import load_calibrated_package
from diagnose import development_dataset
from evaluate import crossed_session_bootstrap, baseline_predictions
from shiftwm.real_video.data import atomic_json, sha256
from core import ReliabilityBlend, gate, load_gate, moments, prefix_forecast

CONFIG = ROOT / "configs/real_video_development/reliability_blend_v1.json"
REGISTRY = ROOT / "configs/real_video_development/reliability_blend_registration_v1.json"
REPORT = ROOT / "reports/real_video_development/reliability_blend_results"
ARMS = ("framewise", "factorized", "equal_blend", "train_prior_blend", "train_query_blend",
        "unshrunk_local", "shrunk_local", "one_step_local", "shuffled_local", "persistence", "constant_velocity")


def now():
    return datetime.now(timezone.utc).isoformat()


def source_dependencies():
    paths = dict(training.source_files())
    for path in (HERE / "core.py", HERE / "run.py", HERE / "tests.py", HERE / "run.slurm", CONFIG,
                 ROOT / "reports/real_video_development/reliability_blend_protocol.md",
                 HERE.parent / "calibrate_residual.py", HERE.parent / "diagnose.py",
                 ROOT / "src/shiftwm/real_video_development.py"):
        paths[str(path)] = sha256(path)
    return paths


def register():
    if REGISTRY.exists():
        raise FileExistsError("Registration is immutable")
    cfg = json.loads(CONFIG.read_text())
    cache = ROOT / cfg["cache_root"]
    deps = source_dependencies()
    for filename in ("manifest.json", "training_statistics.json"):
        deps[str(cache / filename)] = sha256(cache / filename)
    manifest = json.loads((cache / "manifest.json").read_text())
    payloads, counts, donors = {}, Counter(), []
    for row in manifest["episodes"]:
        if row["split"] not in ("train", "val"):
            continue
        source = cache / row["cameras"][cfg["camera"]]["file"]
        expected = row["cameras"][cfg["camera"]]["sha256"]
        if sha256(source) != expected:
            raise ValueError("Training/validation feature identity differs")
        payloads[str(source)] = expected
        counts[row["split"]] += 1
    for seed in cfg["seeds"]:
        for mode in ("framewise", "factorized"):
            source = ROOT / f"configs/real_video_development/calibrations/droid_{mode}_s{seed}.json"
            package = json.loads(source.read_text())
            base = (source.parent / package["base_checkpoint"]).resolve()
            if package["fit"]["fit_split"] != "train" or sha256(base / "model.pt") != package["base_checkpoint_sha256"]:
                raise ValueError("Donor must be unchanged and equally training-calibrated")
            deps[str(source)] = sha256(source)
            for name in ("model.pt", "config.json", "package_manifest.json"):
                deps[str(base / name)] = sha256(base / name)
            donors.append({"seed": seed, "mode": mode, "calibration": str(source), "checkpoint": str(base),
                           "model_sha256": sha256(base / "model.pt"), "residual_scale": package["fit"]["scale"]})
    atomic_json({"status": "registered_before_execution", "created_utc": now(), "configuration": cfg,
                 "dependencies": deps, "train_val_payloads": payloads, "episode_counts": dict(counts),
                 "donors": donors, "test_payloads_permitted": False}, REGISTRY)
    print(json.dumps({"registered": str(REGISTRY), "sha256": sha256(REGISTRY), "counts": dict(counts)}), flush=True)


def verify(reg):
    if reg["status"] != "registered_before_execution" or reg["configuration"] != json.loads(CONFIG.read_text()):
        raise ValueError("Registration/configuration differs")
    training.verify_sources(reg["dependencies"])
    training.verify_sources(reg["train_val_payloads"])


def open_dataset(cfg, split):
    if split not in ("train", "val"):
        raise ValueError("Test payload access is forbidden")
    # Original dataset already validates native frame/action grouping and hashes.
    return development_dataset(ROOT / cfg["cache_root"], split, horizon=20,
                               stride=cfg["stride"], camera=cfg["camera"], verify=True)


def population(dataset):
    counts = Counter(i for i, _ in dataset.windows)
    return {"split": dataset.episodes[0]["split"], "total_episodes": len(dataset.episodes),
            "eligible_episodes": len(counts), "sessions": len({dataset.episodes[i]["session_id"] for i in counts}),
            "windows": len(dataset), "excluded_short": [{"episode_id": row["episode_id"], "session_id": row["session_id"],
                "stored_frames": len(row["features"]), "reason": "fewer_than_23_stored_frames"}
                for i, row in enumerate(dataset.episodes) if i not in counts]}


def donor_pair(reg, seed):
    rows = {row["mode"]: row for row in reg["donors"] if row["seed"] == seed}
    return tuple(load_calibrated_package(rows[mode]["calibration"], device="cpu")
                 for mode in ("framewise", "factorized"))


@torch.inference_mode()
def collect(dataset, donors, cfg, destination):
    metadata_path = destination.with_suffix(".json")
    if destination.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["registration_sha256"] != sha256(REGISTRY) or metadata["sha256"] != sha256(destination):
            raise ValueError("Existing sufficient-statistic cache differs")
        with np.load(destination, allow_pickle=False) as data:
            return {key: np.array(data[key]) for key in data.files}, metadata
    pieces = {key: [] for key in ("episode", "start", "prefix_a", "prefix_b", "one_a", "one_b",
                                  "query_a", "query_b", "query_c", "persistence", "constant_velocity")}
    began, max_arithmetic_error = time.monotonic(), 0.
    std = donors[0].feature_std
    if not torch.equal(std, donors[1].feature_std):
        raise ValueError("Donor normalization differs")
    for batch_no, batch in enumerate(DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)):
        features, actions = batch["features"], batch["actions"]
        prefix, past, future, target = features[:, :13], actions[:, :12], actions[:, 12:], features[:, 13:]
        pred = [prefix_forecast(model, prefix, past) for model in donors]
        a, b, _ = moments(*pred, prefix[:, 3:], std)
        pieces["prefix_a"].append(a.mean(1).numpy()); pieces["prefix_b"].append(b.mean(1).numpy())
        pred = [prefix_forecast(model, prefix, past, True) for model in donors]
        a, b, _ = moments(*pred, prefix[:, 3:], std)
        pieces["one_a"].append(a.mean(1).numpy()); pieces["one_b"].append(b.mean(1).numpy())
        pred = [model.predict(prefix[:, -3:], past[:, -2:], future) for model in donors]
        a, b, c = moments(*pred, target, std)
        for key, value in (("query_a", a), ("query_b", b), ("query_c", c)):
            pieces[key].append(value.numpy())
        if batch_no == 0:
            for coefficient in (0., .25, .5, 1.):
                direct = ((pred[0].double() + coefficient*(pred[1].double()-pred[0].double()) - target.double()) / std.double()).square().mean(-1)
                error = (direct - (c - 2*coefficient*b + coefficient**2*a)).abs().max().item()
                max_arithmetic_error = max(max_arithmetic_error, error)
                if error > 1e-10:
                    raise ValueError("Sufficient-statistic arithmetic parity failed")
        for name, prediction in baseline_predictions(prefix[:, -3:], 10).items():
            pieces[name].append(((prediction.double()-target.double())/std.double()).square().mean(-1).numpy())
        pieces["episode"].append(batch["episode_index"].numpy()); pieces["start"].append(batch["window_start"].numpy())
        if batch_no % 10 == 0:
            print(json.dumps({"stage": "collect", "split": dataset.episodes[0]["split"], "batch": batch_no,
                              "elapsed_seconds": round(time.monotonic()-began, 2)}), flush=True)
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite collected values: {key}")
    arrays["session"] = np.array([dataset.episodes[i]["session_id"] for i in arrays["episode"]])
    arrays["episode_id"] = np.array([dataset.episodes[i]["episode_id"] for i in arrays["episode"]])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".tmp").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    destination.with_suffix(".tmp").replace(destination)
    metadata = {"sha256": sha256(destination), "registration_sha256": sha256(REGISTRY), "population": population(dataset),
                "runtime_seconds": time.monotonic()-began, "direct_moment_max_abs_error": max_arithmetic_error,
                "moments": "FP64 from frozen FP32 donor predictions; no decoder/pixel targets", "completed_utc": now()}
    atomic_json(metadata, metadata_path)
    return arrays, metadata


def episode_mean(values, ids, mask=None):
    values, ids = np.asarray(values), np.asarray(ids)
    if mask is not None:
        values, ids = values[mask], ids[mask]
    if len(values) == 0:
        raise ValueError("Empty fit/scoring population")
    return float(np.mean([values[ids == item].mean() for item in np.unique(ids)]))


def quadratic(data, alpha, horizon):
    h = horizon-1
    alpha = np.asarray(alpha, dtype=np.float64)
    result = data["query_c"][:, h] - 2*alpha*data["query_b"][:, h] + alpha**2*data["query_a"][:, h]
    if not np.isfinite(result).all() or result.min() < -1e-10:
        raise ValueError("Invalid reconstructed MSE")
    return np.maximum(result, 0.)


def fit_moments(data, kind, mask=None):
    a = episode_mean(data[kind+"_a"], data["episode"], mask)
    b = episode_mean(data[kind+"_b"], data["episode"], mask)
    return {"prior": float(np.clip(b/a, 0, 1)) if a > 0 else 0., "energy_scale": a}


def fit_gate(data, cfg, kind, split):
    if split != "train":
        raise ValueError("Gate fitting requires training-only moments")
    fold = np.array([int(hashlib.sha256((cfg["fold_salt"]+session).encode()).hexdigest(), 16) % 3 for session in data["session"]])
    records = []
    for kappa in cfg["kappa_grid"]:
        heldout = np.full(len(fold), np.nan)
        fold_parameters = []
        for number in range(3):
            parameters = fit_moments(data, kind, fold != number)
            alpha = gate(data[kind+"_a"], data[kind+"_b"], parameters["prior"], kappa*parameters["energy_scale"])
            heldout[fold == number] = quadratic(data, alpha, 5)[fold == number]
            fold_parameters.append({"fold": number, **parameters,
                                    "fitting_sessions": len(set(data["session"][fold != number])),
                                    "heldout_sessions": len(set(data["session"][fold == number]))})
        if not np.isfinite(heldout).all():
            raise ValueError("Incomplete training-session cross-validation")
        records.append({"kappa": kappa, "out_of_fold_h5": episode_mean(heldout, data["episode"]), "fold_parameters": fold_parameters})
    selected = min(records, key=lambda row: (row["out_of_fold_h5"], -row["kappa"]))
    fit = fit_moments(data, kind)
    return {**fit, "kappa": selected["kappa"], "regularization": selected["kappa"]*fit["energy_scale"],
            "kind": kind, "selection": "training_session_cross_validation_h5_endpoint", "candidates": records}


def shuffle_across_sessions(alpha, sessions, episode_ids, starts):
    order = np.lexsort((starts, episode_ids, sessions))
    maximum = max(Counter(sessions).values())
    if maximum*2 > len(order):
        return None
    donor = np.roll(order, maximum)
    if np.any(sessions[order] == sessions[donor]):
        raise ValueError("Cross-session permutation failed")
    result = np.empty_like(alpha)
    result[order] = alpha[donor]
    return result


def score(data, parameters):
    prior, shrink = parameters["prefix"]["prior"], parameters["prefix"]["regularization"]
    one = parameters["one"]
    alphas = {"framewise": np.zeros(len(data["episode"])), "factorized": np.ones(len(data["episode"])),
              "equal_blend": np.full(len(data["episode"]), .5), "train_prior_blend": np.full(len(data["episode"]), prior),
              "train_query_blend": np.full(len(data["episode"]), parameters["query_prior"]),
              "unshrunk_local": gate(data["prefix_a"], data["prefix_b"], prior, 0),
              "shrunk_local": gate(data["prefix_a"], data["prefix_b"], prior, shrink),
              "one_step_local": gate(data["one_a"], data["one_b"], one["prior"], one["regularization"])}
    alphas["shuffled_local"] = shuffle_across_sessions(alphas["shrunk_local"], data["session"], data["episode_id"], data["start"])
    errors = {arm: {h: quadratic(data, alpha, h) for h in (5,10)}
              for arm, alpha in alphas.items() if alpha is not None}
    rows = []
    for episode in np.unique(data["episode"]):
        mask = data["episode"] == episode
        row = {"episode_id": data["episode_id"][mask][0], "session_id": data["session"][mask][0],
               "windows": int(mask.sum()), "errors": {}, "mean_gate": float(alphas["shrunk_local"][mask].mean())}
        for arm in ARMS:
            if arm in ("persistence", "constant_velocity"):
                row["errors"][arm] = {f"h{h}": float(data[arm][mask, h-1].mean()) for h in (5,10)}
            elif alphas[arm] is not None:
                row["errors"][arm] = {f"h{h}": float(errors[arm][h][mask].mean()) for h in (5,10)}
        rows.append(row)
    return {"episodes": rows, "summary": {arm: {f"h{h}": float(np.mean([row["errors"][arm][f"h{h}"] for row in rows]))
                for h in (5,10)} for arm in rows[0]["errors"]},
            "gate_window_quantiles": np.quantile(alphas["shrunk_local"], [0,.25,.5,.75,1]).tolist(),
            "shuffled_available": alphas["shuffled_local"] is not None}


@torch.inference_mode()
def export_gate(reg, seed, parameters, donors, example, artifact):
    directory = artifact / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for row in (row for row in reg["donors"] if row["seed"] == seed):
        name = f"droid_{row['mode']}_s{seed}"
        destination = artifact / "donors" / name
        if not destination.exists():
            destination.mkdir(parents=True)
            for filename in ("model.pt", "config.json"):
                shutil.copyfile(Path(row["checkpoint"]) / filename, destination / filename)
            atomic_json({"format_version": 1, "package_kind": training.PACKAGE_KIND,
                         "files": {name: sha256(destination/name) for name in ("model.pt", "config.json")}}, destination / "package_manifest.json")
        package = json.loads(Path(row["calibration"]).read_text())
        package["base_checkpoint"] = f"../donors/{name}"
        calibration = directory / (row["mode"] + "_calibration.json")
        atomic_json(package, calibration)
        manifest[row["mode"]] = {"calibration": calibration.name, "sha256": sha256(calibration)}
    package = {"format_version": 1, "kind": "development_causal_reliability_blend", "fit_split": "train", "seed": seed,
               "parameters": parameters["prefix"], "donors": manifest, "registration_sha256": sha256(REGISTRY),
               "inference_source_sha256": sha256(HERE/"core.py"), "input": "13 observed frames,12 observed action blocks,5 or10 future commands; no query images"}
    atomic_json(package, directory / "gate.json")
    reference = ReliabilityBlend(*donors, parameters["prefix"]["prior"], parameters["prefix"]["regularization"]).eval()
    relocated = load_gate(directory / "gate.json")
    args = (example["features"][:13][None], example["actions"][:12][None], example["actions"][12:][None])
    expected, actual = reference.predict(*args), relocated.predict(*args)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    return {"gate_json": str((directory/"gate.json").relative_to(ROOT)), "sha256": sha256(directory/"gate.json"),
            "offline_reload_max_abs_error": float((expected-actual).abs().max()),
            "combined_donor_parameters": sum(p.numel() for donor in donors for p in donor.parameters())}


def finalize(records, cfg, reg, execution):
    reference_ids = [row["episode_id"] for row in records[0]["scores"]["episodes"]]
    sessions = [row["session_id"] for row in records[0]["scores"]["episodes"]]
    for record in records:
        if [row["episode_id"] for row in record["scores"]["episodes"]] != reference_ids:
            raise ValueError("Seed populations differ")
    arms = list(records[0]["scores"]["summary"])
    arrays = {arm: {f"h{h}": np.array([[row["errors"][arm][f"h{h}"] for row in record["scores"]["episodes"]]
                       for record in records]) for h in (5,10)} for arm in arms}
    summary, intervals = {}, {}
    for arm in arms:
        summary[arm] = {h: float(value.mean()) for h, value in arrays[arm].items()}
        intervals[arm] = {h: crossed_session_bootstrap(value-arrays["train_prior_blend"][h], sessions,
                draws=cfg["bootstrap_draws"], seed=cfg["bootstrap_seed"]) for h, value in arrays[arm].items()}
    ours = summary["shrunk_local"]
    comparators = {"stronger_donor": min(summary["framewise"]["h5"], summary["factorized"]["h5"]),
                   "train_prior_blend": summary["train_prior_blend"]["h5"], "train_query_blend": summary["train_query_blend"]["h5"]}
    gains = {key: 1-ours["h5"]/value for key, value in comparators.items()}
    criteria = {"at_least_one_percent_versus_"+key: value >= cfg["promotion_relative_gain"] for key, value in gains.items()}
    criteria.update({"h5_ci_vs_prior_strictly_negative": intervals["shrunk_local"]["h5"]["ci95"][1] < 0,
                     "all_seeds_improve_h5_vs_prior": bool(np.all((arrays["shrunk_local"]["h5"]-arrays["train_prior_blend"]["h5"]).mean(1)<0)),
                     "h10_no_worse_than_prior": ours["h10"] <= summary["train_prior_blend"]["h10"]})
    report = {"status": "completed", "scope": "original validation development only; not confirmatory", "completed_utc": now(),
              "registration_sha256": sha256(REGISTRY), "execution": execution, "summary": summary,
              "paired_intervals_vs_train_prior": intervals, "relative_h5_gains": gains, "promotion_criteria": criteria,
              "promote": all(criteria.values()), "runs": records,
              "interpretation": "Support reliability/ensembling control; no novel architecture, SOTA, or physical success claim"}
    atomic_json(report, REPORT.with_suffix(".json"))
    lines = ["# Causal support-reliability diagnostic results", "", "Original validation only; exploratory development evidence. All arms share the same 13-frame prefix and eligible windows. Frozen equally calibrated donors; no new neural training.", "",
             "| Arm | h5 MSE | h10 MSE | h5 change vs global-prior blend | h5 paired MSE difference 95% interval |", "|---|---:|---:|---:|---|"]
    for arm in arms:
        interval = intervals[arm]["h5"]
        gain = 100*(1-summary[arm]["h5"]/summary["train_prior_blend"]["h5"])
        lines.append(f"| {arm} | {summary[arm]['h5']:.8f} | {summary[arm]['h10']:.8f} | {gain:+.3f}% | [{interval['ci95'][0]:+.8f}, {interval['ci95'][1]:+.8f}] |")
    lines += ["", f"Prespecified promotion: **{'PASS' if report['promote'] else 'FAIL — do not promote'}**.", "", "| Criterion | Passed |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key,value in criteria.items()]
    lines += ["", "All intervals resample recording sessions and training seeds and are unadjusted development intervals. Prefix-fit improvements are not guarantees on future queries. Shuffled gates are an observational mechanism diagnostic, not a deployment method.", "",
              "## Training-only fitted parameters", "", "| Seed | Prefix prior | Prefix kappa | One-step prior | One-step kappa | Query-fitted constant |", "|---|---:|---:|---:|---:|---:|"]
    for row in records:
        p=row["parameters"]
        lines.append(f"| {row['seed']} | {p['prefix']['prior']:.6f} | {p['prefix']['kappa']} | {p['one']['prior']:.6f} | {p['one']['kappa']} | {p['query_prior']:.6f} |")
    lines += ["", "Complete per-seed/per-episode errors, h10 intervals, length exclusions, gate distributions and artifact identities are retained in the companion JSON and sufficient-statistic caches.", "", json.dumps(execution, indent=2)]
    REPORT.with_suffix(".md").write_text("\n".join(lines)+"\n")
    return report


def run():
    began = time.monotonic()
    reg = json.loads(REGISTRY.read_text()); cfg=reg["configuration"]
    torch.set_num_threads(cfg["cpu_threads"])
    training.seed_everything(20260919)
    verify(reg)
    output, artifact = ROOT/cfg["output"], ROOT/cfg["artifact"]
    output.mkdir(parents=True, exist_ok=True); artifact.mkdir(parents=True, exist_ok=True)
    train_data = open_dataset(cfg, "train")
    fitted, train_metadata, exports = {}, {}, {}
    for seed in cfg["seeds"]:
        donors = donor_pair(reg,seed)
        data, meta=collect(train_data,donors,cfg,output/f"train_s{seed}.npz")
        prefix, one = fit_gate(data,cfg,"prefix","train"), fit_gate(data,cfg,"one","train")
        energy=episode_mean(data["query_a"][:,4],data["episode"])
        alignment=episode_mean(data["query_b"][:,4],data["episode"])
        fitted[str(seed)]={"prefix":prefix,"one":one,"query_prior":float(np.clip(alignment/energy,0,1)) if energy>0 else 0.}
        train_metadata[str(seed)]=meta
        exports[str(seed)]=export_gate(reg,seed,fitted[str(seed)],donors,train_data[0],artifact)
        print(json.dumps({"stage":"train_fit_complete","seed":seed,"kappa":prefix["kappa"],"prior":prefix["prior"]}),flush=True)
        del donors,data
    lock={"registration_sha256":sha256(REGISTRY),"parameters":fitted,"train_statistics":train_metadata,"exports":exports}
    lock_path=output/"training_only_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text())!=lock:raise ValueError("Existing training-only lock differs")
    else:atomic_json(lock,lock_path)
    locked_sha=sha256(lock_path)
    del train_data
    # Earlier integrity checks hash bytes only. Validation arrays are not decoded
    # or passed to a model until all three seeds and exported gates are locked.
    validation_started=now()
    val_data=open_dataset(cfg,"val")
    records=[]
    for seed in cfg["seeds"]:
        donors=donor_pair(reg,seed)
        data,meta=collect(val_data,donors,cfg,output/f"val_s{seed}.npz")
        records.append({"seed":seed,"parameters":fitted[str(seed)],"train":train_metadata[str(seed)],"validation":meta,
                        "export":exports[str(seed)],"scores":score(data,fitted[str(seed)])})
        del donors,data
    if sha256(lock_path)!=locked_sha:raise ValueError("Training-only lock changed after validation")
    verify(reg)
    execution={"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"device":"cpu","cpu_threads":cfg["cpu_threads"],
               "runtime_seconds":time.monotonic()-began,"maximum_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "validation_features_first_decoded_utc":validation_started,"training_only_lock_sha256":locked_sha,
               "train_population":train_metadata["0"]["population"],"validation_population":population(val_data),
               "hashes_verified_at_start_and_end":True,"gate_inference_cost":"2 donor ten-step prefix + 2 donor query rollouts; one-step control uses 20 extra one-step donor predictions during diagnostic collection"}
    result=finalize(records,cfg,reg,execution)
    shutil.copyfile(HERE/"core.py",artifact/"inference_source.py")
    shutil.copyfile(CONFIG,artifact/"study_configuration.json")
    shutil.copyfile(ROOT/"reports/real_video_development/reliability_blend_protocol.md",artifact/"PROTOCOL.md")
    shutil.copyfile(lock_path,artifact/"training_only_lock.json")
    (artifact/"MODEL_CARD.md").write_text("# Causal reliability blend — development control\n\nTwo frozen calibrated donors plus training-fitted scalar gate; no new neural weights. Original validation only; not physical control or SOTA.\n\nLoad `core.load_gate(artifact/seed0/gate.json)` in this repository environment with the new reliability namespace and original calibration loader on PYTHONPATH. Relative donor/calibration paths survive moving the complete artifact directory. Prediction takes thirteen observed feature frames, twelve executed command blocks and five/ten future command blocks; never query images. All features use the original frozen encoder and training normalization. Both donor weights and compute are required.\n\nPromotion: "+str(result["promote"])+". See reports/real_video_development/reliability_blend_results.md for all outcomes and limitations.\n")
    atomic_json({"registration_sha256":sha256(REGISTRY),"files":{str(p.relative_to(artifact)):sha256(p) for p in sorted(artifact.rglob("*")) if p.is_file() and p.name!="artifact_manifest.json"}},artifact/"artifact_manifest.json")
    atomic_json(execution,output/"execution.json")
    print(json.dumps({"status":"completed","promote":result["promote"],"runtime_seconds":execution["runtime_seconds"],"report":str(REPORT.with_suffix('.md'))}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("register","run"));args=parser.parse_args()
    register() if args.command=="register" else run()
