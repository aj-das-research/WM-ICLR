#!/usr/bin/env python3
"""Verify completed local studies and report all paired development outcomes.

This reporting utility does not train, evaluate, mutate study artifacts, or
promote development observations to confirmatory evidence.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from shiftwm.extensions.checkpoint import atomic_json, file_sha256
from shiftwm.extensions.train import scientific_config, validate_completed
from shiftwm.extensions.geometry_revision import validate_completed as validate_geometry

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = {}
SHA_CACHE = {}
LABELS = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics", "factorized": "ShiftWM (ours)"}


def digest(path, expected=None):
    path = Path(path).resolve(strict=True)
    if path not in SHA_CACHE:
        SHA_CACHE[path] = file_sha256(path)
    actual = SHA_CACHE[path]
    if expected is not None and actual != expected:
        raise ValueError(f"Hash mismatch: {path}; expected {expected}, got {actual}")
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    EVIDENCE[label] = actual
    return actual


def read(path):
    digest(path)
    return json.loads(Path(path).read_text())


def key(row):
    return (row["seed"], row["dynamics_id"], row["observation_id"])


def verify_result(path, *, planning):
    result = read(path)
    assert result["status"] == "completed", path
    assert result["identity"] == read(path.parent / "identity.json"), path
    for source, expected in result["identity"]["sources"].items():
        digest(source, expected)
    arguments = result["identity"]["arguments"]
    assert arguments["split"] == "development" and arguments["episodes_per_gain"] == 8
    assert arguments["planner_seed"] == 101
    if planning:
        records = result["records"]
        assert result["tasks"] == len(records) == 8
        assert len({key(r) for r in records}) == 8
        assert result["successes"] == sum(r["success"] for r in records)
        assert result["support_successes"] == sum(r["success_during_support"] for r in records)
        for row in records:
            digest(path.parent / row["trace_file"], row["trace_sha256"])
            saved = read(path.parent / (Path(row["trace_file"]).stem + ".json"))
            assert saved == row, (path, row["trajectory_id"])
            assert row["native_budget"] == 200 and row["support_budget"] == 10
            assert len(row["metrics_per_native_step"]) == row["native_calls"] <= 200
            successful = [i + 1 for i, v in enumerate(row["metrics_per_native_step"]) if v["success"]]
            assert row["success"] == bool(successful)
            assert row["first_success_native_step"] == (successful[0] if successful else None)
    else:
        assert result["sequence_length"] == 8 and result["window_stride"] == 4
        assert result["split"] == "development"
        for horizon in (1, 3, 5):
            values = [r[f"mse_h{horizon}"] for r in result["records"]]
            assert np.isclose(np.mean(values), result["summary"]["o1_d1"][f"mse_h{horizon}"]["mean"], rtol=1e-7)
    return result


def verify_training(row, *, geometry):
    output = Path(row["output"])
    summary = read(output / "training_summary.json")
    (validate_geometry if geometry else validate_completed)(output, summary["training_identity"])
    for name in ("run_config.json", "metrics.jsonl"):
        digest(output / name)
    registered = read(row["config"])
    assert scientific_config(registered) == scientific_config(read(output / "run_config.json"))
    packages = {}
    for which in ("best", "last"):
        directory = (output / which).resolve(strict=True)
        manifest = read(directory / "package_manifest.json")
        for filename, expected in manifest["files"].items():
            digest(directory / filename, expected)
        packages[which] = {"resolved_directory": str(directory.relative_to(ROOT)),
                           "package_manifest_sha256": digest(directory / "package_manifest.json"),
                           "files": manifest["files"]}
    config = read(output / "best/config.json")
    prov = config["provenance"]
    for name, expected in prov["extension_source_hashes"].items():
        digest(ROOT / "src/shiftwm" / name, expected)
    for path, field in [(Path(registered["data_root"]) / "manifest.json", "data_manifest_sha256"),
                        (Path(registered["dataset_kwargs"]["feature_cache"]) / "manifest.json", "cache_manifest_sha256"),
                        (Path(registered["action_stats"]), "action_stats_sha256"),
                        (Path(registered["protocol_path"]), "extension_protocol_sha256")]:
        digest(path, prov[field])
    if geometry:
        digest("src/shiftwm/extensions/geometry_revision.py", prov["geometry_revision"]["implementation_sha256"])
    return {"training_summary": summary, "packages": packages,
            "checkpoint_selection": config["metadata"]["selection"],
            "checkpoint_sha256": packages["best"]["files"]["model.pt"]}


def read_campaign(path, result_root, *, geometry=False):
    campaign = read(path)
    runs, grouped, forecasts = [], defaultdict(dict), defaultdict(dict)
    for row in campaign["runs"]:
        domain, architecture = row.get("domain", "drone"), row.get("architecture", "transformer")
        run_name = Path(row["output"]).name
        print("Verifying", run_name, "geometry" if geometry else "original", flush=True)
        training = verify_training(row, geometry=geometry)
        base = Path(result_root) / run_name
        if not geometry:
            base /= "development"
        forecast = verify_result(base / "forecast/results.json", planning=False)
        planning = verify_result(base / "planning/results.json", planning=True)
        for result in (forecast, planning):
            assert result["identity"]["arguments"]["checkpoint"] == row["output"] + "/best"
            assert result["identity"]["model_config"]["mode"] == row["mode"]
            assert result["identity"]["model_config"]["architecture"] == architecture
        for existing in grouped[(domain, architecture, row["mode"])].values():
            assert existing["selection"] == planning["selection"]
        grouped[(domain, architecture, row["mode"])][row["seed"]] = planning
        forecasts[(domain, architecture, row["mode"])][row["seed"]] = forecast
        runs.append({"name": run_name, "domain": domain, "architecture": architecture,
                     "mode": row["mode"], "training_seed": row["seed"], **training,
                     "forecast_result": str(base / "forecast/results.json"),
                     "planning_result": str(base / "planning/results.json"),
                     "forecast_mse": {f"h{h}": forecast["summary"]["o1_d1"][f"mse_h{h}"]["mean"] for h in (1, 3, 5)},
                     "forecast_trajectories": forecast["summary"]["o1_d1"]["mse_h5"]["clusters"],
                     "forecast_windows": len(forecast["records"]),
                     "successes": planning["successes"], "tasks": planning["tasks"],
                     "support_successes": planning["support_successes"]})
    return runs, grouped, forecasts


def pair(ours, baseline, function):
    result = function(ours, baseline)
    assert result["status"] == "complete"
    inventory = []
    for seed in (0, 1, 2):
        a, b = ({key(v): v for v in x[seed]["records"]} for x in (ours, baseline))
        assert set(a) == set(b)
        for task in sorted(a):
            ar, br = a[task], b[task]
            assert ar["initial_metrics"] == br["initial_metrics"]
            assert ar["metrics_per_native_step"][:10] == br["metrics_per_native_step"][:10]
            inventory.append({"training_seed": seed, "task_seed": task[0], "dynamics_id": task[1],
                              "observation_id": task[2], "ours_success": ar["success"],
                              "baseline_success": br["success"], "ours_native_calls": ar["native_calls"],
                              "baseline_native_calls": br["native_calls"]})
    result["paired_inventory"] = inventory
    result["discordances"] = {
        "ours_only": sum(r["ours_success"] and not r["baseline_success"] for r in inventory),
        "baseline_only": sum(r["baseline_success"] and not r["ours_success"] for r in inventory),
        "both_success": sum(r["baseline_success"] and r["ours_success"] for r in inventory),
        "both_failure": sum(not r["baseline_success"] and not r["ours_success"] for r in inventory)}
    return result


def official_adajepa():
    root = Path("results/baselines/adajepa")
    campaign = read(root / "campaign.json")
    assert campaign["status"] == "complete_required_protocol" and len(campaign["completed"]) == 4
    repo = ROOT / "external/adajepa"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True) == ""
    first_stage_note = read("reports/evidence/adajepa_gpu_start.json")
    digest("scripts/baselines/adajepa_run.py", first_stage_note["current_launcher_sha256"])
    stages = []
    for condition in ("clean", "blur"):
        for method in ("frozen", "adaptive"):
            folder = root / f"{condition}-{method}-seed100"
            row = read(folder / "reproduction_record.json")
            assert row["status"] == "complete" and row["return_code"] == 0
            assert (row["condition"], row["method"], row["episodes"], row["seed"]) == (condition, method, 50, 100)
            assert row["upstream_revision"] == revision
            missing_metadata = []
            for path, field in [("scripts/baselines/adajepa_run.py", "launcher_sha256"),
                                ("environments/adajepa/requirements.lock.txt", "requirements_sha256"),
                                ("data/baselines/adajepa/release/pusht_visual_shift/checkpoints/model_latest.pth", "checkpoint_sha256"),
                                ("data/baselines/adajepa/release/pushobj_eval/val_T/plan_targets.pkl", "goals_sha256"),
                                ("external/adajepa/conf/adajepa_plan_cem_pushobj.yaml", "config_source_sha256")]:
                if field not in row:
                    assert (condition, method) == ("clean", "frozen") and field in {"launcher_sha256", "requirements_sha256"}
                    digest(path)
                    missing_metadata.append(field)
                else:
                    digest(path, row[field])
            digest(folder / "logs.json")
            logs = [json.loads(line) for line in (folder / "logs.json").read_text().splitlines() if line.strip()]
            finals = [r for r in logs if "final_eval/success_rate" in r]
            assert len(finals) == 1 and finals[0] == row["final_metrics"]
            for filename in ("plan_targets.pkl", ".hydra/config.yaml", ".hydra/overrides.yaml", "plan.log", "process.log"):
                digest(folder / filename)
            rate = row["final_metrics"]["final_eval/success_rate"]
            assert np.isclose(rate * 50, round(rate * 50))
            stages.append({**row, "successes": round(rate * 50), "success_rate": rate,
                           "metadata_not_pinned_at_first_stage_start": missing_metadata,
                           "source_record": str(folder / "reproduction_record.json")})
    return {"scope": "Official released RGB+agent-proprioception PushT protocol; not matched to ShiftWM inputs, data, checkpoint, or planner budget.",
            "campaign": campaign, "stages": stages,
            "uncertainty": "One evaluation seed and aggregate success only: no paired per-task interval is claimed.",
            "tracked_upstream_source_clean": True,
            "first_stage_metadata_limitation": "Clean frozen began before launcher/runtime hashes and hardware fields were added to the run record. Its scientific checkpoint, goal set, official configuration and command are pinned; the supplemental GPU observation documents hardware. Current launcher/runtime hashes are recorded here but cannot retroactively establish their exact first-stage bytes.",
            "adaptive_minus_frozen_pp": {c: 100 * (next(r["success_rate"] for r in stages if r["condition"] == c and r["method"] == "adaptive") - next(r["success_rate"] for r in stages if r["condition"] == c and r["method"] == "frozen")) for c in ("clean", "blur")}}


def main():
    torch.set_num_threads(1)
    digest(__file__)
    source = ROOT / "scripts/extensions/summarize.py"
    spec = importlib.util.spec_from_file_location("frozen_summary", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest(source)
    original, groups, forecasts = read_campaign("configs/extensions/campaign.json", "results/extensions_v1")
    geometry, wide, wide_forecasts = read_campaign("configs/geometry_revision/campaign.json", "results/geometry_revision_v1", geometry=True)
    comparisons = []
    for domain in ("drone", "surgery"):
        for family in ("transformer", "gru"):
            ours = groups[(domain, family, "factorized")]
            for baseline in ("framewise", "constant_dynamics"):
                comparisons.append({"domain": domain, "architecture": family, "baseline": baseline,
                                    **pair(ours, groups[(domain, family, baseline)], module.paired_comparison)})
    geometric_pairs = []
    for label, a, b in [
        ("Wide ShiftWM minus wide constant", wide[("drone", "transformer", "factorized")], wide[("drone", "transformer", "constant_dynamics")]),
        ("Wide ShiftWM minus original ShiftWM", wide[("drone", "transformer", "factorized")], groups[("drone", "transformer", "factorized")]),
        ("Wide constant minus original constant", wide[("drone", "transformer", "constant_dynamics")], groups[("drone", "transformer", "constant_dynamics")])]:
        geometric_pairs.append({"comparison": label, **pair(a, b, module.paired_comparison)})
    references = []
    for domain in ("drone", "surgery"):
        for policy in ("random", "replay_oracle"):
            path = Path(f"results/extensions_v1/{domain}_{policy}/development/planning/results.json")
            result = verify_result(path, planning=True)
            references.append({"domain": domain, "policy": policy, "successes": result["successes"],
                               "tasks": result["tasks"], "support_successes": result["support_successes"], "source": str(path)})
    task_table = []
    task_ids = sorted(r["seed"] for r in wide[("drone", "transformer", "factorized")][0]["records"])
    for task in task_ids:
        row = {"task_seed": task}
        for prefix, group in (("original", groups), ("wide", wide)):
            for mode in ("constant_dynamics", "factorized"):
                row[f"{prefix}_{mode}"] = [int(next(r for r in group[("drone", "transformer", mode)][seed]["records"] if r["seed"] == task)["success"]) for seed in (0, 1, 2)]
        task_table.append(row)
    ada = official_adajepa()
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "verified_complete",
              "scope": "All simulator comparisons are prespecified development tasks; geometry is a subsequent development revision, not independent confirmation. AdaJEPA is a separate official reproduction.",
              "completion": {"original_full_30_epoch_runs": len(original), "original_forecasts": len(original),
                             "original_planning": len(original), "geometry_full_30_epoch_runs": len(geometry),
                             "geometry_forecasts": len(geometry), "geometry_planning": len(geometry),
                             "official_adajepa_stages": len(ada["stages"])},
              "original_runs": original, "original_planning_comparisons": comparisons,
              "geometry_runs": geometry, "geometry_planning_comparisons": geometric_pairs,
              "geometry_task_table": task_table, "reference_policies": references, "official_adajepa": ada,
              "mechanistic_scope": "Observed planning changes do not demonstrate that wider gains restored goal geometry or action ranking. The same 16-goal/120-pair representation diagnostic must be repeated on revised checkpoints before a mechanism claim.",
              "verification": {"all_training_completion_and_strict_load_checks": True,
                               "all_result_embedded_identities_equal_sidecars": True,
                               "all_recorded_source_protocol_checkpoint_and_trace_hashes_match": True,
                               "all_planning_counts_recomputed": True,
                               "all_paired_tasks_initial_metrics_and_support_prefixes_equal": True,
                               "all_forecast_means_recomputed": True,
                               "bootstrap": "Unmodified scripts/extensions/summarize.py paired_comparison; RNG417, 5000 crossed draws over 3 training seeds and 8 physical task seeds; unadjusted exploratory percentile95%. Repeated seeds are not24 independent physical tasks."},
              "source_sha256": dict(sorted(EVIDENCE.items()))}
    assert len(original) == 36 and len(geometry) == 6
    atomic_json(report, "reports/completed_extension_results.json")
    write_markdown(report)
    print(json.dumps({"completion": report["completion"], "verified_files": len(EVIDENCE),
                      "geometry_comparisons": [{k:v for k,v in r.items() if k != "paired_inventory"} for r in geometric_pairs]}, indent=2))


def write_markdown(report):
    lines = ["# Completed simulator extensions and official AdaJEPA reproduction", "", report["generated_at_utc"], "",
             "All 36 original simulator models completed 30 epochs, forecasting and closed-loop evaluation. The six observation-gain revision models also completed all three stages. All results below are included, including negative comparisons. Final-test outcomes are not inspected in this report.", "",
             "## Original simulator study", "",
             "Each table entry aggregates training seeds 0/1/2 on the same eight development tasks. Success totals therefore describe 24 matched task–seed instances, not 24 independent tasks. Forecast MSE is the mean across three training seeds on identical development windows, with lower values preferred.", "",
             "| Domain | Predictor | Method | Successes per seed /8 | Total /24 | Success | Forecast MSE@1 | MSE@3 | MSE@5 |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for domain in ("drone", "surgery"):
        for family in ("transformer", "gru"):
            for mode in ("framewise", "constant_dynamics", "factorized"):
                rows = sorted([r for r in report["original_runs"] if (r["domain"], r["architecture"], r["mode"]) == (domain, family, mode)], key=lambda r:r["training_seed"])
                total = sum(r["successes"] for r in rows)
                mses = [np.mean([r["forecast_mse"][f"h{h}"] for r in rows]) for h in (1, 3, 5)]
                lines.append(f"| {domain} | {family} | {LABELS[mode]} | " + "/".join(str(r["successes"]) for r in rows) + f" | {total}/24 | {100*total/24:.2f}% | " + " | ".join(f"{v:.6f}" for v in mses) + " |")
    lines += ["", "Paired planning effects are ShiftWM (ours) minus the named comparator; positive numbers favor ours. The unmodified registered summary function resamples matched training seeds and physical tasks in 5,000 crossed draws. These are unadjusted exploratory intervals with only eight physical tasks.", "",
              "| Domain | Predictor | Comparator | Difference, pp | 95% interval, pp | Ours-only / baseline-only |",
              "|---|---|---|---:|---|---:|"]
    for row in report["original_planning_comparisons"]:
        lo, hi = row["ci95_percentage_points"]
        d = row["discordances"]
        lines.append(f"| {row['domain']} | {row['architecture']} | {LABELS[row['baseline']]} | {row['success_difference_percentage_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {d['ours_only']} / {d['baseline_only']} |")
    lines += ["", "The original transformer gains on drone tasks are small (+4.17 pp); GRU drone results favor both baselines. Surgery transformer results favor both baselines, while surgery GRU gains are +4.17 pp. No original paired interval has a strictly positive lower bound. Lower forecast error alone does not establish better control. The GRU is our in-house recurrent predictor, not Dreamer.", "",
              "| Domain | Reference policy | Successes | Interpretation |", "|---|---|---:|---|"]
    for row in report["reference_policies"]:
        interpretation = "One common seeded random control policy" if row["policy"] == "random" else "Recorded-command replay reachability check; privileged reference, not a learned baseline"
        lines.append(f"| {row['domain']} | {row['policy']} | {row['successes']}/{row['tasks']} | {interpretation} |")
    lines += ["", "The drone random reference reaches 2/8 goals (25%); original drone transformer ShiftWM reaches 5/24 (20.83%). These denominators differ because random has no trained model seeds. The surgical task is SOFA tissue manipulation simulation, not a clinical or real surgical experiment.", "",
              "## Observation-gain revision: development ablation", "",
              "Only the observation FiLM gain function/range changes from [0.9,1.1] to [0.25,4], preserving identity value and first derivative at initialization. Translation, action adapter, parameter shapes, losses, data, validation selection and planner budget remain fixed. Higher-order curvature and optimization paths also change. No causal claim about recovered geometry follows from these counts.", "",
              "| Variant | Method | Successes seeds0/1/2 /8 | Total /24 | Success | Mean forecast MSE@5 |", "|---|---|---|---:|---:|---:|"]
    for variant, runs in (("Original narrow", report["original_runs"]), ("Revised wide", report["geometry_runs"])):
        for mode in ("constant_dynamics", "factorized"):
            rows = sorted([r for r in runs if (r["domain"],r["architecture"],r["mode"]) == ("drone","transformer",mode)], key=lambda r:r["training_seed"])
            total = sum(r["successes"] for r in rows)
            lines.append(f"| {variant} | {LABELS[mode]} | " + "/".join(str(r["successes"]) for r in rows) + f" | {total}/24 | {100*total/24:.2f}% | {np.mean([r['forecast_mse']['h5'] for r in rows]):.6f} |")
    lines += ["", "| Paired comparison | Difference, pp | 95% interval, pp | Per-seed difference, pp | Ours-only / comparator-only |", "|---|---:|---|---|---:|"]
    for row in report["geometry_planning_comparisons"]:
        lo,hi = row["ci95_percentage_points"]; d=row["discordances"]
        lines.append(f"| {row['comparison']} | {row['success_difference_percentage_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | " + ", ".join(f"{x:+.2f}" for x in row["per_training_seed_difference_percentage_points"]) + f" | {d['ours_only']} / {d['baseline_only']} |")
    lines += ["", "Task-level inventory follows. Each three-character string gives success(1)/failure(0) for training seeds 0,1,2 on the identical physical task, gain0.75 and warm appearance. Every comparison uses the same ten native support commands and total 200-command budget.", "",
              "| Physical task seed | Original constant | Wide constant | Original ShiftWM (ours) | Wide ShiftWM (ours) |", "|---:|---|---|---|---|"]
    for row in report["geometry_task_table"]:
        lines.append(f"| {row['task_seed']} | " + " | ".join("".join(map(str,row[k])) for k in ("original_constant_dynamics","wide_constant_dynamics","original_factorized","wide_factorized")) + " |")
    lines += ["", "The revision is a follow-up on the same development tasks. Its paired effects and any plotted increases remain exploratory. Repeat the fixed 16-goal/120-pair geometry diagnostic on these revised checkpoints before claiming reduced goal-distance compression, and evaluate independent tasks before claiming generalization. No new geometry mechanism measurement is included here.", "",
              "## Separate official AdaJEPA reproduction", "",
              "This table uses the authors' released PushT checkpoint, goal set, RGB plus agent proprioception and official CEM configuration. It is not input-, data-, checkpoint-, or budget-matched to ShiftWM, so it must not be used for a direct ranking against our simulator or RGB-only results.", "",
              "| Condition | Official method | Successes | Success rate | Elapsed seconds |", "|---|---|---:|---:|---:|"]
    for row in report["official_adajepa"]["stages"]:
        lines.append(f"| {row['condition']} | AdaJEPA {row['method']} | {row['successes']}/50 | {100*row['success_rate']:.0f}% | {row['elapsed_seconds']:.1f} |")
    lines += ["", "Within the official protocol, adaptation increases aggregate success by 24 pp on clean observations and 20 pp under blur. There is one evaluation seed (100); aggregate logs do not support a paired per-task interval. Official CEM uses horizon25, 200 candidates, 30 elites, ten optimization iterations, at most20 replans and five grouped actions per replan (frameskip5). These numbers reproduce the published implementation locally; they do not establish state of the art.", "",
              "The clean frozen stage started before launcher/runtime hashes and hardware fields were added to the run record. Its checkpoint, goals, official configuration and scientific command were pinned; a supplemental GPU observation documents hardware. Later stages include the added fields. Current launcher/runtime hashes cannot retroactively prove their first-stage bytes.", "",
              "## Audit and reusable artifacts", "",
              f"The machine-readable report pins {len(report['source_sha256'])} input artifacts by SHA256. All 42 training runs pass their existing completion validator, including 30 chronological epochs, complete optimizer/model packages, validation-best selection and strict model loading. Result identities match their sidecars; recorded code, protocol, manifest, checkpoint and trace hashes match current files. Planning counts are recomputed, saved task records agree with summary records, and paired initial states/support prefixes match. Forecast means are recomputed from all recorded windows. AdaJEPA's pinned upstream checkout is clean, and all checkpoint, goal set, launcher, runtime and configuration hashes present in its records match; the first-stage metadata gap is disclosed above.", "",
              "The original 36 portable local checkpoint packages remain in `artifacts/releases/extensions_v1`; the six geometry checkpoints remain in their separate `runs/geometry_revision` namespace and require the version2 geometry loader. Nothing is externally published by this report. This report does not summarize the separate ongoing real DROID model study.", "",
              "- [Machine-readable results and hash ledger](completed_extension_results.json)",
              "- [Original development protocol](domain_extension_protocol.md)",
              "- [Geometry revision protocol](geometry_revision_protocol.md)",
              "- [Report generator](../scripts/extensions/report_completed_results.py)",
              "- [Drone failure-mode audit](drone_failure_modes.md)", ""]
    Path("reports/completed_extension_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
