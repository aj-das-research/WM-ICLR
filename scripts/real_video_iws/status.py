#!/usr/bin/env python3
"""Report live scheduler/checkpoint progress without promoting partial scores."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/real_video_iws"


def refresh():
    now = datetime.now(timezone.utc).isoformat()
    registry_path = ROOT / "configs/real_video_iws/training_registration_v1.json"
    registry = json.loads(registry_path.read_text())
    queue = subprocess.run(["squeue", "--array", "--user", "abhijit.das", "--noheader",
                            "--format=%i|%j|%T|%M|%N|%R"], capture_output=True, text=True, check=True)
    jobs = []
    for line in queue.stdout.splitlines():
        fields = line.strip().split("|", 5)
        if len(fields) != 6:
            raise ValueError("Unexpected scheduler output")
        jobs.append(dict(zip(("id", "name", "state", "elapsed", "nodes", "reason"), fields)))
    runs = []
    for row in registry["runs"]:
        folder = ROOT / row["output"]
        summary = folder / "training_summary.json"
        value = json.loads(summary.read_text()) if summary.exists() else {}
        epochs = value.get("completed_epochs", 0)
        evaluation = REPORT / "evaluations" / (row["name"] + ".json")
        runs.append({**row, "checkpointed_epochs": epochs,
                     "full_training_summary_present": epochs == 30 and value.get("status") == "completed",
                     "evaluation_receipt_present": evaluation.exists()})
    training_jobs = [j for j in jobs if j["name"] == "iws-single-observation"]
    active = [j for j in jobs if j["name"].startswith("iws-")]
    counts = {state: sum(j["state"] == state for j in active) for state in ("RUNNING", "PENDING")}
    training_counts = {state: sum(j["state"] == state for j in training_jobs)
                       for state in ("RUNNING", "PENDING")}
    value = {"schema": "shiftwm_iws_live_progress_v1", "checked_utc": now,
             "registration_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
             "scheduler_jobs": jobs, "campaign_jobs": counts,
             "training_jobs": training_counts, "runs": runs,
             "full_training_summaries": sum(r["full_training_summary_present"] for r in runs),
             "evaluation_receipts": sum(r["evaluation_receipt_present"] for r in runs),
             "complete_study_finalizer_present": (REPORT / "development_finalization.json").exists(),
             "scope": "Progress only. Presence/counts do not replace the scientific completion validator; no partial losses become benchmark results."}
    ablation_registration = ROOT / "configs/real_video_iws_unbounded/registration_v1.json"
    ablation = {"registration_present": ablation_registration.exists(), "runs": [],
                "scope": "Exploratory removal of the innovation bound; separate from the completed v1 study."}
    if ablation_registration.exists():
        next_registry = json.loads(ablation_registration.read_text())
        for row in next_registry["runs"]:
            summary = ROOT / row["output"] / "training_summary.json"
            progress = json.loads(summary.read_text()) if summary.exists() else {}
            ablation["runs"].append({"name": row["name"], "task": row["task"], "seed": row["seed"],
                                     "checkpointed_epochs": progress.get("completed_epochs", 0),
                                     "training_complete": progress.get("status") == "completed" and progress.get("completed_epochs") == 30})
        ablation["registration_sha256"] = hashlib.sha256(ablation_registration.read_bytes()).hexdigest()
    value["unbounded_ablation"] = ablation
    REPORT.mkdir(parents=True, exist_ok=True)
    target = REPORT / "live_status.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=REPORT, prefix=".live-status-", suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(target)
    # Presence is a progress observation, not validation. The renderer verifies
    # the immutable finalizer and frozen reporting sources before accepting it.
    figure = ROOT / "paper/scripts/render_iws_results.py"
    finalized = False
    if value["complete_study_finalizer_present"] and figure.exists():
        subprocess.run([str(ROOT / ".venv/bin/python"), str(figure), "--if-ready"], cwd=ROOT, check=True)
        subprocess.run([str(ROOT / ".venv/bin/python"),
                        str(ROOT / "paper/scripts/render_iws_results_v2.py"), "--if-ready"], cwd=ROOT, check=True)
        subprocess.run([str(ROOT / ".venv/bin/python"),
                        str(ROOT / "paper/scripts/render_iws_main_summary.py"), "--if-ready"], cwd=ROOT, check=True)
        alignment = json.loads((ROOT / "paper/generated/experiment_alignment/status.json").read_text())
        finalized = alignment["iws_development"]["status"] == "complete_validated_development"
    component_reporter = ROOT / "paper/scripts/render_iws_unbounded_results.py"
    if ablation_registration.exists() and component_reporter.exists():
        subprocess.run([str(ROOT / ".venv/bin/python"), str(component_reporter), "--if-ready"],
                       cwd=ROOT, check=True)
    if finalized:
        stage = "All 27 models and the complete development comparison have passed the scientific finalizer."
        iws_finding = "All 27 development evaluations validated; complete signed comparisons are in the paper; official validation remains reserved"
    elif value["full_training_summaries"] == 27:
        stage = ("All 27 training runs are complete. Complete-study evaluation validation is pending. "
                 "The common-CPU recovery preserves the original receipts and unchanged selected checkpoints; "
                 "no partial comparison is reported as a benchmark result.")
        iws_finding = "All 27 models trained; common-CPU evaluation/finalization pending"
    else:
        stage = "Full training and development evaluation are in progress."
        iws_finding = "Predictor comparisons pending complete training and validated evaluation"
    rows = ["| Task | Autoregressive | Additive anchor | ShiftWM (ours) |",
            "|---|---|---|---|"]
    for task in ("pusht", "bimanual_box", "bimanual_rope"):
        cells = []
        for mode in ("autoregressive", "anchored_additive", "bounded_spatial_mix"):
            selected = sorted((r for r in runs if r["task"] == task and r["mode"] == mode), key=lambda r: r["seed"])
            cells.append(", ".join(f"s{r['seed']}: {r['checkpointed_epochs']}/30" for r in selected))
        rows.append("| " + " | ".join([task, *cells]) + " |")
    text = f"""# Current results and GPU status

Checked **{now}** from the live scheduler and checkpoint summaries.

**IWS jobs: {counts['RUNNING']} running, {counts['PENDING']} queued. The v1 study has 27 registered runs.**
V1 training jobs: {training_counts['RUNNING']} running, {training_counts['PENDING']} queued.
{stage}

The account permits two ws-ia jobs plus up to one GPU on the GPU partition.
Evaluation recovery uses CPU-only allocations to keep numerical reduction
behavior consistent across every comparison; scheduler job names distinguish
these from training. Full H60 training uses all 59 future offsets for 30 epochs, with effective batch 64 and
three seeds for each task/method. The frozen recipe and all reviewed scientific
sources are checked at launch and at every epoch.

The first task is real IWS PushT: one actual image plus recorded native commands
predict future DINOv2 visual features. Box and Rope use the same method/recipe
with their native command widths. These are feature forecasts, not RGB videos
or measured physical robot success. Official validation remains reserved.

## Training progress (checkpointed epochs, not benchmark scores)

""" + "\n".join(rows) + f"""

{value['full_training_summaries']}/27 full-training summaries and
{value['evaluation_receipts']}/27 development evaluation receipts are present.
The finalizer independently checks every model, primitive window error,
trajectory, seed and source hash before numerical paper ingestion. Intermediate
training losses are not substituted for completed comparisons.

## Completed results retained in the paper

| Study | Verified completion | Finding and scope |
|---|---|---|
| Current spatial ShiftWM on DROID | 21 models ×30 epochs;32 method contrasts +4 interaction effects | 3.01% h5 /5.30% h10 relative standardized-MSE reduction vs matched AR;136/141 episodes improve at h10; development evidence |
| Spatial components | Six additional follow-up models included above | Mixing has the clearest native h10 benefit; incremental bounding/context effects are inconclusive |
| Historical DROID/context, capacity and h10 studies | Completed in their separate finalizers | Different architectures/protocols; positive and negative findings retained |
| Historical simulation and domain extensions | Completed separate studies | Mixed forecast/planning findings; not final spatial-model cross-domain evidence |
| Real IWS PushT/Box/Rope | 1,804 recordings /360,473 frames cached; {value['full_training_summaries']}/27 full training summaries | {iws_finding} |
| External SOTA comparison | Reproduction work remains incomplete | No current evidence of SOTA superiority |

Training profiles and scheduler limits are in `reports/real_video_iws/`.
The GPU partition has an eight-hour per-job limit. The original training
projection is historical capacity planning, not the remaining evaluation time.
Original GPU receipts and operational recovery provenance are retained under
`reports/real_video_iws/recovery/`; the frozen evaluator and its tolerances
are unchanged.

The paper now states one current proposed decoder, the distinct DROID/IWS
interfaces, the primary feature-error metric, matched internal controls and
pending external comparisons. See `reports/experiment_alignment_2026-09-19.md`.
Source-bound completed DROID evidence remains in
`reports/real_video_spatial/finalization.json` and
`reports/real_video_spatial_components/finalization.json`.
"""
    if ablation["runs"]:
        text += "\n## Controlled innovation-bound ablation\n\nNine separately registered full 30-epoch runs retain the fixed-source mixer and remove only the innovation bound. These are exploratory development follow-ups; their scores do not replace completed v1 results.\n\n"
        text += "| Task | Seed | Checkpointed epochs |\n|---|---:|---:|\n"
        text += "\n".join(f"| {r['task']} | {r['seed']} | {r['checkpointed_epochs']}/30 |" for r in ablation["runs"]) + "\n"
        text += "\nSource, profile, registration, completion and all outcome records live in `reports/real_video_iws_unbounded/`.\n"
    elif (ROOT / "configs/real_video_iws_unbounded/registration_candidate_v1.json").exists():
        text += "\n## Next controlled ablation\n\nThe unbounded-innovation mixer is undergoing full-shape profiling and review before a separate nine-run registration. Candidate status does not imply training has started or that any gain has been measured.\n"
    (ROOT / "reports/current_results_and_gpu_status.md").write_text(text)
    print(json.dumps({"checked_utc": now, "running": counts["RUNNING"], "pending": counts["PENDING"],
                      "full_training_summaries": value["full_training_summaries"],
                      "evaluation_receipts": value["evaluation_receipts"]}, sort_keys=True))
    return value


if __name__ == "__main__":
    refresh()
