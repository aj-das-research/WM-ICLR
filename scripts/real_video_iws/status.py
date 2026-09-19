#!/usr/bin/env python3
"""Report live scheduler/checkpoint progress without promoting partial scores."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

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
    active = [j for j in jobs if j["name"] == "iws-single-observation"]
    counts = {state: sum(j["state"] == state for j in active) for state in ("RUNNING", "PENDING")}
    value = {"schema": "shiftwm_iws_live_progress_v1", "checked_utc": now,
             "registration_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
             "scheduler_jobs": jobs, "campaign_jobs": counts, "runs": runs,
             "full_training_summaries": sum(r["full_training_summary_present"] for r in runs),
             "evaluation_receipts": sum(r["evaluation_receipt_present"] for r in runs),
             "complete_study_finalizer_present": (REPORT / "development_finalization.json").exists(),
             "scope": "Progress only. Presence/counts do not replace the scientific completion validator; no partial losses become benchmark results."}
    REPORT.mkdir(parents=True, exist_ok=True)
    target = REPORT / "live_status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
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

**IWS full training: {counts['RUNNING']} jobs running, {counts['PENDING']} queued; 27 registered runs.**
The account permits two ws-ia GPU jobs plus one GPU-partition GPU. Full H60
training uses all59 future offsets for30epochs, with effective batch64 and
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
| Real IWS PushT/Box/Rope | 1,804 recordings /360,473 frames cached; training progress above | Predictor comparisons pending validated completion |
| External SOTA comparison | Reproduction work remains incomplete | No current evidence of SOTA superiority |

The measured micro64 workload is approximately20.4 GPU-hours for27 runs including
per-epoch validation, or6.8hours at three continuously available GPUs before
loading, checkpointing and final evaluation. This is a resource projection,
not a guaranteed finish time. Full-shape profiles and the scheduler limits are
in `reports/real_video_iws/`. The GPU partition has an eight-hour per-job limit;
its array uses7h50m allocations and preserves epoch checkpoints.

The paper now states one current proposed decoder, the distinct DROID/IWS
interfaces, the primary feature-error metric, matched internal controls and
pending external comparisons. See `reports/experiment_alignment_2026-09-19.md`.
Source-bound completed DROID evidence remains in
`reports/real_video_spatial/finalization.json` and
`reports/real_video_spatial_components/finalization.json`.
"""
    (ROOT / "reports/current_results_and_gpu_status.md").write_text(text)
    figure = ROOT / "paper/scripts/render_iws_results.py"
    if value["complete_study_finalizer_present"] and figure.exists():
        subprocess.run([str(ROOT / ".venv/bin/python"), str(figure), "--if-ready"], cwd=ROOT, check=True)
    print(json.dumps({"checked_utc": now, "running": counts["RUNNING"], "pending": counts["PENDING"],
                      "full_training_summaries": value["full_training_summaries"],
                      "evaluation_receipts": value["evaluation_receipts"]}, sort_keys=True))
    return value


if __name__ == "__main__":
    refresh()
