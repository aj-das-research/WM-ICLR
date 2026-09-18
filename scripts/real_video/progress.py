#!/usr/bin/env python3
"""Write a dated, factual progress report without treating partial runs as results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from shiftwm.real_video.data import atomic_json


def refresh():
    campaign = json.loads(Path("configs/real_video/campaign.json").read_text())
    rows = []
    for run in campaign["runs"]:
        config = json.loads(Path(run["config"]).read_text())
        path = Path(config["output_dir"])/"training_summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {}
        root = Path("results/real_video/droid_selected_v1")/run["name"]
        results = sorted(root.glob("*/h*/results.json"))
        finished = [str(path) for path in results if json.loads(path.read_text()).get("status") == "completed"]
        rows.append({"name":run["name"],"mode":run["mode"],"seed":run["seed"],
                     "completed_epochs":summary.get("completed_epochs",0),
                     "full_training_complete":summary.get("status") == "completed" and summary.get("completed_epochs") == 30,
                     "evaluation_populations_complete":len(finished),"evaluation_files":finished,
                     "checkpoint":str(Path(config["output_dir"])/"best") if (Path(config["output_dir"])/"best").exists() else None})
    report = {"time_utc":datetime.now(timezone.utc).isoformat(),"dataset":"1,126 actual recorded DROID episodes",
              "scope":"prespecified subset, not complete benchmark or closed-loop real-robot results",
              "completed_training":sum(row["full_training_complete"] for row in rows),
              "planned_training":12,"completed_evaluations":sum(row["evaluation_populations_complete"] for row in rows),
              "planned_evaluations":48,"rows":rows}
    atomic_json(report,"reports/real_video_progress.json")
    lines = ["# Real recorded-video progress","",report["time_utc"],"",
             f"Full 30-epoch runs: {report['completed_training']}/12; evaluation populations: {report['completed_evaluations']}/48.","",
             "DROID selection: 1,126 real robot episodes; 851 train / 143 validation / 132 test, separated by recording session.","",
             "| Method | Seed | Epochs completed | Full training | Evaluations |","|---|---:|---:|---|---:|"]
    labels={"factorized":"ShiftWM real-video variant (ours)","framewise":"Framewise","constant_dynamics":"Constant dynamics","action_free":"Action-free"}
    for row in rows:
        lines.append(f"| {labels[row['mode']]} | {row['seed']} | {row['completed_epochs']}/30 | {'complete' if row['full_training_complete'] else 'pending/in progress'} | {row['evaluation_populations_complete']}/4 |")
    lines += ["","Completed epochs are optimization progress, not evidence of superiority. Test comparisons require the complete matched campaign."]
    destination=Path("reports/real_video_progress.md")
    temporary=destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines)+"\n");temporary.replace(destination)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--watch",action="store_true")
    parser.add_argument("--hours",type=float,default=12);args=parser.parse_args()
    deadline=time.monotonic()+args.hours*3600
    while True:
        report=refresh()
        print(json.dumps({key:report[key] for key in ("time_utc","completed_training","completed_evaluations")}),flush=True)
        if not args.watch or report['completed_evaluations']==48 or time.monotonic()>=deadline:break
        time.sleep(120)
