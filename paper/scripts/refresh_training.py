#!/usr/bin/env python3
"""Refresh the manuscript when training, configuration, or evaluation evidence changes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def fingerprint(root):
    paths = [root / "configs/world/full_campaign.json"]
    paths += sorted((root / "configs/world").glob("*.json"))
    paths += sorted((root / "runs/world").glob("*/metrics.jsonl"))
    paths += sorted((root / "runs/world").glob("*/training_summary.json"))
    paths += sorted(path for path in (root / "results/world").glob("*/*.json")
                    if ".progress." not in path.name)
    paths += sorted((root / "results/development_official_budget").glob("*/planning_development.json"))
    paths += sorted((root / "results/development_goal_intervention").glob("*/planning_development.json"))
    paths += sorted((root / "configs/dynamics_revision").glob("*.json"))
    paths += sorted((root / "runs/dynamics_revision").glob("*/training_summary.json"))
    paths += sorted((root / "results/development_dynamics_revision").glob("*/planning_development.json"))
    paths += sorted((root / "configs/rollout_revision").glob("*.json"))
    paths += sorted((root / "runs/rollout_revision").glob("*/training_summary.json"))
    paths += sorted((root / "results/development_rollout_revision").glob("*/planning_development.json"))
    paths += sorted((root / "results/qualitative_diagnostics").glob("*_diagnostics.json"))
    # Source changes should rebuild the draft even between experiment completions.
    paths += [path for path in (root / "paper/main.tex", root / "paper/references.bib",
                               root / "scripts/summarize_goal_intervention.py",
                               root / "scripts/evaluate_goal_calibration_intervention.py",
                               root / "scripts/summarize_paired_planning.py",
                               root / "scripts/train_dynamics_revision.py",
                               root / "scripts/evaluate_dynamics_revision.py",
                               root / "src/shiftwm/dynamics_revision.py",
                               root / "scripts/train_rollout_revision.py",
                               root / "scripts/evaluate_rollout_revision.py",
                               root / "src/shiftwm/rollout_revision.py",
                               root / "reports/rollout_revision_protocol.md",
                               root / "reports/evidence/upstream_planning_results.json") if path.is_file()]
    paths += sorted((root / "paper/figures").glob("*.tex"))
    paths += sorted((root / "paper/tables").glob("*.tex"))
    paths += sorted((root / "paper/sections").glob("*.tex"))
    paths += [root / "scripts/aggregate_results.py"]
    paths += sorted(path for directory in ("assets", "split_assets")
                    for path in (root / "paper/figures" / directory).rglob("*")
                    if path.is_file() and path.suffix in (".png", ".json", ".py", ".txt"))
    paths += sorted((root / "paper/scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=120)
    parser.add_argument("--max-hours", type=float, default=120)
    args = parser.parse_args()
    if args.interval < 10:
        raise ValueError("Refresh interval must be at least ten seconds")
    root = Path(__file__).resolve().parents[2]
    previous = None
    deadline = time.monotonic() + args.max_hours * 3600
    while True:
        try:
            current = fingerprint(root)
            if current != previous:
                commands = [[sys.executable, "paper/scripts/render_training.py"],
                                [sys.executable, "scripts/aggregate_results.py"],
                                [sys.executable, "paper/scripts/render_official_development.py"]]
                if (root / "scripts/summarize_goal_intervention.py").is_file():
                    commands.append([sys.executable, "scripts/summarize_goal_intervention.py"])
                if (root / "scripts/summarize_paired_planning.py").is_file():
                    commands.append([sys.executable, "scripts/summarize_paired_planning.py"])
                    commands.append([sys.executable, "paper/scripts/render_planning_comparison.py"])
                if (root / "paper/scripts/render_planning_controls.py").is_file():
                    commands.append([sys.executable, "paper/scripts/render_planning_controls.py"])
                if (root / "paper/scripts/render_dynamics_revision.py").is_file():
                    commands.append([sys.executable, "paper/scripts/render_dynamics_revision.py"])
                if (root / "paper/scripts/render_rollout_revision.py").is_file():
                    commands.append([sys.executable, "paper/scripts/render_rollout_revision.py"])
                if (root / "paper/scripts/render_qualitative.py").is_file():
                    commands.append([sys.executable, "paper/scripts/render_qualitative.py", "--if-needed"])
                if (root / "paper/scripts/render_positive_qualitative.py").is_file():
                    commands.append([sys.executable, "paper/scripts/render_positive_qualitative.py", "--if-needed"])
                if (root / "paper/scripts/render_technical_contract.py").is_file():
                    commands.append([sys.executable, "paper/scripts/render_technical_contract.py", "--if-needed"])
                diagnostic_inputs = [root / "results/qualitative_diagnostics" / name
                                     for name in ("replay_diagnostics.json", "prediction_diagnostics.json")]
                if (root / "paper/scripts/render_technical_qualitative.py").is_file() and all(
                        path.is_file() and json.loads(path.read_text()).get("status") == "complete"
                        for path in diagnostic_inputs):
                    commands.append([sys.executable, "paper/scripts/render_technical_qualitative.py", "--if-needed"])
                # Each renderer validates completed records before drawing quantitative marks.
                if (root / "paper/scripts/render_forecast.py").is_file():
                    commands.append([sys.executable, "paper/scripts/render_forecast.py"])
                    if (root / "paper/scripts/render_gain_summary.py").is_file():
                        commands.append([sys.executable, "paper/scripts/render_gain_summary.py"])
                    commands.append([sys.executable, "paper/scripts/render_teaser.py"])
                if (root / "paper/generated/goal_calibration.json").is_file():
                    commands.append([sys.executable, "paper/scripts/render_goal_calibration.py"])
                commands.append(["bash", "paper/build.sh"])
                for command in commands:
                    subprocess.run(command, cwd=root, check=True)
                previous = current
                print(json.dumps({"event": "paper_refreshed", "fingerprint": current, "time_unix": time.time()}), flush=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(json.dumps({"event": "refresh_failed", "error": str(error), "time_unix": time.time()}), flush=True)
            if not args.watch:
                raise
        if not args.watch or time.monotonic() >= deadline:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
