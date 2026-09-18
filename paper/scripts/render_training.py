#!/usr/bin/env python3
"""Render logged epochs and verified completion state without predicting results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_jsonl(path):
    if not path.exists():
        return [], None
    raw = path.read_bytes()
    lines = raw.splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                break  # Concurrent writer has not finished the last record.
            raise
        if not math.isfinite(float(row["val"]["prediction_loss"])):
            raise ValueError(f"Nonfinite validation loss in {path}")
        if records and int(row["epoch"]) <= int(records[-1]["epoch"]):
            raise ValueError(f"Duplicate or nonmonotonic epochs in {path}")
        records.append(row)
    return records, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("configs/world/full_campaign.json"))
    parser.add_argument("--output", type=Path, default=Path("paper/generated"))
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    runs = []
    for task in campaign["tasks"]:
        config_file = Path(task["config"])
        config = json.loads(config_file.read_text())
        root = Path(config["output_dir"])
        metrics_file = root / "metrics.jsonl"
        records, digest = read_jsonl(metrics_file)
        summary_file = root / "training_summary.json"
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        last_epoch = int(records[-1]["epoch"]) if records else 0
        complete = (summary.get("status") == "completed"
                    and int(summary.get("completed_epochs", 0)) == int(config["epochs"])
                    and last_epoch == int(config["epochs"])
                    and (root / "last" / "model.pt").is_file())
        state = "complete" if complete else ("interrupted" if summary.get("status") == "interrupted_checkpoint_saved"
                    else "epochs logged; unfinished" if records else "awaiting epoch logs")
        environment = Path(config["pretrained_dir"]).name
        runs.append({"run_id": task["id"], "environment": environment, "mode": config["model"]["mode"],
                     "seed": config["seed"], "planned_epochs": config["epochs"],
                     "last_logged_epoch": last_epoch, "status": state,
                     "config_file": str(config_file.resolve()),
                     "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
                     "metrics_file": str(metrics_file.resolve()), "metrics_sha256": digest,
                     "epochs": records})
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = {"schema_version": 1, "kind": "training_progress_only", "runs": runs,
              "notice": "Validation prediction loss is not planning success. Unfinished runs are not final results."}
    (args.output / "training_ledger.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    completed = sum(run["status"] == "complete" for run in runs)
    started = sum(run["last_logged_epoch"] > 0 for run in runs)
    text = (f"The configured campaign contains {len(runs)} runs. At this artifact snapshot, "
            f"{started} have recorded at least one complete epoch and {completed} have a "
            "verified completion summary and final checkpoint. Missing epoch logs do not "
            "establish whether a scheduler job is queued or running. Training curves show "
            "observed validation prediction loss only; they do not establish closed-loop performance.\n")
    (args.output / "training_status.tex").write_text(text)
    method_names = {"plain": "Unaligned predictor (diagnostic)", "framewise": "Framewise calibration",
                    "single": "Shared context", "factorized_unpaired": "Unpaired contexts",
                    "factorized": "ShiftWM (ours)"}
    table = [r"\begin{longtable}{lllrl}", r"\toprule Environment & Mode & Seed & Epochs & State\\\midrule\endhead"]
    for run in runs:
        state = run["status"].replace("epochs logged; unfinished", "unfinished")
        mode = method_names.get(run["mode"], run["mode"]).replace("_", r"\_")
        table.append(f"{run['environment']} & {mode} & {run['seed']} & {run['last_logged_epoch']}/{run['planned_epochs']} & {state}" + r"\\")
    table.append(r"\bottomrule\end{longtable}")
    (args.output / "training_runs.tex").write_text("\n".join(table) + "\n")
    plotted = [run for run in runs if run["epochs"]]
    if plotted:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import numpy as np
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                             "axes.labelsize": 8.5, "axes.titlesize": 9,
                             "axes.edgecolor": "#647080", "axes.labelcolor": "#243447",
                             "text.color": "#243447", "xtick.color": "#243447",
                             "ytick.color": "#243447", "axes.linewidth": .7,
                             "pdf.fonttype": 42, "svg.fonttype": "none"})
        environments = sorted({r["environment"] for r in runs})
        fig, axes = plt.subplots(1, len(environments), figsize=(5.5, 3.35), squeeze=False)
        colors = {"plain": "#555555", "framewise": "#16876B", "single": "#0072B2",
                  "factorized_unpaired": "#8B62A3", "factorized": "#B75B16"}
        labels = {"plain": "Unaligned predictor (diagnostic)", "framewise": "Framewise calibration", "single": "Shared context",
                  "factorized_unpaired": "Unpaired contexts", "factorized": "ShiftWM (ours)"}
        styles = {"plain": ":", "framewise": "--", "single": "-.",
                  "factorized_unpaired": (0, (5, 2, 1, 2)), "factorized": "-"}
        markers = {"plain": "x", "framewise": "o", "single": "s",
                   "factorized_unpaired": "D", "factorized": "P"}
        plot_records = []
        for panel, (axis, environment) in enumerate(zip(axes.flat, environments)):
            available = [r for r in plotted if r["environment"] == environment]
            for method_index, (mode, color) in enumerate(colors.items()):
                selected = [r for r in available if r['mode'] == mode]
                # Every plotted point uses all configured seeds, without extrapolating a missing epoch.
                expected = [r for r in runs if r['environment'] == environment and r['mode'] == mode]
                if len(selected) != len(expected) or len(selected) < 2:
                    continue
                if len({r['seed'] for r in selected}) != len(selected):
                    raise ValueError('Duplicate training seeds in plotted group')
                by_epoch = [{r['epoch']: r['val']['prediction_loss'] for r in run['epochs']} for run in selected]
                epochs = sorted(set.intersection(*(set(r) for r in by_epoch)))
                values = np.asarray([[r[epoch] for epoch in epochs] for r in by_epoch])
                means, deviations = values.mean(axis=0), values.std(axis=0, ddof=1)
                axis.fill_between(epochs, means - deviations, means + deviations, color=color, alpha=.12, linewidth=0)
                axis.plot(epochs, means, color=color, linestyle=styles[mode],
                          linewidth=1.7 if mode == "factorized" else 1.25,
                          marker=markers[mode], markersize=3.6 if mode == "factorized" else 3.0,
                          markeredgewidth=.65, markerfacecolor="white" if mode != "factorized" else color,
                          markevery=(method_index, 6) if len(epochs) > 1 else 1, zorder=4 if mode == "factorized" else 3)
                plot_records.append({'environment': environment, 'mode': mode,
                    'training_seeds': [r['seed'] for r in selected], 'epochs': epochs,
                    'mean': means.tolist(), 'sample_sd': deviations.tolist()})
            axis.set(xlabel="Epoch")
            axis.set_title(f"{chr(97 + panel)}  {'PushT' if environment == 'pusht' else 'Reacher'}",
                           loc='left', fontweight='bold', pad=9)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis='y', color='#E4E8ED', linewidth=.6)
            axis.set_axisbelow(True)
            axis.set_xticks([1, 10, 20, 30])
            axis.tick_params(length=3, width=.7)
            if not available:
                axis.text(.5, .5, "No complete epoch logged", transform=axis.transAxes, ha="center", fontsize=8)
        legend_order = ("factorized", "framewise", "single", "factorized_unpaired", "plain")
        legends = [Line2D([], [], color=colors[mode], linestyle=styles[mode], linewidth=1.5,
                          marker=markers[mode], markersize=3.7, markeredgewidth=.7,
                          markerfacecolor=colors[mode] if mode == "factorized" else "white", label=labels[mode])
                   for mode in legend_order]
        legend = fig.legend(handles=legends, frameon=False, fontsize=8, loc="lower center", ncol=2,
                            bbox_to_anchor=(.51, .003), columnspacing=1.1, handlelength=2.1, labelspacing=.55)
        legend.get_texts()[0].set_fontweight("bold")
        fig.text(.014, .653, "Validation prediction MSE ↓", rotation=90, va="center", fontsize=8.5)
        fig.text(.13, .975, "Mean ± 1 SD across 3 training seeds", va="top", fontsize=8, color="#465365")
        fig.subplots_adjust(left=.13, right=.985, bottom=.355, top=.85, wspace=.48)
        (args.output / 'training_plot.json').write_text(json.dumps({
            'metric': 'observed validation prediction MSE',
            'uncertainty': 'mean plus/minus sample standard deviation over all configured training seeds at common logged epochs; not a confidence interval',
            'source_ledger': 'training_ledger.json',
            'source_sha256': hashlib.sha256((args.output / 'training_ledger.json').read_bytes()).hexdigest(),
            'curves': plot_records}, indent=2, allow_nan=False) + '\n')
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(args.output / f"training_curves.{suffix}", dpi=240)
        plt.close(fig)
    else:
        # Never retain stale curves when switching to a campaign with no evidence.
        for suffix in ("pdf", "svg", "png"):
            (args.output / f"training_curves.{suffix}").unlink(missing_ok=True)
    print(json.dumps({"runs": len(runs), "logged_runs": started, "completed_runs": completed}))


if __name__ == "__main__":
    main()
