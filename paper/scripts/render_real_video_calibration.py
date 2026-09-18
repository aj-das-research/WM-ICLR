#!/usr/bin/env python3
"""Generate the clearly labeled validation-development calibration table."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics",
         "factorized": r"\textbf{ShiftWM (ours)}", "action_free": "Action-free"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gain(value):
    text = f"{value:+.2f}"
    return r"\positivegain{" + text + "}" if value > 0 else text


def generate():
    source = ROOT / "reports/real_droid_residual_calibration_results.json"
    report = json.loads(source.read_text())
    if report["status"] != "completed" or report["test_evaluated"] or len(report["individual"]) != 12:
        raise ValueError("Require complete train-fitted/validation-only calibration report")
    expected = {(mode, seed) for mode in NAMES for seed in (0, 1, 2)}
    if {(row["mode"], row["seed"]) for row in report["individual"]} != expected:
        raise ValueError("Incomplete matched method/seed population")
    lookup = {(row["mode"], row["horizon"]): (index, row) for index, row in enumerate(report["grouped"])}
    table = [r"% Generated from the completed full-training calibration campaign; VALIDATION ONLY.",
             r"\begin{table}[t]\centering\footnotesize",
             r"\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular}{lrrrrrr}\toprule",
             r"& \multicolumn{3}{c}{Validation: 5 blocks} & \multicolumn{3}{c}{Validation: 10 blocks}\\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Method & Original & Calibrated & Gain (\%) & Original & Calibrated & Gain (\%)\\\midrule"]
    cells = []
    for mode in NAMES:
        values = []
        for horizon in (5, 10):
            index, row = lookup[(mode, horizon)]
            independent = 100 * (1 - row["calibrated"] / row["base"])
            if abs(independent - row["relative_error_reduction_percent"]) > 1e-10:
                raise ValueError("Gain differs from source errors")
            individuals = [item for item in report["individual"] if item["mode"] == mode]
            for version, key in (("base", "base"), ("calibrated", "calibrated")):
                mean = sum(item[f"h{horizon}_{version}"] for item in individuals) / 3
                if abs(mean - row[key]) > 1e-12:
                    raise ValueError("Three-seed mean differs from complete source rows")
            values.extend([f"{row['base']:.6f}", f"{row['calibrated']:.6f}", gain(independent)])
            cells.append({"method": mode, "horizon": horizon, "source_location": f"grouped[{index}]",
                          "original": row["base"], "calibrated": row["calibrated"], "relative_reduction_percent": independent})
        if mode == "factorized":
            table.append(r"\rowcolor{orange!9}")
        table.append(NAMES[mode] + " & " + " & ".join(values) + r"\\")
    table += [r"\bottomrule\end{tabular}"]
    fair = {row["horizon"]: row for row in report["comparisons"] if row["reference"] == "framewise"}
    h5, h10 = fair[5], fair[10]
    intervals = {h: f"${row['mean_difference']:+.6f}\\;[{row['ci95'][0]:+.6f},{row['ci95'][1]:+.6f}]$"
                 for h, row in fair.items()}
    caption = (r"\caption{\textbf{Validation development: matched residual calibration, not a new test.} "
               r"Train-standardized feature MSE, lower is better; all four methods and three seeds are included. "
               r"One scalar per frozen best checkpoint is fitted on 830 eligible training episodes (7,721 windows), "
               r"with no validation targets used for fitting. Each Gain column is the reduction from that row's "
               r"own uncalibrated model. \positivegain{Bold green} marks positive point reductions, not significance. "
               f"Against also-calibrated Framewise, ours improves {h5['relative_reduction_percent']:.2f}\\% at five "
               f"blocks and {h10['relative_reduction_percent']:.2f}\\% at ten; the paired MSE differences with "
               f"95\\% session/seed bootstrap intervals are {intervals[5]} and {intervals[10]}, respectively. "
               r"The latter includes zero. These are exploratory validation intervals, unadjusted for multiple "
               r"comparisons; the original held-out results remain unchanged.}")
    table += [caption, r"\label{tab:real-video-calibration-development}\end{table}", ""]
    destination = ROOT / "paper/generated/real_video/residual_calibration.tex"
    destination.write_text("\n".join(table))
    ledger = {"status": "numeric_checks_passed_visual_review_pending", "split": "validation",
              "source": str(source.relative_to(ROOT)), "source_sha256": sha(source),
              "generator_sha256": sha(__file__), "table_sha256": sha(destination),
              "cells": cells, "fair_comparisons": list(fair.values()),
              "scope": "Full 12-model train-fitted calibration development control; no new test claim",
              "validation_population": report["populations"],
              "uncertainty": "10,000 session-cluster draws crossed with three training seeds; exploratory validation; unadjusted",
              "green_bold_semantics": "positive point reduction relative to the same uncalibrated method, not significance"}
    destination.with_suffix(".sources.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"status": "generated_and_numerically_checked", "table": str(destination), "visual_review": "pending root manuscript build"}))


if __name__ == "__main__":
    generate()
