#!/usr/bin/env python3
"""Render every frozen fresh-DROID endpoint, including negative comparisons."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/real_droid_fresh_evaluation_results.json"
VERIFICATION = ROOT / "reports/real_droid_fresh_evaluation_verification.json"
DESTINATION = ROOT / "paper/generated/real_video"
MODES = ("framewise", "constant_dynamics", "factorized", "action_free")
LABELS = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics",
          "factorized": r"\textbf{ShiftWM (ours)}", "action_free": "Action-free"}
POPULATIONS = (("exterior_image_1_left", 5), ("exterior_image_1_left", 10),
               ("exterior_image_2_left", 5), ("exterior_image_2_left", 10))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(population, variant):
    metric = f"h{population['horizon']}_standardized_mse"
    rows = [row for row in population["paired_comparisons"]
            if row["method"] == variant + "/factorized" and row["reference"] == variant + "/framewise"
            and row["metric"] == metric]
    if len(rows) != 1:
        raise ValueError("Missing or duplicated matched comparison")
    row = rows[0]
    ours = population["methods"][variant + "/factorized"][metric]["mean"]
    reference = population["methods"][variant + "/framewise"][metric]["mean"]
    if not math.isclose(ours - reference, row["mean_difference"], rel_tol=0, abs_tol=1e-14):
        raise ValueError("Paired comparison and reported endpoint differ")
    return row


def gain(value):
    formatted = f"{value:+.3f}"
    return r"\positivegain{" + formatted + "}" if value > 0 else formatted


def main():
    report, verification = json.loads(REPORT.read_text()), json.loads(VERIFICATION.read_text())
    if report["status"] != "completed" or verification["status"] != "passed" or verification["result_sha256"] != sha256(REPORT):
        raise ValueError("Require independently verified completed fresh results")
    if len(report["sources"]) != 96:
        raise ValueError("Require all 96 frozen evaluations")
    for path, expected in report["sources"].items():
        if sha256(path) != expected:
            raise ValueError("Result source changed: " + path)
    populations = [report["populations"][f"{camera}/h{horizon}"] for camera, horizon in POPULATIONS]
    for population in populations:
        if set(population["methods"]) != {f"{variant}/{mode}" for variant in ("original", "calibrated") for mode in MODES}:
            raise ValueError("Incomplete matched methods")
        for metrics in population["methods"].values():
            for metric in metrics.values():
                if len(metric["per_seed"]) != 3:
                    raise ValueError("Incomplete three-seed estimate")
    lines = [r"% Generated from verified reports/real_droid_fresh_evaluation_results.json; never edit numbers manually.",
             r"\begin{table}[t]\centering\small", r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}",
             r"\begin{tabular}{lrrrr}\toprule",
             r"& \multicolumn{2}{c}{External camera 1} & \multicolumn{2}{c}{External camera 2}\\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Method & 5 blocks & 10 blocks & 5 blocks & 10 blocks\\\midrule"]
    for variant, heading in (("original", "Original selected checkpoints"), ("calibrated", "Train-fitted residual calibration")):
        lines.append(r"\multicolumn{5}{l}{\emph{" + heading + r"}}\\")
        for mode in MODES:
            if mode == "factorized":
                lines.append(r"\rowcolor{orange!9}")
            values = [population["methods"][variant + "/" + mode][f"h{population['horizon']}_standardized_mse"]["mean"] for population in populations]
            lines.append(LABELS[mode] + " & " + " & ".join(f"{value:.6f}" for value in values) + r"\\")
        lines.append(r"\midrule")
    for baseline, name in (("persistence", "Persistence"), ("constant_velocity", "Constant velocity")):
        values = [population["fixed_support_baselines"][baseline][f"h{population['horizon']}_standardized_mse"] for population in populations]
        lines.append(name + " & " + " & ".join(f"{value:.6f}" for value in values) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{\textbf{Fresh real-DROID sessions: complete original and calibrated comparison.} Final-horizon training-standardized feature MSE ($\downarrow$), averaged equally over episodes and three training seeds. Every method uses its original validation-selected checkpoint; the second block applies its own scalar fitted on original training data only. The two support-only baselines are unchanged across variants. Five-block evaluation uses all 65 episodes from 52 new site/date sessions (866 windows); ten-block evaluation uses 64 eligible episodes from the same 52 sessions (801 windows). The one short episode remains in the audited manifest. Calibrated camera-1/five-block is the prespecified primary population; other populations are separate secondary analyses. These are feature forecasts of actual recorded observations, not physical control outcomes.}",
              r"\label{tab:fresh-real-video-complete}\end{table}", ""]
    complete = DESTINATION / "fresh_confirmatory_complete.tex"
    complete.write_text("\n".join(lines))
    lines = [r"% All original and calibrated ours-minus-Framewise comparisons, including regressions.",
             r"\begin{table}[t]\centering\small", r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}",
             r"\begin{tabular}{lrr}\toprule",
             r"Population & Gain (\%) $\uparrow$ & Paired MSE difference [95\% CI]\\\midrule"]
    for variant, heading in (("original", "Original selected checkpoints"), ("calibrated", "Train-fitted residual calibration")):
        lines.append(r"\multicolumn{3}{l}{\emph{" + heading + r"}}\\")
        for index, population in enumerate(populations):
            row = pair(population, variant)
            camera = 1 if population["camera"] == POPULATIONS[0][0] else 2
            name = f"Camera {camera}, {population['horizon']} blocks"
            if variant == "calibrated" and index == 0:
                name = r"\textbf{" + name + " (primary)}"
                lines.append(r"\rowcolor{orange!9}")
            lo, hi = row["ci95"]
            estimate = f"${row['mean_difference']:+.6f}\\;[{lo:+.6f},{hi:+.6f}]$"
            lines.append(name + " & " + gain(row["relative_reduction_percent"]) + " & " + estimate + r"\\")
        if variant == "original":
            lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{\textbf{Fresh-session gains and uncertainty versus matched Framewise.} ShiftWM (ours) is compared with Framewise within the same original or calibrated variant. Gains are relative error reductions, not percentage points; \positivegain{bold green} denotes a positive point estimate, not statistical significance. Intervals use 10,000 paired session-cluster bootstrap draws crossed with training-seed resampling. Only the calibrated camera-1/five-block comparison is confirmatory. Its interval excludes zero; both calibrated ten-block intervals cross zero. All secondary intervals are descriptive and unadjusted for multiplicity. The uncalibrated ten-block regressions are retained explicitly.}",
              r"\label{tab:fresh-real-video-paired}\end{table}", ""]
    paired = DESTINATION / "fresh_confirmatory_paired.tex"
    paired.write_text("\n".join(lines))
    evidence = {"status": "complete", "result_sha256": sha256(REPORT), "independent_verification_sha256": sha256(VERIFICATION),
                "source_sha256": sha256(__file__), "evaluation_freeze_sha256": report["evaluation_freeze_sha256"],
                "primary_comparison": report["primary_comparison"],
                "source_evaluations": len(report["sources"]), "population_order": POPULATIONS,
                "all_original_and_calibrated_modes_retained": True, "original_h10_regressions_retained": True,
                "numeric_tables": {str(path.relative_to(ROOT)): sha256(path) for path in (complete, paired)},
                "render_review": "not compiled by this generator; root performs final integrated build and visual review"}
    (DESTINATION / "fresh_confirmatory.sources.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "tables": list(evidence["numeric_tables"]), "primary": report["primary_comparison"]}))


if __name__ == "__main__":
    main()
