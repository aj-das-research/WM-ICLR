#!/usr/bin/env python3
"""Render explicit, source-linked comparison deltas without ranking partial results."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
COMPARATORS = (("framewise", "Framewise calibration"),
               ("factorized_unpaired", "Unpaired contexts"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_script(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forecast_deltas(ledger):
    """Ratios of complete three-seed means; never per-seed percentage averages."""
    rows = {}
    for row in ledger["results"]:
        key = row["environment"], row["split"], row["mode"]
        if key in rows:
            raise ValueError("Duplicate forecast comparison row")
        rows[key] = row
    result = []
    for environment in ("pusht", "reacher"):
        for split in ("test", "extrapolation"):
            for comparator, label in COMPARATORS:
                chosen = [rows[environment, split, mode] for mode in ("factorized", comparator)]
                for row in chosen:
                    if (row["training_seeds"] != [0, 1, 2]
                            or set(row["per_seed_values"]) != {"0", "1", "2"}):
                        raise ValueError("Forecast gains require all three prescribed training seeds")
                    values = list(row["per_seed_values"].values())
                    if (row["metric"] != "fixed_reference_latent_mse_h5"
                            or row["direction"] != "lower"
                            or not all(math.isfinite(x) and x >= 0 for x in values)
                            or not math.isclose(statistics.mean(values), row["value"], rel_tol=1e-11, abs_tol=1e-12)):
                        raise ValueError("Forecast gain input differs from its metric or seed means")
                ours, reference = chosen
                if reference["value"] <= 0:
                    raise ValueError("Relative forecast reduction is undefined for a zero reference")
                result.append({"environment": environment, "split": split,
                    "method": "factorized", "method_label": "ShiftWM (ours)",
                    "comparator": comparator, "comparator_label": label,
                    "status": "complete", "ours_value": ours["value"],
                    "comparator_value": reference["value"],
                    "relative_error_reduction_percent": 100 * (reference["value"] - ours["value"]) / reference["value"],
                    "training_seeds": [0, 1, 2],
                    "source_ids": {"ours": ours["source_ids"], "comparator": reference["source_ids"]}})
    return result


def planning_deltas(rows, primary):
    """Use validated paired intervals and cross-check displayed primary means."""
    lookup = {(row["environment"], row["mode"], row["metric"]): row for row in primary["results"]}
    result = []
    for environment in ("pusht", "reacher"):
        for split in ("test", "extrapolation"):
            source = rows[environment, split, "single"]
            row = {"environment": environment, "split": split, "method": "factorized",
                   "comparator": "single", "status": source["status"],
                   "mean_difference_pp": None, "conditional_ci95_pp": None}
            if source["status"] == "complete":
                metric = "heldout_success" if split == "test" else "extrapolation_success"
                ours, reference = [lookup[environment, mode, metric] for mode in ("factorized", "single")]
                if any(x["status"] != "complete" or x["available_seeds"] != [0, 1, 2]
                       for x in (ours, reference)):
                    raise ValueError("Planning gain cannot accompany partial primary means")
                value = source["metrics"]["mean_difference_pp"]
                if not math.isclose(ours["mean"] - reference["mean"], value, abs_tol=1e-10):
                    raise ValueError("Paired gain differs from the displayed primary comparison")
                row.update(mean_difference_pp=value,
                           conditional_ci95_pp=source["metrics"]["conditional_ci95_pp"],
                           training_seeds=[0, 1, 2],
                           interval_scope="conditional_on_observed_training_seeds; paired_initial_state_clusters")
            result.append(row)
    return result


def build_summary(root):
    root = Path(root)
    paths = {"forecast": root / "paper/generated/forecast_comparison.json",
             "planning": root / "reports/evidence/paired_planning_results.json",
             "primary": root / "paper/generated/primary_results.json"}
    ledgers = {key: json.loads(path.read_text()) for key, path in paths.items()}
    # Reuse the established scientific validators rather than introduce a
    # weaker independent route from unverified point estimates into the paper.
    forecast_report = load_script("render_forecast")
    fresh = forecast_report.build_ledger(root)
    for key in ("status", "sources", "results", "primary_ledger_sha256", "source_script_sha256"):
        if fresh[key] != ledgers["forecast"][key]:
            raise ValueError(f"Regenerate the forecast ledger before gain reporting: {key}")
    planning_report = load_script("render_planning_comparison")
    planning_report.validate_sources(ledgers["planning"], root)
    paired_rows = planning_report.select_rows(ledgers["planning"])
    return {"schema_version": 1, "status": "source_validated",
        "renderer_sha256": digest(__file__),
        "source_ledgers": {key: {"path": str(path.relative_to(root)), "sha256": digest(path)}
                           for key, path in paths.items()},
        "forecast": forecast_deltas(ledgers["forecast"]),
        "planning": planning_deltas(paired_rows, ledgers["primary"]),
        "definitions": {
            "forecast": "100 * (comparator mean MSE - ShiftWM mean MSE) / comparator mean MSE; positive means lower error; ratio of three-seed means, not mean of per-seed percentages.",
            "planning": "ShiftWM raw success minus Shared context raw success in percentage points; intervals are the existing conditional 95% paired initial-state-cluster intervals, not seed SD."},
        "limits": ["No significance, state-of-the-art, or overall-best claim.",
                   "Comparators are explicitly named; no ranking over incomplete planning columns.",
                   "Unaligned predictor is a diagnostic in the full table and has lower PushT held-out forecast error than ShiftWM.",
                   "Negative forecast reductions and planning differences are retained."]}


def format_gain(value):
    """Highlight a favorable displayed mean, without implying significance."""
    text = f"{value:+.2f}"
    return r"\positivegain{" + text + "}" if float(text) > 0 else text


def render_planning_rows(summary):
    lookup = {(row["environment"], row["split"]): row for row in summary["planning"]}
    values, intervals = [], []
    for environment in ("pusht", "reacher"):
        for split in ("test", "extrapolation"):
            row = lookup[environment, split]
            if row["status"] != "complete":
                values.append(r"\missing"); intervals.append(r"\missing")
                continue
            values.append(format_gain(row["mean_difference_pp"]))
            lo, hi = row["conditional_ci95_pp"]
            intervals.append(r"{\scriptsize " + f"[{lo:+.2f}, {hi:+.2f}]" + "}")
    return "\n".join([r"% Source-linked paired differences; units differ from success percentages above.",
        r"\def\ShiftWMPlanningGainRows{",
        r"\midrule", r"{\footnotesize $\Delta$ vs Shared context (pp)} & " + " & ".join(values) + r"\\",
        r"{\footnotesize 95\% paired CI} & " + " & ".join(intervals) + r"\\", "}"]) + "\n"


def render_forecast_rows(summary):
    lookup = {(row["environment"], row["split"], row["comparator"]): row for row in summary["forecast"]}
    lines = [r"% Positive relative reduction means lower MSE; negative means regression.",
             r"\begingroup\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.1}",
             r"\begin{tabular}{@{}lrrrr@{}}",
             r"\multicolumn{5}{@{}l}{\textbf{Relative MSE reduction (\%) $\uparrow$}}\\[3pt]\toprule",
             r"& \multicolumn{2}{c}{PushT} & \multicolumn{2}{c}{Reacher}\\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Reference & Held-out & Extrap. & Held-out & Extrap.\\\midrule"]
    for comparator, label in COMPARATORS:
        cells = [format_gain(lookup[environment, split, comparator]["relative_error_reduction_percent"])
                 for environment in ("pusht", "reacher") for split in ("test", "extrapolation")]
        lines.append(" & ".join([label, *cells]) + r"\\")
    lines += [r"\bottomrule\end{tabular}\endgroup"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = build_summary(args.root)
    output = args.root / "paper/generated"
    (output / "primary_gain_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "planning_gain_rows.tex").write_text(render_planning_rows(summary))
    (output / "forecast_gain_summary.tex").write_text(render_forecast_rows(summary))
    print(json.dumps({"forecast_deltas": len(summary["forecast"]),
                      "planning_deltas_complete": sum(row["status"] == "complete" for row in summary["planning"])}))


if __name__ == "__main__":
    main()
