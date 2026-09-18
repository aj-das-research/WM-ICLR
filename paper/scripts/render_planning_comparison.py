#!/usr/bin/env python3
"""Plot paired planning contrasts from the validated reporting ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
COMPARATORS = (("framewise", "Framewise calibration"),
               ("single", "Shared context"),
               ("factorized_unpaired", "Unpaired contexts"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_rows(ledger):
    """Preserve the three-seed gate and verify plotted units and arithmetic."""
    rows = {}
    for row in ledger["results"]:
        if row["metric"] != "raw_success":
            continue
        key = (row["environment"], row["split"], row["comparator"])
        if key in rows:
            raise ValueError(f"Duplicate planning contrast: {key}")
        if row["comparison"] != "factorized_minus_" + row["comparator"]:
            raise ValueError("Planning contrast direction changed")
        if row["required_training_seeds"] != [0, 1, 2]:
            raise ValueError("Planning contrast requires three prescribed training seeds")
        metrics = row["metrics"]
        if row["status"] == "pending":
            if metrics is not None:
                raise ValueError("Pending contrasts must not contain numerical estimates")
        elif row["status"] == "complete":
            if any(row["availability"][mode].get(str(seed)) != "complete"
                   for mode in ("factorized", row["comparator"]) for seed in (0, 1, 2)):
                raise ValueError("Incomplete sources cannot produce a plotted contrast")
            values = [metrics["mean_difference_pp"], *metrics["conditional_ci95_pp"]]
            if len(values) != 3 or not all(math.isfinite(x) and -100 <= x <= 100 for x in values):
                raise ValueError("Invalid percentage-point estimate or interval")
            if values[1] > values[2]:
                raise ValueError("Reversed planning interval")
            per_seed = metrics["per_training_seed"]
            if sorted(x["training_seed"] for x in per_seed) != [0, 1, 2]:
                raise ValueError("Missing or duplicate training seeds")
            differences = []
            for seed in per_seed:
                n = seed["n"]
                if n <= 0 or seed["wins"] + seed["losses"] + seed["both_success"] + seed["neither_success"] != n:
                    raise ValueError("Invalid paired outcome population")
                value = 100 * (seed["wins"] - seed["losses"]) / n
                if not math.isclose(value, seed["paired_difference_pp"], abs_tol=1e-10):
                    raise ValueError("Paired estimate disagrees with wins/losses")
                differences.append(value)
            if not math.isclose(statistics.mean(differences), values[0], abs_tol=1e-10):
                raise ValueError("Mean planning difference disagrees with seed estimates")
        else:
            raise ValueError("Unknown planning completion status")
        rows[key] = row
    expected = {(env, split, mode) for env in ("pusht", "reacher")
                for split in ("test", "extrapolation") for mode, _ in COMPARATORS}
    if set(rows) != expected:
        raise ValueError("Missing or unexpected planning contrasts")
    return rows


def validate_sources(ledger, root):
    for field, path in (("reporter_sha256", "scripts/summarize_paired_planning.py"),
                        ("validator_sha256", "scripts/aggregate_results.py")):
        if digest(root / path) != ledger[field]:
            raise ValueError(f"Regenerate paired ledger after changing {path}")
    for path, source in ledger["evaluation_sources"].items():
        if digest(root / path) != source["sha256"]:
            raise ValueError(f"Changed planning evidence: {path}")
    for source in ledger["data_sources"].values():
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Changed planning data manifest")


def render(rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "shiftwm-paired-planning"})
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.45), sharex=True, sharey=True)
    fig.subplots_adjust(left=.285, right=.98, top=.80, bottom=.18, hspace=.72, wspace=.20)
    extents = [abs(value) for row in rows.values() if row["status"] == "complete"
               for value in [row["metrics"]["mean_difference_pp"], *row["metrics"]["conditional_ci95_pp"]]]
    bound = max(5, math.ceil(max(extents, default=5) * 1.15 / 5) * 5)
    accent, ink = "#B75B16", "#243447"
    for i, (env, env_label) in enumerate((("pusht", "PushT"), ("reacher", "Reacher"))):
        for j, split in enumerate(("test", "extrapolation")):
            ax = axes[i, j]
            ax.axvline(0, color="#89929C", linewidth=.8, linestyle=(0, (3, 3)), zorder=1)
            for y, (mode, _) in enumerate(COMPARATORS):
                row = rows[env, split, mode]
                if row["status"] == "complete":
                    metric = row["metrics"]
                    lo, hi = metric["conditional_ci95_pp"]
                    ax.hlines(y, lo, hi, color=accent, linewidth=1.8, zorder=3)
                    ax.vlines([lo, hi], y-.075, y+.075, color=accent, linewidth=1.0, zorder=3)
                    ax.scatter(metric["mean_difference_pp"], y, marker="D", s=24,
                               color=accent, edgecolors="white", linewidths=.5, zorder=4)
                    ax.text(.97, y - .22,
                            f"{metric['mean_difference_pp']:+.1f} pp".replace("-", "−"),
                            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                            fontsize=7.5, fontweight="bold", color=accent)
                else:
                    # Axes-relative text is not a zero-valued measurement.
                    ax.text(.03, y, "pending", transform=ax.get_yaxis_transform(),
                            va="center", color="#727A83", fontsize=7.5,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
            ax.set(yticks=range(3), yticklabels=[label for _, label in COMPARATORS],
                   xlim=(-bound, bound), ylim=(2.55, -.55))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10], integer=True))
            ax.tick_params(axis="both", length=0, labelsize=8, pad=5)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#CBD1D8")
            ax.grid(axis="x", color="#E9EDF1", linewidth=.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title("Held-out composition" if j == 0 else "Extrapolation",
                             fontsize=8.5, color=ink, pad=9)
        axes[i, 0].text(-.41, 1.12, env_label, transform=axes[i, 0].transAxes,
                        fontsize=9, fontweight="bold", color=ink)
    fig.text(.04, .965, "Does better prediction translate into better planning?",
             fontsize=9.1, fontweight="bold", color=ink, va="top")
    fig.text(.04, .907, "ShiftWM (ours) minus each matched control", fontsize=8.2, color=ink)
    fig.text(.63, .075, "Success difference (percentage points)", ha="center", fontsize=8)
    fig.text(.63, .025, "Positive favors ours  ·  95% paired task-cluster intervals",
             ha="center", fontsize=7.4, color="#505D6B")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix("."+suffix), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/evidence/paired_planning_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/generated/planning_comparison")
    args = parser.parse_args()
    ledger = json.loads(args.input.read_text())
    validate_sources(ledger, ROOT)
    rows = select_rows(ledger)
    render(rows, args.output)
    record = {"source": str(args.input.relative_to(ROOT)), "source_sha256": digest(args.input),
              "renderer_sha256": digest(__file__), "width_inches": 5.5,
              "metric": "raw_success_difference_percentage_points",
              "uncertainty": ledger["uncertainty"], "results": list(rows.values()),
              "complete": sum(row["status"] == "complete" for row in rows.values()),
              "total": len(rows), "missing_policy": "Pending text only; no numerical mark."}
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"planning_contrasts_plotted": record["complete"], "total": len(rows)}))


if __name__ == "__main__":
    main()
