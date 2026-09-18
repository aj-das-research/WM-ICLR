#!/usr/bin/env python3
"""Compose an illustrative task asset with a source-linked forecast comparison.

No inference, checkpoint selection, smoothing, or image generation occurs here.
Run render_forecast.py first to produce the independently validated raw ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
INK, MUTED = "#243447", "#657589"
BLUE, VIOLET = "#0072B2", "#7751A6"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evidence():
    source = ROOT / "paper/generated/forecast_comparison.json"
    data = json.loads(source.read_text())
    if data["status"] != "source_validated":
        raise ValueError("Forecast ledger must be fully source validated")
    if digest(ROOT / "paper/scripts/render_forecast.py") != data["source_script_sha256"]:
        raise ValueError("Regenerate the forecast ledger after renderer changes")
    if digest(ROOT / data["primary_ledger"]) != data["primary_ledger_sha256"]:
        raise ValueError("Regenerate the forecast ledger after result changes")
    for path, info in data["sources"].items():
        if digest(ROOT / path) != info["sha256"]:
            raise ValueError(f"Forecast source changed: {path}")
    lookup = {(r["environment"], r["split"], r["mode"]): r for r in data["results"]}
    rows = []
    for env in ("pusht", "reacher"):
        for split in ("test", "extrapolation"):
            proposed = lookup[env, split, "factorized"]
            baseline = lookup[env, split, "framewise"]
            for item in (proposed, baseline):
                if item["metric"] != "fixed_reference_latent_mse_h5" or item["training_seeds"] != [0, 1, 2]:
                    raise ValueError("Expected matched three-seed MSE@5 summaries")
                mean = sum(item["per_seed_values"].values()) / 3
                if not math.isclose(mean, item["value"], rel_tol=1e-10):
                    raise ValueError("Mean does not match the three source values")
            change = 100 * (proposed["value"] / baseline["value"] - 1)
            rows.append({"environment": env, "split": split, "percent_error_change": change,
                         "paired_mean": proposed["value"], "framewise_mean": baseline["value"],
                         "source_result_ids": [proposed["id"], baseline["id"]],
                         "source_ids": sorted(set(proposed["source_ids"] + baseline["source_ids"]))})
    return source, rows


def main():
    source, rows = evidence()
    asset = ROOT / "paper/figures/assets/world_concept.png"
    asset_meta = json.loads((asset.parent / "concept_asset.json").read_text())
    if digest(asset) != asset_meta["sha256"]:
        raise ValueError("Illustrative asset changed without provenance update")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "text.color": INK, "axes.labelcolor": INK,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "world_teaser", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(5.5, 2.18), facecolor="white")
    # The saved transparent illustration is displayed unchanged over white.
    scene = fig.add_axes([.00, .075, .405, .81])
    scene.imshow(plt.imread(asset))
    scene.set_axis_off()
    fig.text(.015, .93, "a  What changed?", fontsize=9, weight="bold")
    scene.annotate("Appearance", xy=(.345, .76), xytext=(.68, .98),
                   xycoords="axes fraction", textcoords="axes fraction", color=BLUE,
                   fontsize=8, ha="center", va="top",
                   arrowprops={"arrowstyle": "-", "color": BLUE, "lw": .8,
                               "connectionstyle": "angle,angleA=-90,angleB=0,rad=5"})
    scene.annotate("Dynamics", xy=(.44, .42), xytext=(.035, .13),
                   xycoords="axes fraction", textcoords="axes fraction", color="#B75B16",
                   fontsize=8, ha="left", va="center",
                   arrowprops={"arrowstyle": "-", "color": "#B75B16", "lw": .8,
                               "connectionstyle": "angle,angleA=0,angleB=-90,rad=5"})
    fig.text(.015, .095, "Concept illustration", fontsize=7.8, color=MUTED)
    fig.text(.47, .93, "b  ShiftWM (ours)", fontsize=9, weight="bold")
    fig.text(.47, .825, "vs. Framewise calibration", fontsize=8, color=MUTED)
    ax = fig.add_axes([.55, .295, .395, .465])
    ax.axvline(0, color=MUTED, linewidth=.85, zorder=1)
    ax.set_xlim(-32, 11)
    ax.set_ylim(-.48, 1.48)
    ax.set_yticks([1, 0], ["PushT", "Reacher"])
    ax.tick_params(axis="y", length=0, pad=7, labelcolor=INK)
    ax.set_xticks([-30, -20, -10, 0, 10])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    ax.tick_params(axis="x", labelsize=8, length=0, pad=5, colors=MUTED)
    ax.grid(axis="x", color="#E9EDF2", linewidth=.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    value_labels = []
    for row in rows:
        held = row["split"] == "test"
        y = (1 if row["environment"] == "pusht" else 0) + (.16 if held else -.16)
        value = row["percent_error_change"]
        color, marker = (BLUE, "o") if held else (VIOLET, "^")
        ax.hlines(y, min(value, 0), max(value, 0), color=color, linewidth=2.0, zorder=2)
        ax.plot(value, y, marker=marker, color=color, markersize=5, zorder=3)
        value_labels.append(ax.annotate(
            f"{value:+.1f}%", (value, y), xytext=(-5 if value < 0 else 5, 0),
            textcoords="offset points", ha="right" if value < 0 else "left",
            va="center", fontsize=8, color=color))
    ax.set_xlabel("Forecast error change (%)  ↓", labelpad=6, fontsize=8)
    fig.legend([Line2D([], [], color=BLUE, marker="o", lw=0, markersize=5),
                Line2D([], [], color=VIOLET, marker="^", lw=0, markersize=5)],
               ["Held-out pair", "Extrapolation"], loc="lower right",
               bbox_to_anchor=(.98, .01), ncol=2, frameon=False,
               handletextpad=.3, columnspacing=.85, fontsize=8)
    # Extreme negative values can place their left-hand label over a row name.
    # Use rendered extents at the actual 5.5-inch size; keep the number/mark fixed
    # and move only a colliding label above/right with a 2-point clearance guard.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row_boxes = [text.get_window_extent(renderer).padded(2 * fig.dpi / 72)
                 for text in ax.get_yticklabels()]
    for label in value_labels:
        if any(label.get_window_extent(renderer).overlaps(box) for box in row_boxes):
            label.set_position((6, 6))
            label.set_ha("left")
            label.set_va("bottom")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if any(label.get_window_extent(renderer).overlaps(box)
           for label in value_labels for box in row_boxes):
        raise ValueError("A value label still overlaps an environment label; revise the figure layout")
    output = ROOT / "paper/generated/world_teaser"
    for ext in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300}
        if ext == "svg":
            kwargs["metadata"] = {"Date": None}
        elif ext == "pdf":
            kwargs["metadata"] = {"Creator": "paper/scripts/render_teaser.py", "CreationDate": None, "ModDate": None}
        fig.savefig(output.with_suffix("." + ext), **kwargs)
    plt.close(fig)
    ledger = {"status": "source_linked", "width_inches": 5.5, "height_inches": 2.18,
              "source": str(source.relative_to(ROOT)), "source_sha256": digest(source),
              "script_sha256": digest(__file__), "asset_sha256": digest(asset),
              "asset_status": asset_meta["status"], "asset_placement": "left concept panel; unchanged RGBA composited over white",
              "raster_export_dpi": 300,
              "rows": rows, "formula": "100 * (mean paired MSE@5 / mean framewise MSE@5 - 1)",
              "aggregation": "ratio of three-training-seed means; no uncertainty estimated for this ratio",
              "limits": ["Concept art is not a simulator frame, prediction, or result.",
                         "Only pointwise RGB changes were evaluated, not camera viewpoint changes.",
                         "Forecast evidence, not planning success; unfavorable extrapolation is retained.",
                         "Full raw metric and per-seed variability appear in forecast_comparison."]}
    output.with_suffix(".json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": ledger["status"], "comparisons": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
