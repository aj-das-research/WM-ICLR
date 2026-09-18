#!/usr/bin/env python3
"""Source-validated, fixed-width latent forecasting point-range figure.

This script only reads completed evaluation/training artifacts. It never runs
inference, chooses checkpoints, changes results, or reads planning outcomes.
"""
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from shiftwm.data import split_combinations, split_family, validate_manifest


ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    ("frozen", "Frozen LeWM", "#696B70", "X"),
    ("framewise", "Framewise calibration", "#16876B", "o"),
    ("single", "Shared context", "#0072B2", "s"),
    ("factorized_unpaired", "Unpaired contexts", "#8B62A3", "D"),
    ("factorized", "ShiftWM (ours)", "#B75B16", "P"),
)
CAPTION = (
    "Forecast accuracy under held-out composition and extrapolation. Points show "
    "MSE at the fifth autoregressive latent transition (25 native actions after "
    "the three-observation support), using recorded future actions and a fixed "
    "canonical visual reference; lower is better. Held-out composition is "
    "appearance2/dynamics2. Extrapolation averages the three prespecified "
    "conditions (appearance3,dynamics0), (appearance0,dynamics3), and "
    "(appearance3,dynamics3). Points and horizontal ranges are mean and sample "
    "standard deviation across three independently trained seeds; frozen LeWM "
    "has one checkpoint and no estimated training-seed spread. Each condition "
    "has 64 initial-state seeds and 768 overlapping trajectory windows per "
    "checkpoint. Horizontal axes use logarithmic scales with panel-specific "
    "limits. Factorization improves the held-out comparison but does not improve "
    "every extrapolation comparison; the frozen model has the lowest Reacher "
    "extrapolation error among these methods. These are forecasting results, "
    "not planning-success measurements or confidence intervals. Signed delta "
    "annotations are relative MSE changes versus Framewise calibration, computed "
    "from ratios of three-seed means; negative is lower error, with no inferred "
    "interval or significance. The unaligned "
    "plain diagnostic is retained in the results table and omitted here because "
    "it does not calibrate observed/goal features; it has lower PushT held-out "
    "forecast error than ShiftWM (ours)."
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12)


def expected_keys(manifest, split):
    """Exact complete window population; matches TrajectoryDataset H8/stride5."""
    return {(episode["trajectory_id"], start, episode["seed"], appearance, dynamics)
            for episode in manifest["episodes"] if episode["split"] == split_family(split)
            for appearance, dynamics in split_combinations(split)
            if dynamics == episode["dynamics_id"]
            for start in range(0, episode["steps"] + 2 - 8, 5)}


def validate_forecast(record, environment, mode, seed, split, sources, keys):
    headers = {"status": "complete", "environment": environment, "model_mode": mode,
               "training_seed": None if mode == "frozen" else seed,
               "checkpoint_sha256": sources["checkpoint_sha256"],
               "data_manifest_sha256": sources["data_manifest_sha256"],
               "evaluator_sha256": sources["evaluator_sha256"]}
    if any(record.get(key) != value for key, value in headers.items()):
        raise ValueError(f"Incomplete, mislabelled or changed forecast sources: {environment}/{mode}/{seed}/{split}")
    forecast = record["forecast"]
    contract = {"kind": "fixed_reference_forecasting", "split": split,
                "sequence_length": 8, "window_stride": 5}
    if any(forecast.get(key) != value for key, value in contract.items()):
        raise ValueError("Forecast metric/window protocol differs")
    rows = forecast["records"]
    actual_keys = [(row["trajectory_id"], row["start"], row["seed"],
                    row["observation_id"], row["dynamics_id"]) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != keys:
        raise ValueError("Forecast has duplicated, missing or substituted trajectory windows")
    groups = {"all": rows}
    for row in rows:
        if not math.isfinite(float(row["mse_h5"])) or row["mse_h5"] < 0:
            raise ValueError("Forecast MSE must be finite and nonnegative")
        groups.setdefault(f"o{row['observation_id']}_d{row['dynamics_id']}", []).append(row)
    if set(groups) != set(forecast["summary"]):
        raise ValueError("Forecast summary conditions differ from the actual records")
    for condition, group in groups.items():
        summary = forecast["summary"][condition]["mse_h5"]
        if not close(statistics.mean(row["mse_h5"] for row in group), summary["mean"]):
            raise ValueError("Forecast summary mean differs from per-window records")
        if summary["observations"] != len(group) or summary["clusters"] != len({row["seed"] for row in group}):
            raise ValueError("Forecast uncertainty population differs from actual records")
    return forecast


def build_ledger(root=ROOT):
    root = Path(root)
    primary_path = root / "paper/generated/primary_results.json"
    primary = read(primary_path)
    source_rows = {(row["environment"], row["mode"], row["metric"]): row for row in primary["results"]}
    evaluator_sha = digest(root / "src/shiftwm/evaluate.py")
    sources, plotted, components = {}, [], []
    for environment in ("pusht", "reacher"):
        data_name = "pusht_relative" if environment == "pusht" else "reacher"
        manifest_path = root / "data/world" / data_name / "manifest.json"
        manifest = read(manifest_path)
        validate_manifest(manifest)
        if manifest["environment"] != environment or manifest["action_block"] != 5:
            raise ValueError("Unexpected simulator family/action grouping")
        if environment == "pusht" and manifest.get("action_interface") != "relative":
            raise ValueError("PushT comparison requires relative controls")
        data_sha = digest(manifest_path)
        keys = {split: expected_keys(manifest, split) for split in ("test", "extrapolation")}
        coordinate_identity = None
        for mode, label, color, marker in METHODS:
            seeds = [0] if mode == "frozen" else [0, 1, 2]
            values = {"test": [], "extrapolation": []}
            source_ids = {split: [] for split in values}
            for seed in seeds:
                package = root / "runs/world" / f"{environment}_{mode}_s{seed}" / "best"
                config = read(package / "config.json")
                if config["model_config"]["mode"] != mode or config["model_config"]["history_length"] != 3:
                    raise ValueError("Unexpected checkpoint model/history interface")
                if not config["model_config"]["freeze_visual"] or config["base_config"]["predictor"]["input_dim"] != 192:
                    raise ValueError("Forecasts require frozen shared 192-dimensional visual coordinates")
                provenance = config["provenance"]
                stats = Path(provenance["action_stats"])
                if digest(stats) != provenance["action_stats_sha256"]:
                    raise ValueError("Action statistics changed since training/export")
                coordinates = (provenance["weights_sha256"], provenance["action_stats_sha256"],
                               config["action_mean"], config["action_std"], config["base_config"])
                if coordinate_identity is not None and coordinates != coordinate_identity:
                    raise ValueError("Incompatible pretrained coordinates/action units within an environment")
                coordinate_identity = coordinates
                best_epochs = None
                training_hashes = {}
                if mode != "frozen":
                    run = read(package.parent / "run_config.json")
                    summary = read(package.parent / "training_summary.json")
                    history = [json.loads(line) for line in (package.parent / "metrics.jsonl").read_text().splitlines() if line.strip()]
                    if run["seed"] != seed or summary.get("status") != "completed" or summary["completed_epochs"] != run["epochs"]:
                        raise ValueError("Training is incomplete or wrongly labelled")
                    if sorted(row["epoch"] for row in history) != list(range(1, run["epochs"] + 1)):
                        raise ValueError("Incomplete/duplicate training metric history")
                    best = min(row["val"]["prediction_loss"] for row in history)
                    if not close(best, summary["best_validation_prediction_loss"]) or provenance["data_manifest_sha256"] != data_sha:
                        raise ValueError("Changed training dataset or validation-best metric")
                    best_epochs = [row["epoch"] for row in history if close(row["val"]["prediction_loss"], best)]
                    training_hashes = {name: digest(package.parent / name)
                                       for name in ("run_config.json", "training_summary.json", "metrics.jsonl")}
                checkpoint_sha = digest(package / "model.pt")
                for split, condition in (("test", "o2_d2"), ("extrapolation", "all")):
                    path = root / "results/world" / f"{environment}_{mode}_s{seed}" / f"forecast_{split}.json"
                    source_id = str(path.relative_to(root))
                    record = read(path)
                    identity = {"checkpoint_sha256": checkpoint_sha, "data_manifest_sha256": data_sha,
                                "evaluator_sha256": evaluator_sha}
                    forecast = validate_forecast(record, environment, mode, seed, split, identity, keys[split])
                    if best_epochs is not None and record["checkpoint_epoch"] not in best_epochs:
                        raise ValueError("Forecast does not use the completed validation-best epoch")
                    file_sha = digest(path)
                    if split == "test":
                        recorded_source = primary["sources"][source_id]
                        if recorded_source != {"sha256": file_sha, **identity}:
                            raise ValueError("Primary-results ledger differs from actual forecast sources")
                    source_ids[split].append(source_id)
                    values[split].append(forecast["summary"][condition]["mse_h5"]["mean"])
                    sources[source_id] = {"sha256": file_sha, **identity,
                        "checkpoint_config_sha256": digest(package / "config.json"),
                        "checkpoint_epoch": record["checkpoint_epoch"], "training_source_hashes": training_hashes,
                        "action_stats_sha256": provenance["action_stats_sha256"],
                        "source_location": f"forecast.summary.{condition}.mse_h5.mean; verified against forecast.records[*].mse_h5",
                        "selected_condition": condition,
                        "observations": forecast["summary"][condition]["mse_h5"]["observations"],
                        "initial_state_seeds": forecast["summary"][condition]["mse_h5"]["clusters"]}
                    if split == "extrapolation":
                        components.append({"environment": environment, "mode": mode, "training_seed": seed,
                            "source_id": source_id,
                            "conditions": {key: value["mse_h5"]["mean"] for key, value in forecast["summary"].items() if key != "all"}})
            for split in ("test", "extrapolation"):
                mean = statistics.mean(values[split])
                sd = statistics.stdev(values[split]) if len(seeds) > 1 else None
                if split == "test":
                    primary_row = source_rows[environment, mode, "heldout_mse_h5"]
                    if primary_row["status"] != "complete" or primary_row["available_seeds"] != seeds:
                        raise ValueError("Primary forecast aggregation has incomplete training seeds")
                    if not close(mean, primary_row["mean"]) or ((sd is None) != (primary_row["training_seed_sd"] is None)) or (sd is not None and not close(sd, primary_row["training_seed_sd"])):
                        raise ValueError("Primary forecast aggregate differs from source values")
                    if any(not close(primary_row["per_seed_values"][str(seed)], value) for seed, value in zip(seeds, values[split])):
                        raise ValueError("Primary per-seed values differ from source records")
                plotted.append({"id": f"{environment}_{split}_{mode}", "environment": environment,
                    "split": split, "condition": "o2_d2" if split == "test" else "mean_over_o3_d0_o0_d3_o3_d3",
                    "mode": mode, "label": label, "color": color, "marker": marker,
                    "metric": "fixed_reference_latent_mse_h5", "direction": "lower",
                    "value": mean, "uncertainty": {"type": "sample_sd", "value": sd,
                        "variation": "independently_trained_seeds", "n": len(seeds), "ddof": 1,
                        "not_a_confidence_interval": True},
                    "training_seeds": seeds, "per_seed_values": dict(zip(map(str, seeds), values[split])),
                    "source_ids": source_ids[split]})
    lookup = {(row["environment"], row["split"], row["mode"]): row for row in plotted}
    deltas = []
    for environment in ("pusht", "reacher"):
        for split in ("test", "extrapolation"):
            ours = lookup[environment, split, "factorized"]
            reference = lookup[environment, split, "framewise"]
            if reference["value"] <= 0:
                raise ValueError("Relative forecast change requires a positive reference MSE")
            deltas.append({"environment": environment, "split": split,
                           "method": "factorized", "reference": "framewise",
                           "ours_value": ours["value"], "reference_value": reference["value"],
                           "relative_change_percent": 100 * (ours["value"] / reference["value"] - 1),
                           "source_ids": ours["source_ids"] + reference["source_ids"],
                           "interpretation": "Ratio of three-seed means; negative is lower MSE; no interval or significance claim."})
    return {"schema_version": 1, "status": "source_validated", "width_inches": 5.5,
            "height_inches": 3.8, "minimum_font_points": 8,
            "source_script_sha256": digest(__file__), "primary_ledger_sha256": digest(primary_path),
            "primary_ledger": str(primary_path.relative_to(root)),
            "metric": {"name": "MSE@5", "units": "mean squared fixed-reference latent coordinate error",
                       "latent_dimensions": 192, "forecast_transitions": 5,
                       "native_actions_per_transition": 5, "history_observations": 3,
                       "future_actions": "recorded", "direction": "lower"},
            "comparison": {"paired_window_populations": True, "sequence_length": 8, "window_stride": 5,
                           "test_conditions": [list(x) for x in split_combinations("test")],
                           "extrapolation_conditions": [list(x) for x in split_combinations("extrapolation")],
                           "trained_checkpoint_selection": "validation-best prediction loss after completed training",
                           "frozen_checkpoint_selection": "unchanged upstream pretrained checkpoint"},
            "sources": sources, "results": plotted, "extrapolation_components": components,
            "comparison_deltas": deltas,
            "caption": CAPTION,
            "limits": ["No planning-success or significance claim.", "Three training seeds; frozen has one checkpoint.",
                       "Overlapping forecast windows are not independent replicates.",
                       "Unaligned predictor (diagnostic) remains in the table; it beats ShiftWM on PushT held-out forecasts.",
                       "Aggregate extrapolation contains different appearance/dynamics regimes; see component values.",
                       "Panel-specific log axes; point-range marks, not bars."]}


def render(ledger, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "forecast_comparison", "axes.linewidth": .7,
        "text.color": "#25282D", "axes.labelcolor": "#25282D", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8))
    fig.subplots_adjust(left=.285, right=.985, top=.855, bottom=.225, wspace=.20, hspace=.58)
    # Shared condition headings and metric label leave more room for evidence.
    for col, heading in enumerate(("Held-out combination", "Extrapolation")):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, .972, heading, ha="center", va="top",
                 fontsize=9.2, fontweight="bold", color="#243447")
    limits = {("pusht", "test"): (.47, 1.32), ("pusht", "extrapolation"): (.9, 1.23),
              ("reacher", "test"): (.0058, .073), ("reacher", "extrapolation"): (.76, 1.34)}
    ticks = {("pusht", "test"): [.5, .7, 1.0], ("pusht", "extrapolation"): [.9, 1., 1.1, 1.2],
             ("reacher", "test"): [.006, .01, .02, .06], ("reacher", "extrapolation"): [.8, 1., 1.2]}
    lookup = {(row["environment"], row["split"], row["mode"]): row for row in ledger["results"]}
    changes = {(row["environment"], row["split"]): row["relative_change_percent"]
               for row in ledger["comparison_deltas"]}
    for row_index, environment in enumerate(("pusht", "reacher")):
        for col_index, split in enumerate(("test", "extrapolation")):
            ax = axes[row_index, col_index]
            panel = environment, split
            ax.axhspan(3.55, 4.45, color="#FBF1E8", lw=0, zorder=0)
            for method_index, (mode, label, color, marker) in enumerate(METHODS):
                result = lookup[environment, split, mode]
                value, sd = result["value"], result["uncertainty"]["value"]
                if value - (sd or 0) <= limits[panel][0] or value + (sd or 0) >= limits[panel][1]:
                    raise ValueError("A fixed plot limit would crop measured evidence; revise the composition")
                ax.errorbar(value, method_index, xerr=sd, marker=marker, markersize=6.0 if mode == "factorized" else 5.3,
                            markeredgecolor=color, markeredgewidth=.75, markerfacecolor=color,
                            color=color, elinewidth=1.3, capsize=2.6, capthick=1.0,
                            linestyle="none", zorder=3)
            # Put ratios in the clear end of the proposal row, away from data marks.
            change = changes[panel]
            right = panel != ("reacher", "extrapolation")
            label = (f"{change:+.1f}%").replace("-", "−")
            ax.text(.97 if right else .03, 4, label, transform=ax.get_yaxis_transform(),
                    ha="right" if right else "left", va="center", fontsize=8.5,
                    fontweight="bold", color="#9F4B11" if change < 0 else "#8D3444", zorder=5)
            ax.set_xscale("log")
            ax.set_xlim(*limits[panel])
            ax.set_ylim(4.6, -.65)
            ax.set_yticks(range(5))
            ax.set_yticklabels([method[1] for method in METHODS] if col_index == 0 else [])
            ax.tick_params(axis="y", length=0, pad=9)
            if col_index == 0:
                ax.get_yticklabels()[-1].set_fontweight("bold")
                ax.get_yticklabels()[-1].set_color("#9F4B11")
            ax.tick_params(axis="x", length=3, width=.7, color="#8A8D92", pad=4)
            ax.xaxis.set_major_locator(FixedLocator(ticks[panel]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis="x", color="#E5E7EB", linewidth=.6, zorder=0)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#A0A4AA")
            letter = "abcd"[row_index * 2 + col_index]
            env_label = "PushT" if environment == "pusht" else "Reacher"
            ax.set_title(f"{letter}  {env_label}", loc="left", fontweight="bold", pad=8)
    fig.text(.635, .118, "Latent MSE@5 ↓  ·  logarithmic axes", ha="center", fontsize=8.5)
    fig.text(.04, .06, "Points: mean ± 1 SD (3 seeds). Δ: relative MSE vs Framewise.",
             fontsize=8, color="#465365")
    fig.text(.04, .025, "Frozen LeWM: one checkpoint; no seed SD. Negative Δ is better.",
             fontsize=8, color="#465365")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), metadata={"Creator": "paper/scripts/render_forecast.py", "CreationDate": None, "ModDate": None})
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/generated/forecast_comparison")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    ledger = build_ledger(args.root)
    if not args.validate_only:
        render(ledger, args.output)
        args.output.with_suffix(".json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": ledger["status"], "source_files": len(ledger["sources"]),
                      "plotted_points": len(ledger["results"]), "width_inches": ledger["width_inches"]}))


if __name__ == "__main__":
    main()
