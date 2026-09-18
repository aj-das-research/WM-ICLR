#!/usr/bin/env python3
"""Render actual shiftwm evaluation records; no fabricated or pooled results.

Example: .venv/bin/python paper/scripts/render_results.py results/pusht/*.json
Each run retains its own task-cluster interval. This script deliberately does
not combine training seeds or rank incompatible planner/data configurations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def escape(value):
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
               "#": r"\#", "{": r"\{", "}": r"\}", "$": r"\$"}
    return "".join(mapping.get(c, c) for c in str(value))


def measured(summary, metric, scale=1.0):
    if metric not in summary:
        return None
    item = summary[metric]
    values = [float(item["mean"]), *map(float, item["ci95"])]
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError(f"Invalid interval for {metric}")
    if values[1] > values[2]:
        raise ValueError(f"Reversed interval for {metric}")
    if metric in {"success", "final_success"} and not all(0 <= x <= 1 for x in values):
        raise ValueError("Success must be a fraction, not a percent")
    return {"mean": values[0] * scale, "lower": values[1] * scale,
            "upper": values[2] * scale, "clusters": item["clusters"],
            "observations": item["observations"], "method": item["interval_method"]}


def display(item, decimals=2):
    if item is None:
        return r"\missing"
    if item["lower"] is None:
        return f"{item['mean']:.{decimals}f}"
    return f"{item['mean']:.{decimals}f} [{item['lower']:.{decimals}f}, {item['upper']:.{decimals}f}]"


def latency_eligible(result):
    """Efficiency claims require explicit uncontended main-evaluation provenance."""
    context = result.get("execution_context", {})
    return (context.get("main_efficiency_claim_eligible") is True
            and context.get("shared_gpu_with_training") is False
            and result.get("planning", {}).get("split") in {"test", "extrapolation"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("paper/generated"))
    args = parser.parse_args()
    sources, rows = [], []
    for path in args.inputs:
        raw = path.read_bytes()
        result = json.loads(raw)
        if result.get("status") != "complete":
            raise ValueError(f"Refusing incomplete evaluation: {path}")
        if not result.get("checkpoint_sha256") or not result.get("data_manifest_sha256"):
            raise ValueError(f"Missing checkpoint/data provenance: {path}")
        planning = result.get("planning", {})
        forecast = result.get("forecast", {})
        conditions = sorted(set(planning.get("summary", {})) | set(forecast.get("summary", {})))
        if not conditions:
            raise ValueError(f"No evaluation summaries: {path}")
        run = f"{path.parent.name}/{path.stem}"
        source = {"run_id": run, "file": str(path.resolve()),
                  "source_sha256": hashlib.sha256(raw).hexdigest(),
                  "checkpoint_sha256": result["checkpoint_sha256"],
                  "data_manifest_sha256": result["data_manifest_sha256"],
                  "model_mode": result["model_mode"], "planner": planning.get("planner"),
                  "forecast_split": forecast.get("split"), "planning_split": planning.get("split"),
                  "execution_context": result.get("execution_context"),
                  "efficiency_claim_eligible": latency_eligible(result)}
        sources.append(source)
        for condition in conditions:
            ps = planning.get("summary", {}).get(condition, {})
            eligible = planning.get("eligible_summary", {}).get(condition, {})
            fs = forecast.get("summary", {}).get(condition, {})
            costs = planning.get("aggregate_costs", {})
            latency = None
            if source["efficiency_claim_eligible"] and condition == "all" and costs.get("seconds_per_replan") is not None:
                latency = {"mean": float(costs["seconds_per_replan"]) * 1000,
                           "lower": None, "upper": None, "clusters": None,
                           "observations": costs["total_replans"],
                           "method": "total solve time / actually executed planning calls; no uncertainty interval"}
            rows.append({"run_id": run, "mode": result["model_mode"], "condition": condition,
                         "success_percent": measured(ps, "success", 100),
                         "eligible_success_percent": measured(eligible, "success", 100),
                         "forecast_mse_h5": measured(fs, "mse_h5"),
                         "replan_ms": latency,
                         "source_sha256": source["source_sha256"],
                         "source_keys": {"success_percent": f"planning.summary.{condition}.success",
                                         "forecast_mse_h5": f"forecast.summary.{condition}.mse_h5",
                                         "eligible_success_percent": f"planning.eligible_summary.{condition}.success",
                                         "replan_ms": "planning.aggregate_costs.seconds_per_replan"}})
    if len({s["run_id"] for s in sources}) != len(sources):
        raise ValueError("Ambiguous duplicate run IDs; use distinct source parent directories")
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = {"schema_version": 1, "status": "measured", "sources": sources, "results": rows,
              "aggregation": "None; rows retain distinct runs and condition summaries.",
              "uncertainty": "95% trajectory-cluster bootstrap supplied by evaluator; not variation over training seeds."}
    (args.output / "result_ledger.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    with (args.output / "evaluation_summary.csv").open("w", newline="") as stream:
        keys = ["run_id", "mode", "condition", "metric", "mean", "lower", "upper", "clusters", "observations", "source_sha256"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            for metric in ("success_percent", "eligible_success_percent", "forecast_mse_h5", "replan_ms"):
                if row[metric] is not None:
                    item = {key: row[key] for key in ("run_id", "mode", "condition", "source_sha256")}
                    item.update({key: row[metric][key] for key in ("mean", "lower", "upper", "clusters", "observations")})
                    writer.writerow(dict(item, metric=metric))
    # One condition per row. No missing value is converted to zero.
    tex = [r"\begingroup\scriptsize", r"\begin{longtable}{p{0.25\linewidth}p{0.17\linewidth}p{0.17\linewidth}p{0.17\linewidth}p{0.04\linewidth}}",
           r"\caption{Measured evaluation records. Intervals are 95\% trajectory-cluster bootstrap intervals within each run; they are not training-seed uncertainty. Different conditions and budgets are not pooled. Latency requires explicit dedicated main-evaluation provenance; shared-GPU development timings are excluded.}\\",
           r"\toprule Run / condition & Success (\%) & MSE@5 & ms/replan & Tasks\\\midrule\endhead"]
    for row in rows:
        count = row["success_percent"] or row["forecast_mse_h5"] or row["replan_ms"]
        label = escape(row["run_id"] + " / " + row["condition"])
        tex.append(" & ".join([label, display(row["success_percent"], 1),
                               display(row["forecast_mse_h5"], 4), display(row["replan_ms"], 1),
                               str(count["clusters"])]) + r"\\")
    tex += [r"\bottomrule\end{longtable}\endgroup"]
    (args.output / "evaluations.tex").write_text("\n".join(tex) + "\n")
    # Optional plotting dependency is already present in the project environment.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    for source in sources:
        plotted = [r for r in rows if r["run_id"] == source["run_id"]
                   and r["condition"] != "all" and r["success_percent"] is not None]
        if not plotted:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 2.7), layout="constrained")
        for i, row in enumerate(plotted):
            value = row["success_percent"]
            ax.vlines(i, value["lower"], value["upper"], color="#0072B2", linewidth=1.2)
            ax.scatter(i, value["mean"], color="#0072B2", s=25, zorder=3)
        ax.set(xticks=range(len(plotted)), xticklabels=[r["condition"] for r in plotted],
               ylabel="Planning success (%)", ylim=(-2, 102), title=source["run_id"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.set_axisbelow(True)
        fig.text(.5, -.02, "95% trajectory-cluster intervals; single evaluation run", ha="center", fontsize=7)
        name = source["run_id"].replace("/", "_") + "_planning"
        for extension in ("pdf", "svg", "png"):
            fig.savefig(args.output / f"{name}.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps({"sources": len(sources), "rows": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
