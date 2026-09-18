#!/usr/bin/env python3
"""Render primary comparisons only when every prescribed training seed exists.

Mean +/- SD describes variation across independently trained models. It is not
a confidence interval over overlapping trajectory windows. Per-task clustered
intervals remain available in the source evaluation files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

MODES = [("frozen", "Frozen LeWM"), ("framewise", "Framewise calibration"),
         ("single", "Shared context"), ("factorized_unpaired", "Unpaired contexts"),
         ("factorized", "ShiftWM (ours)"), ("plain", "Unaligned predictor (diagnostic)")]
METRICS = {
    "canonical_success": ("planning", "test", "summary", "o0_d0", "success", 100),
    "heldout_success": ("planning", "test", "summary", "o2_d2", "success", 100),
    "heldout_eligible_success": ("planning", "test", "eligible_summary", "o2_d2", "success", 100),
    "extrapolation_success": ("planning", "extrapolation", "summary", "all", "success", 100),
    "heldout_mse_h5": ("forecast", "test", "summary", "o2_d2", "mse_h5", 1),
}
MAIN_PLANNER_BUDGET = {"samples": 300, "iterations": 30, "elites": 30,
                       "horizon": 5, "native_budget": 50, "action_block": 5,
                       "receding_horizon": 1, "history_steps_charged": 10}
MAIN_PLANNER_IDENTITY = {"implementation": "stable_worldmodel.planning.solver.cem.CEMSolver",
                         "search_coordinates": "checkpoint_training_action_z_scores; native actions clipped to environment bounds",
                         "goal_offset_from_end_of_history": 5}


def validate_main_planning(record, path, split):
    """Only the locked main world-model protocol belongs in primary tables."""
    planning = record["planning"]
    if planning.get("policy") in ("random", "replay_oracle"):
        return False  # Controls are separate artifacts; their CEM fields are unused.
    if planning.get("policy") != "world_model":
        raise ValueError(f"Missing or invalid planning policy: {path}")
    protocol = planning.get("protocol", {})
    for key, expected in MAIN_PLANNER_BUDGET.items():
        if planning.get("planner", {}).get(key) != expected:
            raise ValueError(f"Incompatible main planner budget for {key}: {path}")
        if key in ("samples", "iterations", "elites", "horizon", "native_budget") and protocol.get(key) != expected:
            raise ValueError(f"Incompatible hashed planning protocol for {key}: {path}")
    for key, expected in (("policy", "world_model"), ("episodes_per_dynamics", 64), ("goal_offset", 5), ("seed", 1701)):
        if protocol.get(key) != expected:
            raise ValueError(f"Incompatible main planning protocol for {key}: {path}")
    for key, expected in MAIN_PLANNER_IDENTITY.items():
        if planning.get("planner", {}).get(key) != expected:
            raise ValueError(f"Incompatible main planner identity for {key}: {path}")
    if protocol.get("search_coordinates") != MAIN_PLANNER_IDENTITY["search_coordinates"]:
        raise ValueError(f"Incompatible hashed planner search coordinates: {path}")
    if protocol.get("split") != split or planning.get("split") != split:
        raise ValueError(f"Planning split metadata disagrees: {path}")
    if protocol.get("evaluator_sha256") != record["evaluator_sha256"]:
        raise ValueError(f"Planning evaluator identity disagrees: {path}")
    identity = protocol.get("run_identity", {})
    if any(identity.get(key) != record[key] for key in ("checkpoint_sha256", "data_manifest_sha256")):
        raise ValueError(f"Planning checkpoint/data identity disagrees: {path}")
    expected_signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if planning.get("signature") != expected_signature:
        raise ValueError(f"Planning protocol signature disagrees: {path}")
    return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/world"))
    parser.add_argument("--output", type=Path, default=Path("paper/generated"))
    args = parser.parse_args()
    rows, sources = [], {}
    loaded = {}
    run_checkpoints, checkpoint_owners = {}, {}
    environment_data, evaluator_versions = {}, set()

    def validated_record(path, environment, mode, seed, kind, split):
        if path in loaded:
            return loaded[path]
        raw = path.read_bytes()
        record = json.loads(raw)
        if record.get("status") != "complete":
            loaded[path] = (raw, record)
            return raw, record
        if kind == "planning" and not validate_main_planning(record, path, split):
            loaded[path] = (raw, record)
            return raw, record
        if record["environment"] != environment or record["model_mode"] != mode:
            raise ValueError(f"Mislabelled evaluation: {path}")
        if "training_seed" in record:
            expected_seed = None if mode == "frozen" else seed
            if record["training_seed"] != expected_seed:
                raise ValueError(f"Wrong training seed in completed evaluation: {path}")
        run = (environment, mode, seed)
        checkpoint = record["checkpoint_sha256"]
        if run in run_checkpoints and run_checkpoints[run] != checkpoint:
            raise ValueError(f"Incompatible checkpoint identities within {environment}/{mode}/seed{seed}")
        run_checkpoints[run] = checkpoint
        if mode != "frozen":
            previous = checkpoint_owners.get(checkpoint)
            if previous is not None and previous != run:
                raise ValueError(f"Identical checkpoint reused across trained runs: {previous} and {run}")
            checkpoint_owners[checkpoint] = run
        data = record["data_manifest_sha256"]
        if environment in environment_data and environment_data[environment] != data:
            raise ValueError(f"Incompatible data versions across {environment} comparisons")
        environment_data[environment] = data
        evaluator_versions.add(record["evaluator_sha256"])
        if len(evaluator_versions) > 1:
            raise ValueError("Incompatible evaluator versions across comparisons")
        loaded[path] = (raw, record)
        return raw, record
    for environment in ("pusht", "reacher"):
        for mode, label in MODES:
            seeds = [0] if mode == "frozen" else [0, 1, 2]
            for metric, (kind, split, group, condition, name, scale) in METRICS.items():
                values, hashes, protocols, available = [], set(), set(), []
                for seed in seeds:
                    path = args.results / f"{environment}_{mode}_s{seed}" / f"{kind}_{split}.json"
                    if not path.exists():
                        continue
                    raw, record = validated_record(path, environment, mode, seed, kind, split)
                    if record.get("status") != "complete":
                        continue
                    if kind == "planning" and record["planning"].get("policy") != "world_model":
                        continue
                    item = record[kind].get(group, {}).get(condition, {}).get(name)
                    if item is None:
                        continue
                    hashes.add(record["data_manifest_sha256"])
                    protocols.add(record["evaluator_sha256"])
                    values.append(float(item["mean"]) * scale)
                    available.append(seed)
                    sources[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                          "checkpoint_sha256": record["checkpoint_sha256"],
                                          "data_manifest_sha256": record["data_manifest_sha256"],
                                          "evaluator_sha256": record["evaluator_sha256"]}
                if len(hashes) > 1 or len(protocols) > 1:
                    raise ValueError(f"Incompatible data/evaluation versions within {environment}/{mode}/{metric}")
                complete = len(values) == len(seeds)
                rows.append({"environment": environment, "mode": mode, "label": label,
                             "metric": metric, "status": "complete" if complete else "pending",
                             "available_seeds": available, "required_seeds": seeds,
                             "mean": statistics.mean(values) if complete else None,
                             "training_seed_sd": statistics.stdev(values) if complete and len(values) > 1 else None,
                             "per_seed_values": dict(zip(available, values)),
                             "interpretation": "diagnostic_only" if mode == "plain" else "primary_comparison"})
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = {"schema_version": 1, "sources": sources, "results": rows,
              "main_planner_budget": MAIN_PLANNER_BUDGET,
              "main_planner_identity": MAIN_PLANNER_IDENTITY,
              "uncertainty": "Sample standard deviation across three trained seeds; frozen has one checkpoint.",
              "missing_policy": "No partial-seed mean is inserted into the primary table."}
    (args.output / "primary_results.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    with (args.output / "primary_results.csv").open("w", newline="") as f:
        columns = ["environment", "mode", "metric", "status", "mean", "training_seed_sd", "available_seeds"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    lookup = {(r["environment"], r["mode"], r["metric"]): r for r in rows}
    def display(environment, mode, metric):
        row = lookup[environment, mode, metric]
        if row["status"] != "complete":
            return r"\missing"
        digits = 4 if "mse" in metric else 1
        value = f"{row['mean']:.{digits}f}"
        if row["training_seed_sd"] is not None:
            value += r" $\pm$ " + f"{row['training_seed_sd']:.{digits}f}"
        return value
    tex = [r"% Generated primary planning comparison; diagnostics are reported separately.",
           r"\begin{table}[t]\centering\small",
           r"\providecommand{\ShiftWMPlanningGainRows}{}",
           r"\InputIfFileExists{generated/planning_gain_rows.tex}{}{}",
           r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}",
           r"\begin{tabular}{lrrrr}\toprule",
           r"& \multicolumn{2}{c}{PushT} & \multicolumn{2}{c}{Reacher}\\",
           r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
           r"Method & Held-out $\uparrow$ & Extrap. $\uparrow$ & Held-out $\uparrow$ & Extrap. $\uparrow$\\\midrule"]
    for mode, label in MODES:
        if mode == "plain":
            continue
        if mode == "factorized":
            tex.append(r"\addlinespace[2pt]")
            tex.append(r"\rowcolor{orange!9}")
            label = r"\textbf{ShiftWM (ours)}"
        cells = [label]
        cells += [display(environment, mode, metric) for environment in ("pusht", "reacher")
                  for metric in ("heldout_success", "extrapolation_success")]
        tex.append(" & ".join(cells) + r"\\")
    tex += [r"\ShiftWMPlanningGainRows",
            r"\bottomrule\end{tabular}",
            r"\caption{\textbf{Main comparison: closed-loop planning success (\%).} Higher is better. Trained entries require all three seeds and show mean $\pm$ sample SD; Frozen LeWM uses one released checkpoint per environment. The final rows give ShiftWM minus Shared context in percentage points and conditional 95\% paired task-cluster intervals, not SD. \positivegain{Bold green} marks positive mean differences, not statistical significance. Dashes denote incomplete evidence, not zero success. Held-out is $(v_2,p_2)$; extrapolation averages three prespecified conditions. All methods share the paid support and planning budget. Framewise calibration, Shared context and Unpaired contexts are in-house controls on the same LeWM backbone. Full metrics and the unaligned diagnostic appear in Table~\ref{tab:measured-full}.}",
            r"\label{tab:measured-primary}\label{tab:primary}\end{table}"]
    (args.output / "primary_results.tex").write_text("\n".join(tex) + "\n")
    detailed = [r"% Generated full comparison; diagnostics are visually separated.",
                r"\begin{table}[t]\centering\scriptsize",
                r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}",
                r"\begin{tabular}{lrrrrr}\toprule",
                r"& \multicolumn{4}{c}{Planning success (\%) $\uparrow$} & Forecast $\downarrow$\\",
                r"\cmidrule(lr){2-5}\cmidrule(l){6-6}",
                r"Method & Canonical & Held-out & Eligible & Extrap. & MSE@5\\\midrule"]
    for environment in ("pusht", "reacher"):
        name = "PushT" if environment == "pusht" else "Reacher"
        detailed.append(r"\multicolumn{6}{l}{\textbf{" + name + r"}}\\[2pt]")
        for mode, label in MODES:
            if mode == "factorized":
                detailed.append(r"\rowcolor{orange!9}")
                label = r"\textbf{ShiftWM (ours)}"
            if mode == "plain":
                detailed.append(r"\addlinespace[3pt]")
                label = r"\shortstack[l]{Unaligned predictor\\(diagnostic)}"
            cells = [label]
            cells += [display(environment, mode, metric) for metric in
                      ("canonical_success", "heldout_success", "heldout_eligible_success",
                       "extrapolation_success", "heldout_mse_h5")]
            detailed.append(" & ".join(cells) + r"\\")
        if environment == "pusht":
            detailed.append(r"\midrule")
    detailed += [r"\bottomrule\end{tabular}",
                 r"\caption{\textbf{Complete metric ledger.} Trained entries require three completed seeds and show mean $\pm$ sample SD; Frozen LeWM has one checkpoint. Dashes denote incomplete evidence. Canonical is $(v_0,p_0)$; held-out and forecast MSE@5 use $(v_2,p_2)$; extrapolation averages three prespecified conditions. Eligible success excludes held-out tasks solved during paid support acquisition; other success columns include them. MSE@5 is five-step latent prediction error in fixed reference coordinates. The unaligned predictor lacks a learned visual calibration path and is a diagnostic, not evidence for factorization.}",
                 r"\label{tab:measured-full}\end{table}"]
    (args.output / "primary_results_appendix.tex").write_text("\n".join(detailed) + "\n")
    methods = [r"% Sources: src/shiftwm/model.py and configs/world/*_s0.json.",
               r"\begingroup\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}",
               r"\begin{tabular}{@{}llll@{}}\toprule",
               r"Method & Visual calibration & Context & Consistency\\\midrule",
               r"Frozen LeWM & None & None & None\\",
               r"Framewise calibration & Per frame & None & None\\",
               r"Shared context & Contextual & Shared & None\\",
               r"Unpaired contexts & Contextual & Separate & None\\",
               r"\textbf{ShiftWM (ours)} & Contextual & Separate & Paired\\",
               r"\bottomrule\end{tabular}\endgroup"]
    (args.output / "method_comparison.tex").write_text("\n".join(methods) + "\n")
    pending_controls = [label for mode, label in MODES if mode in {"framewise", "factorized_unpaired"}
                        and any(lookup[environment, mode, metric]["status"] != "complete"
                                for environment in ("pusht", "reacher")
                                for metric in ("heldout_success", "extrapolation_success"))]
    if pending_controls:
        names = " and ".join(pending_controls)
        status_text = (f"The prescribed three-seed planning results for {names} are not yet complete "
                       "across both environments and evaluation populations. "
                       "Table~\\ref{tab:measured-primary} shows completed cells; "
                       "partial-seed outcomes remain excluded.")
    else:
        status_text = ("All prescribed three-seed planning comparisons with Framewise calibration and "
                       "Unpaired contexts are available in Table~\\ref{tab:measured-primary}, "
                       "with paired uncertainty reported in Table~\\ref{tab:paired-planning}.")
    (args.output / "primary_comparison_status.tex").write_text(status_text + "\n")
    print(json.dumps({"evaluation_sources": len(sources), "complete_cells": sum(r["status"] == "complete" for r in rows),
                      "total_cells": len(rows)}))


if __name__ == "__main__":
    main()
