#!/usr/bin/env python3
"""Presentation-only simulator tables from completed, source-bound reports.

No model imports, experiments, checkpoint loads, or new physical-error estimands.
The existing scientific reporters remain responsible for experimental validation.
This renderer verifies their result-file hashes, checks complete-seed arithmetic,
and records exact source pointers for every displayed row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACK_RELATIVE = Path("paper/table_sources/simulator_tables")
CORE_SOURCE = "paper/generated/primary_results.json"
EXTENSION_SOURCE = "reports/completed_extension_results.json"
MODES = (
    ("frozen", "Frozen LeWM"),
    ("framewise", "Framewise calibration"),
    ("single", "Shared context"),
    ("factorized_unpaired", "Unpaired contexts"),
    ("factorized", "ShiftWM (ours)"),
    ("plain", "Unaligned diagnostic"),
)
CORE_METRICS = (
    "canonical_success", "heldout_success", "heldout_eligible_success",
    "extrapolation_success", "heldout_mse_h5",
)
EXTENSION_MODES = (
    ("framewise", "Framewise"), ("constant_dynamics", "Constant dynamics"),
    ("factorized", "ShiftWM (ours)"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object, description: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), description)
    number = float(value)
    require(math.isfinite(number), description)
    return number


def close(a: object, b: object, description: str) -> None:
    require(math.isclose(finite(a, description), finite(b, description),
                         rel_tol=1e-10, abs_tol=1e-12), description)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Sources:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}

    def read(self, name: str, expected: str | None = None) -> dict:
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), f"Source outside project: {name}")
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        require(expected is None or actual == expected, f"Stale result source: {name}")
        self.hashes[str(path.relative_to(self.root))] = actual
        return json.loads(payload)

    def unchanged(self) -> None:
        for name, expected in self.hashes.items():
            require(digest(self.root / name) == expected, f"Source changed during render: {name}")


def core_rows(report: dict) -> list[dict]:
    indexed = {}
    for i, item in enumerate(report["results"]):
        key = item["environment"], item["mode"], item["metric"]
        require(key not in indexed, f"Duplicate core result: {key}")
        require(key[0] in ("pusht", "reacher") and key[1] in dict(MODES)
                and key[2] in CORE_METRICS, f"Unknown core result: {key}")
        indexed[key] = i, item
    rows = []
    for environment in ("pusht", "reacher"):
        for mode, label in MODES:
            metrics, pointers = {}, []
            seeds = [0] if mode == "frozen" else [0, 1, 2]
            for metric in CORE_METRICS:
                entry = indexed.get((environment, mode, metric))
                if entry is None or entry[1]["status"] != "complete":
                    metrics[metric] = None
                    continue
                index, item = entry
                require(item["required_seeds"] == item["available_seeds"] == seeds,
                        f"Incomplete seed contract: {environment}/{mode}/{metric}")
                require(set(item["per_seed_values"]) == {str(s) for s in seeds},
                        f"Unexpected per-seed values: {environment}/{mode}/{metric}")
                values = [finite(item["per_seed_values"][str(s)], metric) for s in seeds]
                require(all(x >= 0 for x in values), f"Negative metric: {metric}")
                if metric.endswith("success"):
                    require(all(x <= 100 for x in values), "Success outside percentage bounds")
                close(item["mean"], statistics.mean(values), "Core mean disagrees with seeds")
                sd = item["training_seed_sd"]
                if len(seeds) == 3:
                    close(sd, statistics.stdev(values), "Core SD disagrees with seeds")
                else:
                    require(sd is None, "Frozen checkpoint must not have a training-seed SD")
                metrics[metric] = {"mean": item["mean"], "training_seed_sd": sd,
                                   "per_seed_values": item["per_seed_values"]}
                pointers.append(f"{CORE_SOURCE}#/results/{index}")
            rows.append({"environment": environment, "model_family": "historical_context",
                         "backbone": "LeWM transformer", "mode": mode, "label": label,
                         "status": "complete" if all(v is not None for v in metrics.values()) else "pending",
                         "training_seeds": seeds, "metrics": metrics, "source_pointers": pointers})
    return rows


def verify_extension_results(report: dict, sources: Sources) -> None:
    require(report["status"] == "verified_complete", "Extension report is not finalized")
    require(len(report["original_runs"]) == 36 and len(report["geometry_runs"]) == 6,
            "Finalized extension report must contain all 42 runs")
    flags = report["verification"]
    for key in ("all_training_completion_and_strict_load_checks",
                "all_result_embedded_identities_equal_sidecars",
                "all_recorded_source_protocol_checkpoint_and_trace_hashes_match",
                "all_planning_counts_recomputed",
                "all_paired_tasks_initial_metrics_and_support_prefixes_equal",
                "all_forecast_means_recomputed"):
        require(flags.get(key) is True, f"Extension validation missing: {key}")
    common_tasks, common_windows = {}, {}
    for collection in ("original_runs", "geometry_runs"):
        for row in report[collection]:
            plan = sources.read(row["planning_result"], report["source_sha256"][row["planning_result"]])
            forecast = sources.read(row["forecast_result"], report["source_sha256"][row["forecast_result"]])
            require(plan["status"] == forecast["status"] == "completed", "Incomplete extension result")
            require(len(plan["records"]) == plan["tasks"] == row["tasks"] == 8,
                    "Extension task denominator changed")
            require(sum(r["success"] is True for r in plan["records"]) == row["successes"] == plan["successes"],
                    "Extension successes disagree with records")
            require(sum(r["success_during_support"] is True for r in plan["records"])
                    == row["support_successes"] == plan["support_successes"] == 0,
                    "Extension support eligibility changed")
            keys = [(r["trajectory_id"], r["seed"], r["observation_id"], r["dynamics_id"])
                    for r in plan["records"]]
            require(len(set(keys)) == 8, "Duplicate extension physical task")
            domain = row["domain"]
            require(common_tasks.setdefault(domain, keys) == keys, "Unmatched extension planning tasks")
            for horizon in (1, 3, 5):
                summary = forecast["summary"]["o1_d1"][f"mse_h{horizon}"]
                close(summary["mean"], row["forecast_mse"][f"h{horizon}"],
                      "Extension forecast disagrees with pinned result")
                require(summary["clusters"] == row["forecast_trajectories"] == 16
                        and summary["observations"] == row["forecast_windows"] == 144,
                        "Extension forecast population changed")
            window_keys = [(r["trajectory_id"], r["start"], r["observation_id"], r["dynamics_id"])
                           for r in forecast["records"]]
            require(len(window_keys) == len(set(window_keys)) == 144, "Invalid forecast windows")
            require(common_windows.setdefault(domain, window_keys) == window_keys,
                    "Unmatched extension forecast windows")


def extension_rows(report: dict) -> list[dict]:
    result = []
    for collection, campaign in (("original_runs", "original"), ("geometry_runs", "wide_gain")):
        indexed = {}
        for i, item in enumerate(report[collection]):
            key = item["domain"], item["architecture"], item["mode"], item["training_seed"]
            require(key not in indexed, f"Duplicate extension result: {campaign}/{key}")
            require(key[0] in ("drone", "surgery") and key[1] in ("transformer", "gru")
                    and key[2] in dict(EXTENSION_MODES) and key[3] in (0, 1, 2),
                    f"Unknown extension result: {campaign}/{key}")
            if campaign == "wide_gain":
                require(key[:2] == ("drone", "transformer") and key[2] != "framewise",
                        "Unregistered wide-gain configuration")
            indexed[key] = i, item
        domains = ("drone", "surgery") if campaign == "original" else ("drone",)
        backbones = ("transformer", "gru") if campaign == "original" else ("transformer",)
        modes = EXTENSION_MODES if campaign == "original" else EXTENSION_MODES[1:]
        for domain in domains:
            for architecture in backbones:
                for mode, label in modes:
                    entries = [indexed.get((domain, architecture, mode, seed)) for seed in range(3)]
                    base = {"domain": domain, "backbone": architecture, "campaign": campaign,
                            "model_family": "historical_context", "mode": mode, "label": label,
                            "reference": "framewise" if campaign == "original" else "constant_dynamics",
                            "training_seeds": [0, 1, 2], "physical_tasks": 8,
                            "forecast_trajectories": 16, "forecast_windows": 144,
                            "physical_error_mean": None,
                            "physical_error_unavailable_reason": "No comparable aggregate in the finalized source report; per-task traces are not silently re-aggregated."}
                    if any(entry is None for entry in entries):
                        result.append({**base, "status": "pending", "metrics": None, "source_pointers": []})
                        continue
                    runs = [entry[1] for entry in entries]
                    for run in runs:
                        require(run["training_summary"]["status"] == "completed"
                                and run["training_summary"]["completed_epochs"] == 30,
                                "Extension run has not completed 30 epochs")
                        require(run["tasks"] == 8 and run["support_successes"] == 0,
                                "Extension population changed")
                        require(type(run["successes"]) is int and 0 <= run["successes"] <= 8,
                                "Invalid extension success count")
                        require(run["forecast_trajectories"] == 16 and run["forecast_windows"] == 144,
                                "Extension forecast population changed")
                    metrics = {"success_percent": 100 * sum(r["successes"] for r in runs) / 24,
                               **{f"mse_h{h}": statistics.mean(finite(r["forecast_mse"][f"h{h}"], "forecast MSE")
                                                               for r in runs) for h in (1, 3, 5)}}
                    require(all(v >= 0 for v in metrics.values()), "Negative extension metric")
                    result.append({**base, "status": "complete", "metrics": metrics,
                                   "successes_per_seed": [r["successes"] for r in runs],
                                   "source_pointers": [f"{EXTENSION_SOURCE}#/{collection}/{entry[0]}" for entry in entries]})
    return result


def gain(value: float | None, reference: float | None, lower_better: bool = False) -> float | None:
    if value is None or reference is None or (lower_better and reference == 0):
        return None
    return 100 * (reference - value) / reference if lower_better else value - reference


def show_gain(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "--"
    text = f"{value:+.2f}" + suffix
    return r"\positivegain{" + text + "}" if value > 0 else text


def best_value(value: float, candidates: list[float], decimals: int, lower_better: bool) -> str:
    best = min(candidates) if lower_better else max(candidates)
    text = f"{value:.{decimals}f}"
    # Compare unrounded measurements; preserve exact ties, never select by row order.
    return r"\textbf{" + text + "}" if value == best else text


def core_table(rows: list[dict]) -> str:
    lines = [r"% Generated by render_simulator_tables.py; historical context model only.",
             r"\begin{table}[!htb]\centering\fontsize{8}{9.5}\selectfont",
             r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
             r"& \multicolumn{4}{c}{Planning success (\%) $\uparrow$} & Feature error $\downarrow$\\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-6}",
             r"Method & Canonical & Held-out & Eligible & Extrap. & MSE@5\\\midrule"]
    for environment, title in (("pusht", "PushT"), ("reacher", "Reacher")):
        group = [r for r in rows if r["environment"] == environment]
        lines.append(r"\multicolumn{6}{@{}l}{\textbf{" + title + r" / LeWM transformer}}\\[2pt]")
        for row in group:
            cells = [row["label"]]
            for metric in CORE_METRICS:
                item = row["metrics"][metric]
                if item is None:
                    cells.append("--")
                    continue
                decimals = 4 if metric == "heldout_mse_h5" else 1
                values = [r["metrics"][metric]["mean"] for r in group if r["metrics"][metric] is not None]
                number = best_value(item["mean"], values, decimals, metric == "heldout_mse_h5")
                if item["training_seed_sd"] is not None:
                    number += r" $\pm$ " + f"{item['training_seed_sd']:.{decimals}f}"
                cells.append(number)
            lines.append(" & ".join(cells) + r"\\")
        ours = next(r for r in group if r["mode"] == "factorized")
        reference = next(r for r in group if r["mode"] == "framewise")
        values = []
        for metric in CORE_METRICS:
            a, b = ours["metrics"][metric], reference["metrics"][metric]
            delta = gain(a["mean"] if a else None, b["mean"] if b else None, metric == "heldout_mse_h5")
            values.append(show_gain(delta, r"\%" if metric == "heldout_mse_h5" else ""))
        lines.append(r"\addlinespace[2pt]Ours $-$ Framewise / gain & " + " & ".join(values) + r"\\")
        lines.append(r"\midrule" if environment == "pusht" else r"\bottomrule")
    lines += [r"\end{tabular}",
              r"\caption{\textbf{Historical context model: all core method--metric results.} "
              r"Trained entries are three-seed means $\pm$ sample SD; Frozen LeWM is one released checkpoint. "
              r"Black bold marks the best point mean per environment and metric (including exact ties); "
              r"the unaligned row remains a diagnostic, not an aligned baseline. Canonical is $(v_0,p_0)$; "
              r"held-out, eligible and MSE@5 use $(v_2,p_2)$; extrapolation averages three registered conditions. "
              r"Eligible success excludes tasks solved during common support. The gain rows compare ShiftWM (ours) "
              r"with Framewise: the four success differences are percentage points, and the last column is relative "
              r"MSE reduction in percent. \positivegain{Bold green} means a favorable point difference, not significance. "
              r"Dashes mean incomplete evidence. Paired planning uncertainty is unchanged in the existing contrast tables. "
              r"These are Ours-1 results, not simulator evaluation of the newer spatial model.}",
              r"\label{tab:measured-full}\label{tab:simulator-core-consolidated}\end{table}"]
    return "\n".join(lines) + "\n"


def extension_table(rows: list[dict]) -> str:
    lines = [r"% Generated by render_simulator_tables.py; replaces original + wide-gain absolute rows.",
             r"\begingroup\fontsize{8}{9.5}\selectfont\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}",
             r"\begin{tabular}{@{}lrrrrrrr@{}}\toprule",
             r"& \multicolumn{2}{c}{Success $\uparrow$} & \multicolumn{3}{c}{MSE ($10^{-3}$) $\downarrow$} & \multicolumn{2}{c}{Gain vs ref.}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-8}",
             r"Method & S0/1/2 & \% & h1 & h3 & h5 & pp $\uparrow$ & h5 \% $\uparrow$\\\midrule"]
    groups = list(dict.fromkeys((r["domain"], r["backbone"], r["campaign"]) for r in rows))
    for group_key in groups:
        group = [r for r in rows if (r["domain"], r["backbone"], r["campaign"]) == group_key]
        domain, backbone, campaign = group_key
        name = "Drone" if domain == "drone" else "Tissue manipulation"
        architecture = "Transformer" if backbone == "transformer" else "GRU"
        reference_mode = group[0]["reference"]
        ref_name = "Framewise" if reference_mode == "framewise" else "wide Constant"
        suffix = " / wide gain" if campaign == "wide_gain" else ""
        lines.append(r"\multicolumn{8}{@{}l}{\emph{" + f"{name} / {architecture}{suffix}; ref.: {ref_name}" + r"}}\\")
        reference = next(r for r in group if r["mode"] == reference_mode)
        for row in group:
            if row["metrics"] is None:
                lines.append(row["label"] + " & " + " & ".join(["--"] * 7) + r"\\")
                continue
            metrics = row["metrics"]
            cells = [row["label"], ",".join(str(s) for s in row["successes_per_seed"])]
            for metric in ("success_percent", "mse_h1", "mse_h3", "mse_h5"):
                scale = 1 if metric == "success_percent" else 1000
                candidates = [r["metrics"][metric] * scale for r in group if r["metrics"] is not None]
                cells.append(best_value(metrics[metric] * scale, candidates, 2 if scale == 1 else 3, scale != 1))
            for metric, lower in (("success_percent", False), ("mse_h5", True)):
                if row["mode"] == reference_mode:
                    cells.append("ref.")
                else:
                    ref = reference["metrics"][metric] if reference["metrics"] else None
                    cells.append(show_gain(gain(metrics[metric], ref, lower)))
            lines.append(" & ".join(cells) + r"\\")
        lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule\end{tabular}\endgroup"]
    return "\n".join(lines) + "\n"


EXTENSION_CAPTION = r"""\textbf{Historical context model: all extension methods and forecast horizons.}
Original groups contain Framewise, Constant dynamics and ShiftWM (ours); the last group contains
both registered wide-observation-gain variants. S0/1/2 gives successes out of eight tasks for each
training seed. Success rates pool 24 task--seed instances, which are only eight physical tasks;
all support-only counts are zero. MSE averages the three seed means on the same 16 development
trajectories and 144 windows per domain. Black bold identifies the best point mean in each group
(including exact ties). Gain columns use the explicitly named within-group reference: success
differences are percentage points, and h5 gains are relative error reductions in percent.
\positivegain{Bold green} identifies favorable point differences, not statistical significance.
All original paired planning intervals include zero; retained paired tables give uncertainty.
All 42 runs completed 30 epochs and both evaluations. Final test evaluation remains unrun.
Physical-error means are omitted because the finalized ledger does not aggregate them.
These adapted simulations do not establish clinical performance or performance of spatial ShiftWM.
"""


def geometry_effects(report: dict) -> str:
    lines = [r"% Same three source-validated contrasts; absolute scores moved to consolidated table.",
             r"\begingroup\fontsize{8}{9.5}\selectfont\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}lrr@{}}\toprule",
             r"Paired contrast & $\Delta$ (pp) & 95\% interval (pp)\\\midrule"]
    expected = ("Wide ShiftWM minus wide constant", "Wide ShiftWM minus original ShiftWM",
                "Wide constant minus original constant")
    comparisons = report["geometry_planning_comparisons"]
    require([r["comparison"] for r in comparisons] == list(expected), "Geometry comparison roster changed")
    for row in comparisons:
        require(row["status"] == "complete", "Geometry contrast is incomplete")
        label = row["comparison"].replace("ShiftWM", "ShiftWM (ours)").replace(" minus ", r" $-$ ")
        delta = finite(row["success_difference_percentage_points"], "geometry difference")
        lo, hi = row["ci95_percentage_points"]
        require(finite(lo, "CI low") <= finite(hi, "CI high"), "Invalid geometry interval")
        lines.append(label + " & " + show_gain(delta) + f" & $[{lo:+.2f}, {hi:+.2f}]$" + r"\\")
    return "\n".join(lines + [r"\bottomrule\end{tabular}\endgroup"]) + "\n"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == text:
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def portable_pack(root: Path, core: dict, extensions: dict, sources: Sources) -> None:
    """Export compact verified summaries; raw private results are provenance only."""
    row_fields = ("name", "domain", "architecture", "mode", "training_seed", "training_summary",
                  "forecast_mse", "forecast_trajectories", "forecast_windows", "successes", "tasks", "support_successes")
    compact_extensions = {
        key: [{field: row[field] for field in row_fields} for row in extensions[key]]
        for key in ("original_runs", "geometry_runs")
    }
    compact_extensions["geometry_planning_comparisons"] = [
        {field: row[field] for field in ("comparison", "status", "success_difference_percentage_points", "ci95_percentage_points")}
        for row in extensions["geometry_planning_comparisons"]
    ]
    payload = {"schema_version": 1, "core_report": {"results": core["results"]},
               "extension_report": compact_extensions, "original_source_sha256": sources.hashes,
               "extraction_renderer_sha256": digest(Path(__file__)),
               "scope": "Previously validated presentation summaries only; original paths are provenance, not portable runtime dependencies.",
               "fresh_extraction_checks": [f"{len(sources.hashes)} report/result file hashes",
                                            "core complete seed arithmetic", "extension matched tasks and forecast windows"]}
    pack = root / PACK_RELATIVE
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(pack / "data.json", encoded)
    manifest = {"schema_version": 1, "kind": "portable_simulator_presentation_source_pack",
                "files_sha256": {"data.json": hashlib.sha256(encoded.encode()).hexdigest()},
                "scope": payload["scope"], "extraction_renderer_sha256": payload["extraction_renderer_sha256"]}
    atomic_write(pack / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_inputs(root: Path, sources: Sources, refresh_pack: bool, portable: bool) -> tuple[dict, dict, str, dict]:
    """Use current private evidence when available; never disguise stale files as fallback."""
    live = not portable and (root / CORE_SOURCE).is_file() and (root / EXTENSION_SOURCE).is_file()
    if live:
        core = sources.read(CORE_SOURCE)
        extensions = sources.read(EXTENSION_SOURCE)
        result_files = list(core["sources"]) + [r[key] for section in ("original_runs", "geometry_runs")
                                               for r in extensions[section] for key in ("forecast_result", "planning_result")]
        live = all((root / p).is_file() for p in result_files)
    if not live:
        # Missing private files permit portable rendering. A hash mismatch with
        # existing files in the live path below remains a hard failure.
        sources.hashes.clear()
        manifest_name = str(PACK_RELATIVE / "manifest.json")
        manifest = sources.read(manifest_name)
        require(manifest["kind"] == "portable_simulator_presentation_source_pack", "Wrong source-pack kind")
        payload = sources.read(str(PACK_RELATIVE / "data.json"), manifest["files_sha256"]["data.json"])
        require(payload["schema_version"] == 1, "Unknown portable source-pack version")
        require(payload["extraction_renderer_sha256"] == manifest["extraction_renderer_sha256"],
                "Portable extraction identity mismatch")
        return payload["core_report"], payload["extension_report"], "portable_verified_summary", payload["original_source_sha256"]
    # Read small result records, never multi-GB model/data artifacts. The source
    # reports retain their wider scientific validator/checkpoint provenance.
    for name, metadata in core["sources"].items():
        sources.read(name, metadata["sha256"])
    verify_extension_results(extensions, sources)
    # Validate arithmetic before creating a pack that can outlive private inputs.
    core_rows(core)
    extension_rows(extensions)
    geometry_effects(extensions)
    sources.unchanged()
    if refresh_pack:
        portable_pack(root, core, extensions, sources)
    return core, extensions, "live_report_and_result_validation", {}


def build(root: Path, output: Path, refresh_pack: bool = False, portable: bool = False) -> dict:
    sources = Sources(root)
    core, extensions, input_mode, historical_hashes = load_inputs(root, sources, refresh_pack, portable)
    original_rows = core_rows(core)
    extended_rows = extension_rows(extensions)
    outputs = {
        "core_complete.tex": core_table(original_rows),
        "extensions_complete.tex": extension_table(extended_rows),
        "extensions_caption.tex": r"\providecommand{\SimulatorExtensionsCaption}{" + EXTENSION_CAPTION + "}\n",
        "geometry_effects.tex": geometry_effects(extensions),
    }
    sources.unchanged()
    for name, contents in outputs.items():
        atomic_write(output / name, contents)
    ledger = {"schema_version": 1,
              "status": "complete" if all(r["status"] == "complete" for r in original_rows + extended_rows) else "pending",
              "scope": "Presentation-only consolidation of historical context-model simulations; not current spatial ShiftWM.",
              "core_rows": original_rows, "extension_rows": extended_rows,
              "core_method_rows": len(original_rows), "extension_method_rows": len(extended_rows),
              "training_runs_covered": {"original_core": 30, "original_extensions": 36, "wide_gain": 6},
              "input_mode": input_mode,
              "source_sha256": sources.hashes,
              "portable_extraction_original_source_sha256": historical_hashes,
              "renderer_sha256": digest(Path(__file__)),
              "output_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in outputs.items()},
              "verification": {"upstream_report_result_hashes_checked": input_mode == "live_report_and_result_validation",
                               "core_complete_seed_means_and_sd_checked": True,
                               "extension_task_and_window_pairing_checked": input_mode == "live_report_and_result_validation",
                               "portable_summary_hashes_checked": input_mode == "portable_verified_summary",
                               "all_source_files_unchanged_during_render": True,
                               "new_training_inference_or_checkpoint_validation": False,
                               "interpretation": "Relies on existing scientific validation reports; fresh checks cover report/result bytes and displayed arithmetic, not full checkpoint/data revalidation."},
              "formatting": {"font_pt": 8, "no_resizebox": True,
                             "black_bold": "best unrounded absolute point mean within a task/backbone/campaign group, preserving ties",
                             "green_bold": "strictly positive named-reference point gain only; no significance inference"},
              "omitted_metrics": {"physical_error_means": "Not aggregated in the finalized reports; no new estimand was introduced.",
                                  "runtime": "No cross-campaign comparable isolated latency aggregate in these inputs."},
              "replacement_map": {"paper/generated/primary_results_appendix.tex": "core_complete.tex",
                                  "paper/generated/extensions_completed/all_development.tex": "extensions_complete.tex",
                                  "paper/generated/extensions_completed/geometry_ablation.tex": "geometry_effects.tex"},
              "retained_separate_tables": ["original primary planning comparison", "original paired planning effects",
                                            "extension paired planning effects", "development dynamics/rollout revisions",
                                            "random/replay controls", "official LeWM/AdaJEPA reproductions"]}
    atomic_write(output / "evidence.json", json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/generated/simulator_tables")
    parser.add_argument("--refresh-source-pack", action="store_true", help="Refresh portable summaries after live result validation; portable fallback remains read-only.")
    parser.add_argument("--portable", action="store_true", help="Render using the compact verified public source pack only.")
    args = parser.parse_args()
    ledger = build(args.root, args.output, args.refresh_source_pack, args.portable)
    print(json.dumps({"status": ledger["status"], "core_rows": ledger["core_method_rows"],
                      "extension_rows": ledger["extension_method_rows"],
                      "checked_source_files": len(ledger["source_sha256"])}))


if __name__ == "__main__":
    main()
