#!/usr/bin/env python3
"""Render the presentation-only benchmark index; never evaluate or rank models."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/generated/benchmark_scorecards"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Follow the source-bound numerical display, including in a public checkout.
    # The scorecard renderer verifies scientific completion before emitting it.
    pack = ROOT / "paper/figure_sources/current_real_scorecards/data.json"
    pack_manifest = pack.with_name("manifest.json")
    iws_count = 4
    if pack.exists():
        identity = json.loads(pack_manifest.read_text())
        if identity.get("data_sha256") != hashlib.sha256(pack.read_bytes()).hexdigest():
            raise ValueError("Changed current real scorecard pack")
        evidence = json.loads(pack.read_text())
        if evidence.get("status") != "complete_validated_scores":
            raise ValueError("Current real scorecard pack is not complete")
        counts = {len(task["methods"]) for task in evidence["iws"]["tasks"].values()}
        if len(counts) != 1 or not counts.issubset({4, 5}):
            raise ValueError("Inconsistent IWS method coverage")
        iws_count = counts.pop()
        if evidence["iws"]["unbounded_included"] != (iws_count == 5):
            raise ValueError("Unbounded reporting gate and method count disagree")
    reserved_pack = ROOT / "paper/figure_sources/iws_reserved_evidence"
    reserved_output = ROOT / "paper/generated/iws_reserved_evidence"
    reserved_sources = []
    reserved_ready = False
    if (reserved_output / "evidence.json").exists():
        reserved_manifest = json.loads((reserved_pack / "manifest.json").read_text())
        reserved_evidence = json.loads((reserved_output / "evidence.json").read_text())
        if (reserved_manifest.get("schema") != "iws_reserved_evidence_pack_recovery_v2"
                or reserved_manifest.get("status") != "complete36_validated"
                or len(reserved_manifest.get("runs", [])) != 36):
            raise ValueError("Reserved comparison index requires complete 36-run evidence")
        for name, expected in reserved_manifest["files_sha256"].items():
            if hashlib.sha256((reserved_pack / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Reserved portable source changed: " + name)
        if reserved_evidence["source_bindings"]["pack_manifest_sha256"] != hashlib.sha256((reserved_pack / "manifest.json").read_bytes()).hexdigest():
            raise ValueError("Reserved display has a different source pack")
        for name, expected in reserved_evidence["outputs_sha256"].items():
            if hashlib.sha256((reserved_output / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Reserved display changed: " + name)
        reserved_sources = [reserved_pack / "manifest.json", reserved_pack / "data.json",
                            reserved_output / "evidence.json", reserved_output / "scores.tex"]
        reserved_ready = True
    iws_scope = (f"{iws_count} predictors; development + reserved" if reserved_ready
                 else f"{iws_count} predictors; development")
    iws_location = (r"\ref{tab:iws-secondary}, \ref{tab:iws-reserved-scores}" if reserved_ready
                    else r"\ref{tab:iws-secondary}")
    external_sources = []
    external_ready = False
    external_pack = ROOT / 'paper/generated/external_dinowm'
    if (external_pack / 'evidence.json').exists():
        display = json.loads((external_pack / 'evidence.json').read_text())
        renderer = ROOT / 'paper/scripts/render_external_dinowm.py'
        if (display.get('status') != 'complete_verified'
                or display.get('renderer_sha256') != hashlib.sha256(renderer.read_bytes()).hexdigest()):
            raise ValueError('External index requires the reviewed complete display')
        for name, expected in display['outputs_sha256'].items():
            path = external_pack / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Changed external comparison display: ' + name)
            external_sources.append(path)
        spec = importlib.util.spec_from_file_location('inventory_external', renderer)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.from_pack(external_pack)
        external_sources += [renderer, external_pack / 'evidence.json']
        external_ready = True
    coverage = [
        ("DROID", "Recorded robot forecasting", "Native / pooled MSE; h5 / h10", "10 predictors; development" if external_ready else "8 predictors; development", r"\ref{tab:editorial-spatial}" if external_ready else r"\ref{tab:current-droid-scorecard}"),
        ("IWS PushT", "Recorded pushing forecast", "MSE / MAE / L1 / cosine", iws_scope, iws_location),
        ("IWS Box", "Recorded bimanual forecast", "MSE / MAE / L1 / cosine", iws_scope, iws_location),
        ("IWS Rope", "Recorded deformable forecast", "MSE / MAE / L1 / cosine", iws_scope, iws_location),
        ("Sim. PushT", "Closed-loop goal reaching", "Success; MSE@1/3/5", "6 predictors; original test", r"\ref{tab:simulator-core-consolidated}"),
        ("Sim. Reacher", "Closed-loop goal reaching", "Success; MSE@1/3/5", "6 predictors; original test", r"\ref{tab:simulator-core-consolidated}"),
        ("Sim. drone", "Planar flight goal reaching", "Success; MSE@1/3/5", "6 configurations + 2 revisions; development", r"\ref{tab:extension-drone}"),
        ("Sim. tissue", "Tissue-point goal reaching", "Success; MSE@1/3/5", "6 configurations; development", r"\ref{tab:extension-drone}"),
        ("Open-H", "Physical-phantom input audit", "No performance metric", "0 evaluated methods", r"\ref{app:comparison-inventory}"),
    ]
    external = [
        ("PushT / Reacher", "Frozen LeWM", "Success; forecast MSE", "Completed in historical grid; separate upstream check"),
        ("PushT", "AdaJEPA", "Success under matched budget", "Official reproduction complete; matched comparison pending"),
        ("PushT / Reacher", "DINO-WM; PLDM", "Success; forecast MSE", "Not run; match visual inputs and planner budget"),
        ("Drone", "PPO; PID", "Success; physical error; failures", "Not run; disclose privileged state inputs if used"),
        ("Tissue", "LapGym PPO", "Success; goal distance; failures", "Not run; match task and observation contract"),
        ("DROID", "LeWM; DINO-WM", "Feature MSE by horizon", "Not run under current forecast contract"),
        ("IWS PushT / Box / Rope", "RLA-WM", "Common-coordinate feature error", "Not run; resolve backbone access and representation mismatch"),
    ]
    if external_ready:
        external[5:6] = [
            ('DROID', 'Adapted official DINO-WM', 'Native / pooled MSE; h1--h10',
             r'Complete: two objectives, three seeds each; App.~\ref{app:external-dinowm}'),
            ('DROID', 'LeWM', 'Feature MSE by horizon', 'Not run under current forecast contract'),
        ]
    prefix = r"""% Generated by render_comparison_inventory.py; no performance values are inferred.
\begin{table}[!htb]\centering\small
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.10}
"""
    tex = prefix + r"""\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.13\linewidth}>{\raggedright\arraybackslash}p{.23\linewidth}>{\raggedright\arraybackslash}p{.23\linewidth}>{\raggedright\arraybackslash}p{.32\linewidth}@{}}
\toprule Benchmark & Task & Reported metrics & Completed comparison / location \\ \midrule
"""
    for task, description, metrics, scope, label in coverage:
        if label.startswith(r"\ref{tab:"):
            label = "Table~" + label
        elif task == "Open-H":
            label = "acquisition only"
        else:
            label = "App.~" + label
        tex += f"{task} & {description} & {metrics} & {scope}; {label} \\\\\n"
    tex += r"""\bottomrule\end{tabular}
\caption{\textbf{Benchmark and result-table index.} Counts include ours and internal
ablations, but exclude repeated seeds. DROID and IWS evaluate the current spatial model and comparison predictors;
the simulations use the historical context model. Drone/tissue have three method
families on two backbones; their final model tests remain pending. The separately
registered IWS unbounded component study is counted only after its complete-campaign
gate passes. MSE/MAE/L1/cosine
are feature errors (lower is better); success is higher is better. Open-H is an
input illustration, not an evaluated forecasting benchmark.}
\label{tab:comparison-coverage}\end{table}
"""
    plan = prefix + r"""\begin{tabular}{@{}>{\raggedright\arraybackslash}p{.15\linewidth}>{\raggedright\arraybackslash}p{.23\linewidth}>{\raggedright\arraybackslash}p{.22\linewidth}>{\raggedright\arraybackslash}p{.31\linewidth}@{}}
\toprule Benchmark & External method & Target metrics & Execution status / required match \\ \midrule
"""
    for row in external:
        plan += " & ".join(row) + " \\\\\n"
    plan += r"""\bottomrule\end{tabular}
\caption{\textbf{External-comparison completion plan for this research draft.}
Pending comparisons are not claimed as completed baselines. Internal controls and
component ablations remain in the numerical tables. AdaJEPA uses different inputs
and planning settings in its completed reproduction. RLA-WM uses a different
feature representation; native scores are not interchangeable with ours. There is
currently no completed matched external-method comparison for the spatial
recorded-video model. Pending cells are unavailable, not zero performance.}
\label{tab:comparison-pending}\end{table}
"""
    if external_ready:
        plan = plan.replace('There is\ncurrently no completed matched external-method comparison for the spatial\nrecorded-video model.',
            'The DROID DINO-WM adaptation uses the same recorded-video forecasting population; '
            'its pooled-token interface, parameter count, objectives and training budget are explicit '
            'in Appendix~\\ref{app:external-dinowm}. It is not a reproduction of the original planning benchmark.')
    sources = ["reports/benchmark_metrics_and_comparators_2026-09-20.md",
               "paper/sections/main_evaluation.tex", "paper/sections/domain_extensions.tex",
               "paper/sections/appendix/simulation_protocols.tex"]
    manifest = {
        "schema": "shiftwm_presentation_comparison_index_v1",
        "scope": "reporting_inventory_not_experiment_registration",
        "coverage": [dict(zip(("benchmark", "task", "metrics", "completed_scope", "location"), r)) for r in coverage],
        "external_plan": [dict(zip(("benchmark", "method", "target_metrics", "status"), r)) for r in external],
        "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    }
    if pack.exists():
        manifest["sources"].update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (pack, pack_manifest)})
    manifest["reserved_complete36_display_present"] = reserved_ready
    manifest['external_dinowm_complete_display_present'] = external_ready
    manifest["sources"].update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in reserved_sources})
    manifest['sources'].update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in external_sources})
    for name, content in (("coverage.tex", tex), ("external_plan.tex", plan),
                          ("inventory.json", json.dumps(manifest, indent=2) + "\n")):
        path = OUT / name
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    print(json.dumps({"status": "rendered", "benchmarks_evaluated": 8, "input_audits": 1}))


if __name__ == "__main__":
    main()
