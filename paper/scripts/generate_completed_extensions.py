#!/usr/bin/env python3
"""Typeset the audited, completed extension results; never launch experiments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "paper/generated/extensions_completed"
SOURCE = ROOT / "reports/completed_extension_results.json"
LABEL = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics", "factorized": "ShiftWM (ours)"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gain(value):
    text = f"{value:+.2f}"
    return r"\positivegain{" + text + "}" if value > 1e-10 else text


def interval(row):
    lo, hi = row["ci95_percentage_points"]
    return f"$[{lo:+.2f}, {hi:+.2f}]$"


def rows_for(rows, domain, family, mode):
    selected = sorted([r for r in rows if (r["domain"], r["architecture"], r["mode"]) == (domain, family, mode)], key=lambda r:r["training_seed"])
    assert [r["training_seed"] for r in selected] == [0, 1, 2]
    assert all(r["tasks"] == 8 and r["training_summary"]["completed_epochs"] == 30 for r in selected)
    return selected


def method_row(rows, label):
    total = sum(r["successes"] for r in rows)
    mse = 1000 * sum(r["forecast_mse"]["h5"] for r in rows) / 3
    counts = " & ".join(f"{r['successes']}/8" for r in rows)
    return f"{label} & {counts} & {100 * total / 24:.2f} & {mse:.3f}" + r" \\"


def main():
    report = json.loads(SOURCE.read_text())
    assert report["status"] == "verified_complete"
    assert report["completion"] == {"original_full_30_epoch_runs": 36, "original_forecasts": 36,
        "original_planning": 36, "geometry_full_30_epoch_runs": 6, "geometry_forecasts": 6,
        "geometry_planning": 6, "official_adajepa_stages": 4}
    assert all(report["verification"][key] for key in (
        "all_training_completion_and_strict_load_checks", "all_result_embedded_identities_equal_sidecars",
        "all_recorded_source_protocol_checkpoint_and_trace_hashes_match", "all_planning_counts_recomputed",
        "all_paired_tasks_initial_metrics_and_support_prefixes_equal", "all_forecast_means_recomputed"))
    DEST.mkdir(parents=True, exist_ok=True)
    header = [r"\begin{tabular}{lrrrrr}", r"\toprule",
              r"Method & Seed 0 & Seed 1 & Seed 2 & Success (\%) & MSE@5 ($10^{-3}$) \\", r"\midrule"]
    lines = header.copy()
    for domain in ("drone", "surgery"):
        for family in ("transformer", "gru"):
            name = "Drone" if domain == "drone" else "Tissue manipulation"
            lines.append(r"\multicolumn{6}{l}{\emph{" + name + " / " + ("Transformer" if family == "transformer" else "GRU") + r"}} \\")
            for mode in ("framewise", "constant_dynamics", "factorized"):
                lines.append(method_row(rows_for(report["original_runs"], domain, family, mode), LABEL[mode]))
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DEST / "all_development.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lllrr}", r"\toprule",
             r"Domain & Predictor & Comparator & $\Delta$ (pp) & Paired 95\% interval (pp) \\", r"\midrule"]
    for row in report["original_planning_comparisons"]:
        domain = "Drone" if row["domain"] == "drone" else "Tissue"
        family = "Transformer" if row["architecture"] == "transformer" else "GRU"
        lines.append(f"{domain} & {family} & {LABEL[row['baseline']]} & {gain(row['success_difference_percentage_points'])} & {interval(row)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DEST / "paired_development.tex").write_text("\n".join(lines) + "\n")

    lines = header.copy()
    for variant, runs in (("Original narrow gain", report["original_runs"]), ("Revised wide gain", report["geometry_runs"])):
        lines.append(r"\multicolumn{6}{l}{\emph{" + variant + r"}} \\")
        for mode in ("constant_dynamics", "factorized"):
            lines.append(method_row(rows_for(runs, "drone", "transformer", mode), LABEL[mode]))
        lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{5pt}", r"\begin{tabular}{lrr}", r"\toprule",
              r"Paired contrast & $\Delta$ (pp) & Paired 95\% interval (pp) \\", r"\midrule"]
    names = {"Wide ShiftWM minus wide constant": "Wide ShiftWM (ours) $-$ wide constant",
             "Wide ShiftWM minus original ShiftWM": "Wide ShiftWM (ours) $-$ original ShiftWM (ours)",
             "Wide constant minus original constant": "Wide constant $-$ original constant"}
    for row in report["geometry_planning_comparisons"]:
        lines.append(f"{names[row['comparison']]} & {gain(row['success_difference_percentage_points'])} & {interval(row)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DEST / "geometry_ablation.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Observation & Official frozen & Official adaptive & $\Delta$ (pp) \\", r"\midrule"]
    for condition in ("clean", "blur"):
        stages = {r["method"]:r for r in report["official_adajepa"]["stages"] if r["condition"] == condition}
        assert all(r["episodes"] == 50 and r["seed"] == 100 for r in stages.values())
        cells = [f"{stages[method]['successes']}/50 ({100*stages[method]['success_rate']:.0f}\\%)" for method in ("frozen", "adaptive")]
        delta = report["official_adajepa"]["adaptive_minus_frozen_pp"][condition]
        lines.append(f"{condition.title()} & {cells[0]} & {cells[1]} & {gain(delta)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DEST / "official_adajepa.tex").write_text("\n".join(lines) + "\n")
    ledger = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha(SOURCE),
              "source_verified_at_utc": report["generated_at_utc"],
              "generator_sha256": sha(Path(__file__)),
              "scope": report["scope"], "completed": report["completion"],
              "positive_style": "positivegain macro marks positive point differences only; intervals are retained, with no significance claim",
              "outputs_sha256": {str(p.relative_to(ROOT)):sha(p) for p in sorted(DEST.glob("*.tex"))}}
    (DEST / "table_provenance.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"source_sha256":ledger["source_sha256"], "outputs":list(ledger["outputs_sha256"])}))


if __name__ == "__main__":
    main()
