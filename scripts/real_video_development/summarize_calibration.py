#!/usr/bin/env python3
"""All-method, all-seed validation report for the frozen calibration campaign."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/real_video"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_residual import verify_registration
from evaluate import crossed_session_bootstrap, summarize_episode_errors
from shiftwm.real_video.data import atomic_json, sha256
from shiftwm.real_video_development import residual_scale

MODES = ("framewise", "constant_dynamics", "factorized", "action_free")


def summarize(campaign_root, output):
    root, output = Path(campaign_root), Path(output)
    campaign = json.loads((root / "campaign.json").read_text())
    expected = {(mode, seed) for mode in MODES for seed in (0, 1, 2)}
    if campaign["status"] != "completed" or len(campaign["runs"]) != 12 or {(row["mode"], row["seed"]) for row in campaign["runs"]} != expected:
        raise ValueError("All 12 registered calibrated models must complete")
    if not campaign["payloads_verified_before_and_after"] or not campaign["all_offline_cpu_reloads_exact"] or campaign["test_evaluated"]:
        raise ValueError("Missing campaign audit or test leakage")
    registration_path = Path(campaign["registration_path"])
    if sha256(registration_path) != campaign["registration_sha256"]:
        raise ValueError("Changed registration")
    registration = json.loads(registration_path.read_text())
    verify_registration(registration)
    records, ledger, individual = {}, {}, []
    for mode, seed in sorted(expected):
        path = root / f"droid_{mode}_s{seed}"
        package = json.loads((path / "calibration.json").read_text())
        record = json.loads((path / "validation.json").read_text())
        if (record["mode"], record["seed"]) != (mode, seed) or record["split"] != "val":
            raise ValueError("Wrong development population or duplicate run")
        fit = package["fit"]
        if fit["fit_split"] != "train" or fit["scale"] != residual_scale(fit["displacement_energy"], fit["displacement_alignment"]):
            raise ValueError("Coefficient differs from training-only sufficient statistics")
        if package["registration_sha256"] != campaign["registration_sha256"]:
            raise ValueError("Package registration differs")
        if record["offline_reload"]["status"] != "passed" or record["offline_reload"]["max_abs_error"] != 0:
            raise ValueError("Offline reloaded predictions differ")
        if sha256(Path(package["base_checkpoint"]) / "model.pt") != package["base_checkpoint_sha256"]:
            raise ValueError("Original selected checkpoint changed")
        for horizon in (5, 10):
            for version in ("base", "calibrated"):
                result = record["horizons"][str(horizon)][version]
                if summarize_episode_errors(result["episodes"]) != result["summary"]:
                    raise ValueError("Validation summary does not match episode errors")
        records[(mode, seed)] = record
        ledger[str(path / "calibration.json")] = sha256(path / "calibration.json")
        ledger[str(path / "validation.json")] = sha256(path / "validation.json")
        individual.append({"mode": mode, "seed": seed, "scale": fit["scale"],
                           "train_episodes": fit["eligible_episodes"], "train_windows": fit["windows"],
                           **{f"h{horizon}_{version}": record["horizons"][str(horizon)][version]["summary"]["model"][f"h{horizon}_standardized_mse"]
                              for horizon in (5, 10) for version in ("base", "calibrated")}})
    grouped, comparisons = [], []
    populations = {}
    for horizon in (5, 10):
        metric = f"h{horizon}_standardized_mse"
        canonical = records[("framewise", 0)]["horizons"][str(horizon)]["base"]
        ids = [(row["episode_id"], row["session_id"], row["window_starts"]) for row in canonical["episodes"]]
        populations[str(horizon)] = {key: canonical[key] for key in ("episode_count", "session_count", "window_count")}
        for record in records.values():
            for version in ("base", "calibrated"):
                result = record["horizons"][str(horizon)][version]
                if [(row["episode_id"], row["session_id"], row["window_starts"]) for row in result["episodes"]] != ids:
                    raise ValueError("Unmatched validation episodes or windows")
                for row, reference in zip(result["episodes"], canonical["episodes"]):
                    if row["errors"]["persistence"] != reference["errors"]["persistence"]:
                        raise ValueError("Frozen-target persistence baseline differs")
        for mode in MODES:
            values = {version: float(np.mean([records[(mode, seed)]["horizons"][str(horizon)][version]["summary"]["model"][metric]
                                             for seed in (0, 1, 2)])) for version in ("base", "calibrated")}
            grouped.append({"mode": mode, "horizon": horizon, **values, "relative_error_reduction_percent": 100 * (1 - values["calibrated"] / values["base"])})
        for reference in ("framewise", "constant_dynamics", "action_free"):
            difference = []
            for seed in (0, 1, 2):
                ours = records[("factorized", seed)]["horizons"][str(horizon)]["calibrated"]["episodes"]
                comparator = records[(reference, seed)]["horizons"][str(horizon)]["calibrated"]["episodes"]
                difference.append([a["errors"]["model"][metric] - b["errors"]["model"][metric] for a, b in zip(ours, comparator)])
            bootstrap = crossed_session_bootstrap(difference, [row["session_id"] for row in canonical["episodes"]], draws=10000, seed=5192026 + horizon)
            ours_mean = next(row["calibrated"] for row in grouped if row["mode"] == "factorized" and row["horizon"] == horizon)
            reference_mean = next(row["calibrated"] for row in grouped if row["mode"] == reference and row["horizon"] == horizon)
            comparisons.append({"horizon": horizon, "reference": reference, "relative_reduction_percent": 100 * (1 - ours_mean / reference_mean), **bootstrap})
    report = {"status": "completed", "scope": "training-fitted calibration; validation development results only", "test_evaluated": False,
              "registration_sha256": campaign["registration_sha256"], "campaign_sha256": sha256(root / "campaign.json"),
              "summarizer_sha256": sha256(__file__), "individual": individual, "grouped": grouped,
              "comparisons": comparisons, "populations": populations, "inputs_sha256": ledger,
              "offline_reload": "12/12 exact CPU prediction parity", "calibration_is_novel_method": False}
    atomic_json(report, output.with_suffix(".json"))
    lines = ["# Full matched real-DROID residual-calibration development campaign", "",
             "All 12 frozen best checkpoints were calibrated using all eligible training recordings, with one scalar fitted independently per checkpoint. No gradients, architecture changes, or test payloads were used. Every method received the same calibration opportunity; all results below are validation development evidence.", "",
             "## All methods, three seeds", "", "| Method | Horizon | Original error | Calibrated error | Reduction from its own original model |", "|---|---:|---:|---:|---:|"]
    for row in grouped:
        lines.append(f"| {row['mode']} | {row['horizon']} | {row['base']:.6f} | {row['calibrated']:.6f} | {row['relative_error_reduction_percent']:+.3f}% |")
    lines += ["", "Errors are train-standardized frozen-feature MSE, averaging windows within episodes, then equally across episodes and the three seeds. Positive reductions favor calibration. All gains and losses are shown.", "",
              "## Fairly calibrated baselines", "", "| Horizon | Comparator (also calibrated) | Ours relative error reduction | Ours minus comparator [paired 95% interval] |", "|---:|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['horizon']} | {row['reference']} | {row['relative_reduction_percent']:+.3f}% | {row['mean_difference']:+.6f} [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] |")
    lines += ["", "Intervals resample recording sessions and training seeds (10,000 draws), preserving matched episode differences. They are exploratory validation intervals, unadjusted for multiple comparisons, and do not turn validation into a new test set.", "",
              "## Fitted scalar and complete per-seed outcomes", "", "| Method | Seed | Alpha | Fitting episodes / windows | Original h5 | Calibrated h5 | Original h10 | Calibrated h10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in individual:
        lines.append(f"| {row['mode']} | {row['seed']} | {row['scale']:.6f} | {row['train_episodes']} / {row['train_windows']} | {row['h5_base']:.6f} | {row['h5_calibrated']:.6f} | {row['h10_base']:.6f} | {row['h10_calibrated']:.6f} |")
    lines += ["", "All 851 train and 143 validation episode payloads were hash-verified against the immutable feature manifest at registration and before/after execution; short recordings remain in the audited manifests but cannot provide every horizon's windows.", ""]
    for horizon, counts in populations.items():
        lines.append(f"- Validation h{horizon}: {counts['episode_count']} episodes, {counts['session_count']} sessions, {counts['window_count']} windows.")
    lines += ["", "## Reuse and interpretation", "", "Each run's `calibration.json` records the fitted scalar, its training sufficient statistics, training payload hashes, source/protocol hashes, and immutable base checkpoint SHA256. `load_calibrated_package(path, device='cpu', base_checkpoint=relocated_base_directory)` supports relocation without changing the base weights. All 12 packages passed exact offline CPU prediction parity on four ten-step validation windows. The wrappers reuse trained checkpoints; no newly trained neural weights are claimed.", "",
              "The scalar is a standard constrained least-squares correction around persistence, not a new architectural contribution. Any positive outcome should motivate a carefully controlled causal adaptation study, not a claim of novelty, SOTA, or physical robot control. Preserve the original completed campaign. Freeze the final method and a fresh recording-session-disjoint data manifest before confirmatory testing; the already inspected original test set is not eligible for new confirmatory claims.", ""]
    output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"status": "completed", "runs": 12, "report": str(output.with_suffix('.md')), "fair_framewise": [row for row in comparisons if row["reference"] == "framewise"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--campaign", default="runs/real_video_development/residual_calibration_v1")
    parser.add_argument("--output", default="reports/real_droid_residual_calibration_results")
    args = parser.parse_args()
    summarize(args.campaign, args.output)
