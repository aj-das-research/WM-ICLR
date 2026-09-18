#!/usr/bin/env python3
"""Render validation-only development diagnosis with no test result inputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def _mean(runs, metric, stratum="all", split="validation"):
    arrays = [run[split]["strata"][stratum]["metrics"][metric] for run in runs]
    return np.asarray(arrays, dtype=float).mean(0)


def interpolation_error(persistence, displacement, alignment, scale):
    """Exact ||scale * predicted_displacement - true_displacement||² identity."""
    return np.asarray(persistence) + scale * scale * np.asarray(displacement) - 2 * scale * np.asarray(alignment)


def render(data):
    if data["status"] != "completed":
        raise ValueError("Only complete diagnostics can be summarized")
    groups = {(mode, which): [row for row in data["runs"] if row["mode"] == mode and row["checkpoint"] == which]
              for mode, which in [("framewise", "best"), ("constant_dynamics", "best"),
                                  ("factorized", "best"), ("action_free", "best"), ("factorized", "last")]}
    if any(len(rows) != 3 or {row["seed"] for row in rows} != {0, 1, 2} for rows in groups.values()):
        raise ValueError("All 12 best and three late-checkpoint diagnostics are required")
    first = data["runs"][0]["validation"]
    lines = ["# Real DROID development diagnosis", "",
             "This report uses only training and validation recordings. No test feature/video payload was read by the diagnostic or this summary. The original completed campaign remains unchanged.", "",
             f"The validation population contains {first['episodes']} episodes from {first['sessions']} sessions and {first['windows']} windows. All horizons below use the same ten-block-eligible recordings. Windows are averaged within episodes, then episodes and the three training seeds are weighted equally. Results are exploratory development evidence, without confirmatory intervals.", "",
             "## Recursive forecasting and observed motion", "",
             "| Model | h1 error | h5 error | h10 error | h10 predicted displacement | h10 observed displacement | h10 oracle-refresh error |", "|---|---:|---:|---:|---:|---:|---:|"]
    for (mode, which), rows in groups.items():
        error = _mean(rows, "model_mse")
        lines.append(f"| {mode}, {which} | {error[0]:.6f} | {error[4]:.6f} | {error[9]:.6f} | {_mean(rows, 'predicted_displacement_mse')[9]:.6f} | {_mean(rows, 'persistence_mse')[9]:.6f} | {_mean(rows, 'oracle_observation_refresh_mse')[9]:.6f} |")
    persistence = _mean(groups[("factorized", "best")], "persistence_mse")
    lines += [f"| Persistence | {persistence[0]:.6f} | {persistence[4]:.6f} | {persistence[9]:.6f} | 0 | {persistence[9]:.6f} | not applicable |", "",
              "Errors and displacement magnitudes are standardized feature mean-square values, not physical distances. Oracle refresh supplies the actual three immediately preceding observations for each one-step prediction; it changes the available information and refreshes inferred contexts. Its lower error cannot be claimed as a deployable algorithm improvement or attributed solely to one source of drift.", "",
              "## Stronger future-command sensitivity", "",
              "Positive error change means the perturbation made prediction worse against the original recorded future. Prediction change measures output sensitivity separately from this error. None of these perturbed commands was executed on hardware.", "",
              "| Model / checkpoint | Future-command perturbation | Standardized command change | h10 prediction change MSE | h5 error change | h10 error change |", "|---|---|---:|---:|---:|---:|"]
    for mode, which in groups:
        rows = groups[(mode, which)]
        base = _mean(rows, "model_mse")
        for name in ("reversed", "permuted_recording", "raw_zero", "train_mean", "hold_support_command"):
            changed = _mean(rows, name + "_mse")
            lines.append(f"| {mode}, {which} | {name} | {float(_mean(rows, name + '_command_change')):.5f} | {_mean(rows, name + '_prediction_change')[9]:.7f} | {(changed[4] / base[4] - 1) * 100:+.3f}% | {(changed[9] / base[9] - 1) * 100:+.3f}% |")
    lines += ["", "Raw-zero and training-mean values may be out of context for absolute robot position commands. Donor commands come from different recording sessions. These stress tests can expose dependence on actions but cannot assess physical counterfactual accuracy.", "",
              "## Where errors occur", "",
              "Tertile boundaries are fixed from all training windows. Future-motion strata are descriptive uses of targets, not inputs to an online decision rule.", "",
              "| Stratum | Eligible validation windows | Framewise h10 | Ours h10 | Persistence h10 | Ours h5 gain vs persistence |", "|---|---:|---:|---:|---:|---:|"]
    for stratum in first["strata"]:
        ours = _mean(groups[("factorized", "best")], "model_mse", stratum)
        framewise = _mean(groups[("framewise", "best")], "model_mse", stratum)
        persistence = _mean(groups[("factorized", "best")], "persistence_mse", stratum)
        count = first["strata"][stratum]["windows"]
        lines.append(f"| {stratum} | {count} | {framewise[9]:.6f} | {ours[9]:.6f} | {persistence[9]:.6f} | {(1 - ours[4] / persistence[4]) * 100:+.2f}% |")
    lines += ["", "## Training-versus-validation overfitting", "",
              "| Seed-0 factorized checkpoint | Train-subset h5 | Train-subset h10 | Validation h5 | Validation h10 | Train-fitted residual scale | Validation h10 with this fixed scale |", "|---|---:|---:|---:|---:|---:|---:|"]
    calibrations = []
    for which in ("best", "last"):
        row = next(row for row in groups[("factorized", which)] if row["seed"] == 0)
        training = row["train_subset"]["strata"]["all"]["metrics"]
        validation = row["validation"]["strata"]["all"]["metrics"]
        # A single global least-squares scale fit on training episodes only.
        # It is deliberately not selected using validation, horizons, or tests.
        scale = float(np.clip(np.mean(training["displacement_alignment"]) / max(np.mean(training["predicted_displacement_mse"]), 1e-20), 0, 1))
        calibrated = interpolation_error(validation["persistence_mse"], validation["predicted_displacement_mse"], validation["displacement_alignment"], scale)
        calibrations.append({"checkpoint": which, "seed": 0, "fit_split": "train_subset", "global_scale": scale,
                             "validation_mse_all_horizons": calibrated.tolist(), "not_a_new_trained_checkpoint": True})
        lines.append(f"| {which} | {training['model_mse'][4]:.6f} | {training['model_mse'][9]:.6f} | {validation['model_mse'][4]:.6f} | {validation['model_mse'][9]:.6f} | {scale:.5f} | {calibrated[9]:.6f} |")
    lines += ["", "The 64 training episodes were selected deterministically without inspecting losses. The residual scale is the clipped least-squares coefficient fitted to train-subset predicted/observed displacement moments, pooled over all ten horizons. This is a numerical diagnosis, not a separately trained architecture or a reported test improvement.", "",
              "## Pooling and spatial information", "",
              f"On {len(data['pooling']['pairs'])} deterministic training frame pairs, median retained raw DINO feature-change energy was " + ", ".join(f"{float(value) * 100:.2f}% at {key}×{key}" for key, value in data["pooling"]["median_retained_fraction"].items()) + ".", "",
              "This establishes strong contraction of spatial variation under pooling if retention is low. It does not establish that manipulated-object information is the particular lost signal, or that restoring patch resolution will improve validation forecasting. No object masks were used, and the current experiment did not train a competing patch model.", "",
              "## Rules for the next controlled experiment", "",
              "1. Preserve all original positive, null, and negative results and released checkpoints.",
              "2. Choose a narrowly defined fix using this train/validation diagnosis; freeze its protocol and source before running new test evaluation.",
              "3. Tune only on training/validation data. Maintain matched architectures, budgets, and validation selection for the corresponding baseline.",
              "4. Obtain an untouched evaluation population from new recording sessions absent from the original 433 sessions. Exclude overlap before viewing frames or results; record deterministic inclusion and all exclusions.",
              "5. Call any reuse of the already revealed original test set exploratory, not confirmatory. Do not claim SOTA or physical robot success from latent forecasting alone.", ""]
    return "\n".join(lines), calibrations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", default="reports/real_droid_development_diagnosis.json")
    args = parser.parse_args()
    path = Path(args.input)
    data = json.loads(path.read_text())
    markdown, calibration = render(data)
    path.with_suffix(".md").write_text(markdown)
    path.with_name(path.stem + "_calibration.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "calibrations": calibration}, indent=2) + "\n")
