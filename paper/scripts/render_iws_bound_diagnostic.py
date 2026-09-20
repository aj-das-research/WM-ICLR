#!/usr/bin/env python3
"""Render the completed analytical diagnostic without opening feature payloads."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/iws_bound_diagnostic_v1"
OUT = ROOT / "paper/generated/iws_bound_diagnostic"
TASKS = ("pusht", "bimanual_box", "bimanual_rope")
LABELS = ("PushT", "Box", "Rope")
MODES = ("bounded_spatial_mix", "unbounded_spatial_mix")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(source=SOURCE):
    paths = {n: source / (n + ".json") for n in
             ("registration", "source_review", "results", "completion")}
    values = {n: json.loads(p.read_text()) for n, p in paths.items()}
    reg, review, result, done = (values[n] for n in paths)
    require(review["status"] == result["status"] == done["status"] == "passed",
            "The entire diagnostic and its independent source review must pass")
    for receipt in (review, result, done):
        require(receipt["registration_sha256"] == sha(paths["registration"]),
                "Diagnostic registration binding differs")
    require(done["results_sha256"] == sha(paths["results"]), "Diagnostic result changed")
    require(result["protocol"] == reg["protocol"], "Diagnostic protocol changed")
    require(set(result["tasks"]) == set(TASKS), "Incomplete three-task result")
    require(not result["new_model_inference"] and result["reserved_or_test_payloads_read"] == 0,
            "Wrong diagnostic scope")
    require(reg["protocol"]["target_offsets"] == list(range(1, 60)), "Incomplete offsets")
    selected = {(r["task"], r["mode"], r["seed"]) for r in reg["selected_models"]}
    require(selected == {(t, m, s) for t in TASKS for m in MODES for s in range(3)},
            "Incomplete selected-model comparison")
    rows = []
    for task, label in zip(TASKS, LABELS):
        data = result["tasks"][task]
        episodes = data["episodes"]
        expected = reg["tasks"][task]["records"]
        require(data["population"] == reg["tasks"][task]["audit"], "Population changed")
        require([(e["episode_id"], e["windows"]) for e in episodes] ==
                [(e["episode_id"], e["windows"]) for e in expected], "Episode identity changed")
        require(all(type(e["windows"]) is int and e["windows"] > 0 for e in episodes),
                "Invalid window counts")
        weights = np.array([e["windows"] for e in episodes])
        for metric in data["equal_trajectory"]:
            matrix = np.array([e[metric] for e in episodes], dtype=np.float64)
            require(matrix.shape == (len(episodes), 59) and np.isfinite(matrix).all()
                    and (matrix >= 0).all(), "Invalid analytical curve")
            if "fraction" in metric or metric == "any_coordinate_violation":
                require((matrix <= 1).all(), "Invalid violation proportion")
            np.testing.assert_allclose(matrix.mean(0), data["equal_trajectory"][metric],
                                       rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(np.average(matrix, axis=0, weights=weights),
                                       data["equal_window"][metric], rtol=1e-12, atol=1e-14)
        curves = data["equal_trajectory"]
        require(np.all(np.array(curves["roundoff_screened_relaxed_mse_lower_bound"]) <=
                       np.array(curves["relaxed_mse_lower_bound"])), "Roundoff screen increased floor")
        endpoints = {}
        for mode in MODES:
            per_seed = []
            for seed in range(3):
                matrix = np.array([e["saved_model_standardized_mse"][mode][str(seed)]
                                   for e in episodes], dtype=np.float64)
                require(matrix.shape == (len(episodes), 59) and np.isfinite(matrix).all()
                        and (matrix >= 0).all(), "Invalid saved-model errors")
                per_seed.append(matrix.mean(0))
            saved = data["saved_model_errors"][mode]
            np.testing.assert_allclose(per_seed, saved["per_seed_curves"], rtol=1e-12, atol=1e-14)
            curve = np.mean(per_seed, axis=0)
            np.testing.assert_allclose(curve, saved["equal_seed_equal_trajectory_curve"],
                                       rtol=1e-12, atol=1e-14)
            endpoints[mode] = float(curve[-1])
        rows.append({"task": task, "label": label, "trajectories": len(episodes),
                     "windows": int(weights.sum()),
                     "outside_percent": 100 * curves["coordinate_violation_fraction"][-1],
                     "relaxed_mse_floor": curves["relaxed_mse_lower_bound"][-1],
                     "roundoff_screened_mse_floor": curves["roundoff_screened_relaxed_mse_lower_bound"][-1],
                     "bounded_mse": endpoints[MODES[0]], "no_tanh_mse": endpoints[MODES[1]]})
    require(sum(r["trajectories"] for r in rows) == 362 and sum(r["windows"] for r in rows) == 10136,
            "Incomplete development population")
    return rows, {str(p.relative_to(ROOT)): sha(p) for p in paths.values()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    if args.if_ready and not (SOURCE / "completion.json").exists():
        require(not (OUT / "evidence.json").exists(), "Existing display has lost its completion receipt")
        print(json.dumps({"status": "pending"}))
        return
    rows, sources = verify()
    tex = r"""% Generated from the complete development-only analytical audit.
\begin{table}[!htbp]\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Task & Outside (\%) & Relaxed floor & Bounded MSE & No-$\tanh$ MSE \\
\midrule
"""
    for row in rows:
        tex += (f"{row['label']} & {row['outside_percent']:.3f} & "
                f"{row['relaxed_mse_floor']:.6f} & {row['bounded_mse']:.6f} & "
                f"{row['no_tanh_mse']:.6f} \\\\\n")
    tex += r"""\bottomrule\end{tabular}
\caption{\textbf{Testing the correction-range limitation.} All 10,136 internal
development windows from 362 trajectories, at offset 59. Outside counts feature
coordinates beyond the observed channel range expanded by one; Relaxed floor
is Equation~\ref{eq:iws-box-floor}'s real-arithmetic box relaxation.
Windows average within trajectories, then trajectories equally; saved predictor
errors additionally average three seeds. Full curves and the fixed roundoff
sensitivity accompany the source evidence.}
\label{tab:iws-bound-diagnostic}\end{table}
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.tex").write_text(tex)
    evidence = {"schema": "iws_bound_diagnostic_display_v1", "status": "complete_verified",
                "sources_sha256": sources, "renderer_sha256": sha(Path(__file__)),
                "rows": rows, "table_sha256": sha(OUT / "table.tex"),
                "new_model_inference": False, "scope": "post-development analytical audit"}
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": "rendered", "rows": rows}))


if __name__ == "__main__":
    main()
