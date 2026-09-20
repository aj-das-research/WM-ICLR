"""Presentation-only tables from independently reviewed fixed diagnostics.

This new renderer is not a dependency of the frozen diagnostic registration.
It never opens model packages, features, simulator traces, or held-out payloads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/extensions_diagnostics_20260920"
DEST = ROOT / "paper/generated/extensions_completed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, text):
    content = text.encode()
    if path.exists() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def payload(source=SOURCE):
    registry = read(source / "registration.json")
    result = read(source / "results.json")
    completion = read(source / "completion.json")
    review = read(source / "execution_review.json")
    binds = {name: sha(source / (name + ".json")) for name in ("registration", "results", "completion")}
    if (review.get("status") != "passed" or
            any(review.get(name + "_sha256") != value for name, value in binds.items())):
        raise ValueError("Independent execution review is missing or stale")
    if (result["status"] != "completed_development_diagnostic" or
            completion["status"] != result["status"] or
            completion["results_sha256"] != binds["results"] or
            completion["registration_sha256"] != binds["registration"] or
            result["registration_sha256"] != binds["registration"] or
            completion["all_registered_inputs_unchanged"] is not True):
        raise ValueError("Diagnostic completion identity failed")
    models = result["drone"]["models"]
    expected = {(f, m, s) for f in ("narrow", "wide") for m in ("constant_dynamics", "factorized") for s in range(3)}
    actual = {(r["family"], r["mode"], r["seed"]) for r in models}
    if len(models) != 12 or actual != expected or actual != {(r["family"], r["mode"], r["seed"]) for r in registry["models"]}:
        raise ValueError("Incomplete or duplicated model population")
    if any(result["drone"][k] != v for k, v in (("goal_count", 16), ("pair_count", 120), ("fixed_support_contexts", 16))):
        raise ValueError("Wrong diagnostic goal/context population")
    rows = []
    for mode in ("constant_dynamics", "factorized"):
        for seed in range(3):
            pair = {f: next(r for r in models if (r["family"], r["mode"], r["seed"]) == (f, mode, seed))
                    for f in ("narrow", "wide")}
            rows.append({"mode": mode, "seed": seed,
                         **{f: {"checkpoint_sha256": pair[f]["checkpoint_sha256"],
                                "selected_epoch": pair[f]["selected_epoch"],
                                **pair[f]["mean_over_fixed_contexts"]} for f in pair}})
    tissue = result["tissue"]
    if len(tissue["runs"]) != 18 or tissue["counts"]["tasks"] != 144:
        raise ValueError("Incomplete tissue audit")
    return {"status": "reviewed_development_diagnostic", "scope": result["drone"]["scope"],
            "job_id": result["job_id"], "goal_count": 16, "pair_count": 120, "fixed_contexts": 16,
            "baselines": result["drone"]["baselines"], "rows": rows,
            "tissue": {"scope": tissue["scope"], "counts": tissue["counts"], "failure_stops": tissue["failure_stops"]},
            "source_sha256": {str((source / name).relative_to(ROOT)): sha(source / name) for name in
                              ("registration.json", "results.json", "completion.json", "source_review.json", "execution_review.json")},
            "renderer_sha256": sha(__file__)}


def geometry_tex(data):
    rows = []
    for r in data["rows"]:
        label = "Constant dynamics" if r["mode"] == "constant_dynamics" else "ShiftWM (ours)"
        a, b = r["narrow"], r["wide"]
        rows.append(f"{label} & {r['seed']} & {100*a['fraction_of_canonical_pairwise_mse']:.3f} & "
                    f"{100*b['fraction_of_canonical_pairwise_mse']:.3f} & "
                    f"{a['spearman_with_squared_xy_distance']:.5f} & {b['spearman_with_squared_xy_distance']:.5f} \\\\")
        if r["seed"] == 2 and r["mode"] == "constant_dynamics":
            rows.append(r"\midrule")
    return r"""% Generated only from the completed, independently reviewed development diagnostic.
\begin{table}[t]
\centering
\fontsize{9}{11}\selectfont
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llrrrr}
\toprule
 & & \multicolumn{2}{c}{Canonical separation (\%)} & \multicolumn{2}{c}{Physical-distance $\rho$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Method & Seed & Narrow & Wide & Narrow & Wide \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Fixed-goal drone geometry after widening the observation gain.}
All twelve selected transformer checkpoints use the same 16 development goals
(120 pairs), averaged over 16 fixed support contexts. Separation is mean
pairwise latent MSE divided by its canonical value (0.01020543); $\rho$ is
Spearman correlation with squared physical XY distance. Canonical and
uncorrected warm $\rho$ are 0.61374 and 0.48985. Warm features retain 17.400\%
of canonical separation. These dependent pairs are descriptive diagnostics,
not independent trials or evidence of planning benefit; final tests remain gated.}
\label{tab:extension-geometry-diagnostic}
\end{table}
"""


def tissue_tex(data):
    c, s = data["tissue"]["counts"], data["tissue"]["failure_stops"]
    rows = [("Successful method/seed/task instances", f"{c['successes']} / {c['tasks']}"),
            ("Failures never entering 2 mm", f"{c['failed_never_enter_2mm']} / {c['failures']}"),
            ("Failure stop: budget / invalid action", f"{s['budget']} / {s['invalid_action']}"),
            ("Native steps / invalid / unstable", f"{c['native_steps']:,} / {c['invalid_native_steps']} / {c['unstable_native_steps']}"),
            ("Complete decisions reducing target distance", f"{c['positive_progress_decisions']:,} / {c['complete_five_command_decisions']:,}")]
    return r"""% Saved original development traces only; no new simulator execution.
\begin{table}[t]
\centering
\fontsize{9}{11}\selectfont
\setlength{\tabcolsep}{7pt}
\begin{tabular}{lr}
\toprule
Saved-trace descriptor & Count \\
\midrule
""" + "\n".join(label + " & " + value + r" \\" for label, value in rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Tissue failure audit on the completed development set.}
All 18 original policy runs are retained (two predictor families, three methods,
three seeds), each on the same eight goals. Invalid actions remain failures.
The decision descriptor uses exactly five executed commands; partial terminal
blocks remain in outcome counts. These are repeated method/seed/task instances,
not independent goals or counterfactual candidate-ranking measurements.}
\label{tab:extension-tissue-failure-audit}
\end{table}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    if args.if_ready and not (SOURCE / "execution_review.json").exists():
        print(json.dumps({"status": "pending", "outputs_written": False}))
        return
    data = payload()
    put(DEST / "geometry_diagnostic_20260920.tex", geometry_tex(data))
    put(DEST / "tissue_failure_audit_20260920.tex", tissue_tex(data))
    put(DEST / "geometry_diagnostic_20260920.json", json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "rendered", "tables": 2}))


if __name__ == "__main__":
    main()
