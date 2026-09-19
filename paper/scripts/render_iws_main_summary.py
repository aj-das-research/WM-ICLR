#!/usr/bin/env python3
"""Compact main-paper IWS comparison, using only the full validated campaign."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/generated/experiment_alignment"
spec = importlib.util.spec_from_file_location("iws_verified_plot", Path(__file__).with_name("render_iws_results.py"))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def gain_tex(value):
    if value is None:
        return "undefined"
    text = f"{value:+.2f}" + r"\%"
    return r"\positivegain{" + text + "}" if value > 0 else text


def secondary_table(development, sources):
    metrics = ("standardized_mae", "raw_dinov2_l1", "feature_cosine_distance")
    methods = ("persistence", "autoregressive", "anchored_additive", "bounded_spatial_mix")
    values = {task: {mode: [] for mode in methods} for task in helper.TASKS}
    for run in development["runs"]:
        path = ROOT / run["evaluation_path"]
        if helper.sha(path) != run["evaluation_sha256"] or sources[run["evaluation_path"]] != run["evaluation_sha256"]:
            raise ValueError("Secondary metric source is not bound by the completed study")
        receipt = json.loads(path.read_text())
        selections = [(run["mode"], "")]
        if run["mode"] == "autoregressive":
            selections.append(("persistence", "persistence_"))
        for mode, prefix in selections:
            rows = np.asarray([[ep[prefix+metric+"_by_offset"][-1] for metric in metrics] for ep in receipt["episodes"]], dtype=np.float64)
            if rows.shape != (receipt["eligible_trajectories"], 3) or not np.isfinite(rows).all() or (rows < 0).any():
                raise ValueError("Invalid complete secondary endpoint metrics")
            values[run["task"]][mode].append({"seed": run["seed"], "means": rows.mean(0).tolist()})
    rows, payload = [], {}
    for task in helper.TASKS:
        payload[task] = {}
        for mode in methods:
            entries = values[task][mode]
            if sorted(row["seed"] for row in entries) != [0, 1, 2]:
                raise ValueError("Incomplete secondary metric seed grid")
            means = np.asarray([row["means"] for row in entries]).mean(0)
            payload[task][mode] = dict(zip(metrics, means.tolist()))
        for mode in methods:
            cells = []
            for metric in metrics:
                value = payload[task][mode][metric]
                text = f"{value:.6f}"
                if value == min(payload[task][m][metric] for m in methods):
                    text = r"\textbf{"+text+"}"
                cells.append(text)
            rows.append(" & ".join([helper.NAMES[task], helper.LABELS[mode], *cells])+r" \\")
    tex = (r"\begin{table}[!htb]\centering"+"\n"
           r"\caption{Complete IWS secondary development endpoints at $H=60$. Lower errors are better. Means weight windows within trajectory, trajectories and three matched seeds equally. The smallest point mean per task and metric is bold; bold does not indicate statistical significance. Raw L1 is in the original DINOv2 coordinates, while standardized MAE uses task-specific training scales. Cosine distance is computed on flattened raw feature vectors. These secondary metrics do not replace the prespecified MSE comparison.}"+"\n"
           r"\label{tab:iws-secondary}"+"\n"
           r"\begingroup\small\setlength{\tabcolsep}{4pt}"+"\n"
           r"\begin{tabular}{@{}llrrr@{}}\toprule Task & Method & Std. MAE & Raw L1 & Cosine distance \\ \midrule"+"\n"
           +"\n".join(rows)+"\n"+r"\bottomrule\end{tabular}\endgroup\end{table}"+"\n")
    return tex, payload


def render(if_ready=False):
    if not helper.FINAL.exists():
        if if_ready:
            return {"status": "pending", "outputs_written": False}
        raise ValueError("Complete campaign finalizer required")
    renderer_sources = {str(Path(__file__).relative_to(ROOT)): helper.sha(__file__),
                        str(Path(helper.__file__).relative_to(ROOT)): helper.sha(helper.__file__)}
    development, sources = helper.load_evidence()
    payload = helper.plot_payload(development)
    secondary_tex, secondary_payload = secondary_table(development, sources)
    rows, gains, ar_gains = [], [], []
    modes = ("persistence", "autoregressive", "anchored_additive", "bounded_spatial_mix")
    for task in helper.TASKS:
        values = payload["tasks"][task]
        endpoints = values["h60_endpoints"]
        best = min(endpoints.values())
        scores = [(r"\textbf{"+f"{endpoints[m]:.5f}"+"}") if endpoints[m] == best else f"{endpoints[m]:.5f}" for m in modes]
        gain = values["primary_gain_percent"]
        ci = values["primary_gain_ci95_percent"]
        if gain is None or ci is None:
            raise ValueError("Main summary requires finite primary gain and interval")
        cell = gain_tex(gain) + f" [{ci[0]:+.2f}, {ci[1]:+.2f}]"
        rows.append(" & ".join([helper.NAMES[task], *scores, cell]) + r" \\")
        gains.append(gain)
        ar_gains.append(development["numerical_results"][task]["h60_comparisons"]["autoregressive"]["relative_mse_reduction_percent"])
    macro = development["macro_primary_relative_gain_percent"]
    ci = development["bootstrap"]["macro_gain_interval"]
    if macro is None or ci is None or any(g is None for g in ar_gains):
        raise ValueError("Main summary requires a defined complete comparison")
    count = ("None", "One", "Two", "All three")[sum(g > 0 for g in gains)]
    signs = f"{count} of three tasks have a favorable point estimate against additive anchoring." if count != "All three" else "All three tasks have a favorable point estimate against additive anchoring."
    if all(g < 0 for g in ar_gains):
        secondary = "Autoregression has lower endpoint error than ShiftWM on all three tasks."
    else:
        secondary = f"ShiftWM has a favorable point estimate against autoregression on {sum(g > 0 for g in ar_gains)} of three tasks."
    text = (r"\paragraph{Single-observation decoder transfer.}" + "\n"
            "A separate 27-model study tests one observed image and 60 native command rows on real IWS PushT, Box and Rope recordings. "
            r"Table~\ref{tab:iws-main} reports all four predictors at stored offset 59 ($H=60$). "
            f"The equal-task mean relative reduction against additive anchoring is {macro:+.2f}" + r"\%"
            + f" (paired 95" + r"\%" + f" interval [{ci[0]:+.2f}, {ci[1]:+.2f}]" + r"\%). "
            + signs + " " + secondary + " "
            "Thus the DROID improvement does not extend to a universal advantage for fixed-source decoding. "
            r"These are separately trained development comparisons; complete curves, uncertainty and interfaces appear in Appendix~\ref{app:experiment-alignment}." + "\n\n"
            r"\begin{table}[!htb]\centering" + "\n"
            r"\caption{Single-observation IWS development endpoints. Lower standardized feature MSE is better; the lowest mean per task is bold. Gains compare ShiftWM with the additive anchor; both point estimates and paired 95\% intervals are in percent. Positive point estimates are bold green, whether or not their interval excludes zero. All tasks, matched seeds and controls are retained.}" + "\n"
            r"\label{tab:iws-main}" + "\n"
            r"\begingroup\small\setlength{\tabcolsep}{3.5pt}" + "\n"
            r"\begin{tabular}{@{}lrrrrl@{}}\toprule" + "\n"
            r"Task & Persistence & AR & \shortstack{Additive\\anchor} & \shortstack{ShiftWM\\(ours)} & \shortstack[l]{Gain vs anchor [95\% CI]} \\ \midrule" + "\n"
            + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\endgroup\end{table}" + "\n")
    record = {"schema": "shiftwm_iws_main_summary_v1", "status": "complete_validated_development",
              "finalization_sha256": development["finalization_sha256"], "source_dependencies": sources,
              "renderer_sha256": renderer_sources[str(Path(__file__).relative_to(ROOT))],
              "renderer_sources_sha256": renderer_sources, "numerical_payload": payload,
              "main_transfer_tex_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "macro_primary_relative_gain_percent": macro, "macro_ci95_percent": ci,
              "secondary_metrics_h60": secondary_payload,
              "secondary_metrics_tex_sha256": hashlib.sha256(secondary_tex.encode()).hexdigest(),
              "gain_intervals_units": "percent relative MSE reduction; not raw MSE difference",
              "scope": "Complete internal development only; no reserved evaluation or SOTA inference."}
    for path, digest in sources.items():
        if helper.sha(ROOT / path) != digest:
            raise ValueError("Evidence changed while rendering the main summary")
    for path, digest in renderer_sources.items():
        if helper.sha(ROOT / path) != digest:
            raise ValueError("Renderer changed while composing the main summary")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in (("main_transfer.tex", text), ("secondary_metrics.tex", secondary_tex),
                          ("main_transfer.json", json.dumps(record, indent=2)+"\n")):
        path = OUT / name
        if path.exists() and path.read_text() == content:
            continue
        with tempfile.NamedTemporaryFile(mode="w", dir=OUT, prefix=".main-summary-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    return {"status": "complete_validated_development", "outputs_written": True, "rows": 3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render(args.if_ready)))
