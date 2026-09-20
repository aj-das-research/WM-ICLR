#!/usr/bin/env python3
"""Two compact landscape figures; source-bound presentation only, no inference."""
from pathlib import Path
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "paper/figure_sources/landscape_evidence_v1"
OUT = ROOT / "paper/generated/editorial"
W, H = 396, 144
INK, MUTED, GRID = "#243447", "#627384", "#dce3e9"
BLUE, TEAL, GREEN, AMBER = "#397BB5", "#16847B", "#166534", "#a45931"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8,
                     "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "svg.fonttype": "none", "svg.hashsalt": "landscape-evidence-v1"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(fig, x, y, label, **kwargs):
    height = fig.get_figheight()*72
    return fig.text(x/W, 1-y/height, label, fontsize=kwargs.pop("fontsize", 8),
                    va="center", color=kwargs.pop("color", INK), **kwargs)


def axis(fig, x, y, width, height, rows):
    figure_height = fig.get_figheight()*72
    ax = fig.add_axes([x/W, 1-(y+height)/figure_height, width/W, height/figure_height])
    ax.set_ylim(rows-.5, -.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8, length=2, pad=3, colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=.55, zorder=0)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    return ax


def audit(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    texts, issues = [], []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        if artist.axes is not None and not artist.axes.get_visible():
            continue
        box = artist.get_window_extent(renderer)
        if not box.width or not box.height:
            continue
        if artist.get_fontsize() < 8:
            issues.append(["font_below_8pt", artist.get_text()])
        if (box.x0 < bounds.x0-.5 or box.y0 < bounds.y0-.5 or
                box.x1 > bounds.x1+.5 or box.y1 > bounds.y1+.5):
            issues.append(["clip", artist.get_text()])
        texts.append((artist.get_text(), box))
    for i, (label, a) in enumerate(texts):
        for other, b in texts[i+1:]:
            if min(a.x1,b.x1)-max(a.x0,b.x0)>1 and min(a.y1,b.y1)-max(a.y0,b.y0)>1:
                issues.append(["text_overlap", label, other])
    return {"status": "passed" if not issues else "needs_repair", "issues": issues,
            "size_inches": list(fig.get_size_inches()), "minimum_effective_font_pt_at_5_5in": 8}


def save(fig, stem):
    result = audit(fig)
    for ext in ("pdf", "svg", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
        fig.savefig(OUT/f"{stem}.{ext}", dpi=240, metadata=metadata, facecolor="white")
    fig.savefig(OUT/f"{stem}_paper_width.png", dpi=120, facecolor="white")
    plt.close(fig)
    Image.open(OUT/f"{stem}_paper_width.png").convert("L").save(OUT/f"{stem}_grayscale.png")
    (OUT/f"{stem}_layout.json").write_text(json.dumps(result, indent=2)+"\n")
    if result["issues"]:
        raise ValueError(result)
    return result


def check_spatial(data):
    assert len(data["rows"]) == 6
    original = data["full_36_contrast_ledger"]
    effects = {e["result_id"]: e for e in original["all_16_effects"]+original["all_20_component_effects"]}
    assert len(effects) == 36
    for row in data["rows"]:
        assert set(row["endpoints"]) == {"5", "10"}
        for horizon, point in row["endpoints"].items():
            e = effects[point["source_result_id"]]
            assert e["metric"] == "native_mse" and e["method"] == "transport"
            assert e["comparator"] == row["comparator"] and e["horizon"] == int(horizon)
            np.testing.assert_allclose(point["gain_x1000"], 1000*(e["comparator_mean"]-e["method_mean"]), atol=1e-13)
            assert point["ci95_x1000"] == [-1000*e["paired_95_percent_interval"][1], -1000*e["paired_95_percent_interval"][0]]
            np.testing.assert_allclose(point["relative_gain_percent"], 100*(1-e["method_mean"]/e["comparator_mean"]), atol=1e-12)


def spatial(data):
    fig = plt.figure(figsize=(5.5, 2.0), dpi=180)
    labels = {"autoregressive": "Autoregressive", "anchored_additive": "No mix / no bound",
              "bounded_additive": "No mix (bounded)", "unbounded_transport": "No bounding",
              "context_off": "No support context", "action_free": "No actions"}
    text(fig, 4, 12, "Comparator", color=MUTED)
    for horizon, x, gain_x, letter, color, marker in (
            (5, 108, 240, "a", BLUE, "o"), (10, 258, 393, "b", TEAL, "D")):
        text(fig, x+48, 12, f"{letter}  h{horizon} endpoint", fontsize=8.8, ha="center", weight="bold")
        text(fig, gain_x, 12, "Gain", ha="right", color=MUTED)
        ax = axis(fig, x, 31, 96, 80, 6)
        ax.set_xlim(-2.8, 13.5)
        ax.set_xticks([0, 5, 10])
        ax.axvline(0, color=MUTED, linewidth=.7, linestyle=(0, (3, 2)), zorder=1)
        for i, row in enumerate(data["rows"]):
            p = row["endpoints"][str(horizon)]
            value, (lo, hi) = p["gain_x1000"], p["ci95_x1000"]
            assert -2.8 < lo <= value <= hi < 13.5
            ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt=marker, color=color,
                        markerfacecolor="white" if lo <= 0 <= hi else color,
                        markersize=3.5, markeredgewidth=.8, elinewidth=1, capsize=1.8, capthick=.8, zorder=3)
            gain = p["relative_gain_percent"]
            text(fig, gain_x, 31+(i+.5)*80/6, f"{gain:+.2f}%", ha="right",
                 color=GREEN if gain > 0 else AMBER, weight="bold" if gain > 0 else "normal")
    for i, row in enumerate(data["rows"]):
        text(fig, 4, 31+(i+.5)*80/6, labels[row["comparator"]])
    text(fig, W/2, 136, r"Native MSE reduction ($\times10^{-3}$); right favors ShiftWM (ours).", ha="center")
    return save(fig, "spatial_ablation_landscape_v1")


def check_forecast(data):
    assert data["status"] == "source_validated" and len(data["results"]) == 20
    expected = {(env, split, mode) for env in ("pusht", "reacher") for split in ("test", "extrapolation")
                for mode in ("frozen", "framewise", "single", "factorized_unpaired", "factorized")}
    assert {(r["environment"],r["split"],r["mode"]) for r in data["results"]} == expected
    for row in data["results"]:
        values = list(row["per_seed_values"].values())
        assert math.isclose(statistics.mean(values), row["value"], rel_tol=1e-12)
        uncertainty = row["uncertainty"]
        assert uncertainty["type"] == "sample_sd" and uncertainty["not_a_confidence_interval"]
        if row["mode"] == "frozen":
            assert len(values) == 1 and uncertainty["value"] is None
        else:
            assert len(values) == 3 and math.isclose(statistics.stdev(values), uncertainty["value"], rel_tol=1e-12)
    assert len(data["comparison_deltas"]) == 4
    for delta in data["comparison_deltas"]:
        assert math.isclose(delta["relative_change_percent"],100*(delta["ours_value"]/delta["reference_value"]-1), abs_tol=1e-12)


def spatial_composite(data, curves):
    """All 50 original curve points beside all 12 matched endpoint intervals."""
    values = curves["means"]
    modes = ("transport", "autoregressive", "persistence", "anchored_additive", "action_free")
    assert set(values) == set(modes)
    original = data["full_36_contrast_ledger"]["absolute_error"]
    for mode in modes:
        expected = original["autoregressive"]["native_persistence_mse"] if mode == "persistence" else original[mode]["native_mse"]
        assert values[mode] == expected and len(values[mode]) == 10
    colors = dict(zip(modes, [TEAL, INK, "#9AA5AD", BLUE, "#B88746"]))
    markers = dict(zip(modes, ["D", "o", "s", "^", "+"]))
    lines = dict(zip(modes, ["-", (0, (5, 2)), (0, (1, 2)), "--", (0, (1, 1))]))
    fig = plt.figure(figsize=(5.5, 2.25), dpi=180)
    text(fig, 30, 10, "a  Forecast error", fontsize=8.8, weight="bold")
    text(fig, 151, 10, "b  Matched endpoint gains", fontsize=8.8, weight="bold")
    ax = fig.add_axes([30/W, 1-112/162, 108/W, 83/162])
    for mode in modes[1:]+modes[:1]:
        ax.plot(range(1, 11), values[mode], color=colors[mode], ls=lines[mode], marker=markers[mode],
                markevery=[0, 4, 9] if mode == "transport" else [1, 5, 8], ms=2.5, lw=1,
                zorder=3 if mode == "transport" else 2)
    ax.set(xlim=(.7, 10.3), ylim=(0, .25), xticks=[1, 5, 10], yticks=[0, .1, .2])
    ax.tick_params(labelsize=8, length=2, pad=2, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.spines[["left", "bottom"]].set_linewidth(.65)
    ax.grid(axis="y", color=GRID, lw=.55)
    text(fig, 6, 69, "Native feature MSE", rotation=90, ha="center")
    text(fig, 84, 132, "Forecast step", ha="center")
    labels = {"autoregressive": "Autoregressive", "anchored_additive": "No mix / no bound",
              "bounded_additive": "No mix (bounded)", "unbounded_transport": "No bounding",
              "context_off": "No support context", "action_free": "No actions"}
    for i, row in enumerate(data["rows"]):
        text(fig, 151, 29+(i+.5)*83/6, labels[row["comparator"]])
    for horizon, x, color, marker in ((5, 248, BLUE, "o"), (10, 327, TEAL, "D")):
        text(fig, x+31, 23, f"h{horizon}", ha="center", color=color)
        bx = axis(fig, x, 29, 62, 83, 6)
        bx.set_xlim(-2.8, 13.5)
        bx.set_xticks([0, 5, 10])
        bx.axvline(0, color=MUTED, linewidth=.7, linestyle=(0, (3, 2)), zorder=1)
        for i, row in enumerate(data["rows"]):
            p = row["endpoints"][str(horizon)]
            value, (lo, hi) = p["gain_x1000"], p["ci95_x1000"]
            bx.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt=marker, color=color,
                        markerfacecolor="white" if lo <= 0 <= hi else color, markersize=3.5,
                        markeredgewidth=.8, elinewidth=1, capsize=1.8, capthick=.8, zorder=3)
    text(fig, 318.5, 132, r"MSE reduction $\times10^{-3}$", ha="center")
    handles = [Line2D([], [], color=colors[m], ls=lines[m], marker=markers[m], ms=2.7, lw=1) for m in modes]
    fig.legend(handles, ["ShiftWM (ours)", "Autoregressive", "Persistence", "Additive anchor", "No actions"],
               loc="lower center", bbox_to_anchor=(.5, .008), ncol=5, frameon=False, fontsize=8,
               handlelength=1.15, handletextpad=.3, columnspacing=.8, borderaxespad=0)
    return save(fig, "spatial_composite_landscape_v1")


def forecast(data):
    fig = plt.figure(figsize=(5.5, 2.0), dpi=180)
    labels = [("frozen", "Frozen LeWM"), ("framewise", "Framewise"), ("single", "Shared context"),
              ("factorized_unpaired", "Unpaired contexts"), ("factorized", "ShiftWM (ours)")]
    text(fig, 4, 12, "Historical model", color=MUTED)
    for i, (_, label) in enumerate(labels):
        text(fig, 4, 36+(i+.5)*70/5, label,
             color=AMBER if i == 4 else INK, weight="bold" if i == 4 else "normal")
    panels = [("pusht", "test", "PushT", "Held-out", (.48,1.31), [.5,1.0], ["0.5","1.0"]),
              ("pusht", "extrapolation", "PushT", "Extrap.", (.89,1.24), [.9,1.2], ["0.9","1.2"]),
              ("reacher", "test", "Reacher", "Held-out", (.0056,.073), [.006,.06], ["0.006","0.06"]),
              ("reacher", "extrapolation", "Reacher", "Extrap.", (.75,1.33), [.8,1.2], ["0.8","1.2"])]
    for j, (environment, split, name, label, limits, ticks, ticklabels) in enumerate(panels):
        x = 103+77*j
        text(fig, x+30, 12, name, fontsize=8.8, weight="bold", ha="center")
        text(fig, x+30, 25, label, ha="center", color=MUTED)
        ax = axis(fig, x, 36, 60, 70, 5)
        ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(ticklabels))
        ax.xaxis.set_minor_locator(NullLocator())
        for i, (mode, _) in enumerate(labels):
            row = next(r for r in data["results"] if (r["environment"],r["split"],r["mode"]) == (environment,split,mode))
            value, sd = row["value"], row["uncertainty"]["value"]
            assert limits[0] < value-(sd or 0) <= value+(sd or 0) < limits[1]
            ax.errorbar(value, i, xerr=sd, fmt=row["marker"], color=row["color"],
                        markersize=3.7, markeredgewidth=.7, elinewidth=.9, capsize=1.8, capthick=.8, zorder=3)
        delta = next(r for r in data["comparison_deltas"] if r["environment"] == environment and r["split"] == split)
        value = delta["relative_change_percent"]
        text(fig, x+30, 136, f"{value:+.2f}%", color=GREEN if value < 0 else AMBER, ha="center", weight="bold")
    text(fig, 4, 119, r"MSE@5 (log) $\downarrow$", color=MUTED)
    text(fig, 4, 136, r"$\Delta$ vs Framewise", color=MUTED)
    return save(fig, "forecast_landscape_v1")


def main():
    manifest = json.loads((PACK/"manifest.json").read_text())
    for name, digest in manifest["runtime_inputs_sha256"].items():
        assert sha(PACK/name) == digest, name
    a = json.loads((PACK/"spatial.json").read_text())
    b = json.loads((PACK/"forecast.json").read_text())
    c = json.loads((PACK/"curves.json").read_text())
    check_spatial(a)
    check_forecast(b)
    OUT.mkdir(parents=True, exist_ok=True)
    qa = {"spatial": spatial(a), "forecast": forecast(b), "spatial_composite": spatial_composite(a, c)}
    for stem, caption, label in (("spatial_ablation_landscape_v1", "spatial_caption.tex", "fig:spatial-versions"),
                                 ("forecast_landscape_v1", "forecast_caption.tex", "fig:results"),
                                 ("spatial_composite_landscape_v1", "spatial_composite_caption.tex", "fig:editorial-spatial")):
        figure = ("\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/"+stem+".pdf}\n"
                  "\\caption[]{\\input{figure_sources/landscape_evidence_v1/"+caption+"}}\n\\label{"+label+"}\n\\end{figure}\n")
        if stem == "spatial_composite_landscape_v1":
            figure = figure.replace("\\end{figure}", "\\label{fig:spatial-versions}\n\\end{figure}")
        (OUT/f"{stem}_figure.tex").write_text(figure)
    evidence = {"schema_version": 1, "status": "source_and_layout_checks_passed",
                "scientific_scope": "Separate current spatial development and historical simulation studies; no pooled superiority claim.",
                "new_model_inference": False, "source_pack_manifest_sha256": sha(PACK/"manifest.json"),
                "runtime_inputs_sha256": manifest["runtime_inputs_sha256"], "renderer_sha256": sha(__file__),
                "layout": qa, "spatial_displayed_intervals": 12, "spatial_preserved_full_contrasts": 36,
                "forecast_displayed_means": 20, "forecast_displayed_sample_sd": 16,
                "forecast_frozen_without_sd": 4, "signed_forecast_changes": b["comparison_deltas"],
                "spatial_composite_forecast_points": c["means"],
                "spatial_points": a["rows"], "forecast_points": b["results"],
                "outputs_sha256": {f"{stem}.{ext}": sha(OUT/f"{stem}.{ext}")
                                   for stem in ("spatial_ablation_landscape_v1", "forecast_landscape_v1", "spatial_composite_landscape_v1")
                                   for ext in ("pdf", "svg", "png")},
                "visual_review": "See source-pack review.md for exact reviewed output hashes; integrated manuscript review belongs to root."}
    (OUT/"landscape_evidence_v1.json").write_text(json.dumps(evidence, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": evidence["status"], "spatial_intervals": 12, "forecast_points": 20,
                      "dimensions_inches": {k: v["size_inches"] for k, v in qa.items()}}))


if __name__ == "__main__":
    main()
