#!/usr/bin/env python3
"""Measured temporal panoramas; no inference, re-selection or image synthesis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import render_qualitative_closest_v1 as base
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT, WIDTH = base.ROOT, base.WIDTH
OUT = ROOT / "paper/generated/qualitative_temporal_v1"
VIOLET = "#7252A1"
COLORS = [base.BLUE, base.GREEN, VIOLET]
STYLES = [":", "-", "--"]
OFFSETS = [14, 29, 44, 59]
CENTERS = [1.13, 1.80, 2.47, 3.14]


def record_frame(case, offset=None):
    index = case["window_start"] + (offset or 0)
    records = [r for r in case["images"] if r["native_index"] == index]
    if len(records) != 1 or records[0]["role"] != ("observed" if offset is None else "target"):
        raise ValueError("Missing exact, role-bound temporal frame")
    rec = records[0]
    path = ROOT / rec["path"]
    if base.sha(path) != rec["sha256"]:
        raise ValueError("Temporal source image hash changed")
    im = Image.open(path).convert("RGB")
    im.thumbnail((256, 192), Image.Resampling.LANCZOS)
    return np.asarray(im), rec


def photo(fig, case, offset, x, y, w=.58, h=.435, query=False):
    pixels, rec = record_frame(case, offset)
    ax = base.axis(fig, x, y, w, h)
    ax.imshow(pixels, interpolation="none")
    ax.axis("off")
    if query:
        hh, ww = pixels.shape[:2]
        for col, lw in (("white", 2.0), (base.INK, .75)):
            ax.add_patch(Rectangle((ww/4-.5, hh/4-.5), ww/4, hh/4,
                                   fill=False, edgecolor=col, lw=lw))
    return rec


def style_curve(ax, ticks, ylabel=None):
    ax.set_xlim(1, 59)
    ax.set_xticks(ticks)
    ax.tick_params(labelsize=8, length=2, width=.5, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(base.RULE)
    ax.grid(axis="y", color=base.RULE, linewidth=.4)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def method_key(fig, x, y):
    for dx, color, style, name in ((0, base.GREEN, "-", "Bounded (ours)"),
                                   (2.14, VIOLET, "--", "No tanh (ours, ablation)")):
        fig.add_artist(plt.Line2D([(x+dx)/WIDTH, (x+dx+.25)/WIDTH],
                                  [y/fig.get_figheight()]*2,
                                  transform=fig.transFigure, color=color, ls=style, lw=1.35))
        base.label(fig, x+dx+.31, y, name)


def gains(case, mode):
    ar = np.asarray(case["methods"]["autoregressive"]["mean"]["mse_curve"])
    other = np.asarray(case["methods"][mode]["mean"]["mse_curve"])
    if np.any(ar <= 0):
        raise ValueError("Relative gain requires a positive AR reference")
    return 100*(ar-other)/ar


def error_reduction(case, mode):
    ar = np.asarray(case["methods"]["autoregressive"]["mean"]["mse_curve"])
    other = np.asarray(case["methods"][mode]["mean"]["mse_curve"])
    return 1000*(ar-other)


def temporal_check(data):
    checked = 0
    for study in data["studies"]:
        for case in study["cases"]:
            for mode in base.method_order(study):
                model = case["methods"][mode]
                fields = {"mse_curve": (model["mean"]["mse_curve"],
                                       [s["mse_curve"] for s in model["per_seed"]])}
                for name, mean in model["mean"].get("mechanism_curves", {}).items():
                    fields[name] = (mean, [s["mechanism_curves"][name] for s in model["per_seed"]])
                for name, (mean, seeds) in fields.items():
                    arr = np.asarray(seeds)
                    if arr.shape != (3, len(study["offsets"])) or not np.isfinite(arr).all():
                        raise ValueError("Invalid temporal field: " + name)
                    np.testing.assert_allclose(arr.mean(0), mean, rtol=3e-6, atol=2e-7)
                    checked += 1
    return {"status": "passed", "three_seed_temporal_mean_checks": checked}


def render_outcomes(studies):
    fig = plt.figure(figsize=(WIDTH, 3.96), dpi=180, facecolor="white")
    base.label(fig, .02, 3.79, "Observed", weight="bold")
    base.label(fig, 2.15, 3.90, "Recorded future · evaluation only", ha="center", weight="bold")
    for center, off in zip(CENTERS, OFFSETS):
        base.label(fig, center, 3.69, f"+{off}", ha="center")
    base.label(fig, 4.66, 3.90, "MSE reduction vs AR", ha="center", weight="bold")
    base.label(fig, 4.66, 3.70, "×10³ · above 0 is better", ha="center", color=base.MUTED)
    cases = [s["cases"][1] for s in studies]
    dmax = max(float(np.abs(base.error_map(c, "autoregressive", t) -
                            base.error_map(c, s["focal_method"], t)).max())
               for s, c in zip(studies, cases) for t in OFFSETS)
    norm = TwoSlopeNorm(0, -dmax, dmax)
    all_gains = np.asarray([error_reduction(c, m) for s,c in zip(studies,cases)
                            for m in base.method_order(s)[1:]])
    lo = min(-10, 10*np.floor(all_gains.min()/10))
    hi = max(10, 10*np.ceil(all_gains.max()/10))
    records = []
    for row, (study, case) in enumerate(zip(studies, cases)):
        y = 2.70 - row*1.03
        base.label(fig, .02, y+.75, f"{'ABC'[row]}  {base.NAMES[study['id']]}", size=9, weight="bold")
        photo(fig, case, None, .02, y+.20, .62, .465)
        base.label(fig, .33, y+.10, f"f{case['window_start']}", ha="center", color=base.MUTED)
        base.label(fig, .02, y-.06, "ΔMSE", color=base.MUTED)
        maps = []
        for center, off in zip(CENTERS, OFFSETS):
            photo(fig, case, off, center-.30, y+.30, .60, .45)
            delta = base.error_map(case, "autoregressive", off)-base.error_map(case, study["focal_method"], off)
            grid_ax=base.grid(fig, delta, center-.195, y-.13, .39, norm, base.DELTA_CMAP)
            for rr,cc in np.argwhere(delta < 0):
                grid_ax.plot([cc-.26,cc+.26],[rr+.26,rr-.26],color=base.INK,
                             lw=.65,solid_capstyle="butt")
            maps.append({"offset": off, "native_index": case["window_start"]+off,
                         "mean_AR_minus_no_tanh_MSE": float(delta.mean()),
                         "patches_with_lower_no_tanh_error": int((delta>0).sum())})
        ax = base.axis(fig, 4.02, y+.025, 1.35, .675)
        ax.axhline(0, color=base.BLUE, lw=.85, ls=":", zorder=2)
        for mode, color, style in zip(base.method_order(study)[1:], COLORS[1:], STYLES[1:]):
            trace = error_reduction(case, mode)
            ax.plot(study["offsets"], trace, color=color, lw=1.35, ls=style)
            ax.scatter(OFFSETS, trace[np.asarray(OFFSETS)-1], color=color,
                       marker="o" if style=="-" else "D", s=8, zorder=4)
        ax.set_ylim(lo-1, hi+1)
        style_curve(ax, [1, 30, 59])
        ax.set_yticks(sorted(set([float(lo), 0., float(hi)])))
        if row < 2:
            base.line(fig, .02, 5.44, y-.24)
        records.append({"study": study["id"], "case": case["id"], "selection": case["selection"],
                        "offset_maps": maps, "gain_at_59_percent":
                            {m: float(gains(case,m)[-1]) for m in base.method_order(study)[1:]}})
    base.colorbar(fig, norm, base.DELTA_CMAP, .75, .28, 2.04,
                  "ΔMSE · slash = higher error", [-dmax,0,dmax])
    base.label(fig, 4.68, .435, "Native forecast offset", ha="center", color=base.MUTED)
    method_key(fig, .27, .075)
    return {"geometry": base.export(fig,"temporal_outcomes"), "cases": records,
            "signed_map_scale": [-dmax,dmax], "scaled_MSE_difference_axis_limits": [lo-1,hi+1],
            "map_quantity": "Difference of mean per-seed patch MSE; negative cells carry a centered diagonal mark; not pixel saliency",
            "curve_quantity": "1000 times AR-minus-method seed-mean MSE; no missing or clipped steps"}


def render_decoder(studies):
    fig = plt.figure(figsize=(WIDTH, 3.72), dpi=180, facecolor="white")
    base.label(fig, .02, 3.53, "Fixed q", weight="bold")
    base.label(fig, 2.01, 3.61, "Observed-source weights W[q, :]", ha="center", weight="bold")
    base.label(fig, 4.63, 3.61, "Whole-grid traces", ha="center", weight="bold")
    for center, off in zip(CENTERS, OFFSETS):
        base.label(fig, center, 3.39, f"+{off}", ha="center")
    base.label(fig, 3.79, 3.37, "gate", color=base.MUTED)
    base.label(fig, 4.63, 3.37, "correction RMS", color=base.MUTED)
    records = []
    for row, study in enumerate(studies):
        case = study["cases"][1]
        y = 2.62-row*.92
        base.label(fig, .02, y+.56, f"{'ABC'[row]}  {base.NAMES[study['id']]}", size=9, weight="bold")
        photo(fig, case, None, .02, y-.02, .62, .465, query=True)
        base.label(fig, .33, y-.12, f"f{case['window_start']}", ha="center", color=base.MUTED)
        weights = []
        for center, off in zip(CENTERS, OFFSETS):
            w = np.asarray(base.values(case, study["focal_method"], off)["M16x16"])[5].reshape(4,4)
            base.grid(fig, w, center-.245, y-.015, .49, Normalize(0,1), base.WEIGHT_CMAP, query=True)
            own = float(w.flat[5])
            base.label(fig, center, y-.13, f"{own:.2f}", ha="center", color=base.MUTED)
            weights.append({"offset":off,"query_index":5,"query_self_weight":own})
        for field, xx, width, limits, ticks in (("mean_gate",3.84,.61,(0,1),[0,1]),
                                               ("correction_rms",4.79,.61,(0,1.1),[0,1])):
            ax = base.axis(fig,xx,y+.03,width,.425)
            for mode,color,style in zip(base.method_order(study)[1:],COLORS[1:],STYLES[1:]):
                trace=case["methods"][mode]["mean"]["mechanism_curves"][field]
                ax.plot(study["offsets"],trace,color=color,lw=1.2,ls=style)
            style_curve(ax,[1,59])
            ax.set_ylim(*limits);ax.set_yticks(ticks)
        if row<2:base.line(fig,.02,5.44,y-.26)
        records.append({"study":study["id"],"case":case["id"],"maps_method":study["focal_method"],
                        "maps":weights,"traces":list(base.method_order(study)[1:])})
    base.colorbar(fig,Normalize(0,1),base.WEIGHT_CMAP,.90,.285,1.88,
                  "No tanh (ours, ablation) · weight",[0,.5,1])
    base.label(fig,.90,.085,"Number below map: W[q,q]",color=base.MUTED)
    base.label(fig,4.65,.49,"Native offset",ha="center",color=base.MUTED)
    # Two stacked keys conserve horizontal room while staying outside data ink.
    for yy,color,ls,title in ((.29,base.GREEN,"-","Bounded (ours)"),(.095,VIOLET,"--","No tanh (abl.)")):
        fig.add_artist(plt.Line2D([3.85/WIDTH,4.12/WIDTH],[yy/3.72]*2,
                                  transform=fig.transFigure,color=color,ls=ls,lw=1.35))
        base.label(fig,4.19,yy,title)
    return {"geometry":base.export(fig,"temporal_decoder"),"cases":records,
            "weight_scale":[0,1],"gate_scale":[0,1],"correction_rms_scale":[0,1.1],
            "query_index":5,"traces_are_all_patch_summaries":True}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--if-ready",action="store_true")
    args=parser.parse_args()
    if not (base.PACK/"derived.json").is_file() and args.if_ready:
        print(json.dumps({"status":"pending","reason":"Reviewed replay unavailable"}));return
    data=base.verified_pack()
    numerical={"maps":base.validate(data),"temporal":temporal_check(data)}
    studies=[next(s for s in data["studies"] if s["id"]==key)
             for key in ("pusht","bimanual_box","bimanual_rope")]
    sources={str(p.relative_to(ROOT)):base.sha(p) for p in
             (Path(__file__),Path(base.__file__),base.PACK/"derived.json",base.PACK/"result_review.json")}
    for study in studies:
        case=study["cases"][1]
        for off in [None,*OFFSETS]:
            _,record=record_frame(case,off)
            sources[record["path"]]=record["sha256"]
        # All temporal curves must fit fixed common display ranges without clipping.
        for mode in base.method_order(study)[1:]:
            curves=case["methods"][mode]["mean"]["mechanism_curves"]
            if not all(0<=v<=1 for v in curves["mean_gate"]):raise ValueError("Gate outside [0,1]")
            if not all(0<=v<=1.1 for v in curves["correction_rms"]):raise ValueError("RMS display clips data")
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=OUT/"figure_evidence.json"
    if evidence.is_file():
        old=json.loads(evidence.read_text())
        if old.get("sources")==sources and old.get("outputs") and all(
                (OUT/p).is_file() and base.sha(OUT/p)==digest for p,digest in old["outputs"].items()):
            print(json.dumps({"status":"unchanged_verified","numerical":numerical}));return
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none",
                         "image.composite_image":False,"axes.labelcolor":base.INK,"text.color":base.INK,
                         "svg.hashsalt":"shiftwm-qualitative-temporal-v1"})
    base.OUT=OUT
    figures={"temporal_outcomes":render_outcomes(studies),"temporal_decoder":render_decoder(studies)}
    own_outputs = [OUT/(name+suffix) for name in ("temporal_outcomes", "temporal_decoder")
                   for suffix in (".pdf", ".svg", ".png", "_paper_width.png", "_grayscale.png")]
    base.atomic(evidence,{"schema":"qualitative_temporal_figures_v1","status":"rendered_pending_visual_review",
                         "sources":sources,"numerical":numerical,"figures":figures,
                         "outputs":{p.name:base.sha(p) for p in own_outputs},
                         "scope":"Existing posthoc middle-ranked cases; no new inference or selection",
                         "images":"Attributed RLA-WM/IWS research excerpts, full field of view reduced to at most256x192; no predicted RGB"})
    print(json.dumps({"status":"completed","figures":2,"numerical":numerical}))


if __name__=="__main__":main()
