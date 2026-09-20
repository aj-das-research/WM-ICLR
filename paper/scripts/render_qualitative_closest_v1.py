#!/usr/bin/env python3
"""Render matched qualitative evidence from the reviewed, derived-only replay.

No training, inference, selection or RGB synthesis. Geometry is defined here;
the JSON and local attributed source frames supply all scientific content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "paper/figure_sources/qualitative_closest_v1"
OUT = ROOT / "paper/generated/qualitative_closest_v1"
WIDTH = 5.5
INK, MUTED, RULE = "#243447", "#59697B", "#DEE5EB"
GREEN, AMBER, BLUE = "#166534", "#A85420", "#245D99"
ERROR_CMAP = LinearSegmentedColormap.from_list("error", ["#F4F7FC", "#749ABE", "#173D64"])
DELTA_CMAP = LinearSegmentedColormap.from_list("paired", ["#B5632D", "#FFFFFF", "#217D69"])
WEIGHT_CMAP = LinearSegmentedColormap.from_list("weight", ["#F3F8F8", "#5CA4A6", "#134D58"])
NAMES = {"droid": "DROID", "pusht": "PushT", "bimanual_box": "Box", "bimanual_rope": "Rope"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if not path.exists() or path.read_text() != text:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def axis(fig, x, y, w, h):
    return fig.add_axes([x/WIDTH, y/fig.get_figheight(), w/WIDTH, h/fig.get_figheight()])


def label(fig, x, y, s, size=8, color=INK, weight="normal", ha="left", va="center"):
    return fig.text(x/WIDTH, y/fig.get_figheight(), s, fontsize=size, color=color,
                    weight=weight, ha=ha, va=va, linespacing=1.15)


def line(fig, x1, x2, y):
    fig.add_artist(plt.Line2D([x1/WIDTH, x2/WIDTH], [y/fig.get_figheight()]*2,
                            transform=fig.transFigure, color=RULE, lw=.55))


def bounds_review(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    errors = []
    labels = []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        box = obj.get_window_extent(renderer)
        if obj.get_fontsize() < 8 - 1e-8:
            errors.append("Font below 8pt: " + obj.get_text())
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
            errors.append("Clipped text: " + obj.get_text())
        if obj in fig.texts:
            labels.append((obj, box))
    for i, (a, ba) in enumerate(labels):
        for b, bb in labels[i+1:]:
            if ba.overlaps(bb):
                errors.append("Overlapping labels: " + a.get_text() + " / " + b.get_text())
        for ax in fig.axes:
            if ax.get_visible() and ba.overlaps(ax.get_window_extent(renderer)):
                errors.append("Figure label overlaps a data/image panel: " + a.get_text())
    if errors:
        raise ValueError("; ".join(errors))
    return {"status": "passed", "min_font_pt": 8, "width_inches": WIDTH,
            "height_inches": fig.get_figheight(), "clipping_or_label_overlap": [], "label_data_panel_overlap": []}


def export(fig, name):
    review = bounds_review(fig)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white",
                    metadata={"Creator": "ShiftWM source-bound qualitative renderer", "CreationDate": None, "ModDate": None}
                    if ext == "pdf" else None)
    fig.savefig(OUT / f"{name}_paper_width.png", dpi=144, facecolor="white")
    with Image.open(OUT / f"{name}_paper_width.png") as im:
        im.convert("L").save(OUT / f"{name}_grayscale.png")
    plt.close(fig)
    return review


def method_order(study):
    methods = study["methods"]
    bounded = "transport" if "transport" in methods else "bounded_spatial_mix"
    unbounded = "unbounded_transport" if "unbounded_transport" in methods else "unbounded_spatial_mix"
    if unbounded not in methods:
        unbounded = next(m for m in methods if "unbounded" in m or "no_tanh" in m)
    return ["autoregressive", bounded, unbounded]


def values(case, mode, endpoint):
    return case["methods"][mode]["mean"]["by_offset"][str(endpoint)]


def error_map(case, mode, endpoint):
    return np.asarray(values(case, mode, endpoint)["patch_mse4x4"], dtype=float)


def frame(case, target=False):
    candidates = sorted(case["images"], key=lambda p: p["native_index"])
    if target:
        record = candidates[-1]
    else:
        observed = [x for x in candidates if x["role"] in ("observed", "last_observed", "observation")]
        if not observed:
            raise ValueError("No explicitly identified observed frame")
        record = observed[-1]
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    if sha(path) != record["sha256"]:
        raise ValueError("Recorded frame changed: " + str(path))
    return np.asarray(Image.open(path).convert("RGB")), record


def photograph(fig, case, target, x, y, w, h, query=False):
    pixels, record = frame(case, target)
    if not case["id"].startswith("droid_"):
        # Attributed research excerpt only: identical aspect-preserving reduction
        # for observed/target frames, with no crop or alteration of raw sources.
        source_image=Image.fromarray(pixels)
        source_image.thumbnail((256,192),Image.Resampling.LANCZOS)
        pixels=np.asarray(source_image)
    ax = axis(fig, x, y, w, h)
    ax.imshow(pixels, interpolation="none")
    ax.axis("off")
    if query:
        height, width = pixels.shape[:2]
        ax.add_patch(Rectangle((width/4-.5, height/4-.5), width/4, height/4,
                               fill=False, edgecolor="white", lw=2.1))
        ax.add_patch(Rectangle((width/4-.5, height/4-.5), width/4, height/4,
                               fill=False, edgecolor=GREEN, lw=.9))
        ax.text(width*.375, height*.375, "q", fontsize=8, color="white", ha="center", va="center",
                bbox={"boxstyle": "square,pad=.08", "facecolor": GREEN, "edgecolor": "none"})
    return record


def grid(fig, array, x, y, width, norm, cmap, query=False):
    ax = axis(fig, x, y, width, width)
    # Keep every measured cell editable in SVG/PDF; only photographs are raster.
    edges=np.arange(5)-.5
    ax.pcolormesh(edges,edges,array,cmap=cmap,norm=norm,shading="flat",rasterized=False,antialiased=False)
    ax.set_aspect("equal")
    ax.set_xlim(-.5,3.5);ax.set_ylim(3.5,-.5)
    ax.set_xticks(np.arange(.5, 3.5), minor=True)
    ax.set_yticks(np.arange(.5, 3.5), minor=True)
    ax.grid(which="minor", color="white", linewidth=.35)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_linewidth(.45)
        spine.set_color(RULE)
    if query:
        ax.add_patch(Rectangle((.5, .5), 1, 1, fill=False, edgecolor=INK, lw=1))
    return ax


def colorbar(fig, norm, cmap, x, y, w, title, ticks):
    ax = axis(fig, x, y, w, .06)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal")
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        # Adjacent PDF vector segments must overlap to avoid viewer hairlines.
        bar.solids.set_edgecolor("face")
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{v:.2g}" for v in ticks])
    bar.ax.tick_params(labelsize=8, length=2, width=.5, pad=1)
    bar.outline.set_linewidth(.45)
    label(fig, x+w/2, y+.15, title, ha="center")
    return bar


def study_scales(study):
    endpoint = study["offsets"][-1]
    modes = method_order(study)
    maps = [error_map(c, m, endpoint) for c in study["cases"] for m in modes]
    vmax = max(float(x.max()) for x in maps)
    dmax = max(float(np.abs(error_map(c, modes[0], endpoint) -
                            error_map(c, study["focal_method"], endpoint)).max()) for c in study["cases"])
    return Normalize(0, vmax), TwoSlopeNorm(0, -dmax, dmax)


def render_droid(study):
    fig = plt.figure(figsize=(WIDTH, 3.05), dpi=180, facecolor="white")
    modes = method_order(study)
    endpoint = study["offsets"][-1]
    norm, delta_norm = study_scales(study)
    columns = [(1.16, "Observed"), (2.04, "Target"), (2.87, "AR"),
               (3.53, "ShiftWM\n(ours)"), (4.19, "No tanh\n(ours, abl.)"), (5.03, "AR − ours")]
    for x, title in columns:
        label(fig, x, 2.83, title, ha="center", color=GREEN if title.startswith("Shift") else INK)
    label(fig, .02, 2.83, "Case / gain", size=8)
    observations = []
    for i, case in enumerate(study["cases"]):
        y = 2.01-i*.76
        role = ("Largest", "Median", "Smallest")[i]
        selection = case["selection"]
        ep, win = selection["gain_percent"], selection["first_window_gain_percent"]
        label(fig, .02, y+.51, role, weight="bold")
        label(fig, .02, y+.32, f"Ep {ep:+.1f}%", color=GREEN if ep>0 else AMBER)
        label(fig, .02, y+.14, f"Win {win:+.1f}%", color=GREEN if win>0 else AMBER)
        for target, x in ((False, .76), (True, 1.64)):
            record = photograph(fig, case, target, x, y+.1, .80, .48)
            label(fig, x+.40, y+.01, f"f{record['native_index']}", ha="center", color=MUTED)
        maps = []
        best_mse=min(float(error_map(case,m,endpoint).mean()) for m in modes)
        for j, mode in enumerate(modes):
            x = 2.62 + j*.66
            a = error_map(case, mode, endpoint)
            maps.append(a)
            grid(fig, a, x, y+.09, .50, norm, ERROR_CMAP)
            label(fig, x+.25, y-.015, f"{a.mean():.4f}", ha="center",
                  color=BLUE if mode == study["focal_method"] else INK,
                  weight="bold" if float(a.mean())==best_mse else "normal")
        difference = maps[0] - error_map(case, study["focal_method"], endpoint)
        difference_axes=grid(fig, difference, 4.755, y+.07, .55, delta_norm, DELTA_CMAP)
        for rr in range(4):
            for cc in range(4):
                value=difference[rr,cc]
                shade=DELTA_CMAP(delta_norm(value))
                luminance=.2126*shade[0]+.7152*shade[1]+.0722*shade[2]
                difference_axes.text(cc,rr,"+" if value>0 else "−" if value<0 else "0",
                                     ha="center",va="center",fontsize=8,
                                     color="white" if luminance<.53 else INK)
        label(fig, 5.03, y-.045, f"{1000*difference.mean():+.1f}", ha="center",
              color=GREEN if difference.mean()>0 else AMBER)
        if i<2:
            line(fig, .02, 5.46, y-.12)
        observations.append({"case": case["id"], "map_means": {m: float(a.mean()) for m,a in zip(modes,maps)},
                             "mean_AR_minus_focal_mse": float(difference.mean()),
                             "patches_lower_error": int((difference>0).sum())})
    colorbar(fig, norm, ERROR_CMAP, .65, .19, 1.7, "Feature MSE · lower is better", [0, norm.vmax])
    colorbar(fig, delta_norm, DELTA_CMAP, 3.09, .19, 1.77,
             "AR − ours · green = lower error", [-delta_norm.vmax, 0, delta_norm.vmax])
    review = export(fig, "droid_matched")
    return {"geometry": review, "case_measurements": observations,
            "absolute_error_scale": [0, norm.vmax], "signed_error_scale": [-delta_norm.vmax, delta_norm.vmax]}


def render_iws_band(studies, index):
    fig = plt.figure(figsize=(WIDTH, 2.08), dpi=180, facecolor="white")
    records = []
    for col, study in enumerate(studies):
        case = study["cases"][index]
        x = .03 + col*1.84
        endpoint = study["offsets"][-1]
        modes = method_order(study)
        norm, _ = study_scales(study)
        label(fig, x, 1.96, NAMES[study["id"]], size=9, weight="bold")
        label(fig, x+1.68, 1.96, f"traj. {case['episode_id'][-2:]}", ha="right", color=MUTED)
        for target, dx in ((False, 0), (True, .87)):
            rec = photograph(fig, case, target, x+dx, 1.30, .79, .46)
            label(fig, x+dx+.395, 1.22, ("Target " if target else "Observed ")+f"f{rec['native_index']}", ha="center")
        ep, win = case["selection"]["gain_percent"], case["selection"]["first_window_gain_percent"]
        label(fig, x+.02, 1.05, f"Ep {ep:+.1f}%", color=GREEN if ep>0 else AMBER)
        label(fig, x+1.64, 1.05, f"Win {win:+.1f}%", ha="right", color=GREEN if win>0 else AMBER)
        measures = {}
        best_mse=min(float(error_map(case,m,endpoint).mean()) for m in modes)
        for j, (mode, title) in enumerate(zip(modes, ("AR", "Bounded", "No tanh"))):
            xx = x+j*.575
            label(fig, xx+.235, .87, title, ha="center", color=GREEN if mode==study["focal_method"] else INK)
            a = error_map(case, mode, endpoint)
            grid(fig, a, xx+.01, .32, .45, norm, ERROR_CMAP)
            label(fig, xx+.235, .21, f"{a.mean():.4f}", ha="center", color=BLUE if mode==study["focal_method"] else INK,
                  weight="bold" if float(a.mean())==best_mse else "normal")
            measures[mode] = float(a.mean())
        label(fig, x+.84, .065, f"MSE scale 0–{norm.vmax:.2f} · 3 seeds", ha="center", color=MUTED)
        records.append({"study": study["id"], "case": case["id"], "map_means": measures,
                        "error_scale": [0,norm.vmax]})
        if col<2:
            fig.add_artist(plt.Line2D([(x+1.74)/WIDTH]*2, [.04/2.08, 2.02/2.08],
                                     transform=fig.transFigure, color=RULE, lw=.6))
    name = ("iws_largest", "iws_median", "iws_smallest")[index]
    return {"geometry": export(fig, name), "case_measurements": records}


def render_mechanism(studies):
    """One fixed query and the declared focal variant on each median case."""
    fig = plt.figure(figsize=(WIDTH, 3.65), dpi=180, facecolor="white")
    title_y = 3.46
    for x, title in ((.55,"Observed / q"),(1.80,"Raw source\nT"),(2.61,"Gated source\nW"),
                     (3.45,"Correction\nRMS"),(4.60,"Own-source\nweight")):
        label(fig,x,title_y,title,ha="center",size=8)
    records=[]
    for row, study in enumerate(studies):
        case=study["cases"][1]
        endpoint=study["offsets"][-1]
        mode=study["focal_method"]
        a=values(case,mode,endpoint)
        y=2.64-row*.69
        photograph(fig,case,False,.03,y+.08,1.05,.48,query=True)
        label(fig,.55,y-.02,NAMES[study["id"]]+(" · bounded" if study["id"]=="droid" else " · no tanh"),ha="center")
        tm=np.asarray(a["T16x16"])[5].reshape(4,4)
        wm=np.asarray(a["M16x16"])[5].reshape(4,4)
        delta=np.asarray(a["correction_rms4x4"])
        grid(fig,tm,1.565,y+.08,.47,Normalize(0,1),WEIGHT_CMAP,query=True)
        grid(fig,wm,2.375,y+.08,.47,Normalize(0,1),WEIGHT_CMAP,query=True)
        # Normalize within a study only; its training-standardization differs.
        correction_max=max(float(np.asarray(values(c,m,endpoint)["correction_rms4x4"]).max())
                           for c in study["cases"] for m in method_order(study)[1:])
        grid(fig,delta,3.215,y+.08,.47,Normalize(0,correction_max),ERROR_CMAP)
        label(fig,3.45,y-.025,f"0–{correction_max:.2f}",ha="center",color=MUTED)
        ax=axis(fig,4.02,y+.15,1.16,.35)
        raw,eff=float(tm.reshape(-1)[5]),float(wm.reshape(-1)[5])
        ax.plot([raw,eff],[.5,.5],color="#8195A0",lw=2)
        ax.scatter([raw],[.5],s=22,color=BLUE,marker="o",zorder=3)
        ax.scatter([eff],[.5],s=24,color=GREEN,marker="D",zorder=3)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_yticks([]);ax.set_xticks([0,1]);ax.tick_params(labelsize=8,length=2,pad=1)
        for spine in ("top","right","left"):ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        label(fig,4.58,y-.025,f"{raw:.2f} → {eff:.2f}",ha="center")
        label(fig,1.80,y-.025,"T[q, : ]",ha="center",color=MUTED)
        label(fig,2.61,y-.025,"W[q, : ]",ha="center",color=MUTED)
        if row<3:line(fig,.03,5.45,y-.115)
        records.append({"study":study["id"],"case":case["id"],"method":mode,"query_index":5,
                        "query_raw_self_weight":raw,"query_effective_self_weight":eff,
                        "correction_mean_patch_rms":float(delta.mean()),"correction_display_scale":[0,correction_max]})
    colorbar(fig,Normalize(0,1),WEIGHT_CMAP,.48,.19,1.65,"Source weight · fixed q",[0,1])
    correction_bar=colorbar(fig,Normalize(0,1),ERROR_CMAP,3.12,.19,1.65,"RMS · limits shown per task",[0,1])
    correction_bar.set_ticklabels(["0","task max"])
    return {"geometry":export(fig,"decoder_internals"),"case_measurements":records}


def validate(data):
    if data["schema"]!="qualitative_closest_derived_v1" or data["status"]!="passed":
        raise ValueError("No completed, compatible replay pack")
    checks=0
    if len(data["studies"])!=4:
        raise ValueError("Incomplete study set")
    for study in data["studies"]:
        if study["seeds"]!=[0,1,2] or len(study["cases"])!=3:
            raise ValueError("Missing seeds/cases")
        for case in study["cases"]:
            for mode in method_order(study):
                model=case["methods"][mode]
                if len(model["per_seed"])!=3:
                    raise ValueError("Incomplete model seeds")
                for offset in study["display_offsets"]:
                    mean=values(case,mode,offset)
                    stack=np.asarray([r["by_offset"][str(offset)]["patch_mse4x4"] for r in model["per_seed"]])
                    if stack.shape!=(3,4,4) or not np.isfinite(stack).all() or (stack<0).any():
                        raise ValueError("Invalid feature-error map")
                    np.testing.assert_allclose(stack.mean(0),mean["patch_mse4x4"],rtol=3e-6,atol=2e-7)
                    np.testing.assert_allclose(stack.mean(),model["mean"]["mse_curve"][study["offsets"].index(offset)],rtol=3e-6,atol=2e-7)
                    if mode!="autoregressive":
                        for field in ("T16x16","M16x16"):
                            per=np.asarray([r["by_offset"][str(offset)][field] for r in model["per_seed"]])
                            np.testing.assert_allclose(per.mean(0),mean[field],rtol=3e-6,atol=2e-7)
                            np.testing.assert_allclose(per.sum(-1),1,rtol=3e-6,atol=2e-7)
                    checks+=1
    return {"map_mean_and_seed_checks":checks,"status":"passed"}


def verified_pack():
    manifest=json.loads((PACK/"manifest.json").read_text())
    if manifest["status"]!="complete_reviewed" or manifest["raw_files_included"] is not False:
        raise ValueError("Expected independently reviewed, derived-only source pack")
    for name,digest in manifest["files"].items():
        if Path(name).name!=name or sha(PACK/name)!=digest:
            raise ValueError("Qualitative pack changed: "+name)
    data=json.loads((PACK/"derived.json").read_text())
    review=json.loads((PACK/"result_review.json").read_text())
    if (review["status"]!="passed" or review["derived_sha256"]!=sha(PACK/"derived.json")
            or review["registration_sha256"]!=data["registration_sha256"]
            or manifest["registration_sha256"]!=data["registration_sha256"]):
        raise ValueError("Replay review/registration mismatch")
    return data


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--if-ready",action="store_true")
    args=parser.parse_args()
    source=PACK/"derived.json"
    if not source.is_file() and args.if_ready:
        print(json.dumps({"status":"pending","reason":"Reviewed qualitative replay is incomplete"}));return
    data=verified_pack()
    numerical=validate(data)
    OUT.mkdir(parents=True,exist_ok=True)
    evidence_path=OUT/"figure_evidence.json"
    if evidence_path.is_file():
        previous=json.loads(evidence_path.read_text())
        if (previous.get("source_sha256")==sha(source) and previous.get("renderer_sha256")==sha(Path(__file__))
                and previous.get("outputs") and all((OUT/name).is_file() and sha(OUT/name)==digest
                for name,digest in previous["outputs"].items())):
            for study in data["studies"]:
                for case in study["cases"]:
                    frame(case,False);frame(case,True)
            print(json.dumps({"status":"unchanged_verified","figures":len(previous["figures"]),
                              "numerical_checks":numerical}));return
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none",
                         "image.composite_image":False,"axes.labelcolor":INK,"text.color":INK,"axes.unicode_minus":True,
                         "svg.hashsalt":"shiftwm-qualitative-closest-v1"})
    studies={s["id"]:s for s in data["studies"]}
    iws=[studies[k] for k in ("pusht","bimanual_box","bimanual_rope")]
    figures={"droid_matched":render_droid(studies["droid"])}
    for i in range(3):figures[("iws_largest","iws_median","iws_smallest")[i]]=render_iws_band(iws,i)
    figures["decoder_internals"]=render_mechanism([studies["droid"],*iws])
    record={"schema":"qualitative_closest_figure_evidence_v1","status":"rendered_pending_visual_review",
            "source_sha256":sha(source),"renderer_sha256":sha(Path(__file__)),"numerical_checks":numerical,
            "figures":figures,"outputs":{p.name:sha(p) for p in OUT.iterdir() if p.suffix in (".pdf",".svg",".png")},
            "image_excerpt_policy":{"DROID":"original320x180 CC BY4.0 frames",
              "IWS":"At most256x192, intact field of view, PIL Lanczos, same reduction for observed and target; originals remain local"},
            "scope":"Matched feature diagnostics; source weights are not semantic attention or causal attribution."}
    atomic(OUT/"figure_evidence.json",record)
    print(json.dumps({"status":"completed","figures":len(figures),"numerical_checks":numerical}))


if __name__=="__main__":main()
