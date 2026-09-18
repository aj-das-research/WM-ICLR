#!/usr/bin/env python3
"""Real held-out frames and measured episode errors; requires finalized results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/generated/real_video"
DATA = ROOT / "data/real_video/droid_selected/processed"
CAMERA = "exterior_image_1_left"
MODES = ("factorized", "framewise", "constant_dynamics")
PLOTTED = (*MODES, "persistence")
COLORS = {"factorized": "#CC651E", "framewise": "#256C9B", "constant_dynamics": "#77678C", "persistence": "#65716F"}
NAMES = {"factorized": "ShiftWM (ours)", "framewise": "Framewise", "constant_dynamics": "Constant dynamics", "persistence": "Persistence"}
MARKERS = {"factorized": "o", "framewise": "s", "constant_dynamics": "^", "persistence": "x"}
HORIZONS = (1, 3, 5)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_inputs():
    report_path = ROOT / "reports/real_droid_results.json"
    release = ROOT / "artifacts/releases/real_droid_v1"
    gate_path = ROOT / "paper/tables/real_video_results.sources.json"
    for path in (report_path, release / "manifest.json", gate_path):
        if not path.is_file():
            raise RuntimeError(f"Complete campaign finalization not yet available: {path}")
    gate, report, manifest = read(gate_path), read(report_path), read(release / "manifest.json")
    if (gate.get("status") != "completed" or report.get("status") != "completed"
            or gate.get("completed_runs") != 12 or gate.get("completed_evaluations") != 48
            or report.get("completed_runs") != 12 or report.get("completed_evaluations") != 48
            or manifest.get("status") != "completed" or len(manifest.get("models", [])) != 12
            or manifest.get("offline_verification", {}).get("status") != "passed"
            or gate["report_json_sha256"] != sha(report_path)
            or gate["release_manifest_sha256"] != sha(release / "manifest.json")
            or sha(release / "reports/results.json") != sha(report_path)):
        raise ValueError("Finalized report/release identity is incomplete or inconsistent")
    population_keys = {(e["aggregate"]["camera"], e["aggregate"]["horizon"]) for e in report["populations"]}
    expected = {(camera, horizon) for camera in (CAMERA, "exterior_image_2_left") for horizon in (5, 10)}
    if len(report["populations"]) != 4 or population_keys != expected:
        raise ValueError("Four complete, separate populations are required")
    primary, checked = None, {}
    for entry in report["populations"]:
        aggregate = entry["aggregate"]
        if len(aggregate["sources"]) != 12:
            raise ValueError("Population does not contain twelve registered evaluations")
        for name, expected_hash in aggregate["sources"].items():
            path = Path(name)
            if sha(path) != expected_hash or gate["evaluation_sources"].get(name) != expected_hash:
                raise ValueError("Raw evaluation hash differs from finalized results")
            checked[name] = expected_hash
        if (aggregate["camera"], aggregate["horizon"]) == (CAMERA, 5):
            primary = aggregate
    if len(checked) != 48:
        raise ValueError("Expected exactly 48 distinct evaluations")
    records = {}
    for name in primary["sources"]:
        row = read(name)
        if row["mode"] in MODES:
            records[(row["mode"], row["seed"])] = row
    if set(records) != {(mode, seed) for mode in MODES for seed in (0, 1, 2)}:
        raise ValueError("Selected comparison lacks matched three-seed results")
    first = records[("factorized", 0)]
    keys = [(r["episode_id"], r["session_id"], r["window_starts"], r["windows"]) for r in first["episodes"]]
    for record in records.values():
        if [(r["episode_id"], r["session_id"], r["window_starts"], r["windows"]) for r in record["episodes"]] != keys:
            raise ValueError("Compared episode/window populations differ")
        if record["dataset_manifest_sha256"] != sha(DATA / "manifest.json"):
            raise ValueError("Recorded RGB comes from a different dataset")
    dataset = read(DATA / "manifest.json")
    audit = read(DATA / "data_audit.json")
    if dataset["status"] != "complete" or audit["status"] != "passed" or audit["dataset_manifest_sha256"] != sha(DATA / "manifest.json"):
        raise ValueError("Source recording audit is incomplete")
    scores = []
    for index, episode in enumerate(first["episodes"]):
        curves = {mode: {str(h): float(np.mean([records[(mode, seed)]["episodes"][index]["errors"]["model"][f"h{h}_standardized_mse"] for seed in (0, 1, 2)])) for h in HORIZONS} for mode in MODES}
        curves["persistence"] = {str(h): episode["errors"]["persistence"][f"h{h}_standardized_mse"] for h in HORIZONS}
        for record in records.values():
            if record["episodes"][index]["errors"]["persistence"] != episode["errors"]["persistence"]:
                raise ValueError("Persistence differs across matched evaluations")
        score = {key: episode[key] for key in ("episode_id", "session_id", "window_starts", "windows")}
        score.update(curves=curves, difference=curves["factorized"]["5"] - curves["framewise"]["5"])
        scores.append(score)
    scores.sort(key=lambda x: x["episode_id"])
    for mode in MODES:
        for h in HORIZONS:
            if not np.isclose(np.mean([s["curves"][mode][str(h)] for s in scores]), primary["methods"][mode][f"h{h}_standardized_mse"]["mean"], rtol=1e-10, atol=1e-12):
                raise ValueError("Episode averages do not reproduce final report")
    for h in HORIZONS:
        if not np.isclose(np.mean([s["curves"]["persistence"][str(h)] for s in scores]), primary["fixed_support_baselines"]["persistence"][f"h{h}_standardized_mse"], rtol=1e-10, atol=1e-12):
            raise ValueError("Persistence episode means do not reproduce final report")
    provenance = {"report": str(report_path.relative_to(ROOT)), "report_sha256": sha(report_path),
                  "release_manifest_sha256": sha(release / "manifest.json"), "publication_gate_sha256": sha(gate_path),
                  "dataset_manifest_sha256": sha(DATA / "manifest.json"), "data_audit_sha256": sha(DATA / "data_audit.json"),
                  "all_48_result_sources": checked}
    return scores, primary, dataset, provenance


def select_cases(scores):
    if len(scores) < 3:
        raise ValueError("At least three common test episodes are required")
    ordered = sorted(scores, key=lambda row: (row["difference"], row["episode_id"]))
    best = ordered[0]
    worst = sorted(scores, key=lambda row: (-row["difference"], row["episode_id"]))[0]
    if best["episode_id"] == worst["episode_id"]:
        worst = sorted(scores, key=lambda row: row["episode_id"])[-1]
    median = float(np.median([row["difference"] for row in scores]))
    middle = min((row for row in scores if row["episode_id"] not in {best["episode_id"], worst["episode_id"]}),
                 key=lambda row: (abs(row["difference"]-median), row["episode_id"]))
    return [("Largest reduction" if best["difference"] < 0 else "Lowest difference", best),
            ("Largest regression" if worst["difference"] > 0 else "Highest difference", worst),
            ("Median case", middle)]


def main():
    scores, aggregate, dataset, provenance = checked_inputs()
    cases = select_cases(scores)
    OUT.mkdir(parents=True, exist_ok=True)
    assets_dir = OUT / "comparison_assets"
    assets_dir.mkdir(exist_ok=True)
    lookup = {e["episode_id"]: e for e in dataset["episodes"]}
    illustrations = []
    for title, score in cases:
        episode = lookup[score["episode_id"]]
        assert episode["split"] == "test" and episode["session_id"] == score["session_id"]
        record = episode["cameras"][CAMERA]
        path = DATA / record["file"]
        if sha(path) != record["sha256"]:
            raise ValueError("Recorded image payload changed")
        start = min(score["window_starts"])
        positions = [start+2, start+7]
        with np.load(path, allow_pickle=False) as arrays:
            images = arrays["images"][positions].copy()
            native = arrays["frame_indices"][positions].tolist()
        assert images.shape == (2, 180, 320, 3) and images.dtype == np.uint8
        assets = []
        for rgb, frame in zip(images, native):
            dest = assets_dir / f"{score['episode_id']}_native{frame:05d}.png"
            Image.fromarray(rgb).save(dest)
            np.testing.assert_array_equal(np.asarray(Image.open(dest)), rgb)
            assets.append({"file":str(dest.relative_to(ROOT)), "sha256":sha(dest),
                           "pixel_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(), "native_frame_index":frame})
        illustrations.append({"title":title, "score":score, "source_rgb":str(path.relative_to(ROOT)),
                              "source_rgb_sha256":sha(path), "native_indices":native,
                              "window_start":start, "assets":assets, "images":images})

    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "pdf.fonttype":42,
                         "ps.fonttype":42, "svg.fonttype":"none", "image.composite_image":False})
    width, height = 5.5, 6.15
    fig = plt.figure(figsize=(width,height), dpi=200, facecolor="white")
    labels, image_axes, plot_axes = [], [], []
    def label(x,y,value,size=8,weight="normal",color="#243447",ha="left"):
        item=fig.text(x/width,y/height,value,fontsize=size,fontweight=weight,color=color,ha=ha,va="center")
        labels.append(item)
        return item
    label(.05,5.98,"Real recordings, measured forecast errors",10.2,"bold")
    differences=np.asarray([score["difference"] for score in scores])
    counts={"lower":int((differences<0).sum()),"tied":int((differences==0).sum()),"higher":int((differences>0).sum())}
    n=len(scores)
    ref=aggregate["methods"]["framewise"]["h5_standardized_mse"]["mean"]
    ours=aggregate["methods"]["factorized"]["h5_standardized_mse"]["mean"]
    gain=100*(ref-ours)/ref if ref>0 else None
    paired_h5=next(row for row in aggregate["paired_comparisons"] if row["reference"]=="framewise" and row["metric"]=="h5_standardized_mse")
    label(.05,5.71,f"All {n} test episodes: ours lower on {counts['lower']}, tied on {counts['tied']}, higher on {counts['higher']}.",8.2)
    gain_text="undefined" if gain is None else f"{gain:+.2f}%"
    label(.05,5.50,f"Mean h5 MSE reduction vs Framewise: {gain_text}  |  3 training seeds",8.2)
    handles=[Line2D([0],[0],color=COLORS[m],marker=MARKERS[m],linestyle="--" if m=="persistence" else "-",lw=1.5,markersize=3,label=NAMES[m]) for m in PLOTTED]
    legend=fig.legend(handles=handles,loc="center",bbox_to_anchor=(.5,5.21/height),ncol=2,frameon=False,
               fontsize=8,columnspacing=1.2,handlelength=1.6,handletextpad=.5)
    row_tops=[4.87,3.29,1.71]
    for row_index, (top, item) in enumerate(zip(row_tops,illustrations)):
        score=item["score"]
        label(.05,top,f"{chr(97+row_index)}  {item['title']}",9.2,"bold")
        label(5.43,top,f"Ours − Framewise: {score['difference']:+.3g} MSE",8,ha="right")
        bottom=top-.41-1.50*180/320
        for column,(rgb,native) in enumerate(zip(item["images"],item["native_indices"])):
            x=.05+column*1.58
            name="Last support" if column==0 else "Recorded target"
            label(x+.75,top-.25,f"{name} · frame {native}",8,ha="center")
            ax=fig.add_axes([x/width,bottom/height,1.5/width,(1.5*180/320)/height])
            ax.imshow(rgb,interpolation="none",aspect="equal");ax.set_axis_off()
            ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,facecolor="none",edgecolor="#C7CDD2",lw=.5))
            image_axes.append(ax)
        plot=fig.add_axes([3.80/width,bottom/height,1.60/width,.96/height])
        plot_axes.append(plot)
        for mode in PLOTTED:
            plot.plot(HORIZONS,[score["curves"][mode][str(h)] for h in HORIZONS],color=COLORS[mode],marker=MARKERS[mode],linestyle="--" if mode=="persistence" else "-",ms=3,lw=1.4)
        plot.set_xticks(HORIZONS);plot.set_xlim(.8,5.2)
        plot.yaxis.set_major_locator(MaxNLocator(3));plot.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:.2g}"))
        plot.set_ylim(bottom=0);plot.grid(axis="y",color="#DDE2E6",lw=.5)
        plot.spines[["top","right"]].set_visible(False)
        plot.spines[["left","bottom"]].set_color("#ADB8C1")
        plot.tick_params(labelsize=8,length=2,pad=2)
        plot.set_ylabel("Std. MSE",fontsize=8,labelpad=2)
        if row_index==2:
            plot.set_xlabel("Future action blocks",fontsize=8,labelpad=1)
        label(.05,bottom-.15,f"Episode averages: {score['windows']} matched windows × 3 seeds",8,color="#536373")
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    collisions=[]; boxes=[]
    checked_labels=list(labels)+list(legend.get_texts())
    for ax in plot_axes:
        checked_labels.extend([ax.xaxis.label,ax.yaxis.label])
        for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            for tick in axis.get_major_ticks():
                if min(limits)<=tick.get_loc()<=max(limits):
                    checked_labels.extend([tick.label1,tick.label2])
    for item in checked_labels:
        if not item.get_visible() or not item.get_text():continue
        box=item.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
            collisions.append({"kind":"text_outside_canvas","text":item.get_text()})
        for earlier,other in boxes:
            if box.overlaps(other):collisions.append({"kind":"text_overlap","text":[earlier,item.get_text()]})
        for ax in image_axes:
            if box.overlaps(ax.get_window_extent(renderer)):collisions.append({"kind":"text_image_overlap","text":item.get_text()})
        boxes.append((item.get_text(),box))
    legend_box=legend.get_window_extent(renderer)
    if legend_box.x0<0 or legend_box.y0<0 or legend_box.x1>fig.bbox.width or legend_box.y1>fig.bbox.height:
        collisions.append({"kind":"legend_outside_canvas"})
    for item in labels:
        if item.get_window_extent(renderer).overlaps(legend_box):collisions.append({"kind":"text_legend_overlap","text":item.get_text()})
    if collisions:raise ValueError(collisions)
    base=OUT/"comparison_recorded_droid"
    for suffix in ("pdf","svg","png"):
        fig.savefig(base.with_suffix("."+suffix),dpi=300)
    plt.close(fig)
    Image.open(base.with_suffix(".png")).convert("L").save(OUT/"comparison_recorded_droid_grayscale.png")
    selected=[{key:value for key,value in item.items() if key!="images"} for item in illustrations]
    ledger={"status":"complete_outcome_descriptive_figure", "population":"primary exterior camera 1; horizon5",
            "protocol":"reports/real_video_comparison_protocol.md", "protocol_sha256":sha(ROOT/"reports/real_video_comparison_protocol.md"),
            "source":str(Path(__file__).relative_to(ROOT)),"source_sha256":sha(__file__),"provenance":provenance,
            "whole_test_counts":counts,"episode_count":n,"whole_test_relative_reduction_percent":gain,
            "whole_test_paired_h5_comparison":paired_h5,
            "selected":selected,"all_episode_scores":scores,"generated_images":False,"crop":"none",
            "curve_unit":"train-standardized feature MSE; all episode windows averaged, then three seeds",
            "plotted_methods":list(PLOTTED), "additional_baseline":"Persistence repeats the last observed feature, requires no training, and does not affect registered case selection",
            "omitted_controls":"Action-free and constant-feature-velocity are reported in the full results tables; this figure focuses on framewise/context comparisons with persistence as a fixed-support reference",
            "image_scope":"first recorded window only, not a predicted RGB output or localized error explanation",
            "width_inches":width,"height_inches":height,"minimum_font_points":8,"layout_collisions":collisions,
            "exports":{str(base.with_suffix("."+suffix).relative_to(ROOT)):sha(base.with_suffix("."+suffix)) for suffix in ("pdf","svg","png")}}
    (OUT/"comparison_recorded_droid_ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
    caption=r"""\textbf{Real DROID recordings with measured feature-forecast errors.}
The primary held-out camera-1 population is summarized above three deterministically
selected episodes: CASE_LABELS in ours-minus-Framewise h5 standardized MSE
(ties by episode ID).
Each image pair shows the last observed support frame and final recorded query
target from the first evaluated window; these are actual photographs, not decoded
predictions. Each curve averages all evaluated windows of that episode, then all
three training seeds, at the three measured horizons. Panel y-axis ranges differ;
methods within each panel share the same axes. PAIRED_INTERVAL_SENTENCE The models observe three support
frames and recorded commands; persistence repeats the last support feature and
requires no training. The displayed images illustrate a recording and do
not localize the aggregate error or establish why a method wins. Selection is
outcome-conditioned and is not a frequency estimate: all-test counts and mean
reduction are reported above. Error reductions are feature-forecast differences,
not robot-control success or evidence of an identified causal mechanism."""
    caption=caption.replace("CASE_LABELS",f"the {cases[0][0].lower()}, the {cases[1][0].lower()}, and an episode nearest the median")
    ci_low,ci_high=paired_h5["ci95"]
    interval_relation="includes" if ci_low<=0<=ci_high else "excludes"
    caption=caption.replace("PAIRED_INTERVAL_SENTENCE",f"Curves show point means; the full-test paired 95\\% h5 MSE-difference interval {interval_relation} zero.")
    (OUT/"comparison_recorded_droid_caption.tex").write_text(caption+"\n")
    snippet=r"""\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{generated/real_video/comparison_recorded_droid.pdf}
\caption{FIGURE_CAPTION}
\label{fig:real-video-comparison}
\end{figure}
"""
    snippet=snippet.replace("FIGURE_CAPTION",caption)
    (OUT/"comparison_recorded_droid_figure.tex").write_text(snippet)
    print(json.dumps({"status":"rendered","figure":str(base),"counts":counts,"selected":[s['score']['episode_id'] for s in selected]}))


if __name__ == "__main__":
    main()
