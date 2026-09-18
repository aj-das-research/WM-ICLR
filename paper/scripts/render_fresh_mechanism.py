#!/usr/bin/env python3
"""A registered matched-case diagnostic and editable fresh-DROID figure."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/generated/real_video"
PREFIX = OUT / "fresh_mechanism"
SELECTION = OUT / "fresh_mechanism_selection.json"
PROTOCOL = OUT / "fresh_mechanism_protocol.md"
MEASUREMENTS = OUT / "fresh_mechanism_measurements.json"
ARRAYS = OUT / "fresh_mechanism_predictions.npz"
CAMERA = "exterior_image_1_left"
METHODS = ("original_ours", "calibrated_ours", "calibrated_framewise")
COLORS = {"original_ours": "#77838C", "calibrated_ours": "#D26B27", "calibrated_framewise": "#2876A2"}
NAMES = {"original_ours": "Ours, original", "calibrated_ours": "Ours, calibrated", "calibrated_framewise": "Framewise, calibrated"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(value, path):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_inputs():
    selection = json.loads(SELECTION.read_text())
    report_path = ROOT / "reports/real_droid_fresh_evaluation_results.json"
    report = json.loads(report_path.read_text())
    if selection["source_result_sha256"] != sha(report_path):
        raise ValueError("Frozen result identity changed")
    for path, expected in selection["sources"].items():
        if sha(path) != expected:
            raise ValueError("Selection source changed")
    scores = sorted(selection["all_65_episode_gains"], key=lambda row: (-row["gain_mse"], row["episode_id"]))
    if len(scores) != 65 or [row["episode_id"] for row in selection["selected_cases"]] != [scores[i]["episode_id"] for i in (0,32,64)]:
        raise ValueError("Registered case selection changed")
    primary = report["primary_comparison"]
    np.testing.assert_allclose(np.mean([row["gain_mse"] for row in scores]), -primary["mean_difference"], rtol=0, atol=1e-14)
    return selection, report


def replay():
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts/real_video"))
    sys.path.insert(0, str(ROOT / "scripts/real_video_development"))
    import train as training
    from fresh_evaluation_common import verify_freeze, FREEZE, CACHE, DECODED
    freeze = verify_freeze()
    selection, report = read_inputs()
    if MEASUREMENTS.exists() or ARRAYS.exists():
        raise ValueError("Replay already exists; preserve measurements and use render only")
    training.seed_everything(20260919)
    torch.set_num_threads(4)
    feature_manifest = json.loads((CACHE / "manifest.json").read_text())
    data_manifest = json.loads((DECODED / "manifest.json").read_text())
    feature_lookup = {row["episode_id"]: row for row in feature_manifest["episodes"]}
    image_lookup = {row["episode_id"]: row for row in data_manifest["episodes"]}
    cases, dependencies = [], {}
    assets = OUT / "fresh_mechanism_assets"
    assets.mkdir(exist_ok=True)
    for selected in selection["selected_cases"]:
        row = feature_lookup[selected["episode_id"]]
        record = row["cameras"][CAMERA]
        path = CACHE / record["file"]
        if sha(path) != record["sha256"]:
            raise ValueError("Feature payload changed")
        with np.load(path, allow_pickle=False) as arrays:
            features, actions = arrays["features"].copy(), arrays["actions"].copy()
            indices = arrays["frame_indices"].copy()
        starts = list(range(0, len(features)-8+1, 5))
        if min(starts) != selected["first_window_start"] or len(starts) != selected["windows"]:
            raise ValueError("Original selected window population changed")
        cases.append({**selected, "features": np.stack([features[start:start+8] for start in starts]),
                      "actions": np.stack([actions[start:start+7] for start in starts]),
                      "native_indices": [int(indices[starts[0]+2]), int(indices[starts[0]+7])],
                      "window_starts": starts})
        dependencies[str(path)] = sha(path)
        image_record = image_lookup[selected["episode_id"]]["cameras"][CAMERA]
        image_path = DECODED / image_record["file"]
        if sha(image_path) != image_record["sha256"]:
            raise ValueError("Recorded RGB payload changed")
        with np.load(image_path, allow_pickle=False) as arrays:
            rgb = arrays["images"][[starts[0]+2, starts[0]+7]].copy()
            np.testing.assert_array_equal(arrays["frame_indices"][[starts[0]+2, starts[0]+7]], cases[-1]["native_indices"])
        asset_rows = []
        for role, image, native in zip(("support", "target"), rgb, cases[-1]["native_indices"]):
            image_path_out = assets / f"{selected['episode_id']}_{role}.png"
            Image.fromarray(image).save(image_path_out)
            np.testing.assert_array_equal(np.asarray(Image.open(image_path_out)), image)
            asset_rows.append({"role": role, "path": str(image_path_out.relative_to(ROOT)), "sha256": sha(image_path_out),
                               "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "native_index": native})
        cases[-1]["assets"] = asset_rows
        dependencies[str(image_path)] = sha(image_path)
    curves = np.empty((len(METHODS), 3, 3, 5), dtype=np.float64)
    episode_curves = np.empty_like(curves)
    predicted = np.empty((len(METHODS), 3, 3, 5, 1536), dtype=np.float32)
    checks = []
    alphas = {}
    first_std = None
    for mode in ("factorized", "framewise"):
        for seed in (0, 1, 2):
            run = next(row for row in freeze["runs"] if row["mode"] == mode and row["seed"] == seed)
            model, _ = training.load_package(run["checkpoint"], device="cpu")
            model.eval()
            alphas[f"{mode}/s{seed}"] = run["scale"]
            dependencies[str(Path(run["checkpoint"])/"model.pt")] = sha(Path(run["checkpoint"])/"model.pt")
            dependencies[run["calibration"]] = sha(run["calibration"])
            if first_std is None:
                first_std = model.feature_std.numpy().copy()
            else:
                np.testing.assert_array_equal(first_std, model.feature_std.numpy())
            for case_index, case in enumerate(cases):
                features = torch.from_numpy(case["features"]).float()
                actions = torch.from_numpy(case["actions"]).float()
                with torch.inference_mode(), torch.autocast("cpu", enabled=False):
                    base = model.predict(features[:, :3], actions[:, :2], actions[:, 2:])
                    calibrated = features[:, 2:3] + torch.tensor(run["scale"], dtype=torch.float32) * (base-features[:, 2:3])
                values = [("calibrated_ours" if mode == "factorized" else "calibrated_framewise", "calibrated", calibrated)]
                if mode == "factorized":
                    values.append(("original_ours", "original", base))
                for name, variant, prediction in values:
                    index = METHODS.index(name)
                    errors = ((prediction-features[:, 3:])/model.feature_std).square().mean(-1).numpy()
                    curves[index,seed,case_index] = errors[0]
                    episode_curves[index,seed,case_index] = errors.astype(np.float64).mean(0)
                    predicted[index,seed,case_index] = prediction[0].numpy()
                    source = ROOT / "results/real_video_development/fresh_confirmatory_v1" / variant / f"droid_{mode}_s{seed}" / CAMERA / "h5.json"
                    if sha(source) != report["sources"][str(source)]:
                        raise ValueError("Original episode-score source changed")
                    saved = next(row for row in json.loads(source.read_text())["episodes"] if row["episode_id"] == case["episode_id"])
                    if saved["window_starts"] != case["window_starts"]:
                        raise ValueError("Replay input windows differ from saved evaluation")
                    for horizon in (1,3,5):
                        measured, reference = episode_curves[index,seed,case_index,horizon-1], saved["errors"]["model"][f"h{horizon}_standardized_mse"]
                        np.testing.assert_allclose(measured, reference, atol=2e-6, rtol=1e-5)
                        checks.append({"case": case_index, "method": name, "seed": seed, "horizon": horizon,
                                       "replayed": float(measured), "saved": reference, "absolute_difference": float(abs(measured-reference))})
                    dependencies[str(source)] = sha(source)
    targets = np.stack([case["features"][0,3:] for case in cases])
    support = np.stack([case["features"][0,:3] for case in cases])
    actions = np.stack([case["actions"][0] for case in cases])
    independent_errors = np.square((predicted-targets[None,None])/first_std).mean(-1, dtype=np.float64)
    np.testing.assert_allclose(independent_errors, curves, rtol=1e-5, atol=2e-6)
    # These are latent feature tensors, not generated prediction images.
    np.savez_compressed(ARRAYS, predictions=predicted, targets=targets, support=support,
                        actions=actions, feature_std=first_std, first_window_errors=curves,
                        episode_mean_errors=episode_curves)
    case_rows = []
    for index, case in enumerate(cases):
        row = {key: value for key, value in case.items() if key not in ("features", "actions")}
        row["first_window_curves"] = {method: curves[j,:,index].mean(0).tolist() for j,method in enumerate(METHODS)}
        row["episode_curves"] = {method: episode_curves[j,:,index].mean(0).tolist() for j,method in enumerate(METHODS)}
        row["first_window_gain_mse"] = float(curves[2,:,index,4].mean()-curves[1,:,index,4].mean())
        row["first_window_sign_matches_episode"] = bool(np.sign(row["first_window_gain_mse"]) == np.sign(row["gain_mse"]))
        case_rows.append(row)
    result = {"status": "replay_verified", "selection_sha256": sha(SELECTION), "protocol_sha256": sha(PROTOCOL),
              "evaluation_freeze_sha256": sha(FREEZE), "figure_source_sha256_at_replay": sha(__file__),
              "cases": case_rows, "alphas": alphas, "methods": list(METHODS), "replay_checks": checks,
              "maximum_saved_score_absolute_discrepancy": max(row["absolute_difference"] for row in checks),
              "independent_numpy_first_window_metric_check": "passed",
              "arrays_sha256": sha(ARRAYS), "dependencies": dependencies,
              "precision": "CPU float32 replay, no autocast/TF32; GPU-saved episode means independently checked within declared tolerance",
              "scope": "Selected post-hoc illustrations; no refit or new model, no synthesized RGB prediction"}
    for path, expected in dependencies.items():
        if sha(path) != expected:
            raise ValueError("Diagnostic dependency changed")
    verify_freeze()
    write_json(result, MEASUREMENTS)
    print(json.dumps({"status": result["status"], "maximum_replay_discrepancy": result["maximum_saved_score_absolute_discrepancy"],
                      "cases": [{k:row[k] for k in ("role", "gain_mse", "first_window_gain_mse", "first_window_sign_matches_episode")} for row in case_rows]}), flush=True)


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle, Arc
    from matplotlib.path import Path as VectorPath
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    from PIL import Image
    selection, report = read_inputs()
    measured = json.loads(MEASUREMENTS.read_text())
    if (measured["status"] != "replay_verified" or measured["selection_sha256"] != sha(SELECTION)
            or measured["protocol_sha256"] != sha(PROTOCOL) or measured["arrays_sha256"] != sha(ARRAYS)):
        raise ValueError("Registered diagnostics changed")
    for path, expected in measured["dependencies"].items():
        if sha(path) != expected:
            raise ValueError("Measured source changed")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "pdf.fonttype":42,
                         "svg.fonttype":"none", "image.composite_image":False, "axes.linewidth":.6})
    width,height=5.5,6.9
    fig=plt.figure(figsize=(width,height),dpi=200,facecolor="white")
    text_artists=[]; image_axes=[]; plot_axes=[]; routes=[]
    def text(x,y,label,size=8,color="#233645",weight="normal",ha="left"):
        artist=fig.text(x/width,y/height,label,fontsize=size,color=color,fontweight=weight,ha=ha,va="center")
        text_artists.append(artist);return artist
    def axes(x,y,w,h):return fig.add_axes([x/width,y/height,w/width,h/height])
    text(.07,6.73,"A fixed residual scale, tested on real recordings",10,"#213747","bold")
    text(.07,6.45,"a  Same forecast; contracted displacement",8.8,weight="bold")
    text(.12,6.13,r"$\widetilde z_h=z_0+\alpha\,(\widehat z_h-z_0)$",10.2)
    text(.12,5.89,"Train-fitted α · fixed at inference",8)
    text(.12,5.67,"Standard calibration control",8,color="#687681")
    # The pictured contraction uses the actual seed-zero scalar. This is a
    # schematic feature-space ray, not a projected empirical trajectory.
    mech=axes(2.80,5.63,2.58,.73);mech.set_xlim(0,2.58);mech.set_ylim(0,.73);mech.axis("off")
    anchor=np.array([.10,.32]); before=np.array([2.33,.32]); alpha=measured["alphas"]["factorized/s0"]
    after=anchor+alpha*(before-anchor)
    mech.plot([anchor[0],before[0]],[anchor[1],before[1]],color="#BCC4CA",lw=1.5,zorder=1)
    mech.plot([anchor[0],after[0]],[anchor[1],after[1]],color=COLORS["calibrated_ours"],lw=2,zorder=2)
    mech.plot(*anchor,"o",ms=4,color="#233645",zorder=5)
    mech.plot(*before,"s",ms=5,color=COLORS["original_ours"],zorder=5)
    mech.plot(*after,"D",ms=5,color=COLORS["calibrated_ours"],zorder=5)
    # A single continuous return arrow above the end of the ray denotes the
    # contraction, with ports outside the endpoint markers.
    start=(before[0],.355);end=(after[0],.369)
    waypoints=[start,(before[0],.52),(after[0],.52),end]
    arrow=FancyArrowPatch(path=VectorPath(waypoints,[VectorPath.MOVETO]+[VectorPath.LINETO]*3),arrowstyle="-|>",
                         mutation_scale=7,lw=1.1,color=COLORS["calibrated_ours"],shrinkA=0,shrinkB=0,zorder=6)
    mech.add_patch(arrow);routes.append({"meaning":"contract predicted displacement", "source_port":"base-square top",
        "target_port":"calibrated-diamond top", "source":list(start),"target":list(end),"waypoints":waypoints,"axes":"mechanism"})
    mech.text(.10,.08,r"$z_0$",ha="center",va="center",fontsize=9,color="#233645")
    mech.text(before[0],.69,r"$\widehat z_h$",ha="center",va="center",fontsize=9,color=COLORS["original_ours"])
    mech.text(after[0],.07,r"$\widetilde z_h$",ha="center",va="center",fontsize=9,color=COLORS["calibrated_ours"])
    text(4.07,5.51,f"Schematic · seed 0: α = {alpha:.3f}",7.8,color="#687681",ha="center")
    handles=[Line2D([0],[0],color=COLORS[m],lw=1.5,linestyle="--" if m=="original_ours" else "-",
                    marker={"original_ours":"^","calibrated_ours":"o","calibrated_framewise":"s"}[m],ms=3,label=NAMES[m]) for m in METHODS]
    legend=fig.legend(handles=handles,loc="center",bbox_to_anchor=(.5,5.23/height),ncol=3,frameon=False,
                      fontsize=7.8,columnspacing=.9,handlelength=1.6,handletextpad=.35)
    text(.09,5.00,"Actual observations from the first eligible window",8.1,weight="bold")
    text(4.33,5.00,"That window’s error",8.1,weight="bold",ha="center")
    text(4.43,4.83,"Future blocks 1–5",7.7,color="#687681",ha="center")
    names={"largest_improvement":"Largest episode gain", "median":"Median episode", "largest_regression":"Largest regression"}
    tops=[4.76,3.58,2.40]
    for index,(case,top) in enumerate(zip(measured["cases"],tops)):
        text(.09,top,f"{chr(98+index)}  {names[case['role']]}",8.5,weight="bold")
        text(3.03,top,f"Episode Δ: {case['gain_mse']:+.4f}",7.8,ha="right",color="#46606D")
        bottom=top-.87
        for j,asset in enumerate(case["assets"]):
            path=ROOT/asset["path"]
            if sha(path)!=asset["sha256"]:raise ValueError("Recorded image asset changed")
            ax=axes(.09+j*1.50,bottom,1.43,.8044)
            ax.imshow(np.asarray(Image.open(path)),interpolation="none",aspect="equal");ax.set_axis_off()
            image_axes.append(ax)
            title="Last support" if j==0 else "Recorded future"
            text(.805+j*1.50,bottom+.071,f"{title} · {asset['native_index']}",7.5,ha="center",color="white")
            # Solid image-footer label band never conceals any source image:
            # extend the axes region below the photograph instead of overlay.
            text_artists[-1].set_position(((.805+j*1.50)/width,(bottom-.12)/height))
            text_artists[-1].set_color("#566671")
        plot=axes(3.67,bottom+.07,1.66,.74);plot_axes.append(plot)
        for method in METHODS:
            plot.plot(np.arange(1,6),case["first_window_curves"][method],color=COLORS[method],
                      lw=1.4,linestyle="--" if method=="original_ours" else "-",
                      marker={"original_ours":"^","calibrated_ours":"o","calibrated_framewise":"s"}[method],ms=3)
        plot.set_xlim(.8,5.2);plot.set_ylim(bottom=0);plot.set_xticks([1,3,5])
        plot.tick_params(labelsize=7.5,length=2,pad=1)
        plot.yaxis.set_major_locator(MaxNLocator(3));plot.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:.2f}"))
        plot.grid(axis="y",lw=.45,color="#DFE5E8");plot.spines[["top","right"]].set_visible(False)
        plot.spines[["left","bottom"]].set_color("#A4B1BA")
        plot.set_ylabel("Std. MSE",fontsize=7.6,labelpad=2)
        mismatch="*" if not case["first_window_sign_matches_episode"] else ""
        text(4.47,bottom-.13,f"h5 Δ: {case['first_window_gain_mse']:+.4f}{mismatch}",7.7,ha="center",
             color="#257456" if case["first_window_gain_mse"]>0 else "#A74945")
    # Complete episode inventory and separate population uncertainty.
    text(.09,1.15,"e  All 65 episodes",8.8,weight="bold")
    gains=np.asarray([r["gain_mse"] for r in selection["all_65_episode_gains"]])
    positive,negative=int((gains>0).sum()),int((gains<0).sum())
    text(3.14,1.15,f"{positive} gains · {negative} regressions",8.1,color="#46606D",ha="right")
    distribution=axes(.68,.36,2.42,.55);plot_axes.append(distribution)
    x=np.arange(1,66)
    distribution.axhline(0,color="#6B7A84",lw=.8)
    distribution.vlines(x,0,gains,color=np.where(gains>0,"#31846A","#B65D58"),lw=.7,alpha=.7)
    distribution.scatter(x,gains,c=np.where(gains>0,"#31846A","#B65D58"),s=7,zorder=4)
    for selected in selection["selected_cases"]:
        distribution.scatter(selected["rank"],selected["gain_mse"],s=22,facecolors="white",edgecolors="#233645",linewidths=.8,zorder=5)
    distribution.set_xlim(0,66);distribution.set_xticks([1,33,65]);distribution.set_xlabel("Episode rank (fixed selection)",fontsize=7.6,labelpad=2)
    distribution.set_ylabel("MSE gain",fontsize=7.6,labelpad=2);distribution.tick_params(labelsize=7.3,length=2,pad=2)
    distribution.yaxis.set_major_locator(MaxNLocator(3));distribution.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:+.3f}"))
    distribution.spines[["top","right"]].set_visible(False)
    for spine in ("left","bottom"):distribution.spines[spine].set_color("#A4B1BA")
    primary=report["primary_comparison"];mean=-primary["mean_difference"];ci=[-primary["ci95"][1],-primary["ci95"][0]]
    text(4.38,1.15,"Population mean + 95% CI",8.1,weight="bold",ha="center")
    interval=axes(3.60,.43,1.68,.40);plot_axes.append(interval)
    interval.axvline(0,color="#B65D58",lw=.8,linestyle=":")
    interval.errorbar([mean],[0],xerr=np.asarray([[mean-ci[0]],[ci[1]-mean]]),fmt="o",color="#257456",lw=1.8,capsize=4,ms=4)
    interval.set_xlim(-.0003,.0025);interval.set_ylim(-.8,.8);interval.set_yticks([]);interval.set_xticks([0,.001,.002])
    interval.set_xticklabels(["0",".001",".002"]);interval.tick_params(axis="x",labelsize=7.5,length=2,pad=2)
    interval.spines[["top","left","right"]].set_visible(False);interval.spines["bottom"].set_color("#A4B1BA")
    text(4.43,.19,f"{primary['relative_reduction_percent']:.3f}% lower h5 MSE",8,ha="center",color="#257456",weight="bold")
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    # Measure text and image bounds; connector continuity is reviewed in crops.
    tracked=list(text_artists)+list(legend.get_texts())
    for ax in plot_axes:
        tracked += [ax.xaxis.label,ax.yaxis.label]
        tracked += [item for item in ax.get_xticklabels()+ax.get_yticklabels() if item.get_visible()]
    boxes=[];issues=[]
    for item in tracked:
        if not item.get_visible() or not item.get_text():continue
        box=item.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
            issues.append({"kind":"outside_canvas","text":item.get_text()})
        for earlier,other in boxes:
            if box.overlaps(other):issues.append({"kind":"text_overlap","text":[earlier,item.get_text()]})
        for ax in image_axes:
            if box.overlaps(ax.get_window_extent(renderer)):issues.append({"kind":"text_image_overlap","text":item.get_text()})
        boxes.append((item.get_text(),box))
    for extension in ("pdf","svg","png"):
        fig.savefig(PREFIX.with_suffix("."+extension),dpi=300)
    plt.close(fig)
    Image.open(PREFIX.with_suffix(".png")).convert("L").save(OUT/"fresh_mechanism_grayscale.png")
    caption=r"""\textbf{Residual calibration on actual DROID recordings.}
(a) One train-fitted scalar contracts each forecast displacement about the last observation; the schematic uses seed 0's $\alpha=0.908$. The base rollout is unchanged.
(b--d) Episodes are selected by their three-seed, episode-average h5 gains: largest improvement, median and largest regression among all 65 fresh recordings. Photographs are recorded support/target frames from each first eligible window (native indices 10 and 35). Curves show that window's five measured feature errors, averaged over three seeds; vertical ranges differ between cases. Positive $\Delta$ favors calibrated ShiftWM over equally calibrated Framewise. *The median episode improves overall but regresses in the displayed window. 
(e) All episode gains and regressions are shown. The separate population interval is the original paired session/seed bootstrap: MSE gain 0.001103, 95\% CI [0.000290,0.002145], or 0.742\% lower mean h5 error. This measures observational forecasting, not physical control."""
    (OUT/"fresh_mechanism_caption.tex").write_text(caption+"\n")
    snippet="\\begin{figure}[p]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/real_video/fresh_mechanism.pdf}\n\\caption{"+caption+"}\n\\label{fig:fresh-mechanism}\n\\end{figure}\n"
    (OUT/"fresh_mechanism_figure.tex").write_text(snippet)
    ledger={"status":"rendered_pending_visual_review", "source_sha256":sha(__file__), "protocol_sha256":sha(PROTOCOL),
            "replay_source_archive_sha256":sha(OUT/"fresh_mechanism_replay_source.txt"),
            "selection_sha256":sha(SELECTION), "measurements_sha256":sha(MEASUREMENTS), "arrays_sha256":sha(ARRAYS),
            "width_inches":width,"height_inches":height,"counts":{"positive":positive,"negative":negative,"ties":int((gains==0).sum())},
            "all_65_episode_gains":gains.tolist(),"population_mean_gain":mean,"population_ci95_gain":ci,
            "primary_relative_reduction_percent":primary["relative_reduction_percent"],"cases":measured["cases"],
            "routed_connectors":routes,"layout_issues":issues,"synthetic_prediction_images":False,"crop":"none",
            "exports":{str(PREFIX.with_suffix('.'+extension).relative_to(ROOT)):sha(PREFIX.with_suffix('.'+extension)) for extension in ('pdf','svg','png')}}
    write_json(ledger,OUT/"fresh_mechanism_ledger.json")
    print(json.dumps({"status":"rendered","issues":issues,"counts":ledger["counts"]}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("operation",choices=("replay","render"));a=p.parse_args()
    {"replay":replay,"render":render}[a.operation]()
