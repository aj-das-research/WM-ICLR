#!/usr/bin/env python3
"""Display-only spatial versions comparison; fixed measured evidence, no inference."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/generated/real_video"
PREFIX = "spatial_versions"
REPORT = ROOT / "reports/real_video_spatial/finalization.json"
REPORT_SHA = "a9d33682dc0922e96682083ad379a447f14cc5213eae87b781c41c0c8afd7511"
CHECKER = ROOT / "scripts/real_video_spatial_reporting/finalize_paper.py"
CHECKER_SHA = "6a61b26aa051305aed14b5cff509186c43ccc1f135c62ab449dd8d5cac350467"
COMPONENT_REG = ROOT / "configs/real_video_spatial_components/v1/registration.json"
COMPONENT_REG_SHA = "4a626e92e6776da80f36b5d7ed11b291bb35e58c50e357dcb9779496385d0ab2"
SKILL = Path(os.environ.get("PAPER_FIGURE_SKILL_DIR", str(Path.home()/".codex/skills/paper-figure-creation"))).expanduser()
GREEN = "#166534"; INK = "#253746"; GRAY = "#657581"; LIGHT = "#E7ECEF"; PALE = "#EFF7F1"
DISPLAY = {"autoregressive": "Autoregressive", "persistence": "Persistence", "anchored_additive": "Ours-2",
           "bounded_additive": "Ours-3", "unbounded_transport": "Ours-4", "transport": "Ours-5",
           "context_off": "Ours-5 − context", "action_free": "Ours-5 − actions"}
ORDER = ("autoregressive", "persistence", "anchored_additive", "bounded_additive", "unbounded_transport", "transport", "context_off", "action_free")
COMPARATORS = ("autoregressive", "anchored_additive", "context_off", "action_free")
PENDING = ("bounded_additive", "unbounded_transport")
VERSIONS = [
    {"id": "Ours-1", "mode": "factorized", "mechanism": "Original context-based ShiftWM", "status": "Completed; different protocol", "spatial_axis": False},
    {"id": "Ours-2", "mode": "anchored_additive", "mechanism": "Observed anchor + additive innovation", "status": "Completed", "spatial_axis": True},
    {"id": "Ours-3", "mode": "bounded_additive", "mechanism": "Observed anchor + bounded innovation", "status": "Training (snapshot)", "spatial_axis": True},
    {"id": "Ours-4", "mode": "unbounded_transport", "mechanism": "Gated mixing + unbounded innovation", "status": "Training (snapshot)", "spatial_axis": True},
    {"id": "Ours-5", "mode": "transport", "mechanism": "Gated mixing + bounded innovation", "status": "Completed", "spatial_axis": True},
]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def write(name, value):
    (OUT / (PREFIX + name)).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_sources():
    for p, expected in ((REPORT, REPORT_SHA), (CHECKER, CHECKER_SHA), (COMPONENT_REG, COMPONENT_REG_SHA)):
        if sha(p) != expected: raise ValueError("Pinned comparison source changed: " + str(p))


def prepare():
    verify_sources()
    if (ROOT / "reports/real_video_spatial_components/finalization.json").exists():
        raise ValueError("Pending arms now have a finalized study. Review/update the display snapshot explicitly; do not preserve stale training labels.")
    spec = importlib.util.spec_from_file_location("spatial_versions_independent_checker", CHECKER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    # collect validates full completion/selected identities/window arithmetic;
    # summarize independently reproduces all sixteen session×seed intervals.
    official, rows, summary, evidence, population = helper.collect()
    effects = []
    for index, effect in enumerate(official["paired_effects"]):
        e = dict(effect)
        d = e["comparator_mean"]-e["method_mean"]
        assert np.isclose(d, -e["method_minus_comparator"], rtol=1e-12, atol=1e-12)
        effects.append({**e, "result_id": f"spatial_effect_{index:02d}", "source_location": f"paired_effects[{index}]",
                        "mse_reduction_x1000": 1000*d,
                        "reduction_ci95_x1000": [-1000*e["paired_95_percent_interval"][1], -1000*e["paired_95_percent_interval"][0]],
                        "display_comparator": DISPLAY[e["comparator"]],
                        "in_main_figure": e["metric"] == "native_mse"})
    evidence[str(CHECKER.relative_to(ROOT))] = CHECKER_SHA
    evidence[str(COMPONENT_REG.relative_to(ROOT))] = COMPONENT_REG_SHA
    ledger = {"status": "measured_source_verified", "snapshot_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "original validation development; no independent test or SOTA claim", "population": population,
              "all_16_effects": effects, "absolute_error": official["aggregate"], "versions": VERSIONS,
              "pending_results": {mode: {"value": None, "missing": True, "status": "training"} for mode in PENDING},
              "ours1_excluded_reason": "Different original context-based protocol; no common-axis numerical comparison",
              "uncertainty": "Paired95% recording-session × training-seed bootstrap;10000draws;seed173;unadjusted",
              "axis_transform": "Gain=1000*(comparator MSE−Ours5 MSE); CI negates and reverses source method-minus-comparator endpoints",
              "absolute_error_marks": "means only; uncertainty shown on paired differences, not invented on absolute means",
              "source_sha256": evidence, "independently_recomputed_intervals": 16,
              "completed_runs": len(rows), "epochs_each": 30, "no_new_predictions": True}
    write("_evidence.json", ledger)
    return official, effects, ledger


def style(ax):
    ax.grid(axis="x", color=LIGHT, lw=.65, zorder=0)
    ax.tick_params(axis="both", labelsize=8, length=2.5, color=GRAY)
    ax.tick_params(axis="y", length=0, pad=7)
    for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY); ax.spines["bottom"].set_linewidth(.7)


def absolute(ax, official, horizon, labels=True):
    ax.set_ylim(-.6, 7.6); ax.set_xlim(.13, .24); ax.set_xticks([.14, .17, .20, .23])
    ax.set_yticks(range(8), [DISPLAY[m] for m in reversed(ORDER)] if labels else [""]*8)
    style(ax); ax.axhspan(1.58, 2.42, color=PALE, zorder=0)
    marks = {"autoregressive": "o", "persistence": "s", "anchored_additive": "^", "transport": "D", "context_off": "x", "action_free": "+"}
    for i, mode in enumerate(ORDER):
        y = 7-i
        if mode in PENDING:
            ax.text(.185, y, "training", color=GRAY, fontsize=8, ha="center", va="center", style="italic")
            continue
        metric = "native_persistence_mse" if mode == "persistence" else "native_mse"
        values = official["aggregate"]["autoregressive" if mode == "persistence" else mode][metric]
        x = values[horizon-1]
        ax.plot(x, y, marker=marks[mode], color=GREEN if mode=="transport" else INK, ms=5.2, mew=1.1, zorder=3)
        if mode == "transport":
            ax.annotate(f"{x:.4f}", (x,y), xytext=(7,0), textcoords="offset points", va="center", color=GREEN, fontsize=8, weight="bold")
    ax.set_xlabel("Mean feature MSE  ↓", fontsize=8, labelpad=6)


def gains(ax, effects, horizon, labels=True):
    ax.set_ylim(-.55, 3.6); ax.set_xlim(-2.8, 13.5); ax.set_xticks([0, 5, 10])
    ax.set_yticks(range(4), [DISPLAY[m] for m in reversed(COMPARATORS)] if labels else [""]*4)
    style(ax); ax.axvline(0, color=GRAY, lw=1, ls=(0,(3,3)), zorder=1)
    for i, mode in enumerate(COMPARATORS):
        e = next(e for e in effects if e["metric"] == "native_mse" and e["horizon"] == horizon and e["comparator"] == mode)
        x = e["mse_reduction_x1000"]; lo, hi = e["reduction_ci95_x1000"]
        y = 3-i
        color = GRAY if e["interval_includes_zero"] else GREEN
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt="D",ms=4.4,color=color,elinewidth=1.3,capsize=3,capthick=.9,zorder=3)
        # Separate numeric column: never cover a confidence interval with text.
        ax.text(1.49,y,f"{e['relative_error_reduction_percent']:+.2f}%",ha="right",va="center",fontsize=8,
                transform=ax.get_yaxis_transform(),clip_on=False,
                weight="normal" if e["interval_includes_zero"] else "bold",color=color,
                )
    ax.set_xlabel("MSE reduction ×10⁻³  →", fontsize=8, labelpad=6)


def render(official, effects):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "svg.fonttype":"none", "pdf.fonttype":42})
    fig=plt.figure(figsize=(5.5,4.9),dpi=160)
    fig.text(.025,.975,"(a) Native spatial error",fontsize=9,weight="bold",va="top",color=INK)
    for horizon,x in ((5,.27),(10,.67)):
        ax=fig.add_axes([x,.555,.315,.325]);absolute(ax,official,horizon,labels=horizon==5)
        ax.set_title(f"Horizon {horizon}",fontsize=9,weight="bold",pad=9,color=INK)
    fig.text(.025,.43,"(b) Ours-5 gains over trained comparators",fontsize=9,weight="bold",color=INK)
    for horizon,x in ((5,.27),(10,.67)):
        ax=fig.add_axes([x,.16,.215,.235]);gains(ax,effects,horizon,labels=horizon==5)
    fig.text(.025,.035,"Bars: 95% paired CI. Labels: relative point gains.",fontsize=8,color=GRAY)
    fig.canvas.draw()
    if (SKILL/"scripts/layout_quality.py").exists():
        sys.path.insert(0,str(SKILL/"scripts"));from layout_quality import audit_figure
        issues=audit_figure(fig,min_font_pt=8,display_width_inches=5.5)
        status="checked"
    else:
        issues=None;status="not_run_optional_skill_unavailable"
    write("_layout_audit.json",{"status":status,"issues":issues,"width_inches":5.5,"height_inches":4.9,"renderer_sha256":sha(__file__)})
    for ext in ("pdf","svg","png"):fig.savefig(OUT/f"{PREFIX}_comparison.{ext}",dpi=300,facecolor="white")
    plt.close(fig)
    return issues


def sketches(official,effects):
    for choice in (1,2,3):
        fig=plt.figure(figsize=(5.5,3.6),dpi=130)
        fig.text(.03,.96,{1:"A  Two horizon columns; errors over paired gains (selected)",2:"B  Paired horizon dumbbells; compact but conflates change",3:"C  Four metric/horizon forests; complete but loses raw error"}[choice],va="top",fontsize=9,weight="bold")
        if choice==1:
            for h,x in ((5,.28),(10,.68)):
                a=fig.add_axes([x,.48,.28,.32]);absolute(a,official,h,h==5)
                b=fig.add_axes([x,.11,.28,.23]);gains(b,effects,h,h==5)
                a.tick_params(labelsize=6);b.tick_params(labelsize=6)
        elif choice==2:
            a=fig.add_axes([.28,.16,.65,.64]);a.set_yticks(range(5),[DISPLAY[m] for m in reversed(tuple(official['aggregate']))]);style(a)
            for i,m in enumerate(official['aggregate']):
                values=official['aggregate'][m]['native_mse'];a.plot([values[4],values[9]],[4-i]*2,'-',color=GRAY)
                a.plot(values[4],4-i,'o',color=INK);a.plot(values[9],4-i,'D',color=GREEN)
            a.set_xlabel("Native feature MSE; circle=h5, diamond=h10")
        else:
            for index,(metric,h) in enumerate((('native_mse',5),('native_mse',10),('original_2x2_mse',5),('original_2x2_mse',10))):
                a=fig.add_axes([.28+.40*(index%2),.55-.40*(index//2),.28,.25]);style(a);a.axvline(0,color=GRAY,ls='--',lw=.8)
                a.set_yticks(range(4),[DISPLAY[m] for m in reversed(COMPARATORS)] if index%2==0 else ['']*4);a.tick_params(labelsize=6)
                for i,m in enumerate(COMPARATORS):
                    e=next(e for e in effects if e['metric']==metric and e['horizon']==h and e['comparator']==m);x=e['mse_reduction_x1000'];lo,hi=e['reduction_ci95_x1000']
                    a.errorbar(x,3-i,xerr=[[x-lo],[hi-x]],fmt='D',ms=3,color=INK)
                a.set_title(f"{'Native' if metric=='native_mse' else 'Coarse'} · h{h}",fontsize=8)
        fig.savefig(OUT/f"{PREFIX}_composition_{choice}.png",dpi=130);plt.close(fig)


def tex_escape(text):
    return text.replace('%',r'\%').replace('×',r'$\times$').replace('−',r'$-$').replace('–','--')


def tables(effects):
    legend=[r'\begin{table}[t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}lp{.49\linewidth}p{.27\linewidth}@{}}',r'\toprule ID & Mechanism & Status \\',r'\midrule']
    for row in VERSIONS: legend.append(f"{row['id']} & {row['mechanism']} & {row['status']} \\\\")
    legend.extend([r'\bottomrule\end{tabular}',r'\caption{Stable mechanism identifiers, not a performance ranking. Ours-1 uses the original context-based protocol and is excluded from the spatial comparison axis. Context-off and action-free are Ours-5 ablations; autoregression and persistence retain baseline labels. Training statuses refer to this display snapshot.}',r'\label{tab:spatial-version-key}',r'\end{table}'])
    (OUT/f"{PREFIX}_legend.tex").write_text('\n'.join(legend)+'\n')
    lines=[r'\begin{table}[p]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}llrrl@{}}',r'\toprule Metric & Comparator & $h$ & Relative gain & MSE reduction $\times10^3$ [95\% CI] \\',r'\midrule']
    for metric in ('native_mse','original_2x2_mse'):
        for comparator in COMPARATORS:
            for horizon in (5,10):
                e=next(e for e in effects if e['metric']==metric and e['comparator']==comparator and e['horizon']==horizon)
                point=f"{e['relative_error_reduction_percent']:+.3f}"+r'\%'
                if e['relative_error_reduction_percent']>0:point=r'\textcolor{gainpositive}{\textbf{'+point+'}}'
                lo,hi=e['reduction_ci95_x1000']; label='Native' if metric=='native_mse' else r'Original $2\!\times\!2$'
                lines.append(f"{label} & {tex_escape(DISPLAY[comparator])} & {horizon} & {point} & {e['mse_reduction_x1000']:+.3f} [{lo:+.3f}, {hi:+.3f}] \\\\")
        if metric=='native_mse':lines.append(r'\midrule')
    lines.extend([r'\bottomrule\end{tabular}',r'\caption{All 16 completed spatial contrasts: Ours-5 versus each comparator. Positive reductions favor Ours-5; bold green percentages indicate positive point estimates, not statistical significance. Paired 95\% recording-session and seed bootstrap intervals use 10,000 draws (seed173), without multiplicity correction. Both unfavorable coarse-coordinate context-ablation point estimates remain included. Native and original $2\times$2 coordinate errors are different metrics and cannot be pooled. Original validation only.}',r'\label{tab:spatial-versions-all}',r'\end{table}'])
    (OUT/f"{PREFIX}_all_comparisons.tex").write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    official,effects,ledger=prepare();sketches(official,effects);issues=render(official,effects);tables(effects)
    caption=("Stable spatial version comparison on original DROID validation. (a) Absolute native 4×4 standardized feature MSE at query endpoints 5 and 10; dots are equal-episode/equal-seed means. Ours-3/4 have no plotted values while their registered full studies train. (b) Paired MSE reductions of Ours-5 against the four registered trained comparators; right of zero favors Ours-5. Bars are 95% paired recording-session × training-seed bootstrap intervals (10,000 draws; seed 173); percentages are relative point reductions. Gray intervals include zero; green intervals exclude zero in the favorable direction, without multiplicity adjustment. All 15 completed models trained 30 epochs on identical data; evaluation has 1,631 windows, 141 episodes and 59 sessions with 3 training seeds. Windows are averaged within episodes before equal episode/seed aggregation. Ours-1 uses a different protocol and is not numerically combined here. Versions identify mechanisms, not ranks. The complete companion table retains all 16 native/coarse contrasts, including both unfavorable coarse context-ablation point estimates. These development comparisons establish neither independent-test generalization nor state-of-the-art superiority.")
    (OUT/f"{PREFIX}_caption.txt").write_text(caption+'\n')
    (OUT/f"{PREFIX}_figure.tex").write_text(r'''\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{generated/real_video/spatial_versions_comparison.pdf}
\caption[Spatial versions and paired improvements.]{'''+tex_escape(caption)+r'''}
\label{fig:spatial-versions}
\end{figure}
''')
    write("_brief.json",{"mode":"experimental comparison","paper_width_inches":5.5,
        "question":"Which completed spatial mechanisms lower native feature prediction error, and how uncertain are paired gains?",
        "representation":"Aligned horizon columns; absolute mean dots above paired difference intervals. Pending arms are text-only gaps; original-context protocol is excluded.",
        "three_compositions":["selected two-horizon error+gain panels","rejected horizon dumbbell conflates horizon deterioration with method gain","rejected four metric forests loses absolute-error context"],
        "scope_decision":"Native4x4 main panels match the registered primary metric; complete companion table preserves all16effects including allcoarse negatives.",
        "asset_choice":"Original vector marks only; generated images cannot substantiate measured numeric comparisons.",
        "pending_snapshot_utc":ledger['snapshot_utc'],"novelty_boundary":"Names identify mechanisms, not demonstrated rank or independent contribution evidence.",
        "visual_review":"pending actual pixels and official-width proof","renderer_sha256":sha(__file__)})
    verify_sources()
    print(json.dumps({"status":"rendered_for_inspection","effects":16,"pending_numeric_marks":0,"layout_issues":issues,"prefix":str(OUT/PREFIX)}))


if __name__=="__main__":main()
