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
COMPONENT_REPORT = ROOT / "reports/real_video_spatial_components/finalization.json"
COMPONENT_REPORT_SHA = "7759884a63ce97de726f36241329590f134a2dc3909416c8afae2f39fd661943"
SKILL = Path(os.environ.get("PAPER_FIGURE_SKILL_DIR", str(Path.home()/".codex/skills/paper-figure-creation"))).expanduser()
GREEN = "#166534"; INK = "#253746"; GRAY = "#657581"; LIGHT = "#E7ECEF"; PALE = "#EFF7F1"
DISPLAY = {"autoregressive": "Autoregressive", "persistence": "Persistence", "anchored_additive": "Ours-2",
           "bounded_additive": "Ours-3", "unbounded_transport": "Ours-4", "transport": "Ours-5",
           "context_off": "Ours-5 − context", "action_free": "Ours-5 − actions"}
ORDER = ("autoregressive", "persistence", "anchored_additive", "bounded_additive", "unbounded_transport", "transport", "context_off", "action_free")
COMPARATORS = ("autoregressive", "anchored_additive", "context_off", "action_free")
PLOT_COMPARATORS = ("autoregressive", "anchored_additive", "bounded_additive", "unbounded_transport", "context_off", "action_free")
PENDING = ()
VERSIONS = [
    {"id": "Ours-1", "mode": "factorized", "mechanism": "Original context-based ShiftWM", "status": "Completed; different protocol", "spatial_axis": False},
    {"id": "Ours-2", "mode": "anchored_additive", "mechanism": "Observed anchor + additive innovation", "status": "Completed", "spatial_axis": True},
    {"id": "Ours-3", "mode": "bounded_additive", "mechanism": "Observed anchor + bounded innovation", "status": "Completed; follow-up", "spatial_axis": True},
    {"id": "Ours-4", "mode": "unbounded_transport", "mechanism": "Gated mixing + unbounded innovation", "status": "Completed; follow-up", "spatial_axis": True},
    {"id": "Ours-5", "mode": "transport", "mechanism": "Gated mixing + bounded innovation", "status": "Completed", "spatial_axis": True},
]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def write(name, value):
    (OUT / (PREFIX + name)).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_sources():
    for p, expected in ((REPORT, REPORT_SHA), (CHECKER, CHECKER_SHA), (COMPONENT_REG, COMPONENT_REG_SHA), (COMPONENT_REPORT, COMPONENT_REPORT_SHA)):
        if sha(p) != expected: raise ValueError("Pinned comparison source changed: " + str(p))


def component_evidence(helper):
    path = ROOT / "scripts/real_video_spatial_components/campaign.py"
    spec = importlib.util.spec_from_file_location("spatial_versions_components", path)
    campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
    registry = campaign.verify()
    report = json.loads(COMPONENT_REPORT.read_text())
    if (report.get("status") != "passed" or report.get("registration_sha256") != COMPONENT_REG_SHA
            or report.get("completed_new_models") != 6 or report.get("completed_frozen_controls") != 6
            or report.get("epochs_per_model") != 30 or len(report.get("offline_cpu_parity", [])) != 6
            or {r['name'] for r in report['offline_cpu_parity']} != {r['name'] for r in registry['runs']}):
        raise ValueError("Incomplete finalized component study")
    sources = {**registry["dependencies"], **report["evidence_sha256"],
               str(COMPONENT_REPORT.relative_to(ROOT)): COMPONENT_REPORT_SHA}
    for name,digest in sources.items():
        if sha(ROOT/name) != digest: raise ValueError("Component source changed: " + name)
    for parity in report["offline_cpu_parity"]:
        package = ROOT/"artifacts/releases/spatial_components_v1/models"/parity["name"]
        manifest = json.loads((package/"package_manifest.json").read_text())
        if (parity.get("status") != "passed" or parity.get("max_abs_error") != 0
                or sha(package/"package_manifest.json") != parity["package_manifest_sha256"]
                or sha(package/"model.pt") != parity["selected_checkpoint_sha256"]
                or manifest.get("package_kind") != "shiftwm_real_video_spatial_components_v1"
                or set(manifest["files"]) != {"model.pt","config.json"}):
            raise ValueError("Component relocated inference proof differs")
        for name,digest in manifest["files"].items():
            if sha(package/name) != digest: raise ValueError("Component inference bytes differ")
        for name in ("model.pt","config.json","package_manifest.json"):
            sources[str((package/name).relative_to(ROOT))] = sha(package/name)
    rows = campaign.collect(registry)
    ledgers = [r[3] for r in rows]
    recomputed = campaign.module("analysis").contrasts(ledgers)
    if len(report["reported_effects"]) != 20: raise ValueError("Missing component contrasts")
    for saved,checked in zip(report["reported_effects"],recomputed["reported_effects"]):
        if {k:saved[k] for k in ('contrast','metric','horizon')} != {k:checked[k] for k in ('contrast','metric','horizon')}:
            raise ValueError("Component contrast ordering differs")
        for key in ('method_mean','comparator_mean','method_minus_comparator','relative_error_reduction_percent','difference_of_differences','paired_95_percent_interval'):
            if key in checked: helper.same(saved[key],checked[key],'component/'+key)
        if saved['contrast'] != 'mixing_x_bounding_interaction':
            independent = helper.independent_comparison([r for r in ledgers if r['mode']==saved['method']],
                            [r for r in ledgers if r['mode']==saved['comparator']],saved['metric'],saved['horizon'])
            for key,value in independent.items():
                if isinstance(value,bool):
                    if saved[key] != value: raise ValueError("Component interval inclusion differs")
                else: helper.same(saved[key],value,'independent-component/'+key)
    for mode,values in report['aggregate'].items():
        for metric,means in values.items():
            actual=np.mean([[np.mean([e[metric][h] for e in r['episodes']]) for h in range(10)]
                            for r in ledgers if r['mode']==mode],axis=0)
            helper.same(actual,means,'component-aggregate/'+mode+'/'+metric)
    return report,sources


def prepare():
    verify_sources()
    spec = importlib.util.spec_from_file_location("spatial_versions_independent_checker", CHECKER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    # collect validates full completion/selected identities/window arithmetic;
    # summarize independently reproduces all sixteen session×seed intervals.
    official, rows, summary, evidence, population = helper.collect()
    component, component_sources = component_evidence(helper)
    evidence.update(component_sources)
    # The old 15-model source report remains authoritative and unmodified.
    # Merge only into this separate display overlay after exact control parity.
    for mode in ('anchored_additive','transport'):
        for metric in official['aggregate'][mode]:
            helper.same(official['aggregate'][mode][metric],component['aggregate'][mode][metric],'shared-control/'+mode+'/'+metric)
    display_official = {**official, 'aggregate':{**official['aggregate'],
        **{mode:component['aggregate'][mode] for mode in ('bounded_additive','unbounded_transport')}}}
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
    component_effects=[]
    for index,e in enumerate(component['reported_effects']):
        row={**e,'result_id':f'component_effect_{index:02d}','source_location':f'reported_effects[{index}]'}
        if e['contrast'] != 'mixing_x_bounding_interaction':
            row.update(mse_reduction_x1000=-1000*e['method_minus_comparator'],
                       reduction_ci95_x1000=[-1000*e['paired_95_percent_interval'][1],-1000*e['paired_95_percent_interval'][0]],
                       display_comparator=DISPLAY[e['comparator']],
                       in_main_figure=e['method']=='transport' and e['metric']=='native_mse')
        else:
            row.update(interaction_x1000=1000*e['difference_of_differences'],
                       interaction_ci95_x1000=[1000*x for x in e['paired_95_percent_interval']],in_main_figure=False)
        component_effects.append(row)
    evidence[str(CHECKER.relative_to(ROOT))] = CHECKER_SHA
    evidence[str(COMPONENT_REG.relative_to(ROOT))] = COMPONENT_REG_SHA
    ledger = {"status": "measured_source_verified", "snapshot_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "original validation development; no independent test or SOTA claim", "population": population,
              "all_16_effects": effects, "all_20_component_effects":component_effects,
              "absolute_error": display_official["aggregate"], "versions": VERSIONS,
              "pending_results": {mode: {"value": None, "missing": True, "status": "training"} for mode in PENDING},
              "ours1_excluded_reason": "Different original context-based protocol; no common-axis numerical comparison",
              "uncertainty": "Paired95% recording-session × training-seed bootstrap;10000draws;seed173;unadjusted",
              "axis_transform": "Gain=1000*(comparator MSE−Ours5 MSE); CI negates and reverses source method-minus-comparator endpoints",
              "absolute_error_marks": "means only; uncertainty shown on paired differences, not invented on absolute means",
              "source_sha256": evidence, "independently_recomputed_intervals": 16,
              "component_pairwise_intervals_independently_recomputed":16,
              "component_all20_recomputed_via_registered_analysis":True,
              "completed_runs": len(rows)+6,"original_spatial_runs":15,"component_factorial_runs_including_reused_controls":12,
              "component_scope":"Follow-up development after original controls were revealed; not independent confirmation",
              "epochs_each": 30, "no_new_predictions": True}
    write("_evidence.json", ledger)
    return display_official, effects, component_effects, ledger


def style(ax):
    ax.grid(axis="x", color=LIGHT, lw=.65, zorder=0)
    ax.tick_params(axis="both", labelsize=8, length=2.5, color=GRAY)
    ax.tick_params(axis="y", length=0, pad=7)
    for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY); ax.spines["bottom"].set_linewidth(.7)


def absolute(ax, official, horizon, labels=True):
    ax.set_ylim(-.6, 7.6); ax.set_xlim(.13, .24); ax.set_xticks([.14, .17, .20, .23])
    ax.set_yticks(range(8), [DISPLAY[m] for m in reversed(ORDER)] if labels else [""]*8)
    style(ax)
    marks = {"autoregressive": "o", "persistence": "s", "anchored_additive": "^", "bounded_additive":"v",
             "unbounded_transport":"h", "transport": "D", "context_off": "x", "action_free": "+"}
    for i, mode in enumerate(ORDER):
        y = 7-i
        if mode in PENDING:
            ax.text(.185, y, "training", color=GRAY, fontsize=8, ha="center", va="center", style="italic")
            continue
        metric = "native_persistence_mse" if mode == "persistence" else "native_mse"
        values = official["aggregate"]["autoregressive" if mode == "persistence" else mode][metric]
        x = values[horizon-1]
        is_version=mode in ('anchored_additive','bounded_additive','unbounded_transport','transport')
        ax.plot(x, y, marker=marks[mode], color=GREEN if is_version else INK, ms=5.2, mew=1.1, zorder=3)
        if is_version:
            ax.annotate(f"{x:.4f}", (x,y), xytext=(7,0), textcoords="offset points", va="center", color=GREEN, fontsize=8, weight="bold")
    ax.set_xlabel("Mean feature MSE  ↓", fontsize=8, labelpad=6)


def gains(ax, effects, horizon, labels=True):
    ax.set_ylim(-.55, 5.6); ax.set_xlim(-2.8, 13.5); ax.set_xticks([0, 5, 10])
    ax.set_yticks(range(6), [DISPLAY[m] for m in reversed(PLOT_COMPARATORS)] if labels else [""]*6)
    style(ax); ax.axvline(0, color=GRAY, lw=1, ls=(0,(3,3)), zorder=1)
    for i, mode in enumerate(PLOT_COMPARATORS):
        e = next(e for e in effects if e["metric"] == "native_mse" and e["horizon"] == horizon and e["comparator"] == mode)
        x = e["mse_reduction_x1000"]; lo, hi = e["reduction_ci95_x1000"]
        y = 5-i
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
    fig=plt.figure(figsize=(5.5,5.3),dpi=160)
    fig.text(.025,.975,"(a) Native spatial error",fontsize=9,weight="bold",va="top",color=INK)
    for horizon,x in ((5,.27),(10,.67)):
        ax=fig.add_axes([x,.61,.315,.29]);absolute(ax,official,horizon,labels=horizon==5)
        ax.set_title(f"Horizon {horizon}",fontsize=9,weight="bold",pad=9,color=INK)
    fig.text(.025,.49,"(b) Paired gains for Ours-5",fontsize=9,weight="bold",color=INK)
    for horizon,x in ((5,.27),(10,.67)):
        ax=fig.add_axes([x,.14,.215,.295]);gains(ax,effects,horizon,labels=horizon==5)
    fig.text(.025,.035,"Bars: 95% paired CI. Labels: relative point gains.",fontsize=8,color=GRAY)
    fig.canvas.draw()
    if (SKILL/"scripts/layout_quality.py").exists():
        sys.path.insert(0,str(SKILL/"scripts"));from layout_quality import audit_figure
        issues=audit_figure(fig,min_font_pt=8,display_width_inches=5.5)
        status="checked"
    else:
        issues=None;status="not_run_optional_skill_unavailable"
    write("_layout_audit.json",{"status":status,"issues":issues,"width_inches":5.5,"height_inches":5.3,"renderer_sha256":sha(__file__)})
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
            modes=tuple(official['aggregate']);n=len(modes)
            a=fig.add_axes([.28,.16,.65,.64]);a.set_yticks(range(n),[DISPLAY[m] for m in reversed(modes)]);style(a)
            for i,m in enumerate(modes):
                values=official['aggregate'][m]['native_mse'];a.plot([values[4],values[9]],[n-1-i]*2,'-',color=GRAY)
                a.plot(values[4],n-1-i,'o',color=INK);a.plot(values[9],n-1-i,'D',color=GREEN)
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
    legend.extend([r'\bottomrule\end{tabular}',r'\caption{Stable mechanism identifiers, not a performance ranking. Ours-1 uses the original context-based protocol and is excluded from the spatial comparison axis. Context-off and action-free are Ours-5 ablations; autoregression and persistence retain baseline labels. Ours-3/4 completed their registered follow-up after the original controls were revealed; all six new models trained 30 epochs.}',r'\label{tab:spatial-version-key}',r'\end{table}'])
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


def component_table(effects):
    lines=[r'\begin{table}[p]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}llrrl@{}}',
           r'\toprule Comparison & Metric & $h$ & Relative gain & Effect $\times10^3$ [95\% CI] \\',r'\midrule']
    previous=None
    for e in effects:
        if previous is not None and previous!=e['contrast']:lines.append(r'\midrule')
        previous=e['contrast'];metric='Native' if e['metric']=='native_mse' else r'$2\!\times\!2$'
        if e['contrast']=='mixing_x_bounding_interaction':
            name='Interaction $D$';point=r'---';value=e['interaction_x1000'];lo,hi=e['interaction_ci95_x1000']
        else:
            name=DISPLAY[e['method']]+' vs '+DISPLAY[e['comparator']]
            point=f"{e['relative_error_reduction_percent']:+.3f}"+r'\%'
            if e['relative_error_reduction_percent']>0:point=r'\textcolor{gainpositive}{\textbf{'+point+'}}'
            value=e['mse_reduction_x1000'];lo,hi=e['reduction_ci95_x1000']
        lines.append(f"{name} & {metric} & {e['horizon']} & {point} & {value:+.3f} [{lo:+.3f}, {hi:+.3f}] \\\\")
    lines.extend([r'\bottomrule\end{tabular}',
        r'\caption{All 20 predeclared component effects from the original-validation follow-up: six new models plus six reused controls, each trained for 30 epochs. The first four blocks report error reductions (second-model MSE minus first-model MSE): positive favors the first model. Bounding is isolated within the Ours-3/Ours-2 and Ours-5/Ours-4 pairs; mixing is compared within Ours-4/Ours-2 and Ours-5/Ours-3. The last block retains the separately signed interaction $D=(\mathrm{MSE}_5-\mathrm{MSE}_4)-(\mathrm{MSE}_3-\mathrm{MSE}_2)$: negative means bounding helps more with mixing; no percentage is defined for $D$. All four interaction intervals include zero. Bold green denotes positive pairwise point estimates, not significance. Paired session/seed 95\% bootstrap intervals use 10,000 draws, seed 173, without multiplicity correction. This follow-up was designed after existing controls were revealed; mixing includes the gate, identity bias and approximately 1.84\% extra active parameters.}',
        r'\label{tab:spatial-versions-components}',r'\end{table}'])
    (OUT/f"{PREFIX}_component_comparisons.tex").write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    official,effects,component_effects,ledger=prepare()
    plot_effects=effects+[e for e in component_effects if e.get('method')=='transport']
    sketches(official,plot_effects);issues=render(official,plot_effects);tables(effects);component_table(component_effects)
    caption=("Completed spatial versions on original DROID validation. (a) Absolute native 4×4 standardized feature MSE at endpoints 5 and 10; green marks identify our mechanism variants, not a winner. (b) Ours-5 is the reference for paired reductions against six comparators; right of zero favors Ours-5. Only registered matching contrasts are used. Ours-4 and Ours-5 are close: both native comparison intervals include zero, and Ours-4 has the lower h5 point estimate. Bars show 95% paired recording-session × training-seed bootstrap intervals (10,000 draws; seed 173); percentages are relative point reductions. Gray intervals include zero; favorable green intervals exclude zero without multiplicity adjustment. All 21 spatial models completed 30 epochs: 15 original models plus 6 new component models; the four-arm factorial subset has 12 models including reused controls. Ours-3/4 were added in a registered follow-up after the original controls were revealed. Evaluation uses 1,631 matched windows, 141 episodes, 59 sessions and 3 seeds. Windows are averaged within episodes, then episodes and seeds equally. Ours-1 has a different protocol and remains off this axis. Companion tables preserve the original 16 comparisons and all 20 component effects, including four signed interactions and every unfavorable or inconclusive result. These are development comparisons, not independent-test or state-of-the-art claims.")
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
        "representation":"Aligned horizon columns; absolute mean dots above paired difference intervals. All completed versions share accent weight; Ours5 is a reference, not an asserted winner. Original-context protocol is excluded.",
        "three_compositions":["selected two-horizon error+gain panels","rejected horizon dumbbell conflates horizon deterioration with method gain","rejected four metric forests loses absolute-error context"],
        "scope_decision":"Native4x4 main panels match the registered primary metric; separate companion tables preserve original16 plus component20effects including allcoarse negatives and separately signed interactions.",
        "asset_choice":"Original vector marks only; generated images cannot substantiate measured numeric comparisons.",
        "completed_snapshot_utc":ledger['snapshot_utc'],"novelty_boundary":"Names identify mechanisms, not demonstrated rank or independent contribution evidence.",
        "visual_review":"pending actual pixels and official-width proof","renderer_sha256":sha(__file__)})
    verify_sources()
    print(json.dumps({"status":"rendered_for_inspection","original_effects":16,"component_effects":20,"completed_models":21,"layout_issues":issues,"prefix":str(OUT/PREFIX)}))


if __name__=="__main__":main()
