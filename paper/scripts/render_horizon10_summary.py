#!/usr/bin/env python3
"""Portable source-bound h10 figure; reads saved ledgers, never predicts.

Run from the repository: python paper/scripts/render_horizon10_summary.py.
Requires the pinned completed report and all 60 evaluation ledgers in their
registered paths; these are release provenance, not manufactured chart data.
Output remains artifacts/qualitative/horizon10_summary_candidate for review.
An absent optional figure skill is explicitly reported as an unrun audit.
Publication copies are intentionally not replaced automatically.
"""
from pathlib import Path
import os
import csv
import hashlib
import importlib.util
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"artifacts/qualitative/horizon10_summary_candidate"
REPORT=ROOT/'reports/real_droid_horizon10_results.json'
EXPECTED_REPORT='38ebb812819d4326c568aeb1d6fdf4c5422856f9e3c8e683de2f5a25bc84fe91'
CHECKER=ROOT/'scripts/real_video_development/finalize_horizon10_paper.py'
EXPECTED_CHECKER='943c8ff410095949a23111252ab92b494ef83a9ace79a9da459272da806c3513'
# Optional local skill: public numerical rendering has no server-path dependency.
# Override with PAPER_FIGURE_SKILL_DIR=/path/to/paper-figure-creation.
SKILL=Path(os.environ.get('PAPER_FIGURE_SKILL_DIR',str(Path.home()/'.codex/skills/paper-figure-creation'))).expanduser()
GREEN='#166534'; INK='#263546'; GRAY='#64748B'; LIGHT='#E4E9EE'; AMBER='#99501B'
MODES=('framewise','constant_dynamics','factorized','action_free')
LABELS={'framewise':'Framewise','constant_dynamics':'Constant dynamics','factorized':'ShiftWM (ours)','action_free':'Action-free'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def prepare():
    assert sha(REPORT)==EXPECTED_REPORT and sha(CHECKER)==EXPECTED_CHECKER
    report=json.loads(REPORT.read_text());assert report['status']=='completed'
    assert len(report['runs'])==12 and len(report['comparisons'])==20
    assert all(r['receipt']['training']['completed_epochs']==30 and r['receipt']['training']['best_epoch']==1 for r in report['runs'])
    records=[];sources={str(REPORT.relative_to(ROOT)):sha(REPORT),str(CHECKER.relative_to(ROOT)):sha(CHECKER)}
    for row in report['runs']:
        r={'mode':row['mode'],'seed':row['seed'],'evaluations':{}}
        for key,value in row['evaluations'].items():
            source=ROOT/value['source'];assert sha(source)==value['source_sha256']
            saved=json.loads(source.read_text());r['evaluations'][key]={'episodes':saved['result']['episodes'],'summary':saved['result']['summary']}
            sources[value['source']]=sha(source)
        records.append(r)
    # Reuse only the independent arithmetic/CI function, not collect(), which
    # also executes model inference. This figure never decodes or predicts.
    spec=importlib.util.spec_from_file_location('h10_figure_independent_checker',CHECKER)
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    _,checked=checker.aggregate(records,report)
    contrasts=[]
    for index,row in enumerate(report['comparisons']):
        delta=row['first_mean']-row['second_mean']
        gain=100*(row['second_mean']-row['first_mean'])/row['second_mean']
        assert np.isclose(delta,row['paired_first_minus_second']['mean_difference'],rtol=1e-12,atol=1e-12)
        contrasts.append({'id':f'contrast_{index:02d}','source_location':f'comparisons[{index}]',**row,
                          'plot_delta_mse_x1000':1000*delta,'plot_ci95_x1000':[1000*v for v in row['paired_first_minus_second']['ci95']],
                          'recomputed_reduction_percent':gain,'ci_independently_recomputed':True})
    a=[next(c for c in contrasts if c['type']=='horizon_training_control' and c['mode']==m and c['horizon']==10 and c['metric']=='mean_standardized_mse') for m in MODES]
    b=[next(c for c in contrasts if c['type']=='matched_h10_training_modes' and c['horizon']==h and c['metric']==metric)
       for h,metric in [(5,'mean_standardized_mse'),(5,'h5_standardized_mse'),(10,'mean_standardized_mse'),(10,'h10_standardized_mse')]]
    negative=next(c for c in contrasts if c['type']=='horizon_training_control' and c['mode']=='factorized' and c['horizon']==5 and c['metric']=='mean_standardized_mse')
    ledger={'status':'source_verified','source_sha256':sources,'all_20_contrasts':contrasts,
            'panels':{'a':[c['id'] for c in a],'b':[c['id'] for c in b]},'explicit_unfavorable_secondary':negative['id'],
            'completed_runs':12,'epochs_each':30,'selected_epochs':[1]*12,'scope':'original validation development only',
            'uncertainty':'95% paired recording-session × training-seed bootstrap;10000 draws;seed5198010;unadjusted',
            'axis_transform':'1000 × signed first-minus-second standardized feature MSE. No percent-CI transformation.',
            'population':'141 episodes/59 sessions. Standardh5:1772 windows;h10 and matchedh5 prefixes:1631 windows.',
            'no_new_model_compute':True,'independent_ci_check_count':len(checked)}
    write(OUT/'evidence_ledger.json',ledger)
    with (OUT/'all20_contrasts.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['id','type','mode','horizon','metric','first_mse','second_mse','relative_reduction_percent','paired_difference','ci_low','ci_high','plotted_panel'])
        for c in contrasts:
            ci=c['paired_first_minus_second']['ci95'];panel='a' if c in a else 'b' if c in b else 'secondary'
            writer.writerow([c['id'],c['type'],c.get('mode','factorized vs framewise'),c['horizon'],c['metric'],c['first_mean'],c['second_mean'],c['recomputed_reduction_percent'],c['first_mean']-c['second_mean'],*ci,panel])
    return a,b,negative,ledger


def axes_style(ax,limits=(-3.1,.35),labels=None):
    ax.set_xlim(limits);ax.set_ylim(-.5,3.5)
    ax.axvline(0,color=GRAY,lw=.9,ls=(0,(3,3)),zorder=1)
    ax.set_yticks(range(4),labels or []);ax.tick_params(axis='y',length=0,pad=8)
    ax.set_xticks([-3,-2,-1,0]);ax.tick_params(axis='x',length=3,color=GRAY)
    ax.grid(axis='x',color=LIGHT,lw=.65,zorder=0)
    for s in ('top','right','left'):ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRAY);ax.spines['bottom'].set_linewidth(.7)


def rows(ax,contrasts,labels,fig=None,gain_x=None):
    for index,c in enumerate(contrasts):
        y=3-index;ours=c.get('mode')=='factorized' or c['type']=='matched_h10_training_modes'
        color=GREEN if ours else INK;marker='D' if ours else 'o'
        d=c['plot_delta_mse_x1000'];lo,hi=c['plot_ci95_x1000']
        ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt=marker,markersize=4.5,color=color,
                    markeredgewidth=.8,elinewidth=1.4,capsize=3,capthick=1.0,zorder=3)
        if fig is not None:
            fy=fig.transFigure.inverted().transform(ax.transData.transform((0,y)))[1]
            fig.text(gain_x,fy,f"+{c['recomputed_reduction_percent']:.3f}%",ha='center',va='center',
                     fontsize=8.5,fontweight='bold',color=GREEN)


def final_figure(a,b,negative):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelcolor':INK,'text.color':INK,
                         'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.default':'regular'})
    fig=plt.figure(figsize=(5.5,4.70),dpi=200)
    fig.text(.035,.965,'Horizon matching on real DROID recordings',fontsize=11.2,fontweight='bold',va='top')
    fig.text(.035,.914,'Validation only  ·  12 full 30-epoch runs  ·  selected epoch 1 for every model',fontsize=8.0)
    fig.text(.035,.845,'a   Match the training horizon',fontsize=9.5,fontweight='bold')
    fig.text(.035,.805,'h10-trained − h5-trained  ·  mean of steps 1–10',fontsize=8.5)
    fig.text(.928,.805,'Reduction',fontsize=8.,ha='center')
    axa=fig.add_axes([.29,.535,.53,.235]);axes_style(axa,labels=list(reversed([LABELS[m] for m in MODES])))
    rows(axa,a,[LABELS[m] for m in MODES],fig,.928)
    axa.tick_params(axis='x',labelbottom=False)
    axa.get_yticklabels()[1].set_fontweight('bold');axa.get_yticklabels()[1].set_color(GREEN)
    fig.text(.035,.468,'b   Compare equally h10-trained models',fontsize=9.5,fontweight='bold')
    fig.text(.035,.428,'ShiftWM (ours) − Framewise',fontsize=8.5)
    fig.text(.928,.428,'Reduction',fontsize=8.,ha='center')
    axb=fig.add_axes([.29,.182,.53,.21]);axes_style(axb,labels=list(reversed(['h5 mean','h5 endpoint','h10 mean','h10 endpoint'])))
    rows(axb,b,['h5 mean','h5 endpoint','h10 mean','h10 endpoint'],fig,.928)
    axb.set_xlabel('Δ standardized feature MSE  (×10⁻³)',labelpad=7,fontsize=8.5)
    fig.text(.555,.055,'←  Lower error  ·  bars: paired 95% CI',ha='center',fontsize=8.)
    # Explicit secondary regression, retained at readable size rather than hidden
    # in a provenance file. CI numbers and window scope are stated in caption.
    fig.text(.035,.013,f"Secondary: ours, matched h5 mean {negative['recomputed_reduction_percent']:.3f}% (CI includes zero)",
             fontsize=8.,color=AMBER)
    for ax in (axa,axb):
        for c in (a if ax is axa else b):assert ax.get_xlim()[0]<c['plot_ci95_x1000'][0]<c['plot_ci95_x1000'][1]<ax.get_xlim()[1]
    if (SKILL/'scripts/layout_quality.py').is_file():
        spec=importlib.util.spec_from_file_location('h10_optional_layout_quality',SKILL/'scripts/layout_quality.py')
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        issues=audit.audit_figure(fig,min_font_pt=8,display_width_inches=5.5,check_data_occlusion=True)
        audit_status='checked'
        reason=None
    else:
        issues=None
        audit_status='not_run'
        reason='Optional paper-figure-creation skill unavailable; set PAPER_FIGURE_SKILL_DIR. No geometry-audit pass is claimed.'
    write(OUT/'layout_audit.json',{'status':audit_status,'reason':reason,'issues':issues,'width_inches':5.5,
          'height_inches':4.7,'min_font_pt':8,'renderer_sha256':sha(__file__)})
    for suffix in ('pdf','svg','png'):fig.savefig(OUT/('horizon10_summary.'+suffix),dpi=300,facecolor='white')
    plt.close(fig)
    return issues


def sketches(a,b,all_contrasts):
    """Three different evidence topologies, all populated with real source values."""
    for choice in (1,2,3):
        fig=plt.figure(figsize=(5.5,3.8),dpi=130)
        fig.text(.04,.96,{1:'1  Stacked forests · common effect scale (selected)',2:'2  Side-by-side forests · compact but crowded',3:'3  Raw-MSE dumbbells + all-contrast strip'}[choice],fontsize=10,weight='bold',va='top')
        if choice==1:
            for panel,rs,y,title in [('A',a,.55,'Training horizon'),('B',b,.12,'Matched method')]:
                ax=fig.add_axes([.28,y,.65,.28]);axes_style(ax,labels=list(reversed([LABELS[c['mode']] if 'mode' in c else f"h{c['horizon']} {c['metric'].split('_')[0]}" for c in rs])))
                rows(ax,rs,[]);ax.set_title(title,loc='left',fontsize=9)
        elif choice==2:
            for rs,x,title in [(a,.20,'Training horizon'),(b,.68,'Matched method')]:
                ax=fig.add_axes([x,.2,.29,.57]);axes_style(ax,labels=list(reversed([LABELS[c['mode']] if 'mode' in c else f"h{c['horizon']} {c['metric'].split('_')[0]}" for c in rs])))
                rows(ax,rs,[]);ax.tick_params(labelsize=7);ax.set_title(title,fontsize=8)
        else:
            ax=fig.add_axes([.27,.47,.65,.32]);ax.set_yticks(range(4),list(reversed([LABELS[m] for m in MODES])))
            for i,c in enumerate(a):
                ax.plot([c['first_mean'],c['second_mean']],[3-i]*2,'-',color=GRAY,lw=1.5)
                ax.plot(c['first_mean'],3-i,'D',color=GREEN,ms=4);ax.plot(c['second_mean'],3-i,'o',color=INK,ms=4)
            ax.set_xlabel('Raw reported standardized MSE',fontsize=8)
            ax2=fig.add_axes([.15,.12,.78,.2]);ax2.axhline(0,color=GRAY,ls='--',lw=.8)
            for i,c in enumerate(all_contrasts):
                d=c['plot_delta_mse_x1000'];lo,hi=c['plot_ci95_x1000']
                ax2.errorbar(i,d,yerr=[[d-lo],[hi-d]],fmt='o',ms=2,color=GREEN if d<0 else AMBER,lw=.8)
            ax2.set_xticks([]);ax2.set_ylabel('Δ ×10⁻³',fontsize=8);ax2.set_xlabel('All 20 registered contrasts',fontsize=8)
        fig.savefig(OUT/f'composition_{choice}.png',dpi=150);plt.close(fig)


def caption_to_latex(caption):
    """Keep the exact caption wording while escaping its measured symbols."""
    for old,new in [('%',r'\%'),('×',r'$\times$'),('−',r'$-$'),('–','--')]:
        caption=caption.replace(old,new)
    return caption


def write_figure_include(path,caption_tex):
    """Inline the reviewed text: input inside a moving caption breaks hyperref."""
    text=(r'''% Reviewed fixed result figure; public renderer stages candidates separately.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{generated/real_video/horizon10_summary.pdf}
\caption[Horizon matching on real DROID recordings.]{'''+caption_tex.strip()+r'''}
\label{fig:horizon10_summary}
\end{figure}
''')
    Path(path).write_text(text)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a,b,negative,ledger=prepare();sketches(a,b,ledger['all_20_contrasts']);issues=final_figure(a,b,negative)
    caption=("Horizon matching on real DROID recordings (original validation only). "
        "(a) For each unchanged mode, h10-trained minus original h5-trained standardized feature MSE, averaged over steps 1–10 on identical h10-eligible windows. Training and checkpoint-selection horizons change together. "
        "(b) ShiftWM (ours; factorized) minus equally h10-trained Framewise, for all-query mean and endpoint errors at h5 and h10. "
        "Dots and bars show signed MSE differences and paired 95% recording-session × training-seed bootstrap intervals (10,000 draws; seed 5198010). Negative differences favor the first model. Bold green percentages are relative point reductions, not confidence intervals. "
        "Both panels use the same absolute-error axis. There are 141 eligible episodes from 59 sessions and 3 model seeds; standard h5 has 1,772 windows, while h10 and matched h5 prefixes have 1,631. Windows are averaged within episodes, then episodes and seeds equally. "
        "All 12 runs completed 30 epochs; all selected epoch 1 by the fixed window-weighted validation criterion. "
        f"The unfavorable secondary matched h5-prefix mean for factorized is {negative['recomputed_reduction_percent']:.3f}% relative reduction, a small increase in error; new-minus-original MSE difference {negative['first_mean']-negative['second_mean']:+.7f}, paired 95% CI [{negative['paired_first_minus_second']['ci95'][0]:+.7f}, {negative['paired_first_minus_second']['ci95'][1]:+.7f}]. "
        "All 20 registered contrasts are retained in the companion evidence ledger. These are exploratory, multiplicity-unadjusted validation comparisons, not independent test or state-of-the-art claims.")
    (OUT/'caption.txt').write_text(caption+'\n')
    caption_tex=caption_to_latex(caption)
    (OUT/'caption.tex').write_text(caption_tex+'\n')
    write_figure_include(OUT/'horizon10_summary_figure.tex',caption_tex)
    assert sha(REPORT)==EXPECTED_REPORT and sha(CHECKER)==EXPECTED_CHECKER
    print(json.dumps({'status':'rendered_for_visual_inspection','all_contrasts':20,'plotted':8,'layout_audit_status':'checked' if issues is not None else 'not_run','layout_issues':issues,'output':str(OUT)},indent=2))


if __name__=='__main__':main()
