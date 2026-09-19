#!/usr/bin/env python3
"""Three shorter presentation candidates; never overwrites the included figure.

Reuses the source-verified values and styling of render_spatial_editorial.py.
Only paper/generated/editorial/spatial_compact_* files are written.
"""
from pathlib import Path
import importlib.util
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'paper/scripts/render_spatial_editorial.py'
spec=importlib.util.spec_from_file_location('spatial_compact_base',BASE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=b.OUT
MAIN_MODES=('transport','autoregressive','persistence','anchored_additive','action_free')
MAIN_LABELS={'transport':'ShiftWM (ours)','autoregressive':'Autoregressive','persistence':'Persistence',
             'anchored_additive':'Anchored additive','action_free':'Without actions'}
CONTROLS=('autoregressive','anchored_additive','action_free')


def paired(d,mode,h):
    return next(e for e in d['all_16_effects'] if e['comparator']==mode and e['metric']=='native_mse' and e['horizon']==h)


def main_curves(fig,values):
    heading(fig,.03,'(a) Forecast error')
    ax=fig.add_axes([.103,.385,.330,.445]);b.error_curves(ax,values,selected=MAIN_MODES[1:]+MAIN_MODES[:1])
    hs=[Line2D([0],[0],color=b.COLORS[m],ls=b.LINES[m],marker=b.MARKERS[m],ms=3,lw=1.1) for m in MAIN_MODES]
    fig.legend(hs,[MAIN_LABELS[m] for m in MAIN_MODES],loc='lower left',bbox_to_anchor=(.020,.025),
               ncol=2,frameon=False,fontsize=8,handlelength=1.50,columnspacing=.8,handletextpad=.4,labelspacing=.40)


def candidate_four(values,d):
    fig=plt.figure(figsize=(5.5,2.60),dpi=200);main_curves(fig,values)
    heading(fig,.51,'(b) ShiftWM gains')
    ax=fig.add_axes([.712,.330,.258,.495])
    ax.axvline(0,color=b.SECONDARY,lw=.7)
    for i,mode in enumerate(CONTROLS):
        for h,dy,marker,color in [(5,.14,'o',b.INK),(10,-.14,'D',b.GREEN)]:
            e=paired(d,mode,h);point=e['mse_reduction_x1000'];lo,hi=e['reduction_ci95_x1000']
            ax.errorbar(point,2-i+dy,xerr=[[point-lo],[hi-point]],fmt=marker,ms=3.4,color=color,lw=1.1,capsize=2)
    ax.set(xlim=(-.5,13.5),ylim=(-.5,2.5),xticks=[0,5,10],yticks=[2,1,0],yticklabels=['Autoregressive','Anchored additive','Without actions'])
    b.axis_style(ax);ax.grid(False);ax.grid(axis='x',color=b.GRID,lw=.55);ax.tick_params(axis='y',length=0,pad=4)
    ax.set_xlabel('MSE reduction ×10⁻³',labelpad=6)
    fig.legend([Line2D([0],[0],marker='o',color=b.INK,lw=1.1,ms=3.4),Line2D([0],[0],marker='D',color=b.GREEN,lw=1.1,ms=3.4)],
               ['h5','h10'],loc='lower center',bbox_to_anchor=(.78,.09),ncol=2,frameon=False,fontsize=8,handlelength=1.2)
    b.text(fig,.77,.035,'Paired 95% intervals',ha='center',color=b.SECONDARY)
    return fig


def candidate_five(values,d):
    fig=plt.figure(figsize=(5.5,2.60),dpi=200);main_curves(fig,values)
    heading(fig,.505,'(b) ShiftWM gains vs controls')
    for i,(mode,label) in enumerate(zip(CONTROLS,('Autoreg.','Anchor','No actions'))):
        ax=fig.add_axes([.586+i*.139,.345,.112,.458])
        es=[paired(d,mode,h) for h in (5,10)];points=[e['mse_reduction_x1000'] for e in es]
        ax.plot([5,10],points,color=b.GREEN,lw=1.05,zorder=2)
        for h,e,marker in zip((5,10),es,('o','D')):
            point=e['mse_reduction_x1000'];lo,hi=e['reduction_ci95_x1000']
            ax.errorbar(h,point,yerr=[[point-lo],[hi-point]],fmt=marker,ms=3.4,color=b.GREEN,lw=1.1,capsize=2,zorder=3)
        ax.set(xlim=(3.5,11.5),ylim=(0,14),xticks=[5,10],yticks=[0,5,10])
        b.axis_style(ax);ax.set_title(label,fontsize=8,pad=5)
        if i==0:ax.set_ylabel('MSE reduction ×10⁻³',fontsize=8.5,labelpad=4)
        else:ax.tick_params(axis='y',labelleft=False,length=0);ax.spines['left'].set_visible(False)
    b.text(fig,.782,.208,'Forecast step',ha='center',fontsize=8.5)
    b.text(fig,.782,.110,'Paired 95% intervals',ha='center',color=b.SECONDARY)
    return fig


def heading(fig,x,label):
    b.text(fig,x,.944,label,fontsize=9.5,weight='bold')


def effect(fig,x,y,e,gap=.050):
    lo,hi=e['reduction_ci95_x1000'];color=b.GREEN if lo>0 else b.SECONDARY
    b.text(fig,x,y,f"{e['mse_reduction_x1000']:+.2f}",ha='center',weight='bold',fontsize=8.5,color=color)
    b.text(fig,x,y-gap,f'[{lo:+.2f}, {hi:+.2f}]',ha='center',fontsize=8,color=color)


def handles():
    return [Line2D([0],[0],color=b.COLORS[m],ls=b.LINES[m],marker=b.MARKERS[m],ms=3,lw=1.1) for m in b.MODES]


def compact_numbered_map(fig,values,d):
    heading(fig,.53,'(b) Mixing × bounding')
    b.text(fig,.62,.86,'Anchor',ha='center',fontsize=8.5,color=b.BLUE)
    b.text(fig,.90,.86,'Gated mixing',ha='center',fontsize=8.5,color=b.GREEN)
    b.text(fig,.523,.715,'Raw δ',rotation=90,ha='center',va='center',color=b.SECONDARY)
    b.text(fig,.523,.275,'tanh δ',rotation=90,ha='center',va='center',color=b.SECONDARY)
    for mode,x,y in [('anchored_additive',.62,.773),('unbounded_transport',.90,.773),
                      ('bounded_additive',.62,.333),('transport',.90,.333)]:b.node(fig,x,y,mode,values)
    b.arrow(fig,(.700,.719),(.816,.719),b.GREEN)
    b.arrow(fig,(.700,.279),(.816,.279),b.GREEN)
    effect(fig,.758,.632,b.component(d,'mixing_without_bounding'))
    effect(fig,.758,.193,b.component(d,'mixing_with_bounding'))
    b.arrow(fig,(.554,.66),(.554,.375))
    b.arrow(fig,(.974,.66),(.974,.375))
    effect(fig,.630,.478,b.component(d,'bounding_without_mixing'))
    effect(fig,.885,.478,b.component(d,'bounding_with_mixing'))
    b.text(fig,.756,.077,'h10 error and reductions ×10⁻³',ha='center',color=b.SECONDARY)
    b.text(fig,.756,.022,'Brackets: paired 95% CI',ha='center',color=b.SECONDARY)


def compact_effect_map(fig,d,with_footer=True):
    """Keep the design cells but leave absolute scores to the exact main table."""
    heading(fig,.53,'(b) Mixing × bounding')
    b.text(fig,.62,.855,'Anchor',ha='center',fontsize=8.5,color=b.BLUE)
    b.text(fig,.90,.855,'Gated mixing',ha='center',fontsize=8.5,color=b.GREEN)
    for mode,x,y in [('anchored_additive',.62,.788),('unbounded_transport',.90,.788),
                      ('bounded_additive',.62,.410),('transport',.90,.410)]:
        b.text(fig,x,y,b.DISPLAY[mode],ha='center',weight='bold',fontsize=8.5,color=b.COLORS[mode])
    b.text(fig,.522,.800,'Raw δ',rotation=90,ha='center',va='center',color=b.SECONDARY)
    b.text(fig,.522,.420,'tanh δ',rotation=90,ha='center',va='center',color=b.SECONDARY)
    b.arrow(fig,(.695,.800),(.82,.800),b.GREEN)
    b.arrow(fig,(.695,.422),(.82,.422),b.GREEN)
    effect(fig,.758,.692,b.component(d,'mixing_without_bounding'),.052)
    effect(fig,.758,.314,b.component(d,'mixing_with_bounding'),.052)
    b.arrow(fig,(.554,.745),(.554,.463))
    b.arrow(fig,(.974,.745),(.974,.463))
    effect(fig,.630,.557,b.component(d,'bounding_without_mixing'),.052)
    effect(fig,.885,.557,b.component(d,'bounding_with_mixing'),.052)
    if with_footer:
        b.text(fig,.758,.192,'h10 reduction ×10⁻³ [paired 95% CI]',ha='center',color=b.SECONDARY)


def candidate_one(values,d):
    fig=plt.figure(figsize=(5.5,2.65),dpi=200)
    heading(fig,.03,'(a) Ten-step forecasts')
    ax=fig.add_axes([.103,.425,.365,.405]);b.error_curves(ax,values)
    fig.legend(handles(),[b.DISPLAY[m] for m in b.MODES],loc='lower left',bbox_to_anchor=(.02,.006),
               ncol=2,frameon=False,fontsize=8,handlelength=1.6,columnspacing=.75,handletextpad=.4,labelspacing=.32)
    compact_numbered_map(fig,values,d)
    return fig


def candidate_two(values,d):
    fig=plt.figure(figsize=(5.5,2.60),dpi=200)
    heading(fig,.03,'(a) Ten-step forecasts')
    ax=fig.add_axes([.103,.345,.365,.485]);b.error_curves(ax,values)
    # One figure-wide legend replaces four rows below the trajectory panel.
    fig.legend(handles(),[b.DISPLAY[m] for m in b.MODES],loc='lower center',bbox_to_anchor=(.50,.018),
               ncol=4,frameon=False,fontsize=8,handlelength=1.65,columnspacing=1.15,handletextpad=.42,labelspacing=.45)
    compact_effect_map(fig,d)
    return fig


def candidate_three(values,d):
    fig=plt.figure(figsize=(5.5,2.60),dpi=200)
    heading(fig,.03,'(a) Endpoint errors')
    b.text(fig,.035,.840,'Method',weight='bold',fontsize=8.5)
    b.text(fig,.340,.840,'h5',ha='right',weight='bold',fontsize=8.5)
    b.text(fig,.474,.840,'h10',ha='right',weight='bold',fontsize=8.5)
    fig.add_artist(Line2D([.033,.481],[.823,.823],transform=fig.transFigure,color=b.INK,lw=.7))
    for i,mode in enumerate(b.MODES):
        y=.767-i*.066
        b.text(fig,.035,y,b.DISPLAY[mode],fontsize=8,color=b.COLORS[mode])
        for x,h in ((.340,5),(.474,10)):
            b.text(fig,x,y,f'{values[mode][h-1]:.6f}',ha='right',fontsize=8)
    fig.add_artist(Line2D([.033,.481],[.275,.275],transform=fig.transFigure,color=b.INK,lw=.7))
    b.text(fig,.035,.220,'Ours-5 vs AR',fontsize=8)
    for x,h in ((.340,5),(.474,10)):
        e=next(e for e in d['all_16_effects'] if e['comparator']=='autoregressive' and e['metric']=='native_mse' and e['horizon']==h)
        b.text(fig,x,.220,f"{e['relative_error_reduction_percent']:+.2f}%",ha='right',fontsize=8,weight='bold',color=b.GREEN)
    compact_effect_map(fig,d)
    b.text(fig,.03,.078,'Native MSE; episode / seed means.',fontsize=8,color=b.SECONDARY)
    b.text(fig,.53,.078,'Trajectories would move to the appendix.',fontsize=8,color=b.SECONDARY)
    return fig


def main():
    b.theme();d,values=b.load_evidence()
    protected=[p for p in OUT.glob('spatial_*') if p.is_file()
               and not p.name.startswith(('spatial_compact_','spatial_architecture'))]
    before={str(p.relative_to(ROOT)):b.sha(p) for p in protected}
    records=[]
    for i,fn in enumerate((candidate_one,candidate_two,candidate_three,candidate_four,candidate_five),1):
        fig=fn(values,d);name=f'compact_candidate_{i}';qa=b.audit(fig);b.save(fig,name)
        records.append({'id':i,'dimensions_inches':fig.get_size_inches().tolist(),'layout':qa,
                        'artifact_sha256':{ext:b.sha(OUT/f'spatial_{name}.{ext}') for ext in ('pdf','svg','png')}})
        plt.close(fig)
    assert before=={str(p.relative_to(ROOT)):b.sha(p) for p in protected},'Current included figure was modified'
    record={'status':'candidate_review_only_current_included_figure_unchanged','created_utc':datetime.now(timezone.utc).isoformat(),
        'renderer_sha256':b.sha(__file__),'base_renderer_sha256':b.sha(BASE),'source_ledger_sha256':b.SOURCE_SHA,
        'source_bindings_verified':len(d['source_sha256'])+3,'all_80_trajectory_means':{m:values[m].tolist() for m in b.MODES},
        'all_4_native_h10_component_effects':[b.component(d,k) for k in ('mixing_without_bounding','mixing_with_bounding','bounding_without_mixing','bounding_with_mixing')],
        'candidates':records,'protected_current_artifacts_sha256':before,
        'main_six_paired_effects':[paired(d,m,h) for m in CONTROLS for h in (5,10)],
        'main_curve_modes':list(MAIN_MODES),'main_method':'transport = ShiftWM (ours)',
        'recommended_candidate':5,
        'recommendation':'One proposed ShiftWM: retain its forecast curves against three named controls plus persistence, and show the six paired native h5/h10 gains in three equal-axis horizon facets. Keep all eight exact model rows in the main table and the full 36-contrast/component study in the appendix. Saves 0.60 inches (18.75%) of graphic height; fonts remain >=8pt.',
        'tradeoffs':{'1':'Ablation draft: retains all current node means, but these duplicate Table1 and leave the trajectory panel shorter.','2':'Recommended compact appendix ablation: removes repeated node errors and retains every curve/component effect. Not the main one-method presentation requested by the latest steering.','3':'Ablation draft: repeats the main table and loses temporal shape.','4':'Main alternative: familiar paired estimate rows, but the long comparator gutter crowds the side-by-side layout.','5':'Main recommendation: one method, two endpoints and three paired controls; consistent zero-based scales and error intervals, with other ablations explicitly retained in table/appendix.'},
        'scope':'Presentation only. Original-validation development; fixed full [0,0.25] trajectory scale, forecast step = five native commands. No data, model, metric or result selection.',
        'remaining_gate':'Root recommendation/selection and independent actual-width review before replacing included spatial_main.pdf.'}
    b.write('compact_candidates.json',record)
    caption=('Native spatial prediction on original DROID validation. (a) Mean endpoint error at each of ten forecast steps; each consumes five native commands. '
             '(b) Arrows compare connected mechanisms at h10, showing error reductions and 95% session/seed bootstrap intervals. '
             'Mixing helps in both rows; incremental bounding and all interaction intervals include zero. '
             'Exact endpoint scores remain in Table 1; all 36 contrasts remain in the appendix. '
             'The follow-up uses revealed controls. Ours-1 uses a different protocol.')
    (OUT/'spatial_compact_candidate_2_caption.txt').write_text(caption+'\n')
    proof=OUT/'spatial_compact_proof';proof.mkdir(exist_ok=True)
    main_caption=('ShiftWM (ours) on original DROID validation. (a) Mean endpoint feature error; each forecast step consumes five native commands. '
                  '(b) Paired error reductions against autoregression, anchored additive prediction and the action-free ablation at steps 5 and 10. '
                  'Error bars are 95% session/seed bootstrap intervals; connecting lines guide the eye. '
                  'All 21 trained models completed 30 epochs. Other ablations remain in Table 1 and the appendix, including inconclusive context/bounding effects and unfavorable outcomes. '
                  'The complete 36-contrast record remains unchanged; this is development evidence.')
    for i in (4,5):(OUT/f'spatial_compact_candidate_{i}_caption.txt').write_text(main_caption+'\n')
    captions=[
        'Candidate 1: 5.5 by 2.65 inches. Retains all current data marks and four node errors; compresses whitespace and uses a two-column, four-row legend.',
        'Candidate 2 (recommended): 5.5 by 2.60 inches. '+caption,
        'Candidate 3: 5.5 by 2.60 inches. Exact eight-row endpoint lookup plus component map; all forecasts would move to the appendix. This duplicates the separate main table, so it is not recommended when that table remains.',
        'Candidate 4: 5.5 by 2.60 inches. '+main_caption,
        'Candidate 5 (recommended main figure): 5.5 by 2.60 inches. '+main_caption]
    tex='\\documentclass{article}\n\\usepackage{iclr2027_conference,times,amsmath,graphicx,hyperref}\n\\iclrfinalcopy\n\\begin{document}\n'
    for i,c in enumerate(captions,1):
        tex+='\\begin{figure}[p]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/spatial_compact_candidate_'+str(i)+'.pdf}\n\\caption{'+c.replace('%',r'\%')+'}\n\\end{figure}\n\\clearpage\n'
    (proof/'proof.tex').write_text(tex+'\\end{document}\n')
    print(json.dumps({'status':record['status'],'candidates':[{'id':r['id'],'height':r['dimensions_inches'][1],'issues':r['layout']['issues']} for r in records]}))


if __name__=='__main__':main()
