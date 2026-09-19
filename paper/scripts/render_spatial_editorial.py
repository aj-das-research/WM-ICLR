#!/usr/bin/env python3
"""Compact, source-bound main-paper compositions; no experiment or inference.

The complete 36-contrast appendix and its existing renderer remain untouched.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/generated/editorial'
SOURCE = ROOT / 'paper/generated/real_video/spatial_versions_evidence.json'
SOURCE_SHA = 'f64ef1ee15bc7e01c26caf6febd1e1c3a1c2352a9f0b283364363b05288774be'
REVIEW = ROOT / 'reports/evidence/spatial_versions_completed_component_review.json'
REVIEW_SHA = '5143777679ccb8930268289569baf48ba7d7b24b83c5c105fd360a376df5427a'
STYLE = ROOT / 'paper/design/editorial_style.json'
STYLE_SHA = '20cdf2ef920ddfc80d337c7bb061a6f00e995134400bfcf3f680737b3604e7af'
SKILL = Path(os.environ.get('PAPER_FIGURE_SKILL_DIR', str(Path.home()/'.codex/skills/paper-figure-creation'))).expanduser()
INK='#243447'; SECONDARY='#536273'; GRID='#E3E8EC'; BLUE='#0072B2'; GREEN='#166534'; WARM='#B75B16'
MODES=('autoregressive','persistence','anchored_additive','bounded_additive','unbounded_transport','transport','context_off','action_free')
DISPLAY=dict(zip(MODES,('Autoregressive','Persistence','Ours-2','Ours-3','Ours-4','Ours-5','Ours-5 − context','Ours-5 − actions')))
COLORS=dict(zip(MODES,(INK,'#7B8289',BLUE,BLUE,GREEN,GREEN,'#806092',WARM)))
MARKERS=dict(zip(MODES,('o','s','^','v','h','D','x','+')))
LINES=dict(zip(MODES,((0,(5,2)),(0,(1,2)),'--','-',(0,(5,2)),'-',(0,(2,1,1,1)),(0,(1,1)))))
MAIN_MODES=('transport','autoregressive','persistence','anchored_additive','action_free')
MAIN_LABELS={'transport':'ShiftWM (ours)','autoregressive':'Autoregressive','persistence':'Persistence',
             'anchored_additive':'Anchored additive','action_free':'Without actions'}
MAIN_CONTROLS=('autoregressive','anchored_additive','action_free')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name, value):
    (OUT/('spatial_'+name)).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def load_evidence():
    for path,digest in ((SOURCE,SOURCE_SHA),(REVIEW,REVIEW_SHA),(STYLE,STYLE_SHA)):
        if sha(path)!=digest:raise ValueError('Pinned source changed: '+str(path))
    d=json.loads(SOURCE.read_text());review=json.loads(REVIEW.read_text())
    if d['completed_runs']!=21 or d['epochs_each']!=30 or d['pending_results']:
        raise ValueError('Expected complete 21-model campaign')
    if d['population']!={'episodes':141,'sessions':59,'windows':1631}:
        raise ValueError('Population changed')
    if len(d['all_16_effects'])!=16 or len(d['all_20_component_effects'])!=20:
        raise ValueError('Incomplete comparison ledger')
    for p,digest in d['source_sha256'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Scientific source changed: '+p)
    old=json.loads((ROOT/'reports/real_video_spatial/finalization.json').read_text())
    component=json.loads((ROOT/'reports/real_video_spatial_components/finalization.json').read_text())
    values={}
    for mode in MODES:
        owner='autoregressive' if mode=='persistence' else mode
        metric='native_persistence_mse' if mode=='persistence' else 'native_mse'
        values[mode]=np.asarray(d['absolute_error'][owner][metric],dtype=float)
        origin=component if mode in ('bounded_additive','unbounded_transport') else old
        np.testing.assert_allclose(values[mode],origin['aggregate'][owner][metric],rtol=0,atol=1e-14)
        if values[mode].shape!=(10,) or not np.isfinite(values[mode]).all() or (values[mode]<0).any():
            raise ValueError('Invalid error trajectory')
    for e in d['all_16_effects']+d['all_20_component_effects']:
        if 'method' not in e:continue
        gain=e['comparator_mean']-e['method_mean']
        np.testing.assert_allclose([e['mse_reduction_x1000'],e['relative_error_reduction_percent']],
                                  [1000*gain,100*gain/e['comparator_mean']],rtol=0,atol=1e-11)
        np.testing.assert_allclose(e['reduction_ci95_x1000'],[-1000*e['paired_95_percent_interval'][1],-1000*e['paired_95_percent_interval'][0]],rtol=0,atol=1e-12)
    return d,values


def component(d,name,h=10):
    return next(e for e in d['all_20_component_effects'] if e['contrast']==name and e['metric']=='native_mse' and e['horizon']==h)


def text(fig,x,y,s,**kwargs):
    return fig.text(x,y,s,fontsize=kwargs.pop('fontsize',8),color=kwargs.pop('color',INK),**kwargs)


def theme():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':8,
        'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white',
        'axes.edgecolor':SECONDARY,'axes.linewidth':.65,'text.color':INK,'axes.labelcolor':INK,
        'xtick.color':SECONDARY,'ytick.color':SECONDARY,'mathtext.fontset':'dejavusans',
        'svg.hashsalt':'shiftwm-spatial-editorial-v1'})


def axis_style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(SECONDARY)
    ax.tick_params(length=2.5,pad=3)
    ax.grid(axis='y',color=GRID,lw=.55,zorder=0)


def error_curves(ax,values,selected=MODES,labels=True):
    for i,mode in enumerate(selected):
        # Offsetting marker schedule reveals coincident series without moving data.
        markers=[1,5,9] if mode in ('autoregressive','anchored_additive','unbounded_transport','context_off') else [0,4,8]
        ax.plot(np.arange(1,11),values[mode],color=COLORS[mode],ls=LINES[mode],lw=1.15,
                marker=MARKERS[mode],ms=3.1,mew=.7,markevery=markers,label=DISPLAY[mode],zorder=3)
    ax.set(xlim=(.7,10.3),ylim=(0,.25),xticks=[1,5,10],yticks=[0,.1,.2])
    if labels:ax.set(xlabel='Forecast step',ylabel='Native feature MSE')
    axis_style(ax)


def main_effect(d,mode,h):
    return next(e for e in d['all_16_effects']
                if e['comparator']==mode and e['metric']=='native_mse' and e['horizon']==h)


def draft_main(values,d):
    """Approved compact candidate 5, drawn directly by the canonical renderer."""
    fig=plt.figure(figsize=(5.5,2.60),dpi=200)
    text(fig,.03,.944,'(a) Forecast error',fontsize=9.5,weight='bold')
    ax=fig.add_axes([.103,.385,.330,.445])
    error_curves(ax,values,selected=MAIN_MODES[1:]+MAIN_MODES[:1])
    handles=[Line2D([0],[0],color=COLORS[m],ls=LINES[m],marker=MARKERS[m],ms=3,lw=1.1)
             for m in MAIN_MODES]
    fig.legend(handles,[MAIN_LABELS[m] for m in MAIN_MODES],loc='lower left',bbox_to_anchor=(.020,.025),
               ncol=2,frameon=False,fontsize=8,handlelength=1.50,columnspacing=.8,
               handletextpad=.4,labelspacing=.40)
    text(fig,.505,.944,'(b) ShiftWM gains vs controls',fontsize=9.5,weight='bold')
    for i,(mode,label) in enumerate(zip(MAIN_CONTROLS,('Autoreg.','Anchor','No actions'))):
        ax=fig.add_axes([.586+i*.139,.345,.112,.458])
        effects=[main_effect(d,mode,h) for h in (5,10)]
        ax.plot([5,10],[e['mse_reduction_x1000'] for e in effects],color=GREEN,lw=1.05,zorder=2)
        for h,e,marker in zip((5,10),effects,('o','D')):
            point=e['mse_reduction_x1000'];lo,hi=e['reduction_ci95_x1000']
            ax.errorbar(h,point,yerr=[[point-lo],[hi-point]],fmt=marker,ms=3.4,color=GREEN,
                        lw=1.1,capsize=2,zorder=3)
        ax.set(xlim=(3.5,11.5),ylim=(0,14),xticks=[5,10],yticks=[0,5,10])
        axis_style(ax);ax.set_title(label,fontsize=8,pad=5)
        if i==0:ax.set_ylabel('MSE reduction ×10⁻³',fontsize=8.5,labelpad=4)
        else:ax.tick_params(axis='y',labelleft=False,length=0);ax.spines['left'].set_visible(False)
    text(fig,.782,.208,'Forecast step',ha='center',fontsize=8.5)
    text(fig,.782,.110,'Paired 95% intervals',ha='center',color=SECONDARY)
    return fig


def arrow(fig,start,end,color=SECONDARY):
    # Continuous comparison arrows; endpoints are separated from text and nodes.
    p=FancyArrowPatch(start,end,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=8,
                      lw=.9,color=color,shrinkA=0,shrinkB=0,zorder=1)
    fig.add_artist(p)


def node(fig,x,y,mode,values):
    color=COLORS[mode]
    text(fig,x,y,DISPLAY[mode],ha='center',weight='bold',fontsize=8.5,color=color)
    text(fig,x,y-.057,f'{1000*values[mode][9]:.2f}',ha='center',fontsize=9.5,color=INK)


def edge_label(fig,x,y,e):
    gain=e['mse_reduction_x1000'];lo,hi=e['reduction_ci95_x1000'];color=GREEN if lo>0 else SECONDARY
    text(fig,x,y,f'{gain:+.2f}',ha='center',weight='bold',fontsize=8.5,color=color)
    text(fig,x,y-.043,f'[{lo:+.2f}, {hi:+.2f}]',ha='center',fontsize=8,color=color)


def factorial_map(fig,values,d):
    text(fig,.53,.956,'(b) Mixing × bounding',fontsize=9.5,weight='bold')
    text(fig,.62,.875,'Anchor',ha='center',fontsize=8.5,color=BLUE)
    text(fig,.90,.875,'Gated mixing',ha='center',fontsize=8.5,color=GREEN)
    text(fig,.523,.743,'Raw δ',rotation=90,ha='center',va='center',color=SECONDARY)
    text(fig,.523,.294,'tanh δ',rotation=90,ha='center',va='center',color=SECONDARY)
    for mode,x,y in [('anchored_additive',.62,.796),('unbounded_transport',.90,.796),
                      ('bounded_additive',.62,.347),('transport',.90,.347)]:node(fig,x,y,mode,values)
    # Horizontal edges add the mixing package; vertical edges bound innovation.
    arrow(fig,(.700,.742),(.816,.742),GREEN)
    arrow(fig,(.700,.292),(.816,.292),GREEN)
    edge_label(fig,.758,.672,component(d,'mixing_without_bounding'))
    edge_label(fig,.758,.223,component(d,'mixing_with_bounding'))
    arrow(fig,(.554,.683),(.554,.380))
    arrow(fig,(.974,.683),(.974,.380))
    edge_label(fig,.630,.502,component(d,'bounding_without_mixing'))
    edge_label(fig,.885,.502,component(d,'bounding_with_mixing'))
    text(fig,.756,.105,'h10 error and reductions ×10⁻³',ha='center',color=SECONDARY)
    text(fig,.756,.048,'Brackets: paired 95% CI',ha='center',color=SECONDARY)


def draft_one(values,d):
    fig=plt.figure(figsize=(5.5,3.2),dpi=200)
    text(fig,.03,.956,'(a) Ten-step forecasts',fontsize=9.5,weight='bold')
    ax=fig.add_axes([.103,.405,.365,.44]);error_curves(ax,values)
    handles=[Line2D([0],[0],color=COLORS[m],ls=LINES[m],marker=MARKERS[m],ms=3,lw=1.1) for m in MODES]
    fig.legend(handles,[DISPLAY[m] for m in MODES],loc='lower left',bbox_to_anchor=(.020,.030),
               ncol=2,frameon=False,fontsize=8,handlelength=1.6,columnspacing=.75,handletextpad=.4,labelspacing=.48)
    factorial_map(fig,values,d)
    return fig


def draft_two(values,d):
    # Genuine alternative: four equal-scale trajectories grouped by factorial cell.
    fig=plt.figure(figsize=(5.5,3.2),dpi=200)
    text(fig,.03,.96,'One mechanism per facet; identical error scale',fontsize=9.5,weight='bold')
    cells=[('anchored_additive',.105,.59,'Anchor · raw'),('unbounded_transport',.575,.59,'Mixing · raw'),
           ('bounded_additive',.105,.18,'Anchor · bounded'),('transport',.575,.18,'Mixing · bounded')]
    for mode,x,y,label in cells:
        ax=fig.add_axes([x,y,.365,.25]);selected=['autoregressive','persistence',mode]
        if mode=='transport':selected+=['context_off','action_free']
        error_curves(ax,values,selected,labels=False)
        ax.set_title(f'{DISPLAY[mode]}: {label}',fontsize=8.5,loc='left',pad=4,color=COLORS[mode])
        if x<.2:ax.set_ylabel('MSE')
        if y<.3:ax.set_xlabel('Forecast step')
    text(fig,.05,.023,'Gray: AR / persistence. Bottom-right: context-off and action-free controls.',fontsize=8,color=SECONDARY)
    return fig


def draft_three(values,d):
    # Genuine alternative: precise endpoint matrix plus four aligned effect strips.
    fig=plt.figure(figsize=(5.5,3.2),dpi=200)
    text(fig,.03,.956,'(a) Endpoint errors',fontsize=9.5,weight='bold')
    ax=fig.add_axes([.235,.145,.25,.69])
    a=np.array([[values[m][4],values[m][9]] for m in MODES])
    ax.imshow(a,cmap='Greys',vmin=0,vmax=.25,aspect='auto',alpha=.22)
    ax.set(xticks=[0,1],xticklabels=['h5','h10'],yticks=np.arange(8),yticklabels=[DISPLAY[m] for m in MODES])
    ax.tick_params(length=0);ax.spines[:].set_visible(False)
    for i in range(8):
        for j in range(2):ax.text(j,i,f'{a[i,j]:.4f}',ha='center',va='center',fontsize=8,color=COLORS[MODES[i]])
    text(fig,.56,.956,'(b) Component effects · h10',fontsize=9.5,weight='bold')
    names=['mixing_without_bounding','mixing_with_bounding','bounding_without_mixing','bounding_with_mixing']
    labels=['Mixing / raw','Mixing / bounded','Bounding / anchor','Bounding / mixing']
    for i,(name,label) in enumerate(zip(names,labels)):
        ax=fig.add_axes([.61,.745-i*.18,.34,.11]);e=component(d,name);lo,hi=e['reduction_ci95_x1000'];x=e['mse_reduction_x1000']
        ax.axvline(0,color=SECONDARY,lw=.7);ax.errorbar(x,0,xerr=[[x-lo],[hi-x]],fmt='o',ms=3,color=GREEN if lo>0 else SECONDARY,capsize=2)
        ax.set(xlim=(-1,10),ylim=(-.5,.5),yticks=[],xticks=[0,5,10]);axis_style(ax)
        ax.set_title(label,loc='left',fontsize=8,pad=1)
    text(fig,.75,.035,'MSE reduction ×10⁻³; 95% CI',ha='center',color=SECONDARY)
    return fig


def audit(fig):
    path=SKILL/'scripts/layout_quality.py'
    if not path.exists():return {'status':'not_run','reason':'Optional paper-figure-creation skill unavailable','issues':None}
    spec=importlib.util.spec_from_file_location('spatial_editorial_layout',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    issues=module.audit_figure(fig,min_font_pt=8,display_width_inches=5.5)
    return {'status':'checked','issues':issues,'display_width_inches':5.5,'min_effective_font_pt':8,'skill_audit_sha256':sha(path)}


def save(fig,name):
    for extension in ('pdf','svg','png'):
        kwargs={'dpi':300} if extension=='png' else {}
        if extension=='pdf':kwargs['metadata']={'CreationDate':None,'ModDate':None}
        if extension=='svg':kwargs['metadata']={'Date':None}
        fig.savefig(OUT/('spatial_'+name+'.'+extension),**kwargs)


def main():
    OUT.mkdir(parents=True,exist_ok=True);theme();d,values=load_evidence()
    # Candidate 5 is the approved composition. Historical alternatives remain
    # separate artifacts; rerendering the main never overwrites that review trail.
    first=draft_main(values,d);qa=audit(first);save(first,'main');plt.close(first)
    caption=('ShiftWM (ours) on original DROID validation. (a) Mean endpoint feature error; each forecast step consumes five native commands. '
             '(b) Paired error reductions against autoregression, anchored additive prediction and the action-free ablation at steps 5 and 10. '
             'Error bars are 95% session/seed bootstrap intervals; connecting lines guide the eye. All 21 trained models completed 30 epochs. '
             'Other ablations remain in Table 1 and the appendix, including inconclusive context/bounding effects and unfavorable outcomes. '
             'The complete 36-contrast record remains unchanged; this is development evidence.')
    (OUT/'spatial_caption.txt').write_text(caption+'\n')
    texcaption=caption.replace('%',r'\%').replace('Table 1',r'Table~\ref{tab:editorial-spatial}')
    (OUT/'spatial_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/spatial_main.pdf}\n\\caption[ShiftWM forecasting and paired control comparisons.]{'+texcaption+'}\n\\label{fig:editorial-spatial}\n\\end{figure}\n')
    write('layout_audit.json',qa)
    write('evidence.json',{'status':'measured_source_verified','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{**d['source_sha256'],str(SOURCE.relative_to(ROOT)):SOURCE_SHA,str(REVIEW.relative_to(ROOT)):REVIEW_SHA,str(STYLE.relative_to(ROOT)):STYLE_SHA},
        'renderer_sha256':sha(__file__),'native_mse_by_mode':{m:v.tolist() for m,v in values.items()},
        'main_proposed_method':'transport = ShiftWM (ours)',
        'main_curve_modes':MAIN_MODES,'main_displayed_means':50,
        'main_paired_effects':[main_effect(d,m,h) for m in MAIN_CONTROLS for h in (5,10)],
        'all_16_original_effects':d['all_16_effects'],'all_20_component_effects':d['all_20_component_effects'],
        'population':d['population'],'aggregation':'mean of episode means, then equally over three seeds',
        'uncertainty':'paired session × seed bootstrap, 10000 draws, seed 173, unadjusted',
        'axis_and_geometry':'Left: full [0,0.25] MSE axis, all integer forecast steps; right: three common [0,14] error-reduction axes at steps 5 and 10.',
        'units':'Right panel: comparator-minus-ShiftWM MSE multiplied by 1000. Intervals are paired difference CIs, not mean-error CIs. Connecting lines guide the eye; no statistical claim about gain growth.',
        'temporal_contract':'One forecast step spans five native transitions and consumes their five 7D recorded commands concatenated into a 35D block; ten steps span fifty native transitions, not ten native commands or calibrated seconds.',
        'all_variants':'Five curves in this main figure; all eight comparable spatial rows remain in the main table and all 36 comparisons remain in the appendix. Original context-based ShiftWM has a different protocol.',
        'scope':'Original-validation development. Component follow-up registered after control results were revealed. No new prediction or test outcome.',
        'limitations':['Mixing includes gate, identity bias and approximately 1.84% extra active parameters.','Coarse original 2x2 outcomes remain in the full appendix; this compact main figure uses the registered native metric.','Unbounded transport has lower native h5 point error than ShiftWM; their h5/h10 paired CIs include zero.','Context, incremental native bounding and all four component interaction intervals include zero.']})
    write('design_brief.json',{'slot':'Main-paper matched spatial result; full comparisons retained unchanged in appendix',
        'dimensions_inches':[5.5,2.6],'min_font_pt':8,
        'one_sentence':'ShiftWM reduces forecast error against registered controls in original-validation development; null and unfavorable ablations remain explicit in the table and appendix.',
        'focal_relationship':'Read temporal error at left, then paired reductions at two forecast steps in aligned comparator facets.',
        'selected':5,'alternatives_ledger':'paper/generated/editorial/spatial_compact_candidates.json',
        'independent_candidate_review':'reports/evidence/spatial_compact_candidate5_independent_review.json',
        'selection_reason':'Retains forecasting behavior and uncertainty in a 2.6-inch-high figure while keeping one proposed algorithm and avoiding a tall ranked list. The earlier factorial map is an ablation/appendix composition.',
        'semantic_marks':{'curves':'Exact observed mean feature prediction errors. Connecting registered integer horizons is a guide, not a continuous-time claim.','interval_points':'Comparator-minus-ShiftWM endpoint error at forecast steps 5 and 10.','interval_bars':'Paired session/seed 95% intervals on a common zero-origin scale.','interval_connectors':'Guides to the eye, not significance tests of gain growth.'},
        'representation_budget':'Five curve labels, three short comparator headings, six error-reduction intervals; complete lookup and null/negative effects in main table and appendix.',
        'skill_references':['publication-polish.md','art-direction.md','visual-story.md','evidence.md','design-system.md'],
        'asset_choice':'Original vectors only; no generated evidence or decorative imagery.',
        'caption_words':len(caption.split()),'review_status':'Approved candidate 5 promoted; final canonical bytes require promotion receipt'})
    # Recheck pinned inputs after drawing; raw science is never mutated.
    for path,digest in ((SOURCE,SOURCE_SHA),(REVIEW,REVIEW_SHA),(STYLE,STYLE_SHA)):
        if sha(path)!=digest:raise ValueError('Input changed during render')
    print(json.dumps({'status':'rendered_for_inspection','layout':qa,'output':str(OUT/'spatial_main.pdf'),'caption_words':len(caption.split())}))


if __name__=='__main__':main()
