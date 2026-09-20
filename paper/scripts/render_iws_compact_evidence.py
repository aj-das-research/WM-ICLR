#!/usr/bin/env python3
"""Display completed IWS curves and paired contrasts from a portable result pack.

No models, dataset payloads, checkpoint selection or new bootstrap is performed.
The original and follow-up comparisons retain their distinct study identities.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / 'paper/figure_sources/iws_compact_evidence'
OUT = ROOT / 'paper/generated/iws_compact_evidence'
TASKS = ('pusht', 'bimanual_box', 'bimanual_rope')
NAMES = ('PushT', 'Box', 'Rope')
MODES = ('persistence', 'autoregressive', 'anchored_additive', 'bounded_spatial_mix', 'unbounded_spatial_mix')
COLORS = ('#956B26', '#536273', '#2458C3', '#166534', '#984A87')
STYLES = (':', '--', '-.', '-', (0, (5, 1.8, 1.3, 1.8)))
MARKS = ('o', 's', 'D', '^', 'v')
LABELS = ('Persistence', 'Autoregression', 'Additive anchor', 'ShiftWM (ours)', 'No tanh (ours, ablation)')
CONTRASTS = ('bounded_vs_additive', 'bounded_vs_ar', 'unbounded_vs_bounded', 'unbounded_vs_ar')
CONTRAST_LABELS = (r'ShiftWM vs additive', r'ShiftWM vs AR', r'No $\tanh$ vs ShiftWM', r'No $\tanh$ vs AR')
PAIRS = dict(zip(CONTRASTS, (
    ('bounded_spatial_mix','anchored_additive'), ('bounded_spatial_mix','autoregressive'),
    ('unbounded_spatial_mix','bounded_spatial_mix'), ('unbounded_spatial_mix','autoregressive'))))
CAPTION = (r'\textbf{Single-observation transfer: forecasting and the effect of the correction bound.} '
    r'Top: all five predictors and all 59 future offsets on recorded IWS tasks; '
    r'$H$ counts native command rows and predicts stored offset $H-1$. '
    r'Bottom: $H=60$ MSE reduction versus the same autoregressive baseline; '
    r'right is better and bars are paired 95\% intervals. '
    r'The bounded model has no consistent advantage here; removing only $\tanh$ '
    r'improves all three endpoint means. Means weight trajectories and three seeds equally. '
    r'The original 27-model study and nine-model follow-up are development comparisons; '
    r'the latter is exploratory. Table~\ref{tab:iws-development-complete} gives every plotted '
    r'contrast and the original primary comparison; Table~\ref{tab:current-iws-scorecard} '
    r'retains all four metrics. Reserved evaluation is excluded.')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(data):
    if (data.get('schema') != 'iws_compact_evidence_v1' or data.get('completed_runs') != 36
            or data.get('official_validation_payloads_read') != 0
            or set(data.get('tasks', {})) != set(TASKS)
            or data.get('offsets') != list(range(2, 61))):
        raise ValueError('Complete original and follow-up result identities required')
    for task in TASKS:
        row = data['tasks'][task]
        if set(row['curves']) != set(MODES) or set(row['effects']) != set(CONTRASTS):
            raise ValueError('Incomplete comparator set')
        for values in row['curves'].values():
            a = np.asarray(values)
            if a.shape != (59,) or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError('Invalid complete mean curve')
        for key, effect in row['effects'].items():
            if (effect['method'],effect['reference']) != PAIRS[key]:
                raise ValueError('Comparator identity does not match the contrast label')
            expected = 'original27' if key.startswith('bounded_') else 'followup9'
            if effect['study'] != expected:
                raise ValueError('Original/follow-up identity changed')
            method = row['curves'][effect['method']][-1]
            reference = row['curves'][effect['reference']][-1]
            if not np.isclose(effect['difference'], method-reference, rtol=0, atol=1e-12):
                raise ValueError('Difference does not match endpoint means')
            if not np.isclose(effect['gain'], 100*(reference-method)/reference, rtol=0, atol=1e-10):
                raise ValueError('Gain does not match endpoint means')
            for kind in ('difference', 'gain'):
                interval = effect['ci'][kind]
                if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]:
                    raise ValueError('Invalid paired interval')
    return data

def load(pack=PACK):
    manifest = json.loads((pack/'manifest.json').read_text())
    if sha(pack/'data.json') != manifest['data_sha256']:
        raise ValueError('Changed portable evidence')
    return validate(json.loads((pack/'data.json').read_text()))

def style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color('#81909C')
        axis.spines[side].set_linewidth(.55)
    axis.tick_params(labelsize=8, length=2, pad=2, colors='#536273')

def make_figure(data):
    plt.rcParams.update({'font.family':'Liberation Sans', 'font.size':8,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'svg.hashsalt':'iws-compact-evidence-v1',
        'text.color':'#243447', 'axes.labelcolor':'#243447'})
    fig = plt.figure(figsize=(5.5, 2.65), facecolor='white')
    handles = [Line2D([], [], color=c, ls=s, marker=m, ms=3, lw=1.1)
               for c,s,m in zip(COLORS, STYLES, MARKS)]
    fig.legend(handles[:3], LABELS[:3], loc='upper center', bbox_to_anchor=(.54,1.015),
               ncol=3, frameon=False, handlelength=1.9, columnspacing=1.1, handletextpad=.35)
    fig.legend(handles[3:], LABELS[3:], loc='upper center', bbox_to_anchor=(.54,.925),
               ncol=2, frameon=False, handlelength=1.9, columnspacing=1.2, handletextpad=.35)
    curve_axes, gain_axes = [], []
    for i, (task, name) in enumerate(zip(TASKS, NAMES)):
        x = .107 + .304*i
        axis = fig.add_axes([x,.415,.246,.34]); curve_axes.append(axis)
        row = data['tasks'][task]
        for j,mode in enumerate(MODES):
            axis.plot(data['offsets'],row['curves'][mode],color=COLORS[j],ls=STYLES[j],
                      marker=MARKS[j],markevery=[0,14+j,29+j,44+j,58],ms=2.4,lw=1.1)
        axis.set(xlim=(1,61),ylim=(0,.68),xticks=[2,30,60],yticks=[0,.3,.6])
        if max(max(values) for values in row['curves'].values()) >= .68:
            raise ValueError('Curve would exceed the displayed range')
        axis.set_title(f'{chr(97+i)}  {name}',loc='left',fontsize=8.5,pad=5)
        axis.grid(axis='y',color='#E3E8EC',lw=.5); style(axis)
        effect_axis = fig.add_axes([x,.113,.246,.115]); gain_axes.append(effect_axis)
        for j,key in enumerate(('bounded_vs_ar','unbounded_vs_ar')):
            e = row['effects'][key]; lo,hi = e['ci']['gain']; val=e['gain']
            if lo <= -7 or hi >= 9: raise ValueError('Paired interval would be clipped')
            effect_axis.errorbar(val,1-j,xerr=[[val-lo],[hi-val]],fmt=MARKS[3+j],
                color=COLORS[3+j],mfc='white' if lo<=0<=hi else COLORS[3+j],
                ms=3.4,mew=.85,lw=1,capsize=2)
        effect_axis.axvline(0,color='#81909C',ls=(0,(2,2)),lw=.65,zorder=0)
        effect_axis.set(xlim=(-7,9),ylim=(-.55,1.55),xticks=[-5,0,5],yticks=[])
        style(effect_axis); effect_axis.spines['left'].set_visible(False)
    fig.text(.025,.58,'Standardized MSE',rotation=90,ha='center',va='center',fontsize=8)
    fig.text(.54,.318,'Command rows H (target stored offset H − 1)',ha='center',fontsize=8)
    fig.text(.54,.261,'Endpoint comparison · same autoregressive reference',ha='center',fontsize=8)
    fig.text(.099,.198,'Bounded',ha='right',va='center',fontsize=8)
    fig.text(.099,.143,'No tanh',ha='right',va='center',fontsize=8)
    fig.text(.54,.017,'H60 MSE reduction (%)  →  better than autoregression',ha='center',fontsize=8)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); issues=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text(): continue
        box=t.get_window_extent(renderer)
        if t.get_fontsize()<8: issues.append('small text: '+t.get_text())
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:
            issues.append('clipped text: '+t.get_text())
    for i,axis in enumerate(curve_axes):
        for line, mode in zip(axis.lines,MODES):
            np.testing.assert_array_equal(line.get_ydata(), data['tasks'][TASKS[i]]['curves'][mode])
    if issues: raise ValueError(issues)
    return fig, {'size_inches':[5.5,2.65], 'minimum_font_pt':8, 'curve_points':885,
                 'paired_intervals':6, 'scope':'same numerical range; task-specific training scales', 'issues':issues}

def contrast_table(data):
    rows=[]
    for task,name in zip(TASKS,NAMES):
        for i,(key,label) in enumerate(zip(CONTRASTS,CONTRAST_LABELS)):
            e=data['tasks'][task]['effects'][key]
            gain=f"{e['gain']:+.2f}"
            if e['gain']>0: gain=r'\positivegain{'+gain+'}'
            ci=e['ci']; diff=f"{e['difference']:+.5f} [{ci['difference'][0]:+.5f}, {ci['difference'][1]:+.5f}]"
            change=gain+f" [{ci['gain'][0]:+.2f}, {ci['gain'][1]:+.2f}]"
            rows.append(' & '.join([name if i==0 else '',label,diff,change])+r' \\')
        rows.append(r'\addlinespace[3pt]')
    macro=data['macro_followup_gain']
    return (r'\begin{table}[!htb]\centering\small'+'\n'+
        r'\caption{\textbf{IWS paired endpoint contrasts at $H=60$ (stored offset 59).} Each task retains both original-study contrasts (ShiftWM versus additive anchoring and autoregression) and the exploratory no-$\tanh$ follow-up. Difference is method minus reference MSE, so negative is better; relative reduction reverses the sign, so positive is better. Brackets are paired 95\% seed--trajectory intervals (10,000 draws, unadjusted). Absolute scores and every secondary metric remain in Table~\ref{tab:current-iws-scorecard}.}'+'\n'+
        r'\label{tab:iws-development-complete}\label{tab:iws-unbounded-component}\label{tab:iws-paired-compact}'+'\n'+
        r'\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llll@{}}\toprule'+'\n'+
        r'Task & Method vs reference & MSE difference [95\% CI] & Reduction (\%) [95\% CI] \\ \midrule'+'\n'+
        '\n'.join(rows)+ '\n'+r'\bottomrule\end{tabular}'+'\n'+
        r'\par\smallskip Equal-task macro reductions (\%; paired 95\% CI): original ShiftWM vs additive, '+f"{data['macro_original_gain']:+.2f} [{data['macro_original_ci'][0]:+.2f}, {data['macro_original_ci'][1]:+.2f}]"+
        r'; follow-up no $\tanh$ vs ShiftWM, '+f"{macro['equal_task_relative_error_reduction_percent']:+.2f} [{macro['paired95'][0]:+.2f}, {macro['paired95'][1]:+.2f}]."+'\n'+r'\end{table}'+'\n')

def render(output=OUT, pack=PACK):
    data=load(pack); output.mkdir(parents=True,exist_ok=True)
    fig,geometry=make_figure(data)
    prefix='forecast_and_gain'
    for ext in ('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(output/(prefix+'.'+ext),dpi=300,metadata=metadata)
    plt.close(fig)
    figure=(r'\begin{figure}[!htb]\centering'+'\n'+r'\includegraphics[width=\linewidth]{generated/iws_compact_evidence/forecast_and_gain.pdf}'+'\n'+
            r'\caption[Single-observation forecasts and paired gains.]{'+CAPTION+'}\n'+r'\label{fig:iws-forecast-transfer}'+'\n'+r'\end{figure}'+'\n')
    (output/'figure.tex').write_text(figure)
    (output/'contrasts.tex').write_text(contrast_table(data))
    names=[prefix+'.'+ext for ext in ('pdf','svg','png')]+['figure.tex','contrasts.tex']
    receipt={'status':'passed_source_and_geometry_checks','geometry':geometry,
             'source_pack_sha256':sha(pack/'data.json'),'renderer_sha256':sha(__file__),
             'outputs_sha256':{p:sha(output/p) for p in names},'new_model_predictions':False,
             'original_and_followup_studies_preserved':True,'all_12_contrasts_retained':True}
    (output/'evidence.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--pack-dir',type=Path,default=PACK)
    args=parser.parse_args()
    r=render(args.output_dir,args.pack_dir)
    print(json.dumps({'status':r['status'],'curve_points':r['geometry']['curve_points'],'contrasts':12}))
