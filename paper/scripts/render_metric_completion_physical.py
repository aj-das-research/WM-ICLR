#!/usr/bin/env python3
"""Render saved historical physical outcomes; never runs a model or simulator."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
COLORS={'factorized':'#007F77','framewise':'#505866','constant_dynamics':'#B27B12',
        'single':'#337AB7','factorized_unpaired':'#96578E','plain':'#A75F45','frozen':'#111111',
        'random':'#666666','replay_oracle':'#398F34'}
LABELS={'factorized':'ShiftWM (ours)','framewise':'Framewise','constant_dynamics':'Constant dynamics',
        'single':'Shared context','factorized_unpaired':'Unpaired','plain':'Unaligned','frozen':'Frozen LeWM',
        'random':'Random reference','replay_oracle':'Replay oracle'}
TASKS=['pusht','reacher','drone','surgery']
NAMES={'pusht':'PushT','reacher':'Reacher','drone':'Drone','surgery':'Tissue'}
METRIC={'pusht':'position_error_px','reacher':'wrapped_joint_error_rad','drone':'goal_distance_mm','surgery':'goal_distance_mm'}
XLABEL={'pusht':'Position error (px)','reacher':'Joint error (rad)','drone':'Goal error (mm)','surgery':'Goal error (mm)'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+'\n')
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def fmt(x,n=2):return '--' if x is None else f'{x:.{n}f}'
def gain(x,n=2):
    text=f'{x:+.{n}f}'
    return r'\positivegain{'+text+'}' if x>0 else text

def selected(data,task,all_modes=False):
    return [r for r in data['runs'] if r['task']==task and
            ((r['family']=='core' and r['split']=='test') if task in ('pusht','reacher') else r['family']=='original')
            and (all_modes or r['mode'] in ('framewise','factorized'))]

def series(runs):
    out={}
    for r in runs:out.setdefault((r['mode'],r['backbone']),[]).extend(r['records'])
    return out

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#ABB1B8')
    ax.tick_params(length=2.2,width=.6,pad=2)
    ax.grid(axis='y',color='#E6E8EB',linewidth=.5,zorder=0)

def draw_error(ax, groups, task):
    for (mode,backbone), rows in groups.items():
        x=np.sort([r['errors'][METRIC[task]] for r in rows]); y=100*np.arange(1,len(x)+1)/len(x)
        ax.step(x,y,where='post',color=COLORS[mode],linestyle='--' if backbone=='gru' else '-',linewidth=1.2,marker='o' if mode=='factorized' else 's',markersize=2.4,markevery=max(1,len(x)//6))
    ax.set_ylim(0,103);ax.set_yticks([0,50,100]);ax.set_xlim(left=0)
    ax.set_xlabel(XLABEL[task],labelpad=2);ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    style(ax)

def draw_success(ax, groups, task):
    budget=50 if task in ('pusht','reacher') else 200
    for (mode,backbone),rows in groups.items():
        y=[100*sum(r['success'] and r['first_success']<=i for r in rows)/len(rows) for i in range(budget+1)]
        ax.step(range(budget+1),y,where='post',color=COLORS[mode],linestyle='--' if backbone=='gru' else '-',linewidth=1.2,marker='o' if mode=='factorized' else 's',markersize=2.4,markevery=max(1,budget//5))
    ax.axvspan(0,10,color='#EEF0F2',zorder=-1)
    ax.set_xlim(0,budget);ax.set_ylim(bottom=0);ax.set_xticks([0,budget//2,budget]);ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('Native commands',labelpad=2);style(ax)

def legend(fig,all_modes=False, y=.93):
    modes=['framewise','factorized'] if not all_modes else ['frozen','framewise','single','factorized_unpaired','factorized','plain','constant_dynamics']
    handles=[Line2D([],[],color=COLORS[m],lw=1.5,marker='o' if m=='factorized' else 's',markersize=3,label=LABELS[m]) for m in modes]
    if not all_modes: handles += [Line2D([],[],color='#555555',lw=1,ls='-',label='Transformer'),Line2D([],[],color='#555555',lw=1,ls='--',label='GRU')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,y),ncol=4 if not all_modes else 4,frameon=False,columnspacing=.9,handlelength=1.5,handletextpad=.4)

def render_a(data,path):
    fig,axs=plt.subplots(2,4,figsize=(5.5,3.3))
    fig.subplots_adjust(left=.095,right=.958,bottom=.12,top=.76,wspace=.38,hspace=.65)
    fig.text(.095,.975,'Physical error and success can disagree',fontsize=9,fontweight='bold',va='top')
    fig.text(.095,.923,'Historical models · core test / extension development',fontsize=8,va='top')
    legend(fig,y=.872)
    for col,task in enumerate(TASKS):
        groups=series(selected(data,task));draw_error(axs[0,col],groups,task);draw_success(axs[1,col],groups,task)
        axs[0,col].set_title(f'({chr(97+col)}) {NAMES[task]}',loc='left',fontsize=8,fontweight='bold',pad=4)
        if col==0:axs[0,col].set_ylabel('Trials ≤ error (%)',labelpad=2);axs[1,col].set_ylabel('Success by call (%)',labelpad=2)
        else:axs[0,col].set_yticklabels([])
    export(fig,path);return fig

def render_b(data,path):
    fig,axs=plt.subplots(2,4,figsize=(5.5,3.1));fig.subplots_adjust(left=.095,right=.958,bottom=.13,top=.76,wspace=.4,hspace=.65)
    fig.text(.095,.975,'Matched final-error differences and success',fontsize=9,fontweight='bold',va='top');legend(fig,y=.91)
    for col,task in enumerate(TASKS):
        rr=selected(data,task);group=series(rr)
        for backbone in ('transformer','gru'):
            left={(r['training_seed'],v['task_seed'],v['observation_id'],v['dynamics_id']):v for r in rr if r['mode']=='factorized' and r['backbone']==backbone for v in r['records']}
            right={(r['training_seed'],v['task_seed'],v['observation_id'],v['dynamics_id']):v for r in rr if r['mode']=='framewise' and r['backbone']==backbone for v in r['records']}
            if not left:continue
            delta=np.sort([right[k]['errors'][METRIC[task]]-left[k]['errors'][METRIC[task]] for k in left])
            axs[0,col].step(delta,100*np.arange(1,len(delta)+1)/len(delta),where='post',color=COLORS['factorized'],ls='--' if backbone=='gru' else '-')
        axs[0,col].axvline(0,color='#555555',lw=.6);axs[0,col].set_ylim(0,103);axs[0,col].set_yticks([0,50,100]);axs[0,col].set_title(NAMES[task],loc='left',fontsize=8,fontweight='bold');style(axs[0,col])
        axs[0,col].set_xlabel('Error gain '+(' (rad)' if task=='reacher' else '(px)' if task=='pusht' else '(mm)'),labelpad=2);axs[0,col].xaxis.set_major_locator(plt.MaxNLocator(3))
        draw_success(axs[1,col],group,task)
    axs[0,0].set_ylabel('Paired CDF (%)');axs[1,0].set_ylabel('Success by call (%)')
    export(fig,path);return fig

def render_c(data,path):
    fig,axs=plt.subplots(2,4,figsize=(5.5,3.5));fig.subplots_adjust(left=.095,right=.958,bottom=.11,top=.80,wspace=.40,hspace=.9)
    fig.text(.095,.98,'Each task: final errors beside executed success',fontsize=9,fontweight='bold',va='top');legend(fig,y=.93)
    for i,task in enumerate(TASKS):
        row,col=divmod(i,2);a,b=axs[row,2*col:2*col+2];group=series(selected(data,task));draw_error(a,group,task);draw_success(b,group,task)
        a.set_title(NAMES[task],loc='left',fontsize=8,fontweight='bold');a.set_ylabel('CDF (%)');b.set_ylabel('Success (%)')
    export(fig,path);return fig

def export(fig,path):
    path=Path(path)
    for suffix in ('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else {'Date':None} if suffix=='svg' else None
        fig.savefig(path.with_suffix('.'+suffix),dpi=240,metadata=metadata)

def table(caption,label,columns,header,rows):
    return '\n'.join([r'\begin{table}[tbp]',r'\centering\fontsize{9}{10.5}\selectfont',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule',*[' & '.join(r)+r' \\' for r in rows],r'\bottomrule\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table}'])+'\n'

def tables(data,out):
    ss=[s for s in data['summaries'] if s['condition']=='all'];by={(s['family'],s['task'],s['backbone'],s['split'],s['mode']):s for s in ss}
    snippets=[]
    for task in ['pusht','reacher']:
        rows=[]
        for split in ['test','extrapolation']:
            for mode in ['frozen','framewise','single','factorized_unpaired','factorized','plain']:
                s=by['core',task,'transformer',split,mode]
                err=s['errors'][METRIC[task]];rows.append([('Test' if split=='test' else 'Extra.'),tex(LABELS[mode]),str(len(s['seeds'])),f"{s['successes']}/{s['n']}",fmt(s['success_percent']),fmt(err['mean'],3 if task=='reacher' else 2),fmt(s['native_calls']['mean'])])
        unit='rad' if task=='reacher' else 'px'
        caption=NAMES[task]+r' historical-model outcomes. Test averages all nine in-range conditions; Extra. all three extrapolation conditions, on independent test initial states. Each condition has 64 physical seeds, repeated across three training seeds (one frozen reference). Error is '+('wrapped joint L2' if task=='reacher' else 'joint pusher-plus-block position L2')+r' at termination/budget; it is not the complete success criterion. Calls are actual attempted native commands, including paid support, maximum 50. Success is at any call. All conditions, other physical errors and distributions are in the source data.'
        caption += (' Paid support already solves 15/576 test and 5/192 extrapolation task conditions per checkpoint; policy-eligible rates exclude these and are retained in the source table.' if task=='pusht' else ' No task is solved during support acquisition.')
        text=table(caption,'tab:physical-'+task,'llrrrrr','Split & Method & Seeds & Hits / trials & SR (\%) & Error ('+unit+') & Calls',rows)
        (out/f'{task}_outcomes.tex').write_text(text);snippets.append(text)
    rows=[]
    for family,task,backbone in [('original',t,b) for t in ['drone','surgery'] for b in ['transformer','gru']]+[('wide_gain','drone','transformer')]:
        for mode in (['framewise','constant_dynamics','factorized'] if family=='original' else ['constant_dynamics','factorized']):
            s=by[family,task,backbone,'development',mode]
            name=NAMES[task]+(' T' if backbone=='transformer' else ' G')+(' wide' if family=='wide_gain' else '')
            rows.append([name,tex(LABELS[mode]),f"{s['successes']}/{s['n']}",fmt(s['success_percent']),fmt(s['errors']['goal_distance_mm']['mean']),fmt(s['native_calls']['mean']),str(s['failure_counts'].get('early_termination',0))])
    text=table(r'Historical simulator development outcomes. T/G denote separate Transformer/GRU predictors; wide is the drone observation-gain follow-up. Each row is the same eight physical tasks repeated across three training seeds, not 24 independent tasks. Final goal error is in millimetres (drone horizontal position; tissue target-point Euclidean distance). Calls are actual attempted native commands including ten support calls, with a 200-call maximum. Early counts unsuccessful terminal runs before budget. No clinical or current-spatial-decoder claim.','tab:physical-extensions','llrrrrr',r'Task & Method & Hits / trials & SR (\%) & Error (mm) & Calls & Early',rows)
    (out/'extension_outcomes.tex').write_text(text);snippets.append(text)
    rows=[]
    for c in data['contrasts']:
        if c['condition']!='all' or c['metric']!=METRIC[c['task']] or c['comparator']!=('constant_dynamics' if c['family']=='wide_gain' else 'framewise'):continue
        sr=next(e for e in data['contrasts'] if all(e[k]==c[k] for k in ['family','task','backbone','split','condition','comparator']) and e['metric']=='success')
        name=NAMES[c['task']]+' '+('test' if c['split']=='test' else 'extra.' if c['split']=='extrapolation' else ('T' if c['backbone']=='transformer' else 'G'))+(' wide' if c['family']=='wide_gain' else '')
        unit='rad' if c['task']=='reacher' else 'px' if c['task']=='pusht' else 'mm'
        rows.append([name,gain(sr['mean']),f"[{sr['ci95'][0]:+.2f}, {sr['ci95'][1]:+.2f}]",gain(c['mean'],3 if unit=='rad' else 2)+f' {unit}',f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" if unit=='rad' else f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"])
    text=table(r'Post-hoc paired physical-outcome contrasts (test: all nine conditions; extra.: all three; T/G: Transformer/GRU) against matched Framewise (wide drone: matched wide Constant dynamics). Success gain is ours minus comparator in percentage points; error gain is comparator minus ours in the named physical unit. Positive green point gains do not assert significance. Core intervals use 20,000 paired initial-state-cluster draws, conditional on the three fixed training seeds; extension intervals use 5,000 crossed training/task-seed draws. All intervals are exploratory and unadjusted; no units/backbones are pooled.','tab:physical-paired','lrrrr',r'Population & SR gain (pp) & 95\% interval & Error gain & 95\% interval',rows)
    (out/'paired_outcomes.tex').write_text(text);snippets.append(text)
    # Complete long-form tables retain every appearance/dynamics cell and error.
    rows=[]
    for s in data['summaries']:
        base={k:s[k] for k in ['family','task','backbone','split','mode','condition','n','physical_task_seeds','success_percent','final_success_percent','support_successes','eligible_n','eligible_success_percent']}
        for metric,values in s['errors'].items():rows.append({**base,'metric':metric,**values,'actual_calls_mean':s['native_calls']['mean'],'success_calls_mean':None if s['success_calls'] is None else s['success_calls']['mean'],'budget_penalized_commands_mean':s['budget_penalized_commands']['mean'],'failures':s['failures']})
    for filename,values in [('all_physical_metrics.csv',rows),('all_paired_effects.csv',data['contrasts'])]:
        buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values);(out/filename).write_text(buf.getvalue())
    (out/'all_tables.tex').write_text('\n'.join(snippets))

def atlas(data,out):
    # Complete distributions, one explicitly identified population per PDF page.
    with PdfPages(out/'complete_distributions.pdf',metadata={'CreationDate':None,'ModDate':None}) as pdf:
        populations=sorted({(r['family'],r['task'],r['backbone'],r['split']) for r in data['runs'] if r['family']!='reference'})
        for family,task,backbone,split in populations:
            rr=[r for r in data['runs'] if (r['family'],r['task'],r['backbone'],r['split'])==(family,task,backbone,split)]
            fig,axs=plt.subplots(1,2,figsize=(5.5,2.5));fig.subplots_adjust(left=.11,right=.95,bottom=.23,top=.70,wspace=.4)
            fig.text(.11,.96,f'{NAMES[task]} · {family} · {backbone} · {split}',fontsize=9,fontweight='bold',va='top')
            groups=series(rr);draw_error(axs[0],groups,task);draw_success(axs[1],groups,task)
            axs[0].set_ylabel('Trials ≤ error (%)');axs[1].set_ylabel('Success by call (%)')
            fig.legend(handles=[Line2D([],[],color=COLORS[m],label=LABELS[m]) for m,b in groups],loc='upper center',bbox_to_anchor=(.5,.89),ncol=3,frameon=False,columnspacing=.8,handlelength=1.2)
            pdf.savefig(fig);plt.close(fig)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,default=ROOT/'reports/metrics_completion_v1/physical/data.json');p.add_argument('--output',type=Path,default=ROOT/'paper/generated/metric_completion_physical');a=p.parse_args()
    data=read(a.source)
    if data['status']!='complete_saved_ledgers_only' or len(data['runs'])!=110:raise ValueError('Complete physical source required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'physical_metrics_v1'})
    a.output.mkdir(parents=True,exist_ok=True)
    figures=[]
    for name,fn in [('composition_A',render_a),('composition_B',render_b),('composition_C',render_c)]:
        fig=fn(data,a.output/name);figures.append(name);plt.close(fig)
    for suffix in ['pdf','svg','png']:shutil.copyfile(a.output/f'composition_A.{suffix}',a.output/f'physical_outcomes.{suffix}')
    tables(data,a.output);atlas(data,a.output)
    caption=r'Historical-model physical outcomes on all nine in-range test conditions (PushT/Reacher) and the eight development tasks (drone/tissue). Upper panels are empirical distributions of final physical error; lower panels give the fraction of all trials that have succeeded by each native command. The gray band marks the first ten paid support commands; successes there precede policy planning. Native commands include this support; unsuccessful trials, including early failures, remain in the denominator and contribute no successes. Solid/dashed lines denote separate Transformer/GRU backbones; all curves average the same three training seeds. PushT error combines pusher and block positions; angle is a separate success requirement. Reacher wrapped joint L2 is diagnostic, while success uses the original per-joint criterion. Task scales differ. Fixed Framewise comparisons are shown here; all methods, extrapolation conditions, units and paired intervals are retained in accompanying tables and the complete distribution supplement. These are historical architectures, not closed-loop results for the current spatial decoder.'
    (a.output/'caption.tex').write_text(caption+'\n')
    (a.output/'figure.tex').write_text('\\begin{figure}[tbp]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/metric_completion_physical/physical_outcomes.pdf}\n\\caption{'+caption+'}\n\\label{fig:physical-outcome-metrics}\n\\end{figure}\n')
    evidence=dict(status='rendered',source=str(a.source.relative_to(ROOT)) if a.source.is_relative_to(ROOT) else str(a.source),source_sha256=sha(a.source),renderer_sha256=sha(__file__),selected='A',width_inches=5.5,height_inches=3.3,minimum_font_pt=8,actual_pixel_review='pending',source_bindings=data['source_sha256'],files_sha256={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.suffix in ('.pdf','.svg','.png','.tex','.csv')})
    dump(a.output/'evidence.json',evidence)
    print(json.dumps({'status':'rendered','outputs':len(evidence['files_sha256']),'source_sha256':evidence['source_sha256']}))
if __name__=='__main__':main()
