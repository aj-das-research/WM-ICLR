#!/usr/bin/env python3
"""Render complete IWS development curves only after the frozen27-run gate.

Never changes registered scientific/reporting files. No missing-data figures or
synthetic fallback. --if-ready is a no-op before complete validated evidence.
Real PDF visual review remains explicitly pending after automatic rendering.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/experiment_alignment'
FINAL=ROOT/'reports/real_video_iws/development_finalization.json'
REPORTER=ROOT/'paper/scripts/refresh_experiment_alignment.py'
REPORTER_SHA='49013f987fdf9e5d8800b21e9ef04458e9c55811cd69ef386448d617fc989664'
TASKS=('pusht','bimanual_box','bimanual_rope')
NAMES={'pusht':'PushT','bimanual_box':'Box','bimanual_rope':'Rope'}
MODES=('autoregressive','anchored_additive','persistence','bounded_spatial_mix')
LABELS={'autoregressive':'Autoregressive','anchored_additive':'Additive anchor',
        'persistence':'Persistence','bounded_spatial_mix':'ShiftWM (ours)'}
COLORS={'autoregressive':'#536273','anchored_additive':'#2458C3','persistence':'#956B26','bounded_spatial_mix':'#166534'}
STYLES={'autoregressive':('--','s'),'anchored_additive':('-.','D'),'persistence':(':','o'),'bounded_spatial_mix':('-','^')}
INK='#243447';MUTED='#536273';GRID='#E3E8EC';NEGATIVE='#A33B32'
SIZE=(5.5,2.75)
CAPTION=(r'\textbf{Single-observation IWS development forecasts.} Each curve retains all 59 future offsets from one observed image and the corresponding causal command prefix. '
         r'The horizontal axis is $H=2,\ldots,60$, whose target is stored offset $H-1$, not physical time. '
         r'Native training-standardized feature MSE averages windows within trajectory, then trajectories and three matched seeds equally; lower is better. '
         r'Each task has its own zero-based vertical scale. Annotations give signed relative $H=60$ MSE reduction versus the learned additive anchor and the registered paired 95\% seed--trajectory interval; positive values favor ShiftWM. '
         r'Intervals are unadjusted exploratory comparisons, with no invented uncertainty bands on the curves. All three tasks, all three learned arms and persistence are retained. Reserved upstream-validation evaluation is separate.')
INCLUDE=('\\begin{figure}[!htb]\n\\centering\n'
         '\\includegraphics[width=\\linewidth]{generated/experiment_alignment/forecast_transfer.pdf}\n'
         '\\caption[Single-observation IWS development forecasts.]{'+CAPTION+'}\n'
         '\\label{fig:iws-forecast-transfer}\n\\end{figure}\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_evidence():
    # Exact helper is frozen by the running scientific registration.
    if sha(REPORTER)!=REPORTER_SHA:
        raise ValueError('Registered IWS reporting helper changed; refusing to render')
    spec=importlib.util.spec_from_file_location('_frozen_iws_result_reporter',REPORTER)
    reporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporter)
    sources={}
    caches=[reporter.cache_metadata(task,sources) for task in TASKS]
    development=reporter.completed_development(sources,caches)
    return development,sources


def plot_payload(development):
    if development.get('status')!='complete_validated_development' or development.get('scope')!='internal_development':
        raise ValueError('Only fully validated internal-development results can be plotted')
    if set(development['numerical_results'])!=set(TASKS) or len(development['runs'])!=27:
        raise ValueError('Incomplete task or run set')
    rows={}
    for task in TASKS:
        item=development['numerical_results'][task]
        if set(item['means_all59_offsets'])!=set(MODES):raise ValueError('Method omitted from curve set')
        curves={mode:np.asarray(item['means_all59_offsets'][mode],dtype=np.float64) for mode in MODES}
        if any(v.shape!=(59,) or not np.isfinite(v).all() or (v<0).any() for v in curves.values()):
            raise ValueError('Invalid complete forecast curve')
        ours=float(curves['bounded_spatial_mix'][-1]);baseline=float(curves['anchored_additive'][-1])
        expected=None if baseline==0 else 100*(baseline-ours)/baseline
        effect=item['h60_comparisons']['anchored_additive'];gain=effect['relative_mse_reduction_percent']
        if ((gain is None)!=(expected is None) or gain is not None and not math.isclose(gain,expected,rel_tol=1e-12,abs_tol=1e-12)
                or not math.isclose(effect['method_minus_comparator'],ours-baseline,rel_tol=1e-12,abs_tol=1e-12)):
            raise ValueError('Primary annotation does not match the plotted H60 endpoints')
        interval=effect['paired95']['gain']
        if interval is not None and (len(interval)!=2 or not np.isfinite(interval).all() or interval[0]>interval[1]):
            raise ValueError('Invalid paired relative-gain interval')
        rows[task]={'curves':{k:v.tolist() for k,v in curves.items()},'h60_endpoints':{k:float(v[-1]) for k,v in curves.items()},
                    'primary_gain_percent':gain,'primary_gain_ci95_percent':interval,
                    'interval_contains_zero':None if interval is None else interval[0]<=0<=interval[1],
                    'eligible_trajectories':item['eligible_trajectories'],'total_windows':item['total_windows']}
    return {'x_H':list(range(2,61)),'stored_offsets':list(range(1,60)),'tasks':rows,'plotted_means':708,
            'metric':'native training-standardized feature MSE','primary_comparator':'anchored_additive',
            'aggregation':'equal windows within trajectory, equal trajectories, equal matched seeds',
            'curve_uncertainty':'not plotted; no per-offset intervals supplied by the frozen reporter'}


def annotation(row):
    gain=row['primary_gain_percent'];interval=row['primary_gain_ci95_percent']
    first='H60 gain undefined' if gain is None else f'H60 gain {gain:+.2f}%'
    second='95% interval undefined' if interval is None else f'95% [{interval[0]:+.2f}, {interval[1]:+.2f}]%'
    third='vs additive anchor' if not row['interval_contains_zero'] else 'vs anchor; CI includes 0'
    return first,second,third


def make_figure(payload):
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.7,
        'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'stix',
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED})
    fig=plt.figure(figsize=SIZE,facecolor='white')
    # Three task panels have equal geometry; annotations occupy a separate band.
    axes=[];notes=[]
    for index,task in enumerate(TASKS):
        left=.105+index*.305;axis=fig.add_axes([left,.385,.257,.43]);axes.append(axis)
        row=payload['tasks'][task]
        maximum=max(max(v) for v in row['curves'].values())
        upper=1.08*maximum if maximum>0 else 1.
        for mode in MODES:
            linestyle,marker=STYLES[mode]
            axis.plot(payload['x_H'],row['curves'][mode],color=COLORS[mode],linestyle=linestyle,
                marker=marker,markevery=[0,13,28,43,58],ms=2.6,lw=1.35 if mode=='bounded_spatial_mix' else 1.,
                zorder=4 if mode=='bounded_spatial_mix' else 2,label=LABELS[mode],clip_on=True)
        axis.set(xlim=(1,61),ylim=(0,upper),xticks=[2,30,60],xlabel='H (target offset H−1)')
        if index==0:axis.set_ylabel('Standardized feature MSE',labelpad=3)
        axis.set_title(f'{chr(97+index)}  {NAMES[task]}',loc='left',pad=8,fontweight='normal')
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=2))
        formatter=ScalarFormatter(useOffset=False,useMathText=True);formatter.set_powerlimits((-2,3));axis.yaxis.set_major_formatter(formatter)
        axis.spines[['top','right']].set_visible(False)
        for edge in ('left','bottom'):axis.spines[edge].set_color(MUTED);axis.spines[edge].set_linewidth(.65)
        axis.tick_params(length=2,pad=2);axis.grid(axis='y',color=GRID,lw=.55);axis.set_axisbelow(True)
        for j,text in enumerate(annotation(row)):
            color=(COLORS['bounded_spatial_mix'] if row['primary_gain_percent'] is not None and row['primary_gain_percent']>0
                   else NEGATIVE if row['primary_gain_percent'] is not None and row['primary_gain_percent']<0 else INK) if j==0 else INK
            notes.append(fig.text(left,.195-.050*j,text,fontsize=8,color=color,ha='left',va='center'))
    handles=[Line2D([],[],color=COLORS[m],ls=STYLES[m][0],marker=STYLES[m][1],ms=3,lw=1.2) for m in MODES]
    legend=fig.legend(handles,[LABELS[m] for m in MODES],loc='upper center',bbox_to_anchor=(.54,1.005),
        ncol=4,frameon=False,handlelength=1.35,handletextpad=.35,columnspacing=.8,fontsize=8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();issues=[]
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():continue
        box=text.get_window_extent(renderer)
        if text.get_fontsize()<8:issues.append({'type':'small_font','text':text.get_text()})
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:
            issues.append({'type':'clipped','text':text.get_text()})
    legend_box=legend.get_window_extent(renderer)
    for index,axis in enumerate(axes):
        if axis.get_window_extent(renderer).overlaps(legend_box):issues.append({'type':'legend_data_overlap','panel':index})
        label=axis.xaxis.label.get_window_extent(renderer)
        for note in notes[index*3:index*3+3]:
            if note.get_window_extent(renderer).overlaps(label):issues.append({'type':'annotation_xlabel_overlap','panel':index})
        if index<2:
            next_left=axes[index+1].get_window_extent(renderer).x0
            for note in notes[index*3:index*3+3]:
                if note.get_window_extent(renderer).x1>next_left-2:issues.append({'type':'annotation_panel_overlap','panel':index})
    if issues:
        plt.close(fig);raise ValueError('Realized figure geometry needs review: '+json.dumps(issues))
    return fig,{'dimensions_inches':list(SIZE),'minimum_font_pt':8,'issues':[],
                'zero_based_y_axes':True,'per_task_y_scales':True,'all59_offsets_all4_methods':True}


def identity(development,sources,payload):
    font=Path(font_manager.findfont(font_manager.FontProperties(family='Liberation Sans'),fallback_to_default=False))
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(__file__),str(REPORTER.relative_to(ROOT)):sha(REPORTER),
                  'paper/figure_sources/iws_results/brief.md':sha(ROOT/'paper/figure_sources/iws_results/brief.md')}
    return {'schema':'shiftwm_iws_forecast_figure_v1','finalization_sha256':development['finalization_sha256'],
            'renderer_dependencies_sha256':dependencies,'evidence_sources_sha256':sources,
            'runtime':{'matplotlib':matplotlib.__version__,'numpy':np.__version__,'font_path':str(font),'font_sha256':sha(font)},
            'numerical_payload':payload,'full_validated_development':development,'caption':CAPTION,'include':INCLUDE}


def existing(fingerprint,content):
    path=OUT/'forecast_transfer.json'
    if not path.exists():return False
    record=json.loads(path.read_text())
    if record.get('fingerprint')!=fingerprint or record.get('bound_payload')!=content:
        raise ValueError('Existing IWS figure belongs to a different source snapshot; explicit rendering revision required')
    for name,expected in record.get('outputs_sha256',{}).items():
        if name not in ('forecast_transfer.pdf','forecast_transfer.svg','forecast_transfer.png','forecast_transfer_figure.tex') or sha(OUT/name)!=expected:
            raise ValueError('Existing IWS figure export is missing or changed')
    if set(record.get('outputs_sha256',{}))!={'forecast_transfer.pdf','forecast_transfer.svg','forecast_transfer.png','forecast_transfer_figure.tex'}:
        raise ValueError('Existing IWS figure has incomplete exports')
    return True


def render(if_ready=False):
    if not FINAL.exists():
        if if_ready:return {'status':'pending','reason':'Complete IWS development finalizer absent','outputs_written':False}
        raise ValueError('Complete IWS development finalizer required')
    development,sources=load_evidence()
    if development.get('status')=='pending':
        if if_ready:return {'status':'pending','reason':development['reason'],'outputs_written':False}
        raise ValueError('IWS development finalizer has not passed')
    payload=plot_payload(development);content=identity(development,sources,payload);fingerprint=digest(content)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.forecast_transfer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if existing(fingerprint,content):return {'status':'unchanged_verified','fingerprint':fingerprint,'outputs_written':False}
        stage=Path(tempfile.mkdtemp(prefix='.forecast_transfer-',dir=OUT))
        try:
            fig,audit=make_figure(payload)
            for extension in ('pdf','svg','png'):fig.savefig(stage/f'forecast_transfer.{extension}',dpi=300)
            plt.close(fig)
            (stage/'forecast_transfer_figure.tex').write_text(INCLUDE)
            outputs={p.name:sha(p) for p in stage.iterdir()}
            record={'status':'numerically_validated_geometry_checked_visual_review_pending','fingerprint':fingerprint,
                    'bound_payload':content,'outputs_sha256':outputs,'geometry_checks':audit,
                    'visual_review':{'actual_pdf_color':'pending','actual_pdf_grayscale':'pending','official_manuscript_page':'pending',
                                     'instructions':'paper/figure_sources/iws_results/brief.md'}}
            (stage/'forecast_transfer.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            # Full source identity still holds before making exports/include visible.
            for name,expected in sources.items():
                if sha(ROOT/name)!=expected:raise ValueError('Scientific evidence changed during rendering')
            for name,expected in content['renderer_dependencies_sha256'].items():
                if sha(ROOT/name)!=expected:raise ValueError('Rendering source changed during export')
            if sha(FINAL)!=development['finalization_sha256']:raise ValueError('Finalizer changed during rendering')
            for name in ('forecast_transfer.pdf','forecast_transfer.svg','forecast_transfer.png','forecast_transfer.json','forecast_transfer_figure.tex'):
                os.replace(stage/name,OUT/name)
        finally:shutil.rmtree(stage,ignore_errors=True)
    return {'status':'rendered_visual_review_pending','fingerprint':fingerprint,'plotted_means':708,'primary_intervals':3,'outputs_written':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--if-ready',action='store_true')
    args=parser.parse_args();print(json.dumps(render(args.if_ready),sort_keys=True))
