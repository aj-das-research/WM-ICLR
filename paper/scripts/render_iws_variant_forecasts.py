#!/usr/bin/env python3
"""Five-predictor IWS curves, gated by the completed portable 9+27 score pack.

Pending writes nothing. --proof-v1 uses the actual completed four-method pack
only in a separately named design proof; it is never a manuscript export.
No private result, model, cache, dataset, or scientific evaluator is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT/'paper/figure_sources/current_real_scorecards'
HELPER = ROOT/'paper/scripts/render_current_real_scorecards.py'
HELPER_SHA = 'f77e425c1edab9996241617e9198b588854a47c0368758e5e64ce4f1825e7660'
OUT = ROOT/'paper/generated/iws_variant_forecasts'
PREFIX = 'forecast_transfer_all_variants'
PROOF_PREFIX = 'FOUR_METHOD_LAYOUT_PROOF'
TASKS = ('pusht','bimanual_box','bimanual_rope')
NAMES = dict(zip(TASKS,('PushT','Box','Rope')))
MODES = ('persistence','autoregressive','anchored_additive','bounded_spatial_mix','unbounded_spatial_mix')
LABELS = {'persistence':'Persistence','autoregressive':'Autoregressive','anchored_additive':'Additive anchor',
          'bounded_spatial_mix':'ShiftWM (ours)','unbounded_spatial_mix':'No tanh (ours, ablation)'}
COLORS = {'persistence':'#956B26','autoregressive':'#536273','anchored_additive':'#2458C3',
          'bounded_spatial_mix':'#166534','unbounded_spatial_mix':'#984A87'}
STYLES = {'persistence':(':','o'),'autoregressive':('--','s'),'anchored_additive':('-.','D'),
          'bounded_spatial_mix':('-','^'),'unbounded_spatial_mix':((0,(5,1.8,1.3,1.8)),'v')}
CAPTION = (r'\textbf{Complete IWS development forecasts and the innovation-bound ablation.} '
    r'All five predictors retain all 59 future offsets from one observed image; learned methods receive supplied causal command prefixes, while persistence ignores commands. '
    r'The horizontal axis is $H=2,\ldots,60$, targeting stored offset $H-1$, not physical time. '
    r'Training-standardized feature MSE weights windows within trajectory, trajectories and three seeds equally; lower is better. '
    r'Each task has its own zero-based vertical scale. The no-$\tanh$ predictor removes only the correction bound and belongs to a separately registered exploratory follow-up after v1 development. '
    r'All nine new and 27 original runs must pass finalization before this figure appears. No uncertainty bands are inferred; paired endpoint gains and intervals are in Table~\ref{tab:iws-unbounded-component}. '
    r'Reserved upstream-validation data remain excluded.')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_pack(pack):
    if sha(HELPER) != HELPER_SHA:
        raise ValueError('Reviewed portable scorecard validator changed')
    spec = importlib.util.spec_from_file_location('_iws_plot_portable_validator',HELPER)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    manifest = json.loads((pack/'manifest.json').read_text())
    if manifest.get('schema') != 'current_real_scorecards_pack_v1' or sha(pack/'data.json') != manifest.get('data_sha256'):
        raise ValueError('Portable scorecard pack hash/schema mismatch')
    data = json.loads((pack/'data.json').read_text());helper.validate_payload(data)
    return data


def plot_payload(data,proof=False):
    iws = data['iws']
    if proof:
        if iws['unbounded_included']:
            raise ValueError('Four-method proof requires the completed four-method pack')
        modes = MODES[:-1]
    else:
        if not iws['unbounded_included']:
            return None
        c = iws['completion']
        if c['v1_runs'] != 27 or c['unbounded_runs'] != 9 or c['epochs_each'] != 30 or not c['unbounded_finalization_sha256']:
            raise ValueError('Complete nine-plus-27 finalization required')
        modes = MODES
    tasks = {}
    for task in TASKS:
        study = iws['tasks'][task]
        if set(study['methods']) != set(modes):
            raise ValueError('Missing or extra forecast method')
        curves = {}
        for mode in modes:
            curve = np.asarray(study['methods'][mode]['mean_curves']['standardized_mse'],dtype=np.float64)
            if curve.shape != (59,) or not np.isfinite(curve).all() or (curve < 0).any():
                raise ValueError('Invalid complete 59-offset curve')
            curves[mode] = curve.tolist()
        tasks[task] = {'curves':curves,'population':study['population']}
    return {'schema':'iws_variant_curve_payload_v1','status':'four_method_layout_proof' if proof else 'complete_9_plus_27',
            'modes':list(modes),'x_H':list(range(2,61)),'stored_offsets':list(range(1,60)),
            'metric':'training-standardized feature MSE','direction':'lower','tasks':tasks,
            'plotted_means':len(modes)*3*59,'uncertainty_bands':'not measured or drawn',
            'completion':iws['completion']}


def make_figure(payload):
    proof = payload['status']=='four_method_layout_proof'
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
        'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix',
        'text.color':'#243447','axes.labelcolor':'#243447','xtick.color':'#536273','ytick.color':'#536273',
        'svg.hashsalt':'iws-variant-forecast-v1'})
    figure = plt.figure(figsize=(5.5,2.4),facecolor='white');axes=[]
    for index,task in enumerate(TASKS):
        axis = figure.add_axes([.105+index*.305,.245,.247,.485]);axes.append(axis)
        row = payload['tasks'][task]
        for mi,mode in enumerate(payload['modes']):
            ls,marker = STYLES[mode]
            axis.plot(payload['x_H'],row['curves'][mode],color=COLORS[mode],ls=ls,marker=marker,
                markevery=[0,13+mi,28+mi,43+mi,58],ms=2.6,mew=.7,
                lw=1.3 if mode in MODES[-2:] else 1.0,label=LABELS[mode],zorder=2+mi)
        maximum = max(max(v) for v in row['curves'].values())
        axis.set(xlim=(1,61),ylim=(0,1.08*maximum if maximum else 1),xticks=[2,30,60])
        axis.set_title(f'{chr(97+index)}  {NAMES[task]}',loc='left',pad=6)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=2))
        axis.ticklabel_format(axis='y',style='plain',useOffset=False)
        axis.tick_params(length=2,pad=2)
        axis.spines[['top','right']].set_visible(False)
        for side in ('left','bottom'):axis.spines[side].set_color('#536273');axis.spines[side].set_linewidth(.65)
        axis.grid(axis='y',color='#E3E8EC',lw=.5);axis.set_axisbelow(True)
        if index == 0:axis.set_ylabel('Standardized feature MSE',labelpad=3)
    figure.text(.54,.075,'H (target stored offset H − 1)',ha='center',fontsize=8)
    legends=[]
    for group,y in [(MODES[:3],.99),(tuple(m for m in MODES[3:] if m in payload['modes']),.915)]:
        handles=[Line2D([],[],color=COLORS[m],ls=STYLES[m][0],marker=STYLES[m][1],ms=3,lw=1.2) for m in group]
        legends.append(figure.legend(handles,[LABELS[m] for m in group],loc='upper center',bbox_to_anchor=(.54,y),
            ncol=len(group),frameon=False,handlelength=2.1,handletextpad=.4,columnspacing=1.2,fontsize=8))
    if proof:
        figure.text(.54,.002,'FOUR-METHOD LAYOUT PROOF · completed v1 values only',ha='center',va='bottom',fontsize=8,color='#536273')
    figure.canvas.draw();renderer=figure.canvas.get_renderer();issues=[];gaps=[]
    for item in figure.findobj(Text):
        if not item.get_visible() or not item.get_text():continue
        box=item.get_window_extent(renderer)
        if item.get_fontsize()<8:issues.append('Font below8pt: '+item.get_text())
        if box.x0<-.5 or box.y0<-.5 or box.x1>figure.bbox.x1+.5 or box.y1>figure.bbox.y1+.5:
            issues.append('Clipped text: '+item.get_text())
    for index,axis in enumerate(axes):
        if len(axis.lines)!=len(payload['modes']):issues.append('Missing line')
        for line,mode in zip(axis.lines,payload['modes']):
            if not np.array_equal(line.get_xdata(),payload['x_H']) or not np.array_equal(line.get_ydata(),payload['tasks'][TASKS[index]]['curves'][mode]):
                issues.append('Line data changed')
        for legend in legends:
            if legend.get_window_extent(renderer).overlaps(axis.get_window_extent(renderer)):
                issues.append('Legend overlaps plotted data')
            for title in (axis.title,axis._left_title):
                if title.get_text() and legend.get_window_extent(renderer).overlaps(title.get_window_extent(renderer)):
                    issues.append('Legend overlaps panel title')
        if index:
            labels=[t for t in axis.get_yticklabels() if t.get_visible() and t.get_text()]
            gap=(min(t.get_window_extent(renderer).x0 for t in labels)-axes[index-1].get_window_extent(renderer).x1)*72/figure.dpi
            gaps.append(float(gap))
            if gap<3:issues.append('Adjacent tick clearance below3pt')
    if issues:
        plt.close(figure);raise ValueError('Figure layout rejected: '+json.dumps(issues))
    return figure,{'dimensions_inches':[5.5,2.4],'minimum_font_pt':8,'adjacent_tick_clearance_pt':gaps,
                  'zero_based_y_axes':True,'per_task_y_scales':True,'all_plotted_points_exact':True,'issues':[]}


def render(if_ready=False,output=None,pack=None,proof=False):
    output = OUT if output is None else Path(output).resolve()
    pack = PACK if pack is None else Path(pack).resolve()
    if proof and (output == OUT.resolve() or output.is_relative_to((ROOT/'paper/generated').resolve())):
        raise ValueError('Four-method proof must stay outside manuscript outputs')
    if not (pack/'manifest.json').exists():
        if if_ready:return {'status':'pending','outputs_written':False}
        raise ValueError('Validated portable pack required')
    data=load_pack(pack);payload=plot_payload(data,proof)
    if payload is None:
        if if_ready:return {'status':'pending','reason':'Complete nine-plus-27 pack absent','outputs_written':False}
        raise ValueError('Complete nine-plus-27 pack required')
    prefix=PROOF_PREFIX if proof else PREFIX
    include='' if proof else ('\\begin{figure}[!htb]\n\\centering\n'
        '\\includegraphics[width=\\linewidth]{generated/iws_variant_forecasts/'+PREFIX+'.pdf}\n'
        '\\caption[Complete IWS development forecasts and bound ablation.]{'+CAPTION+'}\n'
        '\\label{fig:iws-forecast-transfer}\n\\end{figure}\n')
    font=Path(font_manager.findfont(font_manager.FontProperties(family='Liberation Sans'),fallback_to_default=False))
    bound={'renderer_sha256':sha(__file__),'validator_sha256':sha(HELPER),
           'pack_sha256':sha(pack/'data.json'),'manifest_sha256':sha(pack/'manifest.json'),
           'payload':payload,'include':include,'caption':None if proof else CAPTION,
           'runtime':{'matplotlib':matplotlib.__version__,'numpy':np.__version__,'font_sha256':sha(font)}}
    fingerprint=digest(bound);receipt=output/(prefix+'.json')
    if receipt.exists():
        existing=json.loads(receipt.read_text())
        if existing.get('fingerprint')!=fingerprint:raise ValueError('Figure snapshot differs; use a new reviewed output directory')
        for name,expected in existing['outputs_sha256'].items():
            if sha(output/name)!=expected:raise ValueError('Existing figure output changed')
        return {'status':'unchanged_verified','outputs_written':False,'plotted_means':payload['plotted_means']}
    fig,geometry=make_figure(payload)
    output.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.iws-variants-',dir=output))
    try:
        names=[]
        for ext in ('pdf','svg','png'):
            name=prefix+'.'+ext;names.append(name)
            metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
            fig.savefig(stage/name,dpi=300,metadata=metadata)
        if include:
            names.append(prefix+'_figure.tex');(stage/names[-1]).write_text(include)
        if sha(pack/'data.json')!=bound['pack_sha256'] or sha(pack/'manifest.json')!=bound['manifest_sha256']:
            raise ValueError('Portable source changed during plotting')
        record={'status':payload['status'],'fingerprint':fingerprint,'bound_payload':bound,'geometry':geometry,
                'outputs_sha256':{name:sha(stage/name) for name in names}}
        (stage/(prefix+'.json')).write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
        for name in [*names,prefix+'.json']:(stage/name).replace(output/name)
    finally:
        plt.close(fig);shutil.rmtree(stage)
    return {'status':payload['status'],'outputs_written':True,'plotted_means':payload['plotted_means']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--if-ready',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--pack-dir',type=Path)
    parser.add_argument('--proof-v1',action='store_true')
    args=parser.parse_args()
    print(json.dumps(render(args.if_ready,args.output_dir,args.pack_dir,args.proof_v1),sort_keys=True))
