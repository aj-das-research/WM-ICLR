#!/usr/bin/env python3
"""Source-driven candidate only: observed RGB, feature errors and measured mixing."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay import ROOT,OUTPUT,REG,MODES,sha,read,atomic_json,check_rule
from selection import RULE
WIDTH,HEIGHT=5.5,7.0
INK='#193244';MUTED='#62717b';BLUE='#2674a5';ORANGE='#cc681e';GREEN='#18714c';RUST='#a54435'


def check_measurements(output):
    check_rule();measured=read(output/'replay.json')
    if measured['status']!='numeric_replay_passed_visual_review_pending' or len(measured['cases'])!=3:
        raise ValueError('Verified three-case replay required')
    if measured['selection_registration_sha256']!=sha(REG) or measured['arrays_sha256']!=sha(output/'replay_arrays.npz'):
        raise ValueError('Measured replay identity changed')
    for relative,expected in measured['sources'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Measured source changed: '+relative)
    arrays=dict(np.load(output/'replay_arrays.npz',allow_pickle=False))
    for case in measured['cases']:
        if len(case['replay_checks'])!=6:raise ValueError('Every method/seed must be replayed')
        images=arrays[case['prefix']+'_images']
        for entry,index in zip(case['frame_exports'],(0,1,2,12)):
            path=output/entry['path'];pixels=np.asarray(Image.open(path).convert('RGB'))
            if sha(path)!=entry['file_sha256'] or not np.array_equal(pixels,images[index]):
                raise ValueError('Observed PNG no longer matches raw source pixels')
    return measured,arrays


def check_portable_measurements(directory, expected_replay_sha256):
    """Redraw a reviewed replay pack; this is not a fresh scientific audit."""
    if (len(expected_replay_sha256)!=64 or sha(directory/'replay.json')!=expected_replay_sha256):
        raise ValueError('Portable redraw requires the exact reviewed replay SHA256')
    measured=read(directory/'replay.json')
    if (measured.get('status')!='numeric_replay_passed_visual_review_pending'
            or len(measured.get('cases',[]))!=3
            or measured.get('arrays_sha256')!=sha(directory/'replay_arrays.npz')):
        raise ValueError('Portable replay bundle identity differs')
    arrays=dict(np.load(directory/'replay_arrays.npz',allow_pickle=False))
    for case in measured['cases']:
        if len(case['replay_checks'])!=6:raise ValueError('Incomplete portable replay')
        for entry,index in zip(case['frame_exports'],(0,1,2,12)):
            path=directory/entry['path'];pixels=np.asarray(Image.open(path).convert('RGB'))
            if sha(path)!=entry['file_sha256'] or not np.array_equal(pixels,arrays[case['prefix']+'_images'][index]):
                raise ValueError('Portable recorded-frame pixels differ')
    return measured,arrays


def geometry(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();box=fig.bbox
    violations=[];texts=[]
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():continue
        bounds=artist.get_window_extent(renderer)
        if artist.get_fontsize()<7.95:violations.append('Font below 8pt: '+artist.get_text())
        if bounds.x0<box.x0-1 or bounds.y0<box.y0-1 or bounds.x1>box.x1+1 or bounds.y1>box.y1+1:
            violations.append('Text clipped: '+artist.get_text())
        if artist in fig.texts:texts.append((artist,bounds))
    for index,(artist,a) in enumerate(texts):
        for other,b in texts[index+1:]:
            if a.overlaps(b):violations.append('Figure labels overlap: '+artist.get_text()+' / '+other.get_text())
    if violations:raise ValueError('; '.join(violations))
    return {'status':'passed','min_font_pt':8,'page_width_inches':WIDTH,'page_height_inches':HEIGHT,'text_clipping_or_overlap':0}


def render(output=OUTPUT,engineering_fixture=False,portable_replay_sha256=None,input_directory=None):
    output=Path(output)
    if output.resolve().is_relative_to((ROOT/'paper').resolve()):raise ValueError('Unreviewed candidates must remain outside paper')
    if engineering_fixture and output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Engineering fixtures must remain outside the project')
    inputs=Path(input_directory) if input_directory is not None else output
    if portable_replay_sha256 is None:
        measured,arrays=check_measurements(inputs)
    else:
        measured,arrays=check_portable_measurements(inputs,portable_replay_sha256)
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'axes.labelsize':8,
                         'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','image.composite_image':False})
    maps={};curves={};vmax=0.;curve_max=0.
    for case in measured['cases']:
        p=case['prefix']
        for mode in MODES:
            errors=np.stack([arrays[f'{p}_{mode}_s{s}_patch_errors'] for s in (0,1,2)])
            if errors.shape!=(3,10,4,4) or not np.isfinite(errors).all() or (errors<0).any():raise ValueError('Invalid measured patch errors')
            maps[p,mode]=errors[:,9].mean(0);curves[p,mode]=errors.mean((-1,-2))
            vmax=max(vmax,float(maps[p,mode].max()));curve_max=max(curve_max,float(curves[p,mode].max()))
    if vmax<=0 or curve_max<=0:raise ValueError('Nonpositive error display scale')
    fig=plt.figure(figsize=(WIDTH,HEIGHT),dpi=180,facecolor='white')
    def text(x,y,label,size=8,color=INK,weight='normal',ha='left'):
        return fig.text(x/WIDTH,y/HEIGHT,label,fontsize=size,color=color,fontweight=weight,ha=ha,va='center')
    def axis(x,y,w,h):return fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT])
    text(.10,6.83,'ARTIFICIAL ENGINEERING LAYOUT; NOT RESULTS' if engineering_fixture else 'Recorded scenes, measured feature forecasts',10,weight='bold')
    pop=measured['population']
    text(.10,6.59,f"{pop['episodes']} episodes · {pop['positive_episodes']} gains / {pop['negative_episodes']} regressions · pooled {pop['pooled_relative_reduction_percent']:+.2f}%",8,color=MUTED)
    weights=[]
    for i,case in enumerate(measured['cases']):
        p=case['prefix'];top=6.34-2.0*i;shown=case['first_window_gain_percent'];episode_gain=case['gain_percent']
        text(.10,top,f"{'ABC'[i]}  {case['label']}",8.5,weight='bold')
        text(4.12,top,f'episode {episode_gain:+.1f}%',8,color=GREEN if episode_gain>0 else RUST,ha='right')
        text(5.39,top,f'shown {shown:+.1f}%',8,color=GREEN if shown>0 else RUST,ha='right')
        for col,frame in enumerate((0,1,2,12)):
            x=.10+1.34*col
            image=axis(x,top-.96,1.26,.709)
            image.imshow(arrays[p+'_images'][frame]);image.axis('off')
            native=int(arrays[p+'_frame_indices'][frame])
            text(x+.63,top-.20,('Observed ' if col<3 else 'Target ')+f'frame {native}',8,ha='center',color=MUTED)
        bottom=top-1.72
        for j,mode in enumerate(MODES):
            x=.18+j*1.04
            ax=axis(x,bottom,.58,.58)
            error_image=ax.imshow(maps[p,mode],vmin=0,vmax=vmax,cmap='magma',interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
            ax.grid(which='minor',color='white',alpha=.25,linewidth=.5);ax.tick_params(which='minor',length=0)
            for spine in ax.spines.values():spine.set_edgecolor('#d0d7dc');spine.set_linewidth(.5)
            text(x+.29,top-1.09,'AR baseline' if j==0 else 'Transport (ours)',8,ha='center',color=BLUE if j==0 else ORANGE)
            text(x+.29,bottom-.13,f'MSE {maps[p,mode].mean():.4f}',8,ha='center',color=MUTED)
        patch=RULE['target_patch_zero_based'][0]*4+RULE['target_patch_zero_based'][1]
        mixing=np.stack([arrays[f'{p}_transport_s{s}_matrix'][:,patch] for s in (0,1,2)])[:,9].mean(0).reshape(4,4)
        gate=float(np.mean([arrays[f'{p}_transport_s{s}_gate'][9,patch,0] for s in (0,1,2)]))
        weights.append({'case':p,'displayed_target_patch_zero_based':RULE['target_patch_zero_based'],'mean_gate':gate,'transport_row':mixing.tolist()})
        ax=axis(2.30,bottom,.58,.58)
        weight_image=ax.imshow(mixing,vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_edgecolor('#d0d7dc');spine.set_linewidth(.5)
        text(2.59,top-1.09,'Feature mixing',8,ha='center')
        text(2.59,bottom-.13,f'gate {gate:.3f}',8,ha='center',color=MUTED)
        ax=axis(3.52,bottom,1.83,.59)
        for mode,color in zip(MODES,(BLUE,ORANGE)):
            values=curves[p,mode]
            ax.plot(np.arange(1,11),values.mean(0),color=color,lw=1.3,
                    linestyle='-' if mode==MODES[0] else (0,(3,1.5)),
                    marker='o' if mode==MODES[0] else 'D',markevery=[0,4,9],ms=2.8)
            ax.fill_between(np.arange(1,11),values.min(0),values.max(0),color=color,alpha=.13,linewidth=0)
        ax.set_xlim(1,10);ax.set_ylim(0,curve_max*1.06);ax.set_xticks([1,5,10]);ax.yaxis.set_major_locator(MaxNLocator(2))
        ax.tick_params(length=2,pad=2,width=.5)
        ax.spines[['top','right']].set_visible(False);ax.spines[['bottom','left']].set_color('#cbd3d8')
        ax.grid(axis='y',color='#e7ebee',lw=.5);ax.set_axisbelow(True)
        text(4.44,top-1.09,'First-window errors',8,ha='center')
    bar=fig.colorbar(error_image,cax=axis(.17,.23,1.68,.06),orientation='horizontal')
    bar.set_ticks([0,vmax]);bar.ax.set_xticklabels(['0',f'{vmax:.3f}']);bar.ax.tick_params(length=1.5,pad=1,labelsize=8)
    text(1.01,.055,'Standardized feature MSE',8,ha='center',color=MUTED)
    bar=fig.colorbar(weight_image,cax=axis(2.25,.23,.67,.06),orientation='horizontal')
    bar.set_ticks([0,1]);bar.ax.tick_params(length=1.5,pad=1,labelsize=8)
    text(2.59,.055,'Mixing weight',8,ha='center',color=MUTED)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=BLUE,linestyle='-',marker='o',markersize=2.8,label='AR baseline'),
             Line2D([],[],color=ORANGE,linestyle=(0,(3,1.5)),marker='D',markersize=2.8,label='Ours')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(4.43/WIDTH,.23/HEIGHT),
               ncol=2,frameon=False,fontsize=8,handlelength=1.1,handletextpad=.3,columnspacing=.7)
    text(4.44,.055,'Query step · shading: 3-seed range',8,ha='center',color=MUTED)
    checked_geometry=geometry(fig)
    stem='spatial_qualitative_candidate'
    for extension in ('pdf','svg','png'):
        fig.savefig(output/f'{stem}.{extension}',dpi=300,facecolor='white')
    plt.close(fig)
    for suffix,options in [('paper_size',[]),('enlarged',[]),('grayscale',['-gray'])]:
        dpi='300' if suffix=='enlarged' else '100'
        subprocess.run(['pdftoppm','-singlefile','-png','-r',dpi,*options,str(output/f'{stem}.pdf'),str(output/f'{stem}_{suffix}')],check=True,capture_output=True)
    caption=(r'\textbf{Prespecified recorded-video examples and spatial feature diagnostics.} '
        r'Best, median and worst cases are ranked by three-seed episode-level h10 gain of Transport (ours) versus the matched $4\times4$ autoregressive baseline, averaging all eligible windows. '
        r'The first eligible window is always shown; its gain can differ from the episode rank. Images are recorded support and true h10 target frames, not RGB forecasts. '
        r'Error maps average squared standardized errors over 384 channels and three seeds; all six maps share one unclipped scale. Curves show mean error and the three-seed range, not confidence intervals; solid circles denote the baseline and dashed diamonds denote ours. '
        r'Mixing grids show actual h10 weights from anchor source cells to fixed target cell (row 2, column 2), averaged across seeds, with the branch gate below. These are semantic feature mixtures, not physical flow or evidence of causal benefit. '
        r'Population gains and regressions appear above. This gain-conditioned validation gallery is not an independent test.')
    if engineering_fixture:caption=r'\textbf{ARTIFICIAL ENGINEERING LAYOUT; NOT RESULTS.} '+caption
    (output/'caption.tex').write_text(caption+'\n')
    proof=r'''\documentclass{article}
\usepackage{iclr2027_conference,times}
\usepackage{graphicx,amsmath,hyperref}
\begin{document}
\begin{figure}[p]\centering
\includegraphics[width=\linewidth]{spatial_qualitative_candidate.pdf}
\caption{'''+caption+r'''}
\end{figure}
\end{document}
'''
    (output/'proof.tex').write_text(proof)
    env=dict(os.environ,TEXINPUTS=str(ROOT/'paper/template/official/iclr2027')+'//:'+os.environ.get('TEXINPUTS',''))
    for index in range(2):
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','proof.tex'],cwd=output,env=env,capture_output=True,text=True)
        (output/f'proof_pass{index+1}.stdout').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('Candidate caption proof failed')
    log=(output/'proof.log').read_text()
    if any(token in log for token in ('Overfull','Float too large','undefined','multiply defined')):raise ValueError('Candidate ICLR proof has layout/reference warnings')
    for relative,expected in measured['sources'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Measured source changed during rendering')
    review={'status':'numeric_and_geometry_passed_visual_review_pending','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'replay_sha256':sha(inputs/'replay.json'),'renderer_sha256':sha(__file__),'geometry':checked_geometry,
        'input_directory':str(inputs),'portable_redraw':portable_replay_sha256 is not None,
        'scientific_source_validation':'Full original source hashes checked' if portable_replay_sha256 is None else 'Hash-pinned reviewed replay pack only; not a new source/model/scientific audit',
        'shared_error_limits':[0,vmax],'shared_mixing_limits':[0,1],'shared_curve_limits':[0,curve_max*1.06],
        'mixing_panels':weights,'proof_compile_passes':2,'outputs':{p.name:sha(p) for p in output.glob(stem+'*')},
        'proof_pdf_sha256':sha(output/'proof.pdf'),'visual_review':'pending; no manuscript include or publication created',
        'required_visual_checks':['paper_size_readability','observed_frame_identity','shared_scales','case_rank_vs_window_effect',
                                  'feature_mixing_semantics','enlarged_labels_and_grid_cells','grayscale','caption_and_page_fit']}
    atomic_json(review,output/'review_pending.json');print(json.dumps({'status':review['status'],'candidate':str(output)}));return review


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--input',type=Path);parser.add_argument('--portable-replay-sha256')
    args=parser.parse_args();render(args.output,portable_replay_sha256=args.portable_replay_sha256,input_directory=args.input)
