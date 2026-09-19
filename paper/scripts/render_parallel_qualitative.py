#!/usr/bin/env python3
"""Parallel two-case layout of the existing verified DROID replay; no inference."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import render_editorial_qualitative as source

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/editorial'
DESIGN=ROOT/'paper/design/parallel_qualitative'
HELPER=ROOT/'paper/scripts/render_editorial_qualitative.py'
EXPECTED_HELPER='a7ce31f15fce8383f90f83aa236bdcf16f4f45f1cf5aad9c0003c03b4d2e8630'
WIDTH,HEIGHT=5.5,2.8
CAPTION=(r'\textbf{Selected improvement and regression on recorded video.} '
         r'The prespecified largest episode gain and largest regression among 141 DROID development episodes each show their first eligible window. '
         r'The RGB images are recorded observations and withheld targets. '
         r'AR denotes matched autoregression; ShiftWM (ours) uses bounded spatial mixing. '
         r'Maps and numbers show native standardized $h=10$ feature MSE over three seeds, with one unclipped logarithmic scale shared across all six original gallery maps. '
         r'Positive gains mean lower error; episode and shown-window averages differ. '
         r'The appendix retains the median case and full selection record. Images: DROID (CC BY 4.0).')


def text(fig,x,y,label,size=8.5,color='#243447',weight='normal',ha='center'):
    return fig.text(x/WIDTH,y/HEIGHT,label,fontsize=size,color=color,weight=weight,ha=ha,va='center')


def axis(fig,x,y,w,h):
    return fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT])


def preserve_previous_includes():
    archive=DESIGN/'previous_includes';archive.mkdir(parents=True,exist_ok=True)
    for name in ('qualitative_figure.tex','qualitative_caption.tex','qualitative_evidence.json'):
        original=OUT/name;dest=archive/name
        if not dest.exists():
            shutil.copyfile(original,dest)
    return {str(p.relative_to(ROOT)):source.sha(p) for p in sorted(archive.iterdir()) if p.is_file()}


def render():
    if source.sha(HELPER)!=EXPECTED_HELPER:
        raise ValueError('The reviewed source helper changed; require explicit review before reusing it')
    if (source.WIDTH,source.HEIGHT)!=(WIDTH,HEIGHT):
        raise ValueError('Geometry audit physical size changed')
    replay,arrays,maps,checks=source.verified_pack()
    colors=source.initialize_style()
    OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
    archived=preserve_previous_includes()
    vmin=min(float(m.min()) for m in maps.values())
    vmax=max(float(m.max()) for m in maps.values())
    if not 0<vmin<vmax:
        raise ValueError('Log scale requires exact positive source range, without a fabricated floor')
    norm=LogNorm(vmin=vmin,vmax=vmax,clip=False)
    fig=plt.figure(figsize=(WIDTH,HEIGHT),dpi=180,facecolor='white')
    fig.add_artist(plt.Line2D([2.75/WIDTH,2.75/WIDTH],[.39/HEIGHT,2.72/HEIGHT],
                             transform=fig.transFigure,color=colors['grid'],linewidth=.7))
    displayed=[]
    for col,index in enumerate((0,2)):
        case=replay['cases'][index];prefix=case['prefix'];left=.10+2.76*col
        center=left+1.22
        text(fig,center,2.65,'a  Largest episode gain' if col==0 else 'b  Largest regression',
             size=9.5,weight='bold')
        text(fig,center,2.46,f"Episode {case['gain_percent']:+.1f}% · shown {case['first_window_gain_percent']:+.1f}%",
             size=8,color=colors['ours'] if col==0 else colors['secondary'],weight='bold' if col==0 else 'normal')
        for offset,slot,label in ((0.,2,'Observed · f10'),(1.23,12,'Withheld · f60')):
            x=left+offset;w=1.10;h=w*9/16
            text(fig,x+w/2,2.28,label,size=8.5,color=colors['observation'] if slot==2 else colors['secondary'])
            ax=axis(fig,x,1.54,w,h)
            ax.imshow(arrays[prefix+'_images'][slot],interpolation='none');ax.axis('off')
        for offset,mode,label in ((0.,'autoregressive','AR'),(1.23,'transport','ShiftWM (ours)')):
            x=left+offset+.185;w=.73
            text(fig,x+w/2,1.39,label,size=8.5,color=colors['ours'] if mode=='transport' else colors['ink'],
                 weight='bold' if mode=='transport' else 'normal')
            ax=axis(fig,x,.55,w,w)
            last_image=ax.imshow(maps[prefix,mode],cmap='magma',norm=norm,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
            ax.grid(which='minor',color='white',alpha=.28,linewidth=.45);ax.tick_params(which='minor',length=0)
            for spine in ax.spines.values():spine.set_edgecolor(colors['grid']);spine.set_linewidth(.55)
            favorable=(mode=='transport' and float(maps[prefix,mode].mean())<float(maps[prefix,'autoregressive'].mean()))
            text(fig,x+w/2,.42,f'{maps[prefix,mode].mean():.4f}',size=8.5,
                 color=colors['ours'] if favorable else colors['secondary'],weight='bold' if favorable else 'normal')
        displayed.append(checks[index])
    bar=fig.colorbar(last_image,cax=axis(fig,1.99,.22,2.38,.075),orientation='horizontal')
    bar.set_ticks([vmin,.1,vmax]);bar.ax.set_xticklabels([f'{vmin:.4f}','0.1',f'{vmax:.3f}'])
    bar.ax.minorticks_off();bar.ax.tick_params(length=2,pad=2,labelsize=8);bar.outline.set_linewidth(.5)
    text(fig,1.76,.257,'Feature MSE ↓ · log scale',size=8,color=colors['secondary'],ha='right')
    audit=source.geometry(fig)
    stem=OUT/'qualitative_parallel'
    for ext in ('pdf','svg','png'):fig.savefig(stem.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    proofdir=DESIGN/'proof';proofdir.mkdir(exist_ok=True)
    for name,flags,dpi in (('paper_width',[],120),('detail',[],300),('grayscale',['-gray'],120)):
        subprocess.run(['pdftoppm','-singlefile','-png','-r',str(dpi),*flags,str(stem.with_suffix('.pdf')),str(proofdir/name)],check=True,capture_output=True)
    (OUT/'qualitative_caption.tex').write_text(CAPTION+'\n')
    (OUT/'qualitative_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n'
        '\\includegraphics[width=\\linewidth]{generated/editorial/qualitative_parallel.pdf}\n'
        '\\caption{'+CAPTION+'}\n\\label{fig:editorial-qualitative}\n\\end{figure}\n')
    proof='\\documentclass{article}\n\\usepackage{iclr2027_conference,times,graphicx,amsmath,hyperref}\n\\begin{document}\n\\input{generated/editorial/qualitative_figure}\n\\end{document}\n'
    (proofdir/'proof.tex').write_text(proof)
    env=dict(os.environ,TEXINPUTS='template/official/iclr2027:'+os.environ.get('TEXINPUTS',''))
    for i in (1,2):
        r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=design/parallel_qualitative/proof','design/parallel_qualitative/proof/proof.tex'],cwd=ROOT/'paper',env=env,capture_output=True,text=True)
        (proofdir/f'compile_{i}.txt').write_text(r.stdout+r.stderr)
        if r.returncode:raise ValueError('Official ICLR proof failed')
    log=(proofdir/'proof.log').read_text()
    warnings=[line for line in log.splitlines() if any(word in line for word in ('Overfull','Underfull','Float too large','LaTeX Warning','undefined','! LaTeX'))]
    if warnings:raise ValueError('Proof warnings: '+repr(warnings))
    subprocess.run(['pdftoppm','-singlefile','-png','-r','120',str(proofdir/'proof.pdf'),str(proofdir/'official_page')],check=True,capture_output=True)
    outputs=[stem.with_suffix('.'+ext) for ext in ('pdf','svg','png')]+[OUT/'qualitative_caption.tex',OUT/'qualitative_figure.tex',proofdir/'proof.pdf']
    evidence={'status':'numeric_and_geometry_passed_actual_visual_review_pending','created_utc':datetime.now(timezone.utc).isoformat(),
      'source_sha256':source.sha(__file__),'helper_sha256':source.sha(HELPER),'replay_sha256':source.EXPECTED_REPLAY,
      'style_sha256':source.sha(source.STYLE),'layout':'Two equal case columns, recorded observed/withheld images above matched measured maps',
      'selection':'Exact original case0/case2: prespecified episode extremes; first eligible windows; no reselection',
      'recorded_pixel_check':'verified_pack checked all12 original exported PNGs against saved decoded RGB arrays and hashes',
      'shown_observed_frame_index':10,'withheld_frame_index':60,'model_observed_indices':[0,5,10],
      'common_vmin':vmin,'common_vmax':vmax,'scale_population':'All6 maps of original3 cases, including undisplayed median',
      'color_normalization':'LogNorm with true positive measured lower bound; clip=False; no invented floor',
      'coordinate_space':'Native4x4 shared-channel-standardized feature errors, three-seed means at h10',
      'displayed_cases':displayed,'all_recomputed_case_values':checks,'geometry':audit,'proof_warnings':warnings,
      'scientific_scope':'Re-render only; no model inference, new selection, test evaluation, RGB prediction or causal-attribution claim',
      'previous_include_archive':archived,
      'input_files':{str(p.relative_to(ROOT)):source.sha(p) for p in sorted(source.PACK.iterdir()) if p.is_file()},
      'outputs':{str(p.relative_to(ROOT)):source.sha(p) for p in outputs}}
    (OUT/'qualitative_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (OUT/'qualitative_parallel_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'displayed':displayed,'range':[vmin,vmax],'geometry':audit,'proof_warnings':warnings}))


if __name__=='__main__':render()
