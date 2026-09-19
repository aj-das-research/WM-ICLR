#!/usr/bin/env python3
"""Compact task story from unchanged recorded frames and verified feature errors."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.path import Path as MPath
from PIL import Image
from teaser_story_glyphs import feature_tensor, COBALT, TEAL, CORAL, GOLD

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/generated/editorial'
DESIGN = ROOT / 'paper/design/task_story'
PROOF = DESIGN / 'proof'
HELPER = ROOT / 'paper/scripts/render_editorial_task.py'
HELPER_SHA = '2d93b0d64ee9954f73cef476c83eb4a7189db75e3006f4f79325898ed03ed495'
GLYPH = ROOT / 'paper/scripts/teaser_story_glyphs.py'
GLYPH_SHA = 'dc96d56f03423e831d8dc98b0229566373c916f4bf46ae3c42ac1a0284b3ea10'
INCLUDE = ROOT / 'paper/generated/real_video/spatial_task_figure.tex'
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
INK, GRAY, BORDER = '#26364B', '#657587', '#B9C6D2'
W, H = 396., 169.2
CAPTION = (
    r'\textbf{Recorded inputs, predicted features, withheld evaluation.} '
    r'Three observed frames (source indices 0, 5, 10), two past action blocks and '
    r'ten supplied future blocks determine the feature forecast. Frame 60 enters '
    r'only the evaluator through the same frozen DINOv2 encoder. Each block spans '
    r'five source-frame intervals, not seconds. Layered feature objects are schematic; '
    r'the heatmap is the actual three-seed, training-standardized $h=10$ patch MSE '
    r"for the prespecified median-by-episode-gain case's first window "
    r'(mean 0.1727; gain $-0.305\%$ versus autoregression), with the unchanged linear '
    r'0--0.787 scale. Recorded DROID images are unchanged (CC BY 4.0). '
    r'Neither future RGB nor robot actions are generated.'
)


def checked_inputs():
    if SHA(HELPER) != HELPER_SHA or SHA(GLYPH) != GLYPH_SHA:
        raise ValueError('Reviewed input verifier or accepted Figure 1 glyph source changed')
    spec = importlib.util.spec_from_file_location('task_story_verified_inputs', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.checked_inputs()


def theme():
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8,
                         'mathtext.fontset': 'stix', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'shiftwm-task-story-v1', 'pdf.fonttype': 42,
                         'image.composite_image': False})


def drafts():
    """Alternative information-access grammars, before final geometry."""
    DESIGN.mkdir(parents=True, exist_ok=True)
    theme()
    plans = [
        ('A  Aligned forecast / evaluator lanes',
         [(.12,.78,'history'),(.40,.78,'encode'),(.65,.78,'predict'),(.91,.78,'features'),
          (.12,.24,'withheld'),(.40,.24,'same E'),(.64,.24,'target'),(.91,.24,'score')],
         [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7),(3,7)]),
        ('B  Paired features at central comparison',
         [(.15,.82,'history + commands'),(.58,.82,'predict'),(.87,.48,'feature pair'),
          (.15,.20,'withheld RGB'),(.57,.20,'same encoder'),(.87,.13,'patch error')],
         [(0,1),(1,2),(3,4),(4,2),(2,5)]),
        ('C  Temporal strip / scoring sidecar',
         [(.12,.82,'history'),(.46,.82,'future commands'),(.86,.82,'withheld'),
          (.32,.30,'forecast features'),(.75,.30,'private score')],
         [(0,3),(1,3),(2,4),(3,4)])]
    fig, axes = plt.subplots(1,3,figsize=(12,3),facecolor='white')
    for ax,(title,nodes,edges) in zip(axes,plans):
        ax.set(xlim=(-.02,1.04),ylim=(0,1)); ax.axis('off')
        ax.text(.01,1.02,title,fontsize=9,color=INK)
        for a,b in edges:
            x,y,_=nodes[a]; xx,yy,_=nodes[b]
            ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=7,
                                        shrinkA=24,shrinkB=24,lw=.7,color=GRAY))
        for x,y,label in nodes:
            ax.add_patch(FancyBboxPatch((x-.105,y-.07),.21,.14,
                boxstyle='round,pad=0,rounding_size=.015',fc='#F5F7FA',ec=BORDER,lw=.6))
            ax.text(x,y,label,ha='center',va='center',fontsize=8,color=INK)
        if title.startswith('A'):
            ax.plot([0,1],[.50,.50],color=GRAY,lw=.6,ls='--')
    fig.tight_layout()
    fig.savefig(DESIGN/'composition_drafts.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(DESIGN/'composition_drafts.png',dpi=150)
    plt.close(fig)


def render(integrate=False):
    images,error,vmax,verified=checked_inputs()
    OUT.mkdir(parents=True,exist_ok=True); PROOF.mkdir(parents=True,exist_ok=True)
    theme()
    fig=plt.figure(figsize=(5.5,2.35),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
    texts=[];edges=[];nodes={};command_blocks=[]
    def text(x,y,label,size=8,color=INK,ha='center'):
        color={TEAL:'#087F7D',CORAL:'#B64A45',GOLD:'#9B681B'}.get(color,color)
        item=ax.text(x,y,label,ha=ha,va='center',fontsize=size,color=color,zorder=10)
        texts.append(item);return item
    def box(name,x,y,w,h,label,color=INK,frozen=False,fill='white'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2',
                     fc=fill,ec=color,lw=.7,ls='--' if frozen else '-',zorder=3))
        text(x+w/2,y+h/2,label,color=color);nodes[name]=[x,y,w,h]
    def arrow(name,points,color=GRAY):
        ax.add_patch(FancyArrowPatch(path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)),
                     arrowstyle='-|>',mutation_scale=5.5,color=color,lw=.8,zorder=4))
        edges.append({'id':name,'points':points,'color':color})
    def photo(name,x,y,w,pixels):
        h=w*pixels.shape[0]/pixels.shape[1]
        ia=fig.add_axes([x/W,y/H,w/W,h/H]);ia.imshow(pixels,interpolation='none');ia.axis('off')
        nodes[name]=[x,y,w,h]
    def tensor(name,x,y,role):
        obj=feature_tensor(ax,x,y,40,34,role);nodes[name]=list(obj['bounds']);return obj

    # One observed sequence and an explicit 2 + 10 supplied-command strip.
    text(69,153,'Observed frames',size=8.5,color=COBALT)
    for i,index in enumerate((0,5,10)):
        x=8+i*42;photo('observed-'+str(index),x,116,39,images[i]);text(x+19.5,109,str(index),color=GRAY)
    box('forecast-encoder',145,114,51,26,'Frozen\nDINOv2',COBALT,True,'#F3F6FF')
    arrow('observed-only-to-frozen-encoder',[(132,127),(144,127)],COBALT)
    box('predictor',216,114,66,26,'ShiftWM (ours)',TEAL,fill='#F1FAF8')
    arrow('encoded-history-to-predictor',[(197,127),(215,127)],COBALT)
    text(209,163,'Actions: 2 past',color=GOLD)
    text(274,163,'10 supplied future',color=GOLD)
    for i in range(12):
        past=i<2;x=215+9*i if past else 244+5.3*(i-2)
        width=6 if past else 3.8
        ax.add_patch(FancyBboxPatch((x,151),width,6,boxstyle='round,pad=0,rounding_size=.65',
                     fc=GOLD if past else '#FFF0CB',ec=GOLD,lw=.5,zorder=4))
        command_blocks.append({'index':i,'type':'past' if past else 'supplied_future','bounds':[x,151,width,6]})
    ax.plot([215,215,295.5,295.5],[149,147,147,149],color=GOLD,lw=.65,zorder=3)
    arrow('exact-two-past-ten-query-blocks-to-predictor',[(249,147),(249,141)],GOLD)
    tensor('predicted-feature-grid',341,109,'forecast')
    text(357,153,'Predicted features',size=8.5,color=TEAL)
    arrow('predictor-to-feature-forecast',[(283,127),(340,127)],TEAL)

    # The measured target enters only the enclosed evaluation region.
    ax.add_patch(FancyBboxPatch((5,6),386,87,boxstyle='round,pad=0,rounding_size=3',
                 fc='#F5F7FB',ec='#CED7E4',lw=.65,zorder=0))
    text(13,84,'Evaluation only',size=8.5,ha='left')
    text(47,73,'Withheld frame 60')
    photo('withheld-only-for-evaluation',13,29,67,images[3])
    text(47,20,'Recorded target',color=GRAY)
    box('target-encoder',101,36,55,29,'Same frozen\nDINOv2',COBALT,True)
    arrow('withheld-frame-to-same-frozen-encoder',[(81,50.5),(100,50.5)],COBALT)
    tensor('target-feature-grid',196,33,'observed')
    text(216,74,'Target features',color=COBALT)
    text(216,23,r'$Z_{10}$',color=COBALT)
    arrow('target-encoder-to-target-features',[(157,50.5),(195,50.5)],COBALT)
    ax.add_patch(Circle((273,50.5),12,fc='white',ec=CORAL,lw=.85,zorder=4))
    text(273,50.5,'MSE');nodes['mse']=[261,38.5,24,24]
    arrow('target-features-to-mse',[(237,50.5),(260,50.5)],COBALT)
    arrow('prediction-down-to-evaluation-only',[(361,108),(361,99),(273,99),(273,63.5)],TEAL)
    text(310,91,r'$\widehat Z_{10}$',color=TEAL)
    error_ax=fig.add_axes([319/W,34/H,33/W,33/H])
    error_ax.imshow(error,cmap='magma',norm=Normalize(0,vmax),interpolation='nearest')
    error_ax.set_xticks([]);error_ax.set_yticks([])
    for spine in error_ax.spines.values():spine.set_color(GRAY);spine.set_linewidth(.5)
    text(336,74,'Patch error')
    text(336,22,'Lower is better',color=GRAY)
    arrow('mse-to-measured-patch-error',[(286,50.5),(318,50.5)],CORAL)
    cbax=fig.add_axes([360/W,34/H,4/W,33/H])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,vmax),cmap='magma'),cax=cbax,ticks=[0,vmax])
    cb.ax.set_yticklabels(['0',f'{vmax:.3f}']);cb.ax.tick_params(labelsize=8,length=2,pad=2);cb.outline.set_linewidth(.4)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[t.get_window_extent(renderer) for t in texts]
    overlaps=[(texts[i].get_text(),texts[j].get_text()) for i,a in enumerate(bounds)
              for j,b in enumerate(bounds) if j>i and a.overlaps(b)]
    clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height]
    if overlaps or clipped:raise ValueError(json.dumps({'overlaps':overlaps,'clipped':clipped}))
    if len(command_blocks)!=12 or sum(b['type']=='past' for b in command_blocks)!=2:
        raise ValueError('Wrong command block counts')
    for ext in ('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else None)
        fig.savefig(OUT/f'task_story.{ext}',dpi=300,metadata=metadata)
    plt.close(fig)
    (OUT/'task_story_caption.tex').write_text(CAPTION+'\n')
    include=('\\begin{figure}[!htb]\n\\centering\n'
             '\\includegraphics[width=\\linewidth]{generated/editorial/task_story.pdf}\n'
             '\\caption[Recorded inputs and evaluator-only target.]{'+CAPTION+'}\n'
             '\\label{fig:spatial-task}\n\\end{figure}\n')
    (OUT/'task_story_figure.tex').write_text(include)
    if integrate:INCLUDE.write_text(include)
    evidence={'status':'numeric_and_geometry_passed_actual_pixel_review_pending',
        'renderer_sha256':SHA(__file__),'helper_sha256':SHA(HELPER),'glyph_sha256':SHA(GLYPH),
        'verified_inputs':verified,'geometry':{'inches':[5.5,2.35],'minimum_font_pt':8,'maximum_heading_pt':8.5,
            'text_overlaps':overlaps,'clipped':clipped},'edges':edges,'nodes':nodes,'command_blocks':command_blocks,
        'error_scale':{'mapping':'linear','limits':[0,vmax],'colormap':'magma'},
        'illustrative_only':'Colored feature glyphs encode 16 spatial positions and channel depth, not measured feature values or RGB. Three drawn planes do not imply three channels.',
        'outputs':{str((OUT/f'task_story.{ext}').relative_to(ROOT)):SHA(OUT/f'task_story.{ext}') for ext in ('pdf','svg','png')}}
    (OUT/'task_story_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'shape':[5.5,2.35],'mse':verified['error_mean'],'gain_percent':verified['gain_vs_autoregression_percent'],
                      'overlaps':overlaps,'clipped':clipped,'integrated':integrate}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drafts-only',action='store_true');parser.add_argument('--integrate',action='store_true')
    args=parser.parse_args()
    drafts() if args.drafts_only else render(args.integrate)
