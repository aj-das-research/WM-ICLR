#!/usr/bin/env python3
"""Compact method-family interface and decoder overview; no performance data."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.path import Path as MP
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/editorial'
DESIGN=ROOT/'paper/design/method_family'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED_SOURCES = {'src/shiftwm/model.py': '2f783caf8eed699a8d024f689aa956efd126a2c8b6c67aea23e4ff5dfb8536f8', 'src/shiftwm/real_video_spatial/model.py': '054d0aa1c479c1dc722b674bb59dc35015c69ebacec0f0602b84ded40651edf2', 'src/shiftwm/real_video_spatial_components/model.py': '410851c1e138a2b367f1ef2a5e72c5ee57a794a659f9b452e0c99e7039c1dd67', 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_0.png': '192e1c0c55ee427dd158c29fb900a609b63c8caf06ed2e574ac5c5a5d9d4e5a6', 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_5.png': '779ab79d598151047151bd908e46bcde0f94cd6937244dd77dedb1a56e4fbd24', 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_10.png': '344e1956f8ec56f118afe46fa42e71ed05ec25a0dfc45f450bb3f8809fc5ce5d'}
CAPTION = '\\textbf{Two ways to use observed evidence.} Ours-1 calibrates history with $c_o$, then infers $c_d$ from corrected transitions and past actions to condition LeWM. Ours-2--5 build a spatial state with past-only transition context and retain the last observed grid $Z_0$. Columns toggle gated row-stochastic feature mixing; rows toggle innovation bounding. Each cell specifies the future-feature forecast $\\hat Z_h$. Recorded DROID images are unchanged (CC BY 4.0); latent colors are schematic. Shared interfaces do not imply shared weights, representations, or evaluation protocols. Detailed architectures and simulator-only planning appear in Appendices~\\ref{app:spatial-development} and~\\ref{app:original-context-study}.'
STYLE=json.loads((ROOT/'paper/design/editorial_style.json').read_text())
INK=STYLE['colors']['ink']; BLUE=STYLE['colors']['observation']; ORANGE=STYLE['colors']['action'];GREEN=STYLE['colors']['ours'];GRAY=STYLE['colors']['secondary']

def sketches():
    fig,axs=plt.subplots(1,3,figsize=(12,3.0))
    definitions=[('A · Unequal family lanes',[(.09,.53,'images\nactions'),(.42,.8,'context → rollout'),(.39,.40,'fixed anchor'),(.76,.41,'2 × 2\ndecoder variants')],[(0,1),(0,2),(2,3)]),
    ('B · Forked vertical narrative',[(.5,.86,'images + actions'),(.24,.53,'context branch'),(.75,.53,'spatial branch'),(.24,.18,'rollout features'),(.75,.18,'decoder matrix')],[(0,1),(0,2),(1,3),(2,4)]),
    ('C · Radial mechanism atlas',[(.5,.52,'observed evidence'),(.15,.83,'Ours-1'),(.82,.83,'Ours-2'),(.15,.2,'Ours-3'),(.50,.12,'Ours-4'),(.85,.2,'Ours-5')],[(0,1),(0,2),(0,3),(0,4),(0,5)])]
    for ax,(title,nodes,edges) in zip(axs,definitions):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(0,1,title,fontsize=10,weight='bold',va='top')
        for i,j in edges:
            x,y,_=nodes[i];xx,yy,_=nodes[j];ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',lw=1,color=GRAY,mutation_scale=9,shrinkA=25,shrinkB=28))
        for x,y,label in nodes:
            ax.add_patch(FancyBboxPatch((x-.13,y-.095),.26,.19,boxstyle='round,pad=.02',fc='#F2F5F7',ec=GRAY,lw=.7));ax.text(x,y,label,ha='center',va='center',fontsize=8.3)
    fig.tight_layout();fig.savefig(DESIGN/'composition_drafts.png',dpi=160);fig.savefig(DESIGN/'composition_drafts.pdf');plt.close(fig)


def render():
    for name,digest in EXPECTED_SOURCES.items():
        if sha(ROOT/name)!=digest:raise ValueError('Reviewed source or observed image changed: '+name)
    OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    sketches()
    fig=plt.figure(figsize=(5.5,3.4),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,396),ylim=(0,244.8));ax.axis('off')
    texts=[];edges=[];nodes={}
    def text(x,y,s,size=8.5,color=INK,ha='center',weight='normal'):
        t=ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight,zorder=10);texts.append(t);return t
    def box(name,x,y,w,h,fc='#F4F7F9',ec='#B5C1CB',r=3):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=.65,zorder=1));nodes[name]=[x,y,w,h]
    def arrow(name,points,color=GRAY,lw=.85):
        ax.add_patch(FancyArrowPatch(path=MP(points,[MP.MOVETO]+[MP.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=6,lw=lw,color=color,zorder=4));edges.append({'id':name,'points':points,'color':color})
    def bridge_arrow(name,x0,x1,y,crossings,color=BLUE):
        vertices=[(x0,y)];codes=[MP.MOVETO]
        for x in crossings:
            vertices.extend([(x-2,y),(x-2,y+2.8),(x+2,y+2.8),(x+2,y)])
            codes.extend([MP.LINETO,MP.CURVE4,MP.CURVE4,MP.CURVE4])
        vertices.append((x1,y));codes.append(MP.LINETO)
        ax.add_patch(FancyArrowPatch(path=MP(vertices,codes),arrowstyle='-|>',mutation_scale=6,lw=.85,color=color,zorder=7))
        edges.append({'id':name,'path_vertices':vertices,'path_codes':[int(code) for code in codes],'color':color,'semantics':'Continuous bridge over independent action wires; no junction.'})
    palette=np.array([[95,118,173],[124,167,184],[148,190,188],[110,162,152],[153,166,196],[177,157,187],[200,173,169],[203,188,153],[136,164,174],[145,181,167],[184,195,172],[202,199,172],[125,146,177],[156,170,183],[184,173,190],[162,162,185]],float)/255
    blend=.65*palette+.35*np.roll(palette,1,axis=0)
    def grid(x,y,s,mixed=False):
        colors=blend if mixed else palette
        for row in range(4):
            for col in range(4):
                ax.add_patch(Rectangle((x+col*s/4,y+(3-row)*s/4),s/4,s/4,fc=colors[row*4+col],ec='white',lw=.25,zorder=5))
        ax.add_patch(Rectangle((x,y),s,s,fc='none',ec=GRAY,lw=.6,zorder=6))
        ax.add_patch(Rectangle((x+s/4,y+s/2),s/4,s/4,fc='none',ec=INK,lw=.9,zorder=7))
    def bars(x,y,w=26,h=15,color=BLUE):
        for j,f in enumerate((.52,.86,.38,.69)):
            ax.add_patch(Rectangle((x+j*w/4,y),w/6,h*f,fc=color,alpha=.45+j*.1,ec='none',zorder=5))
    # Input strip: all RGB frames are actual observed support and untouched.
    text(32,237,'Observed',size=8.5,weight='bold')
    sources=[]
    for j,frame in enumerate((0,5,10)):
        p=ROOT/f'paper/figure_sources/spatial_qualitative/case1_recorded_frame_{frame}.png';sources.append(str(p.relative_to(ROOT)))
        pixels=np.asarray(Image.open(p));y=201-j*30
        ia=fig.add_axes([7/396,y/244.8,50/396,28.125/244.8]);ia.imshow(pixels);ia.set_axis_off()
    box('encoder',10,115,44,19,fc='#F4F8FB',ec=BLUE);text(32,124.5,r'Frozen $E$')
    arrow('images-to-encoder',[(32,140),(32,135)],BLUE)
    bars(20,91,24,14);arrow('encoder-to-features',[(32,114),(32,108)],BLUE)
    text(32,84,'Features',size=8,color=BLUE)
    text(32,63,'Actions',weight='bold',color=ORANGE)
    for row,y in enumerate((44,28)):
        for j in range(4):ax.add_patch(Rectangle((10+j*5,y),3.5,6,fc='#EDCEB2',ec=ORANGE,lw=.35,zorder=5))
        text(37,y+3,r'$a_S$' if row==0 else r'$a_Q$',ha='left',size=8.5,color=ORANGE)
    # Family boundaries and typed input buses.
    box('original-family',85,157,304,82,fc='#F7F9FB',ec='#D4DFE8')
    box('spatial-family',85,6,304,143,fc='#F8FBF8',ec='#D4E1D7')
    text(95,231,'Ours-1 · Support contexts',size=9.5,ha='left',weight='bold')
    text(95,139,'Ours-2–5 · Fixed observed anchor',size=9.5,ha='left',weight='bold',color=GREEN)
    arrow('features-to-original',[(48,98),(65,98),(65,203),(92,203)],BLUE)
    bridge_arrow('features-to-spatial',65,92,120,[72,78],BLUE)
    ax.add_patch(Circle((65,98),1.1,fc=BLUE,ec='none',zorder=5))
    # History calibration contains c_o inference from support statistics.
    box('observation-calibration',95,187,69,34,fc='white',ec=BLUE)
    text(129.5,212,r'$c_o$: calibrate',size=8.5,color=BLUE)
    bars(102,193,18,11,color=GRAY);text(128,198,r'$\times\!+$',size=9,color=BLUE);bars(140,193,17,11,color=BLUE)
    box('dynamics-context',181,187,65,34,fc='white',ec=ORANGE)
    text(213.5,211,'Infer dynamics',size=8.0,color=ORANGE);text(213.5,197,r'$c_d$',size=12,color=ORANGE)
    arrow('corrected-support-to-context',[(164,203),(180,203)],BLUE)
    box('context-rollout',269,187,63,34,fc='white',ec=GRAY)
    text(300.5,211,'Adapt actions',size=8);text(300.5,197,'LeWM rollout',size=8.5)
    arrow('dynamics-to-rollout',[(246,203),(268,203)],ORANGE)
    arrow('corrected-history-to-rollout',[(158,221),(158,225),(300,225),(300,222)],BLUE)
    arrow('past-actions-to-context',[(56,47),(72,47),(72,173),(212,173),(212,186)],ORANGE)
    arrow('query-actions-to-rollout',[(56,31),(78,31),(78,166),(300,166),(300,186)],ORANGE)
    arrow('past-actions-to-rollout',[(212,173),(286,173),(286,186)],ORANGE)
    ax.add_patch(Circle((212,173),1.0,fc=ORANGE,ec='none',zorder=6))
    text(240,180,r'$a_S$',size=8,color=ORANGE);text(312,176,r'$a_Q$',size=8,color=ORANGE)
    bars(357,195,24,20);arrow('original-feature-output',[(332,203),(352,203)],BLUE)
    text(368,181,'Forecast',size=8,color=BLUE)
    # Spatial state, exact support-only context distinct from the original pair.
    box('spatial-state',95,94,77,33,fc='white',ec=GREEN)
    text(133.5,116,'Causal state',size=8.5);text(133.5,102,'one context',size=8,color=GREEN)
    arrow('actions-to-spatial',[(72,47),(72,103),(94,103)],ORANGE)
    arrow('query-prefix-to-spatial',[(78,31),(78,96),(94,96)],ORANGE)
    text(183,101,r'$H_h$',size=9,color=GREEN)
    # A single last-observed anchor, visibly repeated across all decoders.
    grid(103,50,34);text(120,42,r'$Z_0$',size=9,color=BLUE)
    text(120,31,'Last observed',size=8,color=BLUE)
    ax.plot([65,65],[98,68],color=BLUE,lw=.85,zorder=4)
    bridge_arrow('support-to-anchor',65,100,68,[72,78],BLUE)
    # One decoder-family boundary receives the common state and fixed anchor.
    # Cells denote separately trained alternatives, not a four-output ensemble.
    box('decoder-family',195,31,191,99,fc='#F2F7F2',ec='#BACDBF',r=3)
    text(240,122,'Anchor',size=8.5,weight='bold')
    text(338,122,'Mix + gate',size=8.5,weight='bold',color=GREEN)
    for row,(y,bounded) in enumerate(((76,False),(34,True))):
        for col,(x,mixed,number) in enumerate(((198,False,2+row),(293,True,4+row))):
            box(f'decoder-{number}',x,y,89,39,fc='white',ec='#CCD8D0')
            text(x+6,y+30,f'Ours-{number}',ha='left',size=8.5,weight='bold',color=GREEN)
            grid(x+6,y+5,17,mixed)
            text(x+33,y+13,'+',size=11,color=ORANGE)
            text(x+61,y+13,r'$\tanh R_h$' if bounded else r'$R_h$',size=8.5,color=ORANGE)
            text(x+80,y+30,r'$\hat Z_h$',size=8.0,color=BLUE)
            if bounded:
                curve=np.linspace(-2,2,40);ax.plot(x+54+(curve+2)*3.2,y+29+3*np.tanh(curve),color=ORANGE,lw=.85,zorder=6)
            else:
                ax.plot([x+54,x+67],[y+26,y+32],color=ORANGE,lw=.85,zorder=6)
    arrow('state-to-decoder-family',[(172,111),(194,111)],GREEN)
    arrow('anchor-to-decoder-family',[(137,67),(194,67)],BLUE)
    text(173,60,r'$Z_0$',size=8,color=BLUE)
    text(242,17,r'Mix: $(1-g_h)Z_0+g_h(T_hZ_0)$',size=8.5,color=GREEN)
    # Output and primitive definitions delegated to the compact caption.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[t.get_window_extent(renderer) for t in texts]
    overlap=[]
    for i,a in enumerate(bounds):
        for j,b in enumerate(bounds[i+1:],i+1):
            if a.overlaps(b):overlap.append([texts[i].get_text(),texts[j].get_text()])
    clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.x1>fig.bbox.width or b.y0<0 or b.y1>fig.bbox.height]
    if overlap or clipped:raise ValueError(json.dumps({'overlap':overlap,'clipped':clipped}))
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'method_family.{ext}',dpi=300)
    fig.savefig(OUT/'method_family_paper_size.png',dpi=110);fig.savefig(OUT/'method_family_enlarged.png',dpi=360);plt.close(fig)
    ImageOps.grayscale(Image.open(OUT/'method_family_paper_size.png')).save(OUT/'method_family_grayscale.png')
    source_files=['src/shiftwm/model.py','src/shiftwm/real_video_spatial/model.py','src/shiftwm/real_video_spatial_components/model.py','paper/figure_sources/spatial_method/canonical-graph.json','paper/design/editorial_style.json']+sources
    evidence={'status':'candidate_requires_pixel_review','sources':{p:sha(ROOT/p) for p in source_files},'renderer_sha256':sha(__file__),'width_inches':5.5,'height_inches':3.4,'minimum_font_pt':8,'text_overlaps':overlap,'clipped_text':clipped,'arrows':edges,'nodes':nodes,'image_provenance':'Unchanged DROID support frames, CC BY4.0; all feature glyphs and decoder transformations schematic.'}
    (OUT/'method_family_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (OUT/'method_family_caption.tex').write_text(CAPTION+'\n')
    (OUT/'method_family_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/method_family.pdf}\n\\caption[Method family overview.]{'+CAPTION+'}\n\\label{fig:method-family}\n\\end{figure}\n')
    print(json.dumps({'overlap':overlap,'clipped':clipped}))

if __name__=='__main__':render()
