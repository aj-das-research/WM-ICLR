#!/usr/bin/env python3
"""Refined paper-native spatial architecture; exact frozen scientific graph."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MP
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/editorial'
DESIGN=ROOT/'paper/design/spatial_architecture_refined'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED={
 'src/shiftwm/real_video_spatial/model.py':'054d0aa1c479c1dc722b674bb59dc35015c69ebacec0f0602b84ded40651edf2',
 'src/shiftwm/model.py':'2f783caf8eed699a8d024f689aa956efd126a2c8b6c67aea23e4ff5dfb8536f8',
 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_0.png':'192e1c0c55ee427dd158c29fb900a609b63c8caf06ed2e574ac5c5a5d9d4e5a6',
 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_5.png':'779ab79d598151047151bd908e46bcde0f94cd6937244dd77dedb1a56e4fbd24',
 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_10.png':'344e1956f8ec56f118afe46fa42e71ed05ec25a0dfc45f450bb3f8809fc5ce5d'}
STYLE=json.loads((ROOT/'paper/design/editorial_style.json').read_text())
INK='#243447'; BLUE='#4477AA'; ORANGE='#B88746'; GREEN='#287C7A'; GRAY='#7C8892'
BORDER='#B9C3CC'; NEUTRAL='#F7F9FA'; SY=216/237.6
CAPTION=r'''\textbf{ShiftWM (ours): fixed spatial anchoring with bounded innovation.} (a) Three frozen DINOv2 grids enter spatial encoding. Patch-mean past transitions infer context for FiLM; a chronological past-plus-query-prefix GRU supplies LeWM, without actions beyond horizon $h$. Learned modules after DINOv2 train offline. (b) $Z_0$ is the last observed $16\!\times\!384$ grid in shared per-channel training-normalized coordinates; $E_0$ is its spatial encoding before FiLM. Both $H_h$ labels denote the same horizon state. The fixed $4\!\times\!4$ anchor is mixed by a $16\!\times\!16$ row-stochastic matrix; a per-patch sigmoid gate blends it, and the projected innovation $\Delta_h=\tanh R(H_h)$ is bounded elementwise by one. Here $R$ is layer normalization followed by an affine projection. Grids and weights are schematic; DROID images are unchanged observations (CC BY 4.0). Outputs are features, not RGB; Appendix~\ref{app:spatial-development} gives details.'''

def sketches():
    DESIGN.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8})
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    versions=[('A · Two stacked computation rails',[(.12,.77,'observed'),(.47,.77,'causal state'),(.81,.77,'H'),(.12,.26,'anchor'),(.47,.26,'mix + gate'),(.81,.26,'forecast')],[(0,1),(1,2),(3,4),(4,5),(2,4)]),
      ('B · Three narrow stages',[(.14,.77,'images'),(.14,.26,'commands'),(.49,.61,'context'),(.49,.24,'prefix + state'),(.83,.72,'anchor mixer'),(.83,.27,'bounded output')],[(0,2),(1,3),(2,3),(3,4),(4,5)]),
      ('C · Aligned state and decoder panels',[(.2,.77,'observations'),(.2,.46,'causal state'),(.2,.15,'H'),(.68,.77,'queries / keys'),(.68,.46,'anchor mixing'),(.68,.15,'blend + bound')],[(0,1),(1,2),(3,4),(4,5)])]
    for ax,(title,nodes,edges) in zip(axes,versions):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(0,1,title,fontsize=9,va='top')
        for a,b in edges:
            x,y,_=nodes[a];xx,yy,_=nodes[b];ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',shrinkA=21,shrinkB=22,lw=.65,color=GRAY,mutation_scale=7))
        for x,y,label in nodes:
            ax.add_patch(Rectangle((x-.12,y-.065),.24,.13,fc=NEUTRAL,ec=BORDER,lw=.65));ax.text(x,y,label,ha='center',va='center',fontsize=8)
    fig.tight_layout();fig.savefig(DESIGN/'composition_drafts.pdf');fig.savefig(DESIGN/'composition_drafts.png',dpi=140);plt.close(fig)

def render():
    for name,digest in EXPECTED.items():
        if sha(ROOT/name)!=digest:raise ValueError('Scientific source or image changed: '+name)
    OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8.0,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'stix'})
    W,H=396,216;fig=plt.figure(figsize=(5.5,3.0),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
    texts=[];edges=[];nodes={}
    def text(x,y,s,size=8.0,color=INK,ha='center',weight='normal'):
        obj=ax.text(x,y*SY,s,ha=ha,va='center',fontsize=size,color=color,weight=weight,zorder=10);texts.append(obj);return obj
    def box(name,x,y,w,h,label=None,ec=GRAY,fc='white',size=8.0,dashed=False):
        conventional=name in {'spatial-encoder','past-context','support-film','action-prefix-gru','lewm-temporal'}
        face=NEUTRAL if conventional else fc
        edge=BORDER if conventional else ec
        ax.add_patch(FancyBboxPatch((x,y*SY),w,h*SY,boxstyle='round,pad=0,rounding_size=1.2',fc=face,ec=edge,lw=.6,ls='--' if dashed else '-',zorder=3));nodes[name]=[x,y*SY,w,h*SY]
        if label:text(x+w/2,y+h/2,label,size=size,color=INK if conventional else ec)
    def arrow(name,points,color=GRAY):
        points=[(x,y*SY) for x,y in points]
        ax.add_patch(FancyArrowPatch(path=MP(points,[MP.MOVETO]+[MP.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=4.8,color=color,lw=.65,zorder=4));edges.append({'id':name,'points':points,'color':color})
    palette=np.array([matplotlib.colors.to_rgb(v) for v in ['#A5BCCF','#D2DDE5','#92B8B5','#C3D7D4','#BECEDB','#6C93B2','#B0C9C6','#DCE6E4','#88A9C2','#C2D0DD','#729F9C','#ADC9C6','#D3DFE8','#A8BFCD','#C9DCDA','#8FB4B2']])
    T=.75*np.eye(16)+.2*np.roll(np.eye(16),1,axis=1)+.05*np.ones((16,16))/16
    def grid(name,x,y,s,mixed=False):
        colors=T@palette if mixed else palette
        y=(y+s/2)*SY-s/2
        for r in range(4):
            for c in range(4):ax.add_patch(Rectangle((x+c*s/4,y+(3-r)*s/4),s/4,s/4,fc=colors[4*r+c],ec='white',lw=.35,zorder=5))
        ax.add_patch(Rectangle((x,y),s,s,fc='none',ec=GRAY,lw=.5,zorder=6));ax.add_patch(Rectangle((x+s/4,y+s/2),s/4,s/4,fc='none',ec=INK,lw=.65,zorder=7));nodes[name]=[x,y,s,s]
    def operator(name,x,y,symbol,color=GREEN):
        ax.add_patch(Circle((x,y*SY),5.5,fc='white',ec=color,lw=.65,zorder=5));text(x,y,symbol,size=9,color=color);nodes[name]=[x-5.5,y*SY-5.5,11,11]
    text(8,226,'a  Causal state',size=8.7,ha='left')
    text(184,226,'b  Fixed-anchor forecast',size=8.7,ha='left')
    ax.plot([176,176],[6*SY,218*SY],color='#E0E5E8',lw=.5)
    text(53,210,'Observed images',size=8.0,color=BLUE)
    for j,frame in enumerate((0,5,10)):
        p=ROOT/f'paper/figure_sources/spatial_qualitative/case1_recorded_frame_{frame}.png';pixels=np.asarray(Image.open(p));x=10+31*j;y=183
        ia=fig.add_axes([x/W,y*SY/H,28/W,15.75/H]);ia.imshow(pixels,interpolation='nearest');ia.set_axis_off()
    box('frozen-dino',113,180,52,31,'Frozen\nDINOv2',ec=BLUE,fc='#F3F8FC',dashed=True)
    arrow('observed-rgb-to-frozen-dino',[(101,194),(112,194)],BLUE)
    for j in range(3):grid('observed-grid-'+str(j),119+15*j,160,12)
    arrow('frozen-dino-to-normalized-patch-grids',[(139,179),(139,173)],BLUE)
    text(139,153,r'$Z_{-2},Z_{-1},Z_0$',size=8,color=BLUE)
    box('spatial-encoder',103,119,63,25,'Spatial\nencoder',ec=GREEN,fc='#F5F9F5',size=8.0)
    arrow('support-grids-to-spatial-encoder',[(139,148),(139,145)],BLUE)
    box('past-actions',12,156,63,19,r'Past $a_{-1:0}$',ec=ORANGE,size=8.0)
    box('past-context',11,96,64,26,'Past-only\ncontext',ec=GREEN,fc='#F5F9F5')
    arrow('past-actions-to-transition-context',[(29,155),(29,123)],ORANGE)
    arrow('mean-encoded-support-to-transition-context',[(102,133),(88,133),(88,109),(76,109)],BLUE)
    text(86,146,'Patch\nmean',size=8,color=BLUE)
    box('support-film',104,82,62,27,'FiLM',ec=GREEN,fc='#F5F9F5')
    arrow('encoded-support-to-film',[(135,118),(135,110)],BLUE)
    arrow('context-to-film',[(75,101),(91,101),(91,95),(103,95)],GREEN)
    box('future-action-prefix',12,75,63,17,r'Prefix $a_{1:h}$',ec=ORANGE,size=8.0)
    box('action-prefix-gru',11,40,64,26,'Prefix GRU',ec=GREEN,fc='#F5F9F5')
    arrow('past-actions-to-causal-gru',[(12,164),(5,164),(5,72),(26,72),(26,67)],ORANGE)
    arrow('future-prefix-to-causal-gru',[(49,74),(49,67)],ORANGE)
    box('lewm-temporal',103,40,63,26,'LeWM',ec=GREEN,fc='#F5F9F5')
    arrow('conditioned-support-to-lewm',[(135,81),(135,67)],BLUE)
    arrow('causal-action-states-to-lewm',[(76,53),(102,53)],ORANGE)
    arrow('lewm-to-horizon-state',[(135,39),(135,29)],GREEN)
    text(136,21,r'$H_h$: 16 × 96',size=8.0,color=GREEN)
    text(47,21,'Dashed: frozen',size=8,color=GRAY)
    # Named tensor ports in panel b refer to the same tensors constructed in a.
    box('horizon-state-port',195,193,31,18,r'$H_h$',ec=GREEN,fc='#F5F9F5',size=8.5)
    box('pre-film-key-port',319,193,31,18,r'$E_0$',ec=BLUE,fc='#F3F8FC',size=8.5)
    text(351,201,'pre-FiLM',size=8,color=BLUE,ha='left')
    text(219,181,r'$Q_h=W_QH_h$',size=8,color=GREEN)
    text(335,181,r'$K_0=W_KE_0$',size=8,color=BLUE)
    arrow('horizon-state-to-query',[(210,192),(210,188)],GREEN)
    arrow('pre-film-observation-to-key',[(334,192),(334,188)],BLUE)
    box('row-softmax',184,147,203,22,ec=GREEN,fc='#F5F9F5')
    text(285.5,158,r'$T_h=\mathrm{softmax}_{\rm row}(Q_hK_0^\top/\sqrt{96}+4I)$',size=8,color=GREEN)
    arrow('queries-to-transport-scores',[(217,175),(217,170)],GREEN)
    arrow('keys-to-transport-scores',[(335,175),(335,170)],BLUE)
    mx,my,ms=230,121*SY-16,32
    for r in range(16):
        for c in range(16):
            shade=1-.78*T[r,c]/T.max();ax.add_patch(Rectangle((mx+c*ms/16,my+(15-r)*ms/16),ms/16,ms/16,fc=(shade,shade,shade),ec='white',lw=.05,zorder=4))
    ax.add_patch(Rectangle((mx,my),ms,ms,fc='none',ec=GRAY,lw=.5,zorder=5))
    ax.add_patch(Rectangle((mx,my+10*ms/16),ms,ms/16,fc='none',ec=GREEN,lw=.55,zorder=6))
    arrow('softmax-to-row-stochastic-matrix',[(246,146),(246,140)],GREEN)
    text(276,130,r'$T_h$',size=8.5,color=GREEN)
    grid('fixed-last-observed-anchor',185,72,26);text(198,117,'Fixed',size=8,color=BLUE);text(198,106,r'$Z_0$',size=8.5,color=BLUE)
    operator('matrix-times-anchor',246,85,'×')
    arrow('fixed-anchor-to-feature-product',[(212,85),(240,85)],BLUE)
    arrow('transport-to-feature-product',[(246,102),(246,92)],GREEN)
    grid('mixed-anchor',266,72,26,True);arrow('product-to-mixed-anchor',[(252,85),(265,85)],GREEN)
    text(309,85,r'$T_hZ_0$',size=8,color=GREEN)
    box('patch-sigmoid-gate',294,112,93,27,ec=GREEN,fc='#F5F9F5')
    text(340.5,132,'Patch gate',size=8,color=GREEN)
    text(340.5,120,r'$g_h=\sigma(W_gH_h+b_g)$',size=8,color=GREEN)
    arrow('horizon-state-to-gate',[(227,202),(232,202),(232,214),(392,214),(392,125.5),(388,125.5)],GREEN)
    box('complementary-anchor-blend',222,34,116,24,ec=GREEN,fc='#F5F9F5')
    text(280,46,r'$(1-g_h)Z_0+g_h(T_hZ_0)$',size=8,color=GREEN)
    arrow('identity-anchor-to-complementary-blend',[(198,71),(198,46),(221,46)],BLUE)
    arrow('mixed-anchor-to-complementary-blend',[(278,71),(278,59)],GREEN)
    arrow('gate-to-complementary-blend',[(340,111),(340,68),(326,68),(326,59)],GREEN)
    operator('add-bounded-innovation',350,46,'+',ORANGE)
    arrow('blend-to-addition',[(339,46),(344,46)],GREEN)
    box('bounded-innovation',221,5,157,18,ec=ORANGE,fc='#FCF9F4')
    text(274,14,r'$\Delta_h=\tanh R(H_h)$',size=8,color=INK)
    text(348,14,r'$|\Delta_h|\leq1$',size=8,color=ORANGE)
    arrow('horizon-state-to-projected-innovation',[(392,125.5),(392,14),(379,14)],GREEN)
    ax.add_patch(Circle((392,125.5*SY),.8,fc=GREEN,ec='none',zorder=5))
    arrow('bounded-innovation-to-addition',[(309,24),(309,29),(350,29),(350,39)],ORANGE)
    grid('future-feature-forecast',365,35,23,True)
    text(376.5,68,r'$\hat Z_h$',size=8.5,color=GREEN)
    arrow('addition-to-future-features',[(356,46),(364,46)],GREEN)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[t.get_window_extent(renderer) for t in texts]
    overlaps=[]
    for i,a in enumerate(bounds):
        for j,b in enumerate(bounds[i+1:],i+1):
            if a.overlaps(b):overlaps.append([texts[i].get_text(),texts[j].get_text()])
    clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.x1>fig.bbox.width or b.y0<0 or b.y1>fig.bbox.height]
    evidence={'status':'candidate_pending_actual_pixel_review','sources_sha256':EXPECTED,'renderer_sha256':sha(__file__),'style_sha256':sha(ROOT/'paper/design/editorial_style.json'),'local_style':{'font_family':'Liberation Sans','math':'STIX','palette':{'ink':INK,'observation':BLUE,'method':GREEN,'action_innovation':ORANGE},'headings_pt':8.7},'geometry':{'width_inches':5.5,'height_inches':3.0,'minimum_font_pt':8,'text_overlaps':overlaps,'clipped':clipped},'arrows':edges,'nodes':nodes,'schematic_transport_row_sum_max_error':float(np.max(np.abs(T.sum(1)-1))),'semantic_contract':'Fixed normalized Z0; pre-FiLM E0 keys; horizon Hh queries; row-softmax(QK/sqrt96+4I); sigmoid affine gate; complementary anchor mixture plus tanh-projected innovation. Future RGB never enters predictor. Repeated named tensors are aliases, not new inputs.','image_attribution':'Three unchanged DROID observations, CC BY 4.0 https://droid-dataset.github.io/','limitations':'Schematic patch colors and weights; no measured attribution, generated RGB, performance claim, CEM or paired-simulator supervision.'}
    (OUT/'spatial_architecture_refined_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'spatial_architecture_refined.{ext}',dpi=300)
    fig.savefig(DESIGN/'paper_size.png',dpi=110);plt.close(fig)
    ImageOps.grayscale(Image.open(DESIGN/'paper_size.png')).save(DESIGN/'grayscale.png')
    (OUT/'spatial_architecture_refined_caption.tex').write_text(CAPTION+'\n')
    (OUT/'spatial_architecture_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/spatial_architecture_refined.pdf}\n\\caption[Spatial anchored world model.]{'+CAPTION+'}\n\\label{fig:editorial-spatial-method}\n\\end{figure}\n')
    print(json.dumps({'layout_issues':{'overlaps':overlaps,'clipped':clipped},'size_inches':[5.5,3.0]}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--drafts-only',action='store_true');args=parser.parse_args()
    if args.drafts_only:sketches()
    else:render()
