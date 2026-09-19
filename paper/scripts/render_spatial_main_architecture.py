#!/usr/bin/env python3
"""Compact spatial-only architecture, exact graph and schematic latent marks."""
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
DESIGN=ROOT/'paper/design/spatial_main_architecture'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED={
 'src/shiftwm/real_video_spatial/model.py':'054d0aa1c479c1dc722b674bb59dc35015c69ebacec0f0602b84ded40651edf2',
 'src/shiftwm/model.py':'2f783caf8eed699a8d024f689aa956efd126a2c8b6c67aea23e4ff5dfb8536f8',
 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_0.png':'192e1c0c55ee427dd158c29fb900a609b63c8caf06ed2e574ac5c5a5d9d4e5a6',
 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_5.png':'779ab79d598151047151bd908e46bcde0f94cd6937244dd77dedb1a56e4fbd24',
 'paper/figure_sources/spatial_qualitative/case1_recorded_frame_10.png':'344e1956f8ec56f118afe46fa42e71ed05ec25a0dfc45f450bb3f8809fc5ce5d'}
STYLE=json.loads((ROOT/'paper/design/editorial_style.json').read_text())
INK=STYLE['colors']['ink']; BLUE=STYLE['colors']['observation']; ORANGE=STYLE['colors']['action'];GREEN=STYLE['colors']['ours'];GRAY=STYLE['colors']['secondary']
CAPTION=r'''\textbf{ShiftWM (ours): fixed spatial anchoring with bounded innovation.} (a) Three frozen DINOv2 grids enter spatial encoding. Patch-mean past transitions infer context for FiLM; a causal action-prefix GRU supplies LeWM. Learned modules after DINOv2 train offline. (b) $Z_0$ is the last observed $16\!\times\!384$ grid in shared per-channel training-normalized coordinates; $E_0$ is its spatial encoding before FiLM. Both $H_h$ labels denote the same horizon state. A $16\!\times\!16$ row-stochastic matrix mixes the fixed anchor, a per-patch sigmoid gate blends it, and a tanh-bounded projected innovation completes the forecast. Grids and weights are schematic; DROID images are unchanged observations (CC BY 4.0). Outputs are features, not RGB; Appendix~\ref{app:spatial-development} gives details.'''

def sketches():
    DESIGN.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    versions=[('A · Causal state | anchored decoder',[(.19,.78,'observations'),(.19,.45,'causal state'),(.19,.13,'H'),(.64,.76,'keys + queries'),(.64,.42,'anchor × weights'),(.85,.13,'blend + bound')],[(0,1),(1,2),(2,3),(3,4),(4,5)]),
      ('B · Pipeline + operator inset',[(.12,.72,'frames'),(.39,.72,'state'),(.66,.72,'decoder'),(.9,.72,'features'),(.60,.2,'expanded mixture')],[(0,1),(1,2),(2,3),(2,4)]),
      ('C · Fixed anchor and horizon fan',[(.16,.47,'anchor'),(.53,.8,'h=1'),(.53,.47,'h=5'),(.53,.14,'h=10'),(.86,.47,'features')],[(0,1),(0,2),(0,3),(1,4),(2,4),(3,4)])]
    for ax,(title,nodes,edges) in zip(axes,versions):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(0,1.02,title,fontsize=9.5,weight='bold')
        for a,b in edges:
            x,y,_=nodes[a];xx,yy,_=nodes[b];ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',shrinkA=23,shrinkB=24,lw=.8,color=GRAY,mutation_scale=8))
        for x,y,label in nodes:
            ax.add_patch(FancyBboxPatch((x-.115,y-.075),.23,.15,boxstyle='round,pad=.01',fc='#F3F7F5',ec=GRAY,lw=.7));ax.text(x,y,label,ha='center',va='center',fontsize=8)
    fig.tight_layout();fig.savefig(DESIGN/'composition_drafts.pdf');fig.savefig(DESIGN/'composition_drafts.png',dpi=140);plt.close(fig)

def render():
    for name,digest in EXPECTED.items():
        if sha(ROOT/name)!=digest:raise ValueError('Scientific source or image changed: '+name)
    OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':STYLE['figure_font_family'],'font.size':8.5,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    W,H=396,237.6;fig=plt.figure(figsize=(5.5,3.6),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')
    texts=[];edges=[];nodes={}
    def text(x,y,s,size=8.5,color=INK,ha='center',weight='normal'):
        obj=ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight,zorder=10);texts.append(obj);return obj
    def box(name,x,y,w,h,label=None,ec=GRAY,fc='white',size=8.5,dashed=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=3',fc=fc,ec=ec,lw=.75,ls='--' if dashed else '-',zorder=3));nodes[name]=[x,y,w,h]
        if label:text(x+w/2,y+h/2,label,size=size,color=ec)
    def arrow(name,points,color=GRAY):
        ax.add_patch(FancyArrowPatch(path=MP(points,[MP.MOVETO]+[MP.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=5.5,color=color,lw=.85,zorder=4));edges.append({'id':name,'points':points,'color':color})
    palette=np.array([[95,118,173],[124,167,184],[148,190,188],[110,162,152],[153,166,196],[177,157,187],[200,173,169],[203,188,153],[136,164,174],[145,181,167],[184,195,172],[202,199,172],[125,146,177],[156,170,183],[184,173,190],[162,162,185]],float)/255
    T=.75*np.eye(16)+.2*np.roll(np.eye(16),1,axis=1)+.05*np.ones((16,16))/16
    def grid(name,x,y,s,mixed=False):
        colors=T@palette if mixed else palette
        for r in range(4):
            for c in range(4):ax.add_patch(Rectangle((x+c*s/4,y+(3-r)*s/4),s/4,s/4,fc=colors[4*r+c],ec='white',lw=.35,zorder=5))
        ax.add_patch(Rectangle((x,y),s,s,fc='none',ec=GRAY,lw=.6,zorder=6));ax.add_patch(Rectangle((x+s/4,y+s/2),s/4,s/4,fc='none',ec=INK,lw=.85,zorder=7));nodes[name]=[x,y,s,s]
    def operator(name,x,y,symbol,color=GREEN):
        ax.add_patch(Circle((x,y),7,fc='white',ec=color,lw=.8,zorder=5));text(x,y,symbol,size=11,color=color);nodes[name]=[x-7,y-7,14,14]
    text(8,226,'a  Build a causal state',size=9.5,weight='bold',ha='left')
    text(184,226,'b  Anchor, mix, then correct',size=9.5,weight='bold',ha='left')
    ax.plot([176,176],[6,218],color='#DCE3E7',lw=.75)
    text(53,210,'Observed images',size=8.5,color=BLUE)
    for j,frame in enumerate((0,5,10)):
        p=ROOT/f'paper/figure_sources/spatial_qualitative/case1_recorded_frame_{frame}.png';pixels=np.asarray(Image.open(p));x=10+31*j;y=183
        ia=fig.add_axes([x/W,y/H,28/W,15.75/H]);ia.imshow(pixels,interpolation='nearest');ia.set_axis_off()
    box('frozen-dino',113,180,52,31,'Frozen\nDINOv2',ec=BLUE,fc='#F3F8FC',dashed=True)
    arrow('observed-rgb-to-frozen-dino',[(101,194),(112,194)],BLUE)
    for j in range(3):grid('observed-grid-'+str(j),119+15*j,160,12)
    arrow('frozen-dino-to-normalized-patch-grids',[(139,179),(139,173)],BLUE)
    text(139,153,r'$Z_{-2},Z_{-1},Z_0$',size=8,color=BLUE)
    box('spatial-encoder',103,121,63,25,'Spatial\nencoder',ec=GREEN,fc='#F5F9F5',size=8.5)
    arrow('support-grids-to-spatial-encoder',[(139,148),(139,147)],BLUE)
    box('past-actions',12,156,63,19,r'Past $a_{-1:0}$',ec=ORANGE,size=8.5)
    box('past-context',11,96,64,26,'Past-only\ncontext',ec=GREEN,fc='#F5F9F5')
    arrow('past-actions-to-transition-context',[(29,155),(29,123)],ORANGE)
    arrow('mean-encoded-support-to-transition-context',[(102,133),(88,133),(88,109),(76,109)],BLUE)
    text(86,146,'Patch\nmean',size=8,color=BLUE)
    box('support-film',104,82,62,27,'FiLM',ec=GREEN,fc='#F5F9F5')
    arrow('encoded-support-to-film',[(135,120),(135,110)],BLUE)
    arrow('context-to-film',[(75,101),(91,101),(91,95),(103,95)],GREEN)
    box('future-action-prefix',12,75,63,17,r'Prefix $a_{1:h}$',ec=ORANGE,size=8.5)
    box('action-prefix-gru',11,40,64,26,'Prefix GRU',ec=GREEN,fc='#F5F9F5')
    arrow('past-actions-to-causal-gru',[(12,164),(5,164),(5,72),(26,72),(26,67)],ORANGE)
    arrow('future-prefix-to-causal-gru',[(49,74),(49,67)],ORANGE)
    box('lewm-temporal',103,40,63,26,'LeWM',ec=GREEN,fc='#F5F9F5')
    arrow('conditioned-support-to-lewm',[(135,81),(135,67)],BLUE)
    arrow('causal-action-states-to-lewm',[(76,53),(102,53)],ORANGE)
    arrow('lewm-to-horizon-state',[(135,39),(135,29)],GREEN)
    text(136,21,r'$H_h$: 16 × 96',size=8.5,color=GREEN)
    text(47,21,'Dashed: frozen',size=8,color=GRAY)
    # Named tensor ports in panel b refer to the same tensors constructed in a.
    box('horizon-state-port',195,193,31,18,r'$H_h$',ec=GREEN,fc='#F5F9F5',size=10)
    box('pre-film-key-port',319,193,31,18,r'$E_0$',ec=BLUE,fc='#F3F8FC',size=10)
    text(351,201,'pre-FiLM',size=8,color=BLUE,ha='left')
    text(219,181,r'$Q_h=W_QH_h$',size=8,color=GREEN)
    text(335,181,r'$K_0=W_KE_0$',size=8,color=BLUE)
    arrow('horizon-state-to-query',[(210,192),(210,188)],GREEN)
    arrow('pre-film-observation-to-key',[(334,192),(334,188)],BLUE)
    box('row-softmax',184,147,203,22,ec=GREEN,fc='#F5F9F5')
    text(285.5,158,r'$T_h=\mathrm{softmax}_{\rm row}(Q_hK_0^\top/\sqrt{96}+4I)$',size=8,color=GREEN)
    arrow('queries-to-transport-scores',[(217,175),(217,170)],GREEN)
    arrow('keys-to-transport-scores',[(335,175),(335,170)],BLUE)
    for r in range(16):
        for c in range(16):
            shade=1-.86*T[r,c]/T.max();ax.add_patch(Rectangle((228+c*2.25,103+(15-r)*2.25),2.25,2.25,fc=(shade,shade,shade),ec='white',lw=.06,zorder=4))
    ax.add_patch(Rectangle((228,103),36,36,fc='none',ec=GREEN,lw=.65,zorder=5));ax.add_patch(Rectangle((228,125.5),36,2.25,fc='none',ec=GREEN,lw=.7,zorder=6))
    arrow('softmax-to-row-stochastic-matrix',[(246,146),(246,140)],GREEN)
    text(276,130,r'$T_h$',size=9,color=GREEN)
    grid('fixed-last-observed-anchor',185,72,26);text(198,117,'Fixed',size=8,color=BLUE);text(198,106,r'$Z_0$',size=9,color=BLUE)
    operator('matrix-times-anchor',246,85,'×')
    arrow('fixed-anchor-to-feature-product',[(212,85),(238,85)],BLUE)
    arrow('transport-to-feature-product',[(246,102),(246,93)],GREEN)
    grid('mixed-anchor',266,72,26,True);arrow('product-to-mixed-anchor',[(254,85),(265,85)],GREEN)
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
    arrow('blend-to-addition',[(339,46),(342,46)],GREEN)
    box('bounded-innovation',233,4,118,24,ec=ORANGE,fc='#FCF7F0')
    curve=np.linspace(-2.6,2.6,80)
    ax.plot([238,259],[10,10],color='#D7B996',lw=.4,zorder=5)
    ax.plot([238,259],[22,22],color='#D7B996',lw=.4,zorder=5)
    ax.plot(248.5+4*curve,16+6*np.tanh(curve),color=ORANGE,lw=.9,zorder=6)
    text(309,16,r'$\Delta_h=\tanh R(H_h)$',size=8,color=ORANGE)
    arrow('horizon-state-to-projected-innovation',[(392,125.5),(392,16),(352,16)],GREEN)
    ax.add_patch(Circle((392,125.5),1,fc=GREEN,ec='none',zorder=5))
    arrow('bounded-innovation-to-addition',[(309,29),(309,31),(350,31),(350,38)],ORANGE)
    for r in range(4):
        for c in range(4):
            x=365+c*23/4;y=35+r*23/4;ax.add_patch(Rectangle((x,y),23/4,23/4,fc='white',ec=GREEN,lw=.5,zorder=5))
            for line in (1.4,2.7,4):ax.plot([x+1,x+4.7],[y+line,y+line],color=GREEN,lw=.4,zorder=6)
    nodes['future-feature-forecast']=[365,35,23,23];text(376.5,68,r'$\hat Z_h$',size=10,color=GREEN)
    arrow('addition-to-future-features',[(358,46),(364,46)],GREEN)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[t.get_window_extent(renderer) for t in texts]
    overlaps=[]
    for i,a in enumerate(bounds):
        for j,b in enumerate(bounds[i+1:],i+1):
            if a.overlaps(b):overlaps.append([texts[i].get_text(),texts[j].get_text()])
    clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.x1>fig.bbox.width or b.y0<0 or b.y1>fig.bbox.height]
    evidence={'status':'candidate_pending_actual_pixel_review','sources_sha256':EXPECTED,'renderer_sha256':sha(__file__),'style_sha256':sha(ROOT/'paper/design/editorial_style.json'),'geometry':{'width_inches':5.5,'height_inches':3.6,'minimum_font_pt':8,'text_overlaps':overlaps,'clipped':clipped},'arrows':edges,'nodes':nodes,'schematic_transport_row_sum_max_error':float(np.max(np.abs(T.sum(1)-1))),'semantic_contract':'Fixed normalized Z0; pre-FiLM E0 keys; horizon Hh queries; row-softmax(QK/sqrt96+4I); sigmoid affine gate; complementary anchor mixture plus tanh-projected innovation. Future RGB never enters predictor. Repeated named tensors are aliases, not new inputs.','image_attribution':'Three unchanged DROID observations, CC BY 4.0 https://droid-dataset.github.io/','limitations':'Schematic patch colors and weights; no measured attribution, generated RGB, performance claim, CEM or paired-simulator supervision.'}
    (OUT/'spatial_architecture_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'spatial_architecture_main.{ext}',dpi=300)
    fig.savefig(DESIGN/'paper_size.png',dpi=110);plt.close(fig)
    ImageOps.grayscale(Image.open(DESIGN/'paper_size.png')).save(DESIGN/'grayscale.png')
    (OUT/'spatial_architecture_caption.tex').write_text(CAPTION+'\n')
    (OUT/'spatial_architecture_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/spatial_architecture_main.pdf}\n\\caption[Spatial anchored world model.]{'+CAPTION+'}\n\\label{fig:editorial-spatial-method}\n\\end{figure}\n')
    print(json.dumps({'layout_issues':{'overlaps':overlaps,'clipped':clipped},'size_inches':[5.5,3.6]}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--drafts-only',action='store_true');args=parser.parse_args()
    if args.drafts_only:sketches()
    else:render()
