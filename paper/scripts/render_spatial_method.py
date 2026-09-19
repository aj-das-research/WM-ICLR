#!/usr/bin/env python3
"""Source-grounded schematic only; does not read model predictions or results."""
from pathlib import Path
import argparse
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch,Circle,PathPatch
from matplotlib.path import Path as MPath
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/qualitative/spatial_method_candidate'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'svg.fonttype':'none',
                     'pdf.fonttype':42,'axes.linewidth':.7,'mathtext.fontset':'dejavusans'})
INK='#233247'; MUTED='#59677B'; LINE='#526174'; LIGHT='#EDF1F6'
ANCHOR='#586EB4'; MIX='#147F78'; INNOV='#A66418'; PRED='#2670A5'
COLORS=['#5B70B7','#709CC7','#8EBBC4','#63A795',
        '#91A6CD','#A28FC2','#CDA1AD','#D6B18C',
        '#7195AD','#8CB3A5','#B6C5AA','#D3CAAA',
        '#7085B0','#9AA6B9','#AB9FB6','#C4B5C7']
ILLUSTRATIVE_TRANSPORT=.50*np.eye(16)+.35*np.roll(np.eye(16),1,axis=1)+.15*np.ones((16,16))/16

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sketches():
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.0))
    variants=[('A · Three computation rails',[(.08,.5,'anchor'),(.37,.5,'mix'),(.65,.5,'blend'),(.86,.5,'out')],
               'Identity bypass + transport + bounded innovation'),
              ('B · One target patch',[(.13,.6,'16 inputs'),(.50,.6,'weights'),(.82,.6,'one patch')],
               'Enlarged weighted sum with a small model overview'),
              ('C · Shared-anchor fan',[(.12,.5,'anchor'),(.6,.78,'h=1'),(.6,.50,'h=2'),(.6,.22,'h=10')],
               'One observation fans into independent horizon heads')]
    for ax,(title,nodes,caption) in zip(axs,variants):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        ax.text(0,.98,title,weight='bold',va='top',fontsize=10)
        for x,y,t in nodes:
            ax.add_patch(FancyBboxPatch((x-.075,y-.085),.15,.17,boxstyle='round,pad=.01',fc=LIGHT,ec=LINE,lw=.8))
            ax.text(x,y,t,ha='center',va='center',fontsize=8)
        if title.startswith('A'):
            paths=[[(.16,.5),(.28,.5)],[(.46,.5),(.55,.5)],[(.74,.5),(.77,.5)],[(.08,.6),(.08,.78),(.65,.78),(.65,.60)],[(.37,.23),(.65,.23),(.65,.40)]]
            ax.text(.38,.15,'bounded correction',ha='center',fontsize=8,color=INNOV)
        elif title.startswith('B'):paths=[[(.22,.60),(.40,.60)],[(.60,.60),(.72,.60)]]
        else:paths=[[(.20,.5),(.34,.5),(.34,y),(.5,y)]for y in (.78,.5,.22)]
        for points in paths:
            ax.add_patch(FancyArrowPatch(path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=8,lw=.9,color=LINE))
        ax.text(0,.025,caption,fontsize=8,va='bottom',wrap=True)
    fig.subplots_adjust(left=.025,right=.99,bottom=.09,top=.93,wspace=.20)
    fig.savefig(OUT/'composition_sketches.pdf');fig.savefig(OUT/'composition_sketches.png',dpi=160)
    plt.close(fig)

def render():
    fig=plt.figure(figsize=(5.5,5.7),facecolor='white');ax=fig.add_axes([.015,.015,.97,.97])
    ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    texts=[];edges=[];objects=[]
    def text(x,y,s,size=8.2,color=INK,weight='normal',ha='center',va='center',name=None):
        a=ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,zorder=10)
        texts.append((name or s,a));return a
    def box(x,y,w,h,label=None,color=LINE,fill=LIGHT,size=8.2,name=''):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.25,rounding_size=1.3',ec=color,fc=fill,lw=.8,zorder=3))
        if label:text(x+w/2,y+h/2,label,size=size)
        objects.append({'name':name or label,'bbox':[x,y,x+w,y+h]})
    def arrow(points,color=LINE,width=1.05,name=''):
        p=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=8,lw=width,color=color,zorder=4,joinstyle='round',capstyle='round'))
        edges.append({'name':name,'points':points,'arrowhead':'last endpoint','type':'forward data'})
    def grid(x,y,s,mode='anchor',alpha=1,highlight=False):
        for r in range(4):
            for c in range(4):
                i=r*4+c
                if mode=='anchor':col=COLORS[i]
                elif mode=='mixed':col=ILLUSTRATIVE_TRANSPORT[i]@np.array([matplotlib.colors.to_rgb(v) for v in COLORS])
                elif mode=='pred':
                    original=np.array(matplotlib.colors.to_rgb(COLORS[i]));mixed=ILLUSTRATIVE_TRANSPORT[i]@np.array([matplotlib.colors.to_rgb(v) for v in COLORS])
                    col=np.clip(.55*original+.45*mixed+.025*np.sin(i+np.arange(3)),0,1)
                else:col=('#F2DDC0' if (r+c)%2 else '#B67536')
                ax.add_patch(Rectangle((x+c*s/4,y+(3-r)*s/4),s/4,s/4,facecolor=col,edgecolor='white',linewidth=.7,alpha=alpha,zorder=5))
        ax.add_patch(Rectangle((x,y),s,s,fill=False,edgecolor=LINE,linewidth=.7,zorder=6))
        if highlight:
            ax.add_patch(Rectangle((x+s/4,y+s/2),s/4,s/4,fill=False,ec=INK,lw=1.1,zorder=7))
    def operator(x,y,symbol,color=INK,r=2.3):
        ax.add_patch(Circle((x,y),r,ec=color,fc='white',lw=1.0,zorder=6));text(x,y,symbol,size=11,color=color)

    text(2,97,'a',size=10,weight='bold',ha='left')
    text(7,97,'Build a causal horizon state',size=9.2,weight='bold',ha='left')
    ax.add_patch(FancyBboxPatch((1,68.5),98,25.5,boxstyle='round,pad=.0,rounding_size=1.7',fc='#F6F8FB',ec='#DBE2EB',lw=.7,zorder=0))
    for x,t in [(4,r'$Z_{-2}$'),(10.1,r'$Z_{-1}$'),(16.2,r'$Z_0$')]:
        grid(x,83.1,5.2,alpha=.8 if x<16 else 1);text(x+2.6,81.1,t,size=8.1)
    text(12.7,91.1,'Observed 4×4 grids',size=8.1)
    box(27,81.5,16,10.2,'Spatial\nencoder',name='shared spatial encoder')
    arrow([(22.3,85.8),(26.6,85.8)],name='observed features to spatial encoder')
    box(55,81.5,24,10.2,fill='#F0F1F6',name='LeWM predictor with support-context FiLM')
    text(67,88.5,'LeWM + FiLM',size=8.0,weight='bold')
    text(67,84.2,'support context',size=8.0)
    arrow([(43.5,87),(54.4,87)],name='encoded observed support to temporal predictor')
    box(88,83.3,8.7,6.6,r'$H_h$',fill='#E7EEF8',color=PRED,size=11)
    arrow([(79.5,86.6),(87.5,86.6)],color=PRED,name='conditioned horizon state')
    for i,label in enumerate((r'$a_{-1:0}$',r'$a_{1:h}$')):
        box(4+i*10,71.4,9.1,5.2,label,fill='white',size=8.3,name='past actions' if i==0 else 'causal query action prefix')
    box(27,71.4,16,5.2,'Prefix GRU',size=8.0,name='unidirectional action prefix GRU')
    arrow([(23.7,74),(26.5,74)],name='chronological actions to GRU')
    arrow([(43.5,74),(49,74),(49,83.3),(54.4,83.3)],name='causal action state to predictor')
    text(65,73.6,'observations stay fixed',size=8,color=MUTED)

    text(2,64.7,'b',size=10,weight='bold',ha='left')
    text(7,64.7,'Anchor, mix, then correct',size=9.2,weight='bold',ha='left')
    # The identity bypass is a continuous routed path, never under a label.
    arrow([(12,53),(12,59),(66,59),(66,47.5)],color=ANCHOR,name='identity anchor times one minus gate')
    text(47,61.0,r'$1-g_h$',color=ANCHOR,size=9.0)
    grid(4,37,16,highlight=True);text(12,34.6,r'last observed $Z_0$',size=8.1)
    arrow([(20.5,45),(29.6,45)],color=ANCHOR,name='anchor input to transport product')
    operator(32,45,'×',color=MIX)
    arrow([(34.5,45),(41.5,45)],color=MIX,name='transported anchor grid')
    grid(42,37,16,mode='mixed',highlight=True);text(50,34.6,r'$T_h Z_0$',color=MIX,size=9.2)
    arrow([(58.5,45),(63.6,45)],color=MIX,name='transported anchor times gate')
    text(60.6,49.1,r'$g_h$',color=MIX,size=9.0)
    operator(66,45,'+',color=MIX)
    arrow([(68.5,45),(75.6,45)],name='gated anchor mixture')
    operator(78,45,'+',color=INNOV)
    arrow([(80.5,45),(84.5,45)],color=PRED,name='add bounded innovation to forecast')
    grid(85,39,12,mode='pred');text(91,35.8,r'forecast $\hat Z_h$',size=8.2,color=PRED)
    text(82,55.1,r'$g_h=\sigma(W_g H_h+b_g)$',size=8.0,color=MIX)

    # Full 16 by 16 row-stochastic operator, schematic normalized weights.
    n=16;weights=ILLUSTRATIVE_TRANSPORT
    x,y,s=24,17,16
    for r in range(n):
        for c in range(n):
            value=weights[r,c]
            col=matplotlib.colormaps['Greys'](.13+.70*value/weights.max())
            ax.add_patch(Rectangle((x+c*s/n,y+(n-r-1)*s/n),s/n,s/n,fc=col,ec='white',lw=.13,zorder=3))
    ax.add_patch(Rectangle((x,y),s,s,fill=False,ec=LINE,lw=.7,zorder=4))
    ax.add_patch(Rectangle((x,y+10),s,1,fill=False,ec=MIX,lw=.8,zorder=5))
    text(37,35.5,r'$T_h$',size=9,color=MIX)
    text(32,14.5,'16×16 • row sums = 1',size=8.0)
    arrow([(32,33.5),(32,42.5)],color=MIX,name='row stochastic matrix into anchor product')
    text(6,24.5,r'$H_h,E_0$',size=9)
    arrow([(13.5,25),(22.8,25)],color=MIX,name='query and anchor key make transport scores')
    text(12,20.7,'Q, K → softmax',size=8.0,color=MUTED)

    text(46,23.5,r'$H_h$',size=10,color=INNOV)
    arrow([(49,23.5),(52,23.5)],color=INNOV,name='horizon state to bounded innovation head')
    box(53,18.8,13,9.4,color=INNOV,fill='#FBF4EA',name='project and tanh innovation')
    text(59.5,26.2,'project',size=8.0,color=INNOV)
    xx=np.linspace(-2.1,2.1,50);yy=np.tanh(xx)
    ax.plot(59.5+xx*2,21.5+yy*1.3,color=INNOV,lw=1.0,zorder=6)
    ax.plot([54.7,64.3],[21.5,21.5],color='#D9C1A2',lw=.5,zorder=5)
    arrow([(66.5,23.5),(70,23.5),(78,23.5),(78,42.5)],color=INNOV,name='bounded innovation into final sum')
    text(75.5,20.3,r'$\Delta_h$',size=9.5,color=INNOV)
    text(61,15.1,r'$\Delta_h=\tanh(R_h)$',size=8.1,color=INNOV)
    text(84.5,27.3,r'$|\Delta_h|\leq1$',size=8.5,color=INNOV)

    text(50,8.9,r'$T_h=\mathrm{softmax}_{\mathrm{row}}(Q_hK_0^\top/\sqrt{96}+4I)$',size=9.2,color=MIX)
    text(50,3.0,'Fixed anchor at every horizon • schematic latent features',size=8.1,color=MUTED)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bboxes=[(name,t.get_window_extent(renderer))for name,t in texts]
    collisions=[]
    for i,(a,ba) in enumerate(bboxes):
        for b,bb in bboxes[i+1:]:
            if ba.overlaps(bb):collisions.append([a,b])
    canvas=fig.bbox
    clipped=[name for name,b in bboxes if b.x0<canvas.x0 or b.x1>canvas.x1 or b.y0<canvas.y0 or b.y1>canvas.y1]
    if collisions or clipped:raise RuntimeError(json.dumps({'text_overlap':collisions,'clipped':clipped}))
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'spatial_method_candidate.{ext}',dpi=300,facecolor='white')
    fig.savefig(OUT/'paper_size.png',dpi=110,facecolor='white')
    fig.savefig(OUT/'enlarged.png',dpi=360,facecolor='white')
    audit={'status':'passed','paper_width_inches':5.5,'height_inches':5.7,'minimum_label_font_pt':min(t.get_fontsize()for _,t in texts),
        'text_intersections':collisions,'clipped_text':clipped,'arrow_paths':edges,'objects':objects,
        'source_sha256':{p:sha(ROOT/p)for p in ['src/shiftwm/real_video_spatial/model.py','reports/real_video_development/spatial_protocol.md','configs/real_video_spatial/v1/registration.json']},
        'script_sha256':sha(__file__),'classification':'illustrative architecture schematic; no learned weights/results read',
        'schematic_weight_construction':'0.50*I + 0.35*roll(I,1,axis=1) + 0.15*ones(16,16)/16, solely to illustrate normalized rows; not checkpoint weights',
        'schematic_color_contract':'Mixed symbolic RGB identity colors use the illustrated unit-row matrix; output uses illustrative gate 0.45 and a small bounded color correction. Colors are not image pixels, 384-channel predictions, learned gate values, or experimental results.',
        'visual_review':'pending actual image inspection'}
    (OUT/'geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plt.close(fig)
    from PIL import Image,ImageOps
    im=Image.open(OUT/'enlarged.png');w,h=im.size
    im.crop((0,int(h*.34),w,int(h*.72))).save(OUT/'crop_routing.png')
    im.crop((int(w*.18),int(h*.65),int(w*.95),int(h*.90))).save(OUT/'crop_operators.png')
    ImageOps.grayscale(Image.open(OUT/'paper_size.png')).save(OUT/'grayscale.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output;OUT.mkdir(parents=True,exist_ok=True);sketches();render()
