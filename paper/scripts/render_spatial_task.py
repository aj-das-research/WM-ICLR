#!/usr/bin/env python3
"""Explain the real recorded-action forecasting task with verified observed pixels."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch,Circle
from matplotlib.path import Path as MP
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2]
EXPECTED_REPLAY='12d805f0692f9cd0068f7eb392cf9be996dd778f1fff00f5c1ee4273cfffac20'
INK='#233247';MUTED='#5C6C7B';BLUE='#24719C';ORANGE='#AC601D';TEAL='#197C72'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sketches(output):
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.0))
    variants=[('A · Prediction chain',[(.10,.6,'past'),(.40,.6,'model'),(.73,.6,'features'),(.73,.24,'score'),(.35,.24,'target')],[(0,1),(1,2),(2,3),(4,3)]),
              ('B · Forecast / evaluator lanes',[(.10,.76,'past'),(.40,.76,'model'),(.76,.76,'forecast'),(.10,.25,'target'),(.40,.25,'encoder'),(.76,.25,'error')],[(0,1),(1,2),(3,4),(4,5),(2,5)]),
              ('C · Timeline + scorer',[(.10,.77,'past'),(.72,.77,'withheld'),(.28,.30,'model'),(.70,.30,'score')],[(0,2),(2,3),(1,3)])]
    for ax,(title,nodes,edges) in zip(axs,variants):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(0,.98,title,va='top',weight='bold',fontsize=9)
        for x,y,label in nodes:
            ax.add_patch(FancyBboxPatch((x-.08,y-.075),.16,.15,boxstyle='round,pad=.01',fc='#EDF1F6',ec=INK,lw=.7));ax.text(x,y,label,ha='center',va='center',fontsize=7.5)
        for i,j in edges:
            x1,y1,_=nodes[i];x2,y2,_=nodes[j]
            ax.add_patch(FancyArrowPatch((x1+.085 if x2>x1 else x1,y1-.08 if y2<y1 else y1),(x2-.085 if x2>x1 else x2,y2+.08 if y2<y1 else y2),arrowstyle='-|>',mutation_scale=8,lw=.8,color=BLUE))
    fig.subplots_adjust(left=.03,right=.99,bottom=.07,top=.92,wspace=.16)
    fig.savefig(output/'composition_sketches.png',dpi=140);fig.savefig(output/'composition_sketches.pdf');plt.close(fig)

def render(inputs,output):
    if sha(inputs/'replay.json')!=EXPECTED_REPLAY:raise ValueError('Requires reviewed registered replay')
    r=json.loads((inputs/'replay.json').read_text());assert sha(inputs/'replay_arrays.npz')==r['arrays_sha256'];a=dict(np.load(inputs/'replay_arrays.npz',allow_pickle=False));case=r['cases'][1];prefix=case['prefix'];assert case['rank_descending']==70 and case['first_window_start']==0
    output.mkdir(parents=True,exist_ok=True)
    sketches(output)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig=plt.figure(figsize=(5.5,4.85),facecolor='white');ax=fig.add_axes([.01,.01,.98,.98]);ax.set(xlim=(0,100),ylim=(0,90));ax.axis('off');texts=[];edges=[]
    def text(x,y,s,ha='center',size=8.2,color=INK,weight='normal'):
        t=ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight,zorder=10);texts.append(t)
    def box(x,y,w,h,label,color=INK,fill='#F1F4F7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.3,rounding_size=1.3',facecolor=fill,edgecolor=color,lw=.8,zorder=3));text(x+w/2,y+h/2,label,color=color)
    def arrow(points,color=INK):
        ax.add_patch(FancyArrowPatch(path=MP(points,[MP.MOVETO]+[MP.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=8,color=color,lw=1.05,zorder=4));edges.append(points)
    def image(x,y,w,h,frame):
        pixels=a[prefix+'_images'][frame];iax=fig.add_axes([.01+x*.0098,.01+y*.98/90,w*.0098,h*.98/90]);iax.imshow(pixels);iax.set_axis_off()
    def features(x,y,size,color):
        for row in range(4):
            for col in range(4):
                px=x+col*size/4;py=y+row*size/4
                ax.add_patch(Rectangle((px,py),size/4,size/4,fc='white',ec=color,lw=.65,zorder=4))
                for line in range(3):ax.plot([px+.4,px+size/4-.4],[py+.6+line*.6]*2,color=color,lw=.6,zorder=5)
    text(2,87,'What does the world model predict?',ha='left',size=10,weight='bold')
    text(2,82,'Future visual features, given past images and robot commands',ha='left',size=8.25,color=MUTED)
    text(20,71,'OBSERVED frames',color=BLUE,weight='bold')
    for j,idx in enumerate((0,1,2)):
        image(3+j*11.5,56,10.8,6.15,idx);text(8.4+j*11.5,53.5,str(int(a[prefix+'_frame_indices'][idx])),size=8.0,color=MUTED)
    text(20,49.2,'Source frame index',size=8,color=MUTED)
    box(42,54,13,12,'Frozen\nencoder',color=BLUE,fill='#EFF6FA');arrow([(37,59.8),(41.4,59.8)],BLUE)
    box(66,54,24,12,'World model',color=TEAL,fill='#EFF7F4');arrow([(55.6,59.8),(65.4,59.8)],BLUE)
    text(78,76,'GIVEN recorded commands',color=ORANGE,weight='bold',size=8.0)
    text(78,72.5,'past + future prefix',color=MUTED,size=8.0)
    for i in range(10):ax.add_patch(Rectangle((67+i*2.1,68),1.65,2.0,fc='#EBC49C',ec=ORANGE,lw=.45,zorder=3))
    arrow([(78,67.5),(78,66.5)],ORANGE)
    features(72,32,12,TEAL);arrow([(78,53.5),(78,44.5)],TEAL)
    text(49,39,'PREDICTED\nvisual features',color=TEAL,weight='bold')
    text(49,31.9,'16 patch vectors',size=8,color=MUTED)
    text(20,30.5,'EVALUATION ONLY',color=MUTED,weight='bold',size=8.0)
    ax.add_patch(FancyBboxPatch((1.5,2.5),97,25,boxstyle='round,pad=.2,rounding_size=1.5',fc='#F6F7F9',ec='#BDC7D0',lw=.8,zorder=0))
    text(15.5,25.5,'WITHHELD future frame',color=INK,weight='bold',size=8.0)
    image(4,9,22,12.55,12);text(15,6.2,'Recorded target · frame 60',size=8.0,color=MUTED)
    box(32,11,13,10,'Same frozen\nencoder',color=BLUE,fill='white');arrow([(26.6,15.8),(31.4,15.8)],BLUE)
    features(52,11,10,BLUE);arrow([(45.6,16),(51.4,16)],BLUE);text(56.8,7.4,'Target features',size=8.0,color=MUTED)
    ax.add_patch(Circle((76,16),2.8,fc='white',ec=INK,lw=.8,zorder=5));text(76,16,r'$\Delta^2$',size=9)
    arrow([(62.6,16),(72.8,16)],BLUE);arrow([(78,31.3),(78,24),(76,24),(76,19.3)],TEAL)
    maps=[np.mean([a[f"{c['prefix']}_{m}_s{s}_patch_errors"][9]for s in (0,1,2)],axis=0)for c in r['cases']for m in ('autoregressive','transport')];vmax=max(v.max()for v in maps)
    error=np.mean([a[f'{prefix}_transport_s{s}_patch_errors'][9]for s in (0,1,2)],axis=0)
    iax=fig.add_axes([.01+85*.0098,.01+10*.98/90,12*.0098,12*.98/90]);iax.imshow(error,cmap='magma',vmin=0,vmax=vmax,interpolation='nearest');iax.set_xticks([]);iax.set_yticks([])
    for spine in iax.spines.values():spine.set_color('#A7B5C0');spine.set_linewidth(.6)
    arrow([(79.3,16),(84.4,16)],INK);text(91,25.0,'Patch error',weight='bold',size=8.0);text(91,7.4,'Lower is better',size=8.0,color=MUTED)
    fig.canvas.draw();bboxes=[t.get_window_extent(fig.canvas.get_renderer())for t in texts];collisions=[]
    for i,a1 in enumerate(bboxes):
        for j,b in enumerate(bboxes[i+1:],i+1):
            if a1.overlaps(b):collisions.append([texts[i].get_text(),texts[j].get_text()])
    clipped=[t.get_text()for t,b in zip(texts,bboxes)if b.x0<fig.bbox.x0 or b.x1>fig.bbox.x1 or b.y0<fig.bbox.y0 or b.y1>fig.bbox.y1]
    if collisions or clipped:raise ValueError(json.dumps({'overlap':collisions,'clipped':clipped}))
    for ext in ('pdf','svg','png'):fig.savefig(output/f'spatial_task.{ext}',dpi=300)
    fig.savefig(output/'paper_size.png',dpi=110);fig.savefig(output/'enlarged.png',dpi=360);plt.close(fig);ImageOps.grayscale(Image.open(output/'paper_size.png')).save(output/'grayscale.png')
    caption=(r"\textbf{The recorded-video prediction task.} The model receives three observed images and recorded robot commands, then forecasts visual-feature vectors for a future time. The withheld future RGB frame is used only for evaluation: the same frozen DINO encoder supplies its target features, which are compared with the forecast. The model outputs features, not a generated RGB frame or a chosen robot action. The 16 vector glyphs denote feature vectors schematically; the final heatmap is the actual three-seed $h=10$ per-patch standardized feature error for the prespecified median episode's first window (mean MSE 0.1727; one shared scale 0--0.787 across the companion gallery). Bright patches have larger feature error; this is not a pixel-reconstruction error map. The example's episode-average gain is +5.358\%, but this fixed displayed window has a -0.305\% gain against the autoregressive baseline. Recorded images are unchanged DROID observations (CC BY 4.0); the target frame never enters the forecasting model.")
    (output/'caption.tex').write_text(caption+'\n');(output/'evidence.json').write_text(json.dumps({'status':'candidate_pending_visual_review','replay_sha256':sha(inputs/'replay.json'),'arrays_sha256':sha(inputs/'replay_arrays.npz'),'renderer_sha256':sha(__file__),'case':case['episode_id'],'frame_exports':case['frame_exports'],'actual_error_map':error.tolist(),'error_mean':float(error.mean()),'common_error_scale':[0,float(vmax)],'geometry':{'width_inches':5.5,'height_inches':4.85,'minimum_font_pt':8,'overlap':collisions,'clipped':clipped},'arrows':edges,'semantics':'Upper lane is forecasting; lower lane is evaluator-only. No future-image path into world model. Feature grids are schematic vector glyphs; only heatmap is quantitative.'},indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,default=ROOT/'artifacts/qualitative/spatial_v1_candidate');parser.add_argument('--output',type=Path,default=ROOT/'artifacts/qualitative/spatial_task_candidate');args=parser.parse_args();render(args.input,args.output)
