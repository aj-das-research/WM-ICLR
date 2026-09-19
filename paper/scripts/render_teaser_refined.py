#!/usr/bin/env python3
"""Compact observed-source teaser. Scene and tiles are explicitly conceptual."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
from matplotlib.path import Path as MP
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/teaser_refined'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ASSET=ROOT/'paper/figures/assets/world_concept.png';SOURCE=ROOT/'reports/real_video_spatial/finalization.json'
INK='#243447';BLUE='#4477AA';TEAL='#287C7A';MUTED='#647482';LINE='#DCE3E8';GOLD='#B88746'
CAPTION=r'''\textbf{Forecast change while retaining the observed reference.} Autoregression passes predicted features into the next step. ShiftWM (ours) draws every horizon's mixture from the same observed grid and adds a bounded correction. Supplied actions condition both paths; Figure~\ref{fig:editorial-spatial-method} gives the full computation. The scene and feature tiles are conceptual. The measured callout is the native ten-step endpoint feature-error reduction against matched autoregression on three-seed DROID development data.'''
def main():
 OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
 assert sha(ASSET)=='40e9c67d187cf3f916e4a3923695baa11a6eeb2c14904b57733fe89712251a94'
 d=json.loads(SOURCE.read_text());assert d['status']=='passed'
 e=next(v for v in d['paired_effects'] if v['comparator']=='autoregressive' and v['metric']=='native_mse' and v['horizon']==10)
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'shiftwm-refined-teaser'})
 # Distinct low-detail topology explorations, not numerical evidence.
 f,axs=plt.subplots(1,3,figsize=(10,2.4))
 options=[('A  Paired recursive chains',[(.08,.65,.2,.2),(.4,.65,.2,.2),(.72,.65,.2,.2),(.08,.2,.2,.2),(.4,.2,.2,.2),(.72,.2,.2,.2)]),('B  Shared observed bank',[(.02,.25,.28,.65),(.37,.7,.6,.15),(.4,.4,.16,.15),(.62,.4,.16,.15),(.82,.4,.16,.15),(.37,.12,.61,.13)]),('C  Central source fan',[(.05,.42,.2,.17),(.39,.42,.23,.17),(.78,.12,.18,.15),(.78,.44,.18,.15),(.78,.76,.18,.15)])]
 for ax,(title,boxes) in zip(axs,options):
  ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title(title,fontsize=9)
  for x,y,w,h in boxes:ax.add_patch(Rectangle((x,y),w,h,fc='#F0F5F7',ec=BLUE,lw=.7))
 f.savefig(DESIGN/'alternatives.png',dpi=160);plt.close(f)
 W,H=396,158.4;fig=plt.figure(figsize=(5.5,2.2),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off');texts=[];edges=[]
 def text(x,y,s,size=8,color=INK,weight='normal',ha='center'):
  t=ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=color,weight=weight,zorder=8);texts.append(t)
 def arrow(points,color):
  ax.add_patch(FancyArrowPatch(path=MP(points,[MP.MOVETO]+[MP.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=6.0,lw=.8,color=color,zorder=4));edges.append({'points':points,'color':color})
 def tile(x,y,size,observed=False):
  colors=['#9FBED4','#CCDCE4','#96B8B7','#B6CFCE']
  for row in range(4):
   for col in range(4):
    ax.add_patch(Rectangle((x-size/2+col*size/4+.2,y-size/2+row*size/4+.2),size/4-.4,size/4-.4,fc=colors[(row*3+col)%4],ec='none',zorder=5))
  ax.add_patch(Rectangle((x-size/2-.8,y-size/2-.8),size+1.6,size+1.6,fc='none',ec=BLUE if observed else MUTED,lw=.7,ls='-' if observed else (0,(2,1)),zorder=5))
 text(65,147,'Visual observation',8.5,INK,weight='bold');text(65,134,'Concept illustration',8,MUTED)
 ia=fig.add_axes([2/W,51/H,125/W,(125*2/3)/H]);ia.imshow(Image.open(ASSET));ia.axis('off')
 ax.add_patch(FancyBboxPatch((6,9),118,33,boxstyle='round,pad=0,rounding_size=3',fc='#EEF6F4',ec='none'))
 text(65,31,f"{e['relative_error_reduction_percent']:.2f}% lower error",8.5,TEAL,weight='bold')
 text(65,19,'DROID · h10 vs autoregression',8,MUTED)
 ax.plot([133,133],[8,150],color=LINE,lw=.65)
 # Aligned future states make only the source dependency differ between lanes.
 text(143,147,'a  Autoregression',8.5,INK,weight='bold',ha='left')
 text(387,147,'Recursive state',8,MUTED,ha='right')
 xs=[158,229,298,367]
 for x in xs:tile(x,121,19,observed=x==158)
 for a,b in zip(xs[:-1],xs[1:]):arrow([(a+12,121),(b-12,121)],MUTED)
 for x,label in zip(xs,[r'$Z_0$',r'$\widehat Z_1$',r'$\widehat Z_2$',r'$\widehat Z_3$']):text(x,100,label,8.5)
 ax.plot([143,388],[88,88],color=LINE,lw=.6)
 text(143,77,'b  ShiftWM (ours)',8.5,TEAL,weight='bold',ha='left')
 text(387,77,'Mix + correction',8,MUTED,ha='right')
 # Source bank is an explicit graphical identity: one measured Z0 at every horizon.
 ax.add_patch(FancyBboxPatch((143,11),246,25,boxstyle='round,pad=0,rounding_size=3',fc='#F0F5FA',ec='#B9CBDC',lw=.65,zorder=1))
 tile(158,23.5,16,observed=True);text(179,23.5,r'$Z_0$',8.5,BLUE)
 text(282,23.5,'Fixed observed reference',8,BLUE)
 for x,label in zip(xs[1:],[r'$\widehat Z_1$',r'$\widehat Z_2$',r'$\widehat Z_3$']):
  tile(x,59,17);text(x-20,59,label,8.5,TEAL);arrow([(x,36),(x,48.5)],TEAL)
 fig.canvas.draw();ren=fig.canvas.get_renderer();bounds=[t.get_window_extent(ren) for t in texts]
 overlap=[(texts[i].get_text(),texts[k].get_text()) for i,a in enumerate(bounds) for k,b in enumerate(bounds) if k>i and a.overlaps(b)]
 clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height]
 assert not overlap and not clipped,(overlap,clipped)
 for ext in ('pdf','svg','png'):fig.savefig(OUT/f'teaser_refined.{ext}',dpi=300)
 fig.savefig(DESIGN/'paper_width.png',dpi=120);fig.savefig(DESIGN/'enlarged.png',dpi=360);plt.close(fig)
 ImageOps.grayscale(Image.open(DESIGN/'paper_width.png')).save(DESIGN/'gray.png')
 (OUT/'anchoring_teaser_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/teaser_refined.pdf}\n\\caption[Observation-anchored forecasting.]{'+CAPTION+'}\n\\label{fig:editorial-teaser}\n\\end{figure}\n')
 (OUT/'teaser_refined_caption.tex').write_text(CAPTION+'\n')
 (OUT/'teaser_refined_evidence.json').write_text(json.dumps({'status':'source_geometry_checked_pending_independent_review','script_sha256':sha(__file__),'asset_sha256':sha(ASSET),'source_sha256':sha(SOURCE),'source_effect':e,'geometry':{'inches':[5.5,2.2],'font_points':[8,8.5],'text_overlaps':overlap,'clipping':clipped},'edges':edges,'selected':'B shared source bank with aligned forecast columns','raster_boundary':'unchanged original generated concept image; grids/weights are not measurements','new_observation_or_model_inference':False},indent=2)+'\n')
 print(json.dumps({'size':[5.5,2.2],'label_min_pt':8,'layout_issues':[]}))
if __name__=='__main__':main()
