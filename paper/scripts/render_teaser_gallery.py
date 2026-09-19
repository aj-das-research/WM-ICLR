#!/usr/bin/env python3
"""Candidate only: photographed dataset gallery, source dependencies and DROID evidence."""
from pathlib import Path
import hashlib,importlib.util,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Circle,Polygon
from matplotlib.path import Path as MPath
from PIL import Image,ImageOps
import numpy as np
from teaser_story_glyphs import feature_tensor,mixing_vignette
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/teaser_gallery'
SOURCES=ROOT/'paper/figure_sources/teaser_gallery'
W,H=396,230.4
INK='#182B49';BLUE='#315EDB';TEAL='#008C82';CORAL='#E96E64';GOLD='#E8AE3D';MUTED='#53657C'
HELPER=ROOT/'paper/scripts/render_teaser_story.py'
EXPECTED_HELPER='d8722abfb9f9eda9ccbc2864b425d5498086827a581bfcebb6c1a185afd2770e';EXPECTED_GLYPHS='dc96d56f03423e831d8dc98b0229566373c916f4bf46ae3c42ac1a0284b3ea10'
CAPTION=(r'\textbf{Action-conditioned visual forecasting from a fixed observed reference.} '
 r'Recorded inputs: the prespecified median DROID case and the first internal-training trajectory/frame of each IWS task, selected without outcomes. '
 r'IWS training is in progress; these examples imply neither measured gains nor simultaneous inputs. '
 r'Autoregression feeds forecasts forward; ShiftWM mixes a fixed observed feature grid plus bounded corrections. Both use the same history and supplied actions, omitted from these conceptual routes. '
 r'Camera and feature glyphs are schematic, never predicted RGB; three drawn steps illustrate the dependency through the ten-step endpoint. '
 r'Bottom: all 141 DROID development episodes, with lower-error circles and higher-error crosses against matched autoregression, averaging windows and three seeds. '
 r'The 5.30\% reduction compares population mean native, training-standardized endpoint errors. '
 r'Uncertainty: Figure~\ref{fig:editorial-spatial}. Images: DROID (CC BY 4.0) and IWS \citep{zhang2026rlawm}.')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked():
 assert sha(HELPER)==EXPECTED_HELPER
 assert sha(ROOT/'paper/scripts/teaser_story_glyphs.py')==EXPECTED_GLYPHS
 population=json.loads((SOURCES/'population.json').read_text())
 for rel,expected in population['authoritative_source_sha256'].items():assert sha(ROOT/rel)==expected,rel
 rows=population['episodes_sorted_by_gain'];assert len(rows)==141
 values=np.array([100*(r['baseline_mse']-r['transport_mse'])/r['baseline_mse'] for r in rows])
 np.testing.assert_allclose(values,[r['gain_percent'] for r in rows],rtol=0,atol=1e-11)
 assert int((values>0).sum())==136 and int((values<0).sum())==5
 aggregate=population['aggregate_h10'];gain=100*(aggregate['autoregressive_mse']-aggregate['transport_mse'])/aggregate['autoregressive_mse']
 np.testing.assert_allclose(gain,aggregate['overall_relative_error_reduction_percent'],rtol=0,atol=1e-12)
 manifest=json.loads((SOURCES/'asset_manifest.json').read_text())
 split=json.loads((ROOT/'configs/real_video_iws/split_v1.json').read_text())
 for r in manifest['records']:
  assert r['split']=='internal_train' and r['frame']==0
  assert r['episode_id']==sorted(split['partitions'][r['task']]['internal_train'])[0]
  assert sha(ROOT/r['asset'])==r['asset_sha256']
  assert sha(ROOT/'configs/real_video_iws/split_v1.json')==r['split_sha256']
  assert hashlib.sha256(np.asarray(Image.open(ROOT/r['asset'])).tobytes()).hexdigest()==r['pixel_sha256']
 return population,rows,gain,manifest

def sketches():
 fig,axes=plt.subplots(1,3,figsize=(11,2.8));fig.patch.set_facecolor('white')
 options=[('A  Gallery → mechanism → evidence',[(.02,.70,.96,.26,'Camera + four real datasets'),(.02,.29,.96,.34,'Matched feature-source routes'),(.02,.04,.96,.19,'All DROID episode evidence')]),
 ('B  Dataset contact sheet + mechanism',[(.02,.26,.34,.69,'Camera / 2×2 gallery'),(.42,.50,.56,.45,'AR / fixed source'),(.42,.06,.56,.36,'DROID evidence')]),
 ('C  Mechanism-centered photo wings',[(.02,.62,.25,.32,'DROID input'),(.72,.62,.26,.32,'IWS examples'),(.02,.26,.96,.30,'Observed anchor / feedback'),(.02,.04,.96,.15,'DROID evidence')])]
 for ax,(title,boxes) in zip(axes,options):
  ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title(title,fontsize=9,loc='left')
  for x,y,w,h,label in boxes:
   ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.018',fc='#EEF5FC',ec=BLUE,lw=.7));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
 fig.tight_layout();fig.savefig(DESIGN/'composition_alternatives.png',dpi=150);plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
 population,rows,gain,manifest=checked()
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'shiftwm-teaser-gallery-v1','image.composite_image':False})
 sketches();fig=plt.figure(figsize=(W/72,H/72),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.set_aspect('equal');ax.axis('off')
 texts=[];edges=[];photos=[]
 def tx(x,y,s,size=8,color=INK,ha='left'):
  t=ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',zorder=15);texts.append(t);return t
 def box(x,y,w,h,fc,ec='none',r=3,z=0):
  a=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=.65,zorder=z);ax.add_patch(a);return a
 def arrow(name,pts,color=INK,lw=.9):
  p=FancyArrowPatch(path=MPath(pts,[MPath.MOVETO]+[MPath.LINETO]*(len(pts)-1)),arrowstyle='-|>',mutation_scale=6.2,lw=lw,color=color,zorder=8);ax.add_patch(p);edges.append({'name':name,'points':pts});return p
 def photo(path,x,y,w,label,source):
  im=Image.open(path).convert('RGB');h=w*im.height/im.width
  ia=fig.add_axes([x/W,y/H,w/W,h/H],zorder=4);ia.imshow(im,interpolation='none');ia.axis('off')
  ia.add_patch(Rectangle((-.004,-.006),1.008,1.012,transform=ia.transAxes,fc='none',ec='white',lw=.7,clip_on=False))
  tx(x+w/2,222,label,8,INK,'center')
  photos.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'pixel_sha256':hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),'source':source,'crop':'none','geometry_points':[x,y,w,h]})
 # A small camera identifies recorded observations; it is not an extra model module.
 box(4,183,29,20,INK,ec='#567096',r=3,z=4)
 ax.add_patch(Polygon([(4,203),(10,208),(33,208),(33,203)],fc='#6984A7',ec='none',zorder=3))
 box(8,202,7,5,BLUE,r=1,z=5)
 for radius,color in [(10.2,'#A9BDD2'),(8.5,INK),(6.4,BLUE),(4.4,'#28B5CD'),(2.2,'#B7EEF2')]:ax.add_patch(Circle((25,193),radius,fc=color,ec='none',zorder=6))
 ax.add_patch(Circle((22.5,196),1.4,fc='white',ec='none',zorder=7));tx(20,173,'RGB',8,MUTED,'center')
 photo(ROOT/'paper/figure_sources/spatial_qualitative/case1_recorded_frame_10.png',42,164,85.333,'DROID','Prespecified median DROID input; no crop')
 for r,x,label in zip(manifest['records'],[146,228,310],['IWS PushT','IWS Box','IWS Rope']):photo(ROOT/r['asset'],x,164,64,label,r)
 tx(85,155,'Median input',8,MUTED,'center')
 tx(260,155,'Training in progress · input examples only',8,MUTED,'center')
 # Same high-level source dependency as the accepted teaser; richer math is Fig2.
 box(4,105,388,44,'#F0F4FD',r=4);tx(11,137,'Autoregression',8.5);tx(11,123,'Predict → reuse',8,MUTED)
 centers=[146,220,290,360]
 for i,x in enumerate(centers):
  feature_tensor(ax,x-18,110,36,30,'observed' if i==0 else 'recursive')
  if i:arrow(f'AR forecast {i-1} to {i}',[(centers[i-1]+20,124),(x-20,124)])
 tx(146,146,r'$Z_0$',8,BLUE,'center')
 for x,s in zip(centers[1:],[r'$\widehat Z_1$',r'$\widehat Z_2$',r'$\widehat Z_3$']):tx(x,146,s,8,INK,'center')
 box(4,39,388,64,'#EAF9F4',r=4);tx(11,97,'ShiftWM (ours)',8.5,TEAL)
 vignette=mixing_vignette(ax,7,42,110,44)
 for x in centers[1:]:feature_tensor(ax,x-18,66,36,30,'mixed')
 box(129,39,257,13,'#DAE8FF',ec='#A9C4F1',r=2,z=2)
 for i,col in enumerate([BLUE,TEAL,CORAL,GOLD]):box(134+i*5,42,4,7,col,r=.5,z=5)
 tx(257,45.5,'Fixed observed features',8,BLUE,'center')
 for i,x in enumerate(centers[1:]):arrow(f'Fixed observed source to query {i+1}',[(x,52),(x,66)],TEAL,1)
 tx(64,35,'Mix + bounded correction',8,TEAL,'center')
 ax.plot([6,390],[30,30],color='#D9E3EC',lw=.65)
 tx(8,23,'136 improve · 5 regress',8.5);tx(198,23,'141 recordings',8,MUTED,'right')
 marks=[]
 for i,r in enumerate(rows):
  x=10+(i%47)*4.05;y=13-(i//47)*4.15;positive=r['gain_percent']>0
  ax.plot(x,y,'o' if positive else 'x',color=TEAL if positive else CORAL,ms=1.9 if positive else 2.8,mew=.75,zorder=9)
  marks.append({'episode_id':r['episode_id'],'gain_percent':r['gain_percent'],'shape':'circle' if positive else 'cross'})
 tx(298,22,f'{gain:.2f}% lower error',11,TEAL,'center');tx(298,8,'DROID development · h10',8,MUTED,'center')
 fig.canvas.draw();ren=fig.canvas.get_renderer();bbs=[t.get_window_extent(ren) for t in texts]
 overlaps=[(texts[i].get_text(),texts[j].get_text()) for i,a in enumerate(bbs) for j,b in enumerate(bbs) if j>i and a.overlaps(b)]
 clipped=[t.get_text() for t,b in zip(texts,bbs) if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height]
 if overlaps or clipped:raise ValueError(str({'overlaps':overlaps,'clipped':clipped}))
 for ext in ('pdf','svg','png'):
  kw={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
  fig.savefig(OUT/f'teaser_gallery.{ext}',dpi=300,**kw)
 fig.savefig(DESIGN/'paper_width.png',dpi=120);fig.savefig(DESIGN/'enlarged.png',dpi=360);plt.close(fig)
 ImageOps.grayscale(Image.open(DESIGN/'paper_width.png')).save(DESIGN/'grayscale.png')
 (OUT/'teaser_gallery_caption.tex').write_text(CAPTION+'\n')
 (OUT/'teaser_gallery_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/teaser_gallery.pdf}\n\\caption[Recorded observations and fixed-reference forecasting.]{'+CAPTION+'}\n\\label{fig:editorial-teaser}\n\\end{figure}\n')
 deps={str(p.relative_to(ROOT)):sha(p) for p in [HELPER,ROOT/'paper/scripts/teaser_story_glyphs.py',SOURCES/'population.json',SOURCES/'asset_manifest.json',ROOT/'configs/real_video_iws/split_v1.json']}
 record={'status':'candidate_pending_independent_review','renderer_sha256':sha(__file__),'dependencies_sha256':deps,'population_dependencies_sha256':population['authoritative_source_sha256'],'size_inches':[5.5,3.2],'minimum_font_pt':8,'text_overlaps':overlaps,'clipped':clipped,'photos':photos,'edges':edges,'population_marks':marks,'gain_population_percent':gain,'caption':CAPTION,'outputs_sha256':{ext:sha(OUT/f'teaser_gallery.{ext}') for ext in ['pdf','svg','png']},'scope':'DROID development evidence; IWS input examples only, no reserved/dev IWS payloads','canonical_include_changed':False,'illustrations':'Original vector camera and schematic feature glyphs; all RGB from unchanged recorded data','selected_alternative':'A: preserves wide source-dependency lanes and large real-input examples'}
 (OUT/'teaser_gallery_evidence.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'size_inches':[5.5,3.2],'overlaps':overlaps,'clipped':clipped}))
if __name__=='__main__':main()
