#!/usr/bin/env python3
"""Photographic task story, schematic source dependency, measured population.

RGB frames are unchanged observations. No predicted RGB is illustrated.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MPath
from PIL import Image, ImageOps
import numpy as np
from teaser_story_glyphs import feature_tensor, mixing_vignette

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/editorial'
DESIGN=ROOT/'paper/design/teaser_story'
PACK=ROOT/'paper/figure_sources/spatial_qualitative'
W,H=396,266.4
INK='#182B49';BLUE='#315EDB';TEAL='#008C82';CORAL='#E96E64';GOLD='#E8AE3D';MUTED='#53657C'
CAPTION=(r'\textbf{Predict visual change without replacing the observed reference.} '
 r'Top: unchanged DROID images illustrate the prespecified median case; the future frame is an evaluator target, not predicted RGB. '
 r'Middle: autoregression feeds forecasts forward; ShiftWM mixes the fixed observed feature grid and adds bounded corrections. '
 r'Both paths use the same available history and actions. Feature layers, commands and the mixing inset are schematic; three drawn forecast steps illustrate a dependency that continues to the ten-step endpoint. '
 r'Bottom: one mark per development episode reports lower (circle) or higher (cross) native ten-step feature error versus matched autoregression, averaged over windows and three seeds. '
 r'The 5.30\% reduction uses population mean errors, not mean episode percentages. '
 r'Figure~\ref{fig:editorial-spatial} reports uncertainty; Figure~\ref{fig:editorial-qualitative} gives the matched case-level comparisons. Images: DROID (CC BY 4.0).')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evidence():
 d=json.loads((DESIGN/'population.json').read_text())
 for rel,h in d['authoritative_source_sha256'].items():
  assert sha(ROOT/rel)==h,rel
 rows=d['episodes_sorted_by_gain'];assert len(rows)==141
 vals=np.array([100*(v['baseline_mse']-v['transport_mse'])/v['baseline_mse'] for v in rows])
 np.testing.assert_allclose(vals,[v['gain_percent'] for v in rows],rtol=0,atol=1e-11)
 assert (vals>0).sum()==136 and (vals<0).sum()==5
 a=d['aggregate_h10'];gain=100*(a['autoregressive_mse']-a['transport_mse'])/a['autoregressive_mse']
 np.testing.assert_allclose(gain,a['overall_relative_error_reduction_percent'],atol=1e-12)
 return d,rows,gain

def sketches():
 """Different reading orders considered before the final composition."""
 f,axes=plt.subplots(1,3,figsize=(11,3.1));f.patch.set_facecolor('white')
 structures=[('A  Scene beside comparison',[(.02,.30,.32,.57,'Recorded scene'),(.40,.58,.58,.29,'Two forecast routes'),(.40,.13,.58,.32,'Population evidence')]),
 ('B  Photo / mechanism / evidence',[(.02,.70,.30,.25,'Observed video'),(.68,.70,.30,.25,'Withheld future'),(.02,.29,.96,.35,'Two dependency routes'),(.02,.05,.96,.17,'All 141 recordings')]),
 ('C  One case, split comparison',[(.34,.70,.32,.25,'Common input'),(.02,.25,.44,.30,'Autoregression'),(.54,.25,.44,.30,'ShiftWM'),(.30,.03,.40,.14,'Shared evaluator')])]
 for ax,(name,boxes) in zip(axes,structures):
  ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title(name,fontsize=10,loc='left')
  for x,y,w,h,label in boxes:
   ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.015',fc='#EBF3FF',ec=BLUE,lw=.75))
   ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
 f.tight_layout();f.savefig(DESIGN/'layout_alternatives.png',dpi=160);plt.close(f)

def main():
 OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
 d,rows,gain=evidence()
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix',
  'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'shiftwm-teaser-story-v1',
  'image.composite_image':False})
 sketches()
 f=plt.figure(figsize=(W/72,H/72),facecolor='white')
 ax=f.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.set_aspect('equal');ax.axis('off')
 texts=[];edges=[];photos=[];glyphs=[]
 def tx(x,y,s,size=8,color=INK,ha='left',weight='normal'):
  t=ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',weight=weight,zorder=15);texts.append(t);return t
 def box(x,y,w,h,fc,ec='none',lw=.7,r=4,z=0):
  p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,zorder=z);ax.add_patch(p);return p
 def arrow(name,pts,color=INK,lw=.9):
  p=FancyArrowPatch(path=MPath(pts,[MPath.MOVETO]+[MPath.LINETO]*(len(pts)-1)),arrowstyle='-|>',mutation_scale=6.2,lw=lw,color=color,zorder=8)
  ax.add_patch(p);edges.append({'id':name,'points':pts,'color':color});return p
 def photo(n,x,y,w,z=3):
  p=PACK/f'case1_recorded_frame_{n}.png';im=Image.open(p);assert im.size==(320,180)
  h=w*180/320
  a=f.add_axes([x/W,y/H,w/W,h/H],zorder=z);a.imshow(im,interpolation='nearest');a.axis('off')
  a.add_patch(Rectangle((-.004,-.007),1.008,1.014,transform=a.transAxes,
   fc='none',ec='white',lw=.65,clip_on=False,zorder=10))
  photos.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'frame':n,'crop':'none','geometry_points':[x,y,w,h]})
  return x,y,w,h
 # A concrete photographic task comes first; future pixels are evaluator-only.
 tx(8,257,'Predict future visual features from observations and actions.',8.7,weight='bold')
 tx(68,243,'Observed video',8.5,BLUE,ha='center',weight='bold')
 tx(330,243,'Recorded future',8.5,INK,ha='center',weight='bold')
 for n,x,y in [(0,8,175),(5,12,171),(10,16,167)]:photo(n,x,y,112,z=3+n/10)
 photo(60,274,167,112,z=4)
 tx(72,159,'Frames 0, 5, 10',8,MUTED,ha='center')
 tx(330,159,'Frame 60 · withheld',8,'#B74942',ha='center')
 tx(201,231,'Supplied actions',8.5,INK,ha='center',weight='bold')
 # Command glyphs describe the interface, not measured motion or a new policy.
 arrow('command-axis-x',[(157,203),(178,203)],BLUE)
 arrow('command-axis-y',[(157,203),(157,221)],TEAL)
 arrow('command-axis-z',[(157,203),(168,214)],CORAL)
 ax.add_patch(Circle((157,203),2,fc=INK,ec='none',zorder=9))
 ax.plot([217,217],[222,213],color=INK,lw=2,zorder=8)
 ax.plot([207,227],[213,213],color=INK,lw=2,zorder=8)
 ax.plot([207,207,211],[213,203,200],color=INK,lw=1.5,zorder=8)
 ax.plot([227,227,223],[213,203,200],color=INK,lw=1.5,zorder=8)
 for i in range(10):box(154+9*i,184,6.4,8,GOLD if i<5 else '#F6CE76',r=1,z=5)
 tx(198,171,'10 future blocks',8,MUTED,ha='center')
 # Distinct shared-input schematic. Forecasts are feature tensors, never RGB.
 box(4,106,388,46,'#F0F4FD',r=4)
 tx(11,139,'Autoregression',8.5,INK,weight='bold')
 tx(11,126,'Predict → reuse',8,MUTED)
 centers=[146,220,290,360]
 for i,x in enumerate(centers):
  g=feature_tensor(ax,x-18,113,36,30,'observed' if i==0 else 'recursive');glyphs.append({'lane':'AR','index':i,'bounds':g['bounds'],'semantics':g['semantics']})
  if i:arrow(f'ar-state-{i}',[(centers[i-1]+20,127),(x-20,127)],INK)
 tx(146,149,r'$Z_0$',8.5,BLUE,ha='center')
 for x,label in zip(centers[1:],[r'$\widehat Z_1$',r'$\widehat Z_2$',r'$\widehat Z_3$']):tx(x,149,label,8.5,INK,ha='center')
 box(4,40,388,64,'#EAF9F4',r=4)
 tx(11,99,'ShiftWM (ours)',8.5,TEAL,weight='bold')
 # Open the novel operation: semantic feature mixture plus bounded correction.
 v=mixing_vignette(ax,7,43,110,44)
 glyphs.append({'lane':'ShiftWM','kind':'mixing_inset','bounds':v['bounds'],'semantics':v['semantics']})
 for x in centers[1:]:
  g=feature_tensor(ax,x-18,68,36,30,'mixed');glyphs.append({'lane':'ShiftWM','bounds':g['bounds'],'semantics':g['semantics']})
 # Exactly one observed source persists at every depicted query horizon.
 box(129,40,257,13,'#DAE8FF',ec='#A9C4F1',r=2,z=2)
 for i,col in enumerate([BLUE,TEAL,CORAL,GOLD]):box(134+i*5,43,4,7,col,r=.5,z=5)
 tx(257,46.5,'Fixed observed features',8,BLUE,ha='center')
 for i,x in enumerate(centers[1:]):arrow(f'fixed-anchor-to-forecast-{i+1}',[(x,53),(x,68)],TEAL,lw=1)
 # Route labels disambiguate mini-operator and shared source from numerical data.
 tx(64,36,'Mix + bounded correction',8,TEAL,ha='center')
 # Population context includes every episode, without turning it into robot success.
 ax.plot([6,390],[31,31],color='#D9E3EC',lw=.65)
 tx(8,23,'136 improve · 5 regress',8.5,INK,weight='bold')
 tx(198,23,'141 recordings',8,MUTED,ha='right')
 marks=[]
 for i,row in enumerate(rows):
  x=10+(i%47)*4.05;y=13-(i//47)*4.15;positive=row['gain_percent']>0
  ax.plot(x,y,'o' if positive else 'x',color=TEAL if positive else CORAL,ms=1.9 if positive else 2.8,mew=.75,zorder=9)
  marks.append({'episode_id':row['episode_id'],'gain_percent':row['gain_percent'],'shape':'circle' if positive else 'cross','xy':[x,y]})
 tx(298,22,f'{gain:.2f}% lower error',12,TEAL,ha='center',weight='bold')
 tx(298,8,'DROID development · h10',8,MUTED,ha='center')
 f.canvas.draw();ren=f.canvas.get_renderer();boxes=[t.get_window_extent(ren) for t in texts]
 overlaps=[(texts[i].get_text(),texts[j].get_text()) for i,a in enumerate(boxes) for j,b in enumerate(boxes) if j>i and a.overlaps(b)]
 clipped=[t.get_text() for t,b in zip(texts,boxes) if b.x0<0 or b.y0<0 or b.x1>f.bbox.width or b.y1>f.bbox.height]
 # Report all geometry problems before finalizing; review also inspects actual pixels.
 geom={'size_inches':[W/72,H/72],'font_points':[8,8.5,8.7,12],'text_overlaps':overlaps,'clipped':clipped}
 for ext in ('pdf','svg','png'):
  kw={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
  f.savefig(OUT/f'teaser_story.{ext}',dpi=300,**kw)
 f.savefig(DESIGN/'paper_width.png',dpi=120);f.savefig(DESIGN/'enlarged.png',dpi=360);plt.close(f)
 ImageOps.grayscale(Image.open(DESIGN/'paper_width.png')).save(DESIGN/'grayscale.png')
 (OUT/'teaser_story_caption.tex').write_text(CAPTION+'\n')
 (OUT/'anchoring_teaser_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/teaser_story.pdf}\n\\caption[Observation-anchored forecasting from recorded video.]{'+CAPTION+'}\n\\label{fig:editorial-teaser}\n\\end{figure}\n')
 record={'status':'candidate_pending_independent_review','script_sha256':sha(__file__),'glyph_source_sha256':sha(ROOT/'paper/scripts/teaser_story_glyphs.py'),'population_source_sha256':sha(DESIGN/'population.json'),'sources':d['authoritative_source_sha256'],'geometry':geom,'photos':photos,'edges':edges,'glyphs':glyphs,'population_marks':marks,'overall_relative_error_reduction_percent':gain,'scientific_boundaries':['No generated or predicted RGB.','Task photos are the unchanged prespecified median example, not a winning case.','Three future states schematize dependency, not all10reporting steps.','Feature glyphs and command directions are schematic.','Population evidence is development native h10 feature error, not physical success.']}
 (OUT/'teaser_story_evidence.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(geom))

if __name__=='__main__':main()
