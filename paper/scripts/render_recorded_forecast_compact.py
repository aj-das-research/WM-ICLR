#!/usr/bin/env python3
"""Portable matched-case DROID figure from completed saved evidence only.

Reads its public source pack; never loads a model, raw recording, or private file.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'paper/figure_sources/recorded_forecast_compact'
OUT=ROOT/'paper/generated/editorial'
INK='#243447';MUTED='#5B6A78';TEAL='#287C7A';BLUE='#4477AA';AMBER='#B36B43';GRID='#E2E8ED'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'recorded-forecast-compact','image.composite_image':False,'axes.linewidth':.55})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
DATA=json.loads((HERE/'data.json').read_text());W=396
TITLES=['Largest gain','Median','Largest regression']

class Figure:
 def __init__(self,name,h=230):
  self.name=name;self.h=h;self.fig=plt.figure(figsize=(W/72,h/72));self.texts=[];self.photos=[];self.axes=[]
 def text(self,x,y,s,size=8,color=INK,bold=False,ha='left'):
  t=self.fig.text(x/W,1-y/self.h,s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',ha=ha,va='center');self.texts.append(t);return t
 def ax(self,x,y,w,h):
  a=self.fig.add_axes([x/W,1-(y+h)/self.h,w/W,h/self.h]);self.axes.append(a);return a
 def photo(self,r,x,y,w):
  assert sha(HERE/r['file'])==r['sha256'];im=np.asarray(Image.open(HERE/r['file']));h=w*im.shape[0]/im.shape[1];a=self.ax(x,y,w,h);a.imshow(im,interpolation='none',aspect='equal');a.axis('off');self.photos.append(a)
 def rule(self,x,y,w):self.fig.add_artist(Line2D([x/W,(x+w)/W],[1-y/self.h,1-y/self.h],transform=self.fig.transFigure,color=GRID,lw=.65))
 def finish(self):
  self.fig.canvas.draw();r=self.fig.canvas.get_renderer();bounds=[]
  tracked=list(self.texts)
  for a in self.axes:
   if a in self.photos:continue
   tracked+=a.get_xticklabels()+a.get_yticklabels()
  issues=[]
  for t in tracked:
   if not t.get_visible() or not t.get_text():continue
   b=t.get_window_extent(r);box=[b.x0*72/self.fig.dpi,self.h-b.y1*72/self.fig.dpi,b.x1*72/self.fig.dpi,self.h-b.y0*72/self.fig.dpi]
   if min(box)<0 or box[2]>W or box[3]>self.h:issues.append({'outside':t.get_text(),'box':box})
   for q in bounds:
    z=q['box']
    if min(box[2],z[2])>max(box[0],z[0]) and min(box[3],z[3])>max(box[1],z[1]):issues.append({'overlap':[t.get_text(),q['text']]})
   bounds.append({'text':t.get_text(),'box':box,'font_pt':t.get_fontsize()})
  for ext in ('pdf','svg','png'):self.fig.savefig(OUT/f'{self.name}.{ext}',dpi=300,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None)
  self.fig.savefig(OUT/f'{self.name}_paper_width.png',dpi=120)
  plt.close(self.fig);ImageOps.grayscale(Image.open(OUT/f'{self.name}_paper_width.png')).save(OUT/f'{self.name}_grayscale.png')
  (OUT/f'{self.name}_layout.json').write_text(json.dumps({'canvas_inches':[W/72,self.h/72],'labels':bounds,'issues':issues},indent=2)+'\n')

def axis_style(a,horizons,limit=15,ticks=True):
 a.set_xlim(.8,5.2);a.set_ylim(-limit,limit);a.set_xticks([1,3,5]);a.set_yticks([-limit,0,limit] if ticks else [0]);a.tick_params(labelsize=8,length=2,pad=1)
 a.axhline(0,color='#84929C',lw=.65,zorder=1);a.spines[['top','right']].set_visible(False)
 for s in ('left','bottom'):a.spines[s].set_color('#AFBAC2')

def gains_label(case,brief=False):
 s=f"E {case['episode_gain']*1000:+.2f}"
 if case['shown_window_gain'] is not None:s+=f" · W {case['shown_window_gain']*1000:+.2f}"+('*' if case['role']=='median' else '')
 return s

def rank_strip(f,study,x,y,w,h):
 a=f.ax(x,y,w,h);v=np.array(sorted([r['gain'] for r in study['all_episode_gains']],reverse=True))*1000;xx=np.arange(len(v))
 a.axhline(0,color='#9BA7B0',lw=.5);a.vlines(xx,0,v,color=np.where(v>0,TEAL,AMBER),lw=.65);a.set_xlim(-1,len(v));a.set_ylim(-15,15);a.set_xticks([]);a.set_yticks([])
 a.spines[:].set_visible(False)
 return a

def draft_c():
 f=Figure('recorded_forecast_compact',226)
 colors=[BLUE,'#677985',AMBER];styles=['-','--',':'];markers=['o','s','^']
 for j,s in enumerate(DATA['studies']):
  base=5+j*112
  f.text(6,base+4,f"{'A' if j==0 else 'B'}  {s['title']}",8.5,bold=True)
  f.text(390,base+4,f"{s['episodes']} episodes · {s['positive']} gain / {s['negative']} regress",8,MUTED,ha='right')
  f.text(6,base+16,s['curve_scope']+' · input 10 / target 35',8,MUTED)
  for k,c in enumerate(s['cases']):
   x=6+k*99
   f.fig.add_artist(Line2D([(x+1)/W],[1-(base+29)/f.h],transform=f.fig.transFigure,color=colors[k],marker=markers[k],ms=3,ls='none'))
   f.text(x+6,base+29,TITLES[k]+('*' if j==1 and k==1 else ''),8,colors[k],bold=True)
   for i,r in enumerate(c['assets']):f.photo(r,x+i*47,base+38,44)
   f.text(x,base+71,gains_label(c),8,MUTED)
  a=f.ax(331,base+34,58,37)
  for k,c in enumerate(s['cases']):a.plot(c['horizons'],np.array(c['gain_curve'])*1000,color=colors[k],ls=styles[k],marker=markers[k],ms=2.3,lw=1)
  axis_style(a,[1,3,5]);f.text(359,base+20,'Gain ×1000',8,MUTED,ha='center')
  f.text(6,base+87,'All episodes',8,MUTED)
  rank_strip(f,s,55,base+81,237,13)
  lo,hi=np.array(s['ci95_gain'])*1000
  mean=s['population_gain']*1000
  f.text(6,base+104,f"Gain ×1000: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] (95% CI) · {s['relative_reduction_percent']:.3f}% lower MSE",8,INK)
  if j==0:f.rule(6,115,384)
 return f


def main():
 manifest=json.loads((HERE/'manifest.json').read_text())
 for path,digest in manifest['runtime_inputs'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Changed source input: '+path)
 assert DATA['status']=='saved_evidence_verified_no_new_evaluation'
 plotted=0
 for study,count,positive,negative in zip(DATA['studies'],[132,65],[73,46],[59,19]):
  gains=np.array([r['gain'] for r in study['all_episode_gains']])
  assert len(gains)==count and (int((gains>0).sum()),int((gains<0).sum()))==(positive,negative)
  np.testing.assert_allclose(gains.mean(),study['population_gain'],atol=1e-14,rtol=0)
  assert [c['role'] for c in study['cases']]==['largest_gain','median','largest_regression']
  for c in study['cases']:
   assert len(c['assets'])==2 and [r['native_index'] for r in c['assets']]==[10,35]
   assert max(np.abs(c['gain_curve']))<.015
   if study['id']=='original':
    curves=c['absolute_curves'];want=[curves['framewise'][str(h)]-curves['factorized'][str(h)] for h in c['horizons']]
   else:
    curves=c['absolute_curves'];want=np.array(curves['calibrated_framewise'])-np.array(curves['calibrated_ours'])
   np.testing.assert_array_equal(want,c['gain_curve']);plotted+=len(c['horizons'])
 assert plotted==24
 median=DATA['studies'][1]['cases'][1]
 assert median['episode_gain']>0 and median['shown_window_gain']<0
 OUT.mkdir(parents=True,exist_ok=True);fig=draft_c();fig.finish()
 layout=json.loads((OUT/'recorded_forecast_compact_layout.json').read_text())
 assert not layout['issues'],layout['issues']
 assert min(r['font_pt'] for r in layout['labels'])>=8
 result={'status':'passed_saved_evidence_and_geometry_checks','scope':DATA['scope'],'canvas_inches':[5.5,226/72],
 'runtime_inputs':manifest['runtime_inputs'],'manifest_sha256':sha(HERE/'manifest.json'),
 'case_count':6,'full_photo_count':12,'plotted_paired_gain_values':24,'all_episode_gains':197,
 'same_fixed_case_ids_no_reselection':True,'case_display_order':'largest gain, median, largest regression',
 'study_aggregation':['132 original uncalibrated test: episode-mean h1/3/5','65 fresh calibrated: fixed first-window h1–5'],
 'gain_units':'All signed gains, curves, ranked marks and intervals are multiplied by1000; percentages are relative reductions of population mean MSE',
 'uncertainty':'Only population95%session/seed paired bootstrap CIs are shown. Case curves are three-seed point means.',
 'absolute_values_retained':True,'no_new_model_or_evaluation':True,
 'outputs':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'recorded_forecast_compact.pdf',OUT/'recorded_forecast_compact.svg',OUT/'recorded_forecast_compact.png',OUT/'recorded_forecast_compact_layout.json']}}
 (OUT/'recorded_forecast_compact.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'pdf_sha256':sha(OUT/'recorded_forecast_compact.pdf'),'source_sha256':sha(__file__)}))

if __name__=='__main__':main()
