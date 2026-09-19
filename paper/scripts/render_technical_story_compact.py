from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator,FuncFormatter
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'paper/generated/editorial';PACK=ROOT/'paper/figure_sources/technical_story_compact';D=json.loads((PACK/'data.json').read_text());W=396
INK='#243447';BLUE='#2875A3';GRAY='#73808B';GREEN='#26734A';RED='#B15D3F';LINE='#DAE2E8';PURPLE='#866293'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'technical-story-compact-v2','image.composite_image':False,'axes.linewidth':.5})
class Figure:
 def __init__(self,h):self.h=h;self.fig=plt.figure(figsize=(5.5,h/72));self.photos=[];self.labels=[];self.axes=[]
 def text(self,x,y,s,size=8,color=INK,ha='left',bold=False):
  t=self.fig.text(x/W,1-y/self.h,s,fontsize=size,color=color,ha=ha,va='center',weight='bold' if bold else 'normal');self.labels.append(t);return t
 def ax(self,x,y,w,h):
  a=self.fig.add_axes([x/W,1-(y+h)/self.h,w/W,h/self.h]);self.axes.append(a);return a
 def line(self,x,y,x2,y2):self.fig.add_artist(Line2D([x/W,x2/W],[1-y/self.h,1-y2/self.h],transform=self.fig.transFigure,color=LINE,lw=.6))
 def photo(self,r,x,y,w,crop=None,mask=None,overview=False):
  a=self.ax(x,y,w,w);a.imshow(np.asarray(Image.open(PACK/r['file'])),interpolation='none');a.set_xticks([]);a.set_yticks([])
  for s in a.spines.values():s.set_edgecolor(LINE);s.set_linewidth(.5)
  if mask is not None:a.contour(np.arange(224),np.arange(224),mask,levels=[.5],colors=[PURPLE],linestyles='--',linewidths=.65)
  if overview and crop is not None:a.add_patch(Rectangle((crop[0]-.5,crop[1]-.5),crop[2]-crop[0],crop[3]-crop[1],fill=False,ec=PURPLE,lw=.7))
  if crop is not None and not overview:a.set_xlim(crop[0]-.5,crop[2]-.5);a.set_ylim(crop[3]-.5,crop[1]-.5)
  self.photos.append(a)
 def save(self,name):
  self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer();labels=list(self.labels)
  for a in self.axes:
   if a not in self.photos:labels+=a.get_xticklabels()+a.get_yticklabels()+list(a.texts)
  bounds=[];issues=[]
  for t in labels:
   if not t.get_visible() or not t.get_text():continue
   b=t.get_window_extent(renderer);box=[b.x0*72/self.fig.dpi,self.h-b.y1*72/self.fig.dpi,b.x1*72/self.fig.dpi,self.h-b.y0*72/self.fig.dpi]
   if t.get_fontsize()<8:issues.append({'font':t.get_text()})
   if min(box)<-.2 or box[2]>W+.2 or box[3]>self.h+.2:issues.append({'clip':t.get_text(),'box':box})
   for q in bounds:
    z=q['box']
    if min(box[2],z[2])-max(box[0],z[0])>.25 and min(box[3],z[3])-max(box[1],z[1])>.25:issues.append({'overlap':[t.get_text(),q['text']]})
   bounds.append({'text':t.get_text(),'box':box})
  for e in ['pdf','svg','png']:self.fig.savefig(HERE/f'{name}.{e}',dpi=240,metadata={'CreationDate':None,'ModDate':None} if e=='pdf' else {'Date':None} if e=='svg' else None)
  self.fig.savefig(HERE/f'{name}_paper_width.png',dpi=120);plt.close(self.fig);ImageOps.grayscale(Image.open(HERE/f'{name}_paper_width.png')).save(HERE/f'{name}_gray.png')
  (HERE/f'{name}_layout.json').write_text(json.dumps({'size_inches':[5.5,self.h/72],'minimum_font_pt':8,'issues':issues,'labels':bounds},indent=2)+'\n');return issues

def contract(f,y=0):
 # A compact operation table, not an invented serial dataflow.
 f.text(5,y+7,'A  Historical',8,bold=True);f.text(96,y+7,'Image calibration',8,bold=True);f.text(239,y+7,'Actions',8,bold=True);f.text(320,y+7,'Predict / plan',8,bold=True)
 f.text(5,y+20,'Framewise',8,GRAY);f.text(96,y+20,'Per-image residual',8,GRAY);f.text(239,y+20,r'$E_a(a)$',8,GRAY);f.text(320,y+20,r'$P_F$ / CEM',8,GRAY)
 f.text(5,y+33,'ShiftWM (ours)',8,BLUE);f.text(96,y+33,r'Shared $c_o$ FiLM',8,BLUE);f.text(239,y+33,r'$c_d$ FiLM',8,BLUE);f.text(320,y+33,r'$P_S$ / CEM',8,BLUE)
 f.line(5,y+42,391,y+42)

def style(a):
 a.spines[['top','right']].set_visible(False)
 for s in ['left','bottom']:a.spines[s].set_color('#9DAEBB')
 a.tick_params(labelsize=8,length=2,pad=1);a.set_axisbelow(True);a.grid(axis='y',lw=.4,color='#E6ECEF')

def criterion(f,c,j,x,y,w,h):
 a=f.ax(x,y,w,h);style(a);th=c['criterion']['thresholds'][j]
 a.axhspan(0,th,color='#E4F1E8',zorder=0);a.axhline(th,color=GREEN,ls=':',lw=.8)
 ymax=max(max(np.array(t['criterion_errors'])[10:,j]) for t in c['traces'].values())*1.08
 for m,color,ls,marker in [('factorized',BLUE,'-','o'),('framewise',GRAY,'--','s')]:
  t=c['traces'][m];times=np.array(t['native_times']);v=np.array(t['criterion_errors'])[:,j];select=times>=10
  a.plot(times[select],v[select],color=color,ls=ls,lw=1);a.scatter([20],[v[20]],c=color,s=9,marker=marker,zorder=4)
  a.scatter([times[-1]],[v[-1]],s=15,c=GREEN if m=='factorized' else RED,marker='o' if m=='factorized' else 'x',zorder=5)
 a.set_xlim(9,51);a.set_ylim(0,ymax);a.set_xticks([10,30,50])
 if c['environment']=='pusht':a.set_yticks([0,250,500] if j==0 else [0,.75,1.5])
 else:a.set_yticks([0,.5,1] if j==0 else [0,.25,.5])
 return a

def prediction(f,c,x,y,w,h,groups=True):
 a=f.ax(x,y,w,h);style(a);pairs=c['prediction_pairs'];allv=[r['models'][m]['next_prediction_mse'] for r in pairs for m in ['factorized','framewise']]
 a.set_yscale('log');lo=min(allv)*.6;hi=max(allv)*1.4;a.set_ylim(lo,hi);a.set_xlim(-.45,4.45)
 for i,r in enumerate(pairs):
  ours,base=[r['models'][m]['next_prediction_mse'] for m in ['factorized','framewise']]
  a.plot([i-.13,i+.13],[ours,base],color=GREEN if ours<base else RED,lw=.85)
  a.scatter([i-.13],[ours],color=BLUE,marker='o',s=12,zorder=4);a.scatter([i+.13],[base],color=GRAY,marker='s',s=12,zorder=4)
 a.axvline(1.5,color=LINE,lw=.7)
 a.set_xticks(range(5));a.set_xticklabels([f"{r['anchor_native_time']}→{r['target_native_time']}" for r in pairs])
 a.set_yticks([.02,.1] if c['environment']=='pusht' else [.003,.01]);a.yaxis.set_minor_locator(NullLocator());a.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
 if groups:
  f.text(x+w*.20,y-9,'Ours actions',8,BLUE,ha='center');f.text(x+w*.71,y-9,'Framewise actions',8,GRAY,ha='center')
 return a

def criterion_headers(f,c,x,y):
 labels=['Position (px)','Angle (rad)'] if c['environment']=='pusht' else ['Joint 1 (rad)','Joint 2 (rad)']
 limits=['<20','<π/9'] if c['environment']=='pusht' else ['<0.05','<0.05']
 for j in range(2):f.text(x+56+j*96,y,labels[j],8,ha='center');f.text(x+56+j*96,y+11,'Pass '+limits[j],8,GREEN,ha='center')

def draft_a():
 f=Figure(306);contract(f)
 for i,c in enumerate(D['cases']):
  x=i*198;name='PushT' if i==0 else 'Reacher';f.text(x+5,51,f'{chr(66+i)}  {name} · {c["seed"]}',8.5,bold=True)
  mask=np.asarray(Image.open(PACK/c['goal_outline_mask']))/255
  for j,(im,label) in enumerate(zip(c['images'],['Goal','Ours ·20','Framewise ·20'])):
   f.text(x+32+j*64,64,label,8,BLUE if j==1 else GRAY,ha='center')
   f.photo(im,x+12+j*64,71,40,c['common_display_crop_xyxy'] if i==1 else None,mask if j else None,overview=j==0)

  criterion_headers(f,c,x,123)
  for j in range(2):criterion(f,c,j,x+25+j*96,143,66,40)
  f.text(x+104,200,'Native calls · stop '+('21 /50' if i==0 else '23 /50'),8,ha='center')
  f.text(x+5,209,'Same-transition MSE (log)',8,bold=True);f.text(x+193,209,('0/5 lower' if i==0 else '3/5 lower'),8,RED if i==0 else GREEN,ha='right')
  prediction(f,c,x+27,229,164,35)
 f.line(198,47,198,282)
 f.fig.add_artist(Line2D([55/W,72/W],[1-286/f.h]*2,transform=f.fig.transFigure,color=BLUE,lw=1,marker='o',markevery=[1],ms=3));f.text(77,286,'ShiftWM (ours)',8,BLUE)
 f.fig.add_artist(Line2D([233/W,250/W],[1-286/f.h]*2,transform=f.fig.transFigure,color=GRAY,lw=1,ls='--',marker='s',markevery=[1],ms=3));f.text(255,286,'Framewise',8,GRAY)
 f.text(198,300,'Frozen encoder · separately trained predictors · contexts fixed in CEM',8,ha='center')
 return f


def main():
 HERE.mkdir(parents=True,exist_ok=True)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 m=json.loads((PACK/'manifest.json').read_text())
 for p,h in m['inputs'].items():assert sha(PACK/p)==h
 assert [(c['environment'],c['seed']) for c in D['cases']]==[('pusht',2031024),('reacher',2031004)]
 assert len(D['all39_replay_traces'])==39 and len(D['all70_prediction_pairs'])==70
 for c in D['cases']:
  assert len(c['prediction_pairs'])==5
  for t in c['traces'].values():
   e=np.array(t['criterion_errors']);assert e.shape==(len(t['native_times']),2) and np.isfinite(e).all()
   np.testing.assert_array_equal(np.all(e<np.array(c['criterion']['thresholds']),axis=1),t['success_flags'])
 issues=draft_a().save('technical_story_compact')
 if issues:raise ValueError(issues)
 ev={'status':'source_bound_render_passed','renderer_sha256':sha(__file__),'manifest_sha256':sha(PACK/'manifest.json'),'caption_sha256':sha(PACK/'caption.tex'),'source_scope':D['scope'],'size_inches':[5.5,306/72],'minimum_font_pt':8,'displayed_prediction_pairs':10,'preserved_prediction_pairs':70,'preserved_replayed_paths':39,'four_positive_summary':D['four_positive_cases_summary'],'new_model_or_environment_execution':False,'native_drawio':'Not applicable to quantitative image/trace plate; SVG labels/rules/curves are editable, source images remain raster.','outputs':{ext:sha(HERE/f'technical_story_compact.{ext}') for ext in ['pdf','svg','png']}}
 (HERE/'technical_story_compact.json').write_text(json.dumps(ev,indent=2)+'\n');print(json.dumps(ev['outputs']))
if __name__=='__main__':main()
