#!/usr/bin/env python3
"""Render source-bound decoder mechanism illustrations from portable JSON/PNGs.

No checkpoint, video, private NPZ, network or GPU is accessed at render time.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text
ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'paper/figure_sources/mixing_multibenchmark'
OUT=ROOT/'paper/generated/editorial'
W,H=396,176.4
INK='#243447';MUTED='#667788';BLUE='#4477AA';TEAL='#287C7A';GOLD='#B88746';GRID='#E1E8ED'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def tx(fig,x,y,s,size=8,col=INK,ha='left',weight='normal'):
 return fig.text(x/W,y/H,s,fontsize=size,color=col,ha=ha,va='center',weight=weight)

def ax(fig,x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])

def frame(fig,path,x,y,w):
 a=np.asarray(Image.open(path).convert('RGB'));h=w*a.shape[0]/a.shape[1]
 fig.add_artist(FancyBboxPatch(((x-2)/W,(y-2)/H),(w+4)/W,(h+4)/H,boxstyle='round,pad=0,rounding_size=.014',transform=fig.transFigure,fc='white',ec='#C9D7E1',lw=.65,zorder=-.5))
 z=ax(fig,x,y,w,h);z.imshow(a,interpolation='none');z.set_axis_off();return h

def curves(fig,d,x,y,w,h=45,legend=True):
 n=d['gate'].shape[1];steps=np.arange(1,n+1);z=ax(fig,x,y,w,h);z.set(xlim=(1,n),ylim=(0,1),xticks=[1,max(2,round(n/2)),n],yticks=[0,.5,1]);z.set_yticklabels(['0','0.5','1']);z.tick_params(length=1.5,pad=2,labelsize=8)
 z.spines[['top','right']].set_visible(False)
 for s in ('left','bottom'):z.spines[s].set_color('#ABBBC8');z.spines[s].set_linewidth(.55)
 z.grid(axis='y',color=GRID,lw=.5);z.set_axisbelow(True)
 for k,col,ls in [('effective_self',TEAL,'-'),('gate',GOLD,'--')]:
  v=d[k];z.fill_between(steps,v.min(0),v.max(0),color=col,alpha=.13,lw=0);z.plot(steps,v.mean(0),color=col,ls=ls,lw=1.1)
 if legend:
  tx(fig,x,y+h+9,'Retention',8,TEAL);tx(fig,x+w-2,y+h+9,'Gate',8,GOLD,'right')
 return z

def check(fig):
 fig.canvas.draw();ren=fig.canvas.get_renderer();ts=[];issues=[]
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text().strip():continue
  if t.axes is not None and not t.axes.axison:continue
  bb=t.get_window_extent(ren)
  if t.get_fontsize()<8:issues.append('font '+t.get_text())
  if min(bb.x0,bb.y0)<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:issues.append('clip '+t.get_text())
  ts.append((t.get_text(),bb))
 for i,(t,b) in enumerate(ts):
  for u,c in ts[i+1:]:
   if b.overlaps(c):issues.append('overlap '+t+' / '+u)
 return {'minimum_font_pt':8,'issues':issues,'size_inches':[W/72,H/72]}

def compose_available(paths,d,iw):
 global H
 H=176.4
 fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
 all_data=[d]+iw
 tx(fig,8,103,'Endpoint own-location weight',8.5)
 tx(fig,195,103,'Raw mix',8,BLUE);tx(fig,278,103,'Gated mix',8,TEAL)
 for px,col,mk in ((187,BLUE,'o'),(269,TEAL,'D')):fig.add_artist(Line2D([px/W],[103/H],marker=mk,ms=3.6,mfc=col,mec='white',mew=.4,ls='none',transform=fig.transFigure))
 for i,name in enumerate(('DROID','IWS PushT','IWS Box','IWS Rope')):
  x=i*99;q=all_data[i];n=q['gate'].shape[0];end=q['gate'].shape[1]
  tx(fig,x+49.5,169,name,8.5,ha='center')
  image=paths[0] if i==0 else q['image']
  frame(fig,image,x+15 if i==0 else x+22,120,69 if i==0 else 55)
  tx(fig,x+49.5,113,f'Dev · {n} seeds' if i==0 else f'Train · {n}/3 seeds',8,BLUE if i==0 else MUTED,'center')
  z=ax(fig,x+12,79,78,16);z.set(xlim=(0,1),ylim=(0,1),xticks=[0,1],yticks=[])
  z.spines[['top','left','right']].set_visible(False);z.spines['bottom'].set_color(GRID);z.spines['bottom'].set_linewidth(.6);z.tick_params(length=1.5,pad=1,labelsize=8)
  raw=q['raw_self'][:,-1];eff=q['effective_self'][:,-1];a,b=raw.mean(),eff.mean()
  z.plot([a,b],[.5,.5],color='#8396A2',lw=1.3,zorder=1)
  for v,col,mk in ((raw,BLUE,'o'),(eff,TEAL,'D')):
   val=v.mean();z.errorbar(val,.5,xerr=np.array([[val-v.min()],[v.max()-val]]),fmt=mk,ms=3.4,mfc=col,mec='white',mew=.4,ecolor=col,elinewidth=.8,capsize=1.8,zorder=3)
  tx(fig,x+50,65,f'{a:.3f} → {b:.3f}',8,INK,'center')
  z=curves(fig,q,x+22,19,67,29,legend=False);z.set_yticks([0,1]);z.set_yticklabels(['0','1']);z.set_xticks([1,end])
  tx(fig,x+50,4.5,'Forecast step' if i==0 else 'Stored offset',8,MUTED,'center')
 tx(fig,142,57,'Retention',8,TEAL);tx(fig,237,57,'Gate',8,GOLD)
 for px,col,ls in ((123,TEAL,'-'),(218,GOLD,'--')):fig.add_artist(Line2D([px/W,(px+14)/W],[57/H,57/H],color=col,ls=ls,lw=1.1,transform=fig.transFigure))
 # Shared descriptive legend applies to all four native-axis plots.
 return fig

def load():
 manifest=json.loads((PACK/'manifest.json').read_text());assert manifest['schema']=='shiftwm_mechanism_figure_sources_v1'
 for name,h in manifest['files'].items():assert Path(name).name==name and sha(PACK/name)==h
 p=json.loads((PACK/'metrics.json').read_text());assert p['schema']=='shiftwm_heterogeneous_mechanism_illustration_v1' and p['status']=='validated_mechanism_replays_not_performance_results'
 expected={'droid':[0,1,2],'pusht':[0,1],'bimanual_box':[0,1,2],'bimanual_rope':[0,2]}
 assert [r['task'] for r in p['cases']]==list(expected)
 records=[]
 for r in p['cases']:
  task=r['task'];seeds=r['seed_ids'];N=10 if task=='droid' else 59
  assert seeds==expected[task] and r['missing_seed_ids']==[s for s in range(3) if s not in seeds]
  assert r['offsets']==list(range(1,N+1));assert sorted(x['seed'] for x in r['checkpoints'])==seeds
  q={k:np.asarray(r['curves'][k],dtype=np.float64) for k in ('raw_self','effective_self','gate')}
  for v in q.values():assert v.shape==(len(seeds),N) and np.isfinite(v).all() and v.min()>=0 and v.max()<=1
  assert np.all(q['effective_self']>=q['raw_self']-1e-7)
  f=r['image']['file'];assert Path(f).name==f and sha(PACK/f)==r['image']['sha256']
  rgb=np.asarray(Image.open(PACK/f).convert('RGB'));assert list(rgb.shape)==r['image']['shape'] and hashlib.sha256(rgb.tobytes()).hexdigest()==r['image']['pixel_sha256']
  q.update(task=task,seeds=seeds,image=PACK/f);records.append(q)
 return p,manifest,records

def main():
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'text.color':'#26364B','axes.labelcolor':'#657587','xtick.color':'#657587','ytick.color':'#657587','pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'polished-multibenchmark-preview','image.composite_image':False,'savefig.facecolor':'white'})
 p,m,records=load();fig=compose_available([records[0]['image']],records[0],records[1:]);qa=check(fig);assert not qa['issues'],qa
 OUT.mkdir(parents=True,exist_ok=True)
 for ext in ('pdf','svg','png'):fig.savefig(OUT/f'mixing_multibenchmark.{ext}',dpi=240,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {})
 fig.savefig(OUT/'mixing_multibenchmark_paper_width.png',dpi=120);plt.close(fig)
 evidence={'status':'validated_heterogeneous_mechanism_illustration','renderer_sha256':sha(__file__),'public_sources':m['files'],'public_manifest_sha256':sha(PACK/'manifest.json'),'qa':qa,'cases':p['cases'],'scope':p['heterogeneous_scope'],'metric':p['metric'],'uncertainty':p['uncertainty'],'identity':p['identity'],'backing_sources_sha256':p['backing_sources_sha256'],'endpoint_means':{r['task']:{k:float(r[k][:,-1].mean()) for k in ('raw_self','effective_self','gate')} for r in records},'outputs_sha256':{ext:sha(OUT/f'mixing_multibenchmark.{ext}') for ext in ('pdf','svg','png')}}
 (OUT/'mixing_multibenchmark_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps({'status':evidence['status'],'source':evidence['renderer_sha256'],'outputs':evidence['outputs_sha256'],'qa':qa},indent=2))
if __name__=='__main__':main()
