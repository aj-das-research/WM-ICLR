#!/usr/bin/env python3
"""Measured spatial source-retention and separate historical simulator evidence."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.text import Text
ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'paper/figure_sources/decoder_evidence'
OUT=ROOT/'paper/generated/editorial'
NAME='decoder_evidence'
W,H=396,190
INK='#243447';MUTED='#63768A';TEAL='#267B79';BLUE='#4477AA';GREEN='#25764B';RED='#AA583D';LINE='#CEDCE5'
CM=LinearSegmentedColormap.from_list('retention',['#F2F6F9','#BBD7DF','#6AA4B9','#276A87','#183C58'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ax(fig,x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def txt(fig,x,y,s,size=8,c=INK,ha='left',weight='normal'):
 return fig.text(x/W,y/H,s,fontsize=size,color=c,ha=ha,va='center',weight=weight)
def picture(fig,p,x,y,w,h):
 a=np.asarray(Image.open(p).convert('RGB')); ratio=a.shape[1]/a.shape[0]
 ww=min(w,h*ratio);hh=ww/ratio;x+=(w-ww)/2;y+=(h-hh)/2
 f=FancyBboxPatch(((x-1)/W,(y-1)/H),(ww+2)/W,(hh+2)/H,boxstyle='round,pad=0,rounding_size=.009',transform=fig.transFigure,fc='white',ec=LINE,lw=.55,zorder=0)
 fig.add_artist(f);z=ax(fig,x,y,ww,hh);z.imshow(a,interpolation='none');z.set_axis_off()

def tilemap(fig,v,x,y,side=23):
 z=ax(fig,x,y,side,side);z.set(xlim=(0,4),ylim=(4,0));z.set_axis_off()
 for j,value in enumerate(v):
  rr,cc=divmod(j,4);z.add_patch(Rectangle((cc+.03,rr+.03),.94,.94,facecolor=CM(value),edgecolor='none'))
 return z

def load():
 manifest=json.loads((PACK/'manifest.json').read_text())
 for f,h in manifest['files'].items():assert Path(f).name==f and sha(PACK/f)==h,(f,'hash')
 spatial=json.loads((PACK/'mixing.json').read_text());sim=json.loads((PACK/'simulation.json').read_text())
 assert len(spatial['cases'])==len(sim['cases'])==4
 for c in spatial['cases']:
  d=c['endpoint_diagonals'];assert d['seed_ids']==c['seed_ids'] and d['patch_ids']==list(range(16))
  t,m,g=[np.asarray(d[k],dtype=np.float64) for k in ('raw_T','effective_M','gate')]
  assert t.shape==m.shape==g.shape==(len(c['seed_ids']),16)
  np.testing.assert_allclose(m,(1-g)+g*t,atol=1e-7,rtol=1e-7)
  for a,k in [(t,'raw_self'),(m,'effective_self'),(g,'gate')]:np.testing.assert_allclose(a.mean(1),np.asarray(c['curves'][k])[:,-1],rtol=0,atol=1e-12)
  assert (m>=t-1e-7).all() and min(t.min(),m.min(),g.min())>=0 and max(t.max(),m.max(),g.max())<=1
 for c in sim['cases']:
  assert len(c['rows'])==2
  for r in c['rows']:
   a=np.asarray(r['ours_mse_h5']);b=np.asarray(r['control_mse_h5']);assert a.shape==b.shape==(3,)
   np.testing.assert_allclose(100*(1-a.mean()/b.mean()),r['gain_percent'],atol=1e-12,rtol=0)
   np.testing.assert_allclose(100*(1-a/b),r['per_seed_gain_percent'],atol=1e-12,rtol=0)
 for c in spatial['cases']+sim['cases']:
  p=PACK/c['image']['file'];rgb=np.asarray(Image.open(p).convert('RGB'))
  assert sha(p)==c['image']['sha256'] and hashlib.sha256(rgb.tobytes()).hexdigest()==c['image']['pixel_sha256']
 return spatial,sim,manifest

def render(spatial,sim):
 fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
 txt(fig,5,183,'a  Retaining the observed source',8.7,weight='bold')
 txt(fig,261,183,r'$M_{jj}=(1-g_j)+g_jT_{jj}$',8.3,ha='center')
 for i,(c,title) in enumerate(zip(spatial['cases'],['DROID','IWS PushT','IWS Box','IWS Rope'])):
  x=99*i;txt(fig,x+49.5,167,title,8.6,ha='center',weight='bold')
  txt(fig,x+19,153,'Input',8,MUTED,'center');txt(fig,x+57,153,'Raw',8,MUTED,'center');txt(fig,x+84,153,'Gated',8,MUTED,'center')
  picture(fig,PACK/c['image']['file'],x+3,120,33,29)
  d=c['endpoint_diagonals'];a=np.asarray(d['raw_T']).mean(0);b=np.asarray(d['effective_M']).mean(0)
  tilemap(fig,a,x+45,123);tilemap(fig,b,x+72,123)
  txt(fig,x+56.5,115,f'{a.mean():.3f}',8,ha='center');txt(fig,x+83.5,115,f'{b.mean():.3f}',8,ha='center')
  n=len(c['seed_ids']);txt(fig,x+49.5,103,f'Dev · {n} seeds' if i==0 else f'Train · {n}/3 seeds',8,MUTED,'center')
 txt(fig,5,90,'Each cell: weight on its own source patch',8)
 z=ax(fig,271,87,108,6);z.imshow(np.linspace(0,1,256)[None,:],aspect='auto',cmap=CM,vmin=0,vmax=1);z.set_axis_off()
 txt(fig,264,90,'0',8,MUTED,'center');txt(fig,386,90,'1',8,MUTED,'center')
 fig.add_artist(Line2D([5/W,391/W],[80/H,80/H],transform=fig.transFigure,color=LINE,lw=.65))
 txt(fig,5,71,'b  Historical context model · simulation forecasts',8.7,weight='bold')
 for i,(c,title) in enumerate(zip(sim['cases'],['PushT','Reacher','Drone','Tissue'])):
  x=99*i;txt(fig,x+49.5,57,title,8.6,ha='center',weight='bold')
  picture(fig,PACK/c['image']['file'],x+3,17,31,31)
  for k,r in enumerate(c['rows']):
   y=43-18*k;label=['Pair','Extra'][k] if i<2 else ['Trf.','GRU'][k]
   val=r['gain_percent'];color=GREEN if val>0 else RED
   txt(fig,x+41,y,label,8,MUTED);txt(fig,x+96,y,f'{val:+.2f}%',8,color,'right','bold' if val>0 else 'normal')
   z=ax(fig,x+42,y-9,53,4);z.set(xlim=(-10,30),ylim=(-1,1));z.set_axis_off()
   z.plot([-10,30],[0,0],lw=.6,color='#DAE3E9');z.plot([0,0],[-.9,.9],lw=.65,color=MUTED)
   z.plot([0,val],[0,0],lw=1.25,color=color);z.scatter(r['per_seed_gain_percent'],[0]*3,s=3.5,color=color,alpha=.4,zorder=3)
   z.scatter([val],[0],s=9,marker='D' if val>=0 else 's',color=color,zorder=4)
  if i>=2:txt(fig,x+18,12,'Dev.',8,MUTED,'center')
 txt(fig,5,3.9,'MSE@5 reduction vs Framewise (%) · positive = lower error · 3 seeds',8,MUTED)
 return fig

def check(fig):
 fig.canvas.draw();rr=fig.canvas.get_renderer();ts=[];issues=[]
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text().strip() or (t.axes is not None and not t.axes.axison):continue
  bb=t.get_window_extent(rr);ts.append((t.get_text(),bb))
  if t.get_fontsize()<8:issues.append('font '+t.get_text())
  if min(bb.x0,bb.y0)<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:issues.append('clip '+t.get_text())
 for i,(t,b) in enumerate(ts):
  for u,c in ts[i+1:]:
   if b.overlaps(c):issues.append('overlap '+t+' / '+u)
 return {'size_inches':[W/72,H/72],'minimum_font_pt':8,'issues':issues}

def main():
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'shiftwm-expanded-diag-v2','image.composite_image':False})
 a,b,m=load();fig=render(a,b);qa=check(fig)
 OUT.mkdir(parents=True,exist_ok=True)
 for ext in ('pdf','svg','png'):fig.savefig(OUT/f'{NAME}.{ext}',dpi=240,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {})
 fig.savefig(OUT/f'{NAME}_paper_width.png',dpi=120);plt.close(fig)
 e={'status':'draft_pending_review' if qa['issues'] else 'source_validated','renderer_sha256':sha(__file__),'sources':m,'qa':qa,'figure_scope':'PanelA current spatial decoder snapshot; PanelB separate historical context model forecast results, not generalization of the spatial model.','outputs':{ext:sha(OUT/f'{NAME}.{ext}') for ext in ('pdf','svg','png')}}
 (OUT/f'{NAME}_evidence.json').write_text(json.dumps(e,indent=2)+'\n');print(json.dumps(qa,indent=2))
if __name__=='__main__':main()
