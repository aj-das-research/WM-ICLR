#!/usr/bin/env python3
"""Unique supplementary diagnostic: measured mixing weights before/after the gate."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import render_editorial_qualitative as source
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/mixing_diagnostics'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CAPTION=r'''\textbf{Measured mixing and gate behavior for the same three cases.} At the prespecified target patch (row 2, column 2), the first map shows the learned source weights $T_{10}[j,:]$ and the second shows the effective anchored weights $M_{10}[j,:]$, where $M_h=(1-g_h)I+g_hT_h$ rowwise. The outline marks the source at the target's own location; values underneath give its weight. Maps use one linear $[0,1]$ scale and average three seeds after applying the gate within each seed. Curves show the target patch's gate mean and seed range, not confidence intervals. These are measured feature mixtures, not physical motion or causal explanations of the error differences in Figure~\ref{fig:editorial-qualitative}.'''
def main():
 OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
 r,a,_,_=source.verified_pack();j=5;records=[]
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42})
 W,H=5.5,2.4;fig=plt.figure(figsize=(W,H),facecolor='white');ink='#243447';muted='#647482';teal='#287C7A';blue='#4477AA';gold='#B88746'
 def tx(x,y,s,**kw):fig.text(x/W,y/H,s,fontsize=kw.pop('size',8),ha=kw.pop('ha','center'),va='center',color=kw.pop('color',ink),**kw)
 def axat(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
 for i,c in enumerate(r['cases']):
  key=c['prefix'];left=.10+i*1.83
  T=np.stack([a[f'{key}_transport_s{s}_matrix'] for s in range(3)])
  G=np.stack([a[f'{key}_transport_s{s}_gate'] for s in range(3)])
  E=np.stack([a[f'{key}_transport_s{s}_effective_mixture'] for s in range(3)])
  assert T.shape==(3,10,16,16) and G.shape==(3,10,16,1)
  np.testing.assert_allclose(T.sum(-1),1,rtol=2e-6,atol=2e-7)
  np.testing.assert_allclose(E,(1-G)*np.eye(16,dtype=np.float32)+G*T,rtol=2e-6,atol=2e-7)
  assert np.isfinite(T).all() and T.min()>=0 and T.max()<=1 and G.min()>=0 and G.max()<=1
  tx(left+.77,2.25,['Largest gain','Median','Largest regression'][i],size=8.5,weight='bold')
  tx(left+.33,2.06,r'Raw $T_{10}$');tx(left+1.15,2.06,r'Gated $M_{10}$')
  for offset,v in [(0,T),( .82,E)]:
   arr=v[:,9,j,:].mean(0).reshape(4,4);ax=axat(left+offset+.07,1.44,.53,.53)
   im=ax.imshow(arr,cmap='Blues',vmin=0,vmax=1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   ax.add_patch(Rectangle((.5,.5),1,1,fill=False,ec=gold,lw=.9))
   for s in ax.spines.values():s.set_linewidth(.4);s.set_edgecolor('#CAD6DF')
   tx(left+offset+.33,1.31,f'Self {arr[1,1]:.3f}',color=muted)
  curve=G[:,:,j,0];ax=axat(left+.25,.59,1.30,.49);mean=curve.mean(0)
  ax.fill_between(np.arange(1,11),curve.min(0),curve.max(0),color=teal,alpha=.14,lw=0)
  ax.plot(np.arange(1,11),mean,color=teal,lw=1,marker='o',ms=2.4,markevery=[0,4,9]);ax.set(xlim=(1,10),ylim=(0,1),xticks=[1,10],yticks=[0,1]);ax.tick_params(length=2,pad=2)
  ax.spines[['top','right']].set_visible(False)
  for s in ('left','bottom'):ax.spines[s].set_color('#B6C3CE');ax.spines[s].set_linewidth(.55)
  tx(left+.87,1.17,f'Gate at h10: {mean[9]:.3f}',color=teal)
  tx(left+.90,.36,'Forecast step',color=muted)
  records.append({'case':key,'episode_id':c['episode_id'],'target_patch_zero_based':j,'raw_h10_weights':T[:,9,j,:].mean(0).tolist(),'effective_h10_weights':E[:,9,j,:].mean(0).tolist(),'gate_curve_seedwise':curve.tolist(),'gate_mean_h10':float(mean[9])})
 bar=fig.colorbar(im,cax=axat(2.16,.07,1.12,.055),orientation='horizontal');bar.set_ticks([0,1]);bar.ax.tick_params(length=1.5,labelsize=8,pad=1);bar.outline.set_linewidth(.45);bar.ax.xaxis.set_ticks_position('top')
 tx(1.98,.10,'Source weight',ha='right',color=muted)
 # Keep tick labels within the figure, including the lower color scale.
 fig.canvas.draw();ren=fig.canvas.get_renderer();clip=[]
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text():continue
  b=t.get_window_extent(ren)
  if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:clip.append(t.get_text())
 assert not clip,clip
 for ext in ('pdf','svg','png'):fig.savefig(OUT/f'mixing_diagnostics.{ext}',dpi=300)
 fig.savefig(DESIGN/'paper_width.png',dpi=120);plt.close(fig)
 (OUT/'mixing_diagnostics_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/mixing_diagnostics.pdf}\n\\caption[Measured spatial mixing and gate behavior.]{'+CAPTION+'}\n\\label{fig:mixing-diagnostics}\n\\end{figure}\n')
 (OUT/'mixing_diagnostics_evidence.json').write_text(json.dumps({'status':'numeric_checked_pending_review','script_sha256':sha(__file__),'replay_sha256':source.EXPECTED_REPLAY,'arrays_sha256':sha(source.PACK/'replay_arrays.npz'),'dimensions_inches':[W,H],'target_patch_zero_based':j,'records':records,'scientific_scope':'Existing recorded diagnostic arrays, fixed3cases, no new inference or selection. Raw/effective source weights are distinct from visual error maps. Gate/error co-occurrence is not causal evidence.'},indent=2)+'\n')
 print(json.dumps({'cases':len(records),'effective_mixture_check':'all3seeds/all10horizons/all16rows','dimensions':[W,H]}))
if __name__=='__main__':main()
