#!/usr/bin/env python3
"""Small-multiple forecast profile + paired interval forest; frozen source reuse."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import render_spatial_editorial as source
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/results_refined'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
INK='#243447';MUTED='#647482';BLUE='#4477AA';TEAL='#287C7A';GOLD='#B88746';GRID='#E2E8EC'
CAPTION=r'''\textbf{Forecast accuracy and paired improvements.} (a) Native mean feature error at each step on DROID development data. (b) Comparator-minus-ShiftWM endpoint MSE at steps 5 and 10, with the original paired 95\% session/seed bootstrap intervals. Both error and gain axes retain zero. The interval unit is MSE, not relative percent. Table~\ref{tab:editorial-spatial} retains every spatial arm; Appendix~\ref{app:spatial-development} retains all 36 contrasts, including null and unfavorable ablations. Each forecast step spans five native commands.'''
def main():
 OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
 d,values=source.load_evidence()
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'stix','text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED})
 modes=('transport','autoregressive','persistence','anchored_additive','action_free')
 colors=dict(zip(modes,[TEAL,INK,'#9AA5AD',BLUE,GOLD]));labels=['ShiftWM (ours)','Autoregressive','Persistence','Additive anchor','No actions']
 fig=plt.figure(figsize=(5.5,2.2),facecolor='white');fig.text(.04,.944,'a  Forecast error',fontsize=8.7,weight='bold');fig.text(.53,.944,'b  Paired gains',fontsize=8.7,weight='bold')
 ax=fig.add_axes([.085,.275,.385,.575])
 for mode in modes[1:]+modes[:1]:
  ax.plot(np.arange(1,11),values[mode],color=colors[mode],ls=source.LINES[mode],marker=source.MARKERS[mode],markevery=[0,4,9] if mode=='transport' else [1,5,8],ms=2.7,lw=1.0,zorder=3 if mode=='transport' else 2,label=mode)
 ax.set(xlim=(.7,10.3),ylim=(0,.25),xticks=[1,5,10],yticks=[0,.1,.2],xlabel='Forecast step',ylabel='Native feature MSE')
 ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color(MUTED);ax.spines[['left','bottom']].set_linewidth(.65);ax.tick_params(length=2,pad=2);ax.grid(axis='y',color=GRID,lw=.55)
 bx=fig.add_axes([.675,.275,.29,.575]);effects=[]
 for i,(mode,label) in enumerate(zip(source.MAIN_CONTROLS,['Autoreg.','Additive anchor','No actions'])):
  center=2-i
  bx.axhline(center,color=GRID,lw=.5,zorder=0)
  for h,offset,col,marker in [(5,.13,BLUE,'o'),(10,-.13,TEAL,'D')]:
   e=source.main_effect(d,mode,h);effects.append(e);value=e['mse_reduction_x1000'];lo,hi=e['reduction_ci95_x1000']
   assert 0<=lo<=value<=hi<=14
   bx.errorbar(value,center+offset,xerr=[[value-lo],[hi-value]],fmt=marker,color=col,ms=3,lw=.9,capsize=2,zorder=3)
 bx.set(xlim=(0,14),ylim=(-.48,2.48),xticks=[0,5,10],yticks=[0,1,2],yticklabels=['No actions','Additive anchor','Autoreg.'],xlabel='MSE reduction '+r'$\times10^{-3}$')
 bx.spines[['top','right','left']].set_visible(False);bx.spines['bottom'].set_color(MUTED);bx.spines['bottom'].set_linewidth(.65);bx.tick_params(axis='x',length=2,pad=2);bx.tick_params(axis='y',length=0,pad=5);bx.axvline(0,color=MUTED,lw=.65)
 fig.legend([Line2D([],[],color=BLUE,marker='o',ls='none',ms=3),Line2D([],[],color=TEAL,marker='D',ls='none',ms=3)],['h5','h10'],loc='upper right',bbox_to_anchor=(.982,1.008),ncol=2,frameon=False,handlelength=.5,handletextpad=.35,columnspacing=.65,fontsize=8)
 handles=[Line2D([],[],color=colors[m],ls=source.LINES[m],marker=source.MARKERS[m],ms=2.7,lw=1) for m in modes]
 fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.002),ncol=5,frameon=False,fontsize=8,handlelength=1.15,handletextpad=.3,columnspacing=.8)
 fig.canvas.draw();ren=fig.canvas.get_renderer();clipped=[]
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text():continue
  b=t.get_window_extent(ren)
  if b.x0<-1 or b.y0<-1 or b.x1>fig.bbox.x1+1 or b.y1>fig.bbox.y1+1:clipped.append(t.get_text())
 assert not clipped,clipped
 for ext in ('pdf','svg','png'):fig.savefig(OUT/f'results_refined.{ext}',dpi=300)
 fig.savefig(DESIGN/'paper_width.png',dpi=120);plt.close(fig)
 (OUT/'spatial_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/results_refined.pdf}\n\\caption[Forecast accuracy and paired improvements.]{'+CAPTION+'}\n\\label{fig:editorial-spatial}\n\\end{figure}\n')
 (OUT/'results_refined_evidence.json').write_text(json.dumps({'status':'sources_geometry_checked_pending_review','script_sha256':sha(__file__),'verified_helper_sha256':sha(Path(source.__file__)),'sources':d['source_sha256'],'means':{m:values[m].tolist() for m in modes},'effects':effects,'dimensions_inches':[5.5,2.2],'minimum_font':8,'clipped':clipped,'design':'Forecast profile beside aligned paired interval forest; no guide lines between unrelated effects. Same50means/sixCIs. Exact lookup and ablations retained.','references':['https://matplotlib.org/stable/gallery/statistics/errorbar_features.html','https://observablehq.com/plot/marks/difference']},indent=2)+'\n')
 print(json.dumps({'values':50,'intervals':6,'inches':[5.5,2.2],'clipped':clipped}))
if __name__=='__main__':main()
