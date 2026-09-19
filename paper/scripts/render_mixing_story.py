#!/usr/bin/env python3
"""Compact measured mixing diagnostics, derived only from the reviewed replay.

No inference, new case selection, image modification, or scientific-source edits.
The canonical include is changed only by an explicit --integrate invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.text import Text
import numpy as np
from PIL import Image

import render_editorial_qualitative as source

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/editorial'
DESIGN=ROOT/'paper/design/mixing_story'
W,H=5.5,2.35
INK='#26364B'; MUTED='#657587'; GRID='#DCE4EA'
COLORS=('#4477AA','#287C7A','#B88746')
STYLES=('-', (0,(4,2)), (0,(1,1.6)))
MARKERS=('o','s','D')
CASE_NAMES=('A · gain','B · median','C · regression')
PINS={
 'paper/scripts/render_editorial_qualitative.py':'a7ce31f15fce8383f90f83aa236bdcf16f4f45f1cf5aad9c0003c03b4d2e8630',
 'paper/scripts/render_mixing_diagnostics.py':'ceb6ac88312358e00f385674f93728eed9ef219c71442cb6118b39ae69b34f52',
 'paper/generated/editorial/mixing_diagnostics_evidence.json':'e6780df6799b8df327eb62c0bc7fd0a6d870acadf5ec4a7230c88297900ef3f4',
 'paper/figure_sources/spatial_qualitative/manifest.json':'baf970ebac0c22482fd46243bc97b17f1d162002aebaf793a9d192bc76939577',
 'paper/figure_sources/spatial_qualitative/replay.json':'12d805f0692f9cd0068f7eb392cf9be996dd778f1fff00f5c1ee4273cfffac20',
 'paper/figure_sources/spatial_qualitative/replay_arrays.npz':'816c6e918835c89df64b28f4dabee991546aab6cc7c03741269c2d17daf17cb3',
}
CAPTION=(r'\textbf{Measured anchoring and gate trajectories.} '
 r'A/B/C are the largest-gain, median and largest-regression episodes; each diagnostic uses its first eligible window, as in Figure~\ref{fig:editorial-qualitative}. '
 r'At target patch $j=5$ (row 2, column 2), maps show raw source weights $T_{10}[j,:]$ and effective weights $M_{10}[j,:]$. '
 r'$M_h=(1-g_h)I+g_hT_h$ is computed within each seed before averaging. '
 r'All six maps share a linear $[0,1]$ scale; the amber outline and numbers identify the own-source weight. '
 r'Gate curves show three-seed means and min--max ranges, not confidence intervals; each forecast step uses five native commands. '
 r'Full images are recorded $f10$ observations (DROID, CC BY 4.0). '
 r'These latent-feature diagnostics do not establish physical correspondence or explain error differences causally.')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify():
 for p,h in PINS.items():
  if sha(ROOT/p)!=h:raise ValueError('Pinned source changed: '+p)
 replay,arrays,_,_=source.verified_pack()
 old=json.loads((ROOT/'paper/generated/editorial/mixing_diagnostics_evidence.json').read_text())
 records=[];check_errors=[]
 for i,c in enumerate(replay['cases']):
  k=c['prefix'];T=np.stack([arrays[f'{k}_transport_s{s}_matrix'] for s in range(3)])
  G=np.stack([arrays[f'{k}_transport_s{s}_gate'] for s in range(3)])
  E=np.stack([arrays[f'{k}_transport_s{s}_effective_mixture'] for s in range(3)])
  assert T.shape==(3,10,16,16) and G.shape==(3,10,16,1) and E.shape==T.shape
  assert np.isfinite(T).all() and np.isfinite(G).all() and np.isfinite(E).all()
  assert min(T.min(),G.min(),E.min())>=0 and max(T.max(),G.max(),E.max())<=1
  computed=(1-G)*np.eye(16,dtype=np.float32)+G*T
  np.testing.assert_allclose(E,computed,rtol=2e-6,atol=2e-7)
  np.testing.assert_allclose(T.sum(-1),1,rtol=2e-6,atol=2e-7)
  np.testing.assert_allclose(E.sum(-1),1,rtol=2e-6,atol=2e-7)
  raw=T[:,9,5,:];effective=E[:,9,5,:];gate=G[:,:,5,0]
  r={'case':k,'case_letter':'ABC'[i],'episode_id':c['episode_id'],
   'ranking_basis':'three-seed episode-mean h10 relative error gain, descending; largest, median, worst',
   'window_start':c.get('window_start',0),'target_patch_zero_based':5,'target_patch_one_based':[2,2],
   'episode_gain_percent':c['gain_percent'],'shown_window_gain_percent':c['first_window_gain_percent'],
   'raw_h10_weights_seedwise':raw.tolist(),'effective_h10_weights_seedwise':effective.tolist(),
   'raw_h10_weights':raw.mean(0).tolist(),'effective_h10_weights':effective.mean(0).tolist(),
   'raw_own_source_weight':float(raw.mean(0)[5]),'effective_own_source_weight':float(effective.mean(0)[5]),
   'gate_curve_seedwise':gate.tolist(),'gate_mean':gate.mean(0).tolist(),
   'gate_min':gate.min(0).tolist(),'gate_max':gate.max(0).tolist(),
   'gate_mean_h10':float(gate.mean(0)[9]),
   'frame':'paper/figure_sources/spatial_qualitative/'+f'{k}_recorded_frame_10.png'}
  for key in ('raw_h10_weights','effective_h10_weights','gate_curve_seedwise','gate_mean_h10'):
   assert np.array_equal(np.asarray(r[key]),np.asarray(old['records'][i][key])),key
  np.testing.assert_array_equal(arrays[k+'_images'][2],np.asarray(Image.open(ROOT/r['frame']).convert('RGB')))
  r['frame_sha256']=sha(ROOT/r['frame']);records.append(r)
  check_errors.append(float(np.max(np.abs(E-computed))))
 return replay,arrays,records,check_errors

def style():
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix',
  'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'text.color':INK,
  'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
  'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'shiftwm-mixing-story-v1',
  'image.composite_image':False,'savefig.facecolor':'white'})

def axis(fig,x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def tx(fig,x,y,s,size=8,color=INK,ha='center',**kw):
 return fig.text(x/W,y/H,s,fontsize=size,color=color,ha=ha,va='center',**kw)

def photo(fig,arrays,i,x,y,w):
 ax=axis(fig,x,y,w,w*180/320)
 ax.imshow(arrays[f'case{i}_images'][2],interpolation='none');ax.set_axis_off()

def weightmap(fig,r,key,x,y,w):
 ax=axis(fig,x,y,w,w)
 im=ax.imshow(np.asarray(r[key]).reshape(4,4),cmap='Blues',norm=Normalize(0,1,clip=False),interpolation='nearest')
 ax.set(xticks=[],yticks=[])
 # Four by four source locations, never channel-depth glyphs or physical arrows.
 ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
 ax.grid(which='minor',color='white',lw=.35);ax.tick_params(which='minor',length=0)
 ax.add_patch(Rectangle((.5,.5),1,1,fill=False,ec='#B88746',lw=1.05))
 for spine in ax.spines.values():spine.set_color(GRID);spine.set_linewidth(.45)
 tx(fig,x+w/2,y-.085,f'{r[key][5]:.3f}',color=MUTED)
 return im

def curves(fig,records,x,y,w,h,labels=True):
 ax=axis(fig,x,y,w,h);steps=np.arange(1,11)
 for i,r in enumerate(records):
  ax.fill_between(steps,r['gate_min'],r['gate_max'],color=COLORS[i],alpha=.12,lw=0)
  ax.plot(steps,r['gate_mean'],color=COLORS[i],ls=STYLES[i],lw=1.15,
   marker=MARKERS[i],ms=2.8,markevery=[0,4,9],mew=.35,mec='white')
  if labels:ax.annotate(r['case_letter'],(10,r['gate_mean'][-1]),xytext=(5,0),textcoords='offset points',
    va='center',ha='left',fontsize=8,color=COLORS[i],annotation_clip=False)
 ax.set(xlim=(1,10),ylim=(0,1),xticks=[1,5,10],yticks=[0,.5,1]);ax.set_yticklabels(['0','0.5','1'])
 ax.grid(axis='y',color=GRID,lw=.45,zorder=0);ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False)
 for side in ('left','bottom'):ax.spines[side].set_color('#B7C3CC');ax.spines[side].set_linewidth(.6)
 ax.tick_params(length=2,pad=2)
 return ax

def colorbar(fig,im,x,y,w,h,orientation='vertical'):
 cb=fig.colorbar(im,cax=axis(fig,x,y,w,h),orientation=orientation)
 cb.set_ticks([0,.5,1]);cb.set_ticklabels(['0','0.5','1']);cb.ax.tick_params(length=1.8,pad=2,labelsize=8)
 cb.outline.set_linewidth(.45);cb.outline.set_edgecolor('#B7C3CC')
 return cb

def compose(layout,arrays,records):
 fig=plt.figure(figsize=(W,H),dpi=180,facecolor='white')
 if layout=='A':
  # Case rows expose the gating operation once, beside shared-scale temporal evidence.
  for x,label in ((.495,'Case · f10'),(1.345,r'Raw $T_{10}$'),(2.085,r'Gated $M_{10}$')):
   tx(fig,x,2.20,label,size=8.5)
  tx(fig,2.60,2.20,'Weight',size=8)
  tx(fig,4.17,2.20,'Gate trajectories',size=8.5)
  for i,(y,r) in enumerate(zip((1.54,.92,.30),records)):
   photo(fig,arrays,i,.08,y+.013,.83)
   tx(fig,.495,y-.087,CASE_NAMES[i],color=COLORS[i])
   im=weightmap(fig,r,'raw_h10_weights',1.10,y,.49)
   weightmap(fig,r,'effective_h10_weights',1.84,y,.49)
   fig.add_artist(FancyArrowPatch((1.64/W,(y+.245)/H),(1.79/W,(y+.245)/H),
     transform=fig.transFigure,arrowstyle='-|>',mutation_scale=5.2,lw=.65,color=MUTED))
  colorbar(fig,im,2.48,.30,.055,1.73)
  curves(fig,records,3.16,.42,1.94,1.59)
  tx(fig,4.13,.15,'Forecast step',color=MUTED)
 elif layout=='B':
  # Scene/paired-map triptych above a full-width shared gate chart.
  for i,r in enumerate(records):
   x=.10+i*1.80;tx(fig,x+.77,2.22,CASE_NAMES[i],size=8.5,color=COLORS[i])
   photo(fig,arrays,i,x,1.62,.58)
   im=weightmap(fig,r,'raw_h10_weights',x+.72,1.60,.41)
   weightmap(fig,r,'effective_h10_weights',x+1.23,1.60,.41)
   tx(fig,x+.925,2.095,r'$T_{10}$');tx(fig,x+1.435,2.095,r'$M_{10}$')
  ax=curves(fig,records,.43,.39,4.67,.80)
  tx(fig,2.76,.105,'Forecast step',color=MUTED)
  tx(fig,.11,1.29,'Gate',ha='left');tx(fig,4.24,1.38,'Source weight',ha='right',color=MUTED)
  colorbar(fig,im,4.38,1.35,.73,.055,'horizontal')
 elif layout=='C':
  # Operation-major matrix: three scenes across, raw/effective diagnostic rows below.
  for i,r in enumerate(records):
   x=.44+i*.72;photo(fig,arrays,i,x,1.77,.59)
   tx(fig,x+.295,2.20,r['case_letter'],size=8.5,color=COLORS[i])
   im=weightmap(fig,r,'raw_h10_weights',x+.05,1.075,.49)
   weightmap(fig,r,'effective_h10_weights',x+.05,.37,.49)
  tx(fig,.14,1.32,r'$T_{10}$');tx(fig,.14,.615,r'$M_{10}$')
  colorbar(fig,im,2.60,.37,.05,1.20)
  tx(fig,2.70,1.67,'Weight',color=MUTED)
  tx(fig,4.02,2.20,'Gate trajectories',size=8.5)
  curves(fig,records,3.13,.42,1.98,1.59)
  tx(fig,4.12,.15,'Forecast step',color=MUTED)
 else:raise ValueError(layout)
 return fig

def geometry(fig):
 fig.canvas.draw();ren=fig.canvas.get_renderer();labels=[];issues=[]
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text().strip():continue
  b=t.get_window_extent(ren)
  if t.get_fontsize()<8:issues.append('font below8: '+t.get_text())
  if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:issues.append('clipped: '+t.get_text())
  labels.append((t.get_text(),b))
 for i,(s,b) in enumerate(labels):
  for q,c in labels[i+1:]:
   if b.overlaps(c):issues.append('text overlap: '+s+' / '+q)
 if issues:raise ValueError(issues)
 return {'status':'passed','min_effective_font_pt':8,'dimensions_inches':[W,H],
  'checks':'All visible text bounds/font sizes and figure-level label intersections; actual pixels reviewed separately.'}

def save(fig,stem):
 for ext in ('pdf','svg','png'):
  md={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {}
  fig.savefig(str(stem)+'.'+ext,dpi=300,metadata=md)

def proof():
 out=DESIGN/'proof';out.mkdir(parents=True,exist_ok=True)
 for name,extra,dpi in [('paper_width',[],120),('grayscale',['-gray'],120),('detail',[],300)]:
  subprocess.run(['pdftoppm','-singlefile','-png','-r',str(dpi),*extra,str(OUT/'mixing_story.pdf'),str(out/name)],check=True,capture_output=True)
 tex=r'''\documentclass{article}
\usepackage{iclr2027_conference,times,graphicx,amsmath,hyperref}
\iclrfinalcopy
\begin{document}
\refstepcounter{figure}\label{fig:editorial-qualitative}
\input{generated/editorial/mixing_story_figure}
\end{document}
'''
 (out/'proof.tex').write_text(tex)
 env=dict(os.environ,TEXINPUTS=str(ROOT/'paper')+':'+str(ROOT/'paper/template/official/iclr2027')+':')
 for _ in range(2):
  r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={out}',str(out/'proof.tex')],cwd=ROOT,env=env,capture_output=True,text=True)
  if r.returncode:raise RuntimeError(r.stdout[-6000:])
 subprocess.run(['pdftoppm','-singlefile','-png','-r','120',str(out/'proof.pdf'),str(out/'official_page')],check=True,capture_output=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--integrate',action='store_true');p.add_argument('--proof',action='store_true');args=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True);style()
 replay,arrays,records,errors=verify();audits={}
 with PdfPages(DESIGN/'composition_candidates.pdf',metadata={'CreationDate':None,'ModDate':None}) as pages:
  for name in ('A','B','C'):
   fig=compose(name,arrays,records);audits[name]=geometry(fig);save(fig,DESIGN/f'candidate_{name}');pages.savefig(fig);plt.close(fig)
 fig=compose('A',arrays,records);geometry(fig);save(fig,OUT/'mixing_story');plt.close(fig)
 (OUT/'mixing_story_caption.tex').write_text(CAPTION+'\n')
 include='\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/mixing_story.pdf}\n\\caption[Measured spatial mixing and gate behavior.]{'+CAPTION+'}\n\\label{fig:mixing-diagnostics}\n\\end{figure}\n'
 (OUT/'mixing_story_figure.tex').write_text(include)
 if args.integrate:(OUT/'mixing_diagnostics_figure.tex').write_text(include)
 source_bindings=dict(PINS)
 manifest=json.loads((source.PACK/'manifest.json').read_text())
 for name,rec in manifest['files'].items():source_bindings[str((source.PACK/name).relative_to(ROOT))]=rec['sha256']
 ledger={'status':'numeric_verified_pending_independent_visual_review','created_at_utc':datetime.now(timezone.utc).isoformat(),
  'renderer_sha256':sha(__file__),'sources_sha256':source_bindings,'dimensions_inches':[W,H],
  'selected_layout':'A','selection_reason':'Aligned case rows retain scene identity and paired weight maps; shared gate axes compare all three trajectories without repetitive microplots.',
  'layout_alternatives':{'A':'Case rows plus one shared temporal chart','B':'Scene-and-map triptych above a wide shared temporal chart','C':'Operation-major two-row/three-case map matrix beside a shared temporal chart'},
  'geometry':audits,'records':records,'maps':{'count':6,'cells_each':16,'norm':'linear','vmin':0,'vmax':1,'clipping':False,'color_map':'Blues','own_source_zero_based':5},
  'gate_curves':{'seedwise_values':90,'means':30,'uncertainty':'min/max of three seed values at each step; not confidence intervals','y_limits':[0,1],'x_values':list(range(1,11))},
  'effective_mixture_check':{'scope':'Every entry for all3cases x3seeds x10horizons x16x16','max_abs_error_per_case':errors,'rule':'M=(1-g)I+gT within each seed; then average stored FP32 matrices across seeds'},
  'same_as_existing_diagnostics':'Exact numeric equality checked for all96 displayed map values, all90 gate values, and three gate endpoints against original diagnostic ledger.',
  'image_policy':'Full unchanged recorded frame10 for every case; verified_pack checks exact RGB equality to replay arrays. No cropping, generated evidence or learned channel-feature glyphs.',
  'scope':'Original DROID development diagnostics for prespecified episode-level best/median/worst cases and their first eligible windows. No new model computation, selection, test claim, physical correspondence or causal attribution.',
  'integrated':args.integrate}
 (OUT/'mixing_story_evidence.json').write_text(json.dumps(ledger,indent=2)+'\n')
 if args.proof:proof()
 assert all(sha(ROOT/path)==expected for path,expected in source_bindings.items())
 print(json.dumps({'maps':6,'displayed_map_cells':96,'gate_values':90,'gate_means':30,'exact_original_ledger_parity':True,'dimensions':[W,H],'integrated':args.integrate,'pdf_sha256':sha(OUT/'mixing_story.pdf')}))

if __name__=='__main__':main()
