#!/usr/bin/env python3
"""Compact source-bound historical Reacher outcomes; no model execution."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
ROOT=Path(__file__).resolve().parents[2];PACK=ROOT/'paper/figure_sources/reacher_outcomes_compact';OUT=ROOT/'paper/generated/editorial';W=396
INK='#243447';MUTED='#63768A';BLUE='#4477AA';GREEN='#24734C';RED='#A6563D';LINE='#D7E0E6'
MODES=['factorized','framewise'];NAMES=['ShiftWM','Framewise'];TITLES=['ShiftWM only','Framewise only','Both fail','Both succeed']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'text.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'reacher-outcomes-compact','image.composite_image':False})

def load():
 m=json.loads((PACK/'manifest.json').read_text())
 for f,h in m['files'].items():assert Path(f).name==f and sha(PACK/f)==h
 p=json.loads((PACK/'cases.json').read_text());assert [r['seed'] for r in p['cases']]==[2031004,2031001,2031002,2031000]
 assert [r['category'] for r in p['cases']]==['ours_only','baseline_only','neither','both']
 assert [[r['methods'][s]['native_calls'] for s in MODES] for r in p['cases']]==[[23,50],[50,41],[50,50],[33,36]]
 return p,m

class Scene:
 def __init__(self,H):
  self.H=H;self.fig=plt.figure(figsize=(W/72,H/72),facecolor='white');self.photos=[]
 def text(self,x,y,s,size=8,color=INK,ha='left',bold=False):
  return self.fig.text(x/W,1-y/self.H,s,fontsize=size,color=color,ha=ha,va='center',weight='bold' if bold else 'normal')
 def line(self,x1,y1,x2,y2):self.fig.add_artist(Line2D([x1/W,x2/W],[1-y1/self.H,1-y2/self.H],transform=self.fig.transFigure,color=LINE,lw=.65))
 def photo(self,file,x,y,w,color=LINE):
  a=np.asarray(Image.open(PACK/file).convert('RGB'));assert a.shape==(224,224,3)
  axis=self.fig.add_axes([x/W,1-(y+w)/self.H,w/W,w/self.H]);axis.imshow(a,interpolation='none');axis.set_xticks([]);axis.set_yticks([])
  for spine in axis.spines.values():spine.set_edgecolor(color);spine.set_linewidth(.8 if color!=LINE else .5)
  self.photos.append({'file':file,'x':x,'y':y,'w':w,'pixel_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
 def finish(self,stem):
  self.fig.canvas.draw();ren=self.fig.canvas.get_renderer();boxes=[];issues=[]
  for t in self.fig.findobj(Text):
   if not t.get_visible() or not t.get_text().strip() or (t.axes is not None and not t.axes.axison):continue
   bb=t.get_window_extent(ren);boxes.append((t.get_text(),bb))
   if t.get_fontsize()<8:issues.append('font '+t.get_text())
   if bb.x0<-.1 or bb.y0<-.1 or bb.x1>self.fig.bbox.width+.1 or bb.y1>self.fig.bbox.height+.1:issues.append('clip '+t.get_text())
  for i,(a,b) in enumerate(boxes):
   for c,d in boxes[i+1:]:
    if b.overlaps(d):issues.append('overlap '+a+' / '+c)
  for ext in ['pdf','svg','png']:self.fig.savefig(stem.with_suffix('.'+ext),dpi=240,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {})
  self.fig.savefig(stem.parent/(stem.name+'_paper_width.png'),dpi=120);plt.close(self.fig)
  ImageOps.grayscale(Image.open(stem.parent/(stem.name+'_paper_width.png'))).save(stem.parent/(stem.name+'_gray.png'))
  return {'height_inches':self.H/72,'minimum_font_pt':8,'issues':issues,'images':self.photos}

def card(s,c,i,x,y):
 s.text(x+4,y+7,f'{chr(97+i)}  {TITLES[i]} · {c["seed"]}',8.5,bold=True)
 starts=[x+4,x+51,x+100,x+151];files=[c['goal_png'],c['support_png']]+[c['methods'][m]['final_png'] for m in MODES]
 labels=['Goal','Support']+NAMES
 for j,(xx,file,label) in enumerate(zip(starts,files,labels)):
  s.text(xx+20,y+20,label,8,BLUE if j==2 else MUTED,'center');col=LINE if j<2 else GREEN if c['methods'][MODES[j-2]]['success'] else RED;s.photo(file,xx,y+27,40,col)
 s.text(starts[1]+20,y+76,'t=10',8,MUTED,'center');s.text(x+4,y+87,'Diagnostic L2 (rad)',8,MUTED)
 for j,m in enumerate(MODES):
  r=c['methods'][m];xx=starts[j+2]+20;col=GREEN if r['success'] else RED
  s.text(xx,y+76,f'{"Success" if r["success"] else "Failure"} · {r["native_calls"]}',8,col,'center')
  s.text(xx,y+87,f'{r["joint_l2_rad"]:.3f}',8,INK,'center')

def draft_a(cases):
 s=Scene(190)
 for i,c in enumerate(cases):card(s,c,i,(i%2)*200,(i//2)*96)
 s.line(197,5,197,184);s.line(5,94,391,94);return s


def main():
 OUT.mkdir(parents=True,exist_ok=True);p,m=load()
 qa=draft_a(p['cases']).finish(OUT/'reacher_outcomes_compact')
 if qa['issues']:raise ValueError(qa['issues'])
 evidence={'schema':'shiftwm_reacher_compact_figure_v1','scope':'Historical matched simulated executions; original four preselected strata, not success-frequency estimates.','source_sha256':sha(__file__),'pack_manifest_sha256':sha(PACK/'manifest.json'),'caption_sha256':sha(PACK/'caption.tex'),'qa':qa,'cases':p['cases'],'outputs_sha256':{ext:sha(OUT/f'reacher_outcomes_compact.{ext}') for ext in ['pdf','svg','png']},'native_drawio':'Not applicable: actual-image plate with editable SVG labels and deterministic vector renderer.'}
 (OUT/'reacher_outcomes_compact.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':'rendered','qa':{k:v for k,v in qa.items() if k!='images'},'outputs':evidence['outputs_sha256']},indent=2))
if __name__=='__main__':main()
