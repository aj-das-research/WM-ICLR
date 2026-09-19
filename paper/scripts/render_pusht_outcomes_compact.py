"""Portable PushT endpoint plate from immutable saved historical observations."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'paper/generated/editorial';PACK=ROOT/'paper/figure_sources/pusht_outcomes_compact';W=396
INK='#243447';MUTED='#647487';BLUE='#4477AA';GREEN='#24734C';RED='#A6563D';LINE='#D7E0E6';AMBER='#936B32'
MODES=['factorized','framewise'];NAMES=['ShiftWM','Framewise'];TITLES=['ShiftWM only','Framewise only','Both fail','Support only']
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'text.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'pusht-outcomes-v2','image.composite_image':False})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Scene:
 def __init__(self,height=226):
  self.H=height;self.fig=plt.figure(figsize=(W/72,height/72),facecolor='white');self.photos=[]
 def text(self,x,y,text,size=8,color=INK,ha='left',bold=False):
  self.fig.text(x/W,1-y/self.H,text,fontsize=size,color=color,ha=ha,va='center',weight='bold' if bold else 'normal')
 def line(self,x1,y1,x2,y2):
  self.fig.add_artist(Line2D([x1/W,x2/W],[1-y1/self.H,1-y2/self.H],transform=self.fig.transFigure,color=LINE,lw=.6))
 def photo(self,name,x,y,w,color=LINE):
  a=np.asarray(Image.open(PACK/name).convert('RGB'));assert a.shape==(224,224,3)
  ax=self.fig.add_axes([x/W,1-(y+w)/self.H,w/W,w/self.H]);ax.imshow(a,interpolation='none');ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_edgecolor(color);spine.set_linewidth(.8 if color!=LINE else .5)
  self.photos.append({'file':name,'x':x,'y':y,'w':w,'RGB_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
 def export(self,stem):
  self.fig.canvas.draw();ren=self.fig.canvas.get_renderer();texts=[];issues=[]
  for t in self.fig.findobj(Text):
   if not t.get_visible() or not t.get_text().strip():continue
   if t.axes is not None and not t.axes.axison:continue
   b=t.get_window_extent(ren)
   if t.get_fontsize()<8:issues.append(['small_font',t.get_text()])
   if b.x0<-.1 or b.y0<-.1 or b.x1>self.fig.bbox.width+.1 or b.y1>self.fig.bbox.height+.1:issues.append(['clip',t.get_text()])
   texts.append((t.get_text(),b))
  for i,(a,b) in enumerate(texts):
   for c,d in texts[i+1:]:
    if min(b.x1,d.x1)-max(b.x0,d.x0)>.8 and min(b.y1,d.y1)-max(b.y0,d.y0)>.8:issues.append(['overlap',a,c])
  for ext in ['pdf','svg','png']:
   md={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {}
   self.fig.savefig(HERE/f'{stem}.{ext}',dpi=220,metadata=md)
  self.fig.savefig(HERE/f'{stem}_paper_width.png',dpi=120);plt.close(self.fig)
  ImageOps.grayscale(Image.open(HERE/f'{stem}_paper_width.png')).save(HERE/f'{stem}_grayscale.png')
  qa={'size_inches':[5.5,self.H/72],'minimum_font_pt':8,'issues':issues,'photos':self.photos}
  (HERE/f'{stem}.json').write_text(json.dumps(qa,indent=2)+'\n');return qa

def outcome_card(s,c,i,x,y):
 s.text(x+4,y+7,f'{chr(97+i)}  {TITLES[i]} · {c["seed"]}',8.5,bold=True)
 xs=[x+4,x+51,x+100,x+151];labels=['Goal','Support']+NAMES
 files=[c['goal_png'],c['support_png']]+[c['methods'][m]['final_png'] for m in MODES]
 for j,(xx,name,title) in enumerate(zip(xs,files,labels)):
  s.text(xx+20,y+20,title,color=BLUE if j==2 else MUTED,ha='center')
  color=LINE if j<2 else GREEN if c['methods'][MODES[j-2]]['success'] else RED
  s.photo(name,xx,y+27,40,color)
 s.text(xs[1]+20,y+77,'t=10',color=MUTED,ha='center')
 s.text(x+4,y+91,'Block / pusher (px)',color=MUTED)
 s.text(x+4,y+104,'Angle (rad)',color=MUTED)
 for j,m in enumerate(MODES):
  r=c['methods'][m];xx=xs[j+2]+20;color=GREEN if r['success'] else RED
  s.text(xx,y+77,f'{"Success" if r["success"] else "Failure"} · {r["native_calls"]}',color=color,ha='center')
  s.text(xx,y+91,f'{r["block_px"]:.1f} / {r["pusher_px"]:.1f}',ha='center')
  s.text(xx,y+104,f'{r["angle_rad"]:.3f}',ha='center')

def support_card(s,c,x,y):
 s.text(x+4,y+7,f'd  Support only · {c["seed"]}',8.5,bold=True)
 s.text(x+28,y+20,'Goal',color=MUTED,ha='center')
 s.text(x+91,y+20,'Support = end',color=MUTED,ha='center')
 s.photo(c['goal_png'],x+4,y+27,48)
 s.photo(c['support_png'],x+67,y+27,48,GREEN)
 s.text(x+125,y+36,'Success · 2',color=GREEN)
 s.text(x+125,y+50,'All methods',color=MUTED)
 s.text(x+125,y+65,'No CEM plan',color=MUTED)
 r=c['methods']['factorized']
 s.text(x+4,y+91,f'Block {r["block_px"]:.1f} / pusher {r["pusher_px"]:.1f} px; angle {r["angle_rad"]:.3f} rad')
 s.text(x+4,y+104,'Excluded from eligible policy success.',color=AMBER)

def draft_a(cases):
 s=Scene(226)
 for i,c in enumerate(cases[:3]):outcome_card(s,c,i,(i%2)*200,(i//2)*114)
 support_card(s,cases[3],200,114)
 s.line(197,5,197,220);s.line(5,112,391,112)
 return s.export('pusht_outcomes_compact')

def main():
 m=json.loads((PACK/'manifest.json').read_text())
 for name,h in m['files'].items():
  assert Path(name).name==name and sha(PACK/name)==h,name
 d=json.loads((PACK/'data.json').read_text())
 assert [c['seed'] for c in d['cases']]==[2031024,2031000,2031001,2031026]
 assert [c['category'] for c in d['cases']]==['ours_only','baseline_only','neither','support_success']
 assert [[c['methods'][k]['native_calls'] for k in MODES] for c in d['cases']]==[[21,50],[50,48],[50,50],[2,2]]
 for c in d['cases']:
  for mode,r in c['methods'].items():
   assert bool(r['success'])==(np.hypot(r['block_px'],r['pusher_px'])<20 and r['angle_rad']<np.pi/9)
   if not r['eligible']:
    assert r['native_calls']==2 and not r['record']['executed_action_blocks'] and r['record']['num_replans']==0
    assert np.array_equal(np.asarray(Image.open(PACK/c['support_png'])),np.asarray(Image.open(PACK/r['final_png'])))
 HERE.mkdir(parents=True,exist_ok=True);qa=draft_a(d['cases']);assert not qa['issues'],qa['issues']
 sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),PACK/'manifest.json',PACK/'data.json',PACK/'caption.tex']}
 result={'status':'source_and_geometry_verified','scope':d['scope'],'source_sha256':sources,'layout':qa,
   'case_ids':[c['seed'] for c in d['cases']],'full_original_data_preserved':True,'displayed_photos':14,
   'total_archived_pack_photos':20,'shared_support_duplicate_collapse_verified':True,
   'no_model_compute_or_simulator_replay':True,'source_paths_are_archival_provenance_only':True,
   'exports_sha256':{ext:sha(HERE/f'pusht_outcomes_compact.{ext}') for ext in ['pdf','svg','png']}}
 (HERE/'pusht_outcomes_compact.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':'passed','PDF_sha256':result['exports_sha256']['pdf'],'min_font_pt':8}))

if __name__=='__main__':main()
