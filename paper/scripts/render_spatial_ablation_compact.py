from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'paper/figure_sources/spatial_ablation_compact'
DATA=PACK/'data.json';OUT=ROOT/'paper/generated/editorial'
INK='#243447';MUTED='#627384';LINE='#dce3e9';BLUE='#397BB5';TEAL='#16847B';GREEN='#166534';AMBER='#A45931'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'spatial-ablation-compact-v2'})
W=396

def txt(fig,x,y,text,size=8,**kw):return fig.text(x/W,1-y/fig.get_size_inches()[1]/72,text,fontsize=size,va='center',color=kw.pop('color',INK),**kw)
def axes(fig,x,y,w,h,rows=6):
 H=fig.get_size_inches()[1]*72;ax=fig.add_axes([x/W,1-(y+h)/H,w/W,h/H]);ax.set_xlim(-2.8,13.5);ax.set_xticks([0,5,10]);ax.set_ylim(rows-.5,-.5);ax.set_yticks([]);ax.tick_params(labelsize=8,length=2,pad=3,colors=MUTED)
 ax.axvline(0,c=MUTED,lw=.7,ls=(0,(3,2)));ax.grid(axis='x',color=LINE,lw=.5)
 for side in ['top','left','right']:ax.spines[side].set_visible(False)
 ax.spines['bottom'].set_color(LINE);return ax

def point(ax,e,y,h):
 x=e['gain_x1000'];lo,hi=e['ci95_x1000'];col=BLUE if h==5 else TEAL;marker='o' if h==5 else 'D'
 ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,ms=3.8,color=col,markerfacecolor='white' if lo<=0<=hi else col,elinewidth=1.05,capsize=2,capthick=.8,mew=.9,zorder=4)
 assert ax.get_xlim()[0]<lo<hi<ax.get_xlim()[1]

def audits(fig):
 fig.canvas.draw();ren=fig.canvas.get_renderer();boxes=[];issues=[];bb=fig.bbox
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text().strip():continue
  b=t.get_window_extent(ren)
  if not b.width or not b.height:continue
  if t.get_fontsize()<8:issues.append(['small_font',t.get_text()])
  if b.x0<bb.x0-.5 or b.y0<bb.y0-.5 or b.x1>bb.x1+.5 or b.y1>bb.y1+.5:issues.append(['clip',t.get_text()])
  boxes.append((t.get_text(),b))
 for i,(s,b) in enumerate(boxes):
  for t,c in boxes[i+1:]:
   if min(b.x1,c.x1)-max(b.x0,c.x0)>1 and min(b.y1,c.y1)-max(b.y0,c.y0)>1:issues.append(['overlap',s,t])
 return {'size_inches':list(fig.get_size_inches()),'minimum_font_pt':8,'issues':issues}

def save(fig,name):
 qa=audits(fig)
 for ext in ['pdf','svg','png']:
  md={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
  fig.savefig(OUT/(name+'.'+ext),dpi=240,metadata=md)
 fig.savefig(OUT/(name+'_paper_width.png'),dpi=120);plt.close(fig)
 Image.open(OUT/(name+'_paper_width.png')).convert('L').save(OUT/(name+'_grayscale.png'))
 (OUT/(name+'_layout.json')).write_text(json.dumps(qa,indent=2)+'\n');return qa

def candidate_a(d):
 fig=plt.figure(figsize=(5.5,2.75));H=198
 txt(fig,5,10,'ShiftWM (ours) versus matched spatial controls',8.8,weight='bold')
 txt(fig,5,30,'Comparator',color=MUTED)
 for h,x in [(5,150),(10,281)]:
  txt(fig,x+42,30,f'h{h} endpoint',8.5,ha='center',weight='bold');ax=axes(fig,x,45,83,110)
  for i,row in enumerate(d['rows']):
   e=row['endpoints'][str(h)];point(ax,e,i,h)
   txt(fig,x+111,45+(i+.5)*110/6,f"{e['relative_gain_percent']:+.2f}%",ha='right',color=GREEN if e['relative_gain_percent']>0 else AMBER,weight='bold' if e['relative_gain_percent']>0 else 'normal')
 for i,row in enumerate(d['rows']):txt(fig,5,45+(i+.5)*110/6,row['label'])
 txt(fig,198,179,r'Native MSE reduction ($\times10^{-3}$); right favors ours.',ha='center')
 txt(fig,198,192,'Paired 95% CI; open marks cross zero. Labels: relative gains.',ha='center',color=MUTED)
 return save(fig,'spatial_ablation_compact')


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=json.loads((PACK/'manifest.json').read_text())
    for name,digest in manifest['runtime_inputs'].items():assert sha(ROOT/name)==digest,name
    data=json.loads(DATA.read_text());assert len(data['rows'])==6
    full=data['full_36_contrast_ledger'];source={x['result_id']:x for x in full['all_16_effects']+full['all_20_component_effects']}
    for row in data['rows']:
        for h,e in row['endpoints'].items():
            raw=source[e['source_result_id']]
            assert raw['metric']=='native_mse' and raw['method']=='transport' and raw['comparator']==row['comparator'] and raw['horizon']==int(h)
            np.testing.assert_allclose(e['gain_x1000'],1000*(raw['comparator_mean']-raw['method_mean']),atol=1e-13)
            assert e['ci95_x1000']==[-1000*raw['paired_95_percent_interval'][1],-1000*raw['paired_95_percent_interval'][0]]
            np.testing.assert_allclose(e['relative_gain_percent'],100*(1-raw['method_mean']/raw['comparator_mean']),atol=1e-12)
    OUT.mkdir(parents=True,exist_ok=True);qa=candidate_a(data);assert not qa['issues'],qa
    rec={'status':'passed_source_and_layout_checks','scope':data['scope'],'layout':qa,'source_inputs':manifest['runtime_inputs'],'manifest_sha256':sha(PACK/'manifest.json'),'paired_contrasts_displayed':12,'complete_contrasts_retained':36,'absolute_scores_retained':'Table tab:editorial-spatial and full source data','new_predictions':False,'outputs_sha256':{e:sha(OUT/('spatial_ablation_compact.'+e)) for e in ['pdf','svg','png']}}
    (OUT/'spatial_ablation_compact.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps({'status':rec['status'],'pdf_sha256':rec['outputs_sha256']['pdf']}))
if __name__=='__main__':main()
