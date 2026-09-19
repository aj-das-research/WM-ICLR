"""Portable positive-case plate; saved observations and exact endpoint criteria only."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image, ImageOps, ImageDraw
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'paper/figure_sources/positive_cases_compact'
OUT=ROOT/'paper/generated/editorial'
DATA=json.loads((HERE/'data.json').read_text())
CASES=DATA['cases']
W=396
INK='#243447';MUTED='#65758a';BLUE='#4477aa';GRAY='#6b747e';GREEN='#287c7a';RED='#a65b36';PURPLE='#9a5aa3'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'positive-story-v2','text.color':INK})

class Plate:
    def __init__(self,name,h):
        self.name=name;self.h=h;self.fig=plt.figure(figsize=(W/72,h/72),dpi=180,facecolor='white')
    def text(self,x,y,s,size=8,color=INK,**kw):
        return self.fig.text(x/W,1-y/self.h,s,fontsize=size,color=color,va='center',**kw)
    def ax(self,x,y,w,h):return self.fig.add_axes([x/W,1-(y+h)/self.h,w/W,h/self.h])
    def line(self,x1,y1,x2,y2,color='#d8e0e6',lw=.6):self.fig.add_artist(Line2D([x1/W,x2/W],[1-y1/self.h,1-y2/self.h],transform=self.fig.transFigure,color=color,lw=lw))
    def image(self,c,mode,role,x,y,size,overview=False,overlay=True):
        record=c['methods'][mode]['images'][role];p=HERE/record['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
        rgb=np.array(Image.open(p));a=self.ax(x,y,size,size);a.imshow(rgb,interpolation='none')
        goal=np.array(Image.open(HERE/c['methods']['factorized']['images']['goal']['path']));r,g,b=goal.astype(np.int16).transpose(2,0,1);mask=(r>150)&(r-g>30)&(g-b>30)
        if overlay and role!='goal':a.contour(np.arange(224),np.arange(224),mask.astype(float),levels=[.5],colors=[PURPLE],linewidths=.6,linestyles='--')
        l,t,rr,bb=c['display_crop_xyxy']
        if overview:
            a.set_xlim(-.5,223.5);a.set_ylim(223.5,-.5);a.add_patch(Rectangle((l,t),rr-l,bb-t,fill=False,ec=PURPLE,lw=.6))
        else:a.set_xlim(l-.5,rr-.5);a.set_ylim(bb-.5,t-.5)
        a.set_xticks([]);a.set_yticks([])
        for spine in a.spines.values():spine.set_color((GREEN if mode=='factorized' else RED) if role=='end' else '#bac7d0');spine.set_linewidth(.9 if role=='end' else .45)
    def save(self):
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer();items=[];issues=[]
        for t in self.fig.findobj(Text):
            if not t.get_visible() or not t.get_text().strip():continue
            b=t.get_window_extent(renderer)
            if not b.width or not b.height:continue
            if t.get_fontsize()<8-1e-5:issues.append(['font',t.get_text()])
            if b.x0<-.5 or b.y0<-.5 or b.x1>self.fig.bbox.x1+.5 or b.y1>self.fig.bbox.y1+.5:issues.append(['clip',t.get_text()])
            for s,bb in items:
                if min(b.x1,bb.x1)>max(b.x0,bb.x0)+1 and min(b.y1,bb.y1)>max(b.y0,bb.y0)+1:issues.append(['overlap',s,t.get_text()])
            items.append((t.get_text(),b))
        for ext in ['pdf','svg','png']:
            self.fig.savefig(OUT/f'{self.name}.{ext}',dpi=240,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None)
        self.fig.savefig(OUT/f'{self.name}_paper_width.png',dpi=120)
        ImageOps.grayscale(Image.open(OUT/f'{self.name}_paper_width.png')).save(OUT/f'{self.name}_grayscale.png')
        (OUT/f'{self.name}_layout.json').write_text(json.dumps({'size_inches':[5.5,self.h/72],'minimum_font_pt':8,'issues':issues},indent=2)+'\n')
        plt.close(self.fig)
        print(self.name,issues)

def criterion(p,c,x,y,w,h,labels=True):
    a=p.ax(x,y,w,h);a.axvspan(0,.05,color='#e5f1ee');a.axvline(.05,color=GREEN,ls=':',lw=.9)
    for joint in range(2):
        ov=c['methods']['factorized']['endpoint_criterion_errors'][joint];fv=c['methods']['framewise']['endpoint_criterion_errors'][joint]
        a.plot([ov,fv],[1-joint,1-joint],color='#b4bdc4',lw=.8)
        a.scatter([ov],[1-joint],c=BLUE,s=15,marker='o',zorder=3)
        a.scatter([fv],[1-joint],c=GRAY,s=16,marker='s',zorder=3)
    a.set_xlim(0,.4);a.set_ylim(-.45,1.45);a.set_xticks([0,.2,.4]);a.set_xticklabels(['0','.2','.4']);a.set_yticks([1,0]);a.set_yticklabels(['J1','J2'] if labels else [])
    a.tick_params(labelsize=8,length=2,pad=2);a.spines[['top','right']].set_visible(False)
    for z in ['left','bottom']:a.spines[z].set_color('#aebbc5');a.spines[z].set_linewidth(.5)

def method_legend(p,x,y):
    p.fig.add_artist(Line2D([x/W],[1-y/p.h],marker='o',color=BLUE,ms=3.8,ls='none',transform=p.fig.transFigure));p.text(x+6,y,'ShiftWM (ours)',color=BLUE)
    p.fig.add_artist(Line2D([(x+81)/W],[1-y/p.h],marker='s',color=GRAY,ms=3.8,ls='none',transform=p.fig.transFigure));p.text(x+88,y,'Framewise',color=GRAY)

def draft_a():
    p=Plate('positive_cases_compact',210)
    p.text(6,8,'Two additional Reacher successes · historical context model',8.5,weight='bold')
    for i,c in enumerate(CASES):
        top=25+87*i;p.text(6,top,f"{'(a)' if i==0 else '(b)'}  Episode {c['seed']}",8.5,weight='bold')
        p.text(6,top+14,'Goal',color=MUTED)
        p.image(c,'factorized','goal',6,top+30,23,overview=True)
        p.image(c,'factorized','goal',36,top+27,39)
        for mode,x in [('factorized',84),('framewise',181)]:
            r=c['methods'][mode]['record'];col=BLUE if mode=='factorized' else GRAY
            p.text(x,top+12,'ShiftWM (ours)' if mode=='factorized' else 'Framewise',color=col)
            for role,dx,s in [('matched20',0,'t=20'),('end',45,f"End {r['native_steps']}")]:
                p.text(x+dx+19.5,top+23,s,color=MUTED,ha='center');p.image(c,mode,role,x+dx,top+29,39)
            p.text(x,top+76,f"L2 {r['final_distance']:.3f} · "+('pass' if r['success'] else 'fail'),color=GREEN if r['success'] else RED)
        p.text(347,top+12,'Joint error (rad)',ha='center',color=MUTED)
        criterion(p,c,310,top+32,78,29)
        if i==0:p.line(6,107,390,107)
    method_legend(p,25,204);p.text(214,204,'At stop: both joints < 0.05 rad',color=MUTED)
    p.save()


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for path,digest in manifest['runtime_inputs_sha256'].items():
        assert sha(ROOT/path)==digest,path
    assert [c['seed'] for c in CASES]==[2031008,2031011]
    for c in CASES:
        for mode in ['factorized','framewise']:
            item=c['methods'][mode];errors=np.array(item['endpoint_criterion_errors'])
            assert errors.shape==(2,) and bool(np.all(errors<.05))==bool(item['record']['success'])
            assert abs(np.linalg.norm(errors)-item['record']['final_distance'])<1e-12
            assert item['criterion']['angle_convention']=='raw_unwrapped'
            assert item['native_times'][0]==10 and 20 in item['native_times']
    OUT.mkdir(parents=True,exist_ok=True);draft_a()
    layout=json.loads((OUT/'positive_cases_compact_layout.json').read_text());assert not layout['issues']
    result={'status':'passed_saved_observation_and_criterion_checks','scope':DATA['scope'],'displayed_case_ids':[c['seed'] for c in CASES],'same_original_cases_and_crops':True,'displayed_endpoint_joint_errors':{str(c['seed']):{m:c['methods'][m]['endpoint_criterion_errors'] for m in ['factorized','framewise']} for c in CASES},'selection':DATA['selection'],'no_new_model_or_simulator_execution':True,'layout':layout,'sources_sha256':manifest['runtime_inputs_sha256'],'manifest_sha256':sha(HERE/'manifest.json'),'caption_sha256':sha(HERE/'caption.tex'),'outputs_sha256':{str((OUT/f'positive_cases_compact.{ext}').relative_to(ROOT)):sha(OUT/f'positive_cases_compact.{ext}') for ext in ['pdf','svg','png']}}
    (OUT/'positive_cases_compact.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'pdf_sha256':sha(OUT/'positive_cases_compact.pdf')}))

if __name__=='__main__':main()
