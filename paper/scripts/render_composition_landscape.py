#!/usr/bin/env python3
"""Portable source-bound composition-protocol figure for the historical study.

Only this script, protocol.json, manifest.json and the six bundled PNGs are read.
The experiment, raw trajectories and paper/design are provenance, not runtime inputs.
"""
from pathlib import Path
import base64, hashlib, json, math, zlib, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'paper/figure_sources/composition_landscape'
ASSETS=PUBLIC/'assets'
OUT=ROOT/'paper/generated/editorial'
W,H=396,396*6/19
INK='#233548'; MUTED='#5D6B7D'; BLUE='#2876AC'; ORANGE='#B46A25'
VIOLET='#7952A0'; LINE='#D9E1E7'; PALE='#F2F5F8'; TRAIN='#536C80'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,
                     'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'composition-v2'})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Canvas:
 def __init__(self,name):
  self.name=name;self.objects=[]
 def text(self,x,y,s,size=8,color=INK,align='left',bold=False):
  self.objects.append(dict(kind='text',x=x,y=y,text=s,size=size,color=color,align=align,bold=bold))
 def rect(self,x,y,w,h,fill='white',stroke=None,lw=.6,r=0):
  self.objects.append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,lw=lw,r=r))
 def line(self,pts,color=LINE,lw=.6,dashed=False):
  self.objects.append(dict(kind='line',pts=pts,color=color,lw=lw,dashed=dashed))
 def photo(self,name,x,y,w):
  p=ASSETS/name;im=Image.open(p)
  self.objects.append(dict(kind='photo',path=name,x=x,y=y,w=w,h=w*im.height/im.width,sha256=sha(p)))
 def mark(self,x,y,kind,scale=1):
  self.objects.append(dict(kind='mark',x=x,y=y,mark=kind,scale=scale))
 def render(self,stem):
  fig=plt.figure(figsize=(W/72,H/72));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
  for o in self.objects:
   k=o['kind']
   if k=='text':ax.text(o['x'],o['y'],o['text'],fontsize=o['size'],color=o['color'],ha=o['align'],va='center',fontweight='bold' if o['bold'] else 'normal')
   elif k=='rect':
    kw=dict(facecolor=o['fill'],edgecolor=o['stroke'] or 'none',linewidth=o['lw'])
    if o['r']:p=FancyBboxPatch((o['x'],o['y']),o['w'],o['h'],boxstyle=f"round,pad=0,rounding_size={o['r']}",**kw)
    else:p=Rectangle((o['x'],o['y']),o['w'],o['h'],**kw)
    ax.add_patch(p)
   elif k=='line':
    x,y=zip(*o['pts']);ax.plot(x,y,color=o['color'],lw=o['lw'],ls=(0,(2,2)) if o['dashed'] else '-',solid_capstyle='butt')
   elif k=='photo':ax.imshow(np.asarray(Image.open(ASSETS/o['path'])),extent=(o['x'],o['x']+o['w'],o['y']+o['h'],o['y']),interpolation='none',aspect='auto')
   else:
    x,y,s=o['x'],o['y'],o['scale'];m=o['mark']
    if m=='train':ax.add_patch(Circle((x,y),2.5*s,color=TRAIN))
    elif m=='dev':ax.add_patch(Polygon([(x,y-3.6*s),(x+3.6*s,y),(x,y+3.6*s),(x-3.6*s,y)],facecolor='#FCF1E6',edgecolor=ORANGE,lw=1))
    elif m=='held':
     v=[(x+4.6*s*math.sin(i*math.pi/5)*(1 if i%2==0 else .43),y-4.6*s*math.cos(i*math.pi/5)*(1 if i%2==0 else .43)) for i in range(10)]
     ax.add_patch(Polygon(v,facecolor=BLUE,edgecolor=BLUE,lw=.5))
    elif m=='extra':ax.add_patch(Polygon([(x,y-4*s),(x+3.7*s,y+3*s),(x-3.7*s,y+3*s)],facecolor='white',edgecolor=VIOLET,lw=1))
  stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
  fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=[]
  for t in ax.texts:
   bb=t.get_window_extent(renderer); inv=ax.transData.inverted();p=inv.transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
   texts.append({'text':t.get_text(),'bbox':[float(p[0,0]),float(p[1,1]),float(p[1,0]),float(p[0,1])],'font_pt':t.get_fontsize()})
  fig.savefig(stem.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
  fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None})
  fig.savefig(stem.with_suffix('.png'),dpi=300)
  fig.savefig(stem.parent/(stem.name+'_paper_width.png'),dpi=120)
  plt.close(fig)
  ImageOps.grayscale(Image.open(stem.parent/(stem.name+'_paper_width.png'))).save(stem.parent/(stem.name+'_grayscale.png'))
  self.native(PUBLIC/'composition_landscape.drawio')
  geometry={'canvas_pt':[W,H],'objects':self.objects,'text_bounds':texts}
  (PUBLIC/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
  return geometry
 def native(self,path):
  f=ET.Element('mxfile',{'host':'app.diagrams.net','type':'device'});d=ET.SubElement(f,'diagram',{'name':self.name,'id':'composition-v2'})
  g=ET.SubElement(d,'mxGraphModel',{'page':'1','pageScale':'1','pageWidth':str(W),'pageHeight':str(H),'math':'0','shadow':'0'})
  root=ET.SubElement(g,'root');ET.SubElement(root,'mxCell',{'id':'0'});ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})
  for i,o in enumerate(self.objects):
   k=o['kind'];style='html=0;shadow=0;';attr={'id':f'obj-{i:03}','parent':'1'};x=o.get('x',0);y=o.get('y',0);w=o.get('w',0);h=o.get('h',0)
   if k=='line':
    attr['edge']='1';style+=f"endArrow=none;strokeColor={o['color']};strokeWidth={o['lw']};"+('dashed=1;dashPattern=2 2;' if o['dashed'] else '')
    attr['style']=style;c=ET.SubElement(root,'mxCell',attr);geo=ET.SubElement(c,'mxGeometry',{'relative':'1','as':'geometry'})
    pts=o['pts'];ET.SubElement(geo,'mxPoint',{'x':str(pts[0][0]),'y':str(pts[0][1]),'as':'sourcePoint'});ET.SubElement(geo,'mxPoint',{'x':str(pts[-1][0]),'y':str(pts[-1][1]),'as':'targetPoint'})
    if len(pts)>2:
     arr=ET.SubElement(geo,'Array',{'as':'points'})
     for xx,yy in pts[1:-1]:ET.SubElement(arr,'mxPoint',{'x':str(xx),'y':str(yy)})
    continue
   attr['vertex']='1'
   if k=='text':
    # Native font geometry uses exactly the same point canvas and alignment.
    w=max(12,max(len(s) for s in o['text'].split('\n'))*o['size']*.61);h=o['size']*1.2*len(o['text'].split('\n'))
    if o['align']=='center':x-=w/2
    elif o['align']=='right':x-=w
    y-=h/2;attr['value']=o['text'];style+=f"shape=text;strokeColor=none;fillColor=none;whiteSpace=wrap;align={o['align']};verticalAlign=middle;spacing=0;fontFamily=Liberation Sans;fontSize={o['size']};fontColor={o['color']};fontStyle={1 if o['bold'] else 0};"
   elif k=='photo':
    b=base64.b64encode((ASSETS/o['path']).read_bytes()).decode();style+='shape=image;imageAspect=0;aspect=fixed;image=data:image/png,'+b+';strokeColor=none;'
   elif k=='rect':style+=f"rounded={1 if o['r'] else 0};arcSize=10;fillColor={o['fill']};strokeColor={o['stroke'] or 'none'};strokeWidth={o['lw']};"
   else:
    s=o['scale'];m=o['mark'];w=h=8*s;x-=w/2;y-=h/2
    if m=='train':w=h=5*s;x=o['x']-w/2;y=o['y']-h/2;style+=f'shape=ellipse;fillColor={TRAIN};strokeColor={TRAIN};'
    elif m=='dev':style+=f'shape=rhombus;fillColor=#FCF1E6;strokeColor={ORANGE};strokeWidth=1;'
    elif m=='held':
     # diagrams.net does not recognize bare shape=star; an editable stencil
     # preserves the exact ten canonical vertices rather than a square fallback.
     pts=[(o['x']+4.6*s*math.sin(j*math.pi/5)*(1 if j%2==0 else .43),o['y']-4.6*s*math.cos(j*math.pi/5)*(1 if j%2==0 else .43)) for j in range(10)]
     xs,ys=zip(*pts);x,y=min(xs),min(ys);w,h=max(xs)-x,max(ys)-y
     shape=ET.Element('shape',{'name':'heldout-star','w':str(w),'h':str(h),'aspect':'variable','strokewidth':'inherit'})
     fg=ET.SubElement(shape,'foreground');pathnode=ET.SubElement(fg,'path')
     for j,(xx,yy) in enumerate(pts):ET.SubElement(pathnode,'move' if j==0 else 'line',{'x':str(xx-x),'y':str(yy-y)})
     ET.SubElement(pathnode,'close');ET.SubElement(fg,'fillstroke')
     comp=zlib.compressobj(wbits=-15);raw=ET.tostring(shape,encoding='utf-8');encoded=base64.b64encode(comp.compress(raw)+comp.flush()).decode()
     style+=f'shape=stencil({encoded});fillColor={BLUE};strokeColor={BLUE};strokeWidth=.5;'
    else:style+=f'shape=triangle;direction=north;fillColor=white;strokeColor={VIOLET};strokeWidth=1;'
   attr['style']=style;c=ET.SubElement(root,'mxCell',attr);ET.SubElement(c,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
  ET.indent(f);ET.ElementTree(f).write(path,encoding='utf-8',xml_declaration=True)

def legend(c,y=120,x=8):
 for dx,kind,label in [(0,'train','Train (7)'),(75,'dev','Development'),(172,'held','Held-out pair'),(270,'extra','Extrapolation (3)')]:
  c.mark(x+dx,y,kind,.85);c.text(x+dx+8,y,label)

def matrix(c,x,y,step=22,full=True,head=True,ystep=None):
 dy=step if ystep is None else ystep
 if head:c.text(x+1.5*step,y-17,'Dynamics',8.5,INK,'center',True)
 n=4 if full else 3
 for j in range(n):c.text(x+(j+.5)*step,y-6,f'p{j}',8,VIOLET if j==3 else INK,'center')
 c.rect(x,y,step*3,dy*3,PALE,r=3)
 for j in (1,2):
  c.line([(x+j*step,y),(x+j*step,y+3*dy)],'white',1)
  c.line([(x,y+j*dy),(x+3*step,y+j*dy)],'white',1)
 for i in range(n):
  c.text(x-5,y+(i+.5)*dy,f'v{i}',8,VIOLET if i==3 else INK,'right')
  for j in range(n):
   kind='dev' if (i,j)==(1,1) else 'held' if (i,j)==(2,2) else 'train' if i<3 and j<3 else 'extra' if (i,j) in [(3,0),(0,3),(3,3)] else None
   if kind:c.mark(x+(j+.5)*step,y+(i+.5)*dy,kind)
 if full:
  c.line([(x+3*step,y-1),(x+3*step,y+4*dy)],VIOLET,.6,True)
  c.line([(x-1,y+3*dy),(x+4*step,y+3*dy)],VIOLET,.6,True)

def draft_a():
 c=Canvas('A — interventions and crossing map')
 c.text(5,9,'Appearance · same PushT state',8.5,INK,bold=True)
 for i,x in enumerate([5,49,93,137]):
  c.photo(f'pusht_v{i}.png',x,21,34);c.text(x+17,63,f'v{i}',8,VIOLET if i==3 else INK,'center')
  c.text(x+17,73,['Canonical','Warm','Cool','Dim'][i],8,MUTED,'center')
 c.photo('reacher.png',196,21,42);c.text(217,73,'Reacher',8,INK,'center')
 c.text(5,87,'Physical setting',8.5,INK,bold=True)
 for j,x in enumerate([114,151,189,229]):c.text(x,87,f'p{j}',8,VIOLET if j==3 else INK,'center')
 for y,label,vals in [(99,'PushT damping',['0','.15','.65','.95']),(111,'Reacher density',['1000','650','1350','1500'])]:
  c.text(5,y,label,8,MUTED)
  for j,x in enumerate([114,151,189,229]):c.text(x,y,vals[j],8,VIOLET if j==3 else INK,'center')
 c.line([(250,5),(250,110)],LINE,.65)
 c.text(274,9,'Cross the two factors',8.5,INK,bold=True)
 matrix(c,284,28,22,True,False,19)
 c.text(328,109,'Test: all 9 in-range',8,MUTED,'center')
 # Shared legend sits below both views; test coverage is a separate layer.
 legend(c,120,8)
 return c


def main():
 manifest=json.loads((PUBLIC/'manifest.json').read_text())
 for rel,digest in manifest['runtime_inputs'].items():
  if sha(ROOT/rel)!=digest:raise ValueError('Source identity changed: '+rel)
 p=json.loads((PUBLIC/'protocol.json').read_text())
 train={(0,0),(1,0),(2,0),(0,1),(2,1),(0,2),(1,2)}
 assert set(map(tuple,p['train']))==train and set(map(tuple,p['ordinary_validation']))==train
 assert p['development']==[[1,1]] and p['heldout_composition']==[[2,2]]
 assert set(map(tuple,p['independent_test']))=={(v,d) for v in range(3) for d in range(3)}
 assert p['extrapolation']==[[3,0],[0,3],[3,3]]
 assert p['physical_settings']=={'pusht_damping':[0,.15,.65,.95],'reacher_arm_and_finger_density':[1000,650,1350,1500]}
 c=draft_a();g=c.render(OUT/'composition_landscape')
 texts=g['text_bounds'];collisions=[]
 for i,a in enumerate(texts):
  assert a['font_pt']>=8
  x0,y0,x1,y1=a['bbox'];assert x0>=0 and y0>=0 and x1<=W and y1<=H
  for b in texts[i+1:]:
   z=b['bbox']
   if min(x1,z[2])>max(x0,z[0]) and min(y1,z[3])>max(y0,z[1]):collisions.append([a['text'],b['text']])
 assert not collisions,collisions
 # Mark identities are an exact second check of the displayed cell population.
 cells=[o for o in c.objects if o['kind']=='mark' and o['x']>=284 and o['y']<105]
 counts={k:sum(o['mark']==k for o in cells) for k in ['train','dev','held','extra']}
 assert counts=={'train':7,'dev':1,'held':1,'extra':3}
 evidence={'status':'passed_source_and_layout_checks','scope':p['scope'],
  'canvas_inches':[W/72,H/72],'aspect_ratio':'19:6','font_minimum_pt':8,
  'source_inputs':manifest['runtime_inputs'],'manifest_sha256':sha(PUBLIC/'manifest.json'),
  'cell_counts':counts,'text_overlap_count':len(collisions),
  'images':'Unchanged full RGB; transformed views are exact prescribed display transforms',
  'native_preview':'Native XML provided; actual application preview is a separate review',
  'outputs':{str(path.relative_to(ROOT)):sha(path) for path in [OUT/'composition_landscape.pdf',OUT/'composition_landscape.svg',OUT/'composition_landscape.png',PUBLIC/'composition_landscape.drawio',PUBLIC/'geometry.json']}}
 (OUT/'composition_landscape.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps({'pdf_sha256':sha(OUT/'composition_landscape.pdf'),'renderer_sha256':sha(__file__)}))

if __name__=='__main__':main()
