#!/usr/bin/env python3
"""Editable exploded-view spatial architecture; candidate, never integrates itself.

One point-coordinate scene emits native draw.io XML and deterministic PDF/SVG.
The PDF is not a draw.io export when that application is unavailable.
"""
from pathlib import Path
import argparse, base64, hashlib, html, json, re, zlib
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch
from matplotlib.path import Path as MPath
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'paper/figure_sources/spatial_architecture_exploded'
DESIGN=ROOT/'paper/design/spatial_architecture_exploded'
OUT=ROOT/'paper/generated/editorial'
BASE=ROOT/'paper/figure_sources/spatial_method/canonical_graph.json'
W,H=396,223
INK='#20354C'; MUTED='#687C90'; LINE='#B8C8D8'
BLUE='#315EDB'; TEAL='#129B99'; CORAL='#E96E64'; GOLD='#E8AE3D'
DARKTEAL='#087A78'; DARKCORAL='#AE4741'
PALETTE=[BLUE,TEAL,CORAL,GOLD]
CAPTION=r'''\textbf{ShiftWM (ours), with expanded decoder operations.} (A) The DROID instantiation receives recorded frames 0/5/10 and supplied action blocks. Frozen DINOv2 features are pooled to $4\!\times\!4$ and standardized using fixed training-only channel statistics. The compact causal module contains spatial encoding, past-only patch-mean context, FiLM, a chronological action GRU and the temporal predictor; only the command prefix through $h$ is used. (B) Keys use fixed $E_0\in\mathbb R^{16\times96}$, the last spatial encoding before FiLM; queries use the horizon state $H_h$. $T_h=\operatorname{softmax}_{\rm row}(Q(H_h)K(E_0)^\top/\sqrt{96}+4I)$ mixes the fixed normalized observation $Z_0\in\mathbb R^{16\times384}$. (C) The per-patch affine sigmoid gate combines direct and mixed sources. An independent branch applies LayerNorm and an affine $96\!\to\!384$ projection to the same $H_h$, then $\Delta_h=\tanh(R(H_h))$ is added after blending. (D) $D_h=\operatorname{diag}(g_h)$, $W_h\geq0$, $W_h\mathbf1=\mathbf1$, and $\hat Z_h=W_hZ_0+\Delta_h$; $m_c,M_c$ are the observed channel minimum and maximum. The envelope concerns standardized coordinates, not prediction error. Panels B/C expand the named modules; dashed arrows denote gate coefficients. Feature objects, mixing weights and activation bars are schematic, not generated RGB. Unchanged DROID photographs are CC BY 4.0. Only DINOv2 weights are frozen; predictor modules train offline.'''
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Scene:
 def __init__(self):self.items=[];self.serial=0
 def add(self,kind,**kw):
  self.serial+=1;kw={'id':kw.pop('id',f'{kind}-{self.serial}'),'kind':kind,**kw};self.items.append(kw);return kw['id']
 def rect(self,x,y,w,h,fill='white',stroke=LINE,r=2,lw=.65,dash=False,id=None):
  return self.add('rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,r=r,lw=lw,dash=dash,**({'id':id} if id else {}))
 def text(self,x,y,text,size=8,color=INK,align='center',math=False,id=None):
  return self.add('text',x=x,y=y,text=text,size=size,color=color,align=align,math=math,**({'id':id} if id else {}))
 def edge(self,points,color=MUTED,head=True,dash=False,lw=.8,semantic=None,id=None):
  return self.add('edge',points=points,color=color,head=head,dash=dash,lw=lw,semantic=semantic,**({'id':id} if id else {}))
 def circle(self,x,y,r=5,fill='white',stroke=TEAL,lw=.8):return self.add('circle',x=x,y=y,r=r,fill=fill,stroke=stroke,lw=lw)
 def poly(self,points,fill,stroke='none',lw=.6):return self.add('polygon',points=points,fill=fill,stroke=stroke,lw=lw)
 def photo(self,x,y,w,h,path):return self.add('image',x=x,y=y,w=w,h=h,path=str(path.relative_to(ROOT)),sha256=sha(path))
 def tensor(self,x,y,w=35,h=30,role='anchor'):
  # Spatial patches have stable identities; shallow extrusion denotes channels.
  fw,fh,depth=w-6,h-6,6
  for cc in range(4):
   rgb=np.asarray(matplotlib.colors.to_rgb(PALETTE[(2*cc)%4]));top=matplotlib.colors.to_hex(.40*rgb+.60)
   self.poly([(x+cc*fw/4,y+depth),(x+(cc+1)*fw/4,y+depth),(x+(cc+1)*fw/4+depth,y),(x+cc*fw/4+depth,y)],top,'white',.3)
  for rr in range(4):
   rgb=np.asarray(matplotlib.colors.to_rgb(PALETTE[(rr+6)%4]));side=matplotlib.colors.to_hex(.72*rgb)
   self.poly([(x+fw,y+depth+rr*fh/4),(x+w,y+rr*fh/4),(x+w,y+(rr+1)*fh/4),(x+fw,y+depth+(rr+1)*fh/4)],side,'white',.3)
  for rr in range(4):
   for cc in range(4):
    col=PALETTE[(rr+2*cc)%4]
    if role=='mixed':
     rgb=np.asarray(matplotlib.colors.to_rgb(PALETTE[(rr+2*cc)%4]));other=np.asarray(matplotlib.colors.to_rgb(PALETTE[(rr+2*cc+1)%4]));col=matplotlib.colors.to_hex(.65*rgb+.35*other)
    self.rect(x+cc*fw/4,y+depth+rr*fh/4,fw/4-.6,fh/4-.6,col,'white',r=.5,lw=.15)
  return (x,y,w,h)
 def chip(self,x,y,w,h,color,opacity=1):
  rgb=np.asarray(matplotlib.colors.to_rgb(color));light=matplotlib.colors.to_hex(.5*rgb+.5)
  self.poly([(x,y+2),(x+2,y),(x+w,y),(x+w-2,y+2)],light,'white',.25)
  self.poly([(x+w-2,y+2),(x+w,y),(x+w,y+h-2),(x+w-2,y+h)],matplotlib.colors.to_hex(.7*rgb),'white',.25)
  for j in range(3):self.rect(x,y+2+j*(h-2)/3,w-2,(h-2)/3-.35,matplotlib.colors.to_hex((.8+.1*j)*rgb),'white',r=.35,lw=.2)
 def box(self,x,y,w,h,label,stroke=LINE,fill='#F5F7FA',dash=False,size=8):
  self.rect(x,y,w,h,fill,stroke,dash=dash);self.text(x+w/2,y+h/2,label,size=size)
 def panel(self,letter,title,x,y,w,h,color):
  self.rect(x,y,w,h,'white','#DCE4ED',r=3,lw=.6)
  self.rect(x+4,y+4,12,12,color,color,r=2);self.text(x+10,y+10,letter,color='white',size=8.5)
  self.text(x+21,y+10,title,align='left',size=8.6)

def scene():
 s=Scene()
 s.panel('A','ShiftWM (ours): predict visual features',0,0,396,63,INK)
 for j,f in enumerate((0,5,10)):
  s.photo(7+j*20,21,18,10.125,ROOT/f'paper/figure_sources/spatial_qualitative/case1_recorded_frame_{f}.png')
 s.text(35,39,'0 / 5 / 10',color=BLUE)
 s.box(75,18,41,27,'Frozen\nDINOv2',BLUE,'#F1F5FF',True)
 s.edge([(66,27),(74,27)],BLUE,semantic='observed-rgb-to-frozen-dino')
 # Conventional predictor internals are compact; two action lanes stay distinct.
 s.rect(128,18,108,32,'#F7F9FC',LINE,r=2)
 s.text(182,26,'Spatial · FiLM · temporal')
 s.box(132,35,43,12,'Past ctx',fill='#FFF8E8',stroke=GOLD)
 s.box(187,35,45,12,'Prefix GRU',fill='#FFF8E8',stroke=GOLD)
 s.edge([(117,28),(127,28)],BLUE,semantic='frozen-dino-to-normalized-patch-grids')
 for j in range(2):s.rect(9+j*8,51,6,6,GOLD,GOLD,r=.5)
 for j in range(10):s.rect(30+j*3.6,51,2.7,6,'#FFE4AA',GOLD,r=.4,lw=.3)
 s.rect(7.5,49.5,59.5,9,'none',GOLD,r=1,lw=.35)
 s.edge([(15,50),(15,48),(122,48),(122,41),(131,41)],GOLD,semantic='past-actions-to-transition-context')
 s.edge([(68,54),(181,54),(210,54),(210,48)],GOLD,semantic='future-prefix-to-causal-gru')
 s.edge([(237,29),(269,29)],TEAL,semantic='lewm-to-horizon-state')
 s.text(251,40,r'$H_h$',math=True,color=DARKTEAL)
 s.box(270,18,45,27,'B\nPatch mix',TEAL,'#E7F7F5')
 s.box(338,18,51,27,'C  Gate\n+ correction',CORAL,'#FFF1ED')
 s.edge([(316,31),(337,31)],TEAL,semantic='product-to-mixed-anchor')
 s.edge([(242,29),(242,13),(364,13),(364,17)],CORAL,semantic='horizon-state-to-projected-innovation')
 s.text(293,57,r'Fixed $Z_0,E_0$',math=True,color=BLUE)
 s.edge([(293,51),(293,46)],BLUE,semantic='fixed-anchor-to-feature-product')
 s.edge([(364,46),(364,51)],TEAL,semantic='addition-to-future-features')
 s.text(364,58,r'$\widehat Z_h$',math=True,color=DARKTEAL)
 detail_start=len(s.items)
 s.panel('B','Fixed-source patch mixing',0,101,196,133,TEAL)
 # Compatibility is learned; the source patches themselves remain observed.
 s.box(8,123,34,15,r'$Q(H_h)$',stroke=TEAL,fill='#EEF9F7');s.items[-1]['math']=True
 s.box(8,146,34,15,r'$K(E_0)$',stroke=BLUE,fill='#F1F5FF');s.items[-1]['math']=True
 s.edge([(43,130),(58,130),(58,139),(66,139)],TEAL,semantic='queries-to-transport-scores')
 s.edge([(43,153),(58,153),(58,147),(66,147)],BLUE,semantic='keys-to-transport-scores')
 s.box(67,129,57,30,'Scaled scores\n+ 4I',fill='#F5F9FC')
 score=.7*np.cos((np.arange(16)[:,None]-np.arange(16)[None,:])*.8)+4*np.eye(16)
 matrix=np.exp(score-score.max(1,keepdims=True));matrix/=matrix.sum(1,keepdims=True)
 for rr in range(16):
  for cc in range(16):
   intensity=np.sqrt(matrix[rr,cc]/matrix.max())*.93
   col=matplotlib.colors.to_hex(1-intensity*(1-np.asarray(matplotlib.colors.to_rgb(PALETTE[cc%4]))))
   s.rect(145+cc*2.25,128+rr*1.875,2.05,1.68,col,'white',r=.15,lw=.05)
 for cc in range(16):s.rect(145+cc*2.25,123,2.05,3,PALETTE[cc%4],PALETTE[cc%4],r=.15,lw=.1)
 s.rect(143,128+5*1.875-.5,38,2.9,'none',TEAL,r=.5,lw=.6)
 s.edge([(125,144),(143,144)],TEAL,semantic='softmax-to-row-stochastic-matrix')
 s.text(157,168,r'$T_h$  row-softmax',math=True,color=DARKTEAL)
 # Expanded tensor: highlighted source identities become a feature mixture.
 s.tensor(9,180,43,36,'anchor');s.text(29,225,r'Fixed $Z_0$',math=True,color=BLUE)
 for j,col in enumerate(PALETTE[:3]):
  s.chip(72,176+j*14,14,10,col)
  s.edge([(53,195),(63,195),(71,181+j*14)],BLUE,lw=.65,semantic='fixed-anchor-to-feature-product')
 s.tensor(148,180,39,34,'mixed');s.text(166,225,r'$T_hZ_0$',math=True,color=DARKTEAL)
 for j,col in enumerate(PALETTE[:3]):s.edge([(87,181+j*14),(110,195)],col,lw=.7,semantic='transport-to-feature-product')
 s.edge([(163,173),(163,176),(113,176),(113,191)],TEAL,lw=.7,semantic='product-to-mixed-anchor')
 s.circle(113,195,3,fill='white',stroke=TEAL,lw=.65)
 s.edge([(117,195),(147,195)],TEAL,semantic='product-to-mixed-anchor')
 s.text(108,217,'Weighted patch sum',color=MUTED)
 s.panel('C','Gate + bounded innovation',204,101,192,133,CORAL)
 # Complementary contributions: the retained source is a distinct branch.
 s.chip(214,126,16,15,BLUE);s.text(222,148,r'$Z_0$',math=True,color=BLUE)
 s.chip(214,161,16,15,TEAL);s.text(222,184,r'$T_hZ_0$',math=True,color=DARKTEAL)
 s.circle(263,134,5,stroke=BLUE);s.text(263,134,r'$\times$',math=True)
 s.circle(263,169,5,stroke=TEAL);s.text(263,169,r'$\times$',math=True)
 s.edge([(231,134),(257,134)],BLUE,semantic='identity-anchor-to-complementary-blend')
 s.edge([(231,169),(257,169)],TEAL,semantic='mixed-anchor-to-complementary-blend')
 s.text(280,124,r'$1-g_i$',math=True,color=BLUE)
 s.text(277,181,r'$g_i$',math=True,color=DARKTEAL)
 s.circle(300,151,5.3);s.text(300,151,'+')
 s.edge([(269,134),(288,134),(288,146),(296,146)],BLUE,semantic='identity-anchor-to-complementary-blend')
 s.edge([(269,169),(288,169),(288,156),(296,156)],TEAL,semantic='mixed-anchor-to-complementary-blend')
 s.box(232,144,52,15,r'$\sigma(G(H_h))$',stroke=CORAL,fill='#FFF1ED');s.items[-1]['math']=True
 s.edge([(263,143),(263,140)],CORAL,dash=True,lw=.6,semantic='gate-to-complementary-blend')
 s.edge([(263,160),(263,163)],CORAL,dash=True,lw=.6,semantic='gate-to-complementary-blend')
 s.text(339,127,'Gated blend',color=DARKCORAL)
 s.chip(328,145,16,15,CORAL);s.edge([(306,151),(327,151)],TEAL,semantic='blend-to-addition')
 s.circle(367,155,5.4);s.text(367,155,'+')
 s.edge([(345,152),(361,152)],TEAL,semantic='blend-to-addition')
 s.chip(380,146,13,16,GOLD);s.edge([(373,155),(379,155)],TEAL,semantic='addition-to-future-features')
 s.text(380,176,r'$\widehat Z_h$',math=True,color=DARKTEAL)
 # Independent horizon-state branch; nothing comes from the blend into R.
 s.text(217,209,r'$H_h$',math=True,color=DARKTEAL)
 s.box(233,198,46,23,'LN + Linear',fill='#FFF1ED',stroke=CORAL)
 s.edge([(226,209),(232,209)],CORAL,semantic='horizon-state-to-projected-innovation')
 for yy,col,val in [(202,BLUE,9),(209,TEAL,-7),(216,CORAL,11)]:
  s.edge([(285,yy),(307,yy)],'#D8E0E9',head=False,lw=.4)
  s.edge([(296,yy),(296+val,yy)],col,head=False,lw=1.7)
 s.edge([(280,209),(284,209)],CORAL)
 s.box(312,198,28,23,'tanh',stroke=CORAL,fill='#FFF1ED')
 s.edge([(308,209),(311,209)],CORAL)
 for yy,col,val in [(202,BLUE,5),(209,TEAL,-4),(216,CORAL,6)]:
  s.edge([(347,yy),(361,yy)],'#D8E0E9',head=False,lw=.4)
  s.edge([(354,yy),(354+val,yy)],col,head=False,lw=1.7)
 s.edge([(341,209),(346,209)],CORAL)
 s.edge([(347,194),(347,220)],GOLD,head=False,lw=.6);s.edge([(361,194),(361,220)],GOLD,head=False,lw=.6)
 s.text(346,228,'−1',color='#8C631F');s.text(362,228,'+1',color='#8C631F')
 s.text(383,211,r'$\Delta_h$',math=True,color=DARKCORAL)
 s.edge([(362,209),(371,209),(371,180),(367,180),(367,162)],CORAL,semantic='bounded-innovation-to-addition')
 # Compact landscape arrangement preserves8pt labels and expanded B/C geometry.
 for item in s.items[detail_start:]:
  if 'y' in item:item['y']-=29
  if 'points' in item:item['points']=[(x,y-29) for x,y in item['points']]
 s.edge([(0,209),(396,209)],LINE,head=False,lw=.6,id='math-rail-rule')
 s.rect(1,212,11,11,INK,INK,r=1.5);s.text(6.5,217.5,'D',color='white',size=8.5)
 s.text(22,217,r'$W_h=I-D_h+D_hT_h$',math=True,align='left')
 s.text(200,217,r'$\widehat Z_h=W_hZ_0+\Delta_h$',math=True,color=DARKTEAL)
 s.text(326,217,r'$m_c-1\leq\widehat Z_{h,ic}\leq M_c+1$',math=True)
 return s

def render_scene(s,base):
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
 fig=plt.figure(figsize=(W/72,H/72),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off');texts=[]
 for p in s.items:
  k=p['kind']
  if k=='rect':ax.add_patch(FancyBboxPatch((p['x'],p['y']),p['w'],p['h'],boxstyle=f"round,pad=0,rounding_size={p['r']}",facecolor=p['fill'],edgecolor=p['stroke'],lw=p['lw'],ls='--' if p['dash'] else '-'))
  elif k=='circle':ax.add_patch(Circle((p['x'],p['y']),p['r'],facecolor=p['fill'],edgecolor=p['stroke'],lw=p['lw']))
  elif k=='polygon':ax.add_patch(Polygon(p['points'],facecolor=p['fill'],edgecolor=p['stroke'],lw=p['lw'],zorder=.2 if p['fill'] in ('#F1FBF9','#FFF6F2') else 1))
  elif k=='text':texts.append(ax.text(p['x'],p['y'],p['text'],ha=p['align'],va='center',fontsize=p['size'],color=p['color']))
  elif k=='edge':
   ax.add_patch(FancyArrowPatch(path=MPath(p['points'],[MPath.MOVETO]+[MPath.LINETO]*(len(p['points'])-1)),arrowstyle='-|>' if p['head'] else '-',mutation_scale=4.8,lw=p['lw'],color=p['color'],ls=(0,(2,1.8)) if p['dash'] else '-'))
  elif k=='image':ax.imshow(np.asarray(Image.open(ROOT/p['path'])),extent=(p['x'],p['x']+p['w'],p['y']+p['h'],p['y']),interpolation='none',aspect='auto',zorder=3)
 fig.canvas.draw();ren=fig.canvas.get_renderer();bb=[t.get_window_extent(ren) for t in texts]
 overlap=[[texts[i].get_text(),texts[j].get_text()] for i,b in enumerate(bb) for j,c in enumerate(bb) if i<j and b.overlaps(c)]
 clipped=[t.get_text() for t,b in zip(texts,bb) if b.x0<0 or b.y0<0 or b.x1>fig.bbox.x1 or b.y1>fig.bbox.y1]
 for ext in ('pdf','svg','png'):fig.savefig(base.with_suffix('.'+ext),dpi=300)
 plt.close(fig);return {'text_overlap_pairs':overlap,'clipped_text':clipped,'minimum_font_pt':min(p['size'] for p in s.items if p['kind']=='text')}

def drawio(s,path):
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix'})
 measure=plt.figure(figsize=(W/72,H/72));ma=measure.add_axes([0,0,1,1]);ma.set(xlim=(0,W),ylim=(H,0))
 handles={p['id']:ma.text(p['x'],p['y'],p['text'],ha=p['align'],va='center',fontsize=p['size']) for p in s.items if p['kind']=='text'}
 measure.canvas.draw();mrenderer=measure.canvas.get_renderer()
 measured={k:(t.get_window_extent(mrenderer).width*72/measure.dpi,t.get_window_extent(mrenderer).height*72/measure.dpi) for k,t in handles.items()}
 plt.close(measure)
 root=ET.Element('mxGraphModel',{'dx':str(W),'dy':str(H),'grid':'1','gridSize':'1','page':'1','pageWidth':str(W),'pageHeight':str(H),'math':'1','adaptiveColors':'auto'})
 layer=ET.SubElement(root,'root');ET.SubElement(layer,'mxCell',{'id':'0'});ET.SubElement(layer,'mxCell',{'id':'1','parent':'0'})
 ordered=sorted(s.items,key=lambda p:0 if p.get('fill') in ('#F1FBF9','#FFF6F2') else 1)
 for p in ordered:
  k=p['kind'];at={'id':p['id'],'parent':'1'}
  if k=='edge':
   at.update(edge='1',style=f"edgeStyle=none;rounded=0;html=1;strokeColor={p['color']};strokeWidth={p['lw']};endArrow={'block' if p['head'] else 'none'};endSize=3;startArrow=none;dashed={int(p['dash'])};")
   cell=ET.SubElement(layer,'mxCell',at);geo=ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'});pts=p['points'];ET.SubElement(geo,'mxPoint',{'x':str(pts[0][0]),'y':str(pts[0][1]),'as':'sourcePoint'});ET.SubElement(geo,'mxPoint',{'x':str(pts[-1][0]),'y':str(pts[-1][1]),'as':'targetPoint'})
   arr=ET.SubElement(geo,'Array',{'as':'points'})
   for x,y in pts[1:-1]:ET.SubElement(arr,'mxPoint',{'x':str(x),'y':str(y)})
   continue
  at['vertex']='1';x,y=p.get('x',0),p.get('y',0);w,h=p.get('w',1),p.get('h',1)
  if k=='text':
   # Formula source remains editable MathJax text, never a raster equation.
   label=html.escape(p['text']).replace('\n','<br>')
   label=re.sub(r'\$([^$]+)\$',lambda m:'\\('+m[1]+'\\)',label)
   at.update(value=label,style=f"text;html=1;math=1;align={p['align']};verticalAlign=middle;whiteSpace=nowrap;overflow=visible;spacing=0;fontFamily=Liberation Sans;fontSize={p['size']};fontColor={p['color']};strokeColor=none;fillColor=none;")
   mw,mh=measured[p['id']];w,h=mw+1.4,mh+.8;x=x-(w/2 if p['align']=='center' else 0);y-=h/2
  elif k=='rect':at['style']=f"rounded=1;arcSize=8;fillColor={p['fill']};strokeColor={p['stroke']};strokeWidth={p['lw']};dashed={int(p['dash'])};"
  elif k=='circle':x-=p['r'];y-=p['r'];w=h=p['r']*2;at['style']=f"ellipse;fillColor={p['fill']};strokeColor={p['stroke']};strokeWidth={p['lw']};"
  elif k=='image':
   data=base64.b64encode((ROOT/p['path']).read_bytes()).decode();at['style']=f"shape=image;imageAspect=0;aspect=fixed;image=data:image/png,{data};"
  elif k=='polygon':
   pts=p['points'];x=min(q[0] for q in pts);y=min(q[1] for q in pts);w=max(q[0] for q in pts)-x;h=max(q[1] for q in pts)-y
   shape=ET.Element('shape',{'name':'native-polygon','w':str(w),'h':str(h),'aspect':'variable','strokewidth':'inherit'})
   foreground=ET.SubElement(shape,'foreground');path_node=ET.SubElement(foreground,'path')
   for j,(px,py) in enumerate(pts):ET.SubElement(path_node,'move' if j==0 else 'line',{'x':str(px-x),'y':str(py-y)})
   ET.SubElement(path_node,'close');ET.SubElement(foreground,'fillstroke')
   data=base64.b64encode(zlib.compress(ET.tostring(shape))[2:-4]).decode()
   at['style']=f"shape=stencil({data});fillColor={p['fill']};strokeColor={p['stroke']};strokeWidth={p['lw']};"
  cell=ET.SubElement(layer,'mxCell',at);ET.SubElement(cell,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
 ET.indent(root);path.write_text(ET.tostring(root,encoding='unicode')+'\n')
 ids=[c.attrib['id'] for c in root.findall('.//mxCell')];assert len(ids)==len(set(ids))
 assert all(c.find('mxGeometry') is not None for c in root.findall('.//mxCell[@edge="1"]'))
 return {'unique_cells':len(ids),'edges':len(root.findall('.//mxCell[@edge="1"]')),'xml_well_formed':True,'native_application_render':'pending separate official-browser export; deterministic PDF/SVG render uses same geometry'}

def semantic_contract(s,graph):
 mapping={e['semantic_id']:['explicit native edge '+p['id'] for p in s.items if p.get('semantic')==e['semantic_id']] for e in graph['semantic_edges']}
 collapsed={
 'support-grids-to-spatial-encoder':'A causal module: input normalized grids to trainable spatial encoder',
 'past-actions-to-transition-context':'A causal module: only 2 observed-transition action blocks enter support context',
 'mean-encoded-support-to-transition-context':'A causal module: patch means of three encodings + two past actions form transition context',
 'encoded-support-to-film':'A causal module: all3 spatial encodings pass through FiLM',
 'context-to-film':'A causal module: one past-only32D context supplies FiLM',
 'past-actions-to-causal-gru':'A causal module: normalized two past blocks precede query blocks in separate unidirectional GRU',
 'conditioned-support-to-lewm':'A causal module: same3 FiLM support encodings supplied at each horizon',
 'causal-action-states-to-lewm':'A causal module: first2 past-prefix states plus query-prefix endpoint at h enter temporal predictor',
 'horizon-state-to-query':'B functional query node Q(H_h), same H_h as A/C aliases',
 'pre-film-observation-to-key':'B functional key node K(E0); fixed last spatial encoding bypasses FiLM',
 'horizon-state-to-gate':'C affine per-patch sigmoid equation G(H_h), same state alias',
 }
 for k,v in collapsed.items():mapping.setdefault(k,[]).append(v)
 expected={e['semantic_id'] for e in graph['semantic_edges']}
 assert set(mapping)==expected and all(mapping.values()),[(k,v) for k,v in mapping.items() if not v]
 return {'model_sha256':graph['sources_sha256']['src/shiftwm/real_video_spatial/model.py'],'semantic_dependencies':mapping,'count':len(mapping),'reference_authority':'Generated reference supplies visual grammar only; no generated labels, images or edges are scientific evidence.','normalized_source':'Z0 fixed16x384 last observed grid; E0 fixed16x96 last encoding beforeFiLM. Fixed evidence does not mean frozen trainable key/spatial weights.','mixing':'T=softmax_row(Q(H)K(E0)^T/sqrt96+4I), rows target/cols observed source; no physical motion.','gate':'G affine96to1 per patch; complementary coefficients; no gate as an additive feature.','innovation':'H -> LayerNorm+affine96to384 -> tanh -> additive delta AFTER blend; independent of blend.','output':'Zhat=(I-diag(g)+diag(g)T)Z0+tanh(R(H)); diagram in standardized coordinates. Inverse normalization outside diagram.','zoom_edges':[],'gate_control_edges':'dashed arrows are coefficient control; panelsB/C identify expansion by matching module names','collapsed_module':'A preserves the exact separate context and action-prefix paths in this ledger/caption; it does not imply future commands enter context.','depicted_matrix':'16x16 schematic softmax matrix with one target row highlighted; generated deterministic example, not learned coefficients. Three source pieces exemplify the patch sum, not an assertion of sparsity.','historical_CEM_or_paired_context_losses':False,'future_rgb_as_input':False}

def drafts():
 DESIGN.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(1,3,figsize=(11,3.4),facecolor='white')
 plans=[('Overview + two zoom lenses',[(.04,.73,.92,.22,'A  causal backbone'),(.04,.04,.56,.63,'B  expanded mixer'),(.63,.39,.33,.28,'C  innovation'),(.63,.04,.33,.29,'D  envelope')]),('Radial decoder workbench',[(.30,.67,.40,.28,'A  inputs / state'),(.04,.26,.41,.32,'B  mixer'),(.55,.26,.41,.32,'C  correction'),(.18,.03,.64,.16,'D  output / bound')]),('Aligned processing rails',[(.04,.68,.92,.26,'A  compact forecast'),(.04,.35,.92,.26,'B  source → weights → blend'),(.04,.03,.47,.25,'C  projection → bounded'),(.56,.03,.40,.25,'D  mathematical contract')])]
 for ax,(title,regions) in zip(axs,plans):
  ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.02,1,title,fontsize=9,color=INK)
  for j,(x,y,w,h,t) in enumerate(regions):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.02',facecolor=['#EFF4FB','#E9F8F5','#FFF1ED','#F4F6F9'][j],edgecolor=LINE,lw=.8));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
 fig.tight_layout();fig.savefig(DESIGN/'composition_drafts.pdf');fig.savefig(DESIGN/'composition_drafts.png',dpi=150);plt.close(fig)

def main():
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--drafts-only',action='store_true');args=parser.parse_args()
 drafts()
 if args.drafts_only:return
 for p in (SOURCE,DESIGN,OUT):p.mkdir(parents=True,exist_ok=True)
 graph=json.loads(BASE.read_text())
 for p,h in graph['sources_sha256'].items():
  if sha(ROOT/p)!=h:raise ValueError('Frozen scientific/image source changed: '+p)
 s=scene();(SOURCE/'geometry.json').write_text(json.dumps({'width_pt':W,'height_pt':H,'objects':s.items},indent=2)+'\n')
 contract=semantic_contract(s,graph);(SOURCE/'semantic_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 xml=drawio(s,SOURCE/'spatial-architecture-exploded.drawio')
 visual=render_scene(s,OUT/'spatial_architecture_exploded')
 (OUT/'spatial_architecture_exploded_caption.tex').write_text(CAPTION+'\n')
 (OUT/'spatial_architecture_exploded_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/spatial_architecture_exploded.pdf}\n\\caption[Expanded observation-anchored decoder.]{'+CAPTION+'}\n\\label{fig:editorial-spatial-method}\n\\end{figure}\n')
 report={'status':'candidate_pending_independent_review','renderer_sha256':sha(__file__),'scientific_sources_sha256':graph['sources_sha256'],'geometry_sha256':sha(SOURCE/'geometry.json'),'drawio_sha256':sha(SOURCE/'spatial-architecture-exploded.drawio'),'semantic_contract_sha256':sha(SOURCE/'semantic_contract.json'),'semantic_dependencies_preserved':contract['count'],'composition_reference':{'path':'paper/figure_sources/visual_story_references_v1/architecture_composition_reference.png','sha256':'d435eb3f6b803c751abecda051d4d1276e83f5dc64d283ac03c902a5e2e5d9dc','pixels_inspected':True,'authority':'composition only; generated road imagery and wrong edges rejected'},'dimensions_inches':[W/72,H/72],**xml,**visual,'outputs_sha256':{x:sha(OUT/f'spatial_architecture_exploded.{x}') for x in ('pdf','svg','png')}}
 (OUT/'spatial_architecture_exploded_evidence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
