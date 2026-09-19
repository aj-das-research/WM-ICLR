#!/usr/bin/env python3
"""Observed-memory architecture with one target-patch worked example; candidate only.

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
SOURCE=ROOT/'paper/figure_sources/architecture_visual_design'
DESIGN=ROOT/'paper/design/architecture_visual_design'
OUT=ROOT/'paper/generated/editorial'
BASE=ROOT/'paper/figure_sources/spatial_method/canonical_graph.json'
W,H=396,264
INK='#20354C'; MUTED='#687C90'; LINE='#B8C8D8'
BLUE='#315EDB'; TEAL='#129B99'; CORAL='#E96E64'; GOLD='#E8AE3D'
DARKTEAL='#087A78'; DARKCORAL='#AE4741'
PALETTE=[BLUE,TEAL,CORAL,GOLD]
CAPTION=''
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

def render_scene(s,base):
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'shiftwm-architecture-visual-design-v1','image.composite_image':False})
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
 for ext in ('pdf','svg','png'):
  meta={'Creator':'ShiftWM portable vector renderer','CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Creator':'ShiftWM portable vector renderer','Date':None} if ext=='svg' else {'Software':'ShiftWM portable vector renderer'})
  fig.savefig(base.with_suffix('.'+ext),dpi=300,metadata=meta)
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


# Fixed toy feature identities explain one target patch; never measured values.
IDS=(1,6,11)
IDCOL={1:BLUE,6:TEAL,11:CORAL}
CAPTION=r'''\textbf{ShiftWM (ours): retain observed features, compose a new forecast.} (A) Recorded DROID frames 0/5/10 become three frozen-DINOv2 grids with fixed training-only channel normalization. The learned spatial encoder processes all three grids into $E_{-2:0}$; its last pre-FiLM encoding $E_0$ and source $Z_0$ stay fixed throughout forecasting. Only DINOv2 weights are frozen. (B) Context from patch means, their chronological differences and past commands conditions the three encodings; a separate chronological action GRU and temporal predictor produce $H_h$ from the supplied prefix. (C) $Q(H_h)$ and $K(E_0)$ determine a row-softmax matrix with the $+4I$ bias outside $\sqrt{96}$ scaling. The highlighted row $i=6$ mixes all sixteen observed patch vectors; one-based contributors 1/6/11 are illustrated and patch 6 supplies the direct branch in (D). (D) The same $H_h$ sets the affine sigmoid gate and, independently, a LayerNorm--affine $96\!\to\!384$ projection followed by $\tanh$. The complementary blend receives this unit-bounded update in standardized feature coordinates after summation; see Eqs.~\eqref{eq:editorial-mixing}--\eqref{eq:editorial-forecast}. The lower lane scores the assembled feature grid against the withheld frame through the same frozen encoder and normalization, with no target-to-predictor path. Feature pieces, weights and channel bars are schematic, not RGB forecasts. DROID photographs are unchanged (CC BY 4.0).'''

def patch_grid(s,x,y,w=42,h=35,colors=None,identities=False):
    colors=colors or [IDCOL.get(j+1,'#CED8E4') for j in range(16)]
    fw,fh,d=w-5,h-5,5
    s.poly([(x,y+d),(x+d,y),(x+w,y),(x+fw,y+d)],'#E5EBF2','white',.25)
    s.poly([(x+fw,y+d),(x+w,y),(x+w,y+fh),(x+fw,y+h)],'#91A9BD','white',.25)
    for rr in range(4):
        for cc in range(4):
            j=rr*4+cc;c=colors[j]
            s.rect(x+cc*fw/4,y+d+rr*fh/4,fw/4-.4,fh/4-.4,c,'white',r=.3,lw=.2)
            if identities and j+1 in IDS:s.text(x+(cc+.5)*fw/4,y+d+(rr+.5)*fh/4,str(j+1),size=8,color='white')

def numbered_piece(s,x,y,j,w=14,h=12):
    s.chip(x,y,w,h,IDCOL[j]);s.text(x+(w-2)/2,y+(h+2)/2,str(j),size=8,color='white')

def scene():
    s=Scene()
    # A: immutable observation evidence; fixed does not mean all weights frozen.
    s.panel('A','Fixed memory',0,0,87,224,BLUE)
    s.photo(8,25,22,12.375,ROOT/'paper/figure_sources/spatial_qualitative/case1_recorded_frame_0.png')
    s.photo(8,40,22,12.375,ROOT/'paper/figure_sources/spatial_qualitative/case1_recorded_frame_5.png')
    s.photo(35,25,44,24.75,ROOT/'paper/figure_sources/spatial_qualitative/case1_recorded_frame_10.png')
    s.text(44,60,'Frames 0 / 5 / 10',color=MUTED)
    s.edge([(44,66),(44,73)],BLUE,semantic='observed-rgb-to-frozen-dino')
    s.box(16,74,55,25,'Frozen\nDINOv2',BLUE,'#F3F6FC',True)
    s.text(44,108,'Pool + standardize',color=MUTED)
    s.edge([(44,114),(44,119)],BLUE,semantic='frozen-dino-to-normalized-patch-grids')
    s.text(44,124,r'$Z_{-2:0}$',math=True,color=BLUE)
    # Two quiet backplanes identify three observed grids, front is last Z0.
    s.rect(24,133,39,30,'#F1F4F8',LINE,r=1,lw=.4)
    s.rect(20,136,39,30,'#E5EBF2',LINE,r=1,lw=.4)
    patch_grid(s,16,139,45,36,identities=True)
    s.text(64,167,r'$Z_0$',math=True,color=BLUE)
    s.edge([(44,176),(44,181)],BLUE,semantic='support-grids-to-spatial-encoder')
    s.box(12,182,63,16,'Encode all 3',LINE,'#F6F8FA')
    s.edge([(44,199),(44,202)],BLUE,semantic='pre-film-observation-to-key')
    s.chip(36,203,16,11,'#7890BB');s.text(67,209,r'$E_0$',math=True,color=BLUE)
    s.text(44,220,r'$E_{-2:0}$; keep $E_0$',math=True,color=MUTED)

    # B: separate observation-context and command-prefix paths, compact machinery.
    s.panel('B','Causal state for horizon h',95,0,301,78,INK)
    s.text(119,26,r'$E_{-2:0}$',math=True,color=BLUE)
    s.box(153,19,37,17,'FiLM',fill='#F6F8FA')
    s.edge([(137,26),(152,26)],BLUE,semantic='encoded-support-to-film')
    s.box(101,41,73,16,'Past-only context',fill='#F6F8FA')
    s.edge([(119,32),(119,40)],BLUE,semantic='mean-encoded-support-to-transition-context')
    s.edge([(175,49),(180,49),(180,37)],BLUE,semantic='context-to-film')
    s.box(185,44,43,17,'Prefix GRU',fill='#F6F8FA')
    s.box(241,23,63,33,'Temporal\npredictor',fill='#F6F8FA')
    s.edge([(191,27),(240,27)],BLUE,semantic='conditioned-support-to-lewm')
    s.edge([(229,51),(240,51)],GOLD,semantic='causal-action-states-to-lewm')
    # Two dark past blocks and ten supplied query blocks, read chronologically.
    for j in range(2):s.rect(103+j*7,65,5,6,GOLD,GOLD,r=.5,lw=.2)
    for j in range(10):s.rect(123+j*4.6,65,3.6,6,'#FFE6AF',GOLD,r=.4,lw=.2)
    s.edge([(110,64),(110,58)],GOLD,semantic='past-actions-to-transition-context')
    s.edge([(117,64),(117,62),(179,62),(179,54),(184,54)],GOLD,semantic='past-actions-to-causal-gru')
    s.edge([(169,68),(207,68),(207,62)],GOLD,semantic='future-prefix-to-causal-gru')
    s.text(276,68,'2 past + 10 supplied blocks',color=MUTED)
    s.edge([(305,40),(327,40)],TEAL,semantic='lewm-to-horizon-state')
    s.chip(329,29,20,22,TEAL);s.text(368,39,r'$H_h$',math=True,color=DARKTEAL)
    s.text(353,62,'Changes with h',color=MUTED)

    # C: one identifiable target row and numbered fixed source pieces.
    s.panel('C','Mix fixed patches',95,85,168,139,TEAL)
    s.box(102,108,40,15,r'$Q(H_h)$',TEAL,'#F0F9F7');s.items[-1]['math']=True
    s.box(102,131,40,15,r'$K(E_0)$',BLUE,'#F2F5FC');s.items[-1]['math']=True
    s.edge([(143,115),(148,115)],TEAL,semantic='queries-to-transport-scores')
    s.edge([(143,139),(148,139)],BLUE,semantic='keys-to-transport-scores')
    s.box(149,108,43,38,r'$QK^\top/\sqrt{96}$'+'\n'+r'$+\,4I$',LINE,'#F6F8FA');s.items[-1]['math']=True
    scores=.65*np.cos((np.arange(16)[:,None]-np.arange(16)[None,:])*.8)+4*np.eye(16)
    t=np.exp(scores-scores.max(1,keepdims=True));t/=t.sum(1,keepdims=True)
    for r in range(16):
        for c in range(16):
            col=np.asarray(matplotlib.colors.to_rgb(IDCOL.get(c+1,'#91A9BD')))
            color=matplotlib.colors.to_hex(1-np.sqrt(t[r,c]/t.max())*.94*(1-col))
            s.rect(207+c*2.7,109+r*2.2,2.45,1.95,color,'white',r=.1,lw=.05)
    for j in IDS:s.text(207+(j-.5)*2.7,104,str(j),size=8,color=IDCOL[j])
    s.rect(205.7,109+5*2.2-.7,45.5,3.45,'none',TEAL,r=.3,lw=.65)
    s.text(256,121,'6',color=DARKTEAL)
    s.edge([(193,126),(205,126)],TEAL,semantic='softmax-to-row-stochastic-matrix')
    s.text(226,153,'Row-softmax',color=DARKTEAL)
    s.text(233,163,r'$T_h\;(16\!\times\!16)$',math=True,color=DARKTEAL)
    # Source paths show representatives; omitted pieces remain in exact sum.
    s.edge([(62,153),(89,153),(89,185),(103,185)],BLUE,semantic='fixed-anchor-to-feature-product')
    for j,yy in zip(IDS,(166,183,200)):
        numbered_piece(s,105,yy-6,j)
        s.edge([(99,185),(101,185),(101,yy),(104,yy)],BLUE,lw=.55,semantic='fixed-anchor-to-feature-product')
        s.edge([(120,yy),(165,185)],IDCOL[j],lw=.7,semantic='fixed-anchor-to-feature-product')
    s.text(127,212,'…',color=MUTED)
    s.circle(170,185,5,stroke=TEAL);s.text(170,185,r'$\Sigma$',math=True)
    s.edge([(212,170),(212,174),(170,174),(170,179)],TEAL,semantic='transport-to-feature-product')
    mixcol=matplotlib.colors.to_hex(sum(t[5,j]*np.asarray(matplotlib.colors.to_rgb(IDCOL.get(j+1,'#CED8E4'))) for j in range(16)))
    s.chip(228,177,19,18,mixcol)
    s.edge([(176,185),(227,185)],TEAL,semantic='product-to-mixed-anchor')
    s.text(234,204,r'$m_i$',math=True,color=DARKTEAL)
    s.text(181,219,r'$m_i=\sum_jT_{h,ij}z_{0,j}$',math=True,color=DARKTEAL)

    # D: complementary multipliers followed by an independent bounded update.
    s.panel('D','Blend + correct',272,85,124,139,CORAL)
    s.text(334,111,r'$g_i=\sigma(G(H_h)_i)$',math=True,color=DARKCORAL)
    numbered_piece(s,279,123,6,15,12);s.text(287,145,r'$z_{0,i}$',math=True,color=BLUE)
    s.chip(279,157,15,12,mixcol);s.text(287,180,r'$m_i$',math=True,color=DARKTEAL)
    s.circle(313,130,4.7,stroke=BLUE);s.text(313,130,r'$\times$',math=True)
    s.circle(313,163,4.7,stroke=TEAL);s.text(313,163,r'$\times$',math=True)
    s.edge([(295,130),(307,130)],BLUE,semantic='identity-anchor-to-complementary-blend')
    s.edge([(295,163),(307,163)],TEAL,semantic='mixed-anchor-to-complementary-blend')
    s.text(334,124,r'$1-g_i$',math=True,color=BLUE)
    s.text(334,177,r'$g_i$',math=True,color=DARKTEAL)
    s.text(313,147,r'$g_i$',math=True,color=DARKCORAL)
    s.edge([(313,141),(313,136)],CORAL,dash=True,lw=.55,semantic='gate-to-complementary-blend')
    s.edge([(313,152),(313,157)],CORAL,dash=True,lw=.55,semantic='gate-to-complementary-blend')
    s.circle(344,147,4.8,stroke=TEAL);s.text(344,147,'+')
    s.edge([(319,130),(332,130),(332,142),(339,142)],BLUE,semantic='identity-anchor-to-complementary-blend')
    s.edge([(319,163),(332,163),(332,152),(339,152)],TEAL,semantic='mixed-anchor-to-complementary-blend')
    s.circle(377,147,4.8,stroke=TEAL);s.text(377,147,'+')
    s.edge([(350,147),(371,147)],TEAL,semantic='blend-to-addition')
    s.chip(369,165,19,15,TEAL);s.text(378,187,r'$\hat z_{h,i}$',math=True,color=DARKTEAL)
    s.edge([(377,153),(377,164)],TEAL,semantic='addition-to-future-features')
    s.text(280,201,r'$H_h$',math=True,color=DARKTEAL)
    s.box(298,190,28,22,'LN\nLinear',CORAL,'#FFF4EF')
    s.edge([(289,201),(297,201)],CORAL,semantic='horizon-state-to-projected-innovation')
    s.box(334,191,27,20,'tanh',CORAL,'#FFF4EF')
    s.edge([(327,201),(333,201)],CORAL,semantic='horizon-state-to-projected-innovation')
    s.edge([(362,201),(367,201)],CORAL)
    for yy,col,v in [(195,BLUE,3),(201,TEAL,-4),(207,CORAL,4)]:
        s.edge([(369,yy),(385,yy)],'#DCE4EC',head=False,lw=.4)
        s.edge([(377,yy),(377+v,yy)],col,head=False,lw=1.5)
    s.edge([(369,190),(369,211)],GOLD,head=False,lw=.6);s.edge([(385,190),(385,211)],GOLD,head=False,lw=.6)
    s.text(368,218,'−1',color='#986B20');s.text(385,218,'+1',color='#986B20')
    s.edge([(386,201),(392,201),(392,147),(383,147)],CORAL,semantic='bounded-innovation-to-addition')

    # Clearly separate target-only information; no numerical result or heatmap.
    s.edge([(0,230),(396,230)],LINE,head=False,dash=True,lw=.65)
    s.text(28,241,'Target only',color=MUTED)
    s.text(28,254,r'$h=10$',math=True,color=MUTED)
    s.photo(61,234.5,34,19.125,ROOT/'paper/figure_sources/spatial_qualitative/case1_recorded_frame_60.png')
    s.text(78,258.5,'Frame 60',color=MUTED)
    s.edge([(96,245),(106,245)],MUTED)
    s.box(107,235,63,24,'Frozen DINOv2\nsame train stats',LINE,'#F6F8FA')
    s.edge([(171,245),(185,245)],MUTED)
    patch_grid(s,187,236,23,21,colors=['#C8D3E0']*16)
    s.edge([(212,245),(230,245)],MUTED)
    s.box(231,238,48,19,'Feature MSE',LINE,'#F6F8FA')
    patch_grid(s,306,236,23,21,colors=[mixcol]*16)
    s.edge([(304,245),(280,245)],TEAL)
    s.text(364,247,'Repeat for\n16 patches',color=DARKTEAL)
    return s

def drafts():
    DESIGN.mkdir(parents=True,exist_ok=True)
    plans=[('1  Observed memory / patch workbench',[(.02,.15,.20,.81,'fixed\nobserved\nmemory'),(.25,.73,.73,.23,'causal state + actions'),(.25,.15,.45,.54,'query/key → row\nsource pieces → mix'),(.73,.15,.25,.54,'gate +\nbounded update'),(.02,.02,.96,.10,'target-only scoring')]),('2  Two aligned processing lanes',[(.02,.71,.96,.25,'observations → encoding → causal state'),(.02,.33,.57,.32,'fixed patch gallery + row weights'),(.62,.33,.36,.32,'complementary blend'),(.02,.13,.96,.15,'projected channels → tanh → output'),(.02,.02,.96,.07,'scoring boundary')]),('3  One target patch from left to right',[(.02,.22,.22,.72,'observations\n+ actions'),(.27,.47,.28,.47,'compatibility\nand weights'),(.58,.47,.40,.47,'source pieces → new patch'),(.27,.22,.71,.18,'independent innovation'),(.02,.02,.96,.15,'output equation + target evaluation')])]
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),facecolor='white')
    for ax,(title,boxes) in zip(axs,plans):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.01,1.02,title,fontsize=9,color=INK)
        for j,(x,y,w,h,t) in enumerate(boxes):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.01',fc=['#EFF4FB','#F5F7FA','#EAF8F5','#FFF0EC','#F5F7FA'][j%5],ec=LINE,lw=.8));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
    fig.tight_layout();fig.savefig(DESIGN/'composition_drafts.png',dpi=160);fig.savefig(DESIGN/'composition_drafts.pdf');plt.close(fig)

def main():
    for p in (SOURCE,DESIGN,OUT):p.mkdir(parents=True,exist_ok=True)
    drafts()
    graph=json.loads(BASE.read_text())
    for p,h in graph['sources_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Scientific source changed: '+p)
    target=ROOT/'paper/figure_sources/spatial_qualitative/case1_recorded_frame_60.png'
    if sha(target)!='4dd6b392e56823becfc98728617314a5dbe12e0cb1fffe72072a5455bbd2e871':
        raise ValueError('Target-only recorded frame changed')
    s=scene();(SOURCE/'geometry.json').write_text(json.dumps({'width_pt':W,'height_pt':H,'objects':s.items},indent=2)+'\n')
    expected={e['semantic_id'] for e in graph['semantic_edges']}
    contracts={
      'support-grids-to-spatial-encoder':'A encodes all three standardized observed grids using the same learned spatial encoder; the front grid is the last Z0.',
      'mean-encoded-support-to-transition-context':'B collapses patch means and chronological observed differences before the TransitionContext GRU; only past commands enter.',
      'horizon-state-to-query':'Q(H_h) in C uses the same state H_h emitted by B, separately for each horizon.',
      'pre-film-observation-to-key':'The E0 alias is the last of A encodings E_-2:0 before FiLM; C applies learned K(E0), fixed across forecast horizons.',
      'horizon-state-to-gate':'The G(H_h) functional alias in D uses the same B state; affine scalar per destination patch then sigmoid, broadcast across384channels.',
      'gate-to-complementary-blend':'The repeated g_i labels denote the same scalar from sigma(G(H_h)_i); dashed coefficient edges apply 1-g_i to z0,i and g_i to m_i.',
      'identity-anchor-to-complementary-blend':'D direct observed piece6 is the unchanged vector at Z0 target position i=6 in A.',
      'mixed-anchor-to-complementary-blend':'D m_i is the identical weighted feature piece emitted by C.',
      'addition-to-future-features':'D illustrates one target patch i=6; repeat for all16 positions to assemble the forecast grid in the target-scoring lane.',
    }
    mapping={key:[p['id'] for p in s.items if p.get('semantic')==key]+([contracts[key]] if key in contracts else []) for key in sorted(expected)}
    if not all(mapping.values()):raise ValueError('Missing semantic dependencies: '+str([k for k,v in mapping.items() if not v]))
    contract={'schema':'shiftwm_architecture_visual_design_contract_v1','scientific_model_sha256':sha(ROOT/'src/shiftwm/real_video_spatial/model.py'),
      'semantic_dependencies':mapping,'count':len(expected),'source_graph_sha256':sha(BASE),
      'fixed_evidence':'A observed support Z_-2:0 and learned pre-FiLM E_-2:0 are fixed within a forecast. Z0 is16x384 normalized last observed features; E0 is16x96 last learned spatial encoding. Fixed values are not frozen model weights.',
      'causal_conditioning':'Context uses patch means and observed transitions plus2past35Dblocks. The separate GRU processes2past plus supplied query-prefix blocks. The temporal predictor receives the same3FiLM support grids, first2past action states and query-prefix endpoint throughh.',
      'transport':'T=softmax_row(Q(H_h)K(E0)^T/sqrt96+4I). Target rows, source columns; all16contributors present; no sparsity assertion.',
      'output':'z_hat_h,i=(1-g_i)z0,i+g_i sum_j T_h,ij z0,j+tanh(LN+affine96to384(H_h)_i). g_i is scalar per patch, channel broadcast. Correction independent of mixture; added after complementary blend.',
      'coordinate_scope':'All shown feature algebra is in fixed train-only shared-channel standardized coordinates; tanh gives unit innovation bound, not an accuracy bound. Returned raw features inverse-normalize after this schematic.',
      'worked_example':{'one_based_target_index':6,'representative_one_based_source_indices':[1,6,11],'matrix_shape':[16,16],'all16_sources_used':True,'illustrative_values_not_learned_or_measured':True,'physical_motion_or_RGB_prediction':False},
      'evaluation_only':{'image_source_index':60,'horizon':10,'encoder':'Same frozen DINOv2 weights,4x4 pooling and training-channel normalization as A','score':'Training-standardized feature MSE; this figure does not report a numerical result','target_enters_predictor':False},
      'historical_context_CEM_or_paired_losses_imported':False}
    (SOURCE/'semantic_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    xml=drawio(s,SOURCE/'architecture-visual-design.drawio')
    audit=render_scene(s,OUT/'architecture_visual_design')
    (OUT/'architecture_visual_design_caption.tex').write_text(CAPTION+'\n')
    (OUT/'architecture_visual_design_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/architecture_visual_design.pdf}\n\\caption[ShiftWM observed-memory architecture.]{'+CAPTION+'}\n\\label{fig:editorial-spatial-method}\n\\end{figure}\n')
    input_paths=[BASE,target,*[ROOT/p for p in graph['sources_sha256']]]
    report={'status':'candidate_not_integrated','renderer_sha256':sha(__file__),'dimensions_inches':[W/72,H/72],
      'semantic_dependencies_preserved':len(expected),'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in input_paths},
      'geometry_sha256':sha(SOURCE/'geometry.json'),'semantic_contract_sha256':sha(SOURCE/'semantic_contract.json'),
      'drawio_sha256':sha(SOURCE/'architecture-visual-design.drawio'),
      'composition_reference':'paper/figure_sources/architecture_visual_design/reference/composition_reference.png',
      'reference_pixels_inspected':True,'reference_authority':'Visual hierarchy only; generated photographs,trainable-free keys,action-to-observation and incorrect gate/update wiring rejected.',
      'visual_review':'pending actual-PDF and independent review; layout tests are not visual approval',
      **audit,**xml,'outputs_sha256':{e:sha(OUT/f'architecture_visual_design.{e}') for e in ('pdf','svg','png')}}
    (OUT/'architecture_visual_design_evidence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
