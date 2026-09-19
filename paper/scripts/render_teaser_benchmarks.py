#!/usr/bin/env python3
"""Evidence-bound cinematic teaser; shared primitives emit editable draw.io and PDF.

The native application export is optional and must never be claimed from XML validation.
No model inference, training, or new dataset selection is performed by this script.
"""
from pathlib import Path
import base64, hashlib, json, math, shutil, xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch
from matplotlib.path import Path as MPath
from matplotlib.transforms import Bbox
from PIL import Image, ImageOps
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'paper/figure_sources/teaser_cinematic'
OUT=ROOT/'paper/generated/editorial'
DESIGN=ROOT/'paper/design/teaser_benchmarks'
PUBLIC=ROOT/'paper/figure_sources/teaser_benchmarks'
REF=ROOT/'paper/figure_sources/visual_story_references_v1/teaser_composition_reference.png'
REF_SHA='59bf3592391c1e9e4461ebcf28e4234e4768cb6e30df4f72008410bf6042e3bb'
W,H=396.,234.
INK='#1C304D';MUTED='#56667A';BLUE='#2378B7';TEAL='#008E88';WARM='#DB6947';GOLD='#CF942D';PALE='#E7F5F3'
CAPTION=(r'\textbf{Visual forecasting across the paper\textquotesingle s recorded and simulated settings.} '
 r'Gallery: IWS PushT/Box/Rope are separate inputs to ongoing training; Open-H is a physical-phantom input audit only. '
 r'Historical PushT, Reacher, drone and tissue simulations use distinct context models. '
 r'Only DROID connects to the compared spatial decoder: autoregression reuses predictions; ShiftWM mixes a fixed observed reference with bounded corrections. '
 r'Gold ports denote causal supplied-action prefixes; both routes share history. Three schematic steps depict ten forecast steps, not generated RGB; the camera is illustrative. '
 r'Bottom: all 141 DROID development episodes. The 5.30\% reduction compares population mean native, training-standardized h10 feature MSE with matched autoregression. '
 r'Uncertainty: Figure~\ref{fig:editorial-spatial}. Recorded images: DROID and Open-H (CC BY 4.0); IWS \citep{zhang2026rlawm}. Input-selection rules and simulator attribution accompany the editable figure.')

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def checked_evidence():
    assert sha(REF)==REF_SHA,'Generated composition reference changed'
    p=json.loads((SRC/'population.json').read_text());m=json.loads((SRC/'asset_manifest.json').read_text())
    for rel,h in p['authoritative_source_sha256'].items():assert sha(ROOT/rel)==h,rel
    rows=p['episodes_sorted_by_gain'];assert len(rows)==141
    gains=np.asarray([100*(r['baseline_mse']-r['transport_mse'])/r['baseline_mse'] for r in rows])
    np.testing.assert_allclose(gains,[r['gain_percent'] for r in rows],rtol=0,atol=1e-11)
    assert (int((gains>0).sum()),int((gains<0).sum()),int((gains==0).sum()))==(136,5,0)
    # Reconstruct equal-window-within-episode, then equal-episode / seed aggregation.
    vals={};window_count=0
    for mode in ('autoregressive','transport'):
        by_seed=[]
        for seed in range(3):
            r=json.loads((ROOT/f'reports/real_video_spatial/{mode}_s{seed}_validation.json').read_text())
            grouped={}
            for v in r['windows']:
                grouped.setdefault(v['episode_id'],[]).append(float(v['native_mse'][9]));window_count+=1
            assert len(grouped)==141 and sum(map(len,grouped.values()))==1631
            values={k:float(np.mean(v)) for k,v in grouped.items()}
            for e in r['episodes']:np.testing.assert_allclose(values[e['episode_id']],e['native_mse'][9],rtol=0,atol=1e-12)
            by_seed.append(values)
        vals[mode]={k:float(np.mean([s[k] for s in by_seed])) for k in by_seed[0]}
    for r in rows:
        np.testing.assert_allclose([vals['autoregressive'][r['episode_id']],vals['transport'][r['episode_id']]],[r['baseline_mse'],r['transport_mse']],rtol=0,atol=1e-12)
    a=float(np.mean(list(vals['autoregressive'].values())));b=float(np.mean(list(vals['transport'].values())))
    gain=100*(a-b)/a
    np.testing.assert_allclose(gain,p['aggregate_h10']['overall_relative_error_reduction_percent'],rtol=0,atol=1e-11)
    for v in m['records']:
        assert v['split']=='internal_train' and v['frame']==0
        assert sha(ROOT/v['asset'])==v['asset_sha256']
        assert hashlib.sha256(np.asarray(Image.open(ROOT/v['asset'])).tobytes()).hexdigest()==v['pixel_sha256']
    for v in m['droid']['assets']:assert sha(ROOT/v['path'])==v['sha256']
    return p,m,rows,gain,{'window_values_recomputed':window_count,'seed_episode_values':846,'positive':136,'negative':5,'ties':0,'ar':a,'ours':b,'overall_relative_reduction_percent':gain,'mean_episode_percent':float(gains.mean())}

def checked_gallery():
    manifest=json.loads((ROOT/'paper/figure_sources/benchmark_gallery/asset_manifest.json').read_text())
    for r in manifest['records']:
        p=ROOT/r['asset'];assert sha(p)==r['asset_sha256']
        assert hashlib.sha256(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()==r['pixel_sha256']
    old=json.loads((ROOT/'paper/figures/split_assets/manifest.json').read_text())
    for name in ('pusht.png','reacher.png'):
        rec=next(r for r in old['assets'] if r['file']==name);assert sha(ROOT/'paper/figures/split_assets'/name)==rec['sha256']
    assert {r['task'] for r in manifest['records']}=={'drone','surgery','openh'}
    inventory=json.loads((ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json').read_text())
    assert len(inventory['records'])==9
    for r in inventory['records']:assert sha(ROOT/r['asset'])==r['asset_sha256']
    return manifest

class Canvas:
    def __init__(self,name,w=W,h=H):self.name=name;self.w=w;self.h=h;self.elements=[];self.edges=[];self.photos=[]
    def add(self,kind,**kw):
        kw.update(kind=kind,id=f'{self.name}-{len(self.elements):04d}');self.elements.append(kw);return kw['id']
    def rect(self,x,y,w,h,fill,stroke='none',r=0,lw=.6,alpha=1,role=None):return self.add('rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,r=r,lw=lw,alpha=alpha,role=role)
    def circle(self,x,y,r,fill,stroke='none',lw=.6,alpha=1):return self.add('circle',x=x,y=y,r=r,fill=fill,stroke=stroke,lw=lw,alpha=alpha)
    def poly(self,pts,fill,stroke='none',lw=.6,alpha=1):return self.add('polygon',pts=pts,fill=fill,stroke=stroke,lw=lw,alpha=alpha)
    def line(self,pts,color=INK,lw=.8,arrow=False,dashed=False,alpha=1,semantic=None):
        id=self.add('line',pts=pts,color=color,lw=lw,arrow=arrow,dashed=dashed,alpha=alpha,semantic=semantic)
        if semantic:self.edges.append({'id':id,'meaning':semantic,'points':pts})
        return id
    def text(self,x,y,s,size=8,color=INK,ha='left',bold=False):return self.add('text',x=x,y=y,text=s,size=size,color=color,ha=ha,bold=bold)
    def photo(self,path,x,y,w):
        im=Image.open(path).convert('RGB');h=w*im.height/im.width
        id=self.add('image',x=x,y=y,w=w,h=h,path=str(Path(path).relative_to(ROOT)))
        self.photos.append({'id':id,'path':str(Path(path).relative_to(ROOT)),'sha256':sha(path),'pixel_sha256':hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),'shape':list(np.asarray(im).shape),'crop':'none','geometry':[x,y,w,h]});return h
    def illustration(self,path,x,y,w):
        im=Image.open(path);h=w*im.height/im.width
        return self.add('image',x=x,y=y,w=w,h=h,path=str(Path(path).relative_to(ROOT)),role='Generated conceptual camera; not evidence',sha256=sha(path))
    def render(self,basename,proof=True):
        fig=plt.figure(figsize=(self.w/72,self.h/72),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,self.w),ylim=(self.h,0));ax.set_aspect('equal');ax.axis('off');texts=[];line_artists=[]
        for e in self.elements:
            k=e['kind']
            if k=='rect':
                if e['r']:p=FancyBboxPatch((e['x'],e['y']),e['w'],e['h'],boxstyle=f"round,pad=0,rounding_size={e['r']}",fc=e['fill'],ec=e['stroke'],lw=e['lw'],alpha=e['alpha'])
                else:p=Rectangle((e['x'],e['y']),e['w'],e['h'],fc=e['fill'],ec=e['stroke'],lw=e['lw'],alpha=e['alpha'])
                ax.add_patch(p)
            elif k=='circle':ax.add_patch(Circle((e['x'],e['y']),e['r'],fc=e['fill'],ec=e['stroke'],lw=e['lw'],alpha=e['alpha']))
            elif k=='polygon':ax.add_patch(Polygon(e['pts'],closed=True,fc=e['fill'],ec=e['stroke'],lw=e['lw'],alpha=e['alpha']))
            elif k=='line':
                if e['arrow']:
                    artist=FancyArrowPatch(path=MPath(e['pts'],[MPath.MOVETO]+[MPath.LINETO]*(len(e['pts'])-1)),arrowstyle='-|>',mutation_scale=6.5,color=e['color'],lw=e['lw'],alpha=e['alpha'],linestyle='--' if e['dashed'] else '-')
                    ax.add_patch(artist);line_artists.append((e,artist))
                else:
                    ax.plot(*zip(*e['pts']),color=e['color'],lw=e['lw'],alpha=e['alpha'],linestyle='--' if e['dashed'] else '-',solid_capstyle='round')
                    line_artists.append((e,MPath(e['pts'],[MPath.MOVETO]+[MPath.LINETO]*(len(e['pts'])-1))))
            elif k=='text':texts.append(ax.text(e['x'],e['y'],e['text'],ha=e['ha'],va='center',color=e['color'],fontsize=e['size'],fontweight='bold' if e['bold'] else 'normal'))
            elif k=='image':ax.imshow(Image.open(ROOT/e['path']),extent=(e['x'],e['x']+e['w'],e['y']+e['h'],e['y']),interpolation='none',zorder=2)
        fig.canvas.draw();ren=fig.canvas.get_renderer();b=[t.get_window_extent(ren) for t in texts]
        overlaps=[(texts[i].get_text(),texts[j].get_text()) for i,a in enumerate(b) for j,z in enumerate(b) if i<j and a.overlaps(z)]
        clipped=[t.get_text() for t,z in zip(texts,b) if z.x0<-.05 or z.y0<-.05 or z.x1>fig.bbox.width+.05 or z.y1>fig.bbox.height+.05]
        # Exhaustive actual-font/connector check, including arrowheads. CLOSEPOLY
        # dummy coordinates are never treated as real line segments.
        bounds=[]
        for t,bb in zip(texts,b):
            z=bb.transformed(ax.transData.inverted())
            bounds.append((t.get_text(),Bbox.from_extents(min(z.x0,z.x1),min(z.y0,z.y1),max(z.x0,z.x1),max(z.y0,z.y1))))
        edge_hits=[];segments_count=0
        for e,artist in line_artists:
            path=artist.get_path() if hasattr(artist,'get_path') else artist
            segments=[];cur=start=None
            for v,code in path.iter_segments(curves=False):
                if code==MPath.MOVETO:cur=tuple(v);start=cur
                elif code==MPath.LINETO:nxt=tuple(v);segments.append((cur,nxt));cur=nxt
                elif code==MPath.CLOSEPOLY:
                    if cur!=start:segments.append((cur,start))
                    cur=start
                elif code==MPath.STOP:break
                else:raise ValueError('Unexpected connector path code')
            segments_count+=len(segments)
            for label,bb in bounds:
                pad=.75+e['lw']/2
                padded=Bbox.from_extents(bb.x0-pad,bb.y0-pad,bb.x1+pad,bb.y1+pad)
                if any(MPath([u,v],[MPath.MOVETO,MPath.LINETO]).intersects_bbox(padded,filled=False) for u,v in segments):
                    edge_hits.append({'connector':e['id'],'text':label,'clearance_pt':.75,'stroke_halfwidth_included':True})
        if overlaps or clipped or edge_hits:
            raise ValueError(str({'text_overlaps':overlaps,'clipped':clipped,'connector_text':edge_hits}))
        for ext in ('pdf','svg','png'):
            kw={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
            fig.savefig(str(basename)+'.'+ext,dpi=300,**kw)
        if proof:
            fig.savefig(str(basename)+'_paper_width.png',dpi=120)
            ImageOps.grayscale(Image.open(str(basename)+'_paper_width.png')).save(str(basename)+'_grayscale.png')
        plt.close(fig)
        return {'text_overlaps':overlaps,'clipped':clipped,'minimum_font_pt':min(e['size'] for e in self.elements if e['kind']=='text'),'connector_text_intersections':edge_hits,'connector_objects':len(line_artists),'actual_connector_segments':segments_count,'text_bounds_checked':len(bounds),'connector_clearance_pt':.75,'stroke_halfwidth_included':True}
    def drawio(self,path):
        # Every label, plane, cell, edge and camera part remains an editable native object.
        mx=ET.Element('mxfile',{'host':'app.diagrams.net','type':'device'});d=ET.SubElement(mx,'diagram',{'name':self.name,'id':self.name})
        graph=ET.SubElement(d,'mxGraphModel',{'page':'1','pageWidth':str(self.w),'pageHeight':str(self.h),'grid':'0','guides':'1','adaptiveColors':'auto'})
        root=ET.SubElement(graph,'root');ET.SubElement(root,'mxCell',{'id':'0'});ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})
        def vertex(id,x,y,w,h,style,value=''):
            c=ET.SubElement(root,'mxCell',{'id':id,'value':value,'style':style,'vertex':'1','parent':'1'});ET.SubElement(c,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
        def edge(id,pts,color,lw,arrow=False,dashed=False,alpha=1):
            c=ET.SubElement(root,'mxCell',{'id':id,'value':'','style':f'strokeColor={color};strokeWidth={lw};endArrow={"block" if arrow else "none"};endSize=5;dashed={int(dashed)};opacity={100*alpha};rounded=0;','edge':'1','parent':'1'})
            g=ET.SubElement(c,'mxGeometry',{'relative':'1','as':'geometry'})
            for p,as_ in [(pts[0],'sourcePoint'),(pts[-1],'targetPoint')]:ET.SubElement(g,'mxPoint',{'x':str(p[0]),'y':str(p[1]),'as':as_})
            if len(pts)>2:
                a=ET.SubElement(g,'Array',{'as':'points'})
                for p in pts[1:-1]:ET.SubElement(a,'mxPoint',{'x':str(p[0]),'y':str(p[1])})
        for e in self.elements:
            k=e['kind'];id=e['id']
            if k=='text':
                # Text-box width is deterministic and remains freely editable.
                w=max(15,max(map(len,e['text'].splitlines()))*e['size']*.53+6);h=e['size']*1.3*len(e['text'].splitlines())
                x=e['x'] if e['ha']=='left' else e['x']-w if e['ha']=='right' else e['x']-w/2
                vertex(id,x,e['y']-h/2,w,h,f'text;html=0;whiteSpace=wrap;align={e["ha"]};verticalAlign=middle;fontFamily=Liberation Sans;fontSize={e["size"]};fontStyle={int(e["bold"])};fontColor={e["color"]};strokeColor=none;fillColor=none;spacing=0;',e['text'])
            elif k=='rect':vertex(id,e['x'],e['y'],e['w'],e['h'],f'rounded={int(e["r"]>0)};arcSize=8;fillColor={e["fill"]};strokeColor={e["stroke"]};strokeWidth={e["lw"]};opacity={100*e["alpha"]};')
            elif k=='circle':vertex(id,e['x']-e['r'],e['y']-e['r'],2*e['r'],2*e['r'],f'ellipse;fillColor={e["fill"]};strokeColor={e["stroke"]};strokeWidth={e["lw"]};opacity={100*e["alpha"]};')
            elif k=='line':edge(id,e['pts'],e['color'],e['lw'],e['arrow'],e['dashed'],e['alpha'])
            elif k=='polygon':
                # Native mxGraph stencil, not an embedded raster or flattened SVG.
                import zlib
                xs,ys=zip(*e['pts']);x,y=min(xs),min(ys);w=max(xs)-x;h=max(ys)-y
                s=ET.Element('shape',{'name':id,'w':str(w),'h':str(h),'aspect':'variable','strokewidth':'inherit'})
                f=ET.SubElement(s,'foreground');p=ET.SubElement(f,'path')
                for i,(xx,yy) in enumerate(e['pts']):ET.SubElement(p,'move' if i==0 else 'line',{'x':str(xx-x),'y':str(yy-y)})
                ET.SubElement(p,'close');ET.SubElement(f,'fillstroke')
                co=zlib.compressobj(wbits=-15);raw=ET.tostring(s,encoding='utf-8');encoded=base64.b64encode(co.compress(raw)+co.flush()).decode()
                vertex(id,x,y,w,h,f'shape=stencil({encoded});fillColor={e["fill"]};strokeColor={e["stroke"]};strokeWidth={e["lw"]};opacity={100*e["alpha"]};')
            elif k=='image':
                data=base64.b64encode((ROOT/e['path']).read_bytes()).decode();vertex(id,e['x'],e['y'],e['w'],e['h'],f'shape=image;imageAspect=1;aspect=fixed;image=data:image/png,{data};')
        ET.ElementTree(mx).write(path,encoding='utf-8',xml_declaration=True)
        parsed=ET.parse(path);ids=[c.attrib['id'] for c in parsed.findall('.//mxCell')];assert len(ids)==len(set(ids)) and {'0','1'}<=set(ids)
        return {'native_objects':len(ids)-2,'xml_well_formed':True,'native_cli_preview':False,'shared_geometry':True}

def lighten(color,amount):
    c=np.array([int(color[i:i+2],16) for i in (1,3,5)]);c=c*(1-amount)+255*amount;return '#'+''.join(f'{round(x):02x}' for x in c)

def feature(c,x,y,w=30,color=BLUE):
    """Three layered spatial planes; purely schematic, not inferred channel data."""
    # A plane stack reveals spatial layout and channel depth without fabricated error growth.
    for layer in (2,1,0):
        xx=x+layer*3.6;yy=y-layer*3
        c.poly([(xx,yy),(xx+w,yy),(xx+w+5,yy+5),(xx+5,yy+5)],lighten(color,.82),color,.45,.70)
        for i in range(4):
            for j in range(4):
                u=xx+j*w/4;v=yy+i*w/4
                c.rect(u,v,w/4-.3,w/4-.3,lighten(color,.10+.52*((i+2*j)%5)/4),lighten(color,.05),lw=.25,alpha=.95 if layer==0 else .48)
        c.rect(xx,yy,w,w,'none',color,lw=.55)
    return (x+w/2+3,y+w/2-2)

def camera(c,x,y,scale=1):
    c.rect(x,y+3*scale,34*scale,21*scale,INK,'#718399',r=2*scale,lw=.6)
    c.rect(x+5*scale,y,12*scale,5*scale,'#435A77',r=1)
    c.rect(x+2*scale,y+7*scale,6*scale,12*scale,'#536A81',r=1)
    for r,col in [(12,'#BDCDDA'),(10,INK),(8,BLUE),(5,'#165571'),(2.4,'#75D9DF')]:c.circle(x+28*scale,y+14*scale,r*scale,col)
    c.circle(x+25*scale,y+11*scale,1.5*scale,'white')
    c.line([(x+14*scale,y+24*scale),(x+8*scale,y+35*scale)],MUTED,1)
    c.line([(x+14*scale,y+24*scale),(x+24*scale,y+35*scale)],MUTED,1)

def actions(c,x,y,w=117):
    # Cartesian/gripper glyph is conceptual, not an actual command-value trace.
    c.line([(x+7,y+6),(x+7,y-4)],GOLD,1,True);c.line([(x+7,y+6),(x+18,y+6)],GOLD,1,True)
    c.line([(x+28,y-4),(x+28,y+1),(x+23,y+1),(x+23,y+8),(x+26,y+11)],INK,1)
    c.line([(x+28,y+1),(x+33,y+1),(x+33,y+8),(x+30,y+11)],INK,1)
    for i in range(10):c.rect(x+43+i*(w-43)/10,y-1,(w-47)/10,9,lighten(GOLD,.15+(i%3)*.15),r=.7)

def action_port(c,x,y,w,side=False):
    if side:
        # The fixed-source rails run above the lower forecasts. A separate right
        # input port keeps every command alias below those rails, with no junction.
        c.circle(x+w+15,y+10,1.9,GOLD)
        pts=[(x+w+12.8,y+10),(x+w+7.3,y+10)]
    else:
        c.circle(x+w+7,y-8,1.9,GOLD)
        pts=[(x+w+7,y-6),(x+w+4,y-2)]
    c.line(pts,GOLD,.75,True,
           semantic='Gold alias supplies this forecast with its causal command prefix; values are not depicted')

def corners(c,x,y,w,h,color=BLUE):
    for xx,yy,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:c.line([(xx,yy+dy*7),(xx,yy),(xx+dx*7,yy)],color,1)

def footer(c,rows,gain,y=220):
    c.line([(4,y),(392,y)],'#C9D8E1',.6)
    c.text(5,y+9,'136 improve · 5 regress',8.5,INK)
    for i,r in enumerate(rows):
        x=6+(i%47)*3.28;yy=y+18+(i//47)*4.1
        if r['gain_percent']>0:c.circle(x,yy,1.05,TEAL)
        else:
            c.line([(x-1.2,yy-1.2),(x+1.2,yy+1.2)],WARM,.85);c.line([(x-1.2,yy+1.2),(x+1.2,yy-1.2)],WARM,.85)
    c.text(191,y+9,f'{gain:.2f}%',15,TEAL,bold=True)
    c.text(250,y+8,'lower endpoint error',8.5,INK)
    c.text(191,y+23,'DROID development · vs autoregression',8,MUTED)

def iws_strip(c,m,x=197,y=171,w=191):
    c.text(x,y-8,'Training in progress',8,MUTED)
    for i,(r,name) in enumerate(zip(m['records'],['PushT','Box','Rope'])):
        xx=x+i*(w/3);h=c.photo(ROOT/r['asset'],xx,y,w/3-7)
        c.text(xx+(w/3-7)/2,y+h+6,name,8,MUTED,'center')

def mix_detail(c,x,y):
    # Exact algebraic roles, illustrative vector values: weighted source sum + bounded residual.
    for j,col in enumerate((BLUE,TEAL,GOLD)):
        yy=y-10+j*10
        c.rect(x,yy-3,18,6,lighten(col,.4),col,lw=.4)
        for k in range(3):c.circle(x+4+k*5,yy,1.1,col)
        c.line([(x+20,yy),(x+37,y)],col,.7,True)
    c.circle(x+44,y,6,'white',TEAL,.8);c.text(x+44,y,'Σ',8,TEAL,'center')
    c.line([(x+51,y),(x+66,y)],TEAL,.8,True)
    c.rect(x+68,y-4,22,8,lighten(TEAL,.6),TEAL,lw=.5)
    c.line([(x+92,y),(x+102,y)],TEAL,.8,True)
    c.circle(x+109,y,5.5,'white',GOLD,.8);c.text(x+109,y,'+',9,GOLD,'center')
    c.line([(x+109,y+21),(x+109,y+8)],GOLD,.8,True)
    c.line([(x+97,y+18),(x+121,y+18)],GOLD,.8)
    for a in (97,121):c.line([(x+a,y+15),(x+a,y+21)],GOLD,.8)
    c.rect(x+106,y+15,6,6,GOLD,r=.5)
    c.line([(x+115,y),(x+128,y)],TEAL,.8,True)
    c.rect(x+130,y-4,22,8,lighten(TEAL,.35),TEAL,lw=.5)

def compose_a(m,rows,gain):
    c=Canvas('benchmark-story');extra=checked_gallery();bytask={r['task']:r for r in extra['records']}
    # Paper-wide gallery, separated by actual evaluation/training scope.
    c.text(6,7,'IWS · training inputs',8.5,MUTED)
    for i,(r,name) in enumerate(zip(m['records'],('PushT','Box','Rope'))):
        x=6+i*59;c.photo(ROOT/r['asset'],x,17,48);c.text(x+24,61,name,8,MUTED,'center')
    c.line([(188,14),(188,64)],'#DCE4E8',.6)
    c.text(211,7,'Open-H',8.5,MUTED);c.photo(ROOT/bytask['openh']['asset'],211,17,48);c.text(235,61,'Input audit',8,MUTED,'center')
    c.line([(288,14),(288,64)],'#DCE4E8',.6)
    c.text(313,7,'DROID',8.5,BLUE);c.photo(SRC/'assets/case1_recorded_frame_10.png',313,18,69)
    c.text(347,63,'Development',8,BLUE,'center')
    c.text(6,80,'Historical simulations',8.5,MUTED)
    sims=[('PushT',ROOT/'paper/figures/split_assets/pusht.png'),('Reacher',ROOT/'paper/figures/split_assets/reacher.png'),('Drone',ROOT/bytask['drone']['asset']),('Tissue',ROOT/bytask['surgery']['asset'])]
    for i,(name,p) in enumerate(sims):
        x=6+i*44;c.photo(p,x,93,35);c.text(x+17.5,137,name,8,MUTED,'center')
    # Only DROID supplies this illustrated comparison; gallery entries have no model edges.
    c.line([(383,37),(389,37),(389,72),(196,72),(196,107)],BLUE,.85,True,
           semantic='Only the DROID observation conditions the displayed current spatial decoder; other gallery entries belong to distinct studies')
    feature(c,182,115,25,BLUE);c.text(177,155,'Fixed\nobservation',8,BLUE)
    c.text(239,83,'Autoregression',8.5,WARM)
    for x in (233,291,349):feature(c,x,101,24,WARM);action_port(c,x,101,24)
    c.line([(207,115),(223,110),(231,113)],WARM,1.05,True,
           semantic='Observed initial features condition first autoregressive forecast')
    for x,xx in ((265,289),(323,347)):
        c.line([(x,113),(xx,113)],WARM,1.05,True,
               semantic='Autoregressive predicted features feed the next forecast')
    c.text(267,138,'ShiftWM (ours)',8.5,TEAL)
    for j,x in enumerate((233,291,349)):
        feature(c,x,153,24,TEAL);action_port(c,x,153,24,side=True)
        c.line([(207,138),(220+j*6,144+j*2),(x+12,144+j*2),(x+12,150)],TEAL,1.0,True,
               semantic=f'Fixed DROID observed source independently contributes to forecast {j+1}; action-conditioned mixing plus bounded correction; no feedback')
    c.text(250,187,'Mix + bounded correction',8,TEAL)
    c.illustration(ROOT/'paper/figure_sources/visual_story_references_v1/camera_illustration_cutout.png',1,145,35)
    c.text(53,152,'Supplied actions',8,GOLD);actions(c,51,169,121);c.circle(178,174,2.3,GOLD)
    # All headline marks belong to the same completed DROID development comparison.
    c.line([(4,198),(392,198)],'#CFDAE1',.6)
    c.text(5,207,'136 / 141 improve · 5 regress',8.5,INK)
    for i,r in enumerate(rows):
        x=6+(i%47)*3.52;y=217+(i//47)*4.2
        if r['gain_percent']>0:c.circle(x,y,1.05,TEAL)
        else:
            c.line([(x-1.2,y-1.2),(x+1.2,y+1.2)],WARM,.85)
            c.line([(x-1.2,y+1.2),(x+1.2,y-1.2)],WARM,.85)
    c.text(234,208,f'{gain:.2f}%',17,TEAL,bold=True)
    c.text(307,207,'lower h10 error',8.5,INK)
    c.text(234,226,'DROID development · vs AR',8,MUTED)
    return c

def draft_b(m,rows,gain):
    c=Canvas('central-observation',h=258)
    c.text(5,9,'B  Central observation / contrasting dependencies',8.5)
    c.photo(SRC/'assets/case1_recorded_frame_10.png',104,68,170);corners(c,101,65,176,102)
    camera(c,55,114,.9);c.text(114,59,'Recorded DROID scene',8)
    c.text(14,28,'Forecast → next input',8,WARM)
    for x in (168,230,292):feature(c,x,22,23,WARM)
    for x in (200,262):c.line([(x,34),(x+26,34)],WARM,1,True)
    c.line([(281,94),(340,94),(340,178),(294,178)],TEAL,1,True)
    c.text(96,187,'Fixed observed source → every forecast',8,TEAL)
    for x in (32,147,265):feature(c,x,182,20,TEAL)
    c.text(314,111,'Action',8,GOLD);c.text(314,123,'prefixes',8,GOLD)
    footer(c,rows,gain,220);return c

def draft_c(m,rows,gain):
    c=Canvas('panorama',h=258)
    c.text(5,9,'C  Observation → forecast → scoring boundary',8.5)
    c.photo(SRC/'assets/case1_recorded_frame_10.png',7,25,156);c.photo(SRC/'assets/case1_recorded_frame_60.png',284,45,105)
    c.text(8,120,'Observed',8,BLUE);c.text(280,114,'Withheld real target',8,MUTED)
    feature(c,198,56,35,BLUE)
    c.line([(167,68),(191,68)],BLUE,1,True);c.line([(242,73),(268,73)],MUTED,.8,False,True)
    c.text(177,116,'Feature-space scoring',8,MUTED)
    c.text(14,143,'Recursive forecast',8,WARM);c.text(219,143,'Fixed-source forecast',8,TEAL)
    for x in (19,80,141):feature(c,x,159,22,WARM)
    for x in (223,284,345):feature(c,x,159,22,TEAL)
    for x in (49,110):c.line([(x,170),(x+27,170)],WARM,1,True)
    c.line([(213,201),(373,201)],TEAL,1)
    for x in (233,294,355):c.line([(x,201),(x,187)],TEAL,1,True)
    footer(c,rows,gain,220);return c

def main():
    OUT.mkdir(exist_ok=True,parents=True);DESIGN.mkdir(exist_ok=True,parents=True);PUBLIC.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'shiftwm-cinematic-v1','image.composite_image':False})
    p,m,rows,gain,arithmetic=checked_evidence()
    alternatives=[compose_a(m,rows,gain)]
    qas=[]
    for i,c in enumerate(alternatives):qas.append(c.render(DESIGN/f'composition_{"ABC"[i]}'))
    thumbs=[Image.open(DESIGN/f'composition_{s}_paper_width.png').convert('RGB') for s in 'A'];sheet=Image.new('RGB',(max(i.width for i in thumbs)*len(thumbs),max(i.height for i in thumbs)), 'white')
    for i,im in enumerate(thumbs):sheet.paste(im,(i*im.width,0))
    sheet.save(DESIGN/'composition_contact_sheet.png')
    c=alternatives[0]
    inventory=json.loads((ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json').read_text())
    assert len(c.photos)==9 and {r['sha256'] for r in c.photos}=={r['asset_sha256'] for r in inventory['records']}
    native=c.drawio(OUT/'teaser_benchmarks.drawio') # native XML is written before fallback exports
    qa=c.render(OUT/'teaser_benchmarks')
    (OUT/'teaser_benchmarks_caption.tex').write_text(CAPTION+'\n')
    (OUT/'teaser_benchmarks_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/teaser_benchmarks.pdf}\n\\caption[Forecast from a fixed observed reference.]{'+CAPTION+'}\n\\label{fig:editorial-teaser}\n\\end{figure}\n')
    (PUBLIC/'geometry.json').write_text(json.dumps({'size_points':[W,H],'elements':c.elements},indent=2)+'\n')
    (OUT/'teaser_benchmarks_evidence.json').write_text(json.dumps({'status':'candidate_pending_independent_review','renderer_sha256':sha(__file__),'size_inches':[W/72,H/72],'reference_inspected':{'path':str(REF.relative_to(ROOT)),'sha256':sha(REF),'use':'composition only; generated pixels not embedded'},'source_sha256':{str(s.relative_to(ROOT)):sha(s) for s in [ROOT/'paper/figure_sources/visual_story_references_v1/camera_illustration_cutout.png',ROOT/'paper/figure_sources/visual_story_references_v1/camera_illustration_cutout_manifest.json',SRC/'population.json',SRC/'asset_manifest.json',ROOT/'src/shiftwm/real_video_spatial/model.py',ROOT/'paper/figure_sources/benchmark_gallery/asset_manifest.json',ROOT/'paper/figure_sources/benchmark_gallery/ATTRIBUTION.md',ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json',ROOT/'paper/figures/split_assets/manifest.json']},'authoritative_source_sha256':p['authoritative_source_sha256'],'arithmetic':arithmetic,'all_episode_marks':[{'episode_id':r['episode_id'],'gain_percent':r['gain_percent'],'mark':'circle' if r['gain_percent']>0 else 'cross'} for r in rows],'photo_selection':p['photo_selection'],'photos':c.photos,'generated_camera':{'path':'paper/figure_sources/visual_story_references_v1/camera_illustration_cutout.png','role':'Illustrative camera next to supplied commands; no source image occluded','geometry':[1,145,35,42]},'semantic_edges':c.edges,'qa':qa,'alternatives_qa':qas,'native':native,'scope':'Nine paper-wide source settings: completed DROID development comparison, ongoing IWS3 training inputs,4 historical simulations with different context models, and Open-H physical-phantom input audit only. No new inference or all-task current-model performance claim.','caption':CAPTION,'outputs_sha256':{ext:sha(OUT/f'teaser_benchmarks.{ext}') for ext in ('drawio','pdf','svg','png')}},indent=2)+'\n')
    print(json.dumps({'qa':qa,'alternatives_qa':qas,'native':native,'pdf_sha256':sha(OUT/'teaser_benchmarks.pdf')},indent=2))
if __name__=='__main__':main()
