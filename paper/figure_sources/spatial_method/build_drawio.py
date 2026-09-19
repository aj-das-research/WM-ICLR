#!/usr/bin/env python3
"""Native editable counterpart of the reviewed normalized spatial predictor graph.

All patch and matrix cells, labels, operators and connectors are mxCells.
No raster image is embedded and no scientific model/data is modified.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
INK = '#233247'; MUTED = '#59677B'; BLUE = '#586EB4'; TEAL = '#147F78'; OCHRE = '#A66418'
PALETTE = ['#5B70B7','#709CC7','#8EBBC4','#63A795','#91A6CD','#A28FC2','#CDA1AD','#D6B18C',
           '#7195AD','#8CB3A5','#B6C5AA','#D3CAAA','#7085B0','#9AA6B9','#AB9FB6','#C4B5C7']
MODEL_SHA = '054d0aa1c479c1dc722b674bb59dc35015c69ebacec0f0602b84ded40651edf2'
PDF_SHA = 'a877aafc7dd3aab6a3f9640158a7ebfa389f5970ebeb845aba0f963725bae190'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
NODES = {}; EDGES = []; ALIASES = {}


def create():
    if sha(ROOT/'src/shiftwm/real_video_spatial/model.py') != MODEL_SHA or sha(ROOT/'paper/generated/real_video/spatial_method.pdf') != PDF_SHA:
        raise ValueError('Reviewed model/PDF identity changed; new semantic review required')
    file = ET.Element('mxfile', {'host':'app.diagrams.net','type':'device'})
    page = ET.SubElement(file,'diagram',{'id':'spatial-transport-architecture','name':'Spatial transport — normalized predictor'})
    graph = ET.SubElement(page,'mxGraphModel',{'dx':'1520','dy':'1540','grid':'1','gridSize':'10','guides':'1','tooltips':'1','connect':'1','arrows':'1','fold':'1','page':'1','pageScale':'1','pageWidth':'1520','pageHeight':'1540','math':'0','shadow':'0','adaptiveColors':'auto'})
    root = ET.SubElement(graph,'root'); ET.SubElement(root,'mxCell',{'id':'0'}); ET.SubElement(root,'mxCell',{'id':'1','parent':'0'})

    def vertex(id, label, x,y,w,h,kind='module',fill='#F1F4F7',color=INK,font=22,shape=None,parent='1',semantic=True):
        style = f'whiteSpace=wrap;html=0;align=center;verticalAlign=middle;fontFamily=Helvetica;fontSize={font};fontColor={color};strokeColor={color};strokeWidth=1.6;fillColor={fill};shadow=0;spacing=6;'
        if kind in ('label','heading'):
            style += 'shape=text;strokeColor=none;fillColor=none;'
        elif kind == 'operator': style += 'shape=ellipse;aspect=fixed;'
        elif kind == 'junction': style += 'shape=ellipse;fillColor='+color+';'
        elif kind == 'group': style += 'group;strokeColor=none;fillColor=none;container=1;collapsible=0;'
        elif kind == 'cell': style += 'rounded=0;strokeColor=#FFFFFF;strokeWidth=1;spacing=0;'
        else: style += 'rounded=1;arcSize=12;'
        if kind == 'heading': style += 'fontStyle=1;align=left;'
        if shape: style += shape
        cell = ET.SubElement(root,'mxCell',{'id':id,'value':label,'style':style,'vertex':'1','parent':parent})
        ET.SubElement(cell,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
        NODES[id] = {'id':id,'label':label,'kind':kind,'bbox':[x,y,w,h],'parent':parent,'semantic':semantic}
        return id

    def label(id,text,x,y,w,h,font=21,color=MUTED):
        return vertex(id,text,x,y,w,h,kind='label',font=font,color=color,semantic=False)

    def grid(id,x,y,size,mode='anchor',parent='1',n=4):
        vertex(id,'',x,y,size,size,kind='group',parent=parent)
        def rgb(hex): return tuple(int(hex[i:i+2],16) for i in (1,3,5))
        def hexcolor(rgb): return '#'+''.join(f'{min(255,max(0,round(v))):02X}' for v in rgb)
        for row in range(n):
            for col in range(n):
                i = row*n+col
                if n == 16:
                    # Exact symbolic matrix recipe from the reviewed schematic, not checkpoint weights.
                    weight = .15/16 + (.50 if row==col else 0) + (.35 if col==(row+1)%16 else 0)
                    gray = 248 - 190*weight/.859375
                    color=hexcolor((gray,gray,gray))
                elif mode == 'anchor': color = PALETTE[i]
                else:
                    base = rgb(PALETTE[i]); neighbor = rgb(PALETTE[(i+1)%16]); average=[sum(rgb(c)[k] for c in PALETTE)/16 for k in range(3)]
                    mixed=[.5*base[k]+.35*neighbor[k]+.15*average[k] for k in range(3)]
                    if mode=='predicted': mixed=[.55*base[k]+.45*mixed[k]+6*math.sin(i+k) for k in range(3)]
                    color=hexcolor(mixed)
                vertex(f'{id}-cell-{row}-{col}','',col*size/n,row*size/n,size/n,size/n,kind='cell',fill=color,color='#FFFFFF',parent=id,semantic=False)
        if n == 4:
            vertex(id+'-selected','',size/4,size/4,size/4,size/4,kind='cell',parent=id,semantic=False,
                    shape='fillColor=none;strokeColor='+INK+';strokeWidth=2;')
        return id

    def edge(id,source,target,points=None,exit=(1,.5),entry=(0,.5),color=INK,meaning=''):
        style = f'edgeStyle=segmentEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=0;endArrow=block;endFill=1;endSize=9;strokeColor={color};strokeWidth=2;exitX={exit[0]};exitY={exit[1]};exitPerimeter=0;entryX={entry[0]};entryY={entry[1]};entryPerimeter=0;'
        cell=ET.SubElement(root,'mxCell',{'id':id,'value':'','style':style,'edge':'1','parent':'1','source':source,'target':target})
        geometry=ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})
        if points:
            array=ET.SubElement(geometry,'Array',{'as':'points'})
            for x,y in points: ET.SubElement(array,'mxPoint',{'x':str(x),'y':str(y)})
        EDGES.append({'id':id,'source':source,'target':target,'exit':list(exit),'entry':list(entry),'waypoints':points or [],'meaning':meaning})

    # A: two distinct conditioning branches; no future observations enter either branch.
    vertex('panel-a','',20,55,1480,555,fill='#F6F8FB',color='#D4DCE5',semantic=False)
    vertex('panel-a-heading','A   Build a causal horizon state',35,8,920,40,kind='heading',font=28,semantic=False)
    vertex('observed-history','',55,135,225,90,kind='group')
    for j, text in enumerate(('Z₋₂','Z₋₁','Z₀')):
        grid('observed-'+str(j),j*77,0,58,parent='observed-history')
        vertex('observed-label-'+str(j),text,j*77,59,58,25,kind='label',font=20,parent='observed-history',semantic=False)
    label('observed-caption','Observed 4×4 grids',45,87,245,35)
    vertex('spatial-encoder','Spatial\nencoder',340,130,160,90)
    vertex('encoded-support','E₋₂:₀\npre-FiLM',560,135,160,80,fill='#EEF2F8',color=BLUE)
    vertex('spatial-mean','Patch mean',570,270,170,60)
    vertex('support-context','Support-only\ntransition\ncontext',770,300,190,90,fill='#EEF2F8',color=BLUE)
    vertex('film','FiLM',920,130,140,90,fill='#EEF2F8',color=BLUE)
    vertex('temporal-predictor','LeWM\ntemporal\npredictor',1150,130,180,90)
    vertex('horizon-state','Hₕ',1390,145,80,60,fill='#E7EEF8',color=BLUE,font=27)
    vertex('past-actions','Past actions\na₋₁:₀',60,400,180,60,fill='#FFFFFF',font=21)
    vertex('past-junction','',285,425,10,10,kind='junction')
    vertex('future-actions','Causal prefix\na₁:ₕ',60,520,180,60,fill='#FFFFFF',font=21)
    vertex('action-concat','∥',270,495,40,40,kind='operator',fill='#FFFFFF',font=25)
    vertex('prefix-gru','Prefix GRU',390,480,170,70)
    vertex('action-selector','Select states\nu₋₁, u₀, uₕ',650,480,210,70)
    edge('observations-to-encoder','observed-history','spatial-encoder',[(310,175)],exit=(1,40/90),entry=(0,.5),meaning='Encode each of three normalized observed patch grids')
    edge('encoder-to-support','spatial-encoder','encoded-support',meaning='Shared spatial encoding before context FiLM')
    edge('support-to-film','encoded-support','film',meaning='Fixed encoded support remains the visual input at every horizon')
    edge('support-to-mean','encoded-support','spatial-mean',[(640,240),(655,240)],exit=(.5,1),entry=(.5,0),meaning='Mean over the16 spatial patch tokens, retaining3 observed times')
    edge('means-to-context','spatial-mean','support-context',[(755,300),(755,345)],meaning='Observed support states/differences enter TransitionContext')
    edge('past-to-junction','past-actions','past-junction',meaning='The same normalized past actions feed both branches')
    edge('past-to-context','past-junction','support-context',[(750,430),(750,375)],exit=(1,.5),entry=(0,5/6),meaning='Two past action blocks condition support-only transition context')
    edge('context-to-film','support-context','film',[(990,345)],entry=(.5,1),color=BLUE,meaning='Fixed support context sets FiLM scale and shift')
    edge('film-to-temporal','film','temporal-predictor',meaning='Context-conditioned observed states enter temporal predictor')
    edge('past-to-concat','past-junction','action-concat',exit=(.5,1),entry=(.5,0),meaning='First concatenate the two past actions')
    edge('future-to-concat','future-actions','action-concat',[(290,550)],entry=(.5,1),meaning='Append only the given query action prefix up to horizonh')
    edge('concat-to-gru','action-concat','prefix-gru',meaning='Unidirectional chronological action encoding')
    edge('gru-to-selector','prefix-gru','action-selector',meaning='Select two observed-action states plus current-prefix state')
    edge('actions-to-temporal','action-selector','temporal-predictor',[(1240,515)],entry=(.5,1),meaning='Action states condition LeWM, not the support-context network')
    edge('temporal-to-hidden','temporal-predictor','horizon-state',meaning='One horizon-specific hidden token per patch')
    label('causal-note','Observations and support context stay fixed across horizons',380,566,1010,31,font=20)

    # B: exact normalized decoder, with repeated aliases explicitly identified in the graph ledger.
    vertex('panel-b-heading','B   Anchor, mix, then correct',35,622,900,40,kind='heading',font=28,semantic=False)
    grid('anchor',50,825,110)
    label('anchor-caption','Z₀ • last observation',25,944,245,40)
    grid('mixed-anchor',505,825,110,mode='mixed')
    label('mixed-caption','Tₕ Z₀',490,942,140,40,font=24,color=TEAL)
    grid('normalized-forecast',1290,825,110,mode='predicted')
    label('forecast-caption','Ẑₕ • normalized',1250,942,190,40,font=23,color=BLUE)
    vertex('transport-product','×',365,860,40,40,kind='operator',fill='#FFFFFF',color=TEAL,font=28)
    vertex('gated-mix-product','×',730,860,40,40,kind='operator',fill='#FFFFFF',color=TEAL,font=28)
    vertex('identity-product','×',940,760,40,40,kind='operator',fill='#FFFFFF',color=BLUE,font=28)
    vertex('anchor-blend','+',940,860,40,40,kind='operator',fill='#FFFFFF',color=TEAL,font=28)
    vertex('innovation-merge','+',1125,860,40,40,kind='operator',fill='#FFFFFF',color=OCHRE,font=28)
    vertex('hidden-for-gate','Hₕ',430,710,80,60,fill='#FFFFFF',color=BLUE,font=26)
    vertex('gate','gₕ = σ(Wg Hₕ + bg)',540,710,260,60,fill='#EEF6F3',color=TEAL,font=22)
    vertex('one-minus-gate','1 − gₕ',845,710,90,60,fill='#FFFFFF',color=BLUE,font=23)
    grid('transport-matrix',305,1020,160,n=16)
    label('matrix-caption','Tₕ • 16×16\nrow sums = 1',300,1189,180,66,font=22,color=TEAL)
    vertex('row-softmax','row softmax\n(QₕK₀ᵀ/√96 + 4I)',60,1115,190,70,fill='#EEF6F3',color=TEAL,font=21)
    vertex('hidden-for-query','Hₕ',95,1410,80,55,fill='#FFFFFF',color=BLUE,font=26)
    vertex('query-projection','Qₕ = Wq Hₕ',60,1300,150,60,fill='#FFFFFF',color=TEAL,font=20)
    vertex('encoding-for-key','E₀',337.5,1410,80,55,fill='#FFFFFF',color=BLUE,font=26)
    vertex('key-projection','K₀ = Wk E₀',300,1300,155,60,fill='#FFFFFF',color=TEAL,font=20)
    label('pre-film-key-note','E₀: last observed encoding, before FiLM',45,1482,655,36,font=20)
    vertex('hidden-for-innovation','Hₕ',850,1130,90,60,fill='#FFFFFF',color=BLUE,font=26)
    vertex('residual-projection','Rₕ\nLayerNorm\n+ linear',1030,1110,165,100,fill='#FBF2E7',color=OCHRE,font=20)
    vertex('bounded-innovation','Δₕ = tanh(Rₕ)\n|Δₕ| ≤ 1',1250,1110,150,100,fill='#FBF2E7',color=OCHRE,font=20)
    edge('anchor-to-mixing-product','anchor','transport-product',color=TEAL,meaning='Multiply the semantic mixing matrix by the fixed observation anchor')
    edge('matrix-to-mixing-product','transport-matrix','transport-product',exit=(.5,0),entry=(.5,1),color=TEAL,meaning='Tₕ is a16×16 row-stochastic feature-mixing matrix')
    edge('product-to-mixed-anchor','transport-product','mixed-anchor',color=TEAL,meaning='Tₕ Z₀ in shared-channel standardized coordinates')
    edge('mixed-to-gated-product','mixed-anchor','gated-mix-product',color=TEAL,meaning='Apply per-target-patch sigmoid gate to mixed anchor')
    edge('gate-hidden-input','hidden-for-gate','gate',color=TEAL,meaning='Affine gate includes learned bias bg initialized−3')
    edge('gate-to-mixed-product','gate','gated-mix-product',[(670,810),(750,810)],exit=(.5,1),entry=(.5,0),color=TEAL,meaning='gₕ multiplies each target patch feature vector')
    edge('gate-to-complement','gate','one-minus-gate',None,entry=(0,.5),color=BLUE,meaning='Identity-branch multiplier is exactly1−gₕ')
    edge('anchor-identity-bypass','anchor','identity-product',[(105,680),(960,680)],exit=(.5,0),entry=(.5,0),color=BLUE,meaning='The same last observed Z₀ is fixed at every horizon')
    edge('complement-to-identity-product','one-minus-gate','identity-product',[(890,780)],exit=(.5,1),entry=(0,.5),color=BLUE,meaning='Multiply Z₀ by the complementary gate')
    edge('identity-to-blend','identity-product','anchor-blend',exit=(.5,1),entry=(.5,0),color=BLUE,meaning='Identity anchor contribution')
    edge('gated-mix-to-blend','gated-mix-product','anchor-blend',color=TEAL,meaning='Gated transported-anchor contribution')
    edge('blend-to-innovation-merge','anchor-blend','innovation-merge',meaning='(1−gₕ)Z₀ +gₕTₕZ₀')
    edge('innovation-hidden-input','hidden-for-innovation','residual-projection',color=OCHRE,meaning='Rₕ is affine output projection after LayerNorm, including bias')
    edge('residual-to-tanh','residual-projection','bounded-innovation',color=OCHRE,meaning='Elementwise tanh; innovation_bound=1 in the frozen transport arm')
    edge('innovation-to-merge','bounded-innovation','innovation-merge',[(1325,1040),(1145,1040)],exit=(.5,0),entry=(.5,1),color=OCHRE,meaning='Add the bounded feature correction; not a physical displacement')
    edge('merge-to-forecast','innovation-merge','normalized-forecast',color=BLUE,meaning='Normalized future feature grid; no predicted RGB')
    edge('hidden-to-query','hidden-for-query','query-projection',exit=(.5,0),entry=(.5,1),color=TEAL,meaning='Bias-free query projection of Hₕ')
    edge('encoding-to-key','encoding-for-key','key-projection',exit=(.5,0),entry=(.5,1),color=TEAL,meaning='Bias-free key projection of last support encoding before FiLM')
    edge('queries-to-softmax','query-projection','row-softmax',[(135,1230),(200,1230)],exit=(.5,0),entry=(14/19,1),color=TEAL,meaning='Qₕ forms scaled query/key scores')
    edge('keys-to-softmax','key-projection','row-softmax',[(377.5,1270),(230,1270)],exit=(.5,0),entry=(17/19,1),color=TEAL,meaning='K₀ remains anchored to actual observed pre-FiLM features')
    edge('softmax-to-matrix','row-softmax','transport-matrix',[(275,1150),(275,1100)],color=TEAL,meaning='Row softmax after+4I identity bias, scale√96')
    label('bounded-schematic-note','Schematic latent features • not learned weights or physical flow',650,1450,820,50,font=20)
    ALIASES.update({'anchor':{'source':'observed-history','selection':'last observed grid Z₀'},
                    'hidden-for-query':{'source':'horizon-state'},'hidden-for-gate':{'source':'horizon-state'},
                    'hidden-for-innovation':{'source':'horizon-state'},
                    'encoding-for-key':{'source':'encoded-support','selection':'last observed grid encoding E₀, before FiLM'}})
    ET.indent(file,space='  ')
    target = OUT/'spatial-transport-architecture.drawio'
    ET.ElementTree(file).write(target,encoding='utf-8',xml_declaration=True)
    ledger={'scope':'Normalized _predict_normalized transport-arm computation; frozen encoder and final feature unstandardization lie outside this reviewed panel scope.',
            'nodes':NODES,'edges':EDGES,'aliases':ALIASES,
            'equations':{'transport':'T_h = softmax_row(Q_h K_0^T / sqrt(96) + 4 I)',
                         'gate':'g_h = sigmoid(W_g H_h + b_g)', 'innovation':'Delta_h = tanh(R_h)',
                         'forecast':'Zhat_h = (1-g_h)*Z_0 + g_h*(T_h @ Z_0) + Delta_h'},
            'sources':{'src/shiftwm/real_video_spatial/model.py':sha(ROOT/'src/shiftwm/real_video_spatial/model.py'),
                       'src/shiftwm/model.py':sha(ROOT/'src/shiftwm/model.py'),
                       'paper/generated/real_video/spatial_method.pdf':sha(ROOT/'paper/generated/real_video/spatial_method.pdf')},
            'classification':'Editable illustrative architecture. Patch colors/matrix grayscale are symbolic, not measured data.'}
    (OUT/'canonical-graph.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n')
    return target


if __name__ == '__main__':
    print(create())
