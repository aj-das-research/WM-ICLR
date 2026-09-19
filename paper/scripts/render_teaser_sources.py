#!/usr/bin/env python3
"""Three evidence-preserving alternatives for the teaser's forecast-source story.

No published asset is overwritten. The paired-source sketches are conceptual
feature dependencies, not new predictions, measured mixing weights or RGB output.
"""
from pathlib import Path
import importlib.util
import argparse, base64, json, zlib, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('teaser_benchmarks_base', ROOT/'paper/scripts/render_teaser_benchmarks.py')
B = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(B)
assert B.sha(ROOT/'paper/scripts/render_teaser_benchmarks.py') == '55935cf3f7131e00a772618102adeb29b5254dfd13a66502ffda286e72f964b5'
DESIGN = ROOT/'paper/design/teaser_sources'
OUT = ROOT/'paper/generated/editorial'
PUBLIC = ROOT/'paper/figure_sources/teaser_sources'
INK, MUTED, BLUE, TEAL, WARM, GOLD = B.INK, B.MUTED, B.BLUE, B.TEAL, B.WARM, B.GOLD
CAPTION=(r'\textbf{Retain observations while forecasting.} '
 r'Autoregression feeds its own prediction into the next three-frame input history; ShiftWM retains the observed reference. '
 r'Mix includes the gate in $W=(I-D)+DT$; a projected $\tanh$ correction has unit coordinate bound after training standardization. '
 r'For ShiftWM, blue/gold ports abbreviate conditioning through $H_h$ on fixed observed support and causal action prefixes (Figure~\ref{fig:editorial-spatial-method}). '
 r'Three colored source locations illustrate mixing, not measured weights; grids are schematic features, not generated RGB, and the camera is illustrative. '
 r'Gallery: historical simulations use distinct context models; IWS inputs belong to ongoing training; Open-H is a physical-phantom input audit only. '
 r'Bottom: all 141 DROID development episodes. The 5.30\% reduction compares population mean native, training-standardized h10 feature MSE with matched autoregression; uncertainty appears in Figure~\ref{fig:editorial-spatial}. '
 r'Images: DROID and Open-H (CC BY 4.0); IWS \citep{zhang2026rlawm}. Simulator attribution and selection rules accompany the editable figure.')


def gallery(c, manifest):
    extra = B.checked_gallery(); by = {r['task']:r for r in extra['records']}
    c.text(5,7,'Historical simulations',8,MUTED)
    sims=[('PushT', ROOT/'paper/figures/split_assets/pusht.png'),
          ('Reacher', ROOT/'paper/figures/split_assets/reacher.png'),
          ('Drone', ROOT/by['drone']['asset']), ('Tissue', ROOT/by['surgery']['asset'])]
    for i,(label,path) in enumerate(sims):
        x=5+36*i;c.photo(path,x,19,30);c.text(x+15,58,label,8,MUTED,'center')
    c.text(153,7,'IWS · training inputs',8,MUTED)
    for i,(r,label) in enumerate(zip(manifest['records'],('PushT','Box','Rope'))):
        x=153+37*i;c.photo(ROOT/r['asset'],x,21,33);c.text(x+16.5,58,label,8,MUTED,'center')
    c.text(274,7,'Input audit',8,MUTED)
    c.photo(ROOT/by['openh']['asset'],274,19,43);c.text(295.5,58,'Open-H',8,MUTED,'center')
    c.text(337,7,'DROID',8,BLUE)
    c.photo(B.SRC/'assets/case1_recorded_frame_10.png',329,19,62)
    c.text(360,58,'Development',8,BLUE,'center')
    for x in (146,268,323): c.line([(x,17),(x,62)],'#DCE4E8',.5)
    c.line([(5,69),(391,69)],'#C9D8E1',.6)


def evidence(c, rows, gain):
    c.line([(4,197),(392,197)],'#C9D8E1',.6)
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


def sheet(c,x,y,w=32,color=BLUE,kind='source'):
    """Flat semantic-feature glyph; colors/positions are illustrative, not data."""
    c.rect(x-2,y-2,w+4,w+4,'white',B.lighten(color,.40),r=1,lw=.65)
    for i in range(4):
        for j in range(4):
            shade=(i*3+j*2)%7/7
            c.rect(x+j*w/4,y+i*w/4,w/4-.55,w/4-.55,B.lighten(color,.1+.68*shade))
    return x+w/2,y+w/2


def arc(c,pts,color,lw=.9,arrow=True,meaning=None):
    c.line(pts,color,lw,arrow,semantic=meaning)


def export_native(c,path):
    """A short activation glyph is an editable open stroke, not a graph edge.

    diagrams.net discarded its dense waypoint edge during actual native export.
    The local stencil uses the same24coordinates, while the frozen primitive
    helper and every scientific/dataflow primitive remain unchanged.
    """
    receipt=c.drawio(path)
    glyphs=[e for e in c.elements if e.get('role')=='bounded_activation_glyph']
    assert len(glyphs)==1
    e=glyphs[0];pts=e['pts'];xs,ys=zip(*pts);x,y=min(xs),min(ys);w,h=max(xs)-x,max(ys)-y
    shape=ET.Element('shape',{'name':e['id'],'w':str(w),'h':str(h),'aspect':'variable','strokewidth':'inherit'})
    foreground=ET.SubElement(shape,'foreground');stroke=ET.SubElement(foreground,'path')
    for i,(xx,yy) in enumerate(pts):ET.SubElement(stroke,'move' if i==0 else 'line',{'x':str(xx-x),'y':str(yy-y)})
    ET.SubElement(foreground,'stroke')
    compressor=zlib.compressobj(wbits=-15);raw=ET.tostring(shape,encoding='utf-8')
    encoded=base64.b64encode(compressor.compress(raw)+compressor.flush()).decode()
    tree=ET.parse(path);cell=tree.find(f'.//mxCell[@id="{e["id"]}"]');assert cell is not None
    cell.clear();cell.attrib.update({'id':e['id'],'value':'','style':f'shape=stencil({encoded});fillColor=none;strokeColor={e["color"]};strokeWidth={e["lw"]};opacity={100*e["alpha"]};','vertex':'1','parent':'1'})
    ET.SubElement(cell,'mxGeometry',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'as':'geometry'})
    tree.write(path,encoding='utf-8',xml_declaration=True)
    recovered=ET.fromstring(zlib.decompress(base64.b64decode(encoded),wbits=-15))
    q=[(float(n.attrib['x'])+x,float(n.attrib['y'])+y) for n in recovered.findall('./foreground/path/*')]
    np.testing.assert_allclose(q,pts,rtol=0,atol=1e-12)
    receipt['activation_glyph']={'id':e['id'],'representation':'editable open-path native stencil','coordinates':len(pts),'canonical_coordinates_reconstructed_exactly':True,'no_primary_pdf_geometry_change':True}
    return receipt


def draft_a(manifest,rows,gain):
    """Two source policies as one-step loops, with a fixed mixing workbench."""
    c=B.Canvas('source-policy-workbench');gallery(c,manifest)
    c.text(5,80,'DROID · what supplies the next forecast?',8.5,INK)
    c.text(7,96,'Autoregression',8.5,WARM)
    c.text(211,96,'ShiftWM (ours)',8.5,TEAL)
    c.line([(198,98),(198,188)],'#DCE4E8',.55)
    sheet(c,13,120,30,WARM);sheet(c,146,120,30,WARM)
    c.text(28,112,'Previous',8,WARM,'center');c.text(161,112,'Next',8,WARM,'center')
    c.circle(94,135,16,'#FFF6F2',WARM,.8);c.text(94,135,'Predict',8,WARM,'center')
    arc(c,[(47,135),(76,135)],WARM,1,meaning='Previous forecast is autoregressive decoder input')
    arc(c,[(112,135),(142,135)],WARM,1,meaning='Autoregressive decoder emits next forecast')
    arc(c,[(161,154),(161,169),(28,169),(28,154)],WARM,1,meaning='Next forecast becomes the source for the next recursive step')
    c.text(94,182,'Reuse prediction',8,WARM,'center')
    sheet(c,213,122,32,BLUE);sheet(c,351,122,32,TEAL)
    c.text(229,111,'Observed',8,BLUE,'center');c.text(367,111,'Next',8,TEAL,'center')
    # Internal weighted-source strands expose the workbench operation.
    for j in range(3):
        yy=127+10*j
        arc(c,[(247,yy),(270,yy),(282,138)],BLUE,.75,meaning='Illustrative weighted observed-feature contributions')
    c.circle(293,138,9,'#E7F5F3',TEAL,.8);c.text(293,138,'Σ',9,TEAL,'center')
    arc(c,[(304,138),(318,138)],TEAL,.9)
    c.circle(327,138,7,'white',TEAL,.8);c.text(327,138,'+',9,TEAL,'center')
    arc(c,[(336,138),(347,138)],TEAL,.9,meaning='Fixed-source mixing plus bounded correction emits forecast')
    c.rect(316,164,22,7,B.lighten(TEAL,.68),TEAL,lw=.6)
    for x in (316,338):c.line([(x,161),(x,174)],TEAL,.6)
    arc(c,[(327,162),(327,147)],TEAL,.8,meaning='Bounded correction enters the output sum')
    c.text(211,184,'Fixed source',8,TEAL)
    c.text(327,182,'±1 correction',8,TEAL,'center')
    evidence(c,rows,gain);return c


def draft_b(manifest,rows,gain):
    """A relay updates its source; a source rail remains fixed for all endpoints."""
    c=B.Canvas('source-relay-rail');gallery(c,manifest)
    c.text(5,80,'DROID · changing source versus a persistent reference',8.5,INK)
    c.text(5,110,'Autoregression',8.5,WARM)
    c.text(5,157,'ShiftWM (ours)',8.5,TEAL)
    # Endpoint plates, not stacked feature cubes.
    for x,label in [(115,'Observed'),(204,'Step 1'),(281,'Step 2'),(359,'Step 3')]:
        col=BLUE if label=='Observed' else WARM
        sheet(c,x-12,96,24,col);c.text(x,128,label,8,MUTED,'center')
    for x1,x2 in ((132,188),(220,265),(297,343)):
        arc(c,[(x1,108),(x2,108)],WARM,1.2,meaning='A forecast replaces the previous source in autoregression')
    c.text(148,141,'Reuse the same observation',8,BLUE)
    c.rect(111,152,260,5,B.lighten(BLUE,.70),BLUE,lw=.6)
    for x in (204,281,359):
        arc(c,[(x,159),(x,169)],BLUE,.9,meaning='Each endpoint independently reads the same fixed observed source')
        c.rect(x-20,171,40,13,B.lighten(TEAL,.76),TEAL,lw=.7,r=1)
        c.text(x,177.5,'Mix + Δ',8,TEAL,'center')
    c.line([(115,149),(115,140)],BLUE,.8)
    evidence(c,rows,gain);return c


def draft_c(manifest,rows,gain):
    """One persistent palette, one mixing workbench, and an alternative AR loop."""
    c=B.Canvas('persistent-palette-workbench');gallery(c,manifest)
    c.text(5,80,'DROID · ShiftWM (ours)',8.5,TEAL)
    c.text(222,80,'Fixed reference for every horizon',8,TEAL)
    c.rect(5,95,386,72,'#F5FAFB',r=2)
    # One broad fixed source and a single destination, rather than repeated stacks.
    sheet(c,48,111,42,BLUE);c.text(69,102,'Observed features',8,BLUE,'center')
    c.illustration(ROOT/'paper/figure_sources/visual_story_references_v1/camera_illustration_cutout.png',8,113,28)
    sheet(c,340,112,39,TEAL);c.text(359.5,102,'Forecast',8,TEAL,'center')
    for i,(yy,col) in enumerate(((119,BLUE),(132,TEAL),(146,GOLD))):
        c.circle(95,yy,2,col)
        # Weights are concept-only; the ribbon means a weighted source contribution.
        c.poly([(98,yy-2),(171,yy-2),(207,130),(207,136),(171,yy+2),(98,yy+2)],B.lighten(col,.30),alpha=.55)
        arc(c,[(101,yy),(169,yy),(205,133)],col,.7,meaning='Illustrative learned weighted contribution from fixed observed features')
    c.circle(220,133,10,'white',TEAL,.8);c.text(220,133,'Σ',11,TEAL,'center')
    arc(c,[(232,133),(281,133)],TEAL,1.2,meaning='Row-convex mixing of fixed observed features')
    c.circle(292,133,8,'white',TEAL,.8);c.text(292,133,'+',11,TEAL,'center')
    arc(c,[(302,133),(335,133)],TEAL,1.2,meaning='Observed-feature mixture plus bounded correction gives the output')
    c.rect(275,154,35,5,B.lighten(TEAL,.64),TEAL,lw=.6)
    for x in (275,310):c.line([(x,151),(x,162)],TEAL,.7)
    arc(c,[(292,152),(292,143)],TEAL,.8,meaning='Projected unit-bounded innovation enters the feature output')
    c.text(175,156,'Weighted mix',8,TEAL)
    c.text(286,176,'±1 correction',8,TEAL,'center')
    # Explicitly separate comparison annotation; this path never enters ours.
    c.text(7,184,'AR control:',8,WARM)
    c.text(63,184,'feed prediction into next input history',8,WARM)
    arc(c,[(380,133),(387,133),(387,184),(235,184)],WARM,.95,meaning='Alternative AR decoder rolls a predicted feature into the next three-frame input history; not an input edge to ShiftWM')
    evidence(c,rows,gain);return c


def selected(manifest,rows,gain):
    """Disconnected AR loop and larger, identity-linked fixed-source workbench."""
    c=B.Canvas('forecast-source-workbench');gallery(c,manifest)
    c.text(5,80,'DROID · action-conditioned feature forecasting',8.5,INK)
    c.circle(309,80,2.1,BLUE);c.circle(316,80,2.1,GOLD)
    c.text(323,80,'History + actions',8,MUTED)
    c.text(5,98,'Autoregression',8.5,WARM)
    c.text(151,98,'ShiftWM (ours)',8.5,TEAL)
    # Separate baseline graph: observed initialization, own predictor and output.
    c.text(32,116,'Observed start',8,BLUE,'center')
    c.circle(30,124,1.8,BLUE)
    arc(c,[(30,127),(30,132)],BLUE,.8,meaning='Observed three-frame support initializes the AR rolling history')
    c.circle(64,124,1.8,GOLD)
    arc(c,[(64,127),(64,132)],GOLD,.8,meaning='AR receives the same causal action-prefix information')
    c.rect(18,134,51,22,'#FFF0E9',WARM,r=2,lw=.8);c.text(43.5,145,'Predict',8,WARM,'center')
    sheet(c,105,131,27,WARM);c.text(118.5,120,'Forecast',8,WARM,'center')
    arc(c,[(72,145),(101,145)],WARM,1.05,meaning='Autoregressive predictor emits its own forecast')
    arc(c,[(119,162),(119,173),(5,173),(5,145),(16,145)],WARM,1.05,meaning='AR feeds its own prediction into the next three-frame input history')
    c.text(65,187,'Update input history',8,WARM,'center')

    # An upright feature palette is retained; three illustrative locations are
    # expanded with exactly matching color identities, never RGB motion claims.
    x,y,w=155,125,38;cell=w/4;violet='#8C6BB1'
    colors=(BLUE,TEAL,violet)
    c.rect(x-2,y-2,w+4,w+4,'white',B.lighten(BLUE,.4),r=1,lw=.65)
    for i in range(4):
        for j in range(4):
            col=colors[i] if j==3 and i<3 else B.lighten(BLUE,.57+.27*((i+j)%3)/2)
            c.rect(x+j*cell,y+i*cell,cell-.55,cell-.55,col)
    c.text(174,179,'Fixed observed\nreference',8,BLUE,'center')
    c.illustration(ROOT/'paper/figure_sources/visual_story_references_v1/camera_illustration_cutout.png',133,135,20)
    for i,col in enumerate(colors):
        yy=y+(i+.5)*cell
        # The chip is the same source-location feature represented by the colored
        # palette cell. Line thickness is illustrative, not a measured weight.
        arc(c,[(195,yy),(204,yy)],col,.8,meaning=f'Illustrative source location {i+1} retains its identity in the weighted mixture')
        c.rect(206,yy-4,8,8,col,B.lighten(col,.18),r=.6,lw=.5)
        pts=[(217,yy),(229,yy),(242,144)]
        c.line(pts,B.lighten(col,.60),3.3,False,alpha=.65)
        arc(c,pts,col,.75,meaning='Fixed observed-feature contribution enters the action-conditioned convex mixture')
    # Mix folds the learned gate into W=(I-D)+D*T, rather than omitting its
    # retained-source complement. Full Q/K/g details stay in Figure2.
    c.poly([(245,127),(267,136),(267,152),(245,161)],'#DCF2ED',TEAL,.8)
    c.text(255,144,'Mix',8,TEAL,'center')
    c.circle(247,115,1.8,BLUE);c.circle(257,115,1.8,GOLD)
    arc(c,[(247,118),(247,126)],BLUE,.8,meaning='Alias: fixed observed support conditions H_h and therefore mixing; not a direct raw-feature edge')
    arc(c,[(257,118),(257,130)],GOLD,.8,meaning='Causal action-conditioned state determines mixing weights and gate')
    arc(c,[(270,144),(281,144)],TEAL,1.0,meaning='W=(I-D)+D*T multiplies the fixed observed reference')
    c.circle(290,144,7,'white',TEAL,.8);c.text(290,144,'+',10,TEAL,'center')
    arc(c,[(300,144),(338,144)],TEAL,1.05,meaning='Fixed observed mixture plus independently projected bounded correction emits forecast')
    sheet(c,342,128,32,TEAL);c.text(358,116,'Forecast',8,TEAL,'center')
    # A small update patch contains a saturating activation, not an error bar.
    c.rect(279,166,22,15,B.lighten(TEAL,.88),TEAL,r=1,lw=.7)
    curve=[(282+16*t,173.5-5*np.tanh(4*(t-.5))) for t in np.linspace(0,1,24)]
    c.line(curve,TEAL,.8)
    c.elements[-1]['role']='bounded_activation_glyph'
    arc(c,[(290,164),(290,153)],TEAL,.8,meaning='LayerNorm/affine projected unit-bounded tanh feature correction enters the sum')
    c.circle(269,173,1.8,BLUE)
    arc(c,[(272,173),(277,173)],BLUE,.75,meaning='Alias: observed support conditions the projected correction through H_h; not a direct raw-feature input')
    c.circle(312,173,1.8,GOLD)
    arc(c,[(309,173),(303,173)],GOLD,.75,meaning='Causal state also determines the independent projected correction')
    c.text(319,178,'Bounded Δ',8,TEAL)
    evidence(c,rows,gain);return c


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--drafts',action='store_true',help='Also reproduce the three rejected/rough alternatives under paper/design only')
    args=parser.parse_args()
    DESIGN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'shiftwm-sources-v1','image.composite_image':False})
    population,manifest,rows,gain,arithmetic=B.checked_evidence()
    inventory=json.loads((ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json').read_text())
    qas={}
    if args.drafts:
        for name,fun in [('A',draft_a),('B',draft_b),('C',draft_c)]:
            c=fun(manifest,rows,gain)
            assert len(c.photos)==9 and {r['sha256'] for r in c.photos}=={r['asset_sha256'] for r in inventory['records']}
            qas[name]=c.render(DESIGN/f'composition_{name}')
            (DESIGN/f'geometry_{name}.json').write_text(json.dumps({'elements':c.elements,'edges':c.edges,'photos':c.photos},indent=2)+'\n')
        thumbs=[Image.open(DESIGN/f'composition_{n}_paper_width.png').convert('RGB') for n in 'ABC']
        contact=Image.new('RGB',(thumbs[0].width*3,thumbs[0].height+30),'white')
        from PIL import ImageDraw
        draw=ImageDraw.Draw(contact)
        for i,(n,im) in enumerate(zip('ABC',thumbs)):
            contact.paste(im,(im.width*i,30));draw.text((im.width*i+10,6),n,fill='black')
        contact.save(DESIGN/'contact_sheet.png')
    c=selected(manifest,rows,gain)
    assert len(c.photos)==9 and {r['sha256'] for r in c.photos}=={r['asset_sha256'] for r in inventory['records']}
    native=export_native(c,OUT/'teaser_sources.drawio')
    qas['selected']=c.render(OUT/'teaser_sources')
    (DESIGN/'geometry_selected.json').write_text(json.dumps({'elements':c.elements,'edges':c.edges,'photos':c.photos},indent=2)+'\n')
    (PUBLIC/'geometry.json').write_text(json.dumps({'size_points':[396,234],'elements':c.elements},indent=2)+'\n')
    (OUT/'teaser_sources_caption.tex').write_text(CAPTION+'\n')
    (OUT/'teaser_sources_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/teaser_sources.pdf}\n\\caption[Retain observations while forecasting.]{'+CAPTION+'}\n\\label{fig:editorial-teaser}\n\\end{figure}\n')
    sources=[Path(__file__),ROOT/'paper/scripts/render_teaser_benchmarks.py',B.SRC/'population.json',B.SRC/'asset_manifest.json',ROOT/'src/shiftwm/real_video_spatial/model.py',ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json',ROOT/'paper/figure_sources/benchmark_gallery/asset_manifest.json',ROOT/'paper/figure_sources/benchmark_gallery/ATTRIBUTION.md',ROOT/'paper/figure_sources/forecast_comparison_reference_v2/composition_reference.png',ROOT/'paper/figure_sources/forecast_comparison_reference_v2/generation_manifest.json',ROOT/'paper/figure_sources/visual_story_references_v1/camera_illustration_cutout.png',ROOT/'paper/figure_sources/visual_story_references_v1/camera_illustration_cutout_manifest.json']
    ledger={'status':'candidate_pending_independent_review','size_inches':[5.5,3.25],'source_sha256':{str(p.relative_to(ROOT)):B.sha(p) for p in sources},'authoritative_source_sha256':population['authoritative_source_sha256'],'arithmetic':arithmetic,'qa':qas['selected'],'all_episode_marks':[{'episode_id':r['episode_id'],'gain_percent':r['gain_percent'],'mark':'circle' if r['gain_percent']>0 else 'cross'} for r in rows],'photos':c.photos,'semantic_edges':c.edges,'native':native,'caption':CAPTION,'scope':'Nine settings; completed DROID development evidence only. AR and ShiftWM are independent graphs. Effective gated mixture plus independently conditioned bounded correction. All palette values/weights are illustrative. No new experiments.','selection':'A independent topology plus C larger palette workbench; B rejected for repeated forecast rows and C original rejected for conflating outputs. New generated reference actually inspected.','outputs_sha256':{ext:B.sha(OUT/f'teaser_sources.{ext}') for ext in ('pdf','svg','png','drawio')}}
    (OUT/'teaser_sources_evidence.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(qas,indent=2))

if __name__=='__main__':main()
