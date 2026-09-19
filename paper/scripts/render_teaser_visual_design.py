#!/usr/bin/env python3
"""Personal visual-design skill: exact second-step feature-history comparison.

New figure namespace only. All experimental imagery and outcome ledgers are
reused unchanged. Feature-state glyphs show input identities, not feature values.
"""
from pathlib import Path
import importlib.util,json
import numpy as np
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('teaser_source_helpers',ROOT/'paper/scripts/render_teaser_sources.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
assert S.B.sha(ROOT/'paper/scripts/render_teaser_sources.py')=='b822564b606abde10148795aae834442965b8f20444e51bc3e4270cb44bd0656'
B=S.B
OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/teaser_visual_design';PUBLIC=ROOT/'paper/figure_sources/teaser_visual_design'
INK='#1C304D';BLUE='#2378B7';TEAL='#008E88';CORAL='#D96951';GOLD='#CF942D';MUTED='#64748B'
CAPTION=(r'\textbf{Keep the observed feature history.} '
 r'The $h=2$ illustration contrasts autoregression, using $(Z_{-1},Z_0,\hat Z_1)$, with ShiftWM, retaining $(Z_{-2},Z_{-1},Z_0)$ and the fixed $Z_0$ decoder source. '
 r'Autoregression inserts its own $\hat Z_2$ into the next window $(Z_0,\hat Z_1,\hat Z_2)$; its originally inferred support context remains fixed. '
 r'Both receive the same available causal commands. Solid/dashed cards distinguish observed/predicted features; textures and outlined coordinates are schematic, not RGB predictions or measured feature values. '
 r'Figure~\ref{fig:editorial-spatial-method} details the gated mixture and bounded correction. '
 r'Gallery: historical simulations use distinct context models; IWS training is ongoing; Open-H is a physical-phantom input audit only. '
 r'The measured footer retains all 141 DROID development episodes: 5.30\% lower population mean native, training-standardized $h=10$ feature MSE than matched autoregression (uncertainty in Figure~\ref{fig:editorial-spatial}). '
 r'Images: DROID and Open-H (CC BY 4.0); IWS \citep{zhang2026rlawm}. Source-selection and simulator attribution accompany the editable figure.')

def gallery(c,m):
    # Camera-back/viewfinder surrounds the actual DROID RGB without covering it.
    c.rect(326,16,68,40,'#E4EEF5',BLUE,r=2,lw=.8)
    c.rect(333,13,11,3,BLUE,r=.6)
    S.gallery(c,m)
    # Remove the old sharp divider family; scope labels and whitespace group.
    c.elements=[e for e in c.elements if not(e['kind']=='line' and e.get('color') in ('#DCE4E8','#C9D8E1'))]
    for e in c.elements:
        if e['kind']=='text' and e['text']=='Development':e['y']=63
    # IDs are relabeled because helper elements were removed.
    ids={}
    for i,e in enumerate(c.elements):
        ids[e['id']]=f'{c.name}-{i:04d}';e['id']=ids[e['id']]
    for p in c.photos:p['id']=ids[p['id']]

def state(c,x,y,name,w=25,kind='observed'):
    col=BLUE if kind=='observed' else TEAL
    # A shallow card exposes object identity/depth without plotting a quantity.
    c.rect(x+2,y+2,w,w,'#D6E0E7',alpha=.35,r=.7)
    c.poly([(x+w,y),(x+w+2,y+2),(x+w+2,y+w+2),(x+w,y+w)],B.lighten(col,.63),col,.35)
    c.poly([(x,y+w),(x+2,y+w+2),(x+w+2,y+w+2),(x+w,y+w)],B.lighten(col,.81),col,.35)
    c.rect(x,y,w,w,'white',col,lw=.8,r=.7)
    for i in range(4):
        for j in range(4):
            shade=.20+.58*((i*3+j*2+{'Z−2':0,'Z−1':2,'Z0':4}.get(name,1))%7)/7
            c.rect(x+1+j*(w-2)/4,y+1+i*(w-2)/4,(w-2)/4-.35,(w-2)/4-.35,B.lighten(col,shade))
    # This consistent outlined location is a feature-grid identity, not physical
    # correspondence. The same observed state has the same schematic pattern.
    cell=(w-2)/4;c.rect(x+1+cell,y+1+cell,cell-.35,cell-.35,'none',INK,lw=.65)
    if kind=='predicted':
        c.line([(x-2,y-2),(x+w+2,y-2),(x+w+2,y+w+2),(x-2,y+w+2),(x-2,y-2)],TEAL,.65,dashed=True)
    c.text(x+w/2,y-10,name,8,col,'center')

def history(c,x,y,names,w=25,step=33,fixed=False):
    width=2*step+w
    c.rect(x-6,y-19,width+12,w+28,'#ECF4F9' if fixed else '#F4F6F8','#AAC3D4' if fixed else '#CDD7DE',r=2,lw=.6)
    # Filmstrip rails provide a recognizable bounded three-state memory window.
    for yy,hh in ((y-18,2),(y+w+5,3)):
        c.rect(x-4,yy,width+8,hh,'#C7DAE8' if fixed else '#DAE1E6',r=.7)
        for xx in np.arange(x-1,x+width,9):c.rect(float(xx),yy+.5,4,hh-1,'white',r=.4)
    for i,name in enumerate(names):state(c,x+i*step,y,name,w,'predicted' if '̂' in name else 'observed')
    return width

def port(c,x,y,color,end,meaning):
    c.circle(x,y,1.8,color);c.line([(x,y+3),end],color,.75,True,semantic=meaning)

def footer(c,rows,gain):
    S.evidence(c,rows,gain)
    for e in c.elements:
        if e['kind']=='text' and e['text']=='lower h10 error':e['text']='lower error vs AR'
        if e['kind']=='text' and e['text']=='DROID development · vs AR':e['text']='Measured DROID development · h=10';e['x']=234;e['size']=8

def composition_a(m,rows,gain):
    c=B.Canvas('history-window-lanes');gallery(c,m)
    c.text(5,78,'Mechanism illustration · h=2',8.5,INK)
    c.text(261,79,'Next step: roll history',8,MUTED)
    c.text(6,117,'Autoregression',8.5,MUTED)
    c.text(6,130,'Rolling history',8,MUTED)
    c.text(6,165,'ShiftWM (ours)',8.5,TEAL)
    c.text(6,178,'Fixed observations',8,TEAL)
    history(c,111,110,['Z−1','Z0','Ẑ1'],w=24,step=32)
    history(c,111,158,['Z−2','Z−1','Z0'],w=24,step=32,fixed=True)
    for y in (122,170):
        c.line([(207,y),(249,y)],BLUE,.85,True,semantic='Complete three-state feature history enters the predictor')
        c.rect(252,y-11,47,22,'white',MUTED if y==122 else TEAL,r=2,lw=.8)
        if y==122:c.text(275.5,y,'Predict',8,MUTED,'center')
        else:
            c.text(266,y,'Mix',8,TEAL,'center');c.text(278,y,'+',8,TEAL,'center');c.text(290,y,'Δ',8,CORAL,'center')
        c.line([(302,y),(343,y)],MUTED if y==122 else TEAL,.9,True,semantic='Each method produces its own second-step feature forecast')
        state(c,347,y-12,'Ẑ2',24,'predicted')
        port(c,276,y-21,GOLD,(276,y-13),'Same available causal commands condition each method')
    c.line([(376,122),(383,122),(383,86),(107,86),(107,91)],MUTED,.8,True,semantic='AR own output rolls into the next entire history window [Z0,Zhat1,Zhat2]; arrow lands on container boundary')
    c.text(283,101,'Commands',8,GOLD)
    # Only the fixed last observation has a separate decoder source path.
    c.line([(187,184),(187,194),(276,194),(276,183)],BLUE,.85,True,semantic='Fixed last-observed Z0 remains the source for the gated spatial decoder')
    footer(c,rows,gain);return c

def composition_b(m,rows,gain):
    c=B.Canvas('history-snapshot-pair');gallery(c,m)
    c.text(5,79,'Autoregression',8.5,MUTED)
    c.text(167,79,'ShiftWM (ours)',8.5,TEAL)
    c.text(5,94,'Feature history at step 2',8,MUTED)
    c.text(167,94,'Original observed support',8,BLUE)
    history(c,10,118,['Z−1','Z0','Ẑ1'],w=25,step=34)
    history(c,170,118,['Z−2','Z−1','Z0'],w=25,step=34,fixed=True)
    # Entire history goes into each independent predictor, not only the last card.
    c.line([(52,149),(52,160),(87,160)],MUTED,.9,True,semantic='AR rolling history predicts its next feature state')
    c.rect(90,151,29,18,'white',MUTED,r=1,lw=.7);c.text(104.5,160,'F',9,MUTED,'center')
    c.line([(123,160),(134,160),(134,148),(95,148)],MUTED,.8,True,semantic='AR own prediction is inserted into the next rolling history')
    c.text(7,182,'Observation → prediction',8,MUTED)
    c.line([(210,149),(210,162),(281,162)],BLUE,.85,True,semantic='Unchanged observed support conditions the forecast state')
    c.rect(284,151,41,22,'#F0FAF6',TEAL,r=1,lw=.8);c.text(304.5,162,'Decode',8,TEAL,'center')
    c.line([(327,162),(348,162)],TEAL,.9,True)
    state(c,351,151,'Ẑ2',26,'predicted')
    c.line([(250,149),(250,182),(304,182),(304,176)],BLUE,.9,True,semantic='Same Z0 supplies every decoder query')
    port(c,304,137,GOLD,(304,148),'Causal action prefix conditions the fixed-support decoder')
    port(c,105,136,GOLD,(105,148),'Same causal action prefix conditions the autoregressive predictor')
    c.text(174,191,'Retain the observed reference',8,TEAL)
    footer(c,rows,gain);return c

def composition_c(m,rows,gain):
    c=B.Canvas('history-on-timeline');gallery(c,m)
    c.text(5,78,'A second-step query sees different feature histories',8.5,INK)
    c.text(6,111,'Autoregression',8.5,MUTED);c.text(6,163,'ShiftWM (ours)',8.5,TEAL)
    positions=[115,165,215,265,352]
    names=['Z−2','Z−1','Z0','Ẑ1','Ẑ2']
    c.rect(158,92,138,44,'#F7F8FA','#C5CFD7',r=2,lw=.7)
    c.rect(108,144,138,44,'#F0F8FC','#AACBDC',r=2,lw=.7)
    for y in (105,157):
        for x,name in zip(positions,names):
            if y==157 and name=='Ẑ1':continue
            state(c,x,y,name,24,'predicted' if '̂' in name else 'observed')
    c.line([(299,118),(346,118)],MUTED,.9,True,semantic='AR rolling three-frame window predicts its second output')
    c.line([(249,170),(346,170)],TEAL,.9,True,semantic='Original observed window and fixed Z0 predict the second query')
    c.text(300,91,'Query 2',8,GOLD)
    c.circle(323,105,1.8,GOLD);c.circle(323,157,1.8,GOLD)
    footer(c,rows,gain);return c

def main():
    DESIGN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix','svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'shiftwm-visual-design-v1','image.composite_image':False})
    p,m,rows,gain,a=B.checked_evidence();qas={}
    for k,f in [('A',composition_a),('B',composition_b),('C',composition_c)]:
        c=f(m,rows,gain);qas[k]=c.render(DESIGN/f'composition_{k}')
        (DESIGN/f'geometry_{k}.json').write_text(json.dumps({'elements':c.elements,'edges':c.edges,'photos':c.photos},indent=2)+'\n')
    thumbs=[Image.open(DESIGN/f'composition_{k}_paper_width.png').convert('RGB') for k in 'ABC'];sheet=Image.new('RGB',(thumbs[0].width*3,thumbs[0].height+24),'white');draw=ImageDraw.Draw(sheet)
    for i,(k,im) in enumerate(zip('ABC',thumbs)):sheet.paste(im,(i*im.width,24));draw.text((i*im.width+8,5),k,fill='black')
    sheet.save(DESIGN/'contact_sheet.png')
    (DESIGN/'draft_evidence.json').write_text(json.dumps({'status':'three_composition_sketches','arithmetic':a,'qa':qas,'scope':'No new model outputs; schematic feature-state identities at h2; source population unchanged.'},indent=2)+'\n')
    c=composition_a(m,rows,gain)
    inventory=json.loads((ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json').read_text())
    assert len(c.photos)==9 and {p['sha256'] for p in c.photos}=={r['asset_sha256'] for r in inventory['records']}
    # Native first. This figure has no sampled activation-edge glyph; all edges
    # are short piecewise-linear dataflow paths and all card faces are stencils.
    native=c.drawio(OUT/'teaser_visual_design.drawio')
    qa=c.render(OUT/'teaser_visual_design')
    (PUBLIC/'geometry.json').write_text(json.dumps({'size_points':[396,234],'elements':c.elements},indent=2)+'\n')
    (OUT/'teaser_visual_design_caption.tex').write_text(CAPTION+'\n')
    (OUT/'teaser_visual_design_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/teaser_visual_design.pdf}\n\\caption[Keep the observed feature history.]{'+CAPTION+'}\n\\label{fig:editorial-teaser}\n\\end{figure}\n')
    deps=[Path(__file__),ROOT/'paper/scripts/render_teaser_sources.py',ROOT/'paper/scripts/render_teaser_benchmarks.py',ROOT/'src/shiftwm/real_video_spatial/model.py',B.SRC/'population.json',B.SRC/'asset_manifest.json',ROOT/'paper/figure_sources/benchmark_gallery/benchmark_inventory.json',ROOT/'paper/figure_sources/benchmark_gallery/asset_manifest.json',ROOT/'paper/figure_sources/benchmark_gallery/ATTRIBUTION.md']
    ledger={'status':'candidate_pending_independent_review','source_sha256':{str(q.relative_to(ROOT)):B.sha(q) for q in deps},'authoritative_source_sha256':p['authoritative_source_sha256'],'size_inches':[5.5,3.25],'arithmetic':a,'all_episode_marks':[{'episode_id':r['episode_id'],'gain_percent':r['gain_percent'],'mark':'circle' if r['gain_percent']>0 else 'cross'} for r in rows],'photos':c.photos,'semantics':{'snapshot_horizon':2,'autoregressive_features':['observed_-1','observed_0','predicted_1'],'autoregressive_next_features':['observed_0','predicted_1','predicted_2'],'shiftwm_features':['observed_-2','observed_-1','observed_0'],'shiftwm_decoder_source':'fixed observed_0','context':'both use originally inferred observed-support context; only AR feature history rolls','commands':'same available causal commands; internal GRU-state slicing is not claimed identical','glyphs':'categorical state identities and feature-grid coordinate j=5; no measured values, physical tracking or RGB generation'},'semantic_edges':c.edges,'qa':qa,'native':native,'caption':CAPTION,'skills':['paper-visual-design','paper-figure-creation','codex-paper-figure-skill'],'reference':'Previously inspected generated forecast-comparison reference informs shallow card/object hierarchy; new history-window topology derives from actual model code. No generated artwork embedded.','outputs_sha256':{ext:B.sha(OUT/f'teaser_visual_design.{ext}') for ext in ('pdf','svg','png','drawio')}}
    (OUT/'teaser_visual_design_evidence.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(qas,indent=2))

if __name__=='__main__':main()
