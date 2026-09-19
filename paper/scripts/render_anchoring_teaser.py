#!/usr/bin/env python3
"""A conceptual teaser; all quantitative text binds to completed DROID evidence."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.path import Path as MPath
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/generated/editorial';DESIGN=ROOT/'paper/design/anchoring_teaser'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ASSET=ROOT/'paper/figures/assets/world_concept.png';SOURCE=ROOT/'reports/real_video_spatial/finalization.json'
CAPTION=r'''\textbf{Keep a direct path to the observation.} Autoregression feeds predicted features into the next prediction. ShiftWM (ours) instead mixes a fixed observed grid and adds a bounded correction at each horizon. Both paths are conditioned on supplied actions; the detailed conditioning and decoder appear in Figure~\ref{fig:editorial-spatial-method}. The scene is a generated concept illustration and the grids are schematic. The measured callout uses native ten-step endpoint feature error on DROID development episodes, averaged across three seeds; it is not a planning-success result.'''
def main():
    OUT.mkdir(parents=True,exist_ok=True);DESIGN.mkdir(parents=True,exist_ok=True)
    assert sha(ASSET)=='40e9c67d187cf3f916e4a3923695baa11a6eeb2c14904b57733fe89712251a94'
    j=json.loads(SOURCE.read_text());assert j['status']=='passed'
    e=next(v for v in j['paired_effects'] if v['comparator']=='autoregressive' and v['metric']=='native_mse' and v['horizon']==10)
    gain=e['relative_error_reduction_percent'];assert not e['interval_includes_zero']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    # Three structural studies: stacked paths, radial anchor fanout, and observation + aligned paths.
    fig,axs=plt.subplots(1,3,figsize=(10,3))
    plans=[('A: stacked paths',[(.1,.7,.8,.16),(.1,.3,.8,.16)]),('B: radial anchor',[(.05,.4,.25,.2),(.65,.7,.3,.15),(.65,.4,.3,.15),(.65,.1,.3,.15)]),('C: scene + aligned paths',[(.02,.2,.32,.6),(.4,.65,.58,.16),(.4,.25,.58,.16)])]
    for ax,(title,boxes) in zip(axs,plans):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,fontsize=10)
        for x,y,w,h in boxes:ax.add_patch(Rectangle((x,y),w,h,fc='#eef4f6',ec='#536273'))
    fig.savefig(DESIGN/'layout_candidates.png',dpi=150);plt.close(fig)
    W,H=396,187.2;fig=plt.figure(figsize=(5.5,2.6),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off');texts=[];edges=[]
    ink='#243447';blue='#0072B2';green='#166534';gray='#536273'
    def text(x,y,s,size=8.5,color=ink,weight='normal',ha='center'):
        t=ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',zorder=8);texts.append(t)
    def arrow(points,color):
        ax.add_patch(FancyArrowPatch(path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=7,color=color,lw=.9,zorder=3));edges.append(points)
    def grid(x,y,color):
        for i in range(4):
            for k in range(4):ax.add_patch(Rectangle((x-10+i*5,y-10+k*5),5,5,fc='white',ec=color,lw=.55,zorder=4))
    text(72,164,'Visual observation',9.5,blue,'bold');text(72,149,'Concept illustration',8,gray)
    ia=fig.add_axes([3/W,48/H,138/W,92/H]);ia.imshow(Image.open(ASSET));ia.axis('off')
    text(72,40,'Development result',8.5,blue)
    ax.plot([146,146],[31,178],color='#d8e1e7',lw=.7)
    text(275,174,'Autoregression',9.5,gray,'bold')
    xs=[171,236,301,366]
    for x in xs:grid(x,140,blue if x==171 else gray)
    for start,end in zip(xs[:-1],xs[1:]):arrow([(start+11,140),(end-11,140)],gray)
    for x,label in zip(xs,[r'$Z_0$',r'$\widehat Z_1$',r'$\widehat Z_2$',r'$\widehat Z_3$']):text(x,120,label,8.5)
    text(273,105,'Predicted features become the next input',8,gray)
    text(275,88,'ShiftWM (ours)',9.5,green,'bold')
    for x in xs:grid(x,51,blue if x==171 else green)
    # One observed source, with explicit branches into each horizon's anchored forecast.
    ax.plot([171,171,366],[62,70,70],color=blue,lw=.9,zorder=3)
    for x in xs[1:]:arrow([(x,70),(x,62)],blue)
    for x,label in zip(xs,[r'$Z_0$',r'$\widehat Z_1$',r'$\widehat Z_2$',r'$\widehat Z_3$']):text(x,31,label,8.5)
    text(267,17,'Mix observed features + bounded correction',8,green)
    # Quantitative footer is a source-bound endpoint, not invented visual evidence.
    # The bottom band carries only the comparator and source scope.
    # Put the compact measured outcome alongside the observation rather than another chart.
    text(72,24,f'{gain:.2f}% lower error',9,green,'bold')
    text(72,11,'DROID · h10 vs AR',8,gray)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[t.get_window_extent(renderer) for t in texts]
    overlaps=[(texts[i].get_text(),texts[k].get_text()) for i,a in enumerate(bounds) for k,b in enumerate(bounds) if k>i and a.overlaps(b)]
    clipped=[t.get_text() for t,b in zip(texts,bounds) if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height]
    assert not overlaps and not clipped,(overlaps,clipped)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'anchoring_teaser.{ext}',dpi=300)
    fig.savefig(DESIGN/'paper_width.png',dpi=120);fig.savefig(DESIGN/'enlarged.png',dpi=360);plt.close(fig)
    ImageOps.grayscale(Image.open(DESIGN/'paper_width.png')).save(DESIGN/'grayscale.png')
    (OUT/'anchoring_teaser_caption.tex').write_text(CAPTION+'\n')
    (OUT/'anchoring_teaser_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/editorial/anchoring_teaser.pdf}\n\\caption[Observation-anchored forecasting.]{'+CAPTION+'}\n\\label{fig:editorial-teaser}\n\\end{figure}\n')
    record={'status':'awaiting_pixel_review','renderer_sha256':sha(__file__),'source_sha256':sha(SOURCE),'asset_sha256':sha(ASSET),'source_effect':e,'geometry':{'inches':[5.5,2.6],'minimum_font_pt':8,'text_overlaps':overlaps,'clipped':clipped},'edges':edges,'selected_layout':'C: scene plus aligned paths; directly compares evidence routes without duplicating the complete decoder','illustration_boundary':'Saved generated scene only; no pixel transformations. Exact grids/arrows/labels are vector authored. Scene and grids are conceptual, not model output.','scope':'DROID validation native endpoint h10 versus matched autoregression; not simulation/planning/generalization evidence.'}
    (OUT/'anchoring_teaser_evidence.json').write_text(json.dumps(record,indent=2)+'\n')
    (DESIGN/'brief.md').write_text('# Observation anchoring teaser\n\nOne claim: each future prediction retains a direct path to a measured reference.\nThe three sketches compare stacked paths, radial fanout and aligned paths next to a concept scene. C preserves the earlier pictorial style and is easiest to compare at 5.5-inch width.\nThe generated scene is an illustration only; no claim about a pushing success is made.\n')
    print(json.dumps({'gain':gain,'overlaps':overlaps,'clipped':clipped}))
if __name__=='__main__':main()
