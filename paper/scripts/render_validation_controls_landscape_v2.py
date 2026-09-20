#!/usr/bin/env python3
"""Portable historical validation-control plot: exact saved contrasts; no experiment access."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / 'paper/figure_sources/validation_controls_landscape_v2'
HERE = ROOT / 'paper/generated/editorial'
INK, GRAY, BLUE, TEAL, AMBER = '#243447', '#64717d', '#4477aa', '#287c7a', '#a65b29'
LIGHT = '#e4e9ed'
W, H = 396, 144
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'mathtext.fontset':'stix',
                     'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'historical-validation-controls-landscape-v2',
                     'axes.labelcolor':INK,'text.color':INK,'xtick.color':GRAY,'ytick.color':INK})

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(fig,x,y,text,size=8,**kw):
    return fig.text(x/W,1-y/H,text,fontsize=size,va='center',**kw)
def axis(fig,x0,x1,top,bottom,limits,ticks,n):
    ax=fig.add_axes([x0/W,1-bottom/H,(x1-x0)/W,(bottom-top)/H])
    ax.set_xlim(*limits);ax.set_ylim(n-.5,-.5)
    ax.set_yticks([]);ax.set_xticks(ticks);ax.tick_params(axis='x',length=2,pad=3,labelsize=8)
    ax.axvline(0,color=GRAY,lw=.7,ls=(0,(3,2)),zorder=1)
    ax.grid(axis='x',color=LIGHT,lw=.55,zorder=0)
    for s in ['left','right','top']:ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#bac4cc');ax.spines['bottom'].set_linewidth(.6)
    return ax
def dot(ax,row,y,marker='o'):
    d=1000*row['delta'];lo,hi=np.asarray(row['ci95'])*1000
    color=TEAL if d<0 else AMBER
    filled=lo>0 or hi<0
    ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt=marker,color=color,mfc=color if filled else 'white',
                mec=color,ms=3.8,mew=.9,elinewidth=1.1,capsize=2,capthick=.8,zorder=3)
    assert ax.get_xlim()[0]<lo<=d<=hi<ax.get_xlim()[1]
def footer(fig):
    label(fig,198,125,r'Δ standardized MSE ($\times10^{-3}$); left favors the first named model.',ha='center')
    label(fig,198,137,'Secondary: context h5 mean −0.116% gain (CI crosses zero).',ha='center',color=AMBER)
def audit(fig):
    fig.canvas.draw();rr=fig.canvas.get_renderer();bb=fig.bbox
    ts=[];issues=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        if t.axes is not None and not t.axes.get_visible():continue
        b=t.get_window_extent(rr)
        if not b.width or not b.height:continue
        if t.get_fontsize()<8-1e-6:issues.append(['font',t.get_text()])
        if b.x0<bb.x0-.5 or b.y0<bb.y0-.5 or b.x1>bb.x1+.5 or b.y1>bb.y1+.5:issues.append(['clip',t.get_text()])
        ts.append((t.get_text(),b))
    for i,(a,b) in enumerate(ts):
        for c,d in ts[i+1:]:
            if min(b.x1,d.x1)-max(b.x0,d.x0)>1 and min(b.y1,d.y1)-max(b.y0,d.y0)>1:
                issues.append(['text_overlap',a,c])
    return {'status':'passed' if not issues else 'needs_revision','issues':issues,'minimum_font_pt':8,'size_inches':[5.5,2.0]}
def export(fig,stem):
    quality=audit(fig)
    for ext in ['pdf','svg','png']:
        metadata={'Creator':'ShiftWM staged scientific figure'}
        if ext=='pdf':metadata.update(CreationDate=None,ModDate=None)
        elif ext=='svg':metadata['Date']=None
        else:metadata={'Software':'ShiftWM staged scientific figure'}
        fig.savefig(HERE/f'{stem}.{ext}',dpi=220,facecolor='white',metadata=metadata)
    (HERE/f'{stem}_layout.json').write_text(json.dumps(quality,indent=2)+'\n')
    plt.close(fig);return quality

def composition_a(data):
    fig=plt.figure(figsize=(W/72,H/72),dpi=180)
    label(fig,7,7,'Historical context model (ours) · DROID validation',size=8.7)
    headings=[(7,'a  Optimization / capacity','Context − Framewise'),
              (140,'b  Training horizon','h10-trained − h5-trained'),
              (274,'c  Matched methods','Context − Framewise')]
    for x,title,sub in headings:
        label(fig,x,21,title,size=8.5)
        label(fig,x,33,sub,color=GRAY)
    label(fig,7,44,'Endpoint',color=GRAY)
    label(fig,140,44,'Mean h10; last row h5',color=GRAY)
    label(fig,274,44,'Both h10-trained',color=GRAY)
    # Panel a retains all six effects, paired by arm and differentiated by horizon.
    a=axis(fig,54,128,54,102,(-2.1,6.0),[-2,0,2,4,6],6)
    for i,r in enumerate(data['panels'][0]['rows']):
        y=54+(i+.5)*48/6
        label(fig,7,y,f"{r['label']} {r['horizon']}")
        dot(a,r,i,'o' if r['horizon']==5 else 'D')
    b=axis(fig,200,264,54,102,(-3.05,.8),[-3,-2,-1,0],5)
    for i,r in enumerate(data['panels'][1]['rows']):
        y=54+(i+.5)*48/5
        label(fig,140,y,r['label'],color=AMBER if r.get('secondary') else INK)
        dot(b,r,i,'s' if r.get('secondary') else 'o')
    b.axhline(3.5,color=LIGHT,lw=.6)
    c=axis(fig,325,390,54,102,(-1.5,.15),[-1.5,-1,-.5,0],4)
    for i,r in enumerate(data['panels'][2]['rows']):
        label(fig,274,54+(i+.5)*48/4,r['label'])
        dot(c,r,i,'o' if 'mean' in r['label'] else 'D')
    footer(fig)
    return export(fig,'validation_controls_landscape_v2')

def main():
    manifest=json.loads((PACK/'manifest.json').read_text())
    for name,digest in manifest['runtime_inputs_sha256'].items():
        assert sha(PACK/name)==digest, name
    data=json.loads((PACK/'data.json').read_text())
    assert [len(p['rows']) for p in data['panels']]==[6,5,4]
    assert len(data['historical_horizon_ledger']['all_20_contrasts'])==20
    originals={r['id']:r for r in data['historical_horizon_ledger']['all_20_contrasts']}
    for panel in data['panels']:
        for row in panel['rows']:
            assert np.isclose(row['delta'],row['first_mean']-row['second_mean'],rtol=1e-11,atol=1e-13)
            assert np.isclose(row['gain_percent'],100*(1-row['first_mean']/row['second_mean']),atol=1e-10)
            if panel['id'] in ['b','c']:
                source=originals[row['id']]
                assert row['ci95']==source['paired_first_minus_second']['ci95']
                assert row['first_mean']==source['first_mean'] and row['second_mean']==source['second_mean']
    HERE.mkdir(parents=True,exist_ok=True)
    quality=composition_a(data)
    assert quality['status']=='passed',quality
    img=Image.open(HERE/'validation_controls_landscape_v2.png').convert('RGB')
    img.save(HERE/'validation_controls_landscape_v2_paper_width.png')
    img.convert('L').convert('RGB').save(HERE/'validation_controls_landscape_v2_grayscale.png')
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),PACK/'data.json',PACK/'caption.tex',PACK/'manifest.json']}
    evidence={'status':'passed','scope':data['scope'],'sources_sha256':sources,'layout':quality,
              'displayed_rows':data['panels'],'preserved_horizon_contrasts':20,
              'prior_readonly_extraction_verification':data['verification'],
              'axis_transform':'1000 * signed first-minus-second standardized MSE; no percent-CI transform',
              'no_new_model_compute':True,'historical_source_paths_are_provenance_only':True,
              'exports_sha256':{ext:sha(HERE/('validation_controls_landscape_v2.'+ext)) for ext in ['pdf','svg','png']}}
    (HERE/'validation_controls_landscape_v2.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'passed','displayed_effects':15,'pdf_sha256':evidence['exports_sha256']['pdf']}))

if __name__=='__main__':main()
