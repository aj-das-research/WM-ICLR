#!/usr/bin/env python3
"""Three-case, source-bound main qualitative evidence. No model inference.

The recorded replay, selection, source helper and scientific files are immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

import render_editorial_qualitative as source

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/editorial'
HELPER=ROOT/'paper/scripts/render_editorial_qualitative.py'
HELPER_SHA='a7ce31f15fce8383f90f83aa236bdcf16f4f45f1cf5aad9c0003c03b4d2e8630'
WIDTH,HEIGHT=5.5,2.8
SLATE='#334155';SECONDARY='#64748B';GRID='#E2E8F0'
BLUE='#4477AA';TEAL='#287C7A';AMBER='#B88746'
MODES=('autoregressive','transport')
LABELS={'autoregressive':'AR','transport':'ShiftWM (ours)'}
COLORS={'autoregressive':BLUE,'transport':TEAL}
CASES=('Best episode','Median episode','Worst episode')
CAPTION=(r'\textbf{Prespecified best, median and worst cases.} '
         r'The episode-gain ranks among 141 DROID development recordings each show their first eligible window. '
         r'AR denotes matched autoregression. '
         r'Full RGB frames are recorded observations ($f10$) and withheld targets ($f60$). '
         r'Maps show three-seed native, training-standardized $h=10$ feature MSE on one unclipped logarithmic scale. '
         r'Curves show mean paired AR-minus-ShiftWM error with common $\pm70\times10^{-3}$ MSE limits; bands show seed minima/maxima, not confidence intervals. '
         r'Teal gains mean lower error; amber shows regressions. Each forecast step consumes five native commands. '
         r'Episode (Ep) gains average windows; the shown median window instead regresses. '
         r'Images: DROID (CC BY 4.0).')


def sha(path):return source.sha(path)


def axis(fig,x,y,w,h):return fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT])


def text(fig,x,y,s,size=8,color=SLATE,ha='left',weight='normal',**kw):
    return fig.text(x/WIDTH,y/HEIGHT,s,fontsize=size,color=color,ha=ha,va='center',weight=weight,**kw)


def setup():
    plt.rcParams.update({'font.family':'Liberation Sans','mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':8,'ytick.labelsize':8,'text.color':SLATE,'axes.labelcolor':SLATE,
        'xtick.color':SECONDARY,'ytick.color':SECONDARY,'pdf.fonttype':42,
        'svg.fonttype':'none','svg.hashsalt':'shiftwm-qualitative-compact-v1',
        'image.composite_image':False,'savefig.facecolor':'white'})


def verified():
    if sha(HELPER)!=HELPER_SHA:raise ValueError('Reviewed source helper changed')
    replay,arrays,maps,checks=source.verified_pack()
    if [c['rank_descending'] for c in replay['cases']]!=[0,70,140]:
        raise ValueError('Original prespecified cases changed')
    inventory=replay['inventory']
    if len(inventory)!=141:raise ValueError('Incomplete original episode inventory')
    for case in replay['cases']:
        item=next(x for x in inventory if x['episode_id']==case['episode_id'])
        np.testing.assert_allclose(100*(item['baseline_mse']-item['transport_mse'])/item['baseline_mse'],case['gain_percent'],rtol=0,atol=1e-11)
        assert case['first_window_start']==item['first_window_start']==0
    curves={}
    for c in replay['cases']:
        prefix=c['prefix']
        for mode in MODES:
            native=np.stack([arrays[f'{prefix}_{mode}_s{s}_native_mse'] for s in range(3)])
            patch=np.stack([arrays[f'{prefix}_{mode}_s{s}_patch_errors'] for s in range(3)])
            np.testing.assert_allclose(native,patch.mean((-1,-2)),rtol=2e-6,atol=2e-7)
            for s in range(3):
                record=next(x for x in c['replay_checks'] if x['mode']==mode and x['seed']==s)
                np.testing.assert_allclose(native[s],record['replayed_window_mse'],rtol=0,atol=0)
                np.testing.assert_allclose(native[s],record['source_window_mse'],rtol=record['rtol'],atol=record['atol'])
            curves[prefix,mode]={'mean':native.mean(0),'min':native.min(0),'max':native.max(0)}
    vmin=min(float(m.min()) for m in maps.values());vmax=max(float(m.max()) for m in maps.values())
    if not 0<vmin<vmax:raise ValueError('Log scale needs measured positive bounds')
    np.testing.assert_allclose([vmin,vmax],[.0057522510178387165,.7868967652320862],rtol=0,atol=0)
    return replay,arrays,maps,checks,curves,LogNorm(vmin=vmin,vmax=vmax,clip=False)


def frame(fig,arrays,prefix,index,x,y,w):
    a=arrays[prefix+'_images'][index]
    if a.shape!=(180,320,3):raise ValueError('Recorded frame geometry changed')
    ax=axis(fig,x,y,w,w*9/16);ax.imshow(a,interpolation='none');ax.axis('off')
    return ax


def error_map(fig,maps,prefix,mode,norm,x,y,w,numeric=True):
    ax=axis(fig,x,y,w,w)
    im=ax.imshow(maps[prefix,mode],cmap='cividis',norm=norm,interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='white',alpha=.3,lw=.35);ax.tick_params(which='minor',length=0)
    for s in ax.spines.values():s.set_linewidth(.4);s.set_color(GRID)
    if numeric:text(fig,x+w/2,y-.092,f'{float(maps[prefix,mode].mean()):.4f}',ha='center',color=COLORS[mode])
    return im


def curve(fig,curves,prefix,x,y,w,h,show_ylabel=False,show_xlabel=False):
    ax=axis(fig,x,y,w,h)
    for mode,ls,marker in zip(MODES,('-','--'),('o','D')):
        d=curves[prefix,mode]
        ax.fill_between(np.arange(1,11),d['min'],d['max'],color=COLORS[mode],alpha=.12,lw=0)
        ax.plot(np.arange(1,11),d['mean'],color=COLORS[mode],ls=ls,marker=marker,markevery=[0,4,9],ms=2.2,lw=.95)
    ax.set(xlim=(.7,10.3),ylim=(0,.34),xticks=[1,10],yticks=[0,.3])
    ax.tick_params(length=2,pad=1.5)
    ax.spines[['top','right']].set_visible(False)
    for name in ('left','bottom'):ax.spines[name].set_color(SECONDARY);ax.spines[name].set_linewidth(.5)
    ax.grid(axis='y',color=GRID,lw=.45)
    if show_ylabel:ax.set_ylabel('MSE',labelpad=1)
    if show_xlabel:ax.set_xlabel('Forecast step',labelpad=1.5)
    return ax


def legend(fig,x,y):
    handles=[Line2D([0],[0],color=COLORS[m],lw=1,ls=ls,marker=mark,ms=2.3)
             for m,ls,mark in zip(MODES,('-','--'),('o','D'))]
    return fig.legend(handles,[LABELS[m] for m in MODES],loc='center left',bbox_to_anchor=(x/WIDTH,y/HEIGHT),
                      frameon=False,ncol=2,fontsize=8,handlelength=1.4,handletextpad=.3,columnspacing=.7,borderaxespad=0)


def bar(fig,im,norm,x,y,w,labelx=None):
    cb=fig.colorbar(im,cax=axis(fig,x,y,w,.065),orientation='horizontal')
    cb.set_ticks([norm.vmin,.1,norm.vmax]);cb.ax.set_xticklabels([f'{norm.vmin:.4f}','0.1',f'{norm.vmax:.3f}'])
    cb.ax.minorticks_off();cb.ax.tick_params(length=2,pad=1.5,labelsize=8);cb.outline.set_linewidth(.4)
    if labelx is not None:text(fig,labelx,y+.032,'Feature MSE · log',ha='right',color=SECONDARY)


def paired_gaps(arrays):
    out={}
    for i in range(3):
        prefix=f'case{i}'
        per_seed=np.stack([arrays[f'{prefix}_autoregressive_s{s}_native_mse']-
                           arrays[f'{prefix}_transport_s{s}_native_mse'] for s in range(3)])
        if float(np.abs(per_seed).max())>.07:raise ValueError('Common gap axis would clip a seed')
        out[prefix]={'mean':per_seed.mean(0),'min':per_seed.min(0),'max':per_seed.max(0),'per_seed':per_seed}
    return out


def gap_curve(fig,gap,x,y,w,h):
    ax=axis(fig,x,y,w,h);steps=np.arange(1,11);v=1000*gap['mean']
    ax.axhline(0,color=SECONDARY,lw=.55,zorder=1)
    ax.fill_between(steps,1000*gap['min'],1000*gap['max'],color=SLATE,alpha=.13,lw=0,zorder=2)
    ax.fill_between(steps,0,v,where=v>=0,color=TEAL,alpha=.32,lw=0,interpolate=True,zorder=2)
    ax.fill_between(steps,0,v,where=v<=0,color=AMBER,alpha=.32,lw=0,interpolate=True,zorder=2)
    ax.plot(steps,v,color=SLATE,lw=.9,marker='o',ms=2,markevery=[0,4,9],zorder=3)
    ax.set(xlim=(.7,10.3),ylim=(-70,70),xticks=[1,10],yticks=[-50,0,50])
    ax.tick_params(length=1.8,pad=1.3,labelsize=8)
    ax.spines[['top','right']].set_visible(False)
    for n in ('left','bottom'):ax.spines[n].set_color(GRID);ax.spines[n].set_linewidth(.4)
    return ax


def rows(replay,arrays,maps,curves,norm):
    """A: cases as rows; one aligned column grammar; shared labels and axes."""
    fig=plt.figure(figsize=(WIDTH,HEIGHT),dpi=180)
    gaps=paired_gaps(arrays)
    for x,label in ((.355,'Case / gain'),(1.30,'Observed f10'),(2.22,'Withheld f60'),
                    (3.03,'AR'),(3.685,'ShiftWM\n(ours)'),(4.875,'AR − ShiftWM\nMSE ×10³ vs step')):
        text(fig,x,2.67,label,size=8.5,ha='center',color=TEAL if label.startswith('ShiftWM') else SLATE)
    for i,case in enumerate(replay['cases']):
        prefix=case['prefix'];y=1.90-i*.72
        text(fig,.035,y+.40,('Best','Median','Worst')[i],size=8.5)
        text(fig,.035,y+.225,f"Ep {case['gain_percent']:+.1f}%",color=TEAL if case['gain_percent']>0 else AMBER)
        text(fig,.035,y+.052,f"Shown {case['first_window_gain_percent']:+.1f}%",color=TEAL if case['first_window_gain_percent']>0 else AMBER)
        frame(fig,arrays,prefix,2,.88,y+.029,.84);frame(fig,arrays,prefix,12,1.80,y+.029,.84)
        im=error_map(fig,maps,prefix,'autoregressive',norm,2.79,y+.025,.48)
        im=error_map(fig,maps,prefix,'transport',norm,3.445,y+.025,.48)
        gap_curve(fig,gaps[prefix],4.34,y+.01,1.08,.49)
        if i<2:fig.add_artist(Line2D([.035/WIDTH,5.42/WIDTH],[(y-.165)/HEIGHT]*2,color=GRID,lw=.45))
    handles=[Patch(facecolor=TEAL,alpha=.4),Patch(facecolor=AMBER,alpha=.4)]
    fig.legend(handles,['Lower error','Higher error'],loc='center left',bbox_to_anchor=(.04/WIDTH,.185/HEIGHT),
               ncol=2,frameon=False,fontsize=8,handlelength=.8,handletextpad=.4,columnspacing=.8,borderaxespad=0)
    bar(fig,im,norm,3.36,.205,1.88,labelx=3.24)
    return fig


def columns(replay,arrays,maps,curves,norm):
    """B: three self-contained case cards with paired images, maps and curves."""
    fig=plt.figure(figsize=(WIDTH,HEIGHT),dpi=180)
    for i,case in enumerate(replay['cases']):
        prefix=case['prefix'];x=.065+i*1.83;cx=x+.805
        text(fig,cx,2.67,CASES[i],size=8.5,ha='center')
        text(fig,cx,2.49,f"Ep {case['gain_percent']:+.1f}% / win {case['first_window_gain_percent']:+.1f}%",ha='center')
        for offset,idx,label in ((0,2,'Observed'),(.85,12,'Withheld')):
            text(fig,x+offset+.38,2.29,label,ha='center')
            frame(fig,arrays,prefix,idx,x+offset,1.80,.76)
        for offset,mode in ((.06,'autoregressive'),(.94,'transport')):
            text(fig,x+offset+.275,1.69,'AR' if mode=='autoregressive' else 'ShiftWM',ha='center',color=COLORS[mode])
            im=error_map(fig,maps,prefix,mode,norm,x+offset,1.055,.55)
        curve(fig,curves,prefix,x+.30,.335,1.29,.48,show_ylabel=i==0)
        text(fig,cx,.13,'Step 1 → 10',ha='center',color=SECONDARY)
        if i<2:fig.add_artist(Line2D([(x+1.715)/WIDTH]*2,[.1/HEIGHT,2.72/HEIGHT],color=GRID,lw=.45))
    # Rejected card composition: the scale gets its own narrow inter-panel band.
    bar(fig,im,norm,2.95,.865,2.23,labelx=2.55)
    return fig


def bands(replay,arrays,maps,curves,norm):
    """C: scenes and error maps grouped in bands, all case curves on one axis."""
    fig=plt.figure(figsize=(WIDTH,HEIGHT),dpi=180)
    for x,label in ((.385,'Case'),(1.155,'Observed'),(2.02,'Withheld'),(2.80,'AR'),(3.37,'ShiftWM')):
        text(fig,x,2.67,label,size=8.5,ha='center')
    for i,case in enumerate(replay['cases']):
        x=.76;y=1.92-i*.70;prefix=case['prefix']
        text(fig,.04,y+.385,('Best','Median','Worst')[i],size=8.5)
        text(fig,.04,y+.21,f"Ep {case['gain_percent']:+.1f}%")
        text(fig,.04,y+.05,f"Win {case['first_window_gain_percent']:+.1f}%")
        frame(fig,arrays,prefix,2,x,y,.79);frame(fig,arrays,prefix,12,x+.87,y,.79)
        for j,mode in enumerate(MODES):im=error_map(fig,maps,prefix,mode,norm,2.59+j*.57,y,.43)
    ax=axis(fig,4.02,.64,1.30,1.74)
    for i,case in enumerate(replay['cases']):
        for mode,ls in zip(MODES,('-','--')):
            d=curves[case['prefix'],mode];color=(BLUE,AMBER,TEAL)[i]
            ax.plot(np.arange(1,11),d['mean'],color=color,ls=ls,lw=.95,label=('Best','Median','Worst')[i] if mode=='autoregressive' else None)
    ax.set(xlim=(1,10),ylim=(0,.32),xticks=[1,10],yticks=[0,.1,.2,.3],xlabel='Step',ylabel='Window MSE')
    ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2,pad=1.5);ax.grid(axis='y',color=GRID,lw=.45)
    ax.legend(loc='upper left',frameon=False,fontsize=8,handlelength=1)
    text(fig,4.67,2.66,'All window curves',size=8.5,ha='center')
    text(fig,4.66,.32,'AR — / ShiftWM --',ha='center')
    bar(fig,im,norm,1.57,.19,1.88,labelx=1.44)
    return fig


def audit(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();issues=[]
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():continue
        b=obj.get_window_extent(renderer)
        if obj.get_fontsize()<8:issues.append('Font below 8pt: '+obj.get_text())
        if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.x1+1 or b.y1>fig.bbox.y1+1:issues.append('Clipped: '+obj.get_text())
    labels=[(t,t.get_window_extent(renderer)) for t in fig.texts]
    for i,(t,b) in enumerate(labels):
        for u,c in labels[i+1:]:
            if b.overlaps(c):issues.append('Overlapping labels: '+t.get_text()+' / '+u.get_text())
    return {'issues':issues,'minimum_effective_font_pt':8,'dimensions_inches':[WIDTH,HEIGHT]}


def skill_audit(fig):
    skill=Path(os.environ.get('PAPER_FIGURE_SKILL_DIR',str(Path.home()/'.codex/skills/paper-figure-creation'))).expanduser()
    path=skill/'scripts/layout_quality.py'
    if not path.exists():return {'status':'not_run','reason':'Optional skill not installed'}
    spec=importlib.util.spec_from_file_location('compact_qualitative_layout',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return {'status':'checked','issues':module.audit_figure(fig,min_font_pt=8,display_width_inches=WIDTH),
            'source_sha256':sha(path)}


def save(fig,stem):
    for ext in ('pdf','svg','png'):
        kw={'dpi':300}
        if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None}
        if ext=='svg':kw['metadata']={'Date':None}
        fig.savefig(OUT/(stem+'.'+ext),**kw)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--drafts-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);setup()
    replay,arrays,maps,checks,curves,norm=verified();audits={}
    with PdfPages(OUT/'qualitative_compact_drafts.pdf') as pdf:
        for i,fn in enumerate((rows,columns,bands),start=1):
            fig=fn(replay,arrays,maps,curves,norm);audits[str(i)]=audit(fig)
            save(fig,f'qualitative_compact_draft_{i}');pdf.savefig(fig);plt.close(fig)
    brief={'selected':1,'width_inches':WIDTH,'height_inches':HEIGHT,
       'minimum_font_pt':8,'headings_pt':8.5,'alternatives':[
        {'id':1,'geometry':'Case rows with shared observed/withheld/error/curve columns','tradeoff':'Dense but aligned comparison; smallest frames, no repeated model labels.'},
        {'id':2,'geometry':'Three case columns, vertically stacked paired images/maps/curves','tradeoff':'Strong case identity; repeats headings and has less room for aligned axis labels.'},
        {'id':3,'geometry':'Matched scene/map matrix plus one shared six-curve axis','tradeoff':'Large common quantitative axis; cross-case curve identity adds cognitive load.'}],
       'evidence':'Unchanged verified replay pack; exact same three ranks and first windows; all six common scale bounds.',
       'scope':'Original DROID validation; no inference, selection, generated pixels or causal attribution.',
       'color_roles':{'AR':BLUE,'ShiftWM':TEAL,'regression_text':AMBER,'text':SLATE,'maps':'cividis sequential, exact shared LogNorm bounds'},
       'case_values':checks,'curve_values':{p+'_'+m:{k:v.tolist() for k,v in d.items()} for (p,m),d in curves.items()},
       'geometry':audits,'norm':{'class':'LogNorm','vmin':norm.vmin,'vmax':norm.vmax,'clip':False},
       'selection_reason':'Aligned row grammar shows all three full-frame cases, exact endpoint values and paired horizon-wise gains on one symmetric zero-reference scale. Columns repeat labels; a shared six-curve axis confuses case and method identity.',
       'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(source.PACK.iterdir()) if p.is_file()},
       'helper_sha256':HELPER_SHA,'renderer_sha256':sha(__file__)}
    (OUT/'qualitative_compact_design.json').write_text(json.dumps(brief,indent=2)+'\n')
    print(json.dumps({'draft_audits':audits,'case_values':checks,'norm':brief['norm']}))
    if args.drafts_only:return
    fig=rows(replay,arrays,maps,curves,norm);qa=audit(fig);qa['skill']=skill_audit(fig)
    if qa['issues'] or qa['skill'].get('issues'):raise ValueError('Selected layout audit: '+repr(qa))
    save(fig,'qualitative_compact');plt.close(fig)
    (OUT/'qualitative_caption.tex').write_text(CAPTION+'\n')
    (OUT/'qualitative_figure.tex').write_text('\\begin{figure}[!htb]\n\\centering\n'
        '\\includegraphics[width=\\linewidth]{generated/editorial/qualitative_compact.pdf}\n'
        '\\caption{'+CAPTION+'}\n\\label{fig:editorial-qualitative}\n\\end{figure}\n')
    proofdir=OUT/'qualitative_compact_proof';proofdir.mkdir(exist_ok=True)
    proof='\\documentclass{article}\n\\usepackage{iclr2027_conference,times,graphicx,amsmath,hyperref}\n\\iclrfinalcopy\n\\begin{document}\n\\input{generated/editorial/qualitative_figure}\n\\end{document}\n'
    (proofdir/'proof.tex').write_text(proof)
    env=dict(os.environ,TEXINPUTS=str(ROOT/'paper')+':'+str(ROOT/'paper/template/official/iclr2027')+':')
    for i in (1,2):
        r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={proofdir}',str(proofdir/'proof.tex')],cwd=ROOT,env=env,capture_output=True,text=True)
        (proofdir/f'compile_{i}.txt').write_text(r.stdout+r.stderr)
        if r.returncode:raise ValueError('Official-width proof compilation failed')
    warnings=[line for line in (proofdir/'proof.log').read_text().splitlines() if any(w in line for w in ('Overfull','Underfull','Float too large','undefined','! LaTeX'))]
    if warnings:raise ValueError('Proof warnings: '+repr(warnings))
    for filename,flags,dpi,inputfile in (('paper_width',[],120,OUT/'qualitative_compact.pdf'),
        ('detail',[],300,OUT/'qualitative_compact.pdf'),('grayscale',['-gray'],120,OUT/'qualitative_compact.pdf'),
        ('official_page',[],120,proofdir/'proof.pdf')):
        subprocess.run(['pdftoppm','-singlefile','-png','-r',str(dpi),*flags,str(inputfile),str(proofdir/filename)],check=True,capture_output=True)
    outputs=[OUT/('qualitative_compact.'+ext) for ext in ('pdf','svg','png')]+[OUT/'qualitative_figure.tex',OUT/'qualitative_caption.tex',proofdir/'proof.pdf']
    evidence={'status':'numeric_geometry_passed_visual_review_pending','created_utc':datetime.now(timezone.utc).isoformat(),
       'renderer_sha256':sha(__file__),'helper_sha256':HELPER_SHA,'replay_sha256':source.EXPECTED_REPLAY,
       'input_files':brief['source_sha256'],'source_replay_receipts':replay['sources'],
       'upstream_receipts_scope':'Raw source hashes are preserved from the original replay; this portable presentation independently rechecks the complete exported pack, all maps, 180 seed/horizon errors, selected inventory identities and gain arithmetic.',
       'selection':'Unchanged prespecified descending episode-gain ranks 0, 70, 140 among 141; exact first eligible window of each.',
       'displayed_cases':checks,'curve_values':brief['curve_values'],'curve_uncertainty':'Pointwise minimum to maximum of the three paired-seed gaps; not confidence intervals',
       'paired_gap_values':{p:{k:v.tolist() for k,v in d.items()} for p,d in paired_gaps(arrays).items()},
       'curve_shared_axes':{'forecast_step':[1,10],'AR_minus_ShiftWM_MSE_times_1000':[-70,70]},
       'map_norm':brief['norm'],'map_colormap':'cividis; common sequential log mapping for all six source maps',
       'map_metric':'h10 native shared-channel-standardized 4x4 feature MSE; channel and seed averaged',
       'images':'Full unchanged decoded recorded RGB, indices 2/12 corresponding to native frames 10/60; no crop, generative edit, interpolation or predicted RGB.',
       'model_observed_indices':[0,5,10],'displayed_observed_index':10,'displayed_withheld_index':60,
       'temporal_contract':'One forecast step consumes five native recorded commands; no physical timing claim.',
       'palette':brief['color_roles'],'font':'Liberation Sans regular; 8pt labels and 8.5pt headings',
       'dimensions_inches':[WIDTH,HEIGHT],'geometry':qa,'proof_warnings':warnings,
       'scope':'Original DROID validation selected qualitative examples, not a prevalence estimate, fresh test, RGB prediction or causal attribution.',
       'outputs':{str(p.relative_to(ROOT)):sha(p) for p in outputs}}
    for p,h in brief['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Source pack changed during rendering: '+p)
    for name in ('qualitative_compact_evidence.json','qualitative_evidence.json'):
        (OUT/name).write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'selected':1,'output':str(OUT/'qualitative_compact.pdf'),'geometry':qa,'proof_warnings':warnings}))


if __name__=='__main__':main()
