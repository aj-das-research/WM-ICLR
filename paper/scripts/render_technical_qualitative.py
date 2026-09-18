#!/usr/bin/env python3
"""Exact replay constraints and paired same-transition prediction diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('positive_scene',Path(__file__).with_name('render_positive_qualitative.py'))
scene=importlib.util.module_from_spec(spec); spec.loader.exec_module(scene)
OUT=ROOT/'paper/generated/qualitative'
BLUE,GRAY,GREEN,RED,INK,MUTED='#0072B2','#6B6F78','#166534','#A14032','#243447','#65758A'
MODES=('factorized','framewise')
LABELS={'factorized':'ShiftWM (ours)','framewise':'Framewise'}
COLORS={'factorized':BLUE,'framewise':GRAY}

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def read_inputs():
    paths=[ROOT/'results/qualitative_diagnostics/replay_diagnostics.json', ROOT/'results/qualitative_diagnostics/prediction_diagnostics.json']
    replay,pred=[json.loads(p.read_text()) for p in paths]
    if replay['status']!='complete' or pred['status']!='complete':
        raise ValueError('Only completed, validated diagnostics may be plotted')
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for packet,key in [(replay,'sources'),(pred,'source_hashes')]:
        for name,digest in packet[key].items():
            if sha(ROOT/name)!=digest:
                raise ValueError(f'Diagnostic source changed: {name}')
            sources[name]=digest
    records={}
    for row in replay['records']:
        if row['validation']['status']!='exact' or not row['validation']['claim_eligible']:
            raise ValueError('Replay has unverified frames or states')
        path=ROOT/row['npz']
        if sha(path)!=row['npz_sha256']:
            raise ValueError('Replayed array hash changed')
        with np.load(path,allow_pickle=False) as values:
            arrays={key:values[key].copy() for key in values.files}
        times=arrays['trace_native_times'];errors=arrays['criterion_errors']
        thresholds=np.array(row['criterion']['thresholds'])
        if not np.array_equal(times,np.arange(len(times))) or errors.shape!=(len(times),2):
            raise ValueError('Unaligned physical trace')
        flags=np.all(errors < thresholds,axis=1)
        if not np.array_equal(flags,arrays['success_flags']):
            raise ValueError('Displayed criteria disagree with simulator success')
        records[(row['environment'],row['seed'],row['behavior_mode'])]=(row,arrays)
        sources[row['npz']]=row['npz_sha256']
    for row in pred['records']:
        if row['target_native_time']!=row['anchor_native_time']+5:
            raise ValueError('Prediction is not one complete action block')
        if row['history_native_times']!=[row['anchor_native_time']-10,row['anchor_native_time']-5,row['anchor_native_time']]:
            raise ValueError('Prediction observes inconsistent history')
        target=np.asarray(row['canonical_next_feature'],dtype=np.float64)
        for mode in MODES:
            predicted=np.asarray(row['models'][mode]['predicted_next_feature'],dtype=np.float64)
            expected=np.mean(np.square(predicted-target))
            if not np.isclose(expected,row['models'][mode]['next_prediction_mse'],rtol=2e-5,atol=1e-10):
                raise ValueError('Reported prediction MSE differs from saved vectors')
    return records,pred['records'],sources


def selected_cases(records):
    cases=sorted({(env,seed) for (env,seed,mode),(row,_) in records.items() if mode=='factorized' and row['category']=='ours_only'})
    if len(cases)!=4:
        raise ValueError('Declared exhaustive four-case selection changed')
    return cases


def style_axis(ax):
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom','left'):
        ax.spines[side].set_color('#BBC5CF');ax.spines[side].set_linewidth(.6)
    ax.tick_params(axis='both',labelsize=6.4,length=2,color='#BBC5CF',pad=2)
    ax.grid(axis='y',color='#E7ECF0',lw=.6,zorder=0)
    ax.set_axisbelow(True)


def shared_crop(arrays):
    """Use one crop for all shown states and goal, preserving coordinates."""
    masks=[]
    for values in arrays:
        masks.append(scene.goal_mask(values['goal_shifted_frame'],'reacher'))
        for time in (10,15,20,int(values['native_times'][-1])):
            index=list(values['native_times']).index(time)
            masks.append(scene.goal_mask(values['shifted_frames'][index],'reacher'))
    yy,xx=np.nonzero(np.logical_or.reduce(masks))
    if not len(xx): raise ValueError('No visible arm')
    side=min(224,max(int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))+24)
    x0=int(np.clip((int(xx.min())+int(xx.max())+1-side)//2,0,224-side))
    y0=int(np.clip((int(yy.min())+int(yy.max())+1-side)//2,0,224-side))
    return x0,y0,x0+side,y0+side


def draw_case(env,seed,records,predictions):
    rows=[records[(env,seed,mode)] for mode in MODES]
    fig=plt.figure(figsize=(5.5,6.7),facecolor='white')
    name='PushT' if env=='pusht' else 'Reacher'
    fig.text(.025,.980,f'{name} · where the successful execution differs',fontsize=10,weight='bold',va='top')
    fig.text(.025,.948,f'Development episode {seed} · matched goal/support · positive case',fontsize=7,color=MUTED,va='top')
    fig.text(.025,.914,'A  Observe the decisions along the executed path',fontsize=8.3,weight='bold',va='top')
    crop=shared_crop([values for _,values in rows]) if env=='reacher' else (0,0,224,224)
    goal=rows[0][1]['goal_shifted_frame'];mask=scene.goal_mask(goal,env)
    for i,(mode,(row,values)) in enumerate(zip(MODES,rows)):
        bottom=.756-i*.15
        fig.text(.025,bottom+.092,LABELS[mode],fontsize=7.6,weight='bold',color=COLORS[mode],va='center')
        success=bool(row['original_record']['success'])
        fig.text(.025,bottom+.058,'Goal reached' if success else 'Goal missed',fontsize=7.0,weight='bold',color=GREEN if success else RED)
        for j,time in enumerate((10,15,20,int(values['native_times'][-1]))):
            ax=fig.add_axes([.255+j*.179,bottom,.155,.127])
            idx=list(values['native_times']).index(time)
            ax.imshow(values['shifted_frames'][idx],interpolation='nearest')
            ax.contour(np.arange(224),np.arange(224),mask.astype(float),levels=[.5],colors=[scene.PURPLE],linestyles='--',linewidths=.65)
            ax.set_xlim(crop[0]-.5,crop[2]-.5);ax.set_ylim(crop[3]-.5,crop[1]-.5)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color((GREEN if success else RED) if j==3 else '#CCD4DD');spine.set_linewidth(1 if j==3 else .5)
            ax.set_title(('End ' if j==3 else '')+f't={time}',fontsize=6.8,pad=3)
    fig.text(.255,.584,'Dashed outline: common goal'+(' · same arm crop in every cell' if env=='reacher' else ' · green T is unscored'),fontsize=6.1,color=MUTED)
    fig.text(.025,.553,'B  Both goal conditions must pass at the same time',fontsize=8.3,weight='bold',va='top')
    thresholds=rows[0][0]['criterion']['thresholds']
    titles=(['Combined position error (px)','Block-angle error (rad)'] if env=='pusht' else ['Joint 1 absolute error (rad)','Joint 2 absolute error (rad)'])
    final_metrics={}
    for j in range(2):
        ax=fig.add_axes([.115+j*.485,.346,.355,.157]);style_axis(ax)
        ymax=max(float(values['criterion_errors'][10:,j].max()) for _,values in rows)*1.18
        ax.axhspan(0,thresholds[j],facecolor='#E5F2E8',zorder=0)
        ax.axhline(thresholds[j],color=GREEN,lw=.8,ls=':')
        for mode,(row,values) in zip(MODES,rows):
            times=values['trace_native_times']; choose=times>=10; errors=values['criterion_errors'][:,j]
            ax.plot(times[choose],errors[choose],color=COLORS[mode],lw=1.3,ls='-' if mode=='factorized' else '--')
            pts=[t for t in (10,15,20) if t<=times[-1]]
            ax.scatter(pts,errors[pts],color=COLORS[mode],marker='o' if mode=='factorized' else 's',s=8,zorder=3)
            ax.scatter(times[-1],errors[-1],color=GREEN if row['original_record']['success'] else RED,marker='o' if row['original_record']['success'] else 'x',s=20,zorder=4)
            final_metrics.setdefault(mode,[]).append(float(errors[-1]))
        ax.set_xlim(9,51);ax.set_ylim(0,max(ymax,thresholds[j]*2));ax.set_xticks([10,20,30,40,50])
        ax.set_xlabel('Native actions (support included)',fontsize=6.2,labelpad=2)
        ax.set_title(titles[j],fontsize=7.2,pad=6)
        ax.text(.02,.94,'Pass < '+('20 px' if env=='pusht' and j==0 else ('π/9 rad' if env=='pusht' else '0.05 rad')),transform=ax.transAxes,color=GREEN,fontsize=6.1,va='top')
        left=final_metrics['factorized'][j];right=final_metrics['framewise'][j]
        fmt='.1f' if env=='pusht' and j==0 else '.3f'
        ax.text(.98,.94,'At stop: '+format(left,fmt)+' / '+format(right,fmt),transform=ax.transAxes,ha='right',va='top',fontsize=6.0,color=INK)
    fig.text(.025,.267,'C  Same observed transition, two prediction models',fontsize=8.3,weight='bold',va='top')
    casepred=[r for r in predictions if r['environment']==env and r['seed']==seed]
    all_errors=[r['models'][m]['next_prediction_mse'] for r in casepred for m in MODES]
    if not all_errors or min(all_errors)<=0: raise ValueError('Need finite positive MSE for logarithmic display')
    ylo=min(all_errors)*.7;yhi=max(all_errors)*1.8
    counts={}
    for k,behavior in enumerate(MODES):
        ax=fig.add_axes([.115+k*.485,.072,.355,.13]);style_axis(ax)
        points=sorted([r for r in casepred if r['behavior_mode']==behavior],key=lambda r:r['anchor_native_time'])
        counts[behavior]={'ours_lower':0,'comparisons':len(points)}
        for r in points:
            t=r['anchor_native_time'];a=r['models']['factorized']['next_prediction_mse'];b=r['models']['framewise']['next_prediction_mse']
            counts[behavior]['ours_lower']+=int(a<b)
            ax.plot([t-.5,t+.5],[a,b],color=GREEN if a<b else RED,alpha=.55,lw=.8)
            ax.scatter(t-.5,a,color=BLUE,s=17,marker='o',zorder=3)
            ax.scatter(t+.5,b,color=GRAY,s=17,marker='s',zorder=3)
            change=100*(a/b-1)
            ax.text(t,max(a,b)*1.15,f'{abs(change):.1f}%'+('↓' if change<0 else '↑'),
                    ha='center',va='bottom',fontsize=6.1,weight='bold' if change<0 else 'normal',
                    color=GREEN if change<0 else RED)
        ax.set_xlim(8.5,21.5);ax.set_ylim(ylo,yhi);ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{value:g}'))
        ax.set_xticks([10,15,20]);ax.set_xticklabels(['10→15','15→20','20→25'],fontsize=6)
        ax.set_title('Actions from '+('ShiftWM' if behavior=='factorized' else 'Framewise'),fontsize=7.0,pad=5)
        if k==0:ax.set_ylabel('Latent MSE ↓ (log)',fontsize=6.5,labelpad=3)
        if len(points)<3:
            ax.text(.985,.04,'No full block\nafter success',transform=ax.transAxes,ha='right',va='bottom',fontsize=5.8,color=MUTED)
    handles=[Line2D([0],[0],color=BLUE,marker='o',lw=1.2,label='ShiftWM (ours)'),Line2D([0],[0],color=GRAY,marker='s',ls='--',lw=1.2,label='Framewise')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.06,.005),frameon=False,ncol=2,fontsize=6.8,columnspacing=2,handlelength=2)
    path=OUT/f'technical_{env}'
    for ext in ('pdf','svg','png'):fig.savefig(path.with_suffix('.'+ext),dpi=240,facecolor='white')
    plt.close(fig)
    return {'environment':env,'seed':seed,'display_crop_xyxy':list(crop),'terminal_criterion_errors':final_metrics,'prediction_comparisons':counts}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--if-needed',action='store_true');args=parser.parse_args()
    ledger=OUT/'technical_evidence.json'
    if args.if_needed and ledger.exists():
        old=json.loads(ledger.read_text())
        required=[OUT/f'technical_{env}.{ext}' for env in ('pusht','reacher') for ext in ('pdf','svg','png')]
        required.append(ROOT/'artifacts/qualitative/technical_comparisons.pdf')
        required.append(OUT/'technical_summary.tex')
        if all(p.is_file() for p in required) and all((ROOT/n).is_file() and sha(ROOT/n)==h for n,h in old['sources'].items()):
            print(json.dumps({'status':'reused_identical_technical_diagnostics'}));return
    records,predictions,sources=read_inputs();cases=selected_cases(records)
    selected=[next((env,seed) for env,seed in cases if env==name) for name in ('pusht','reacher')]
    outputs=[draw_case(env,seed,records,predictions) for env,seed in selected]
    sources[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    sources['paper/scripts/render_positive_qualitative.py']=sha(Path(__file__).with_name('render_positive_qualitative.py'))
    combined=ROOT/'artifacts/qualitative/technical_comparisons.pdf'
    subprocess.run(['pdfunite',*[str(OUT/f'technical_{env}.pdf') for env in ('pusht','reacher')],str(combined)],check=True)
    summaries=[]
    for env,seed in cases:
        pairs=[r for r in predictions if r['environment']==env and r['seed']==seed]
        summaries.append({'environment':env,'seed':seed,'paired_transitions':len(pairs),
             'ours_lower_prediction_error':sum(r['models']['factorized']['next_prediction_mse']<r['models']['framewise']['next_prediction_mse'] for r in pairs),
             'ours_lower_support_calibration_error':sum(r['models']['factorized']['support_calibration_mse']<r['models']['framewise']['support_calibration_mse'] for r in pairs)})
    packet={'status':'source_validated','selection':'First lexicographic ShiftWM-only success in each environment; positive subset, not a benchmark estimate','figures':outputs,'positive_case_prediction_summary':summaries,'sources':sources}
    ledger.write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
    total=sum(r['paired_transitions'] for r in summaries)
    lower=sum(r['ours_lower_prediction_error'] for r in summaries)
    support_lower=sum(r['ours_lower_support_calibration_error'] for r in summaries)
    wording=(f'Across the four positive cases, ShiftWM has lower next-observation '
        f'prediction MSE on {lower} of {total} matched transitions and lower '
        f'observed-history calibration MSE on {support_lower} of {total}. '
        'These outcome-selected, temporally dependent transitions are descriptive '
        'counts, not independent benchmark wins or significance estimates. '
        'Success therefore cannot be equated with improvement in every '
        'intermediate model measurement.\n')
    (OUT/'technical_summary.tex').write_text(wording)
    print(json.dumps({'status':'source_validated','technical_case_studies':len(outputs),'figures':outputs}))

if __name__=='__main__':main()
