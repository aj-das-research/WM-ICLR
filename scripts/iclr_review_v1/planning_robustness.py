"""Post-hoc robustness and physical-criterion diagnosis; no training or selection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'paper/scripts'))
from render_current_spatial_planning_completion import load_pack, PACK, MODES, COLORS
OUT=ROOT/'reports/iclr_review_2026-09-22'
FIG=ROOT/'paper/generated/iclr_review_v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')

def holm(pvalues):
    order=np.argsort(pvalues); adjusted=np.empty(len(order)); previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(order)-rank)*pvalues[index]));adjusted[index]=previous
    return adjusted

def main():
    data=load_pack(PACK)
    protocol={'scope':'post-hoc diagnostic on revealed current-spatial simulator tests; no model/endpoint reselection',
      'inputs':{'paper/figure_sources/current_spatial_planning_completion/data.json':sha(PACK/'data.json')},
      'method':'episode-cluster sign flips after averaging three trained seeds; two-sided; 100000 Monte Carlo draws; plus-one correction; Holm over all eight declared success contrasts',
      'limitation':'conditional on the three fitted seeds; symmetric cluster effects required for sign-flip inference; sensitivity analysis, not confirmatory evidence',
      'physical':'terminal criterion failures among all300 seed-case records; joint errors decomposed only for observed terminal states; no causal attribution',
      'curves':'equal-seed/episode mean physical error; terminated successes retain final state for plotting only; no invented post-stop transitions',
      'seed':173}
    OUT.mkdir(exist_ok=True,parents=True);FIG.mkdir(exist_ok=True,parents=True)
    frozen=OUT/'planning_robustness_protocol.json'
    if frozen.exists(): assert json.loads(frozen.read_text())==protocol
    else: write(frozen,protocol)
    rng=np.random.default_rng(173)
    signs=rng.choice(np.array([-1.,1.]),size=(100000,100))
    contrasts=[];failures={};curves={}
    pairs=[('transport','autoregressive'),('transport','bounded_additive'),('unbounded_transport','transport'),('unbounded_transport','autoregressive')]
    for task in ['pusht','reacher']:
        arrays={};failures[task]={};curves[task]={}
        for mode in MODES:
            rows=[data['evaluations'][f'{task}_{mode}_s{s}']for s in range(3)]
            arrays[mode]=np.array([[r['success']for r in rr]for rr in rows],float)
            flat=sum(rows,[])
            metric='joint_pusher_block_xy_l2_px'if task=='pusht'else'joint_l2_rad'
            trajectories=np.array([[r['native_scores'][min(t,len(r['native_scores'])-1)][metric]for t in range(51)]for r in flat])
            curves[task][mode]=trajectories.mean(0).tolist()
            failed=[r for r in flat if not r['success']]
            if task=='reacher':
                f=np.array([r['final_score']['per_joint_unwrapped_error_rad']for r in failed])>=.05
                categories={'joint1_only':int((f[:,0]&~f[:,1]).sum()),'joint2_only':int((~f[:,0]&f[:,1]).sum()),'both_joints':int((f[:,0]&f[:,1]).sum())}
            else:
                pos=np.array([r['final_score']['joint_pusher_block_xy_l2_px']for r in failed])>=20
                ang=np.array([r['final_score']['circular_block_angle_error_rad']for r in failed])>=np.pi/9
                categories={'position_only':int((pos&~ang).sum()),'angle_only':int((~pos&ang).sum()),'both':int((pos&ang).sum())}
            assert sum(categories.values())==len(failed)
            failures[task][mode]={'failed_seed_case_records':len(failed),'all_seed_case_records':300,'categories':categories,
              'seed_success_percent':(100*arrays[mode].mean(1)).tolist()}
        for a,b in pairs:
            differences=arrays[a]-arrays[b]; clusters=differences.mean(0)
            observed=abs(clusters.mean());null=(signs@clusters)/100
            p=(1+int((abs(null)>=observed-1e-12).sum()))/100001
            contrasts.append({'task':task,'first':a,'reference':b,'difference_pp':float(differences.mean()*100),
              'seed_differences_pp':(differences.mean(1)*100).tolist(),'p_two_sided':p,
              'episodes_with_positive_seed_average':int((clusters>0).sum()),'episodes_with_negative_seed_average':int((clusters<0).sum()),'episodes_tied':int((clusters==0).sum())})
    adjusted=holm([r['p_two_sided']for r in contrasts])
    for r,p in zip(contrasts,adjusted):r['p_holm_eight']=float(p)
    result={'status':'complete','protocol_sha256':sha(frozen),'contrasts':contrasts,'terminal_failure_diagnosis':failures,'physical_error_by_call':curves}
    write(OUT/'planning_robustness.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.45),layout='constrained')
    labels={'transport':'Bounded mixing','unbounded_transport':'Unbounded mixing'}
    for ax,task in zip(axes,['pusht','reacher']):
        for i,mode in enumerate(labels):
            row=next(r for r in contrasts if r['task']==task and r['first']==mode and r['reference']=='autoregressive')
            ci=data['finalization']['paired_differences'][task][mode+'_minus_autoregressive']['success']['ci95']
            mean=row['difference_pp'];lo,hi=np.array(ci)*100
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='D',color=COLORS[mode],capsize=3,markersize=5)
            ax.scatter(row['seed_differences_pp'],np.full(3,i+.14),marker='|',s=70,color=COLORS[mode])
        ax.axvline(0,color='.55',lw=.8,ls='--');ax.set_xlim(-70,15)
        ax.set_yticks([0,1],list(labels.values())if task=='pusht'else['',''])
        ax.set_ylim(-.45,1.5);ax.set_title('PushT'if task=='pusht'else'Reacher',loc='left',fontweight='bold')
        ax.set_xlabel('Success difference vs. AR (pp)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='.9',lw=.5)
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'planning_seed_robustness.{ext}',dpi=240)
    plt.close(fig)
    (FIG/'figure.tex').write_text(r'''\begin{figure}[tbp]
\centering\includegraphics[width=\linewidth]{generated/iclr_review_v1/planning_seed_robustness.pdf}
\caption{Current-model control is not a general benefit of observed-state mixing.
Diamonds show success differences against matched autoregression; whiskers are
the original paired seed--episode 95\% bootstrap intervals. Small vertical marks
show each of the three training-seed differences. All 100 held-out starting
episodes per task are retained. Reacher regressions occur for every seed in both
mixing arms; the PushT intervals include zero. This secondary display does not
change the model selection, primary endpoint or test population.}
\label{fig:planning-seed-robustness}\end{figure}
''')
    lines=['# Post-hoc planning robustness','',protocol['scope'],'']
    for r in contrasts:lines.append(f"- {r['task']} {r['first']} vs {r['reference']}: {r['difference_pp']:+.2f} pp; seed differences {r['seed_differences_pp']}; two-sided p={r['p_two_sided']:.6g}, Holm(8)={r['p_holm_eight']:.6g}.")
    lines+=['',protocol['limitation'],'','Terminal failure decomposition (counts are correlated seed-case records, not independent episodes):',json.dumps(failures,indent=2)]
    (OUT/'planning_robustness.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','contrasts':len(contrasts),'records':2400}))
if __name__=='__main__':main()
