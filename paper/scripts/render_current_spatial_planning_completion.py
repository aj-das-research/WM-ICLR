#!/usr/bin/env python3
"""Complete-only current spatial planning tables/curves; no model or binary data access."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/metrics_completion_v1/planning'
PACK=ROOT/'paper/figure_sources/current_spatial_planning_completion'
OUTPUT=ROOT/'paper/generated/current_spatial_planning_completion'
TASKS={'pusht':'PushT','reacher':'Reacher'}
MODES={'autoregressive':'Autoregressive','bounded_additive':'Additive control',
       'transport':'ShiftWM (ours)','unbounded_transport':'No tanh (ours, ablation)'}
COLORS={'autoregressive':'#59616D','bounded_additive':'#AE7922','transport':'#255E91','unbounded_transport':'#82517E'}
STYLES={'autoregressive':('--','s'),'bounded_additive':(':','^'),'transport':('-','o'),'unbounded_transport':('-.','D')}
SCHEMA='current_spatial_planning_public_v1'
PENDING=(r'\paragraph{Current spatial closed-loop control (pending).} '
    r'A separate PushT/Reacher study uses actual frozen DINOv2 $4\times4$ tokens, '
    r'four matched predictor families and three training seeds. Each task has '
    r'1,000 training, 100 development and 100 test episodes. Results will appear '
    r'only after all 24 full-training runs and all 2,400 planning trials validate; '
    r'no partial success rates are reported. Each trial permits 50 native calls '
    r'including ten paid support calls, with shared CEM $128\times8$ and 16 elites. '
    r'These are new current-spatial experiments, separate from historical-model results.'+'\n')
FIG_CAPTION=(r'Current spatial predictors in closed-loop simulator control. Each curve '
    r'is the fraction of all 100 test episodes, averaged equally over three training '
    r'seeds, that has reached the native success criterion by a given call. '
    r'Failures remain in the denominator. The gray band marks the ten paid support '
    r'calls; success there precedes policy planning. Every method uses the same '
    r'goal, start, support controls and CEM $128\times8$/top-16 budget, with '
    r'25 then at most 15 policy calls. Curves show point estimates; endpoint '
    r'uncertainty is in Table~\ref{tab:current-spatial-planning-success}. '
    r'PushT requires joint pusher/block position error below 20 pixels and circular '
    r'angle error below $\pi/9$; Reacher requires each unwrapped joint error below '
    r'0.05 rad. These are current-spatial results, not the historical context model.')


def check(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def json_text(value):return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'


def write_changed(path,value):
    path=Path(path);data=value.encode() if isinstance(value,str)else value
    if path.exists() and path.read_bytes()==data:return False
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_bytes(data);temporary.replace(path);return True


def numeric_tree(value):
    if isinstance(value,dict):
        for x in value.values():numeric_tree(x)
    elif isinstance(value,list):
        for x in value:numeric_tree(x)
    elif isinstance(value,float):check(math.isfinite(value),'Nonfinite report number')


def scalar_metrics(record,task):
    result={'success':float(record['success']),'endpoint_success':float(record['final_score']['success']),
        'support_only_success':float(record['support_only_success']),'actual_native_calls':float(record['native_calls']),
        'failure_penalized_calls':float(record['first_success_native_call'] if record['success']else 50),
        'planning_wall_seconds':record['planning_wall_seconds'],'cem_seconds':sum(record['solve_seconds']),
        'candidate_evaluations':record['candidate_evaluations'],'peak_gpu_allocated_bytes':record['peak_gpu_allocated_bytes']}
    fields=('joint_pusher_block_xy_l2_px','block_xy_l2_px','pusher_xy_l2_px','circular_block_angle_error_rad','block_iou')if task=='pusht'else('joint_l2_rad','wrapped_joint_l2_rad')
    result.update({k:record['final_score'][k]for k in fields})
    if task=='pusht':result['maximum_block_iou']=record['maximum_block_iou']
    return result


def validate(data):
    check(data.get('schema')==SCHEMA and data.get('status')=='complete','Incomplete public pack')
    final=data['finalization'];reg=data['registration']
    check(final['schema']=='current_spatial_planning_finalization_v1' and final['status']=='complete','Incomplete finalization')
    check((final['completed_runs'],final['records'],final['test_episodes_per_task'])==(24,2400,100),'Incomplete population')
    expected={f'{t}_{m}_s{s}'for t in TASKS for m in MODES for s in range(3)}
    check(set(data['evaluations'])==set(final['selected_models'])==expected,'Run grid differs')
    check(len(reg['runs'])==24 and {r['name']for r in reg['runs']}==expected,'Registered run grid differs')
    check(final['registration_sha256']==data['source_bindings']['reports/metrics_completion_v1/planning/registration.json'],'Registration binding differs')
    check(reg['planning']['native_budget']==50 and reg['planning']['support_native_calls']==10 and reg['planning']['solve_horizons_if_no_early_success']==[5,3],'Control protocol differs')
    check((reg['planning']['samples'],reg['planning']['iterations'],reg['planning']['elites'])==(128,8,16),'CEM budget differs')
    numeric_tree(data)
    matrices={}
    for row in reg['runs']:
        name,task,mode,seed=(row[k]for k in('name','task','mode','seed'))
        records=data['evaluations'][name]
        check(len(records)==100 and [r['episode_id']for r in records]==[r['episode_id']for r in reg['datasets'][task]['partitions']['test']],'Case population differs')
        for r in records:
            first=r['first_success_native_call']
            check(first is None or type(first)is int and 0<=first<=50,'Invalid first success')
            check(r['success']==(first is not None),'First-hit outcome differs')
            check(r['native_calls']==(first if first is not None else 50),'Stopped calls/failure budget differs')
            check(r['support_only_success']==(first is not None and first<=10),'Support success differs')
            check(r['policy_eligible']==(first is None or first>10),'Eligibility differs')
            check(len(r['native_scores'])==r['native_calls']+1,'Native score trace missing')
            hits=[i for i,v in enumerate(r['native_scores'])if v['success']]
            check((hits[0] if hits else None)==first and r['native_scores'][-1]==r['final_score'],'Per-call/final scores differ')
            check(r['candidate_evaluations']==r['replans']*1024 and 0<=r['replans']<=2,'Candidate count differs')
            check(r['planning_wall_seconds']>=sum(r['solve_seconds'])>=0 and r['peak_gpu_allocated_bytes']>=0,'Cost accounting differs')
        for metric in scalar_metrics(records[0],task):
            matrices.setdefault((task,mode,metric),np.zeros((3,100)))[seed]=[scalar_metrics(r,task)[metric]for r in records]
    for (task,mode,metric),arr in matrices.items():
        summary=final['summaries'][task][mode][metric]
        check(abs(float(arr.mean())-summary['mean'])<=1e-10*max(1,abs(summary['mean'])),'Primitive mean differs: '+str((task,mode,metric)))
        check(np.allclose(arr.mean(1),summary['seed_means'],rtol=1e-12,atol=1e-12),'Seed means differ')
        check(len(summary['ci95'])==2 and summary['ci95'][0]<=summary['ci95'][1],'Invalid reported interval')
    for task in TASKS:
        reference=data['evaluations'][f'{task}_transport_s0']
        for mode in MODES:
            records=[r for seed in range(3)for r in data['evaluations'][f'{task}_{mode}_s{seed}']]
            summary=final['summaries'][task][mode]
            eligible=[r for r in records if r['policy_eligible']]
            check(summary['policy_eligible_success']['mean']==(sum(r['success']for r in eligible)/len(eligible) if eligible else None),'Conditional success differs')
            check(summary['maximum_peak_gpu_allocated_bytes']==max(r['peak_gpu_allocated_bytes']for r in records),'Memory maximum differs')
            for seed in range(3):
                for a,b in zip(reference,data['evaluations'][f'{task}_{mode}_s{seed}']):
                    check(all(a[k]==b[k]for k in('episode_id','planning_start','initial_render_sha256','goal_render_sha256','source_support_actions_sha256','support_only_success')),'Unpaired inputs')
        for a,b in(('transport','autoregressive'),('transport','bounded_additive'),('unbounded_transport','transport'),('unbounded_transport','autoregressive')):
            for metric,item in final['paired_differences'][task][a+'_minus_'+b].items():
                expected_mean=float((matrices[task,a,metric]-matrices[task,b,metric]).mean())
                check(abs(item['mean']-expected_mean)<=1e-10*max(1,abs(expected_mean)),'Signed paired difference differs')
    return data


def load_pack(pack):
    manifest=read(pack/'manifest.json')
    check(manifest.get('schema')==SCHEMA+'_manifest' and manifest.get('status')=='complete','Invalid pack manifest')
    check(manifest['runtime_inputs_sha256']=={'data.json':sha(pack/'data.json')},'Portable source changed')
    data=validate(read(pack/'data.json'))
    check(manifest['source_bindings']==data['source_bindings'],'Provenance mapping differs')
    return data


def live_pack(report,pack):
    final_path=report/'finalization.json'
    if not final_path.exists():return None
    final=read(final_path)
    check(final.get('status')=='complete','Present finalization is incomplete')
    reg_path=report/'registration.json';reg=read(reg_path);review=read(report/'source_review.json')
    check(final['registration_sha256']==sha(reg_path) and review.get('status')=='passed' and review['registration_sha256']==sha(reg_path),'Registration/review stale')
    # Source bytes only: do not open models, dataset arrays or local trace archives.
    for relative,expected in reg['dependencies'].items():
        if relative.endswith(('.py','.slurm','.md')):check(sha(ROOT/relative)==expected,'Frozen source changed: '+relative)
    bindings={'reports/metrics_completion_v1/planning/'+p.name:sha(p)for p in(final_path,reg_path,report/'source_review.json')}
    records={}
    for row in reg['runs']:
        relative='reports/metrics_completion_v1/planning/evaluations/test/'+row['name']+'.json'
        path=report/'evaluations/test'/(row['name']+'.json')
        check(final['sources'].get(relative)==sha(path),'Evaluation ledger changed')
        payload=read(path)
        check(payload['status']=='complete' and payload['run']==row and payload['registration_sha256']==sha(reg_path),'Wrong evaluation identity')
        check(payload['checkpoint_sha256']==final['selected_models'][row['name']]['checkpoint_sha256'],'Wrong model identity')
        records[row['name']]=payload['records'];bindings[relative]=sha(path)
    data=validate({'schema':SCHEMA,'status':'complete','registration':reg,'finalization':final,'evaluations':records,
                   'source_bindings':bindings,'raw_rgb_actions_states_features_included':False,
                   'physical_trace_validation':'performed by complete source-bound finalizer; renderer reads JSON only'})
    write_changed(pack/'data.json',json_text(data))
    write_changed(pack/'manifest.json',json_text({'schema':SCHEMA+'_manifest','status':'complete','completed_runs':24,'case_records':2400,
        'runtime_inputs_sha256':{'data.json':sha(pack/'data.json')},'source_bindings':bindings}))
    return data


def success_curves(data):
    result={}
    for task in TASKS:
        result[task]={}
        for mode in MODES:
            rows=[r for seed in range(3)for r in data['evaluations'][f'{task}_{mode}_s{seed}']]
            result[task][mode]=[100*sum(r['first_success_native_call'] is not None and r['first_success_native_call']<=n for r in rows)/300 for n in range(51)]
    return result


def number(item,scale=1,precision=1,interval=False):
    if item['mean']is None:return 'undefined'
    value=f"{item['mean']*scale:.{precision}f}"
    if interval:value+=r'\,['+f"{item['ci95'][0]*scale:.{precision}f}, {item['ci95'][1]*scale:.{precision}f}"+']'
    return '$'+value+'$'


def table(label,caption,columns,header,rows):
    return '\n'.join([r'\begin{table}[tbp]',r'\centering\begingroup\fontsize{9}{10.8}\selectfont',r'\setlength{\tabcolsep}{3pt}',
       r'\begin{tabular}{@{}'+columns+r'@{}}\toprule',header+r' \\',r'\midrule',*rows,
       r'\bottomrule\end{tabular}\endgroup',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table}',''])


def signed_cell(item,scale,precision):
    point=item['mean']*scale;lo,hi=sorted(v*scale for v in item['ci95'])
    text=f'{point:+.{precision}f}'
    if point>0:text=r'\textcolor{green!40!black}{\textbf{'+text+'}}'
    return r'\shortstack{'+text+r'\\{}['+f'{lo:+.{precision}f}, {hi:+.{precision}f}'+']}'


def tables(data):
    summary=data['finalization']['summaries'];success=[];cost=[];physical=[]
    for task,name in TASKS.items():
        success.append(r'\multicolumn{5}{@{}l}{\textbf{'+name+r'}} \\')
        cost.append(r'\multicolumn{5}{@{}l}{\textbf{'+name+r'}} \\')
        for mode,label in MODES.items():
            row=summary[task][mode]
            success.append(' & '.join([(r'\shortstack[l]{No tanh (ours,\\ablation)}'if mode=='unbounded_transport'else label),number(row['success'],100,1,True),number(row['support_only_success'],100),number(row['policy_eligible_success'],100),number(row['actual_native_calls'])])+r' \\')
            cost.append(' & '.join([(r'\shortstack[l]{No tanh (ours,\\ablation)}'if mode=='unbounded_transport'else label),number(row['planning_wall_seconds'],1,2),number(row['cem_seconds'],1,2),number(row['candidate_evaluations'],1,0),f"{row['maximum_peak_gpu_allocated_bytes']/1048576:.1f}"])+r' \\')
        if task=='pusht':
            physical.append(r'\multicolumn{6}{@{}l}{\textbf{PushT}} \\')
            physical.append(r'Method & \shortstack{Pusher\\px} & \shortstack{Block\\px} & \shortstack{Angle\\rad} & \shortstack{Final\\IoU} & \shortstack{Max.\\IoU} \\')
            for mode,label in MODES.items():
                row=summary[task][mode]
                physical.append(' & '.join([(r'\shortstack[l]{No tanh (ours,\\ablation)}'if mode=='unbounded_transport'else label),*[number(row[k],1,3 if k in('circular_block_angle_error_rad','block_iou','maximum_block_iou')else 1)for k in('pusher_xy_l2_px','block_xy_l2_px','circular_block_angle_error_rad','block_iou','maximum_block_iou')]])+r' \\')
        else:
            physical.extend([r'\midrule',r'\multicolumn{6}{@{}l}{\textbf{Reacher}} \\',r'Method & \multicolumn{2}{c}{Unwrapped L2 (rad)} & \multicolumn{3}{c}{Wrapped L2 (rad)} \\'])
            for mode,label in MODES.items():
                row=summary[task][mode]
                physical.append((r'\shortstack[l]{No tanh (ours,\\ablation)}'if mode=='unbounded_transport'else label)+' & '+r'\multicolumn{2}{c}{'+number(row['joint_l2_rad'],1,3)+'} & '+r'\multicolumn{3}{c}{'+number(row['wrapped_joint_l2_rad'],1,3)+r'} \\')
    captions={
      'success':r'Current spatial planning on 100 held-out episodes per task and three seeds per method. SR is any-call success (\%); brackets are 95\% paired seed--episode crossed-bootstrap intervals (10,000 draws, seed 173). Support is success within the first ten calls; eligible SR excludes those cases. Calls include support and end at first success or the 50-call failure budget, so actual and failure-penalized calls coincide. Endpoint success equals any-call success under this stopping rule. All methods share real DINOv2 $4\times4$ tokens, the whole-episode split and reduced-compute CEM protocol. No historical result is reused.',
      'cost':r'Complete online planning cost for the same current-spatial trials, averaged equally across all seeds and episodes, including support-only successes and failures. Online time includes image encoding, context computation, CEM, simulation, rendering and scoring; it excludes environment construction and trace-file writing. CEM is synchronized solver wall time, included in Online. Candidates counts full candidate sequences (128 per iteration, eight iterations per replan). GPU memory is the maximum PyTorch allocated peak across all 300 trials in the row, including resident encoder and predictor, not whole-device memory. Hardware identities and per-case times remain in the bound execution receipts.',
      'physical':r'Physical endpoint diagnostics for current spatial planning; means include all successes and failures. PushT pusher/block errors are Euclidean pixel distances; angle is circular radians; polygon IoU is a secondary diagnostic, not the success criterion. Max. IoU is the maximum along the actual trace. Reacher unwrapped and wrapped joint L2 are diagnostics; native success requires each unwrapped joint error below 0.05 rad. All uncertainties, combined PushT position error and signed paired differences remain in the complete portable numeric pack.'}
    contrasts=[]
    short={'transport':'ShiftWM','autoregressive':'AR','bounded_additive':'Additive','unbounded_transport':'No tanh'}
    for task,name in TASKS.items():
        metric='joint_pusher_block_xy_l2_px' if task=='pusht' else 'joint_l2_rad'
        unit='px' if task=='pusht' else 'rad'
        contrasts.append(r'\multicolumn{3}{@{}l}{\textbf{'+name+'}; error gain in '+unit+r'} \\')
        for a,b in(('transport','autoregressive'),('transport','bounded_additive'),('unbounded_transport','transport'),('unbounded_transport','autoregressive')):
            row=data['finalization']['paired_differences'][task][a+'_minus_'+b]
            contrasts.append(short[a]+' vs. '+short[b]+' & '+signed_cell(row['success'],100,1)+' & '+signed_cell(row[metric],-1,3 if task=='reacher'else 1)+r' \\')
    return {
      'success.tex':table('tab:current-spatial-planning-success',captions['success'],'lrrrr',r'Method & SR [95\% CI] & Support & Eligible SR & Calls',success),
      'cost.tex':table('tab:current-spatial-planning-cost',captions['cost'],'lrrrr',r'Method & Online (s) & CEM (s) & Candidates & GPU (MiB)',cost),
      'paired.tex':table('tab:current-spatial-planning-paired',r'Paired current-spatial planning effects. SR gain is named first minus reference in percentage points; error gain is reference minus first in the named physical units. Green bold positive points indicate favorable point estimates, not significance. Brackets are the registered 95\% paired seed--episode crossed-bootstrap intervals; every favorable and unfavorable comparison is retained. ShiftWM is ours; No tanh is our ablation; AR is the matched autoregressive predictor. PushT error combines pusher and block position, with angle retained separately in the physical table. Reacher error is unwrapped joint L2.','lrr',r'Comparison & SR gain (pp) & Error gain',contrasts),
      'physical.tex':table('tab:current-spatial-planning-physical',captions['physical'],'lrrrrr',r'\multicolumn{6}{l}{Physical errors and goal overlap}',physical)}


def plot(data,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'current_spatial_planning_v1'})
    curves=success_curves(data);fig,axes=plt.subplots(1,2,figsize=(5.5,2.35))
    fig.subplots_adjust(left=.105,right=.975,bottom=.21,top=.70,wspace=.25)
    handles=[Line2D([],[],color=COLORS[m],ls=STYLES[m][0],marker=STYLES[m][1],markersize=3,lw=1.3,label=label)for m,label in MODES.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,frameon=False,columnspacing=1.1,handlelength=2,handletextpad=.5)
    for ax,(task,name)in zip(axes,TASKS.items()):
        ax.axvspan(0,10,color='#E9ECEF',zorder=-2)
        for mode in MODES:
            line,marker=STYLES[mode]
            ax.step(range(51),curves[task][mode],where='post',color=COLORS[mode],ls=line,marker=marker,lw=1.25,markersize=3,markevery=[0,10,20,30,40,50])
        ax.set(xlim=(0,50),ylim=(0,102),xticks=[0,10,25,35,50],yticks=[0,25,50,75,100],xlabel='Native calls (support included)')
        ax.set_title(name,loc='left',fontweight='bold',fontsize=9,pad=4)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2.5,pad=2)
        ax.grid(axis='y',color='#DDE1E5',lw=.5,zorder=-1)
    axes[0].set_ylabel('Success by call (%)',labelpad=3)
    for suffix in('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None}if suffix=='pdf'else{'Date':None}if suffix=='svg'else None
        fig.savefig(directory/f'success_by_budget.{suffix}',dpi=240,metadata=metadata)
    plt.close(fig)


def render(data,output,pack):
    validate(data)
    output.mkdir(parents=True,exist_ok=True);written=False
    for name,text in tables(data).items():written|=write_changed(output/name,text)
    figure='\\begin{figure}[tbp]\n\\centering\n\\includegraphics[width=\\linewidth]{generated/current_spatial_planning_completion/success_by_budget.pdf}\n\\caption{'+FIG_CAPTION+'}\n\\label{fig:current-spatial-planning-budget}\n\\end{figure}\n'
    written|=write_changed(output/'figure.tex',figure)
    written|=write_changed(output/'section.tex','\n'.join(r'\input{generated/current_spatial_planning_completion/'+x+'}'for x in('success.tex','physical.tex','cost.tex','paired.tex','figure.tex'))+'\n')
    with tempfile.TemporaryDirectory(prefix='current-spatial-report-')as temp:
        plot(data,Path(temp))
        for path in sorted(Path(temp).iterdir()):written|=write_changed(output/path.name,path.read_bytes())
    files=['success.tex','physical.tex','cost.tex','paired.tex','figure.tex','section.tex','success_by_budget.pdf','success_by_budget.svg','success_by_budget.png']
    evidence={'schema':SCHEMA+'_render','status':'complete','completed_runs':24,'case_records':2400,'renderer_sha256':sha(__file__),
      'pack_manifest_sha256':sha(pack/'manifest.json'),'pack_data_sha256':sha(pack/'data.json'),
      'outputs_sha256':{name:sha(output/name)for name in files},'curve_points':408,'figure_inches':[5.5,2.35],
      'minimum_plot_font_pt':8,'table_font_pt':9,'actual_complete_data_pixel_review':'required_after_first_complete_render',
      'model_dataset_or_npz_payloads_opened':False}
    written|=write_changed(output/'evidence.json',json_text(evidence))
    return {'status':'complete','outputs_written':written,'completed_runs':24,'case_records':2400}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--if-ready',action='store_true');parser.add_argument('--from-pack',action='store_true')
    parser.add_argument('--report-dir',type=Path,default=REPORT);parser.add_argument('--pack-dir',type=Path,default=PACK)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT);args=parser.parse_args()
    if args.from_pack:data=load_pack(args.pack_dir)
    elif (args.report_dir/'finalization.json').exists():data=live_pack(args.report_dir,args.pack_dir)
    elif (args.pack_dir/'manifest.json').exists():data=load_pack(args.pack_dir)
    else:data=None
    if data is None:
        check(args.if_ready,'Complete planning finalization required')
        check(not(args.output_dir/'evidence.json').exists(),'Complete outputs exist but their bound source is missing')
        changed=write_changed(args.output_dir/'section.tex',PENDING)
        print(json.dumps({'status':'pending_complete24','outputs_written':changed,'accuracy_values_emitted':False}));return
    print(json.dumps(render(data,args.output_dir,args.pack_dir)))


if __name__=='__main__':main()
