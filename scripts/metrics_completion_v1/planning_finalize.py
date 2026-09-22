"""All-24, all-100-case finalization; reconstruct physical outcomes from traces."""
import argparse
import json
import numpy as np
from planning_common import MODES, REG, REPORT, ROOT, TASKS, all_training_complete, atomic_json, checked_registration, read, require, sha


def ci(values, seed=173, draws=10000):
    values=np.asarray(values,dtype=np.float64)
    require(values.shape==(3,100) and np.isfinite(values).all(),'Expected complete three-seed x100case matrix')
    rng=np.random.default_rng(seed);seeds=rng.integers(3,size=(draws,3));cases=rng.integers(100,size=(draws,100))
    estimates=values[seeds[:,:,None],cases[:,None,:]].mean((1,2))
    return {'mean':float(values.mean()),'ci95':np.quantile(estimates,[.025,.975]).tolist(),
            'seed_means':values.mean(1).tolist(),'aggregation':'equal seed and episode; paired crossed percentile bootstrap10000/seed173'}


def eligible_success_ci(success,eligible,seed=173,draws=10000):
    success=np.asarray(success,dtype=np.float64);eligible=np.asarray(eligible,dtype=bool)
    require(success.shape==eligible.shape==(3,100),'Incomplete policy eligibility matrix')
    if not eligible.any():
        return {'status':'undefined_no_policy_eligible_cases','mean':None,'ci95':None,'eligible_seed_case_records':0}
    rng=np.random.default_rng(seed);s=rng.integers(3,size=(draws,3));e=rng.integers(100,size=(draws,100))
    weight=eligible[s[:,:,None],e[:,None,:]];denominator=weight.sum((1,2))
    numerator=(success[s[:,:,None],e[:,None,:]]*weight).sum((1,2))
    estimates=numerator[denominator>0]/denominator[denominator>0]
    return {'status':'complete','mean':float(success[eligible].mean()),'ci95':np.quantile(estimates,[.025,.975]).tolist(),
            'eligible_seed_case_records':int(eligible.sum()),'empty_eligible_bootstrap_draws':int((denominator==0).sum()),
            'aggregation':'paired crossed resampling of full seed/episode population; report conditional success among eligible records'}


def validate_trace(task,record):
    path=ROOT/record['trace_file'];require(sha(path)==record['trace_sha256'],'Planning trace changed')
    with np.load(path,allow_pickle=False) as f:
        states=f['states'];actions=f['actions'];goal=f['goal_state'];flags=f['success'];images=f['images']
        require(actions.shape==(record['native_calls'],2) and states.shape[0]==len(actions)+1,'Native trace count differs')
        require(images.dtype==np.uint8 and images.shape==(len(states),224,224,3),'Missing actual native RGB trace')
        require(np.isfinite(states).all() and np.isfinite(actions).all(),'Nonfinite physical trace')
        if task=='pusht':
            difference=np.abs(states[:,4]-goal[4]);angle=np.minimum(difference,2*np.pi-difference)
            expected=(np.linalg.norm(states[:,:4]-goal[:4],axis=1)<20)&(angle<np.pi/9)
        else:expected=np.all(np.abs(states[:,:2]-goal)<.05,axis=1)
        require(np.array_equal(flags,expected),'Saved success disagrees with native physical criterion')
        hits=np.flatnonzero(expected);first=int(hits[0]) if len(hits) else None
        require(record['success']==bool(len(hits)) and record['first_success_native_call']==first,'First-hit summary differs')
        require(record['support_only_success']==(first is not None and first<=10),'Support success mislabeled')
        require(record['policy_eligible']==(first is None or first>10),'Policy eligibility differs')
        require(record['native_calls']==(first if first is not None else 50),'Failure removed or budget/stop time changed')
        require(record['candidate_evaluations']==record['replans']*128*8,'Candidate work count differs')
        require(record['replans']<=2 and record['planning_wall_seconds']>=sum(record['solve_seconds'])>=0,'Invalid work/time accounting')
        require(len(record['native_scores'])==len(states),'Missing per-native diagnostics')
        for a,b in zip(record['native_scores'],expected):require(a['success']==bool(b),'Per-call success differs')


def finalize():
    reg=checked_registration();selected=all_training_complete(reg);evaluations={};bindings={}
    for row in reg['runs']:
        path=REPORT/'evaluations/test'/(row['name']+'.json')
        require(path.is_file(),'Full evaluation grid incomplete: '+row['name'])
        result=read(path)
        require(result['status']=='complete' and result['registration_sha256']==sha(REG) and result['run']==row,'Stale/incomplete evaluation')
        require(result['checkpoint_sha256']==selected[row['name']]['checkpoint_sha256'],'Wrong selected checkpoint')
        records=result['records'];expected=reg['datasets'][row['task']]['partitions']['test']
        require([r['episode_id']for r in records]==[r['episode_id']for r in expected] and len(records)==100,'Incomplete/changed test population')
        for record in records:validate_trace(row['task'],record)
        evaluations[row['name']]=records;bindings[str(path.relative_to(ROOT))]=sha(path)
    metrics={'success':lambda r:float(r['success']),'endpoint_success':lambda r:float(r['final_score']['success']),
             'support_only_success':lambda r:float(r['support_only_success']),'actual_native_calls':lambda r:float(r['native_calls']),
             'failure_penalized_calls':lambda r:float(r['first_success_native_call'] if r['success'] else 50),
             'planning_wall_seconds':lambda r:r['planning_wall_seconds'],'cem_seconds':lambda r:sum(r['solve_seconds']),
             'candidate_evaluations':lambda r:r['candidate_evaluations'],
             'peak_gpu_allocated_bytes':lambda r:r['peak_gpu_allocated_bytes']}
    summaries={};contrasts={}
    for task in TASKS:
        taskmetrics=dict(metrics)
        keys=('joint_pusher_block_xy_l2_px','block_xy_l2_px','pusher_xy_l2_px','circular_block_angle_error_rad','block_iou') if task=='pusht' else('joint_l2_rad','wrapped_joint_l2_rad')
        for key in keys:taskmetrics[key]=lambda r,key=key:r['final_score'][key]
        if task=='pusht':taskmetrics['maximum_block_iou']=lambda r:r['maximum_block_iou']
        arrays={mode:{metric:np.asarray([[fn(r)for r in evaluations[f'{task}_{mode}_s{s}']]for s in range(3)])
                      for metric,fn in taskmetrics.items()}for mode in MODES}
        # Verify genuinely paired starts/goals/support, across all methods/seeds.
        reference=evaluations[f'{task}_transport_s0']
        for mode in MODES:
            for seed in range(3):
                for a,b in zip(reference,evaluations[f'{task}_{mode}_s{seed}']):
                    require(all(a[k]==b[k]for k in('episode_id','planning_start','initial_render_sha256','goal_render_sha256','source_support_actions_sha256','support_only_success')),'Unpaired planning inputs')
        summaries[task]={mode:{metric:ci(v)for metric,v in data.items()}for mode,data in arrays.items()}
        for mode in MODES:
            eligible=np.asarray([[r['policy_eligible']for r in evaluations[f'{task}_{mode}_s{s}']]for s in range(3)])
            summaries[task][mode]['policy_eligible_success']=eligible_success_ci(arrays[mode]['success'],eligible)
            summaries[task][mode]['maximum_peak_gpu_allocated_bytes']=float(arrays[mode]['peak_gpu_allocated_bytes'].max())
        contrasts[task]={f'{a}_minus_{b}':{metric:ci(arrays[a][metric]-arrays[b][metric])for metric in taskmetrics}
                         for a,b in(('transport','autoregressive'),('transport','bounded_additive'),('unbounded_transport','transport'),('unbounded_transport','autoregressive'))}
    out={'schema':'current_spatial_planning_finalization_v1','status':'complete','registration_sha256':sha(REG),
         'completed_runs':24,'records':2400,'test_episodes_per_task':100,'selected_models':selected,
         'sources':bindings,'summaries':summaries,'paired_differences':contrasts,
         'difference_direction':'named first minus second; positive success/IoU favors first, positive error/cost favors second',
         'scope':reg['scope'],'protocol':reg['planning'],'metric_contract':reg['metric_contract']}
    atomic_json(out,REPORT/'finalization.json');return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.parse_args();print(json.dumps({'status':finalize()['status']}))
