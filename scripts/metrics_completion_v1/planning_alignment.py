"""Training-only native-action/restoration audit; no predictor or test access."""
import argparse
import json
import numpy as np
from planning_common import DATA, REG, REPORT, ROOT, atomic_json, checked_registration, require, sha
from planning_evaluate import environment,full_state,pusht_iou,render,restore,set_goal,source_case,task_state


def audit(task):
    reg=checked_registration()
    import h5py
    import hdf5plugin  # noqa: F401
    row=reg['datasets'][task]['partitions']['train'][0]
    with h5py.File(ROOT/DATA[task],'r') as f:
        initial,goal,support=source_case(f,task,row)
        start=row['offset']+row['planning_start']
        recorded=np.asarray(f['state' if task=='pusht' else'qpos'][start:start+11])
        steps=np.asarray(f['step_idx'][start:start+11])
        require(np.array_equal(steps,np.arange(row['planning_start'],row['planning_start']+11)),'Native steps not consecutive')
    require(np.max(np.abs(support))<10,'Recorded controls inconsistent with proposed relative/torque action coordinates')
    env=environment(task,173+row['episode_id'])
    try:
        set_goal(env,task,goal);restore(env,task,initial)
        observed=[task_state(env,task)];actual_images=[render(env,task)]
        for action in support:
            env.step(action);observed.append(task_state(env,task));actual_images.append(render(env,task))
        observed=np.asarray(observed)
        diagnostic={'native_state_l2_difference_to_recording':np.linalg.norm(observed-recorded,axis=1).tolist(),
                    'not_a_bitwise_replay_claim':'PushT lacks full block velocity/contact state and _set_state advances physics; discrepancies are measured, not corrected or hidden'}
        if task=='pusht':
            require(env.relative is True and env.action_scale==100,'Wrong PushT relative action interface')
            self_iou=pusht_iou(env,env._get_obs())
            require(abs(self_iou-1.)<1e-9,'Goal/body polygon self-IoU parity failed')
            diagnostic.update(relative=True,action_scale=env.action_scale,self_pose_polygon_iou=self_iou)
        else:
            require(env.action_repeat==2,'Reacher native action-repeat mismatch')
            diagnostic.update(action_repeat=env.action_repeat,qpos_threshold=float(env.env.task.qpos_threshold))
    finally:env.close()
    output={'schema':'current_spatial_training_alignment_v1','status':'passed','registration_sha256':sha(REG),
            'task':task,'episode_id':row['episode_id'],'split':'train','native_rows':steps.tolist(),
            'support_action_min':support.min(0).tolist(),'support_action_max':support.max(0).tolist(),
            'action_alignment_source':'pinned world/world.py episode_iter rotates collected successor-attached action left; terminal reset sentinel is excluded',
            'read_scope':'first preregistered training episode only; no validation/test data, predictor, or goal outcome used for selection',
            'diagnostic':diagnostic}
    atomic_json(output,REPORT/f'{task}_training_alignment.json');print(json.dumps(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=('pusht','reacher'),required=True)
    a=p.parse_args();audit(a.task)
