"""Actual matched closed-loop control, with scorer-only simulator state access."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from planning_common import (DATA, REG, REPORT, ROOT, all_training_complete, array_sha, atomic_json,
                             checked_registration, digest, install_simulator_runtime, read, require, select_run, sha)
from planning_train import load_package


def environment(task, seed):
    install_simulator_runtime()
    if task=='pusht':
        from stable_worldmodel.envs.pusht.env import PushT
        env=PushT(relative=True,resolution=224)
    else:
        from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper
        env=ReacherDMControlWrapper(task='qpos_match',seed=seed)
    env.reset(seed=seed)
    return env


def render(env, task):
    image=env.render() if task=='pusht' else env.render(width=224,height=224)
    require(image.shape==(224,224,3) and image.dtype==np.uint8,'Simulator RGB contract differs')
    return image


def task_state(env,task):
    return env._get_obs().copy() if task=='pusht' else env.env.physics.data.qpos.copy()


def full_state(env,task):
    if task=='pusht':
        return np.r_[env._get_obs(),np.asarray(env.block.velocity),env.block.angular_velocity]
    return np.r_[env.env.physics.data.qpos,env.env.physics.data.qvel]


def restore(env,task,data):
    # Match the upstream initializer exactly. PushT _set_state performs one
    # physics tick and stores agent velocity; observations are freshly rendered.
    if task=='pusht':env._set_state(data['state'])
    else:env.set_state(data['qpos'],data['qvel'])


def set_goal(env,task,goal):
    if task=='pusht':env._set_goal_state(goal.copy())
    else:env.set_target_qpos(goal.copy())


def pusht_iou(env,goal):
    """Scoring-only union of the actual rigid block's convex Pymunk polygons."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    body=env._get_goal_pose_body(np.asarray(goal[2:5]))
    current=[];target=[]
    for shape in env.block.shapes:
        vertices=shape.get_vertices()
        current.append(Polygon([tuple(env.block.local_to_world(v)) for v in vertices]))
        target.append(Polygon([tuple(body.local_to_world(v)) for v in vertices]))
    a,b=unary_union(current),unary_union(target)
    require(a.is_valid and b.is_valid and a.area>0 and b.area>0,'Invalid PushT block geometry')
    return float(a.intersection(b).area/a.union(b).area)


def score(env,task,goal):
    current=task_state(env,task)
    if task=='pusht':
        success,distance=env.eval_state(goal,current)
        angle=float(abs(goal[4]-current[4]));angle=min(angle,2*np.pi-angle)
        return {'success':bool(success),'full_state_l2_mixed_units':float(distance),
                'joint_pusher_block_xy_l2_px':float(np.linalg.norm(goal[:4]-current[:4])),
                'block_xy_l2_px':float(np.linalg.norm(goal[2:4]-current[2:4])),
                'pusher_xy_l2_px':float(np.linalg.norm(goal[:2]-current[:2])),
                'circular_block_angle_error_rad':angle,'block_iou':pusht_iou(env,goal)}
    error=np.abs(current-goal);threshold=float(env.env.task.qpos_threshold)
    require(threshold==.05,'Upstream Reacher success threshold changed')
    return {'success':bool(np.all(error<threshold)),'joint_l2_rad':float(np.linalg.norm(error)),
            'per_joint_unwrapped_error_rad':error.tolist(),
            'wrapped_joint_l2_rad':float(np.linalg.norm((current-goal+np.pi)%(2*np.pi)-np.pi))}


class SpatialGoalCost(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.candidate_evaluations=0

    def to_native(self,z):
        return (z*self.model.action_std+self.model.action_mean).clamp(-1.,1.)

    def get_cost(self,info,actions):
        b,k,h,d=actions.shape
        x=info['history_features'].flatten(0,1);past=info['past_actions'].flatten(0,1)
        target=info['goal_features'].flatten(0,1)
        predictions=self.model.predict(x,past,self.to_native(actions).reshape(b*k,h,d))
        cost=((predictions[:,-1]-target)/self.model.feature_std).square().mean(-1)
        require(torch.isfinite(cost).all(),'Nonfinite candidate cost')
        self.candidate_evaluations+=b*k
        return cost.reshape(b,k)


def source_case(source,task,row):
    start=row['offset']+row['planning_start'];end=start+35
    key='episode_idx' if task=='pusht' else 'ep_idx'
    require(int(source[key][start])==int(source[key][end])==row['episode_id'],'Planning rows cross episodes')
    require(int(source['step_idx'][end])-int(source['step_idx'][start])==35,'Nonconsecutive source time')
    fields=('state',) if task=='pusht' else ('qpos','qvel')
    initial={k:np.asarray(source[k][start]) for k in fields}
    goal_state=np.asarray(source['state' if task=='pusht' else 'qpos'][end])
    support=np.asarray(source['action'][start:start+10],dtype=np.float32)
    require(support.shape==(10,2) and np.isfinite(support).all(),'Invalid fixed support actions')
    return initial,goal_state,support


def planning_horizon(native_used):
    require(type(native_used)is int and 10<=native_used<50 and native_used%5==0,'Invalid remaining native-call budget')
    return min(5,(50-native_used)//5)


@torch.inference_mode()
def run_case(model,encoder,source,task,row,protocol,trace_path):
    from gymnasium.spaces import Box
    from stable_worldmodel.solver import CEMSolver
    initial,goal,support=source_case(source,task,row)
    # Goal pixels are re-rendered under the same task marker, not assumed equal
    # to an HDF5 frame after the upstream initializer's physics tick.
    env=environment(task,173+row['episode_id']);goal_env=environment(task,173+row['episode_id'])
    try:
        set_goal(env,task,goal);restore(env,task,initial)
        goal_init={'state':goal} if task=='pusht' else {'qpos':goal,'qvel':np.zeros_like(goal)}
        set_goal(goal_env,task,goal);restore(goal_env,task,goal_init)
        if task=='pusht':
            # The goal is a desired pose, not a simulated successor. Undo the
            # initializer's physics advance for this visualization-only body.
            goal_env.block.angle=float(goal[4]);goal_env.block.position=tuple(goal[2:4])
            goal_env.agent.position=tuple(goal[:2]);goal_env.agent.velocity=(0.,0.)
            goal_env.block.velocity=(0.,0.);goal_env.block.angular_velocity=0.
            goal_env.space.reindex_shapes_for_body(goal_env.block)
            goal_env.space.reindex_shapes_for_body(goal_env.agent)
            require(np.allclose(goal_env._get_obs()[:5],goal[:5],rtol=0,atol=1e-8),'Goal rendering pose differs')
        goal_rgb=render(goal_env,task)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
        frames=[render(env,task)];states=[full_state(env,task)];scores=[score(env,task,goal)];actions=[]
        feature_history=[torch.from_numpy(encoder(np.stack(frames),64)).cuda()]
        goal_features=torch.from_numpy(encoder(goal_rgb[None],64)).cuda()
        past=[];first_success=0 if scores[0]['success'] else None
        support_success=first_success is not None;solve_times=[];model_calls=0
        def execute(native):
            nonlocal first_success
            env.step(np.asarray(native,dtype=np.float32));actions.append(np.asarray(native,dtype=np.float32))
            frames.append(render(env,task));states.append(full_state(env,task));scores.append(score(env,task,goal))
            if scores[-1]['success'] and first_success is None:first_success=len(actions)
        if first_success is None:
            for block in support.reshape(2,5,2):
                for action in block:
                    execute(action)
                    if first_success is not None:break
                if first_success is not None:break
                past.append(torch.from_numpy(block.reshape(1,10)).cuda())
                feature_history.append(torch.from_numpy(encoder(frames[-1][None],64)).cuda())
            support_success=first_success is not None
        cost=SpatialGoalCost(model).cuda()
        solver=CEMSolver(model=cost,batch_size=1,num_samples=protocol['samples'],n_steps=protocol['iterations'],
                         topk=protocol['elites'],device='cuda',seed=protocol['cem_seed']+row['episode_id'])
        solver.configure(action_space=Box(-1,1,shape=(1,2),dtype=np.float32),n_envs=1,
                         config=SimpleNamespace(horizon=5,action_block=5))
        # Execute five planned blocks per replan, then the remaining three.
        # Actual observed RGB/history is refreshed after every complete block.
        while len(actions)<50 and first_success is None:
            require(len(feature_history)>=3 and len(past)>=2,'Insufficient real observed support')
            horizon=planning_horizon(len(actions))
            solver.configure(action_space=Box(-1,1,shape=(1,2),dtype=np.float32),n_envs=1,
                             config=SimpleNamespace(horizon=horizon,action_block=5))
            info={'history_features':torch.stack(feature_history[-3:],1),
                  'past_actions':torch.stack(past[-2:],1),'goal_features':goal_features}
            torch.cuda.synchronize();start=time.perf_counter();solution=solver.solve(info)
            torch.cuda.synchronize();solve_times.append(time.perf_counter()-start)
            plan=cost.to_native(solution['actions'][0].cuda()).reshape(horizon,5,2).cpu().numpy()
            model_calls+=1
            for block in plan:
                for action in block:
                    execute(action)
                    if first_success is not None:break
                if first_success is not None:break
                past.append(torch.from_numpy(block.reshape(1,10)).cuda())
                feature_history.append(torch.from_numpy(encoder(frames[-1][None],64)).cuda())
        torch.cuda.synchronize();wall=time.perf_counter()-began
        trace_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=trace_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,images=np.asarray(frames),states=np.asarray(states),
                actions=np.asarray(actions,dtype=np.float32).reshape(-1,2),goal_image=goal_rgb,goal_state=goal,
                success=np.array([x['success'] for x in scores]),support_actions=support)
        os.replace(temporary,trace_path)
        return {'episode_id':row['episode_id'],'planning_start':row['planning_start'],'goal_native_offset':35,
                'success':first_success is not None,'first_success_native_call':first_success,
                'support_only_success':support_success,'policy_eligible':not support_success,
                'native_calls':len(actions),'native_call_budget':50,'replans':model_calls,
                'candidate_evaluations':cost.candidate_evaluations,'planning_wall_seconds':wall,
                'cost_scope':'online wall time includes observed/goal encoding, context inference, CEM, rendering, simulation and scoring; excludes environment construction and trace serialization',
                'solve_seconds':solve_times,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                'initial_score':scores[0],'final_score':scores[-1],'native_scores':scores,
                'maximum_block_iou':max(s['block_iou'] for s in scores) if task=='pusht' else None,
                'source_support_actions_sha256':array_sha(support),'initial_render_sha256':array_sha(frames[0]),
                'goal_render_sha256':array_sha(goal_rgb),'trace_file':str(trace_path.relative_to(ROOT)),
                'trace_sha256':sha(trace_path),'initialization_note':'unchanged upstream restore; actual post-restore RGB encoded; no assumption of byte-equality with HDF5'}
    finally:
        env.close();goal_env.close()


def evaluate(row,split='test'):
    reg=checked_registration();require(split in ('development','test'),'Unknown planning split')
    selected=all_training_complete(reg) if split=='test' else None
    torch.set_num_threads(8);require(torch.cuda.is_available(),'CUDA allocation required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    install_simulator_runtime()
    from shiftwm.real_video_iws.features import DinoSpatialEncoder
    import h5py
    import hdf5plugin  # noqa: F401
    model,state=load_package(ROOT/row['output']/'best','cuda');model.requires_grad_(False)
    encoder=DinoSpatialEncoder(ROOT,ROOT/'data/pretrained/dinov2-small','cuda')
    directory=REPORT/'evaluations'/split;directory.mkdir(parents=True,exist_ok=True)
    target=directory/(row['name']+'.json')
    signature=digest({'registration':sha(REG),'checkpoint':sha(ROOT/row['output']/'best/model.pt'),'split':split})
    with target.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        records=[]
        if target.exists():
            previous=read(target);require(previous['signature']==signature,'Evaluation identity changed')
            records=previous['records']
            require(len({r['episode_id'] for r in records})==len(records),'Duplicate existing cases')
            for r in records:require(sha(ROOT/r['trace_file'])==r['trace_sha256'],'Committed trace changed')
        done={r['episode_id'] for r in records};cases=reg['datasets'][row['task']]['partitions'][split]
        require(done<={c['episode_id']for c in cases},'Unknown evaluated episode')
        allocation_start=time.monotonic()
        with h5py.File(ROOT/DATA[row['task']],'r') as source:
            for case in cases:
                if case['episode_id'] in done:continue
                trace=ROOT/'artifacts/planning/current_spatial_v1'/split/row['name']/f'{case["episode_id"]:06d}.npz'
                records.append(run_case(model,encoder,source,row['task'],case,reg['planning'],trace))
                result={'schema':'current_spatial_planning_evaluation_v1','status':'complete'if len(records)==len(cases) else'running',
                        'signature':signature,'registration_sha256':sha(REG),'run':row,'split':split,
                        'selected_epoch':state['epoch'],'checkpoint_sha256':sha(ROOT/row['output']/'best/model.pt'),
                        'all_training_complete_selection':selected,'records':records}
                atomic_json(result,target)
                print(json.dumps({'run':row['name'],'planning_cases':len(records),'required':len(cases),'native_calls':records[-1]['native_calls']}),flush=True)
                if len(records)<len(cases) and time.monotonic()-allocation_start>=5400:
                    return result
        return read(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=int,required=True);p.add_argument('--split',choices=('development','test'),default='test')
    a=p.parse_args();evaluate(select_run(checked_registration(),index=a.index),a.split)
