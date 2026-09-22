"""Frozen development-only candidate ranking experiment for all 24 spatial models."""
from pathlib import Path
import sys, json, os, time, socket, argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/metrics_completion_v1'))
from planning_common import checked_registration, sha, atomic_json, ROOT, REG, DATA, require, install_simulator_runtime
OUT=ROOT/'reports/iclr_review_2026-09-22/action_ranking'
REGISTRATION=OUT/'registration.json'
MODES=('autoregressive','bounded_additive','transport','unbounded_transport')

def register():
    original=checked_registration(); cases={}
    for task in ('pusht','reacher'):
        population=sorted(original['datasets'][task]['partitions']['development'],key=lambda r:r['episode_id'])
        cases[task]=[population[i]for i in np.linspace(0,99,8,dtype=int)]
    document={'schema':'spatial_development_action_ranking_v1','scope':'post-hoc development diagnostic motivated by revealed test failure; not new held-out evidence',
      'planning_registration_sha256':sha(REG),'cases':cases,'seeds':[0,1,2],'modes':list(MODES),
      'candidates':33,'horizon_native':25,'support_native':10,'random_seed':173,
      'candidate_rule':'32 iid uniform native [-1,1] sequences, then one zero sequence; same candidates for every model/seed; no CEM or demonstration oracle',
      'outcomes':'terminal joint physical goal distance, native success, and actual standardized encoder goal cost; fixed horizon without success stopping',
      'metrics':'Spearman prediction-cost versus physical-distance and versus actual-feature-cost; physical regret of minimum predicted-cost candidate; rollout feature MSE',
      'aggregation':'equal cases then seeds; descriptive development means; no test reselection',
      'models':{r['name']:{'directory':r['output']+'/best','sha256':sha(ROOT/r['output']/'best/model.pt')}for r in original['runs']},
      'sources':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                 **{str(Path('scripts/metrics_completion_v1')/n):sha(ROOT/'scripts/metrics_completion_v1'/n)for n in ['planning_evaluate.py','planning_train.py','planning_common.py']}}}
    OUT.mkdir(parents=True,exist_ok=True)
    if REGISTRATION.exists():require(json.loads(REGISTRATION.read_text())==document,'Existing diagnostic registration differs')
    else:atomic_json(document,REGISTRATION)
    return document

def check():
    doc=json.loads(REGISTRATION.read_text());require(sha(REG)==doc['planning_registration_sha256'],'Original registration changed')
    checked_registration()
    for name,digest in doc['sources'].items():require(sha(ROOT/name)==digest,'Diagnostic source changed')
    for item in doc['models'].values():require(sha(ROOT/item['directory']/'model.pt')==item['sha256'],'Selected model changed')
    return doc

def rho(x,y):
    from scipy.stats import spearmanr
    if np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(spearmanr(x,y).statistic)

def run(task):
    import torch, h5py, hdf5plugin
    from planning_evaluate import environment, restore, set_goal, render, score, source_case, full_state
    from planning_train import load_package
    from shiftwm.real_video_iws.features import DinoSpatialEncoder
    doc=check();require(os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'GPU scheduler allocation required')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    install_simulator_runtime();encoder=DinoSpatialEncoder(ROOT,ROOT/'data/pretrained/dinov2-small','cuda')
    models={}
    for mode in MODES:
        for seed in range(3):
            model,_=load_package(ROOT/doc['models'][f'{task}_{mode}_s{seed}']['directory'],'cuda')
            models[mode,seed]=model.eval().requires_grad_(False)
    destination=OUT/(task+'.json');records=[]
    if destination.exists():
        previous=json.loads(destination.read_text());require(previous['registration_sha256']==sha(REGISTRATION),'Stale partial result');records=previous['records']
    begun=time.monotonic()
    with h5py.File(ROOT/DATA[task],'r')as source, torch.inference_mode():
        for row in doc['cases'][task]:
            if any(r['episode_id']==row['episode_id']for r in records):continue
            initial,goal,support=source_case(source,task,row)
            rng=np.random.default_rng(173+row['episode_id']);candidates=np.concatenate([rng.uniform(-1,1,(32,25,2)).astype('float32'),np.zeros((1,25,2),np.float32)])
            terminal=[];physical=[];success=[];reference_history=None;reference_state=None
            for actions in candidates:
                env=environment(task,173+row['episode_id'])
                try:
                    set_goal(env,task,goal);restore(env,task,initial)
                    frames=[render(env,task)]
                    for i,a in enumerate(support):
                        env.step(a)
                        if i in (4,9):frames.append(render(env,task))
                    start_state=full_state(env,task)
                    if reference_history is None:reference_history=np.stack(frames);reference_state=start_state
                    else:
                        require(np.array_equal(reference_history,np.stack(frames)) and np.array_equal(reference_state,start_state),'Candidate starts not exactly paired')
                    for a in actions:env.step(a)
                    measured=score(env,task,goal);terminal.append(render(env,task));success.append(measured['success'])
                    physical.append(measured['joint_pusher_block_xy_l2_px'if task=='pusht'else'joint_l2_rad'])
                finally:env.close()
            env=environment(task,173+row['episode_id'])
            try:
                set_goal(env,task,goal);restore(env,task,{'state':goal}if task=='pusht'else{'qpos':goal,'qvel':np.zeros_like(goal)})
                if task=='pusht':
                    env.block.angle=float(goal[4]);env.block.position=tuple(goal[2:4]);env.agent.position=tuple(goal[:2]);env.agent.velocity=(0.,0.);env.block.velocity=(0.,0.);env.block.angular_velocity=0.
                    env.space.reindex_shapes_for_body(env.block);env.space.reindex_shapes_for_body(env.agent)
                    require(np.allclose(env._get_obs()[:5],goal[:5],rtol=0,atol=1e-8),'Wrong exact goal pose')
                goal_rgb=render(env,task)
            finally:env.close()
            history=torch.from_numpy(encoder(reference_history,64)).cuda()
            targets=torch.from_numpy(encoder(np.stack(terminal),64)).cuda()
            goal_features=torch.from_numpy(encoder(goal_rgb[None],64)).cuda()
            query=torch.from_numpy(candidates.reshape(33,5,10)).cuda()
            past=torch.from_numpy(support.reshape(1,2,10)).cuda().expand(33,-1,-1)
            observed=history[None].expand(33,-1,-1)
            physical=np.array(physical);summaries={}
            for (mode,seed),model in models.items():
                prediction=model.predict(observed,past,query)[:,-1]
                predicted=((prediction-goal_features)/model.feature_std).square().mean(-1).cpu().numpy()
                actual=((targets-goal_features)/model.feature_std).square().mean(-1).cpu().numpy()
                forecast=((prediction-targets)/model.feature_std).square().mean(-1).cpu().numpy()
                require(np.isfinite(predicted).all() and np.isfinite(forecast).all(),'Nonfinite diagnostic')
                chosen=int(predicted.argmin());regret=float(physical[chosen]-physical.min())
                summaries[f'{mode}_s{seed}']={'predicted_cost':predicted.tolist(),'actual_feature_cost':actual.tolist(),'forecast_mse':forecast.tolist(),
                   'rho_physical':rho(predicted,physical),'rho_actual_feature':rho(predicted,actual),'selected_candidate':chosen,
                   'selected_physical_regret':regret,'normalized_physical_regret':regret/float(np.ptp(physical))if np.ptp(physical)>0 else None,
                   'forecast_mse_mean':float(forecast.mean()),'predicted_cost_range':float(np.ptp(predicted)),'actual_feature_cost_range':float(np.ptp(actual))}
            records.append({'episode_id':row['episode_id'],'models':summaries,'physical_goal_distance':physical.tolist(),'terminal_success':success,
               'candidate_actions_sha256':__import__('hashlib').sha256(candidates.tobytes()).hexdigest(),
               'support_images_sha256':__import__('hashlib').sha256(reference_history.tobytes()).hexdigest()})
            atomic_json({'status':'complete'if len(records)==8 else'running','registration_sha256':sha(REGISTRATION),'task':task,
                'records':records,'gpu':torch.cuda.get_device_name(),'hostname':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],
                'scope':doc['scope'],'elapsed_seconds_this_invocation':time.monotonic()-begun},destination)
            print(json.dumps({'task':task,'cases':len(records),'total':8}),flush=True)
    check()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--task',choices=['pusht','reacher']);a=p.parse_args()
    if a.register:register();print('registered')
    else:require(a.task is not None,'Task required');run(a.task)
