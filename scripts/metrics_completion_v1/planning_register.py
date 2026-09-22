"""Freeze an outcome-independent metadata split and complete new source closure."""
import argparse
import json
from planning_common import (DATA,HERE,MODES,REG,REPORT,ROOT,RUNTIME,SCHEMA,TASKS,atomic_json,checked_registration,require,sha)
from planning_data import metadata_inventory


def dependencies():
    paths=list(HERE.glob('planning_*.py'))+list(HERE.glob('planning_*.slurm'))+[HERE/'test_planning.py',HERE/'planning_README.md']
    paths += [ROOT/p for p in ('scripts/real_video/train.py','src/shiftwm/real_video_spatial/model.py',
        'src/shiftwm/real_video_spatial_components/model.py','src/shiftwm/real_video_iws/features.py',
        'src/shiftwm/real_video_iws/data.py','src/shiftwm/real_video/data.py','src/shiftwm/model.py',
        'src/shiftwm/upstream.py','src/shiftwm/checkpoint.py','src/shiftwm/vendor/lewm/module.py',
        'src/shiftwm/vendor/lewm/NOTICE.json','references/real_dinov2_sources.json',
        'data/pretrained/dinov2-small/provenance.json','data/pretrained/dinov2-small/model.safetensors',
        'data/pretrained/dinov2-small/config.json','data/pretrained/dinov2-small/preprocessor_config.json')]
    paths += list((RUNTIME/'stable_worldmodel').rglob('*.py'))+[RUNTIME/'.source_revision']
    require(all(p.is_file() for p in paths),'Missing expected source dependency')
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def freeze():
    require(not REG.exists(),'Registration already exists; no in-place scientific revision')
    deps=dependencies();datasets={task:metadata_inventory(task) for task in TASKS}
    # The split is already fixed without decoding payload arrays. Hash the
    # acquired HDF5 bytes only for provenance; never inspect reserved values.
    atomic_json({'selection_complete':True,'datasets':datasets},REPORT/'metadata_selection.json')
    for task in TASKS:
        print(json.dumps({'event':'hashing_acquired_source_bytes','task':task,'decoded_payload_arrays':False}),flush=True)
        datasets[task]['hdf5_sha256']=sha(ROOT/DATA[task])
    runs=[{'name':f'{task}_{mode}_s{seed}','task':task,'mode':mode,'seed':seed,
           'output':f'runs/current_spatial_planning_v1/{task}_{mode}_s{seed}'}
          for seed in range(3) for task in TASKS for mode in MODES]
    document={'schema':SCHEMA,'status':'frozen_before_training_and_test_access','methods':list(MODES),'seeds':[0,1,2],
       'dependencies':deps,'datasets':datasets,'runs':runs,
       'model_config':{'feature_dim':6144,'channels':384,'grid_size':4,'action_dim':10,'hidden_dim':96,
           'history_length':3,'depth':4,'context_dim':32,'context_hidden':128,'innovation_bound':1.,'identity_bias':4.,'initial_gate_logit':-3.},
       'training':{'epochs':30,'batch_size':128,'lr':1e-4,'min_lr':1e-6,'weight_decay':.01,'grad_clip':1.,'bf16':True},
       'planning':{'maximum_horizon':5,'remaining_budget_horizon_rule':'min(5, remaining_native_calls//5)',
           'solve_horizons_if_no_early_success':[5,3],'action_block':5,'support_native_calls':10,'native_budget':50,'receding_horizon':5,
           'samples':128,'iterations':8,'elites':16,'cem_seed':1701,'goal_native_offset_from_start':35,
           'goal_native_offset_from_observed_anchor':25,'native_action_bounds':[-1.,1.],
           'search_space':'train-only action z scores, transformed then clipped in native coordinates',
           'comparison_to_official':'matched reduced-compute CEM128x8/top16; not official LeWM300x30/top30 results'},
       'selection':'all5 forecast standardized MSE, equal development episode means; never planning/test outcomes',
       'test_gate':'all24 full30epoch packages and selected-weight identities validated before any test RGB/action/state reads',
       'metric_contract':{'primary':'any-native-call upstream task success, with initial/support-only successes separately identified',
           'pusht_success':'norm(concatenated pusherXY+blockXY difference)<20 and circular angle difference<pi/9',
           'reacher_success':'both unwrapped joint absolute errors<0.05rad',
           'secondary':['endpoint success','success among policy-eligible cases','actual native calls','failure-penalized calls50',
                        'physical endpoint errors','PushT final/max block polygon IoU','end-to-end online wall time','GPU peak allocated bytes','candidate evaluations'],
           'uncertainty':'paired seed+episode crossed percentile bootstrap; 10000 draws seed173; retain every failure'},
       'resource_profile':{'training_inputs_only':True,'batch_size':128,'warmup_steps':3,'measured_steps':5,'full_horizon':5},
       'scope':'new current spatial architecture closed-loop simulator study; no claim that historical context-model results belong to this method',
       'metadata_payload_reads':'selection uses only episode offsets/lengths/IDs/native indices; complete HDF5 bytes subsequently SHA256-hashed for provenance only; no RGB/actions/physical states/scores decoded'}
    atomic_json(document,REG)
    checked_registration(require_review=False)
    print(json.dumps({'registration':str(REG.relative_to(ROOT)),'sha256':sha(REG),'runs':24,
                      'source_bindings':len(deps),'counts':{t:{p:len(r)for p,r in d['partitions'].items()}for t,d in datasets.items()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true',required=True);p.parse_args();freeze()
