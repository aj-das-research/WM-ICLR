"""Complete-three decoder gate; no forecast RGB scores are invented or implied."""
import argparse
import json
import torch
from rgb_core import REPORT, REG, TASKS, RECIPE, sha, read, require, atomic_json, checked_registration


def finalize(if_ready=False):
    reg=checked_registration()
    directories=[REPORT/'runs'/f'{task}_s173' for task in TASKS]
    missing=[task for task,path in zip(TASKS,directories) if not (path/'completion.json').exists()]
    if missing:
        require(if_ready,'Incomplete three-decoder grid')
        return {'status':'pending','missing_tasks':missing}
    rows=[];sources={}
    for task,path in zip(TASKS,directories):
        complete=read(path/'completion.json');summary=read(path/'training_summary.json')
        require(complete['status']==summary['status']=='completed' and complete['epochs']==summary['epochs']==30,'Incomplete decoder epochs')
        require(complete['summary_sha256']==sha(path/'training_summary.json') and complete['selected_sha256']==summary['selected_sha256']==sha(path/'selected.pt'),'Decoder completion binding changed')
        identity=summary['identity']
        require(identity==complete['identity'] and identity['registration_sha256']==sha(REG) and identity['task']==task and identity['seed']==173 and identity['recipe']==RECIPE,'Wrong decoder identity')
        require(identity['statistics_sha256']==reg['tasks'][task]['statistics_sha256'],'Wrong decoder statistics')
        require(identity['target_manifest_sha256']==sha(REPORT/'targets'/task/'manifest.json'),'Wrong RGB cache')
        history=summary['history'];require([r['epoch'] for r in history]==list(range(1,31)),'Missing/duplicate decoder epochs')
        for row in history:
            require(row['train_frames']==reg['tasks'][task]['counts']['internal_train']['frames'] and row['development_frames']==reg['tasks'][task]['counts']['internal_development']['frames'],'Incomplete decoder frame population')
            expected=[r for r in reg['tasks'][task]['records'] if r['split']=='internal_development']
            actual=row['validation']['episodes']
            require([r['episode_id'] for r in actual]==[r['episode_id'] for r in expected] and [r['frames'] for r in actual]==[len(r['selected_indices']) for r in expected],'Decoder validation trajectory population changed')
            score=sum(r['unclipped_mse'] for r in actual)/len(actual)
            require(abs(score-row['validation']['equal_trajectory_unclipped_mse'])<1e-12,'Reconstruction aggregation mismatch')
        selected=min(history,key=lambda r:r['validation']['equal_trajectory_unclipped_mse'])
        require(summary['selected_epoch']==complete['selected_epoch']==selected['epoch'] and summary['reconstruction_quality']==selected['validation'],'Decoder checkpoint selector mismatch')
        package=torch.load(path/'selected.pt',map_location='cpu',weights_only=False)
        require(package['kind']=='iws_shared_rgb_decoder_v1' and package['identity']==identity and package['epochs']==30
                and package['selected_epoch']==selected['epoch'] and package['validation']==selected['validation'], 'Selected decoder package metadata changed')
        require(package['model'] and all(torch.is_tensor(v) and torch.isfinite(v).all() for v in package['model'].values()), 'Invalid selected decoder state')
        for name in ('completion.json','training_summary.json','selected.pt'):
            sources[str((path/name).relative_to(REPORT))]=sha(path/name)
        rows.append({'task':task,'decoder_seed':173,'selected_epoch':selected['epoch'],
                     'gt_feature_reconstruction':summary['reconstruction_quality'],'checkpoint_sha256':summary['selected_sha256']})
    result={'schema':'iws_shared_rgb_decoder_completion_v1','status':'three_decoders_complete',
            'registration_sha256':sha(REG),'sources':sources,'tasks':rows,
            'scope':RECIPE['scope'],'uncertainty':RECIPE['uncertainty'],
            'forecast_metrics':{'status':'not_run','reason':'separate common-decoder forecast evaluator and its frozen source review required'},
            'LPIPS':{'status':'not_measured','prerequisite':'official VGG backbone and LPIPS v0.1 calibrated weights/code pinned before metric execution'},
            'FID_FVD':{'status':'not_measured','prerequisite':'separate frozen official weights, feature preprocessing, sample/clip and finite-sample protocols; no proxy'}}
    target=REPORT/'decoder_finalization.json'
    if target.exists():require(read(target)==result,'Refusing changed decoder finalization')
    else:atomic_json(result,target)
    return {'status':result['status'],'finalization_sha256':sha(target)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--if-ready',action='store_true');args=parser.parse_args()
    print(json.dumps(finalize(args.if_ready),sort_keys=True))
