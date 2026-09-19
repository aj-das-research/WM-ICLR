#!/usr/bin/env python3
"""Inspect prespecified native-length examples and all authorized command shapes."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from shiftwm.real_video_iws_tasks.data import InternalInventory,TASK_WIDTHS,sha,load_commands,decode_native_rgb,require
from shiftwm.real_video_iws.features import DinoSpatialEncoder
from shiftwm.real_video_iws.cache import atomic_json


def inspect(root=ROOT):
    torch.set_num_threads(4)
    results={};encoder=DinoSpatialEncoder(root,root/'data/pretrained/dinov2-small','cpu')
    for task,width in TASK_WIDTHS.items():
        inventory=InternalInventory(root,task);shapes={};command_rows=0
        for eid in inventory.selected():
            values=load_commands(inventory,eid)
            require(values.shape[1]==width,'Task command width differs')
            shapes[str(len(values))]=shapes.get(str(len(values)),0)+1;command_rows+=len(values)
        # First internal-training identity at each observed training-native length.
        examples={}
        for eid in inventory.selected('internal_train'):
            examples.setdefault(inventory.rows[eid]['shapes']['target_qpos'][0],eid)
        decoded=[]
        for length,eid in sorted(examples.items()):
            row,split,metadata,video=inventory.paths(eid,'internal_train')
            frames=decode_native_rgb(inventory,eid,sha(video),'internal_train')
            item={'episode_id':eid,'split':split,'frames':len(frames),'rgb_shape':list(frames.shape),
                  'command_shape':list(load_commands(inventory,eid,'internal_train').shape),
                  'source_video_sha256':sha(video),'source_metadata_sha256':sha(metadata),
                  'decoded_rgb_sha256':hashlib.sha256(frames.tobytes()).hexdigest()}
            if eid==inventory.selected('internal_train')[0]:
                images=frames[[0,-1]];actual=encoder(images,32)
                # Independent explicit formulation of the pinned spatial encoder recipe.
                pixels=torch.from_numpy(images.copy()).permute(0,3,1,2).float()/255
                pixels=F.interpolate(pixels,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
                pixels=(pixels-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
                with torch.inference_mode():
                    tokens=encoder.model(pixel_values=pixels).last_hidden_state[:,1:]
                    expected=F.adaptive_avg_pool2d(tokens.float().transpose(1,2).reshape(-1,384,16,16),(4,4)).flatten(1).numpy()
                require(np.array_equal(actual,expected),'Frozen encoder extraction parity differs')
                item.update(first_last_feature_shape=list(actual.shape),independent_cpu_recipe_max_abs=float(np.max(np.abs(actual-expected))))
            decoded.append(item)
        results[task]={'episodes':len(inventory.assignment),'split_counts':{s:len(inventory.selected(s)) for s in ('internal_train','internal_development')},
            'command_width':width,'command_rows':command_rows,'native_length_counts':shapes,'decoded_examples':decoded,
            'selection':'First sorted internal-training identity for each distinct native training length; first/last frame CPU encoder check for first training identity.'}
    sources=['src/shiftwm/real_video_iws_tasks/data.py','src/shiftwm/real_video_iws_tasks/cache.py',
        'scripts/real_video_iws_tasks/prepare_cache.py','scripts/real_video_iws_tasks/inspect_training.py',
        'src/shiftwm/real_video_iws/data.py','src/shiftwm/real_video_iws/cache.py','src/shiftwm/real_video_iws/features.py',
        'configs/real_video_iws/split_v1.json']
    report={'status':'passed_actual_training_data_and_cpu_encoder_preflight','utc':datetime.now(timezone.utc).isoformat(),
        'tasks':results,'source_sha256':{p:sha(root/p) for p in sources},'official_validation_payloads_opened':0,
        'scope':'Input compatibility only; no predictor training, evaluation window definition, physical time/unit inference or benchmark score.'}
    atomic_json(report,root/'reports/evidence/iws_box_rope_cache_actual_preflight.json')
    print(json.dumps(report,indent=2))

if __name__=='__main__':inspect()
