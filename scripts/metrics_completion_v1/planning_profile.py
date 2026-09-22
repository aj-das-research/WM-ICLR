"""Training-input-only full-shape resource profile, separate from study outcomes."""
import argparse
import json
import socket
import time
import numpy as np
import torch
from planning_common import CACHE, REG, REPORT, atomic_json, checked_registration, read, require, sha
from planning_data import PlanningDataset, load_cache
from planning_train import make_model, base


def profile(task):
    reg=checked_registration();torch.set_num_threads(8)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'Allocated BF16 CUDA required')
    _,episodes=load_cache(task,'train',sha(REG));dataset=PlanningDataset(episodes)
    b=reg['training']['batch_size']
    rows=[dataset[i] for i in range(b)]
    batch={k:torch.stack([r[k] for r in rows]).cuda() for k in ('features','actions')}
    stats=read(CACHE/task/'training_statistics.json');result=[]
    for mode in reg['methods']:
        base.seed_everything(0);model=make_model({**reg['model_config'],'mode':mode},stats).cuda().train()
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=reg['training']['lr'],weight_decay=.01)
        timings=[];torch.cuda.reset_peak_memory_stats()
        for i in range(8):
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();began=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(batch)['loss']
            require(torch.isfinite(loss),'Nonfinite profile loss');loss.backward()
            norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
            require(torch.isfinite(norm),'Nonfinite profile gradient');optimizer.step();torch.cuda.synchronize()
            if i>=3:timings.append(time.perf_counter()-began)
        result.append({'mode':mode,'full_shape_training_step_seconds':timings,'median_seconds':float(np.median(timings)),
                       'train_windows':len(dataset),'estimated_training_only_30_epoch_seconds':float(np.median(timings))*int(np.ceil(len(dataset)/b))*30,
                       'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                       'parameters':{'total':sum(p.numel()for p in model.parameters()),'trainable':sum(p.numel()for p in model.parameters()if p.requires_grad)}})
        del model,optimizer;torch.cuda.empty_cache()
    out={'schema':'current_spatial_training_resource_profile_v1','status':'passed','registration_sha256':sha(REG),'task':task,
         'scope':'disposable training-only first B128 windows,3 warmup/5 measured optimization steps; estimates exclude validation,IO,planning',
         'dataset_payload_scope':'train only; no development/test arrays read','batch_size':b,'gpu':torch.cuda.get_device_name(),'hostname':socket.gethostname(),'rows':result}
    atomic_json(out,REPORT/f'{task}_profile.json');return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=('pusht','reacher'),required=True)
    a=p.parse_args();print(json.dumps(profile(a.task)),flush=True)
