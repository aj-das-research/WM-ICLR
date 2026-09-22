from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/metrics_completion_v1'))
from rgb_core import module,configure_cpu,atomic_json
configure_cpu(); ev=module(ROOT/'scripts/real_video_iws_reserved_recovery_v2/evaluate.py','_rgb_replay_diagnostic')
reg=ev.checked_registration(ROOT)
row=next(r for r in reg['runs']if r['name']=='bimanual_rope_bounded_spatial_mix_s2')
from shiftwm.real_video_iws_reserved.cache import ReservedFeatureCache
cache=ReservedFeatureCache(ROOT,row['task'],registration_path=reg['cache_registration_path'])
model,_=ev.load_selected(ROOT,row);store={};results=[]
with torch.inference_mode():
 for repeat in range(2):
  errors=[]
  for i in range(0,200,64):
   b=ev.make_batch(cache,cache.handles[i:i+64],row['task'],cache.episode_ids,store)
   p=model.predict(b['initial_features'],b['commands'])
   errors.append(ev.feature_errors(p,b['targets'],model.feature_std)['standardized_mse'][:,-1].double().numpy())
  results.append(np.concatenate(errors))
with np.load(ROOT/'reports/real_video_iws_reserved_recovery_v2/evaluations'/f"{row['name']}.npz")as f:original=f['standardized_mse'][:,-1]
result={'scope':'numerical replay only; no RGB accuracy accessed','row':row['name'],'repeat_exact':bool(np.array_equal(*results)),
 'original_max_absolute_difference':float(abs(results[0]-original).max()),'original_max_relative_difference':float((abs(results[0]-original)/np.maximum(abs(original),1e-30)).max()),
 'different_windows':int((results[0]!=original).sum()),'original':original.tolist(),'replay':results[0].tolist()}
atomic_json(result,ROOT/'reports/iclr_review_2026-09-22/rgb_replay_diagnostic.json');print(json.dumps({k:v for k,v in result.items()if k not in ['original','replay']}))
