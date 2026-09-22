"""Validate all45 RGB endpoint rows, paired trajectory/seed inference and reports."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'reports/iclr_review_2026-09-22/rgb_endpoint_v2';OUT=ROOT/'paper/generated/iclr_review_v1'
TASKS=['pusht','bimanual_box','bimanual_rope'];MODES=['autoregressive','anchored_additive','bounded_spatial_mix','unbounded_spatial_mix'];REFS=['rgb_persistence','feature_persistence','gt_feature_reconstruction']
LABELS={'autoregressive':'Autoregressive','anchored_additive':'Additive anchor','bounded_spatial_mix':'Bounded ShiftWM','unbounded_spatial_mix':'Unbounded mixing','rgb_persistence':'RGB persistence','feature_persistence':'Feature persistence','gt_feature_reconstruction':'Target-feature reconstruction'}
METRICS=['rgb_mse','psnr_db','ssim','lpips_vgg','uiqi']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def ci(x):
 rng=np.random.default_rng(173);s=rng.integers(3,size=(10000,3));e=rng.integers(10,size=(10000,10));v=x[s[:,:,None],e[:,None,:]].mean((1,2));return np.quantile(v,[.025,.975]).tolist()
def main():
 reg=read(BASE/'registration.json');sources={str((BASE/'registration.json').relative_to(ROOT)):sha(BASE/'registration.json')};summary={};contrasts={}
 for task in TASKS:
  path=BASE/(task+'.json');doc=read(path);assert doc['status']=='complete'and doc['registration_sha256']==sha(BASE/'registration.json')and len(doc['rows'])==15
  sources[str(path.relative_to(ROOT))]=sha(path);expected={f'{task}_{m}_s{s}'for m in MODES for s in range(3)}|set(REFS)
  assert {r['name']for r in doc['rows']}==expected
  matrices={};population=None
  for row in doc['rows']:
   p=BASE/task/(row['name']+'.npz');assert sha(p)==row['npz_sha256'];sources[str(p.relative_to(ROOT))]=sha(p)
   with np.load(p,allow_pickle=False)as z:
    values=z['metrics'];ids=z['episode_index'];starts=z['window_start'];assert values.shape==(200,5)and np.isfinite(values).all()and set(ids)==set(range(10))
    identity=np.stack([ids,starts],1)
    if population is None:population=identity
    else:np.testing.assert_array_equal(population,identity)
    assert len(set(map(tuple,identity)))==200
    per=np.stack([values[ids==i].mean(0)for i in range(10)]);np.testing.assert_allclose(per,row['trajectory_metrics'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(per.mean(0),row['equal_trajectory_mean'],rtol=1e-12,atol=1e-12);matrices[row['name']]=per
  means={}
  for mode in MODES+REFS:
   x=np.stack([matrices[f'{task}_{mode}_s{s}']for s in range(3)])if mode in MODES else np.repeat(matrices[mode][None],3,axis=0)
   means[mode]=x
  summary[task]={m:{metric:{'mean':float(x[:,:,j].mean()),'ci95':ci(x[:,:,j])}for j,metric in enumerate(METRICS)}for m,x in means.items()}
  contrasts[task]={}
  for a,b in [('bounded_spatial_mix','autoregressive'),('unbounded_spatial_mix','autoregressive'),('unbounded_spatial_mix','bounded_spatial_mix')]:
   diff=means[a]-means[b];contrasts[task][a+'_minus_'+b]={metric:{'mean':float(diff[:,:,j].mean()),'ci95':ci(diff[:,:,j])}for j,metric in enumerate(METRICS)}
 result={'status':'complete','scope':reg['scope'],'rows':45,'scored_endpoints':9000,'source_sha256':sources,'summary':summary,'paired_differences':contrasts,
  'uncertainty':'10000 paired crossed trajectory/predictor-seed bootstrap draws; seed173; conditional on one fixed decoder; posthoc unadjusted',
  'numerical_revision':reg['numerical_revision']}
 (BASE/'finalization.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=[r'\begin{table}[tbp]\centering\begingroup\fontsize{8.5}{10.2}\selectfont',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}lrrrrr@{}}\toprule',r'Method & RGB MSE $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & UIQI $\uparrow$ \\ \midrule']
 for task,title in zip(TASKS,['PushT','Box','Rope']):
  lines.append(r'\multicolumn{6}{@{}l}{\textbf{'+title+r'}} \\')
  best={metric:(min if metric in ('rgb_mse','lpips_vgg') else max)(summary[task][m][metric]['mean']for m in MODES+REFS if m!='gt_feature_reconstruction')for metric in METRICS}
  for m in REFS+MODES:
   cells=[]
   for metric in METRICS:
    value=summary[task][m][metric]['mean'];cell=f'{value:.6f}'if metric=='rgb_mse'else f'{value:.4f}'
    if m!='gt_feature_reconstruction'and value==best[metric]:cell=r'\textbf{'+cell+'}'
    cells.append(cell)
   lines.append(LABELS[m]+' & '+' & '.join(cells)+r' \\')
  lines.append(r'\addlinespace[2pt]')
 lines.extend([r'\bottomrule\end{tabular}\endgroup',r'\caption{Decoded RGB endpoints on the same 600 reserved IWS handles. All methods use the same task-specific decoder selected on development frames; target-feature reconstruction diagnoses readout error and is not a guaranteed lower bound. RGB persistence copies observed pixels, while feature persistence decodes the observed features. Scores average handles within each of ten trajectories per task, then trajectories and three predictor seeds equally. Images are quantized resized 224$\times$224 RGB in [0,1]; predictions are clamped. PSNR is in dB, SSIM uses 11$\times$11 Gaussian windows, UIQI uses 8$\times$8 uniform windows, and LPIPS uses the official VGG calibration v0.1. This secondary endpoint study uses stored offset 59, not the original IWS 192-step protocol or a reproduction of published scores. Bold marks the best forecasting point mean, excluding target-feature reconstruction. Full conventions and paired intervals accompany the source ledger.}',r'\label{tab:rgb-endpoint-comparison}\end{table}'])
 (OUT/'rgb_endpoint_table.tex').write_text('\n'.join(lines)+'\n')
 # Paired differences retain negative results, units and uncertainty.
 lines=[r'\begin{table}[tbp]\centering\small',r'\begin{tabular}{@{}llrr@{}}\toprule',r'Task & Comparison & RGB MSE difference & LPIPS difference \\ \midrule']
 for task,title in zip(TASKS,['pusht','box','rope']):
  for key,label in [('bounded_spatial_mix_minus_autoregressive','Bounded vs. AR'),('unbounded_spatial_mix_minus_autoregressive','Unbounded vs. AR'),('unbounded_spatial_mix_minus_bounded_spatial_mix','Unbounded vs. bounded')]:
   cells=[]
   for metric in ['rgb_mse','lpips_vgg']:
    d=contrasts[task][key][metric];lo,hi=d['ci95'];cells.append(f"${d['mean']:+.5f}$ [${lo:+.5f}$, ${hi:+.5f}$]")
   lines.append(title+' & '+label+' & '+' & '.join(cells)+r' \\')
 lines.extend([r'\bottomrule\end{tabular}',r'\caption{Post-hoc paired RGB endpoint differences: named first minus reference; negative favors the first model. Brackets are unadjusted 95\% trajectory--seed bootstrap intervals, conditional on one frozen decoder per task. The complete ledger includes the other three metric contrasts.}',r'\label{tab:rgb-endpoint-paired}\end{table}'])
 (OUT/'rgb_endpoint_paired.tex').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'complete','rows':45,'endpoint_images_scored':9000}))
if __name__=='__main__':main()
