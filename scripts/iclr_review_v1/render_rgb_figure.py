"""Plot all learned RGB endpoint comparisons against the same AR reference."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'paper/generated/iclr_review_v1'
p=ROOT/'reports/iclr_review_2026-09-22/rgb_endpoint_v2/finalization.json';d=json.loads(p.read_text());assert d['status']=='complete'
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(5.5,2.7))
fig.subplots_adjust(left=.12,right=.985,bottom=.25,top=.85,wspace=.38)
for ax,metric,scale,label in zip(axes,['rgb_mse','lpips_vgg'],[1000,1000],['RGB MSE difference × 1,000','LPIPS difference × 1,000']):
 for i,task in enumerate(['pusht','bimanual_box','bimanual_rope']):
  for delta,mode,color in [(-.12,'bounded_spatial_mix','#255E91'),(.12,'unbounded_spatial_mix','#82517E')]:
   v=d['paired_differences'][task][mode+'_minus_autoregressive'][metric];m=v['mean']*scale;lo,hi=np.array(v['ci95'])*scale
   ax.errorbar(m,i+delta,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=color,ms=4,label=('Bounded mixing'if delta<0 else'Unbounded mixing')if i==0 else None)
 ax.axvline(0,color='.5',ls='--',lw=.8);ax.set_yticks(range(3),['PushT','Box','Rope']);ax.set_ylim(2.45,-.65);ax.set_xlabel(label+'\n← lower than AR');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',lw=.5,color='.9')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,frameon=False,loc='upper center',ncol=2,fontsize=8)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'rgb_metric_tradeoffs.{ext}',dpi=240)
plt.close(fig)
(OUT/'rgb_metric_figure.tex').write_text(r'''\begin{figure}[tbp]\centering
\includegraphics[width=\linewidth]{generated/iclr_review_v1/rgb_metric_tradeoffs.pdf}
\caption{Decoded endpoint conclusions depend on the metric. Points are named
mixing variant minus autoregression, with post-hoc paired trajectory--seed 95\%
bootstrap intervals, conditional on one fixed decoder per task. Negative favors
mixing. Unbounded mixing improves LPIPS relative to AR in all three tasks, but
its Rope RGB MSE is worse. Bounded mixing has worse RGB MSE in all tasks. Raw RGB
persistence has lower LPIPS than every decoded predictor and is retained in
Table~\ref{tab:rgb-endpoint-comparison}; these panels are not an all-baseline
superiority claim.}
\label{fig:rgb-metric-tradeoffs}\end{figure}
''')
(OUT/'rgb_figure_provenance.json').write_text(json.dumps({'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'all_paired_learned_vs_ar_mse_lpips_contrasts':True,'figure_width_inches':5.5},indent=2)+'\n')
