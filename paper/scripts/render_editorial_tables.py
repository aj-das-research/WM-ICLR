#!/usr/bin/env python3
"""Generate compact main-paper lookup tables from completed, audited ledgers."""
from pathlib import Path
import hashlib,json,math,importlib.util
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/generated/editorial';OUT.mkdir(parents=True,exist_ok=True)
paths=['reports/real_video_spatial/finalization.json','reports/real_video_spatial_components/finalization.json','paper/generated/primary_gain_summary.json']
a,b,c=[json.loads((ROOT/p).read_text()) for p in paths]
assert a['status']=='passed' and b['status']=='passed'
assert a['completed_models']==15 and b['completed_new_models']==6
assert a['epochs_per_model']==30 and b['epochs_per_model']==30
external = None
external_path = ROOT/'paper/generated/external_dinowm'
external_evidence = external_path/'evidence.json'
if external_evidence.exists():
 evidence = json.loads(external_evidence.read_text())
 renderer = ROOT/'paper/scripts/render_external_dinowm.py'
 assert evidence['status']=='complete_verified'
 assert evidence['renderer_sha256']==hashlib.sha256(renderer.read_bytes()).hexdigest()
 for name, expected in evidence['outputs_sha256'].items():
  assert hashlib.sha256((external_path/name).read_bytes()).hexdigest()==expected
 spec=importlib.util.spec_from_file_location('editorial_external_display',renderer)
 api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
 external=api.from_pack(external_path)['finalization']
 for mode in api.INTERNAL:
  for metric in api.METRICS:
   assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in
              zip(external['results']['aggregate'][mode][metric],a['aggregate'][mode][metric]))
 paths += [str(p.relative_to(ROOT)) for p in (renderer,external_evidence)]
 paths += [str((external_path/name).relative_to(ROOT)) for name in evidence['outputs_sha256']]
modes=[('Autoregressive','autoregressive',a),('Persistence','persistence',a),(r'\textbf{ShiftWM (ours)}','transport',a),('No mixing (bounded anchor)','bounded_additive',b),('No bounding','unbounded_transport',b),('No mixing or bounding (additive anchor)','anchored_additive',a),('No support context','context_off',a),('No actions','action_free',a)]
rows=[];records=[]
for label,mode,d in modes:
 if mode=='bounded_additive':
  if external is not None:
   rows.append((external_path/'external_main_rows.tex').read_text().rstrip())
  rows.extend([r'\midrule',r'\multicolumn{3}{@{}l}{\emph{Component ablations}} \\'])
 vals=d['aggregate']['autoregressive']['native_persistence_mse'] if mode=='persistence' else d['aggregate'][mode]['native_mse']
 assert len(vals)==10 and all(math.isfinite(v) and v>=0 for v in vals)
 vals=[vals[4],vals[9]]
 rows.append(f'{label} & {vals[0]:.6f} & {vals[1]:.6f} \\\\')
 records.append({'label':label,'mode':mode,'values':vals})
gains=[]
for h in (5,10):
 e=next(x for x in a['paired_effects'] if x['comparator']=='autoregressive' and x['metric']=='native_mse' and x['horizon']==h)
 assert not e['interval_includes_zero']
 gains.append(e['relative_error_reduction_percent'])
rows+=['\\midrule',r'ShiftWM gain vs autoregression & '+ ' & '.join(r'\positivegain{+'+f'{v:.2f}'+r'\%}' for v in gains)+r' \\']
if external is not None:
 rows.append((external_path/'external_main_gains.tex').read_text().rstrip())
table=r'''\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{10pt}\renewcommand{\arraystretch}{1.08}
\begin{tabular}{@{}lrr@{}}
\toprule
Method & h5 MSE $\downarrow$ & h10 MSE $\downarrow$ \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\caption{\textbf{Matched spatial forecasting on DROID development data.}
Native $4\times4$ standardized endpoint feature errors average windows within episodes,
then episodes and three seeds equally. All21 trained models completed30epochs;
persistence has no trained predictor. The last row reports relative reductions
for ShiftWM against autoregression. Green identifies positive point gains;
paired uncertainty and component effects appear in
Figure~\ref{fig:editorial-spatial} and Appendix~\ref{app:spatial-development}.}
\label{tab:editorial-spatial}
\end{table}
'''
if external is not None:
 table=table.replace('All21','All27').replace('The last row reports relative reductions\nfor ShiftWM against autoregression.',
  'Gain rows report relative reductions for ShiftWM against each named reference.')
 table=table.replace('Figure~\\ref{fig:editorial-spatial} and Appendix~\\ref{app:spatial-development}.',
  'Figure~\\ref{fig:editorial-spatial} and Appendix~\\ref{app:spatial-development}. '
  'The two DINO-WM objectives use the adapted official 19.4M-parameter predictor '
  '\\citep{zhou2024dinowm}; interface and budget differences are explicit in '
  'Appendix~\\ref{app:external-dinowm}.')
table=table.replace('ShiftWM gain vs autoregression','Gain vs autoregression')
table=table.replace('ShiftWM gain vs DINO-WM (shifted one-step)','Gain vs adapted DINO-WM (one-step)')
table=table.replace('ShiftWM gain vs DINO-WM (recursive H10)','Gain vs adapted DINO-WM (recursive H10)')
(OUT/'spatial_main_table.tex').write_text(table.replace('All21','All 21').replace('All27','All 27').replace('completed30epochs','completed 30 epochs'))
rows=[];context=[]
for ref,label in [('framewise','Framewise'),('factorized_unpaired','Unpaired contexts')]:
 for env,name in [('pusht','PushT'),('reacher','Reacher')]:
  chosen=[next(x for x in c['forecast'] if x['comparator']==ref and x['environment']==env and x['split']==split) for split in ('test','extrapolation')]
  values=[x['relative_error_reduction_percent'] for x in chosen]
  for x in chosen:
   assert x['status']=='complete' and x['training_seeds']==[0,1,2]
   assert math.isclose(x['relative_error_reduction_percent'],100*(x['comparator_value']-x['ours_value'])/x['comparator_value'],abs_tol=1e-10)
  text=[r'\positivegain{'+f'{v:+.2f}'+r'\%}' if v>0 else f'{v:+.2f}'+r'\%' for v in values]
  rows.append(f'{name} & {label} & '+ ' & '.join(text)+r' \\')
  context.extend(chosen)
 if ref=='framewise':rows.append(r'\midrule')
table=r'''\begin{table}[!htb]
\centering\small\setlength{\tabcolsep}{7pt}\renewcommand{\arraystretch}{1.08}
\begin{tabular}{@{}llrr@{}}
\toprule Environment & Comparator & Held-out & Extrapolation \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\caption{\textbf{Ours-1 forecast-error reductions in simulation.} Each percentage
compares the same environment, split and five-step feature target; positive
means lower error. Ratios use three-seed means, without a significance claim.
The complete absolute scores, seed dispersion, all aligned controls and
original primary planning results remain in
Appendix~\ref{app:original-context-study}.}
\label{tab:editorial-context}
\end{table}
'''
(OUT/'context_main_table.tex').write_text(table)
ledger={'status':'source_checked','sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'spatial_rows':records,'context_rows':context,'scope':'No new experiment or interval; reorganized main-paper display of completed evidence.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
ledger['external_complete_comparison_included']=external is not None
if external is not None:
 ledger['external_native_curves']={mode:external['results']['aggregate'][mode]['native_mse'] for mode in api.EXTERNAL}
(OUT/'main_tables_evidence.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps({'status':'rendered','internal_spatial_rows':8,'external_spatial_rows':2 if external is not None else 0}))
