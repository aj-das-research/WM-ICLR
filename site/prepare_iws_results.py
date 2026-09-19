#!/usr/bin/env python3
"""Prepare a static IWS section from the completed, reviewed paper artifacts.

This is a presentation adapter, not a scientific evaluator. It checks the
immutable completion receipt, all27 evaluation identities, current rendering
sources and reviewed plot bytes. Checkpoints/datasets are never opened.
"""
from __future__ import annotations
import hashlib
import html
import json
import math
from pathlib import Path
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BEGIN='<!-- BEGIN SOURCE-DERIVED IWS RESULTS -->'
END='<!-- END SOURCE-DERIVED IWS RESULTS -->'
TASKS=('pusht','bimanual_box','bimanual_rope')
NAMES={'pusht':'PushT','bimanual_box':'Box','bimanual_rope':'Rope'}
MODES=('persistence','autoregressive','anchored_additive','bounded_spatial_mix')
FILES={
 'summary':'paper/generated/experiment_alignment/main_transfer.json',
 'plot_v1':'paper/generated/experiment_alignment/forecast_transfer.json',
 'plot_v2':'paper/generated/experiment_alignment/forecast_transfer_v2.json',
 'plot_pdf':'paper/generated/experiment_alignment/forecast_transfer_v2.pdf',
 'review':'reports/evidence/iws_results_display_v2_review.json',
 'finalizer':'reports/real_video_iws/development_finalization.json',
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verified_results():
    sources={p:digest(ROOT/p) for p in FILES.values()}
    data={name:json.loads((ROOT/path).read_text()) for name,path in FILES.items() if name!='plot_pdf'}
    summary,v1,v2,review,final=(data[k] for k in ('summary','plot_v1','plot_v2','review','finalizer'))
    final_sha=sources[FILES['finalizer']]
    if (final.get('schema')!='shiftwm_iws_development_finalization_v1' or final.get('status')!='passed'
            or final.get('expected_runs')!=27 or final.get('completed_runs')!=27
            or summary.get('status')!='complete_validated_development' or summary.get('finalization_sha256')!=final_sha):
        raise ValueError('Complete27-run finalized development evidence required')
    for record in (v1,v2):
        if canonical(record['bound_payload'])!=record['fingerprint'] or record['bound_payload']['finalization_sha256']!=final_sha:
            raise ValueError('Plot fingerprint or finalization differs')
    payload=summary['numerical_payload']
    if payload!=v1['bound_payload']['numerical_payload'] or payload!=v2['bound_payload']['numerical_payload']:
        raise ValueError('Main table and both plots must contain the exact same708 means')
    if payload['plotted_means']!=708 or set(payload['tasks'])!=set(TASKS):raise ValueError('Incomplete curve population')
    if (review.get('status')!='passed' or review['source_dependencies'][FILES['plot_pdf']]!=sources[FILES['plot_pdf']]
            or v2['outputs_sha256']['forecast_transfer_v2.pdf']!=sources[FILES['plot_pdf']]):
        raise ValueError('Web plot must use the reviewed v2 PDF bytes')
    # Verify the artifact-producing sources, not the private scientific binaries.
    dependencies=dict(summary['renderer_sources_sha256'])
    dependencies.update(v1['bound_payload']['renderer_dependencies_sha256'])
    dependencies['paper/scripts/render_iws_results_v2.py']=v2['bound_payload']['renderer_sha256']
    dependencies['configs/real_video_iws/training_registration_v1.json']=v1['bound_payload']['full_validated_development']['registration_sha256']
    for path,expected in dependencies.items():
        if digest(ROOT/path)!=expected:raise ValueError('Artifact source changed: '+path)
        sources[path]=expected
    runs=v1['bound_payload']['full_validated_development']['runs']
    grid={(t,m,s) for t in TASKS for m in MODES[1:] for s in range(3)}
    if len(runs)!=27 or {(r['task'],r['mode'],r['seed']) for r in runs}!=grid:
        raise ValueError('Incomplete matched task/method/seed grid')
    for run in runs:
        path=run['evaluation_path'];expected=run['evaluation_sha256']
        if digest(ROOT/path)!=expected:raise ValueError('Completed evaluation changed: '+path)
        receipt=json.loads((ROOT/path).read_text())
        if (receipt['status']!='passed' or receipt['scope']!='internal_development' or receipt['completed_epochs']!=30
                or receipt['official_validation_payloads_read']!=0
                or any(receipt[key]!=run[key] for key in ('task','mode','seed','selected_checkpoint_sha256'))):
            raise ValueError('Wrong completed development identity')
        sources[path]=expected
    rows=[]
    for task in TASKS:
        item=payload['tasks'][task];means=item['h60_endpoints']
        if set(means)!=set(MODES) or any(not math.isfinite(v) or v<0 for v in means.values()):raise ValueError('Invalid endpoint')
        ours,anchor=means['bounded_spatial_mix'],means['anchored_additive']
        gain=100*(anchor-ours)/anchor
        if not math.isclose(gain,item['primary_gain_percent'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Gain does not match means')
        ci=item['primary_gain_ci95_percent']
        if len(ci)!=2 or not all(math.isfinite(v) for v in ci) or ci[0]>ci[1]:raise ValueError('Invalid relative-gain interval')
        rows.append({'task':task,'name':NAMES[task],'means':means,'best_methods':[m for m in MODES if means[m]==min(means.values())],
            'gain_vs_anchor_percent':gain,'gain_ci95_percent':ci,'eligible_trajectories':item['eligible_trajectories'],'total_windows':item['total_windows']})
    return {'schema':'shiftwm_iws_web_results_v1','status':'complete_validated_development','scope':'internal_development',
        'finalization_sha256':final_sha,'source_files_sha256':sources,'preparer_sha256':digest(__file__),
        'completed_models':27,'training_seeds':3,'methods':list(MODES),'rows':rows,
        'metric':'Native training-standardized feature MSE at H60 (stored offset59)',
        'aggregation':'Equal windows within trajectory, equal trajectories, equal matched seeds',
        'gain_interval_units':'Percent relative MSE reduction against additive anchoring; paired seed-by-trajectory 95% interval',
        'checkpoints':{'local_iws_inference_bundles':27,'published_iws_bundles':0,'adds_to_public_predictor_count':False},
        'reviewed_plot':{'asset':'assets/iws-forecast.svg','pdf_source':FILES['plot_pdf'],'pdf_sha256':sources[FILES['plot_pdf']]}}


def section(result):
    rows=[]
    for row in result['rows']:
        cells=[]
        for mode in MODES:
            value=f"{row['means'][mode]:.5f}"
            if mode in row['best_methods']:value='<strong>'+value+'</strong>'
            cells.append('<td>'+value+'</td>')
        gain=row['gain_vs_anchor_percent'];ci=row['gain_ci95_percent']
        point=f'{gain:+.2f}%'
        if gain>0:point='<strong class="iws-positive">'+point+'</strong>'
        cells.append(f'<td>{point} <span class="iws-ci">[{ci[0]:+.2f}, {ci[1]:+.2f}]</span></td>')
        rows.append('<tr><th scope="row">'+html.escape(row['name'])+'</th>'+''.join(cells)+'</tr>')
    wins=sum(r['gain_vs_anchor_percent']>0 for r in result['rows'])
    ar_wins=sum(r['means']['autoregressive']<r['means']['bounded_spatial_mix'] for r in result['rows'])
    ar_text=('Autoregression has lower endpoint MSE on all three tasks.' if ar_wins==3
             else f'Autoregression has lower endpoint MSE on {ar_wins} of three tasks.')
    return BEGIN+f'''
<section class="iws-transfer" id="iws-results" aria-labelledby="iws-title">
  <span class="iws-label">Single-observation transfer · internal development</span>
  <h3 id="iws-title">One image. Three manipulation tasks.</h3>
  <p>Separately trained IWS adapters forecast 59 future feature states from one observed image and 60 recorded native command rows. All 27 models completed training and validated evaluation: three tasks, three learned methods and three matched seeds.</p>
  <p class="fineprint iws-scroll-hint">Scroll the table to see every method and interval →</p>
  <div class="table-wrap iws-table-wrap" tabindex="0" aria-label="IWS results; scroll horizontally for all methods and intervals">
    <table class="results-table iws-table"><caption>H=60 feature MSE · lower is better · target offset 59</caption>
      <thead><tr><th scope="col">Task</th><th scope="col">Persistence</th><th scope="col">Autoregressive</th><th scope="col">Additive anchor</th><th scope="col">ShiftWM (ours)</th><th scope="col">Gain vs anchor [95% CI]</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  <p class="fineprint iws-table-note">Bold means mark the lowest exact task mean. Green marks a favorable point estimate; intervals quantify uncertainty. Gains and intervals are percentages. Scores average windows within trajectory, then trajectories and matched seeds equally.</p>
  <p class="iws-reading">ShiftWM improves on the additive anchor on {wins} of three tasks by point estimate. {ar_text} Development results do not establish an external state-of-the-art advantage; reserved validation remains separate.</p>
  <details class="protocol-details iws-curves"><summary>Inspect all forecast horizons</summary>
    <figure class="comparison-figure"><a href="assets/iws-forecast.svg"><img src="assets/iws-forecast.svg" width="1650" height="825" alt="IWS PushT, Box and Rope development curves for persistence, autoregression, additive anchoring and ShiftWM. Each task retains all 59 forecast offsets; signed endpoint gains and paired 95% intervals compare ShiftWM with additive anchoring." loading="lazy"></a>
      <figcaption>One zero-based MSE scale per task. H is the supplied command count; its target is stored offset H−1, not physical time. Curves show three-seed means; endpoint annotations give paired seed–trajectory intervals. <a href="assets/iws-forecast.svg">Open the reviewed plot</a>.</figcaption></figure>
  </details>
  <p class="fineprint iws-local">The 27 IWS inference bundles are local artifacts and are not included in the 117 published predictors.</p>
  <div class="source-links"><a href="iws-results.json">Machine-readable IWS results</a><a href="https://github.com/aj-das-research/WM-ICLR/blob/main/paper/generated/experiment_alignment/main_transfer.json">Validated paper source ↗</a></div>
</section>
'''+END


def prepare():
    result=verified_results();path=HERE/'index.html';original=path.read_text()
    if original.count(BEGIN)!=1 or original.count(END)!=1:raise ValueError('Exactly one IWS section marker pair required')
    before,tail=original.split(BEGIN);_,after=tail.split(END)
    updated=before+section(result)+after
    for name,expected in result['source_files_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Evidence changed while preparing site')
    for target,value in ((HERE/'iws-results.json',json.dumps(result,indent=2,allow_nan=False)+'\n'),(path,updated)):
        if target.exists() and target.read_text()==value:continue
        with tempfile.NamedTemporaryFile('w',dir=HERE,prefix='.iws-site-',delete=False) as stream:
            stream.write(value);temporary=Path(stream.name)
        temporary.replace(target)
    return result


if __name__=='__main__':
    value=prepare();print(json.dumps({'status':value['status'],'tasks':3,'completed_models':27,'published_iws_bundles':0}))
