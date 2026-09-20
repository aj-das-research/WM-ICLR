#!/usr/bin/env python3
"""Source-bound presentation of complete36 IWS development and reviewed reserved results.
No checkpoint/cache/raw-data/primitive-NPZ reads. Missing reserved pack is pending;
a present invalid pack fails closed. Only the marked IWS section is replaced.
"""
from __future__ import annotations
import hashlib, html, importlib.util, json, math, sys, tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BEGIN='<!-- BEGIN SOURCE-DERIVED IWS RESULTS -->'
END='<!-- END SOURCE-DERIVED IWS RESULTS -->'
TASKS=('pusht','bimanual_box','bimanual_rope')
NAMES=dict(zip(TASKS,('PushT','Box','Rope')))
MODES=('persistence','autoregressive','anchored_additive','bounded_spatial_mix','unbounded_spatial_mix')
LABELS=dict(zip(MODES,('Persistence','Autoregressive','Additive anchor','ShiftWM (ours)','No-tanh (ours, ablation)')))
METRICS=('standardized_mse','standardized_mae','raw_dinov2_l1','feature_cosine_distance')
METRIC_NAMES=dict(zip(METRICS,('Standardized MSE','Standardized MAE','Raw feature L1','Cosine distance')))
DEV='paper/figure_sources/current_real_scorecards'
EFFECTS='paper/figure_sources/iws_compact_evidence'
RESERVED='paper/figure_sources/iws_reserved_evidence'
PLOT='paper/generated/iws_compact_evidence/forecast_and_gain.svg'
ASSET='assets/iws-development-forecast.svg'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def bound(root,name,sources,expected=None):
    path=root/name;require(path.resolve().is_relative_to(root.resolve()),'Source escapes workspace')
    actual=digest(path);require(expected is None or actual==expected,'Changed reviewed source: '+name)
    sources[name]=actual;return json.loads(path.read_text())
def load_module(root,name,sources):
    sources[name]=digest(root/name)
    spec=importlib.util.spec_from_file_location('_site_'+Path(name).stem,root/name)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def check_effect(effect,means):
    a,b=effect['method'],effect['reference'];require(a in MODES and b in MODES,'Unknown comparison')
    require(math.isclose(effect['gain'],100*(means[b]-means[a])/means[b],rel_tol=0,abs_tol=1e-10),'Gain differs from means')
    ci=effect['ci'];require(len(ci)==2 and all(math.isfinite(x) for x in ci) and ci[0]<=ci[1],'Invalid interval')

def development(root,sources):
    review=bound(root,'reports/evidence/iws_unbounded_completed_independent_review.json',sources)
    visual=bound(root,'reports/evidence/iws_compact_evidence_independent_review.json',sources)
    require(review['status']==visual['status']=='passed','Completed development reviews required')
    bindings=review['source_sha256'];pb=visual['source_and_output_sha256']
    manifest=bound(root,DEV+'/manifest.json',sources,bindings[DEV+'/manifest.json'])
    data=bound(root,DEV+'/data.json',sources,bindings[DEV+'/data.json'])
    require(manifest['data_sha256']==sources[DEV+'/data.json'],'Changed scorecard pack')
    helper=load_module(root,'paper/scripts/render_current_real_scorecards.py',sources)
    require(sources['paper/scripts/render_current_real_scorecards.py']==bindings['paper/scripts/render_current_real_scorecards.py'],'Changed scorecard validator')
    helper.validate_payload(data)
    iws=data['iws'];c=iws['completion'];require(iws['unbounded_included'] and c['v1_runs']==27 and c['unbounded_runs']==9,'Complete36 required')
    for path,key in [('reports/real_video_iws/development_finalization.json','v1_finalization_sha256'),('reports/real_video_iws_unbounded/development_finalization.json','unbounded_finalization_sha256')]:
        require(data['source_sha256'][path]==c[key]==bindings[path],'Finalization identity differs')
    em=bound(root,EFFECTS+'/manifest.json',sources,pb[EFFECTS+'/manifest.json'])
    effects=bound(root,EFFECTS+'/data.json',sources,pb[EFFECTS+'/data.json'])
    require(em['data_sha256']==sources[EFFECTS+'/data.json'] and em['sources'][DEV+'/data.json']==sources[DEV+'/data.json'],'Different contrast population')
    ph=load_module(root,'paper/scripts/render_iws_compact_evidence.py',sources)
    require(sources['paper/scripts/render_iws_compact_evidence.py']==pb['paper/scripts/render_iws_compact_evidence.py'],'Changed plot validator');ph.validate(effects)
    require(digest(root/PLOT)==pb[PLOT],'Plot differs from independent visual review');sources[PLOT]=pb[PLOT]
    rows=[]
    for task in TASKS:
        item=iws['tasks'][task];e=effects['tasks'][task]
        means={metric:{mode:item['methods'][mode]['mean_curves'][metric][-1] for mode in MODES} for metric in METRICS}
        curves={mode:item['methods'][mode]['mean_curves']['standardized_mse'] for mode in MODES}
        require(item['population']==e['population'],'Different development population')
        for mode in MODES:
            require(all(math.isclose(a,b,rel_tol=0,abs_tol=2e-14) for a,b in zip(curves[mode],e['curves'][mode])),'Different plot curves')
        contrasts={key:{**{k:v[k] for k in ('method','reference','study','gain')},'ci':v['ci']['gain']} for key,v in e['effects'].items()}
        for value in contrasts.values():check_effect(value,means['standardized_mse'])
        rows.append({'task':task,'name':NAMES[task],'means':means,'curves':curves,'effects':{'standardized_mse':contrasts},'population':item['population']})
    return {'status':'complete_validated_development','scope':'internal_development','completed_models':36,'original_models':27,'exploratory_followup_models':9,'original_finalization_sha256':c['v1_finalization_sha256'],'followup_finalization_sha256':c['unbounded_finalization_sha256'],'rows':rows,'plot':ASSET,
        'macro':{'bounded_vs_additive':{'gain':effects['macro_original_gain'],'ci':effects['macro_original_ci']},'unbounded_vs_bounded':{'gain':effects['macro_followup_gain']['equal_task_relative_error_reduction_percent'],'ci':effects['macro_followup_gain']['paired95']}},'unmeasured':iws['not_measured']}

def reserved(root,sources):
    pack=root/RESERVED
    if not (pack/'manifest.json').exists():return {'status':'pending_complete_reviewed_evidence','scope':'reserved_upstream_validation','rows':[]}
    helper=load_module(root,'paper/scripts/render_iws_reserved_evidence.py',sources)
    value,manifest,validation=helper.load_pack(pack)
    sources[RESERVED+'/manifest.json']=digest(pack/'manifest.json')
    sources.update({RESERVED+'/'+p:s for p,s in manifest['files_sha256'].items()})
    results=value['results'];rows=[]
    for task in TASKS:
        means={};effects={}
        for metric in METRICS:
            item=results['task_results'][task][metric];means[metric]=item['equal_trajectory']['horizon_means']['60']
            require(set(means[metric])==set(MODES),'Missing reserved method');effects[metric]={}
            for key,(method,reference,_) in helper.PAIRS.items():
                e=item['h60_comparisons'][key];effect={'method':method,'reference':reference,'study':'followup9' if method=='unbounded_spatial_mix' else 'original27','gain':e['relative_error_reduction_percent'],'ci':e['paired95']['gain_percent']['percentile95']}
                check_effect(effect,means[metric]);effects[metric][key]=effect
        rows.append({'task':task,'name':NAMES[task],'means':means,'effects':effects,'population':{'trajectories':10,'windows':200}})
    macro={metric:{key:{'gain':e['equal_task_relative_error_reduction_percent'],'ci':e['paired95']['percentile95']} for key,e in entries.items()} for metric,entries in results['macro_h60'].items()}
    return {'status':'complete_reviewed_reserved','scope':'reserved_upstream_validation','completed_models':36,'rows':rows,'macro':macro,'finalization_sha256':manifest['finalization_sha256'],'registration_sha256':manifest['registration_sha256'],'backend':'one_command_row_gru_cpu_fp32_v1','numerical_recovery':manifest['numerical_recovery'],'unmeasured':['RGB quality','physical control success']}

def verified_results(root=ROOT):
    sources={}
    return {'schema':'shiftwm_iws_web_results_v2','status':'complete_validated_development','development':development(root,sources),'reserved':reserved(root,sources),'methods':list(MODES),'metrics':list(METRICS),'horizon':60,'target_offset':59,'aggregation':'Equal handles within trajectory, equal trajectories, equal matched seeds','source_files_sha256':sources,'preparer_sha256':digest(__file__),'checkpoints':{'local_iws_inference_bundles':36,'published_iws_bundles':0,'published_other_predictors':117,'adds_to_public_predictor_count':False}}

def gain_cell(e):
    point=f"{e['gain']:+.2f}%"
    if e['gain']>0:point='<strong class="iws-positive">'+point+'</strong>'
    return point+f' <span class="iws-ci">[{e["ci"][0]:+.2f}, {e["ci"][1]:+.2f}]</span>'
def table(headers,rows,caption):
    return '<div class="table-wrap iws-table-wrap" tabindex="0" aria-label="Scrollable IWS results table"><table class="results-table iws-table"><caption>'+html.escape(caption)+'</caption><thead><tr>'+''.join('<th scope="col">'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr><th scope="row">'+html.escape(row[0])+'</th>'+''.join('<td>'+cell+'</td>' for cell in row[1:])+'</tr>' for row in rows)+'</tbody></table></div>'
def scores(study,metric='standardized_mse'):
    rows=[]
    for row in study['rows']:
        means=row['means'][metric];best=min(means.values());cells=[]
        for mode in MODES:
            cell=f'{means[mode]:.5f}';cells.append('<strong>'+cell+'</strong>' if means[mode]==best else cell)
        rows.append([row['name'],*cells])
    return table(['Task',*[LABELS[m] for m in MODES]],rows,METRIC_NAMES[metric]+' at H60 · lower is better')
def contrasts(study,metric='standardized_mse'):
    rows=[]
    for task in study['rows']:
        for e in task['effects'][metric].values():rows.append([task['name'],html.escape(LABELS[e['method']]+' vs '+LABELS[e['reference']]),gain_cell(e)])
    return table(['Task','Named comparison','Relative reduction [95% interval]'],rows,METRIC_NAMES[metric]+' reductions at H60 · positive is better')
def metric_details(study):
    return '<details class="protocol-details iws-metrics"><summary>Inspect every measured metric</summary>'+''.join(scores(study,m) for m in METRICS[1:])+'<p class="fineprint">All metrics are errors: lower is better. MSE and MAE use fixed task-specific training scales; raw L1 and cosine distance use DINOv2 coordinates. RGB quality and physical control success were not measured.</p></details>'
def section(result):
    dev,held=result['development'],result['reserved']
    macro='; '.join(('ShiftWM vs additive anchor' if k=='bounded_vs_additive' else 'No-tanh vs ShiftWM')+': '+gain_cell(v) for k,v in dev['macro'].items())
    population='; '.join(f"{r['name']}: {r['population']['trajectories']} trajectories / {r['population']['windows']:,} windows" for r in dev['rows'])
    held_html='<h4>Reserved upstream validation</h4>'
    if held['status']=='pending_complete_reviewed_evidence':
        held_html+='<p class="fineprint">Results await complete 36-model finalization and independent numerical review. This separate evaluation uses 600 fixed handles from 30 trajectories; no partial accuracy values are displayed.</p>'
    else:
        held_html+='<p>All 36 selected predictors use the same 200 reserved handles per task from ten trajectories. Checkpoint selection remained fixed after development. These scores are separate from the development results above.</p>'+scores(held)
        held_html+='<details class="protocol-details iws-reserved-contrasts"><summary>Inspect reserved comparisons and uncertainty</summary>'+''.join(contrasts(held,m) for m in METRICS)
        for metric,entries in held['macro'].items():held_html+='<p class="fineprint">'+html.escape(METRIC_NAMES[metric])+' macro: '+'; '.join(html.escape(k.replace('_',' '))+': '+gain_cell(v) for k,v in entries.items())+'</p>'
        held_html+='</details>'+metric_details(held)+'<p class="fineprint">Numerical recovery: all 36 models use an explicit row-wise command-GRU backend after an input-only prefix audit conducted after initial reserved access. Weights, inputs, metrics and tolerances are unchanged. The original native-backend results are retained; this is not a new training run.</p>'
    return BEGIN+'''
<section class="iws-transfer" id="iws-results" aria-labelledby="iws-title">
  <span class="iws-label">Single-observation transfer · two distinct evaluation scopes</span>
  <h3 id="iws-title">One image. Three manipulation tasks.</h3>
  <p>Separately trained IWS predictors forecast 59 future feature states from one observed image and 60 recorded native command rows. All 36 models completed development training and evaluation: the original 27-model comparison plus nine no-tanh component-ablation models, with three matched seeds per task and learned method.</p>
  <h4>Completed internal development</h4>
  <p class="fineprint">The no-tanh (ours, ablation) study is a separately registered exploratory follow-up. It removes only the correction bound and does not replace the original proposed model.</p>
  <p class="fineprint iws-scroll-hint">Scroll the table to see every method →</p>
'''+scores(dev)+'''
  <p class="fineprint iws-table-note">Bold black marks the lowest exact task mean. Green marks a favorable point reduction against its named reference, not statistical significance. Scores average windows within trajectory, then trajectories and matched seeds equally.</p>
  <p class="iws-reading">The bounded ShiftWM model has higher endpoint MSE than autoregression on all three tasks; the PushT interval includes zero. Removing tanh lowers all three endpoint means in the exploratory development follow-up. These comparisons do not establish an external state-of-the-art advantage.</p>
'''+ '<p class="fineprint">'+html.escape(population)+'.</p>'+ '<details class="protocol-details iws-contrasts"><summary>Inspect all signed development comparisons</summary>'+contrasts(dev)+'<p class="fineprint">Equal-task macro reductions: '+macro+'.</p></details>'+metric_details(dev)+f'''
  <details class="protocol-details iws-curves"><summary>Inspect all 59 forecast offsets</summary>
    <figure class="comparison-figure"><a href="{dev['plot']}"><img src="{dev['plot']}" width="1650" height="795" alt="All five predictors across 59 forecast offsets for PushT, Box and Rope. Signed endpoint reductions and paired intervals preserve the bounded model's regressions against autoregression and the exploratory no-tanh improvement." loading="lazy"></a>
      <figcaption>H counts supplied native command rows; the target is stored offset H−1, not physical time. Three-seed means and paired seed–trajectory 95% intervals. Training scales differ by task. <a href="{dev['plot']}">Open the reviewed development plot</a>.</figcaption></figure>
  </details>
'''+held_html+'''
  <p class="fineprint iws-table-note">Intervals use 10,000 paired seed–trajectory draws and are unadjusted. The original primary comparison is ShiftWM versus additive anchoring; no-tanh comparisons are follow-up analyses. Persistence ignores commands.</p>
  <p class="fineprint iws-local">All 36 IWS predictors have local inference exports. They are separate from the 117 published predictors and are not public checkpoint downloads.</p>
  <div class="source-links"><a href="iws-results.json">Machine-readable results and source hashes</a><a href="https://github.com/aj-das-research/WM-ICLR/blob/main/paper/figure_sources/current_real_scorecards/data.json">Development evidence ↗</a></div>
</section>
'''+END

def prepare(root=ROOT,here=HERE):
    result=verified_results(root);path=here/'index.html';original=path.read_text()
    require(original.count(BEGIN)==original.count(END)==1,'Exactly one IWS marker pair required')
    before,tail=original.split(BEGIN);_,after=tail.split(END);updated=before+section(result)+after
    for name,expected in result['source_files_sha256'].items():require(digest(root/name)==expected,'Evidence changed while preparing site')
    for target,value in [(here/'iws-results.json',(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()),(here/ASSET,(root/PLOT).read_bytes()),(path,updated.encode())]:
        if target.exists() and target.read_bytes()==value:continue
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile('wb',dir=target.parent,prefix='.iws-site-',delete=False) as stream:stream.write(value);temporary=Path(stream.name)
        temporary.replace(target)
    return result

if __name__=='__main__':
    value=prepare();print(json.dumps({'status':value['status'],'tasks':3,'completed_models':36,'reserved_status':value['reserved']['status'],'published_iws_bundles':0}))
