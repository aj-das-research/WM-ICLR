#!/usr/bin/env python3
"""Second arithmetic pass directly against original JSON and published intervals.

This is author-side validation, not an independent reviewer receipt.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-10):raise ValueError((a,b))
def main():
    path=ROOT/'reports/metrics_completion_v1/physical/data.json';d=read(path)
    for p,h in {**d['source_sha256'],**d['definition_source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('Source changed: '+p)
    cells=0
    for run in d['runs']:
        raw=read(ROOT/run['source']);rows=raw['planning']['records'] if run['family']=='core' else raw['records']
        for a,b in zip(rows,run['records'],strict=True):
            if (a['seed'],a['observation_id'],a['dynamics_id'],bool(a['success']))!=(b['task_seed'],b['observation_id'],b['dynamics_id'],b['success']):raise ValueError('Row identity')
            for metric,value in b['errors'].items():
                if metric=='position_error_px':ref=(a['block_translation_error_px']**2+a['agent_position_error_px']**2)**.5
                elif metric=='goal_distance_mm':ref=a['final_distance_m']*1000
                elif metric=='speed_mm_s':ref=a['final_metrics']['speed_m_s']*1000
                elif metric=='altitude_error_mm':ref=a['final_metrics']['altitude_error_m']*1000
                else:ref=a[metric]
                close(ref,value);cells+=1
            close(a['native_steps'] if run['family']=='core' else a['native_calls'],b['native_calls'])
            if not b['success'] and (b['first_success'] is not None or b['budget_penalized_commands']!=b['budget']):raise ValueError('Failure censoring')
    # Check every existing comparable raw-success interval, without new draws.
    prior=read(ROOT/'reports/evidence/paired_planning_results.json');ci=0
    for r in prior['results']:
        if r['metric']!='raw_success':continue
        condition=[2,2] if r['split']=='test' else 'all'
        c=next(c for c in d['contrasts'] if c['family']=='core' and c['task']==r['environment'] and c['split']==r['split'] and c['comparator']==r['comparator'] and c['condition']==condition and c['metric']=='success')
        close(c['mean'],r['metrics']['mean_difference_pp'])
        for a,b in zip(c['ci95'],r['metrics']['conditional_ci95_pp']):close(a,b)
        ci+=1
    ext=read(ROOT/'reports/completed_extension_results.json')
    for r in ext['original_planning_comparisons']:
        c=next(c for c in d['contrasts'] if c['family']=='original' and c['task']==r['domain'] and c['backbone']==r['architecture'] and c['comparator']==r['baseline'] and c['condition']=='all' and c['metric']=='success')
        close(c['mean'],r['success_difference_percentage_points'])
        for a,b in zip(c['ci95'],r['ci95_percentage_points']):close(a,b)
        ci+=1
    c=next(c for c in d['contrasts'] if c['family']=='wide_gain' and c['condition']=='all' and c['metric']=='success');r=ext['geometry_planning_comparisons'][0]
    close(c['mean'],r['success_difference_percentage_points'])
    for a,b in zip(c['ci95'],r['ci95_percentage_points']):close(a,b)
    ci+=1
    result=dict(status='passed_author_arithmetic_check',data_sha256=sha(path),validator_sha256=sha(__file__),
                source_count=len(d['source_sha256'])+len(d['definition_source_sha256']),physical_scalar_cells=cells,
                records=sum(len(r['records']) for r in d['runs']),existing_paired_success_intervals_exact=ci,
                limitations='Author validation, not independent review. New physical-error intervals are post-hoc descriptive; core uncertainty conditional on fixed training seeds.')
    output=ROOT/'reports/metrics_completion_v1/physical/numerical_validation.json';output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
