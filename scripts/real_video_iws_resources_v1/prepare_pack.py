#!/usr/bin/env python3
"""Portable presentation pack for an already-complete resource measurement; no inference."""
import argparse
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[2]
TASKS=('pusht','bimanual_box','bimanual_rope')
MODES=('autoregressive','anchored_additive','bounded_spatial_mix','unbounded_spatial_mix','persistence')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_text()!=text:raise ValueError('Refusing to change a completed portable pack')
    if not p.exists():p.write_text(text)
def jsontext(x):return json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n'


def prepare(source_dir,output_dir):
    source_dir=Path(source_dir);output_dir=Path(output_dir)
    summary=read(source_dir/'summary.json');validation=read(source_dir/'postrun_validation.json')
    if summary['status']!='passed' or summary['cases']!=39 or validation['status']!='passed':raise ValueError('Complete measured report required')
    if validation['summary_sha256']!=sha(source_dir/'summary.json'):raise ValueError('Review binding differs')
    rows=[];source={}
    for path,h in summary['case_receipts'].items():
        f=ROOT/path
        if sha(f)!=h:raise ValueError('Case record changed')
        row=read(f)
        if row['status']!='passed' or row['registration_sha256']!=summary['registration_sha256']:raise ValueError('Incomplete/mixed case')
        for dev in ('cpu','cuda'):
            d=row['devices'][dev];samples=d['latency']['samples_ms']
            if d['status']!='passed' or len(samples)!=30 or any(not math.isfinite(x) or x<=0 for x in samples):raise ValueError('Missing/invalid timing samples')
            if statistics.median(samples)!=d['latency']['median_ms']:raise ValueError('Stored timing statistic differs')
        rows.append(row);source[path]=h
    expected={(t,m,s) for t in TASKS for m in MODES[:-1] for s in range(3)}|{(t,'persistence',None) for t in TASKS}
    if len(rows)!=39 or {(x['case']['task'],x['case']['mode'],x['case']['seed']) for x in rows}!=expected:raise ValueError('Missing/duplicate predictor')
    summaries=[]
    for task in TASKS:
        for mode in MODES:
            group=sorted([r for r in rows if r['case']['task']==task and r['case']['mode']==mode],key=lambda r:r['case']['seed'] or 0)
            counts=group[0]['parameters']
            if any(r['parameters']!=counts for r in group):raise ValueError('Unmatched architecture counts')
            item={'task':task,'mode':mode,'seeds':[r['case']['seed'] for r in group],'parameters':counts,'measured_models':len(group),'devices':{}}
            for dev in ('cpu','cuda'):
                values=[r['devices'][dev]['latency']['median_ms'] for r in group]
                memory=[r['devices'][dev]['memory'] for r in group]
                if dev=='cpu':peaks={'process_peak_rss_bytes_per_model':[v['process_peak_rss_bytes'] for v in memory],'maximum_process_peak_rss_bytes':max(v['process_peak_rss_bytes'] for v in memory),'isolated_tensor_allocator_peak_bytes':None}
                else:peaks={k+'_per_model':[v[k] for v in memory] for k in ['peak_allocated_bytes','peak_reserved_bytes','incremental_peak_allocated_bytes']};peaks.update({'maximum_'+k:max(v[k] for v in memory) for k in ['peak_allocated_bytes','peak_reserved_bytes','incremental_peak_allocated_bytes']})
                item['devices'][dev]={'per_seed_median_ms':values,'median_of_seed_medians_ms':statistics.median(values),'mean_of_seed_medians_ms':statistics.mean(values),'range_of_seed_medians_ms':[min(values),max(values)],'memory':peaks}
            summaries.append(item)
    comparisons=[]
    for task in TASKS:
        for method,base in [('bounded_spatial_mix','autoregressive'),('unbounded_spatial_mix','autoregressive'),('unbounded_spatial_mix','bounded_spatial_mix')]:
            for dev in ('cpu','cuda'):
                a=next(x for x in summaries if x['task']==task and x['mode']==method)['devices'][dev]['median_of_seed_medians_ms'];b=next(x for x in summaries if x['task']==task and x['mode']==base)['devices'][dev]['median_of_seed_medians_ms']
                comparisons.append({'task':task,'device':dev,'method':method,'comparator':base,'latency_reduction_percent':100*(b-a)/b,'comparator_over_method_latency_ratio':b/a,'scope':'Descriptive ratio of three-seed timing medians for one fixed input; no uncertainty or accuracy inference.'})
    for path in [source_dir/'summary.json',source_dir/'postrun_validation.json',Path(__file__)]:source[str(path.resolve().relative_to(ROOT))]=sha(path)
    data={'schema':'iws_predictor_resource_portable_v1','status':'passed','job_id':'201732','registration_sha256':summary['registration_sha256'],'measurement_protocol':summary['protocol'],'aggregation':'Median of three model-level medians, each based on30 warmed full59-offset calls; brackets are min/max of the three medians, not CI. Persistence has one zero-parameter case/task. Registered mean-of-medians is also retained. Memory columns use maximum recorded peak across the three cases.','measurements_scope':'Batch1 FP32 raw-feature-input predictor only on fixed training episode000011/frame0; excludes encoder, loading, transfer and control.','hardware':{'cpu':rows[0]['devices']['cpu']['hardware'],'cuda':rows[0]['devices']['cuda']['hardware'],'node':rows[0]['node'],'torch':rows[0]['torch'],'numpy':rows[0]['numpy']},'rows':summaries,'descriptive_comparisons':comparisons,'all39_case_measurements':rows,'unsupported_states':[],'cpu_memory_caveat':validation['cpu_memory_caveat'],'official_validation_payloads_read':0,'accuracy_metrics_computed':False,'source_sha256':source}
    data_text=jsontext(data);write(output_dir/'data.json',data_text)
    manifest={'schema':'iws_predictor_resource_pack_manifest_v1','status':'complete_validated_resource_measurement','runtime_inputs_sha256':{'data.json':sha(output_dir/'data.json')},'source_sha256':source,'source_registration_sha256':summary['registration_sha256'],'portable_without_checkpoints_or_datasets':True,'learned_models':36,'persistence_cases':3,'timing_samples':2340,'unsupported_rows':0}
    write(output_dir/'manifest.json',jsontext(manifest))
    lines=['# Complete predictor-only IWS resource comparison','',data['aggregation'],'',data['measurements_scope'],'','CPU: Intel Xeon w7-2495X,2 PyTorch threads, recorded logical affinity[8,32]. GPU: RTX5000 Ada; synchronized eager FP32 wall latency. CPU peak is whole-process RSS; GPU peak is PyTorch allocated bytes (not device-total memory).','', '| Task | Method | Trainable / total | CPU ms [seed range] | CUDA ms [seed range] | CPU RSS peak MiB | CUDA allocated peak MiB |','|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        c,g=row['devices']['cpu'],row['devices']['cuda'];fmt=lambda d:f"{d['median_of_seed_medians_ms']:.3f} [{d['range_of_seed_medians_ms'][0]:.3f}, {d['range_of_seed_medians_ms'][1]:.3f}]"
        lines.append(f"| {row['task']} | {row['mode']} | {row['parameters']['trainable']:,} / {row['parameters']['total']:,} | {fmt(c)} | {fmt(g)} | {c['memory']['maximum_process_peak_rss_bytes']/1048576:.2f} | {g['memory']['maximum_peak_allocated_bytes']/1048576:.2f} |")
    lines+=['','All39 cases and78 device rows passed. No unsupported rows were omitted. All2,340 raw latency samples and both original mean-of-medians and display median-of-medians are retained in data.json. There is no CPU tensor-only peak claim or GPU accuracy/prefix-equivalence certificate.','',data['cpu_memory_caveat'],'','This pack is derived only from completed timing receipts; it contains no input feature arrays, command arrays, targets or weights. The bound reports retain timing/identity provenance.']
    write(output_dir/'README.md','\n'.join(lines)+'\n')
    print(json.dumps({'status':'passed','output':str(output_dir),'rows':15,'timing_samples':2340,'data_sha256':sha(output_dir/'data.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source-dir',type=Path,default=ROOT/'reports/real_video_iws_resources_v1/job_201732');p.add_argument('--output-dir',type=Path,default=ROOT/'paper/table_sources/iws_predictor_resources_v1');args=p.parse_args();prepare(args.source_dir,args.output_dir)
