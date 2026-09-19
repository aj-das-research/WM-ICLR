#!/usr/bin/env python3
"""Display-only IWS tick-clearance revision from the completed public v1 figure.

No scientific evaluator, model, cache or checkpoint is loaded. The immutable v1
figure helper supplies the same curves, caption and typography; only axis widths
change. --output-dir supports an isolated public replay, and --verify-against
requires exact PNG and rasterized-PDF pixel equality before publishing outputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
V1_RENDERER = ROOT / 'paper/scripts/render_iws_results.py'
V1_EVIDENCE = ROOT / 'paper/generated/experiment_alignment/forecast_transfer.json'
V1_RENDERER_SHA = 'c762da602f874336891628bb13ed55815c3f01145f684806ef2e3609f4655671'
V1_EVIDENCE_SHA = 'b6c58f470bea98379dd77266001bcfed60660b0ef04257774fb5d8e9e718866d'
PREFIX = 'forecast_transfer_v2'
AXIS_WIDTH = .247
EXPORTS = tuple(PREFIX+s for s in ('.pdf','.svg','.png','_figure.tex'))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@contextmanager
def offline():
    originals=[]
    def reject(*args,**kwargs):
        raise RuntimeError('Public figure rendering is offline')
    for owner,name in ((socket.socket,'connect'),(socket.socket,'connect_ex'),
                       (socket,'create_connection'),(socket,'getaddrinfo')):
        originals.append((owner,name,getattr(owner,name)));setattr(owner,name,reject)
    try:yield
    finally:
        for owner,name,value in originals:setattr(owner,name,value)


def pixels(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:return np.asarray(im.convert('RGBA')).copy()


def pdf_pixels(path):
    data=subprocess.check_output(['pdftoppm','-f','1','-l','1','-singlefile','-r','144','-png',str(path)])
    return pixels(BytesIO(data))


def load_public(evidence,helper_path):
    if sha(evidence)!=V1_EVIDENCE_SHA or sha(helper_path)!=V1_RENDERER_SHA:
        raise ValueError('Display revision requires the exact completed v1 public snapshot')
    record=json.loads(evidence.read_text());content=record['bound_payload']
    if digest(content)!=record['fingerprint'] or content['schema']!='shiftwm_iws_forecast_figure_v1':
        raise ValueError('Invalid v1 figure identity')
    if set(record['outputs_sha256'])!={f'forecast_transfer{s}' for s in ('.pdf','.svg','.png','_figure.tex')}:
        raise ValueError('Incomplete authoritative v1 exports')
    sources={str(evidence):sha(evidence),str(helper_path):sha(helper_path),str(Path(__file__).resolve()):sha(__file__)}
    for name,expected in record['outputs_sha256'].items():
        path=evidence.parent/name
        if sha(path)!=expected:raise ValueError('Changed v1 export: '+name)
        sources[str(path)]=expected
    spec=importlib.util.spec_from_file_location('_immutable_public_iws_v1',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    # Pure display helpers only: never helper.load_evidence()/render().
    payload=helper.plot_payload(content['full_validated_development'])
    if payload!=content['numerical_payload'] or payload['plotted_means']!=708:
        raise ValueError('All708 numerical values must remain unchanged')
    font=Path(helper.font_manager.findfont(helper.font_manager.FontProperties(family='Liberation Sans'),fallback_to_default=False))
    runtime=content['runtime']
    if (helper.matplotlib.__version__!=runtime['matplotlib'] or helper.np.__version__!=runtime['numpy']
            or sha(font)!=runtime['font_sha256']):raise ValueError('Recorded plotting runtime/font required')
    return helper,payload,content,sources


def make_figure(helper,payload):
    figure,_=helper.make_figure(payload)
    original_positions=[]
    for axis in figure.axes:
        position=axis.get_position();original_positions.append(list(position.bounds))
        axis.set_position([position.x0,position.y0,AXIS_WIDTH,position.height])
    figure.canvas.draw();renderer=figure.canvas.get_renderer();clearances=[]
    for index,axis in enumerate(figure.axes):
        assert list(axis.get_position().bounds)[0:2]==original_positions[index][0:2]
        assert axis.get_position().height==original_positions[index][3]
        for line,mode in zip(axis.lines,helper.MODES):
            helper.np.testing.assert_array_equal(line.get_xdata(),payload['x_H'])
            helper.np.testing.assert_array_equal(line.get_ydata(),payload['tasks'][helper.TASKS[index]]['curves'][mode])
        if len(axis.lines)!=4:raise ValueError('Missing plotted method')
        if index:
            labels=[t for t in axis.get_yticklabels() if t.get_visible() and t.get_text()]
            gap=(min(t.get_window_extent(renderer).x0 for t in labels)
                 -figure.axes[index-1].get_window_extent(renderer).x1)*72/figure.dpi
            clearances.append(float(gap))
            if gap<3:raise ValueError('Adjacent y-axis tick clearance below3pt')
    from matplotlib.text import Text
    for item in figure.findobj(Text):
        if not item.get_visible() or not item.get_text():continue
        box=item.get_window_extent(renderer)
        if item.get_fontsize()<8 or box.x0<-.5 or box.y0<-.5 or box.x1>figure.bbox.x1+.5 or box.y1>figure.bbox.y1+.5:
            raise ValueError('Text size or canvas clipping changed')
    return figure,{'dimensions_inches':[5.5,2.75],'minimum_font_pt':8,
        'axis_width_before':.257,'axis_width_after':AXIS_WIDTH,'other_axis_geometry_unchanged':True,
        'adjacent_y_tick_clearance_pt':clearances,'minimum_required_clearance_pt':3,
        'all708_plotted_values_exact':True,'x_coordinates_and_zero_based_y_domains_unchanged':True}


def render(evidence,helper_path,output,verify_against=None,if_ready=False):
    evidence,helper_path,output=map(lambda p:Path(p).resolve(),(evidence,helper_path,output))
    if not evidence.exists() and if_ready:return {'status':'pending','outputs_written':False}
    with offline():
        helper,payload,base,sources=load_public(evidence,helper_path)
        include=base['include'].replace('forecast_transfer.pdf',PREFIX+'.pdf')
        content={'schema':'shiftwm_iws_forecast_display_v2','revision':'Only three axis widths .257→.247; no data/caption/font changes',
            'v1_evidence_sha256':V1_EVIDENCE_SHA,'v1_renderer_sha256':V1_RENDERER_SHA,'renderer_sha256':sha(__file__),
            'finalization_sha256':base['finalization_sha256'],'runtime':base['runtime'],
            'numerical_payload':payload,'numerical_payload_sha256':digest(payload),'caption':base['caption'],'include':include}
        fingerprint=digest(content);existing=output/(PREFIX+'.json')
        if existing.exists():
            old=json.loads(existing.read_text())
            if old.get('fingerprint')!=fingerprint:raise ValueError('Different v2 snapshot already exists')
            for name,expected in old['outputs_sha256'].items():
                if sha(output/name)!=expected:raise ValueError('Changed v2 output')
            return {'status':'unchanged_verified','outputs_written':False}
        output.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.iws-display-v2-',dir=output))
        try:
            helper.plt.rcParams['svg.hashsalt']='shiftwm-iws-display-v2'
            figure,geometry=make_figure(helper,payload)
            for extension in ('pdf','svg','png'):
                metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None} if extension=='svg' else None
                figure.savefig(stage/(PREFIX+'.'+extension),dpi=300,metadata=metadata)
            helper.plt.close(figure);(stage/(PREFIX+'_figure.tex')).write_text(include)
            comparison=None
            if verify_against:
                reference=Path(verify_against).resolve();prior=json.loads((reference/(PREFIX+'.json')).read_text())
                if prior['fingerprint']!=fingerprint:raise ValueError('Replay target source snapshot differs')
                comparison={}
                for ext,loader in (('png',pixels),('pdf',pdf_pixels)):
                    path=reference/(PREFIX+'.'+ext)
                    if sha(path)!=prior['outputs_sha256'][path.name]:raise ValueError('Replay target changed')
                    a,b=loader(path),loader(stage/path.name)
                    if a.shape!=b.shape or not helper.np.array_equal(a,b):raise ValueError('Actual '+ext+' pixels differ')
                    comparison[ext]={'equal':True,'shape':list(a.shape),'rgba_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
            if any(sha(path)!=expected for path,expected in sources.items()):raise ValueError('Source changed during display revision')
            result={'status':'rendered_geometry_checked','fingerprint':fingerprint,'bound_payload':content,'geometry_checks':geometry,
                'outputs_sha256':{name:sha(stage/name) for name in EXPORTS},'visual_review':'separate source-bound review receipt',
                'private_scientific_inputs_read':False,'v1_sources_and_exports_unchanged':True}
            (stage/(PREFIX+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            if comparison:
                (stage/'pixel_replay.json').write_text(json.dumps({'status':'passed','fingerprint':fingerprint,'comparison':comparison,
                    'scope':'Public figure replay only; no private scientific evaluation recomputed','network_attempts':0},indent=2)+'\n')
            for path in stage.iterdir():
                if (output/path.name).exists():raise ValueError('Refusing to replace existing output')
                os.replace(path,output/path.name)
            return {'status':'passed_pixel_replay' if comparison else 'rendered_geometry_checked','outputs_written':True,'geometry':geometry}
        finally:shutil.rmtree(stage,ignore_errors=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--evidence',type=Path,default=V1_EVIDENCE)
    parser.add_argument('--v1-renderer',type=Path,default=V1_RENDERER)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'paper/generated/experiment_alignment')
    parser.add_argument('--verify-against',type=Path)
    parser.add_argument('--if-ready',action='store_true')
    args=parser.parse_args()
    print(json.dumps(render(args.evidence,args.v1_renderer,args.output_dir,args.verify_against,args.if_ready),sort_keys=True))
