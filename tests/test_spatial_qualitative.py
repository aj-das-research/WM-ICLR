"""Prespecified selection, numerical replay contracts, and unpublished layout QA."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/real_video_spatial_qualitative'))
import selection
spec=importlib.util.spec_from_file_location('spatial_qualitative_replay_test',ROOT/'scripts/real_video_spatial_qualitative/replay.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
spec=importlib.util.spec_from_file_location('spatial_qualitative_render_test',ROOT/'scripts/real_video_spatial_qualitative/render.py')
render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)


def test_selection_is_best_lower_median_worst_with_stable_ties():
    rows=[{'episode_id':str(i),'gain_percent':v} for i,v in enumerate([10,3,3,-2,-20,1])]
    result=selection.select_episodes(rows)
    assert [r['episode_id'] for r in result]==['0','2','4']
    assert [r['rank_descending'] for r in result]==[0,2,5]
    assert result[-1]['label']=='Largest regression'
    for row in rows:row['gain_percent']=abs(row['gain_percent'])
    assert selection.select_episodes(rows)[-1]['label']=='Smallest episode gain'


def test_registration_was_frozen_before_scores():
    registered=replay.check_rule()
    assert registered['spatial_validation_result_files_present']==0
    assert registered['rule']['replay_window'].startswith('smallest')
    assert registered['rule']['target_patch_zero_based']==[1,1]


def score_fixture():
    population={'e':{'session_id':'s','starts':[0,5]}}
    rows=[]
    for start,value in ((0,.2),(5,.4)):
        rows.append({'episode_id':'e','session_id':'s','window_start':start,**{m:[value]*10 for m in replay.METRICS}})
    summary={m:[.3]*10 for m in replay.METRICS}
    document={'status':'complete','scope':'original_validation_development_only','windows':rows,
              'episodes':[{'episode_id':'e','session_id':'s','windows':2,**deepcopy(summary)}],'summary':summary}
    return document,population


@pytest.mark.parametrize('damage',['window','mean','summary','scope','negative','nan','length'])
def test_source_ledger_rejects_changed_or_incomplete_metrics(damage):
    doc,population=score_fixture();replay.check_scores(doc,population)
    if damage=='window':doc['windows'][0]['window_start']=10
    elif damage=='mean':doc['episodes'][0]['native_mse'][9]=.4
    elif damage=='summary':doc['summary']['native_mse'][9]=.4
    elif damage=='scope':doc['scope']='test'
    elif damage=='negative':doc['windows'][0]['native_mse'][9]=-.1
    elif damage=='nan':doc['windows'][0]['native_mse'][9]=float('nan')
    else:doc['windows'][0]['native_mse'].pop()
    with pytest.raises(ValueError):replay.check_scores(doc,population)


def test_patch_error_uses_correct_channel_major_order_and_shared_scale_units():
    target=torch.zeros(2,10,6144);prediction=target.clone();std=torch.ones(6144)*2
    # One channel, patch (1,1), squared standardized residual=4.
    prediction[:,:,5]=4
    patches=replay.patch_errors(prediction,target,std)
    assert patches.shape==(2,10,4,4)
    torch.testing.assert_close(patches[:,:,1,1],torch.ones(2,10)*(4/384))
    assert torch.count_nonzero(patches)==20
    torch.testing.assert_close(patches.mean((-1,-2)),((prediction-target)/std).square().mean(-1))


def test_incomplete_campaign_never_creates_candidate(monkeypatch,tmp_path):
    registration=tmp_path/'registry.json';registration.write_text('{}')
    final=tmp_path/'final.json';final.write_text('{"status":"in_progress","completed_models":14}')
    monkeypatch.setattr(replay,'SPATIAL_REG',registration);monkeypatch.setattr(replay,'FINAL',final)
    with pytest.raises(ValueError,match='fully verified fifteen'):
        replay.checked_campaign()


def test_pipeline_never_writes_unreviewed_images_into_paper():
    with pytest.raises(ValueError,match='outside paper'):
        replay.replay(ROOT/'paper/generated/would_be_wrong')
    with pytest.raises(ValueError,match='outside paper'):
        render.render(ROOT/'paper/generated/would_be_wrong')


def engineering_fixture():
    rng=np.random.default_rng(8);arrays={};cases=[]
    for i,(label,gain) in enumerate([('Largest episode gain',12.4),('Median episode gain',1.2),('Largest regression',-23.4)]):
        p=f'case{i}';arrays[p+'_images']=rng.integers(0,255,(13,180,320,3),dtype=np.uint8)
        arrays[p+'_frame_indices']=np.arange(13)*5
        cases.append({'prefix':p,'label':label,'gain_percent':gain,'first_window_gain_percent':gain})
        for mode in replay.MODES:
            for s in (0,1,2):
                arrays[f'{p}_{mode}_s{s}_patch_errors']=rng.random((10,4,4))*.25
                if mode=='transport':
                    weights=rng.random((10,16,16));weights/=weights.sum(-1,keepdims=True)
                    arrays[f'{p}_{mode}_s{s}_matrix']=weights
                    arrays[f'{p}_{mode}_s{s}_gate']=np.ones((10,16,1))*.123
    measured={'sources':{},'cases':cases,'population':{'episodes':123,'positive_episodes':90,'negative_episodes':33,'pooled_relative_reduction_percent':1.23}}
    return measured,arrays


def test_unpublished_engineering_layout_has_native_width_proof(monkeypatch,tmp_path):
    # Artificial fixture is visibly watermarked and outside the project.
    monkeypatch.setattr(render,'check_measurements',lambda output:engineering_fixture())
    (tmp_path/'replay.json').write_text('{"engineering_fixture":true}')
    result=render.render(tmp_path,engineering_fixture=True)
    assert result['status']=='numeric_and_geometry_passed_visual_review_pending'
    assert result['shared_error_limits'][0]==0 and result['shared_mixing_limits']==[0,1]
    assert result['proof_compile_passes']==2
    assert 'ARTIFICIAL ENGINEERING' in (tmp_path/'caption.tex').read_text()
