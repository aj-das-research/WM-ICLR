"""Presentation-only gates and scope; synthetic branches stay in tmp directories."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('site_iws',ROOT/'site/prepare_iws_results.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def test_all_completed_development_metrics_and_signed_effects():
    value=s.verified_results();d=value['development']
    assert d['completed_models']==36
    assert len(d['rows'])==3
    assert sum(len(curve) for row in d['rows'] for curve in row['curves'].values())==885
    assert sum(len(row['effects']['standardized_mse']) for row in d['rows'])==12
    for row in d['rows']:
        assert set(row['means'])==set(s.METRICS)
        assert all(set(means)==set(s.MODES) for means in row['means'].values())
        assert row['effects']['standardized_mse']['bounded_vs_ar']['gain']<0
    push=d['rows'][0]['effects']['standardized_mse']['bounded_vs_ar']['ci']
    assert push[0]<0<push[1]
    assert value['checkpoints']['local_iws_inference_bundles']==36
    assert value['checkpoints']['published_iws_bundles']==0


def test_reserved_absent_is_pending_without_loader(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'load_module',lambda *a:pytest.fail('No loader needed for absent pack'))
    value=s.reserved(tmp_path,{})
    assert value['status']=='pending_complete_reviewed_evidence'
    assert not value['rows']


def test_reviewed_reserved_mapping_matches_portable_final_results():
    pack=ROOT/s.RESERVED
    if not (pack/'manifest.json').exists():
        pytest.skip('Reviewed complete reserved pack has not arrived')
    source=json.loads((pack/'data.json').read_text())['results']
    value=s.reserved(ROOT,{})
    assert value['status']=='complete_reviewed_reserved'
    assert value['completed_models']==36
    assert sum(len(row['effects'][metric]) for row in value['rows'] for metric in s.METRICS)==60
    for row in value['rows']:
        assert row['population']=={'trajectories':10,'windows':200}
        for metric in s.METRICS:
            original=source['task_results'][row['task']][metric]
            assert row['means'][metric]==original['equal_trajectory']['horizon_means']['60']
            for key,effect in row['effects'][metric].items():
                expected=original['h60_comparisons'][key]
                assert effect['gain']==expected['relative_error_reduction_percent']
                assert effect['ci']==expected['paired95']['gain_percent']['percentile95']
    for metric,entries in value['macro'].items():
        for key,effect in entries.items():
            expected=source['macro_h60'][metric][key]
            assert effect['gain']==expected['equal_task_relative_error_reduction_percent']
            assert effect['ci']==expected['paired95']['percentile95']


def test_present_invalid_reserved_pack_is_rejected(tmp_path):
    pack=tmp_path/s.RESERVED;pack.mkdir(parents=True)
    (pack/'manifest.json').write_text('{"schema":"partial","status":"pending"}')
    scripts=tmp_path/'paper/scripts';scripts.mkdir(parents=True)
    # Real frozen presentation validator; no fixture data or model payload.
    (scripts/'render_iws_reserved_evidence.py').write_bytes((ROOT/'paper/scripts/render_iws_reserved_evidence.py').read_bytes())
    with pytest.raises(ValueError,match='Incomplete portable pack'):
        s.reserved(tmp_path,{})


def test_missing_method_and_bad_interval_rejected():
    means={m:float(i+1) for i,m in enumerate(s.MODES)}
    for e in [{'method':'made_up','reference':'autoregressive','gain':0,'ci':[0,1]},
              {'method':'bounded_spatial_mix','reference':'anchored_additive','gain':100*(3-4)/3,'ci':[2,1]},
              {'method':'bounded_spatial_mix','reference':'anchored_additive','gain':9,'ci':[0,1]}]:
        with pytest.raises(ValueError):s.check_effect(e,means)


def test_negative_rendering_preserves_sign_and_no_green():
    cell=s.gain_cell({'gain':-4.12,'ci':[-5.2,-3.1]})
    assert '-4.12%' in cell and '[-5.20, -3.10]' in cell and 'iws-positive' not in cell
    assert 'iws-positive' in s.gain_cell({'gain':1.0,'ci':[-1,2]})


def test_section_keeps_evaluation_scopes_and_followup_labels():
    value=s.verified_results();value['reserved']={'status':'pending_complete_reviewed_evidence','rows':[]}
    text=s.section(value)
    for expected in ['No-tanh (ours, ablation)','Completed internal development','Reserved upstream validation',
                     'All 36 IWS predictors','117 published predictors','not public checkpoint downloads',
                     'no partial accuracy values','higher endpoint MSE','PushT interval includes zero']:
        assert expected in text
    assert text.count('bounded')>=1


def test_idempotence_and_non_iws_bytes_preserved(tmp_path,monkeypatch):
    value=s.verified_results()
    # Keep ready/absent branch stable even if live reserved finalization arrives.
    monkeypatch.setattr(s,'verified_results',lambda root:copy.deepcopy(value))
    original='outside-before\n'+s.BEGIN+'old'+s.END+'\noutside-after'
    (tmp_path/'index.html').write_text(original)
    s.prepare(ROOT,tmp_path)
    first={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    s.prepare(ROOT,tmp_path)
    assert first=={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    text=(tmp_path/'index.html').read_text()
    assert text.split(s.BEGIN)[0]=='outside-before\n'
    assert text.split(s.END)[1]=='\noutside-after'
    assert (tmp_path/s.ASSET).read_bytes()==(ROOT/s.PLOT).read_bytes()


def test_publisher_manifest_maps_scopes_without_public_checkpoint_inflation():
    spec=importlib.util.spec_from_file_location('iws_site_publish',ROOT/'site/publish.py')
    publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
    value=s.verified_results()
    metadata=publisher.iws_publication_metadata(value)
    assert 'iws-development-forecast.svg' in publisher.ASSETS
    assert metadata['iws_development']['completed_models']==36
    assert metadata['iws_development']['original_models']==27
    assert metadata['iws_development']['exploratory_followup_models']==9
    assert metadata['iws_checkpoint_availability']['published_other_predictors']==117
    assert metadata['iws_checkpoint_availability']['published_iws_bundles']==0
    assert metadata['iws_checkpoint_availability']['local_iws_inference_bundles']==36
    value['reserved']={'status':'pending_complete_reviewed_evidence','scope':'reserved_upstream_validation','rows':[]}
    pending=publisher.iws_publication_metadata(value)['iws_reserved']
    assert set(pending)=={'status','scope'}
    value['development']['completed_models']=27
    with pytest.raises(ValueError,match='complete 36-model'):
        publisher.iws_publication_metadata(value)
