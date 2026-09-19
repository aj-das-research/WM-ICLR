"""Gates/arithmetic/geometry tests; synthetic values exist only in test memory/tmp."""
import copy
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('iws_figure_test',ROOT/'paper/scripts/render_iws_results.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def fixture():
    result={'status':'complete_validated_development','scope':'internal_development','runs':[{}]*27,'numerical_results':{}}
    for i,task in enumerate(r.TASKS):
        anchor=np.linspace(.05,.8,59)*(i+1)
        # Preserve favorable, unfavorable and zero point estimates.
        factors=(.9,1.1,1.)
        curves={'autoregressive':(anchor*1.2).tolist(),'anchored_additive':anchor.tolist(),
                'persistence':(anchor*1.4).tolist(),'bounded_spatial_mix':(anchor*factors[i]).tolist()}
        gain=100*(anchor[-1]-curves['bounded_spatial_mix'][-1])/anchor[-1]
        result['numerical_results'][task]={'means_all59_offsets':curves,
            'h60_comparisons':{'anchored_additive':{'relative_mse_reduction_percent':gain,
                'method_minus_comparator':curves['bounded_spatial_mix'][-1]-anchor[-1],
                'paired95':{'gain':[gain-2,gain+2]}}},'eligible_trajectories':120+i,'total_windows':3000+i}
    return result


def test_pending_creates_no_directory_or_files(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'FINAL',tmp_path/'missing_finalizer.json')
    monkeypatch.setattr(r,'OUT',tmp_path/'must-not-exist')
    assert r.render(True)['outputs_written'] is False
    assert not r.OUT.exists()
    with pytest.raises(ValueError):r.render(False)


def test_incomplete_gate_does_not_draw_or_write(tmp_path,monkeypatch):
    final=tmp_path/'final.json';final.write_text('{}')
    monkeypatch.setattr(r,'FINAL',final);monkeypatch.setattr(r,'OUT',tmp_path/'not-created')
    monkeypatch.setattr(r,'load_evidence',lambda:({'status':'pending','reason':'not passed'},{}))
    assert r.render(True)['status']=='pending' and not r.OUT.exists()


def test_registered_reporter_change_is_rejected_before_import(tmp_path,monkeypatch):
    path=tmp_path/'changed.py';path.write_text('raise AssertionError("must not import")')
    monkeypatch.setattr(r,'REPORTER',path)
    with pytest.raises(ValueError,match='helper changed'):r.load_evidence()


def test_all708_values_and_all_signed_outcomes_are_retained():
    evidence=fixture();payload=r.plot_payload(evidence)
    assert payload['plotted_means']==708 and payload['x_H']==list(range(2,61))
    assert payload['stored_offsets']==list(range(1,60))
    for task in r.TASKS:
        assert payload['tasks'][task]['curves']==evidence['numerical_results'][task]['means_all59_offsets']
    assert payload['tasks']['pusht']['primary_gain_percent']==pytest.approx(10)
    assert payload['tasks']['bimanual_box']['primary_gain_percent']==pytest.approx(-10)
    assert payload['tasks']['bimanual_rope']['interval_contains_zero']
    assert '-10.00%' in r.annotation(payload['tasks']['bimanual_box'])[0]
    assert 'CI includes 0' in r.annotation(payload['tasks']['bimanual_rope'])[2]


@pytest.mark.parametrize('mutation',[
    lambda d:d.update(status='running'),lambda d:d.update(scope='official_validation'),
    lambda d:d['runs'].pop(),lambda d:d['numerical_results'].pop('pusht'),
    lambda d:d['numerical_results']['pusht']['means_all59_offsets'].pop('persistence'),
    lambda d:d['numerical_results']['pusht']['means_all59_offsets']['autoregressive'].pop(),
    lambda d:d['numerical_results']['pusht']['means_all59_offsets']['autoregressive'].__setitem__(0,np.nan),
    lambda d:d['numerical_results']['pusht']['means_all59_offsets']['autoregressive'].__setitem__(0,-1),
    lambda d:d['numerical_results']['pusht']['h60_comparisons']['anchored_additive'].update(relative_mse_reduction_percent=99),
    lambda d:d['numerical_results']['pusht']['h60_comparisons']['anchored_additive']['paired95'].update(gain=[4,1])])
def test_incomplete_or_contradictory_numerical_payload_fails(mutation):
    evidence=fixture();mutation(evidence)
    with pytest.raises(ValueError):r.plot_payload(evidence)


def test_undefined_ratio_is_not_zero():
    evidence=fixture();task=evidence['numerical_results']['pusht']
    task['means_all59_offsets']['anchored_additive']=[0.]*59
    effect=task['h60_comparisons']['anchored_additive'];effect.update(relative_mse_reduction_percent=None,
        method_minus_comparator=task['means_all59_offsets']['bounded_spatial_mix'][-1]);effect['paired95']['gain']=None
    row=r.plot_payload(evidence)['tasks']['pusht']
    assert row['primary_gain_percent'] is None and 'undefined' in r.annotation(row)[0]


def test_in_memory_geometry_all_curves_zero_axes_and_no_file_exports():
    payload=r.plot_payload(fixture());fig,audit=r.make_figure(payload)
    try:
        assert audit['issues']==[] and fig.get_size_inches().tolist()==[5.5,2.75]
        for axis,task in zip(fig.axes,r.TASKS):
            assert len(axis.lines)==4 and axis.get_ylim()[0]==0
            for line,mode in zip(axis.lines,r.MODES):
                np.testing.assert_array_equal(line.get_xdata(),payload['x_H'])
                np.testing.assert_array_equal(line.get_ydata(),payload['tasks'][task]['curves'][mode])
            assert axis.get_ylim()[1]>max(max(v) for v in payload['tasks'][task]['curves'].values())
    finally:plt.close(fig)


def test_idempotent_checksum_gate_and_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path)
    content={'fixture_only':True};fingerprint=r.digest(content)
    outputs={}
    for name in ('forecast_transfer.pdf','forecast_transfer.svg','forecast_transfer.png','forecast_transfer_figure.tex'):
        p=tmp_path/name;p.write_text('TEST FIXTURE, NOT A SCIENTIFIC FIGURE');outputs[name]=r.sha(p)
    (tmp_path/'forecast_transfer.json').write_text(json.dumps({'fingerprint':fingerprint,'bound_payload':content,'outputs_sha256':outputs}))
    before={p.name:p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    assert r.existing(fingerprint,content)
    assert before=={p.name:p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    (tmp_path/'forecast_transfer.pdf').write_text('tampered')
    with pytest.raises(ValueError,match='missing or changed'):r.existing(fingerprint,content)


def test_source_revision_cannot_silently_replace_old_render(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path)
    (tmp_path/'forecast_transfer.json').write_text(json.dumps({'fingerprint':'old','bound_payload':{}}))
    with pytest.raises(ValueError,match='different source snapshot'):r.existing('new',{})
