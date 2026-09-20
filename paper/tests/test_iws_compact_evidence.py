"""Protect data/labels while consolidating the completed IWS display."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/render_iws_compact_evidence.py'
spec=importlib.util.spec_from_file_location('iws_compact',P)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

@pytest.fixture
def data(): return r.load()

def test_all_curve_values_and_both_signs_retained(data):
    fig,qa=r.make_figure(data)
    try:
        assert qa['curve_points']==885 and qa['paired_intervals']==6
        for axis,task in zip(fig.axes[::2],r.TASKS):
            assert len(axis.lines)==5
            for line,mode in zip(axis.lines,r.MODES):
                np.testing.assert_array_equal(line.get_ydata(),data['tasks'][task]['curves'][mode])
            assert data['tasks'][task]['effects']['bounded_vs_ar']['gain'] < 0
            assert data['tasks'][task]['effects']['unbounded_vs_ar']['gain'] > 0
        assert data['tasks']['pusht']['effects']['bounded_vs_ar']['ci']['gain'][1]>0
        assert data['tasks']['bimanual_rope']['effects']['bounded_vs_additive']['gain']<0
    finally:r.plt.close(fig)

@pytest.mark.parametrize('change',['partial','missing_curve','wrong_reference','wrong_sign','bad_interval','wrong_scope','reserved'])
def test_invalid_evidence_rejected(data,change):
    row=data['tasks']['pusht'];e=row['effects']['unbounded_vs_ar']
    if change=='partial':data['completed_runs']=35
    elif change=='missing_curve':row['curves']['persistence'].pop()
    elif change=='wrong_reference':e['reference']='anchored_additive'
    elif change=='wrong_sign':e['gain']=-e['gain']
    elif change=='bad_interval':e['ci']['gain'].reverse()
    elif change=='wrong_scope':e['study']='original27'
    else:data['official_validation_payloads_read']=1
    with pytest.raises(ValueError):r.validate(data)

def test_compact_table_contains_all_twelve_nonempty_contrasts(data):
    text=r.contrast_table(data)
    for label in r.CONTRAST_LABELS:assert text.count(' & '+label+' & ')==3
    assert r'\missing' not in text and '---' not in text
    for task in r.TASKS:
        for e in data['tasks'][task]['effects'].values():
            assert f"{e['difference']:+.5f}" in text
            for x in e['ci']['difference']: assert f'{x:+.5f}' in text

def test_public_pack_replay_has_identical_exports(data,tmp_path):
    one=r.render(tmp_path/'one');two=r.render(tmp_path/'two')
    assert one['outputs_sha256']==two['outputs_sha256']
