"""Only in-memory/temp synthetic five-method tests; no production mock outputs."""
import copy
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('variant_forecast_test',ROOT/'paper/scripts/render_iws_variant_forecasts.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


@pytest.fixture
def current():return r.load_pack(r.PACK)


@pytest.fixture
def synthetic_five(current):
    """Explicit synthetic adverse ablation; NOT an experimental result."""
    data=copy.deepcopy(current);iws=data['iws'];iws['unbounded_included']=True
    source='reports/real_video_iws_unbounded/development_finalization.json'
    data['source_sha256'][source]='0'*64
    iws['completion']['unbounded_runs']=9;iws['completion']['unbounded_finalization_sha256']='0'*64
    iws['unbounded_scope']='SYNTHETIC TEST FIXTURE ONLY'
    for task in r.TASKS:
        old=iws['tasks'][task]['methods']['bounded_spatial_mix']
        extra=copy.deepcopy(old)
        for seed,metrics in extra['per_seed_curves'].items():
            for metric,curve in metrics.items():metrics[metric]=[1.2*v for v in curve]
        extra['mean_curves']={m:np.asarray([extra['per_seed_curves'][str(s)][m] for s in range(3)]).mean(0).tolist() for m in old['mean_curves']}
        iws['tasks'][task]['methods']['unbounded_spatial_mix']=extra
    return data


def save_pack(path,data):
    path.mkdir();(path/'data.json').write_text(json.dumps(data))
    (path/'manifest.json').write_text(json.dumps({'schema':'current_real_scorecards_pack_v1','data_sha256':r.sha(path/'data.json')}))


def test_current_pending_writes_zero_files(current,tmp_path):
    out=tmp_path/'never-created'
    assert r.render(True,out)['status']=='pending'
    assert not out.exists()
    with pytest.raises(ValueError,match='nine-plus-27'):r.render(False,out)
    assert not out.exists()


def test_absent_pack_pending_writes_nothing(tmp_path):
    assert r.render(True,tmp_path/'output',tmp_path/'missing')['outputs_written'] is False
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('change',['partial','missing_method','bad_curve','reserved','bad_hash'])
def test_invalid_present_pack_fails_before_output(synthetic_five,tmp_path,change):
    data=copy.deepcopy(synthetic_five)
    if change=='partial':data['iws']['completion']['unbounded_runs']=3
    elif change=='missing_method':del data['iws']['tasks']['pusht']['methods']['autoregressive']
    elif change=='bad_curve':data['iws']['tasks']['pusht']['methods']['unbounded_spatial_mix']['mean_curves']['standardized_mse'][58]=99
    elif change=='reserved':data['official_validation_payloads_read']=1
    pack=tmp_path/'SYNTHETIC-pack';save_pack(pack,data)
    if change=='bad_hash':(pack/'data.json').write_text('{}')
    with pytest.raises(ValueError):r.render(True,tmp_path/'must-not-exist',pack)
    assert not (tmp_path/'must-not-exist').exists()


def test_all885_values_and_adverse_ablation_retained(synthetic_five):
    payload=r.plot_payload(synthetic_five);figure,geometry=r.make_figure(payload)
    try:
        assert payload['plotted_means']==885 and payload['stored_offsets']==list(range(1,60))
        assert figure.get_size_inches().tolist()==[5.5,2.4] and geometry['minimum_font_pt']==8
        assert min(geometry['adjacent_tick_clearance_pt'])>=3
        assert geometry['issues']==[]
        for axis,task in zip(figure.axes,r.TASKS):
            assert len(axis.lines)==5 and len(axis.collections)==0 and axis.get_ylim()[0]==0
            for line,mode in zip(axis.lines,r.MODES):
                np.testing.assert_array_equal(line.get_xdata(),np.arange(2,61))
                np.testing.assert_array_equal(line.get_ydata(),synthetic_five['iws']['tasks'][task]['methods'][mode]['mean_curves']['standardized_mse'])
            assert axis.lines[-1].get_ydata()[-1]>axis.lines[-2].get_ydata()[-1]
        assert r.COLORS['bounded_spatial_mix']!=r.COLORS['unbounded_spatial_mix']
        assert r.STYLES['bounded_spatial_mix']!=r.STYLES['unbounded_spatial_mix']
    finally:plt.close(figure)


def test_four_method_proof_cannot_write_manuscript_output(tmp_path):
    with pytest.raises(ValueError,match='outside manuscript'):r.render(proof=True)
    result=r.render(output=tmp_path/'DESIGN-PROOF',proof=True)
    assert result['plotted_means']==708 and not list((tmp_path/'DESIGN-PROOF').glob('*.tex'))
    assert (tmp_path/'DESIGN-PROOF'/f'{r.PROOF_PREFIX}.pdf').exists()


def test_full_fixture_export_idempotence_caption_and_tamper(synthetic_five,tmp_path):
    pack=tmp_path/'SYNTHETIC-pack';save_pack(pack,synthetic_five)
    out=tmp_path/'SYNTHETIC-OUTPUT-NOT-PAPER'
    assert r.render(output=out,pack=pack)['plotted_means']==885
    before={p.name:p.read_bytes() for p in out.iterdir()}
    assert r.render(output=out,pack=pack)['outputs_written'] is False
    assert before=={p.name:p.read_bytes() for p in out.iterdir()}
    include=(out/f'{r.PREFIX}_figure.tex').read_text()
    assert r'\label{fig:iws-forecast-transfer}' in include and r'\ref{tab:iws-unbounded-component}' in include
    (out/f'{r.PREFIX}.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='output changed'):r.render(output=out,pack=pack)
