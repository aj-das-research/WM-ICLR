"""Synthetic checks only; these tests never open training or reserved inputs."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shiftwm.real_video_iws_reserved_recovery.prefix_backend import install_rowwise_gru, rowwise_gru_context
spec=importlib.util.spec_from_file_location('prefix_probe',ROOT/'scripts/real_video_iws_reserved_diagnostics_20260920/probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
torch.set_num_threads(2)


def model():
    torch.manual_seed(173)
    value=torch.nn.Module();value.action_prefix=torch.nn.GRU(14,96,batch_first=True)
    return value.eval()


def test_rowwise_exact_prefix_and_state_identity():
    value=model();inputs=torch.randn(4,60,14);before=probe.state_sha(value)
    keys=list(value.state_dict());counts=sum(x.numel() for x in value.parameters())
    assert install_rowwise_gru(value) is value
    assert install_rowwise_gru(value) is value
    with torch.inference_mode():
        full,_=value.action_prefix(inputs)
        for horizon in (15,30,45):
            prefix,hidden=value.action_prefix(inputs[:,:horizon])
            assert torch.equal(prefix,full[:,:horizon])
            assert torch.equal(hidden[0],full[:,horizon-1])
    assert before==probe.state_sha(value)
    assert keys==list(value.state_dict())
    assert counts==sum(x.numel() for x in value.parameters())


def test_context_restores_original_execution_after_exception():
    value=model();x=torch.randn(2,8,14)
    expected=value.action_prefix(x)[0]
    with pytest.raises(RuntimeError):
        with rowwise_gru_context(value):
            assert 'forward' in value.action_prefix.__dict__
            raise RuntimeError('test')
    assert 'forward' not in value.action_prefix.__dict__
    assert torch.equal(expected,value.action_prefix(x)[0])


@pytest.mark.parametrize('change',('training','float64','rank','empty'))
def test_invalid_execution_rejected(change):
    value=install_rowwise_gru(model());x=torch.randn(2,8,14)
    if change=='training':value.train()
    if change=='float64':x=x.double()
    if change=='rank':x=x[0]
    if change=='empty':x=x[:,:0]
    with pytest.raises(ValueError):value.action_prefix(x)


def test_incompatible_gru_rejected():
    value=model();value.action_prefix=torch.nn.GRU(14,96,2,batch_first=True)
    with pytest.raises(ValueError):install_rowwise_gru(value)


def test_rows_stream_without_full_array_load(tmp_path,monkeypatch):
    array=np.arange(100*7,dtype=np.float32).reshape(100,7)
    path=tmp_path/'input.npz';np.savez_compressed(path,features=array)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Full array loader forbidden'))
    assert np.array_equal(probe.rows_npz(path,'features',20,1,(100,7)),array[20:21])
    assert np.array_equal(probe.rows_npz(path,'features',3,60,(100,7)),array[3:63])
    with pytest.raises(ValueError):probe.rows_npz(path,'features',99,2,(100,7))
    with pytest.raises(ValueError):probe.rows_npz(path,'features',0,1,(99,7))


def test_discrepancy_quantifies_failure_and_exact_match():
    reference=torch.ones(2,3,4);value=reference.clone();value[1,2,3]+=.01
    result=probe.discrepancy(value,reference)
    assert not result['allclose'] and result['failed_coordinates']==1
    assert result['maximum_tolerance_ratio']>300
    assert result['maximum_ratio_coordinate']==[1,2,3]
    assert probe.discrepancy(reference,reference)['bitwise_equal']


def test_gru_first_divergence():
    reference=torch.zeros(2,8,3);value=reference.clone();value[1,3,2]=1e-7
    assert probe.gru_difference(value,reference)['first_bitwise_different_command_row']==3
    assert probe.gru_difference(reference,reference)['first_bitwise_different_command_row'] is None
