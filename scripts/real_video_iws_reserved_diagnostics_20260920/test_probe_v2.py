"""Synthetic allocation-path regression checks; no model/data payload reads."""
import importlib.util
from pathlib import Path
import torch
import pytest

spec=importlib.util.spec_from_file_location('probe_v2_test',Path(__file__).with_name('probe_v2.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_model_ordinary_inputs_inference_and_context_restored(monkeypatch):
    model=torch.nn.Linear(2,2).eval();initial=torch.zeros(2,2);commands=torch.zeros(2,3,2)
    def inspect(value,z,a):
        assert value is model and not next(value.parameters()).is_inference()
        assert torch.is_inference_mode_enabled() and z.is_inference() and a.is_inference()
        assert torch.equal(z,initial) and torch.equal(a,commands)
        return {'checked':True}
    monkeypatch.setattr(probe,'diagnose_inference',inspect)
    assert probe.diagnose(model,initial,commands)=={'checked':True}
    assert not torch.is_inference_mode_enabled()


def test_wrong_model_allocation_rejected():
    with torch.inference_mode():model=torch.nn.Linear(2,2).eval()
    with pytest.raises(ValueError,match='allocation'):
        probe.diagnose(model,torch.zeros(2,2),torch.zeros(2,3,2))
