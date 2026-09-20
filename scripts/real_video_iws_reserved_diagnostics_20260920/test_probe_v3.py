import importlib.util
from pathlib import Path
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('probe_v3_test',Path(__file__).with_name('probe_v3.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_scorer_only_gets_zero_synthetic_targets_and_discards_metrics(monkeypatch):
    model=torch.nn.Module();model.action_prefix=torch.nn.GRU(14,6,batch_first=True);model.eval()
    initial=torch.ones(2,6144);commands=torch.ones(2,60,14)
    calls=[]
    def score(value,batch,offset):
        assert value is model and offset==64
        assert torch.is_inference_mode_enabled() and batch['initial_features'].is_inference()
        assert not next(model.parameters()).is_inference()
        assert batch['targets'].shape==(2,59,6144) and torch.count_nonzero(batch['targets'])==0
        assert torch.equal(batch['initial_features'],initial) and torch.equal(batch['commands'],commands)
        calls.append(True)
        return {k:np.full((2,1),123.456) for k in probe.probe.evaluation.metric_keys()},[
            {'command_rows':60,'prediction_sha256':'a'*64},
            *[{'command_rows':h,'maximum_absolute_difference':0.,'maximum_tolerance_ratio':0.} for h in (15,30,45)]]
    monkeypatch.setattr(probe.probe.evaluation,'score_batch',score)
    for rowwise in (False,True):
        result=probe.score_synthetic(model,initial,commands,64,rowwise)
        assert result['status']=='passed' and result['all_synthetic_metric_values_discarded']
        assert '123.456' not in str(result)
    assert len(calls)==2 and 'forward' not in model.action_prefix.__dict__


def test_original_exception_preserved_without_metric_output(monkeypatch):
    model=torch.nn.Linear(2,2).eval()
    def fail(*args):raise ValueError('registered prefix tolerance exceeded')
    monkeypatch.setattr(probe.probe.evaluation,'score_batch',fail)
    result=probe.score_synthetic(model,torch.zeros(2,6144),torch.zeros(2,60,14),0,False)
    assert result['status']=='failed' and result['reason']=='registered prefix tolerance exceeded'
    assert result['all_synthetic_metric_values_discarded']
