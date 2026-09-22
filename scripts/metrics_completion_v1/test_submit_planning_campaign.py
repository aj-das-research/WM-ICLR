"""Operational scheduling contracts; no Slurm submission or payload reads."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import submit_planning_campaign as s


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'REPORT',tmp_path)
    monkeypatch.setattr(s,'checked_registration',lambda: {})
    monkeypatch.setattr(s,'sha',lambda p:'a'*64)
    return tmp_path


def test_dry_run_has_no_submission(sandbox,monkeypatch):
    monkeypatch.setattr(s.subprocess,'run',lambda *a,**k:pytest.fail('Dry run submitted'))
    plan=s.submit(['101','102'])
    assert len(plan['stages'])==5
    assert not (sandbox/'campaign_submission.json').exists()
    with pytest.raises(ValueError):s.submit(['101','abc'])


def test_durable_exact_dependencies_and_quota(sandbox,monkeypatch):
    calls=[]
    def fake(command,**unused):
        calls.append(command)
        return SimpleNamespace(stdout=str(200+len(calls))+';cluster\n')
    monkeypatch.setattr(s.subprocess,'run',fake)
    result=s.submit(['101','102'],True)
    assert result['status']=='submitted'
    assert '--dependency=afterok:101:102' in calls[0] and '--dependency=afterok:101:102' in calls[1]
    assert '--dependency=afterok:201:202' in calls[2] and '--dependency=afterok:201:202' in calls[3]
    assert '--dependency=afterok:203:204' in calls[4]
    assert next(x for x in calls[0]if x.startswith('--array=')).endswith('%2')
    assert next(x for x in calls[1]if x.startswith('--array=')).endswith('%1')
    with pytest.raises(ValueError):s.submit(['101','102'],True)
    assert len(calls)==5


def test_failure_retains_submitted_ids_and_refuses_duplicate(sandbox,monkeypatch):
    calls=[]
    def fake(command,**unused):
        calls.append(command)
        if len(calls)==2:raise RuntimeError('scheduler unavailable')
        return SimpleNamespace(stdout='201\n')
    monkeypatch.setattr(s.subprocess,'run',fake)
    with pytest.raises(RuntimeError):s.submit(['101','102'],True)
    saved=json.loads((sandbox/'campaign_submission.json').read_text())
    assert saved['status']=='submission_failed_inspect_saved_jobs_before_retry'
    assert saved['jobs']['train_ws']['job_id']=='201'
    with pytest.raises(ValueError):s.submit(['101','102'],True)
