"""Completion, population, paired uncertainty and interruption regressions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name="_unbounded_finalizer_tests",file="finalize.py"):
    p=ROOT/"scripts/real_video_iws_unbounded"/file
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def fixtures(m):
    audits={t:{"eligible_episodes":2,"records":[{"episode_id":f"{i:06d}","windows":2} for i in range(2)]} for t in m.TASKS}
    rows=[]
    for ti,t in enumerate(m.TASKS):
        for mode in m.MODES:
            if mode=="persistence":continue
            value=(2. if ti==1 else .5) if mode==m.OURS else 1.
            for seed in range(3):
                episodes=[]
                for i in range(2):
                    e={"episode_id":f"{i:06d}","windows":2}
                    for metric in m.METRICS:
                        e[metric+"_by_offset"]=[value]*59
                        e["persistence_"+metric+"_by_offset"]=[2.]*59
                    episodes.append(e)
                rows.append({"task":t,"mode":mode,"seed":seed,"receipt":{"episodes":episodes}})
    return rows,audits


def test_complete_population_and_all_negative_signs_retained():
    m=load();rows,audits=fixtures(m);arrays=m.arrays_from_receipts(rows,audits)
    report=m.numerical_report(arrays)
    assert report["bootstrap"]["draws"]==10000 and report["bootstrap"]["seed"]==173
    for metric in m.METRICS:
        for ti,t in enumerate(m.TASKS):
            effect=report["task_results"][t][metric]["h60_comparisons"]["bounded_spatial_mix"]
            gain=-100. if ti==1 else 50.
            assert effect["relative_error_reduction_percent"]==gain
            assert effect["paired95"]["gain"]==[gain,gain]
        assert report["macro_h60"][metric]["bounded_spatial_mix"]["equal_task_relative_error_reduction_percent"]==0.
    text=m.table_text({"results":report})
    assert "-100.000000" in text and all(metric in text for metric in m.METRICS)
    assert all(str(h) in report["task_results"]["pusht"][m.METRICS[0]]["horizon_means"] for h in (15,30,45,60))


def test_missing_duplicate_or_substituted_run_population_rejected():
    m=load();rows,audits=fixtures(m)
    with pytest.raises(ValueError,match="all9"):m.arrays_from_receipts(rows[:-1],audits)
    rows[-1]=deepcopy(rows[0])
    with pytest.raises(ValueError,match="all9"):m.arrays_from_receipts(rows,audits)


def test_episode_window_counts_and_persistence_must_match():
    m=load();rows,audits=fixtures(m)
    changed=deepcopy(rows);changed[0]["receipt"]["episodes"][0]["windows"]+=1
    with pytest.raises(ValueError,match="population"):m.arrays_from_receipts(changed,audits)
    changed=deepcopy(rows);changed[0]["receipt"]["episodes"][0]["persistence_standardized_mse_by_offset"][0]+=.1
    with pytest.raises(ValueError,match="persistence"):m.arrays_from_receipts(changed,audits)


def test_primitive_persistence_change_rejected_even_when_episode_mean_equal(tmp_path):
    m=load();a={"episode_index":np.zeros(2,dtype=np.int64),"window_start":np.array([0,5],dtype=np.int64)}
    for metric in m.METRICS:a["persistence_"+metric]=np.ones((2,59),dtype=np.float64)
    b=deepcopy(a);b["persistence_standardized_mse"][0]=0;b["persistence_standardized_mse"][1]=2
    np.savez(tmp_path/"a.npz",**a);np.savez(tmp_path/"b.npz",**b)
    with pytest.raises(ValueError,match="Primitive"):m.primitive_persistence_equal(tmp_path/"a.npz",tmp_path/"b.npz")


def test_partial_campaign_writes_no_numeric_report(tmp_path,monkeypatch):
    m=load();monkeypatch.setattr(m.campaign,"REPORT",tmp_path/"reports")
    monkeypatch.setattr(m.campaign,"check_registration",lambda p:{"runs":[{"name":str(i)} for i in range(9)]})
    r=m.finalize(if_ready=True)
    assert r=={"status":"pending","completed_new_runs":0,"outputs_written":False}
    assert not (tmp_path/"reports").exists()


@pytest.mark.parametrize("partition",["ws-ia","gpu"])
def test_requeue_budget_is_below_eight_hour_allocation(partition,monkeypatch,tmp_path):
    m=load("_unbounded_budget_test","campaign.py");config=m.read(m.CONFIG);rows=m.grid(config)
    monkeypatch.setattr(m,"ROOT",tmp_path)
    monkeypatch.setattr(m,"check_registration",lambda p:{"runs":rows})
    monkeypatch.setenv("SLURM_JOB_PARTITION",partition);monkeypatch.setenv("SLURM_JOB_ID","123")
    calls=[]
    def run(command,**kw):calls.append(command);return SimpleNamespace(returncode=75 if "train" in command else 0)
    monkeypatch.setattr(m.subprocess,"run",run);m.run_index(m.CONFIG,0)
    command=calls[0];seconds=float(command[command.index("--max-runtime-seconds")+1])
    assert seconds==7.25*3600 and seconds<8*3600
    assert calls[-1]==["scontrol","requeue","123"]


def test_completed_training_retry_does_not_reenter_fit_or_rewrite_packages(tmp_path,monkeypatch):
    m=load("_unbounded_retry_test","campaign.py");config=m.read(m.CONFIG);rows=m.grid(config);row=rows[0]
    directory=tmp_path/row["output"];directory.mkdir(parents=True)
    summary=directory/"training_summary.json";summary.write_text(json.dumps({"status":"completed","completed_epochs":30,"resume_semantics":"keep"}))
    before=summary.read_bytes();report=tmp_path/"reports";ev=report/"evaluations"/(row["name"]+".json");ev.parent.mkdir(parents=True);ev.write_text('{}')
    recipe={"task":row["task"],"mode":row["mode"],"seed":row["seed"],"model":config["model"],"training":config["training"],"task_config":config["tasks"][row["task"]],"study_config_sha256":m.sha(m.CONFIG)}
    checks=[]
    fake=SimpleNamespace(validate_completed=lambda p:checks.append(p),read_package=lambda p:(p,{"config":{"metadata":{"identity":{"scientific_config":recipe}}}}))
    monkeypatch.setattr(m,"ROOT",tmp_path);monkeypatch.setattr(m,"REPORT",report)
    monkeypatch.setattr(m,"trainer",lambda:fake);monkeypatch.setattr(m,"check_registration",lambda p:{"runs":rows})
    calls=[]
    def run(command,**kw):
        calls.append(command)
        assert "train" not in command
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,"run",run)
    m.run_index(m.CONFIG,0)
    marker=report/"completions"/(row["name"]+".json");marker_bytes=marker.read_bytes()
    m.run_index(m.CONFIG,0)
    assert summary.read_bytes()==before and marker.read_bytes()==marker_bytes
    assert len(checks)==2 and sum(any(str(x).endswith("finalize.py") for x in c) for c in calls)==2


def test_committed_cpu_json_recovers_missing_adapter_without_new_evaluation(tmp_path,monkeypatch):
    m=load("_unbounded_adapter_retry_test","evaluate.py")
    real=m.campaign;config=real.read(real.CONFIG);row=real.grid(config)[0]
    report=tmp_path/"reports";output=report/"evaluations"/(row["name"]+".json")
    registration=tmp_path/"registration.json";registration.write_text('{}')
    package=tmp_path/row["output"]/"best";package.mkdir(parents=True);(package/"model.pt").write_bytes(b"selected")
    recipe={"task":row["task"],"mode":row["mode"],"seed":row["seed"],"model":config["model"],"training":config["training"],"task_config":config["tasks"][row["task"]],"study_config_sha256":real.sha(real.CONFIG)}
    trainer=SimpleNamespace(validate_completed=lambda p:{"best_epoch":30},read_package=lambda p:(package,{"config":{"metadata":{"identity":{"scientific_config":recipe}}}}),open_cache=lambda *a:None)
    interrupted=[False];calls=[]
    def atomic(value,path):
        if str(path).endswith('.adapter.json') and not interrupted[0]:
            interrupted[0]=True;raise RuntimeError('interruption after frozen JSON commit')
        real.atomic_json(value,path)
    c=SimpleNamespace(REPORT=report,REGISTRATION=registration,read=real.read,sha=real.sha,atomic_json=atomic,
                      check_registration=lambda p:{"runs":[row]},trainer=lambda:trainer)
    monkeypatch.setattr(m,"campaign",c);monkeypatch.setattr(m,"ROOT",tmp_path)
    def evaluate(*args,**kwargs):
        calls.append('inference');output.parent.mkdir(parents=True,exist_ok=True)
        np.savez(output.with_suffix('.npz'),metric=np.ones(1))
        output.write_text(json.dumps({'status':'passed','ledger_sha256':real.sha(output.with_suffix('.npz'))}))
    monkeypatch.setattr(m.frozen,"evaluate",evaluate)
    validations=[]
    def validate(path,*args):
        assert real.read(path)['ledger_sha256']==real.sha(path.with_suffix('.npz'))
        validations.append('full ledger')
    validator=SimpleNamespace(metadata_audit=lambda p:{},validate_evaluation=validate)
    monkeypatch.setattr(m,"load",lambda *a:validator)
    with pytest.raises(RuntimeError,match='interruption'):
        m.evaluate(real.CONFIG,row['name'],output)
    assert output.exists() and not output.with_suffix('.adapter.json').exists()
    before=output.read_bytes();result=m.evaluate(real.CONFIG,row['name'],output)
    assert result['status']=='passed' and calls==['inference'] and len(validations)==2
    assert output.read_bytes()==before and output.with_suffix('.adapter.json').exists()
