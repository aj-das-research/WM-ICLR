"""Bounded finalization tests: gates, honest signs, exact portable local weights."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from test_real_video_training import model, settings, identity, TinyDataset

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("real_video_finalizer", ROOT / "scripts/real_video/finalize_campaign.py")
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


def aggregate_fixture(reference=2., ours=3., interval=(.2, 1.8)):
    metric = "mean_standardized_mse"
    values = {"mean": ours, "seed_sd": .1, "per_seed": [ours-.1, ours, ours+.1]}
    return {"camera":"exterior_image_1_left","horizon":5,"population":"primary",
            "methods":{"factorized":{metric:values},
                       "framewise":{metric:{**values,"mean":reference}}},
            "fixed_support_baselines":{},
            "paired_comparisons":[{"method":"factorized","reference":"framewise","metric":metric,
                                   "mean_difference":ours-reference,"ci95":list(interval)}]}


@pytest.mark.parametrize("ours,interval,percent,status", [
    (3.,(.2,1.8),-50.,"interval above zero"),
    (1.,(-1.8,-.2),50.,"interval below zero"),
    (1.9,(-.8,.3),5.,"interval includes zero"),
    (2.,(0.,0.),0.,"interval includes zero")])
def test_gain_signs_keep_regressions_and_inconclusive_intervals(ours,interval,percent,status):
    rows = finalizer.gain_rows(aggregate_fixture(ours=ours,interval=interval))
    assert rows[0]["relative_mse_reduction_percent"] == pytest.approx(percent)
    assert rows[0]["interval_description"] == status
    assert rows[0]["mean_difference"] == ours-2.


def test_zero_reference_has_no_invented_relative_gain():
    result = finalizer.gain_rows(aggregate_fixture(reference=0.,ours=1.))
    assert result[0]["relative_mse_reduction_percent"] is None
    wrong = aggregate_fixture()
    wrong["paired_comparisons"][0]["mean_difference"] = -1
    with pytest.raises(ValueError,match="Point estimates"):
        finalizer.gain_rows(wrong)


def test_campaign_fails_closed_before_any_release_on_missing_runs(tmp_path):
    rows = []
    for mode in finalizer.train.RealVideoWorldModel.MODES:
        for seed in (0,1,2):
            config = tmp_path/f"{mode}_{seed}.json"
            finalizer.train.atomic_json({"output_dir":str(tmp_path/f"run_{mode}_{seed}")},config)
            rows.append({"name":f"{mode}_{seed}","mode":mode,"seed":seed,"config":str(config)})
    campaign = tmp_path/"campaign.json"
    finalizer.train.atomic_json({"runs":rows},campaign)
    table = tmp_path/"paper/real_video_results.tex"
    table.parent.mkdir()
    table.write_text("Previously generated result table")
    table.with_suffix(".sources.json").write_text('{"status":"completed"}')
    with pytest.raises(ValueError,match="Incomplete campaign"):
        finalizer.finalize(campaign,tmp_path/"results",tmp_path/"report",tmp_path/"release",tmp_path/"encoder",
                           paper_table=table)
    assert not (tmp_path/"report.json").exists() and not (tmp_path/"release").exists()
    assert not table.exists() and not table.with_suffix(".sources.json").exists()
    archive = list((tmp_path/".real_droid_table_history").glob("*/real_video_results.tex"))
    assert len(archive) == 1 and archive[0].read_text() == "Previously generated result table"
    rows[-1]["mode"] = "framewise"
    finalizer.train.atomic_json({"runs":rows},campaign)
    with pytest.raises(ValueError,match="exactly"):
        finalizer.collect_campaign(campaign,tmp_path/"results")


def test_copy_rejects_wrong_source_and_encoder_identity(tmp_path):
    source = tmp_path/"source.txt"
    source.write_text("actual source")
    with pytest.raises(ValueError,match="Source differs"):
        finalizer.checked_copy(source,tmp_path/"copy.txt",expected="wrong")
    encoder = tmp_path/"encoder"
    encoder.mkdir()
    finalizer.train.atomic_json({"files":[],"revision":"actual"},encoder/"provenance.json")
    with pytest.raises(ValueError,match="encoder differs"):
        finalizer.encoder_bundle(tmp_path/"stage",encoder,{"files":[],"revision":"wrong"})


def test_report_prints_negative_relative_gain_and_correct_units():
    aggregate = aggregate_fixture()
    for values in aggregate["methods"].values():
        for h in (1,3,5):
            values[f"h{h}_standardized_mse"] = values["mean_standardized_mse"].copy()
    report = {"completed_runs":12,"completed_evaluations":48,
              "populations":[{"aggregate":aggregate,"gains":finalizer.gain_rows(aggregate)}]}
    markdown = finalizer.render_report(report)
    assert "-50.00%" in markdown and "+1 [+0.2, +1.8]" in markdown
    assert "not percentage points" in markdown and "ShiftWM real-video variant (ours)" in markdown
    assert "not closed-loop robot success" in markdown and "inconclusive" in markdown


def table_report(ours=1.9):
    populations = []
    for camera,horizon in finalizer.POPULATIONS:
        aggregate = aggregate_fixture(ours=ours,interval=(-.4,.2))
        aggregate.update(camera=camera,horizon=horizon,population="primary" if camera==finalizer.evaluate.CAMERAS[0] and horizon==5 else "transfer")
        for mode in ("constant_dynamics","action_free"):
            aggregate["methods"][mode] = deepcopy(aggregate["methods"]["framewise"])
        for values in aggregate["methods"].values():
            for h in ((1,3,5) if horizon==5 else (10,)):
                values[f"h{h}_standardized_mse"] = values["mean_standardized_mse"].copy()
        for baseline in ("persistence","constant_velocity"):
            aggregate["fixed_support_baselines"][baseline] = {
                metric:3. for metric in aggregate["methods"]["factorized"]}
        aggregate["paired_comparisons"] = [
            {"method":"factorized","reference":mode,"metric":f"h{h}_standardized_mse",
             "mean_difference":ours-2.,"ci95":[-.4,.2]} for mode in ("framewise","constant_dynamics")
             for h in ((1,3,5) if horizon==5 else (10,))]
        populations.append({"aggregate":aggregate,"gains":finalizer.gain_rows(aggregate)})
    return {"completed_runs":12,"completed_evaluations":48,"populations":populations}


def test_paper_tables_gate_green_only_positive_ours_and_keep_both_intervals(tmp_path):
    report = table_report()
    latex = finalizer.render_paper_tables(report)
    assert latex.count(r"\begin{table}") == 2
    assert latex.count(r"\positivegain{+5.00}") == 1
    assert "Framewise: $-0.1" in latex and "Constant dynamics: $-0.1" in latex
    assert "includes zero" in latex and "not significance" in latex
    assert "not percentage points" in latex and "closed-loop robot success" in latex
    negative = finalizer.render_paper_tables(table_report(ours=3.))
    assert r"\positivegain{-50.00}" not in negative and " & -50.00" in negative
    report["completed_evaluations"] = 47
    with pytest.raises(ValueError,match="complete registered"):
        finalizer.render_paper_tables(report)
    if shutil.which("pdflatex"):
        (tmp_path/"table.tex").write_text(latex)
        (tmp_path/"proof.tex").write_text(
            r"\documentclass{article}\usepackage[paperwidth=7.5in,margin=1in]{geometry}\usepackage{booktabs,xcolor,colortbl}"
            r"\newcommand{\positivegain}[1]{\textcolor{green!50!black}{\textbf{#1}}}"
            r"\begin{document}\input{table}\end{document}")
        result = subprocess.run(["pdflatex","-halt-on-error","-interaction=batchmode","proof.tex"],
                                cwd=tmp_path,capture_output=True,text=True)
        assert result.returncode == 0, (tmp_path/"proof.log").read_text()
        assert "Overfull" not in (tmp_path/"proof.log").read_text()


def test_inference_export_roundtrip_and_isolated_bundled_source(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run = tmp_path/"run"
        cfg = settings(run)
        summary = finalizer.train.fit(model(),cfg,TinyDataset(),TinyDataset(),identity(cfg))
        record = {"row":{"name":"droid_factorized_s0","mode":"factorized","seed":0},
                  "summary":summary,"run":run}
        stage = tmp_path/"portable"
        sample = TinyDataset()[0]
        export = finalizer.export_model(record,stage,sample)
        assert export["checkpoint_sha256"] == finalizer.sha256(run/"best/model.pt")
        assert not (stage/"models/droid_factorized_s0/training_state.pt").exists()
        loaded,state = finalizer.train.load_package(stage/"models/droid_factorized_s0")
        assert state["epoch"] == summary["best_epoch"]
        # Twelve independently loadable copies exercise the isolated package
        # loader. They are explicitly test fixtures, not twelve scientific runs.
        for i in range(1,12):
            shutil.copytree(stage/"models/droid_factorized_s0",stage/"models"/f"fixture_{i:02}")
        source_files = ["src/shiftwm/__init__.py","src/shiftwm/model.py","src/shiftwm/upstream.py",
                        "src/shiftwm/checkpoint.py","src/shiftwm/real_video/__init__.py",
                        "src/shiftwm/real_video/model.py","src/shiftwm/real_video/data.py",
                        "src/shiftwm/vendor/lewm/module.py","src/shiftwm/vendor/lewm/NOTICE.json",
                        "scripts/real_video/train.py"]
        for path in source_files:
            finalizer.checked_copy(ROOT/path,stage/"source"/path)
        (stage/"verification").mkdir()
        np.savez(stage/"verification/sample.npz",features=sample["features"].numpy(),actions=sample["actions"].numpy())
        check = finalizer.verify_offline(stage)
        assert check["status"] == "passed" and len(check["models"]) == 12
        assert all(row["maximum_absolute_difference"] < 1e-6 for row in check["models"])
        assert not list((stage/"source").rglob("*.pyc"))
    finally:
        torch.set_num_threads(old_threads)
