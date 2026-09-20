"""Portable resource tables must not silently change timing coverage or aggregation."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/scripts/render_iws_predictor_resources.py"
PACK = ROOT / "paper/table_sources/iws_predictor_resources_v1"
spec = importlib.util.spec_from_file_location("resource_table", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def pack(tmp_path):
    destination = tmp_path / "pack"
    shutil.copytree(PACK, destination)
    return destination


def mutate(pack, operation):
    data = json.loads((pack / "data.json").read_text())
    operation(data)
    (pack / "data.json").write_text(json.dumps(data))
    manifest = json.loads((pack / "manifest.json").read_text())
    manifest["runtime_inputs_sha256"]["data.json"] = module.sha(pack / "data.json")
    (pack / "manifest.json").write_text(json.dumps(manifest))


def test_real_complete_data_and_idempotence(pack, tmp_path):
    output = tmp_path / "out"
    assert module.render(pack, output)["outputs_written"]
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    assert not module.render(pack, output)["outputs_written"]
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    text = (output / "predictor_resources.tex").read_text()
    assert text.count("No tanh (ablation)") == 3
    assert "0.042" in text and "not a confidence interval" in text
    assert "tab:iws-predictor-resources" in text


def test_raw_timing_tamper_fails_even_with_new_pack_hash(pack):
    mutate(pack, lambda d: d["all39_case_measurements"][0]["devices"]["cpu"]["latency"].update(samples_ms=[1.0] * 30))
    with pytest.raises(ValueError, match="Case median differs"):
        module.checked_pack(pack)


def test_missing_seed_fails(pack):
    mutate(pack, lambda d: d["all39_case_measurements"].pop())
    with pytest.raises(ValueError, match="model grid"):
        module.checked_pack(pack)


def test_mean_cannot_replace_median(pack):
    def change(data):
        device = data["rows"][0]["devices"]["cpu"]
        device["median_of_seed_medians_ms"] = device["mean_of_seed_medians_ms"]
    mutate(pack, change)
    with pytest.raises(ValueError, match="Display median"):
        module.checked_pack(pack)


def test_memory_aggregation_tamper_fails(pack):
    mutate(pack, lambda d: d["rows"][0]["devices"]["cuda"]["memory"].update(maximum_peak_allocated_bytes=1))
    with pytest.raises(ValueError, match="Memory maximum"):
        module.checked_pack(pack)


@pytest.mark.parametrize("field,value", [("unsupported_states", ["cuda"]), ("official_validation_payloads_read", 1)])
def test_scope_or_unsupported_fails(pack, field, value):
    mutate(pack, lambda d: d.update({field: value}))
    with pytest.raises(ValueError):
        module.checked_pack(pack)


def test_public_copy_needs_only_three_files(tmp_path):
    root = tmp_path / "isolated"
    source = root / "paper/scripts" / SOURCE.name
    source.parent.mkdir(parents=True)
    shutil.copy2(SOURCE, source)
    pack = root / "paper/table_sources/iws_predictor_resources_v1"
    pack.mkdir(parents=True)
    for name in ("data.json", "manifest.json"):
        shutil.copy2(PACK / name, pack / name)
    subprocess.run([sys.executable, "-I", str(source)], cwd=root, check=True, capture_output=True, text=True)
    assert (root / "paper/generated/iws_resources/predictor_resources.tex").read_bytes() == (ROOT / "paper/generated/iws_resources/predictor_resources.tex").read_bytes()
