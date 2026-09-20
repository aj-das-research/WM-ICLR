"""Presentation gates; all mutations stay in temporary fixture directories."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("diagnostic_tables", Path(__file__).with_name("render_tables.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def test_reviewed_rows_and_units():
    data = r.payload()
    text = r.geometry_tex(data)
    assert len(data["rows"]) == 6
    assert text.count("Constant dynamics &") == 3
    assert text.count("ShiftWM (ours) &") == 3
    assert r"Canonical separation (\%)" in text
    for row in data["rows"]:
        for family in ("narrow", "wide"):
            assert f"{100*row[family]['fraction_of_canonical_pairwise_mse']:.3f}" in text
            assert f"{row[family]['spearman_with_squared_xy_distance']:.5f}" in text
    assert "132 / 132" in r.tissue_tex(data)


def test_missing_review_writes_nothing(tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    monkeypatch.setattr(r, "SOURCE", source)
    monkeypatch.setattr(r, "DEST", dest)
    monkeypatch.setattr(sys, "argv", ["render_tables.py", "--if-ready"])
    r.main()
    assert not dest.exists()


@pytest.mark.parametrize("mutation", ["status", "results_sha256"])
def test_corrupt_present_review_rejected(tmp_path, monkeypatch, mutation):
    for name in ("registration", "results", "completion", "execution_review"):
        shutil.copy2(r.SOURCE / (name + ".json"), tmp_path / (name + ".json"))
    review = r.read(tmp_path / "execution_review.json")
    review[mutation] = "bad"
    (tmp_path / "execution_review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match="review"):
        r.payload(tmp_path)


def test_idempotent_presentation_write(tmp_path):
    path = tmp_path / "table.tex"
    r.put(path, "same\n")
    stamp = path.stat().st_mtime_ns
    r.put(path, "same\n")
    assert path.stat().st_mtime_ns == stamp
