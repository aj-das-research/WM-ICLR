"""Release integrity tests; synthetic fixtures never become research reports."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generalization_release", ROOT / "scripts/publishing/publish_generalization_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def fixture_documents():
    rows = [{"name": f"{arm}_{mode}_s{seed}", "arm": arm, "mode": mode, "seed": seed,
             "training": {"status": "completed", "completed_epochs": 30, "best_epoch": 2}}
            for arm in release.ARMS for mode in release.MODES for seed in range(3)]
    aggregates = [{"arm": arm, "horizon": horizon,
        "methods": {mode: {"mean": 0.1, "gain_percent": -4.0 if mode == "factorized" else 0.0} for mode in release.MODES}}
        for arm in release.ARMS for horizon in (5, 10)]
    report = {"status": "completed", "completed_training_runs": 36, "completed_validation_evaluations": 72,
        "registered_checkpoint_parity_passes": 36, "independent_cpu_reload_parity_passes": 36,
        "registration_sha256": "registry", "review": {"status": "passed"}, "paper_build": {"status": "passed"},
        "scope": "original validation development only; no original or fresh test payloads read",
        "runs": rows, "aggregate": aggregates}
    return report, {"runs": deepcopy(rows)}


def test_complete_matched_study_passes_only_all_required_gates():
    report, registry = fixture_documents()
    release.validate_completion_gate(report, registry, "registry")


@pytest.mark.parametrize("mutation", ["count", "evaluations", "parity", "paper", "epoch", "scope", "duplicate", "registration", "arm", "nan"])
def test_incomplete_or_wrong_study_cannot_be_published(mutation):
    report, registry = fixture_documents()
    if mutation == "count": report["completed_training_runs"] = 35
    elif mutation == "evaluations": report["completed_validation_evaluations"] = 71
    elif mutation == "parity": report["independent_cpu_reload_parity_passes"] = 35
    elif mutation == "paper": report["paper_build"]["status"] = "pending"
    elif mutation == "epoch": report["runs"][0]["training"]["completed_epochs"] = 29
    elif mutation == "scope": report["scope"] = "fresh test"
    elif mutation == "duplicate": report["runs"][-1] = deepcopy(report["runs"][0])
    elif mutation == "registration": report["registration_sha256"] = "other"
    elif mutation == "arm": report["aggregate"] = report["aggregate"][:-1]
    elif mutation == "nan": report["aggregate"][0]["methods"]["framewise"]["mean"] = float("nan")
    with pytest.raises(ValueError, match="Completion gate"):
        release.validate_completion_gate(report, registry, "registry")


def test_readme_retains_regressions_and_development_scope():
    report, _ = fixture_documents()
    text = release.readme(report)
    assert text.count("-4.000%") == 6
    assert "not RGB video" in text
    assert "within an arm" in text
    assert "neither" in text.lower()


def fixture_bundle(tmp_path):
    bundle = tmp_path / "shiftwm-generalization-v1"
    (bundle / "verification").mkdir(parents=True)
    for name in ("README.md", "MODEL_CARD.md", "verification/offline_packages.json"):
        (bundle / name).write_text("Synthetic integrity-test fixture\n")
    manifest = {"files": {str(p.relative_to(bundle)): {"bytes": p.stat().st_size, "sha256": release.sha(p)}
                          for p in bundle.rglob("*") if p.is_file()}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle, manifest


def test_archive_is_verified_against_the_complete_file_inventory(tmp_path):
    bundle, manifest = fixture_bundle(tmp_path)
    files = release.archive_bundle(bundle, manifest, tmp_path)
    assert len(files) == 6
    assert "shiftwm-generalization-v1.tar.gz" in files
    manifest["files"]["README.md"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="differs"):
        release.archive_bundle(bundle, manifest, tmp_path)


def test_archive_rejects_parent_traversal(tmp_path):
    bundle, manifest = fixture_bundle(tmp_path)
    with tarfile.open(tmp_path / "shiftwm-generalization-v1.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("../outside")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        release.archive_bundle(bundle, manifest, tmp_path)
