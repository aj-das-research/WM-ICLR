"""Synthetic fail-closed publication tests; no credentials or network access."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/publishing/publish_iws_rowwise_release.py"
spec = importlib.util.spec_from_file_location("iws_publication_test", SOURCE)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def proof_fixture():
    rows, models = [], []
    for index in range(36):
        row = {"name": f"model_{index}", "selected_epoch": 17,
               "checkpoint_sha256": f"{index:064x}", "package_kind": "selected-kind"}
        rows.append(row)
        models.append({"name": row["name"], "status": "passed", "backend": publisher.BACKEND,
            "epoch": 17, "checkpoint_sha256": row["checkpoint_sha256"], "selected_package_kind": "selected-kind",
            "cases": [{"batch_size": batch, "shape": [batch, 59, 6144], "byte_identity_verified": True,
                "maximum_absolute_difference": 0., "prediction_sha256": "a" * 64,
                "prefix_checks": [{"horizon": h, "allclose": True, "maximum_tolerance_ratio": .25,
                    "maximum_absolute_difference": 1e-6} for h in (15, 30, 45)]} for batch in (1, 64, 8)]})
    return {"schema": "iws_rowwise_relocation_proof_v2", "status": "passed", "backend": publisher.BACKEND,
        "device": "cpu", "precision": "float32", "threads": 8, "interop_threads": 1,
        "isolated_python": True, "original_workspace_reads_denied": True, "network_attempts": 0,
        "original_workspace_read_attempts": 0, "dataset_payloads_read": 0,
        "accuracy_evaluation_performed": False, "models": models}, {"models": rows}


def test_complete_36_synthetic_proof_accepted():
    publisher.validate_proof(*proof_fixture())


@pytest.mark.parametrize("field,value", [("threads", 2), ("device", "cuda"), ("precision", "bfloat16"),
    ("isolated_python", False), ("original_workspace_reads_denied", False), ("network_attempts", 1),
    ("original_workspace_read_attempts", 1), ("dataset_payloads_read", 1), ("accuracy_evaluation_performed", True)])
def test_wrong_runtime_or_read_scope_rejected(field, value):
    proof, manifest = proof_fixture(); proof[field] = value
    with pytest.raises(ValueError): publisher.validate_proof(proof, manifest)


@pytest.mark.parametrize("mutation", ["missing_model", "duplicate_model", "wrong_checkpoint", "wrong_epoch",
    "missing_batch", "duplicate_batch", "partial_forecast", "failed_parity", "missing_prefix", "bad_ratio", "nan_ratio"])
def test_incomplete_or_forged_parity_rejected(mutation):
    proof, manifest = proof_fixture(); model = proof["models"][0]; case = model["cases"][0]
    if mutation == "missing_model": proof["models"].pop()
    elif mutation == "duplicate_model": proof["models"][-1] = deepcopy(model)
    elif mutation == "wrong_checkpoint": model["checkpoint_sha256"] = "f" * 64
    elif mutation == "wrong_epoch": model["epoch"] = 18
    elif mutation == "missing_batch": model["cases"].pop()
    elif mutation == "duplicate_batch": model["cases"][-1] = deepcopy(case)
    elif mutation == "partial_forecast": case["shape"][1] = 58
    elif mutation == "failed_parity": case["maximum_absolute_difference"] = 1e-7
    elif mutation == "missing_prefix": case["prefix_checks"].pop()
    elif mutation == "bad_ratio": case["prefix_checks"][0]["maximum_tolerance_ratio"] = 1.0001
    else: case["prefix_checks"][0]["maximum_tolerance_ratio"] = float("nan")
    with pytest.raises(ValueError): publisher.validate_proof(proof, manifest)


def test_independent_review_binds_every_final_artifact():
    bindings = {k: str(i) * 64 for i, k in enumerate(("readiness_sha256", "manifest_sha256", "archive_sha256",
        "proof_sha256", "registration_sha256", "source_review_sha256"))}
    review = {"status": "passed", "models": 36, **bindings}
    publisher.validate_independent_review(review, bindings)
    for key in bindings:
        changed = {**review, key: "stale"}
        with pytest.raises(ValueError): publisher.validate_independent_review(changed, bindings)
    with pytest.raises(ValueError): publisher.validate_independent_review({**review, "models": 35}, bindings)


def staged(tmp_path):
    payload = b"synthetic archive bytes"
    (tmp_path / publisher.ARCHIVE_NAME).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    publication = {"assets_sha256": {publisher.ARCHIVE_NAME: digest}, "archive_sha256": digest}
    (tmp_path / "PUBLICATION_MANIFEST.json").write_text(publisher.encoded(publication))
    names = sorted([publisher.ARCHIVE_NAME, "PUBLICATION_MANIFEST.json"])
    (tmp_path / "SHA256SUMS").write_text("".join(f"{publisher.sha(tmp_path/name)}  {name}\n" for name in names))
    return names + ["SHA256SUMS"], publication


def test_staging_changes_cannot_be_uploaded(tmp_path):
    names, publication = staged(tmp_path)
    publisher.validate_staged_assets(tmp_path, names, publication)
    (tmp_path / publisher.ARCHIVE_NAME).write_bytes(b"changed after review")
    with pytest.raises(ValueError, match="Staged asset changed"):
        publisher.validate_staged_assets(tmp_path, names, publication)


def test_unlisted_or_symlink_staging_rejected(tmp_path):
    names, publication = staged(tmp_path)
    (tmp_path / "credential-file").write_bytes(b"synthetic")
    with pytest.raises(ValueError): publisher.validate_staged_assets(tmp_path, names, publication)
    (tmp_path / "credential-file").unlink()
    p = tmp_path / publisher.ARCHIVE_NAME; data = p.read_bytes(); p.unlink()
    outside = tmp_path.parent / (tmp_path.name + "-outside"); outside.write_bytes(data); p.symlink_to(outside)
    with pytest.raises(ValueError): publisher.validate_staged_assets(tmp_path, names, publication)


def remote(draft=False):
    return {"tag_name": publisher.TAG, "prerelease": True, "draft": draft,
        "target_commitish": "c"*40, "assets": [{"name": "archive"}],
        "html_url": f"https://github.com/{publisher.REPOSITORY}/releases/tag/" +
            ("untagged-placeholder" if draft else publisher.TAG)}


def test_draft_untagged_link_is_allowed_but_public_scope_is_exact():
    publisher.validate_remote_release(remote(True), ["archive"], "c"*40)
    publisher.validate_remote_release(remote(), ["archive"], "c"*40)
    for field, value in (("tag_name", "old-release"), ("prerelease", False), ("target_commitish", "d"*40),
        ("assets", [{"name": "archive"}, {"name": "archive"}]), ("assets", [{"name": "unreviewed"}]),
        ("html_url", f"https://github.com/{publisher.REPOSITORY}/releases/tag/wrong")):
        with pytest.raises(ValueError): publisher.validate_remote_release({**remote(), field: value}, ["archive"], "c"*40)


class Download:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def raise_for_status(self): pass
    def iter_content(self, size): yield self.value


def test_anonymous_download_full_bytes_required():
    data = b"complete public archive"
    row = {"url": f"https://github.com/{publisher.REPOSITORY}/releases/download/{publisher.TAG}/archive",
           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    api = SimpleNamespace(get=lambda *a, **k: Download(data))
    assets = {"archive": dict(row)}; publisher.verify_public_assets(assets, api)
    assert assets["archive"]["public_download_verified"] is True
    with pytest.raises(ValueError): publisher.verify_public_assets({"archive": row}, SimpleNamespace(get=lambda *a, **k: Download(data[:-1])))
    with pytest.raises(ValueError): publisher.verify_public_assets({"archive": {**row, "url": "https://outside.example/archive"}}, api)


def test_missing_independent_review_cannot_create_staging(tmp_path):
    with pytest.raises(ValueError): publisher.stage_release(tmp_path)
    assert not (tmp_path / publisher.STAGE).exists()


def test_existing_secret_scanner_blocks_token_without_echoing_value(tmp_path):
    marker = "ghp" + "_" + "synthetic" * 5
    (tmp_path / "notes.md").write_text(marker)
    with pytest.raises(ValueError, match="Secret scan finding") as raised:
        publisher.scan_tree(SOURCE.parents[2], tmp_path)
    assert marker not in str(raised.value)
