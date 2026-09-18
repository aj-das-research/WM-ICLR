"""Offline tests for artifact integrity, resumption, and preservation rules."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_artifacts.py"
spec = importlib.util.spec_from_file_location("shiftwm_artifact_downloader_test", SCRIPT)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def item_for(content=b"known-model-weights", path="data/pretrained/test/weights.pt"):
    return {"name": "weights.pt", "path": path,
            "url": "https://huggingface.co/test/repo/resolve/" + "a" * 40 + "/weights.pt",
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def test_existing_wrong_final_file_is_never_overwritten(tmp_path):
    item = item_for()
    target = tmp_path / item["path"]
    target.parent.mkdir(parents=True)
    original = b"x" * item["bytes"]
    target.write_bytes(original)
    called = []
    with pytest.raises(ValueError, match="unexpected SHA256"):
        downloader.fetch_artifact(tmp_path, item, runner=lambda *args, **kwargs: called.append(True))
    assert target.read_bytes() == original and not called


def test_verify_only_missing_file_does_not_create_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        downloader.fetch_artifact(tmp_path, item_for(), verify_only=True)
    assert list(tmp_path.iterdir()) == []


def test_verified_existing_file_never_downloads(tmp_path):
    payload = b"known-model-weights"
    item = item_for(payload)
    target = tmp_path / item["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    def no_network(*args, **kwargs):
        raise AssertionError("Existing correct artifact should not download")
    assert downloader.fetch_artifact(tmp_path, item, runner=no_network)["status"] == "verified"


def test_interrupted_transfer_retains_identity_and_resumes_atomically(tmp_path):
    payload = b"known-model-weights"
    item = item_for(payload)
    calls = []
    def curl(command, check):
        assert check and command[command.index("--continue-at") + 1] == "-"
        target = Path(command[command.index("--output") + 1])
        calls.append(command)
        if len(calls) == 1:
            target.write_bytes(payload[:5])
            raise subprocess.CalledProcessError(18, command)
        assert target.read_bytes() == payload[:5]
        with target.open("ab") as handle:
            handle.write(payload[5:])
    with pytest.raises(subprocess.CalledProcessError):
        downloader.fetch_artifact(tmp_path, item, runner=curl)
    final = tmp_path / item["path"]
    assert not final.exists()
    assert final.with_name(final.name + ".partial.source.json").exists()
    result = downloader.fetch_artifact(tmp_path, item, runner=curl)
    assert result["status"] == "downloaded_and_verified"
    assert final.read_bytes() == payload
    assert not final.with_name(final.name + ".partial").exists()
    assert not final.with_name(final.name + ".partial.source.json").exists()


def test_unknown_partial_is_preserved(tmp_path):
    item = item_for()
    final = tmp_path / item["path"]
    final.parent.mkdir(parents=True)
    partial = final.with_name(final.name + ".partial")
    partial.write_bytes(b"unidentified")
    with pytest.raises(ValueError, match="unrecognized partial"):
        downloader.fetch_artifact(tmp_path, item, runner=lambda *args, **kwargs: None)
    assert partial.read_bytes() == b"unidentified" and not final.exists()


def test_concurrent_final_file_is_not_overwritten_during_promotion(tmp_path):
    payload = b"known-model-weights"
    item = item_for(payload)
    final = tmp_path / item["path"]
    def curl(command, check):
        partial = Path(command[command.index("--output") + 1])
        partial.write_bytes(payload)
        final.write_bytes(b"created by an independent writer")
    with pytest.raises(FileExistsError):
        downloader.fetch_artifact(tmp_path, item, runner=curl)
    assert final.read_bytes() == b"created by an independent writer"
    assert final.with_name(final.name + ".partial").read_bytes() == payload


def test_metadata_restore_is_exact_and_never_clobbers(tmp_path):
    text = '{"repo": "test/repo", "revision": "abc"}\n'
    item = {"path": "data/pretrained/test/provenance.json", "content_utf8": text,
            "bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    downloader.restore_metadata(tmp_path, item)
    final = tmp_path / item["path"]
    assert final.read_bytes() == text.encode()
    final.write_text("user modified provenance")
    with pytest.raises(ValueError, match="Preserved existing"):
        downloader.restore_metadata(tmp_path, item)
    assert final.read_text() == "user modified provenance"


def test_destination_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="escapes root"):
        downloader.artifact_path(tmp_path, "../elsewhere")


def test_manifest_rejects_unpinned_urls_and_bad_metadata_checksum():
    item = item_for()
    source = {"repo": "test/repo", "revision": "a" * 40, "files": [item]}
    manifest = {"schema_version": 1, "models": [source], "datasets": [], "metadata_files": []}
    downloader.validate_manifest(manifest)
    item["url"] = item["url"].replace("a" * 40, "main")
    with pytest.raises(ValueError, match="pinned repository"):
        downloader.validate_manifest(manifest)
    item["url"] = item["url"].replace("main", "a" * 40)
    manifest["metadata_files"] = [{"path": "data/bad", "content_utf8": "bad", "bytes": 3, "sha256": "0" * 64}]
    with pytest.raises(ValueError, match="metadata checksum"):
        downloader.validate_manifest(manifest)


def test_default_verification_does_not_read_dataset_archives(tmp_path, monkeypatch):
    payload = b"known-model-weights"
    item = item_for(payload)
    archive = item_for(b"large-archive", "data/upstream/archive.tar.zst")
    archive["name"] = "archive.tar.zst"
    archive["url"] = "https://huggingface.co/datasets/test/repo/resolve/" + "b" * 40 + "/archive.tar.zst"
    manifest = {"schema_version": 1,
                "models": [{"repo": "test/repo", "revision": "a" * 40, "files": [item]}],
                "datasets": [{"repo": "test/repo", "revision": "b" * 40, "files": [archive]}],
                "metadata_files": []}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    model_file = tmp_path / item["path"]
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(payload)
    # Archive deliberately absent: default verification must still succeed.
    result = downloader.run(path, tmp_path, verify_only=True)
    assert result["dataset_archives_included"] is False
    assert len(result["artifacts"]) == 1
    with pytest.raises(FileNotFoundError):
        downloader.run(path, tmp_path, datasets=True, verify_only=True)
