"""Protect real user edits across the two independent publishing histories."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/publishing"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("shiftwm_sync_project", SCRIPTS / "sync_project.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_only_server_commit_refs_push_failure_is_retried(monkeypatch, tmp_path):
    calls, delays = [], []
    results = iter([
        subprocess.CompletedProcess([], 1, b"", b"remote: fatal error in commit_refs\n"),
        subprocess.CompletedProcess([], 0, b"pushed", b""),
    ])

    def execute(command, **kwargs):
        calls.append(list(command))
        return next(results)

    monkeypatch.setattr(sync.subprocess, "run", execute)
    monkeypatch.setattr(sync.time, "sleep", delays.append)
    command = ["git", "push", "origin", "HEAD:main"]
    assert sync.run(command, tmp_path) == b"pushed"
    assert calls == [command, command]
    assert delays == [1]


@pytest.mark.parametrize("stderr", [
    b"remote: authentication failed",
    b"! [rejected] HEAD -> main (non-fast-forward)",
])
def test_push_conflict_or_authentication_failure_is_not_retried(monkeypatch, tmp_path, stderr):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, b"", stderr)

    monkeypatch.setattr(sync.subprocess, "run", execute)
    monkeypatch.setattr(sync.time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    with pytest.raises(RuntimeError, match="git failed"):
        sync.run(["git", "push", "origin", "HEAD:main"], tmp_path)
    assert len(calls) == 1


def test_server_commit_refs_retry_is_bounded(monkeypatch, tmp_path):
    calls, delays = [], []

    def execute(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, b"", b"remote: fatal error in commit_refs")

    monkeypatch.setattr(sync.subprocess, "run", execute)
    monkeypatch.setattr(sync.time, "sleep", delays.append)
    with pytest.raises(RuntimeError, match="git failed"):
        sync.run(["git", "push", "origin", "HEAD:main"], tmp_path)
    assert len(calls) == 3 and delays == [1, 3]


def entry(data, executable=False):
    return {"data": data, "executable": executable}


def setup_bases(tmp_path, github=None, overleaf=None):
    state = tmp_path / "private"
    state.mkdir()
    return state, {name: {"files": sync.put_blobs(state, files or {})}
                   for name, files in (("github", github), ("overleaf", overleaf))}


def test_independent_edits_merge_without_dropping_lines():
    base = b"title\na\nb\nc\nd\ncaption\n"
    local = base.replace(b"title", b"new title")
    remote = base.replace(b"caption", b"new caption")
    assert sync.merge_bytes(base, local, remote) == local.replace(b"caption", b"new caption")


@pytest.mark.parametrize("base,local,remote", [
    (b"same\n", b"mine\n", b"theirs\n"),
    (None, b"mine", b"theirs"),
    (b"base", None, b"modified"),
    (b"base", b"modified", None),
    (b"\x00base", b"\x00mine", b"\x00theirs"),
])
def test_conflicting_edits_are_never_overwritten(base, local, remote):
    with pytest.raises(ValueError):
        sync.merge_bytes(base, local, remote)


def test_both_remotes_reconcile_into_one_workspace(tmp_path):
    root = tmp_path / "work"
    (root / "paper").mkdir(parents=True)
    base = b"title\na\nb\nc\nd\ne\ncaption\n"
    (root / "paper/main.tex").write_bytes(base)
    state, bases = setup_bases(tmp_path, {"paper/main.tex": entry(base)}, {"main.tex": entry(base)})
    remote = {"github": {"paper/main.tex": entry(base.replace(b"title", b"Github title"))},
              "overleaf": {"main.tex": entry(base.replace(b"caption", b"Overleaf caption"))}}
    planned, observed, modes, imports = sync.plan_imports(root, state, bases, remote)
    sync.apply_imports(root, state, planned, observed, modes)
    result = (root / "paper/main.tex").read_bytes()
    assert b"Github title" in result and b"Overleaf caption" in result
    assert len(imports) == 2
    assert list((state / "imports").rglob("main.tex"))[0].read_bytes() == base


def test_any_conflict_stops_all_imports(tmp_path):
    root = tmp_path / "work"
    (root / "paper").mkdir(parents=True)
    (root / "README.md").write_bytes(b"old")
    (root / "paper/main.tex").write_bytes(b"local")
    state, bases = setup_bases(tmp_path, {"README.md": entry(b"old")}, {"main.tex": entry(b"base")})
    remote = {"github": {"README.md": entry(b"new")}, "overleaf": {"main.tex": entry(b"remote")}}
    with pytest.raises(ValueError, match="conflict"):
        sync.plan_imports(root, state, bases, remote)
    assert (root / "README.md").read_bytes() == b"old"
    assert json.loads((state / "conflicts.json").read_text())["files"][0]["path"] == "main.tex"


def test_add_delete_and_mode_are_propagated(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (root / "README.md").write_bytes(b"delete me")
    state, bases = setup_bases(tmp_path, {"README.md": entry(b"delete me")})
    remote = {"github": {"scripts/new.py": entry(b"print('ok')\n", True)}, "overleaf": {}}
    planned, observed, modes, _ = sync.plan_imports(root, state, bases, remote)
    sync.apply_imports(root, state, planned, observed, modes)
    assert not (root / "README.md").exists()
    assert (root / "scripts/new.py").read_text() == "print('ok')\n"
    assert (root / "scripts/new.py").stat().st_mode & 0o111


def test_edit_during_import_is_preserved(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (root / "README.md").write_bytes(b"base")
    state, bases = setup_bases(tmp_path, {"README.md": entry(b"base")})
    planned, observed, modes, _ = sync.plan_imports(root, state, bases,
                            {"github": {"README.md": entry(b"remote")}, "overleaf": {}})
    (root / "README.md").write_bytes(b"new local edit")
    with pytest.raises(ValueError, match="Workspace changed"):
        sync.apply_imports(root, state, planned, observed, modes)
    assert (root / "README.md").read_bytes() == b"new local edit"


def test_frozen_experiment_remote_edit_stops(tmp_path):
    root = tmp_path / "work"
    (root / "src/shiftwm/real_video").mkdir(parents=True)
    path = "src/shiftwm/real_video/model.py"
    (root / path).write_bytes(b"frozen")
    state, bases = setup_bases(tmp_path, {path: entry(b"frozen")})
    with pytest.raises(ValueError):
        sync.plan_imports(root, state, bases, {"github": {path: entry(b"changed")}, "overleaf": {}})
    assert (root / path).read_bytes() == b"frozen"


def test_paths_and_symlinks_cannot_escape(tmp_path):
    (tmp_path / "escape").symlink_to("/tmp")
    for name in ("../outside", "/tmp/absolute", "escape/file"):
        with pytest.raises(ValueError):
            sync.safe_path(tmp_path, name)
    with pytest.raises(ValueError):
        sync.map_remote("overleaf", ".latexmkrc")


def test_merge_bases_detect_corruption(tmp_path):
    records = sync.put_blobs(tmp_path, {"README.md": entry(b"original")})
    record = records["README.md"]
    (tmp_path / "blobs" / record["sha256"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        sync.get_blob(tmp_path, record)


def test_real_git_round_trip_and_race_rejection(tmp_path):
    """Exercise normal fast-forward publishing and protect a concurrent editor."""
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(bare)], check=True, capture_output=True)
    local = tmp_path / "local"
    subprocess.run(["git", "clone", str(bare), str(local)], check=True, capture_output=True)
    for key, value in (("user.name", "Sync Test"), ("user.email", "test@example.invalid")):
        sync.git(local, "config", key, value)
    (local / "README.md").write_text("initial\n")
    sync.commit_push(local, "main", None, "Initial")
    editor = tmp_path / "editor"
    subprocess.run(["git", "clone", str(bare), str(editor)], check=True, capture_output=True)
    for key, value in (("user.name", "Editor Test"), ("user.email", "test@example.invalid")):
        sync.git(editor, "config", key, value)
    (editor / "README.md").write_text("remote edit\n")
    published = sync.commit_push(editor, "main", None, "Edit")
    assert sync.refresh(local, "main", None) == published
    assert (local / "README.md").read_text() == "remote edit\n"
    (editor / "README.md").write_text("concurrent edit\n")
    sync.commit_push(editor, "main", None, "Concurrent edit")
    (local / "README.md").write_text("local competing edit\n")
    outgoing = {}
    with pytest.raises(RuntimeError):
        sync.commit_push(local, "main", None, "Must not overwrite remote", lambda h: outgoing.update(commit=h))
    assert sync.git(bare, "show", "main:README.md") == b"concurrent edit\n"
    assert (local / "README.md").read_text() == "local competing edit\n"
    assert sync.recover_outgoing(local, "main", None, outgoing) is False
    assert (local / "README.md").read_text() == "concurrent edit\n"
    backup = sync.git(local, "for-each-ref", "--format=%(refname)", "refs/heads/sync-backup/").decode().strip()
    assert sync.git(local, "show", backup + ":README.md") == b"local competing edit\n"
    # The caller still has canonical local edits and can reconcile them against
    # the untouched remote base; recovery only moves the dedicated publisher.


def test_replace_tree_handles_both_directory_file_transitions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    sync.git(repo, "init")
    (repo / "a").mkdir()
    (repo / "a/old.txt").write_text("old")
    (repo / "b").write_text("old file")
    sync.git(repo, "add", ".")
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_text("new file")
    (source / "b").mkdir()
    (source / "b/new.txt").write_text("new child")
    sync.replace_tree(repo, source)
    assert (repo / "a").read_text() == "new file"
    assert (repo / "b/new.txt").read_text() == "new child"


def test_pushurl_cannot_redirect_publication(tmp_path):
    sync.git(tmp_path, "init")
    sync.git(tmp_path, "remote", "add", "origin", "https://github.com/owner/repo.git")
    sync.check_remote_url(tmp_path, "https://github.com/owner/repo.git")
    sync.git(tmp_path, "remote", "set-url", "--push", "origin", "https://example.invalid/other")
    with pytest.raises(ValueError, match="destination"):
        sync.check_remote_url(tmp_path, "https://github.com/owner/repo.git")


def test_remote_edit_to_generated_artifact_is_not_lost_in_rebuild(tmp_path):
    root = tmp_path / "work"
    (root / "paper").mkdir(parents=True)
    (root / "paper/world_model_draft.pdf").write_bytes(b"old")
    state, bases = setup_bases(tmp_path, {"paper/world_model_draft.pdf": entry(b"old")})
    with pytest.raises(ValueError, match="conflict"):
        sync.plan_imports(root, state, bases, {"github": {"paper/world_model_draft.pdf": entry(b"edited")}, "overleaf": {}})
    assert "Generated artifact" in (state / "conflicts.json").read_text()


STATUS_JSON = "reports/real_video_iws/live_status.json"
STATUS_MARKDOWN = "reports/current_results_and_gpu_status.md"
HEARTBEAT_1 = "2026-09-20T09:48:35.050524+00:00"
HEARTBEAT_2 = "2026-09-20T09:58:37.578388+00:00"


def status_payload(checked=HEARTBEAT_1, **changes):
    return json.dumps({"schema": "shiftwm_iws_live_progress_v1",
                       "checked_utc": checked, "full_training_summaries": 27,
                       "counts": {"RUNNING": 0, "PENDING": 0}, **changes}).encode()


def status_markdown(checked=HEARTBEAT_1):
    return ("# Current results and GPU status\n\n"
            f"Checked **{checked}** from the live scheduler and checkpoint summaries.\n\n"
            "All 27 models completed.\n").encode()


@pytest.mark.parametrize("path,factory", [
    (STATUS_JSON, status_payload), (STATUS_MARKDOWN, status_markdown),
])
def test_only_recognized_heartbeat_is_ignored(path, factory):
    before, after = factory(HEARTBEAT_1), factory(HEARTBEAT_2)
    assert before != after
    assert sync.fingerprint_bytes(path, before) == sync.fingerprint_bytes(path, after)
    assert before == factory(HEARTBEAT_1)  # Input/publication bytes are untouched.


@pytest.mark.parametrize("change", [
    {"full_training_summaries": 26},
    {"counts": {"RUNNING": 1, "PENDING": 0}},
    {"counts": {"RUNNING": 0, "PENDING": 1}},
    {"complete_study_finalizer_present": True},
    {"nested": {"checked_utc": HEARTBEAT_2}},
])
def test_substantive_status_updates_still_change_fingerprint(change):
    before = status_payload(nested={"checked_utc": HEARTBEAT_1})
    after = status_payload(HEARTBEAT_2, **({"nested": {"checked_utc": HEARTBEAT_1}} | change))
    assert sync.fingerprint_bytes(STATUS_JSON, before) != sync.fingerprint_bytes(STATUS_JSON, after)


def test_status_markdown_results_and_other_dates_remain_significant():
    before = status_markdown() + f"Finalized at {HEARTBEAT_1}.\n".encode()
    for after in (before.replace(b"27 models", b"36 models"),
                  before.replace(f"Finalized at {HEARTBEAT_1}".encode(),
                                 f"Finalized at {HEARTBEAT_2}".encode())):
        assert sync.fingerprint_bytes(STATUS_MARKDOWN, before) != sync.fingerprint_bytes(STATUS_MARKDOWN, after)


@pytest.mark.parametrize("path,data", [
    ("reports/other/live_status.json", status_payload()),
    ("reports/evidence/live_status.json", status_payload()),
    ("reports/other_status.md", status_markdown()),
    ("paper/sections/main_results.tex", status_markdown()),
])
def test_heartbeat_exception_is_exact_path_only(path, data):
    assert sync.fingerprint_bytes(path, data) == data


@pytest.mark.parametrize("data", [
    b"{invalid", b"[]", b"null", b"\xff",
    status_payload(checked=None), status_payload(checked=123),
    status_payload(checked="2026-02-30T09:00:00+00:00"),
    status_payload(checked="2026-09-20T09:00:00"),
    status_payload(checked="2026-09-20T09:00:00+04:00"),
    status_payload(schema="other_schema"),
    status_payload().replace(b'"counts":', b'"checked_utc": "duplicate", "counts":'),
    status_payload(nested={"x": 1}).replace(b'"x": 1', b'"x": 1, "x": 2'),
    status_payload(invalid=float("nan")),
])
def test_malformed_or_unrecognized_json_keeps_raw_fingerprint(data):
    assert sync.fingerprint_bytes(STATUS_JSON, data) == data


@pytest.mark.parametrize("data", [
    b"\xff", b"unrelated prose",
    status_markdown("2026-02-30T09:00:00+00:00"),
    status_markdown().replace(b"# Current results", b"# Different results"),
    status_markdown().replace(b"Checked **", b"Checked at **"),
    status_markdown() + status_markdown(),
    status_markdown().replace(b"checkpoint summaries.", b"different source."),
])
def test_malformed_or_unrecognized_markdown_keeps_raw_fingerprint(data):
    assert sync.fingerprint_bytes(STATUS_MARKDOWN, data) == data


def test_workspace_settle_and_publish_checks_ignore_only_heartbeats(tmp_path):
    sync.git(tmp_path, "init")
    for path, data in [(STATUS_JSON, status_payload()), (STATUS_MARKDOWN, status_markdown())]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    initial = sync.workspace_fingerprint(tmp_path)
    # These are the same before/after comparisons used by watch-cycle settling
    # and the publication pending flag; the live writer may run in either gap.
    (tmp_path / STATUS_JSON).write_bytes(status_payload(HEARTBEAT_2))
    (tmp_path / STATUS_MARKDOWN).write_bytes(status_markdown(HEARTBEAT_2))
    assert sync.workspace_fingerprint(tmp_path) == initial
    assert (tmp_path / STATUS_JSON).read_bytes() == status_payload(HEARTBEAT_2)
    assert (tmp_path / STATUS_MARKDOWN).read_bytes() == status_markdown(HEARTBEAT_2)
    (tmp_path / STATUS_JSON).write_bytes(status_payload(HEARTBEAT_2, full_training_summaries=36))
    assert sync.workspace_fingerprint(tmp_path) != initial


def test_workspace_paths_deletions_and_executable_modes_remain_significant(tmp_path):
    sync.git(tmp_path, "init")
    path = tmp_path / "README.md"
    path.write_text("Scientific result.\n")
    first = sync.workspace_fingerprint(tmp_path)
    path.chmod(0o755)
    assert sync.workspace_fingerprint(tmp_path) != first
    path.chmod(0o644)
    assert sync.workspace_fingerprint(tmp_path) == first
    path.rename(tmp_path / "REPRODUCING.md")
    assert sync.workspace_fingerprint(tmp_path) != first
    (tmp_path / "REPRODUCING.md").unlink()
    assert sync.workspace_fingerprint(tmp_path) != first
