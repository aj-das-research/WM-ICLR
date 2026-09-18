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
