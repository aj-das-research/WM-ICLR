#!/usr/bin/env python3
"""Synchronize the working project, GitHub, Overleaf and the Pages demo.

Private configuration and merge bases live outside the repository. All pushes
are fast-forward, and conflicts stop before any workspace file is changed.
Run --initialize once against the clean, previously published checkouts, then
run normally (or --watch-cycle from a user systemd timer).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from prepare_public_snapshot import include, TOKEN_RULES

ROOT = Path(__file__).resolve().parents[2]
GENERATED_GITHUB = {"PUBLIC_SNAPSHOT.json", ".gitignore"}
OVERLEAF_MANIFEST = ".shiftwm-sync-manifest.json"
PAPER_EXTENSIONS = {".tex", ".bib", ".sty", ".bst", ".cls", ".pdf", ".png",
                    ".jpg", ".jpeg", ".svg", ".eps"}
DEFAULT_PROTECTED = ["src/shiftwm/real_video/*", "configs/real_video/*",
                     "scripts/real_video/train.py", "scripts/real_video/evaluate.py",
                     "scripts/real_video/run_campaign.py", "reports/real_droid_protocol.md",
                     "paper/template/official/*"]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(command, cwd, *, timeout=600):
    result = subprocess.run(command, cwd=cwd, capture_output=True,
                            env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), timeout=timeout)
    if result.returncode:
        # No credentials are passed in commands. Do not echo remote file content.
        raise RuntimeError(f"{Path(command[0]).name} failed (exit {result.returncode}); "
                           "inspect the relevant checkout/build logs locally")
    return result.stdout


def git(repo, *args, credential=None):
    command = ["git"]
    if credential:
        credential = Path(credential).expanduser().resolve()
        if credential.stat().st_mode & 0o077:
            raise ValueError("Credential file permissions must be 0600")
        command += ["-c", "credential.helper=", "-c",
                    "credential.helper=store --file=" + str(credential)]
    return run(command + list(args), repo)


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe synchronized path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Synchronized paths must stay inside their root without symlinks")
    return path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)


def tracked_files(repo):
    """Read clean regular Git files, rejecting links and submodules."""
    result = {}
    for row in git(repo, "ls-files", "-s", "-z").split(b"\0"):
        if not row:
            continue
        metadata, name = row.split(b"\t", 1)
        mode = metadata.split()[0].decode()
        if mode not in ("100644", "100755"):
            raise ValueError("Links/submodules are outside automatic synchronization")
        rel = name.decode()
        path = safe_path(repo, rel)
        result[rel] = {"data": path.read_bytes(), "executable": mode == "100755"}
    return result


def scan_secret(data, relative):
    if any(pattern.search(data) for pattern in TOKEN_RULES.values()):
        raise ValueError(f"Secret scan failed for {relative}; content suppressed")


def put_blobs(state_dir, files):
    records = {}
    blobs = state_dir / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for name, entry in files.items():
        key = digest(entry["data"])
        path = blobs / key
        if not path.exists():
            path.write_bytes(entry["data"])
            path.chmod(0o600)
        records[name] = {"sha256": key, "executable": entry["executable"]}
    return records


def get_blob(state_dir, record):
    if record is None:
        return None
    content = (state_dir / "blobs" / record["sha256"]).read_bytes()
    if digest(content) != record["sha256"]:
        raise ValueError("Synchronization base is corrupt")
    return content


def merge_bytes(base, local, remote):
    """Three-way merge, including additions/deletions and binary conflicts."""
    if local == remote or remote == base:
        return local
    if local == base:
        return remote
    if base is None or local is None or remote is None:
        raise ValueError("Concurrent addition/deletion conflict")
    for value in (base, local, remote):
        if b"\0" in value:
            raise ValueError("Concurrent binary edits")
        try:
            value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Concurrent binary edits") from exc
    with tempfile.TemporaryDirectory(prefix="shiftwm-merge-") as tmp:
        paths = [Path(tmp) / name for name in ("local", "base", "remote")]
        for path, data in zip(paths, (local, base, remote)):
            path.write_bytes(data)
        merged = subprocess.run(["git", "merge-file", "-p", *map(str, paths)], capture_output=True)
        if merged.returncode:
            raise ValueError("Overlapping text edits")
        return merged.stdout


def map_remote(name, relative):
    if name == "github":
        if relative in GENERATED_GITHUB:
            return None
        if not include(Path(relative)):
            raise ValueError("Remote path is outside the public source policy: " + relative)
        return relative
    if relative == OVERLEAF_MANIFEST:
        return None
    path = Path(relative)
    if any(part.startswith(".") for part in path.parts) or path.suffix not in PAPER_EXTENSIONS:
        raise ValueError("Unsupported Overleaf source path: " + relative)
    if len(path.parts) == 1 and path.suffix in {".sty", ".bst"}:
        return "paper/template/official/iclr2027/" + relative
    return "paper/" + relative


def plan_imports(root, state_dir, bases, remote_files, protected=DEFAULT_PROTECTED):
    virtual, observed, modes, imports, conflicts = {}, {}, {}, [], []
    for name in ("github", "overleaf"):
        base_files = bases[name]["files"]
        remote = remote_files[name]
        for rel in sorted(base_files.keys() | remote.keys()):
            base = get_blob(state_dir, base_files.get(rel))
            new = remote.get(rel, {}).get("data")
            old_mode = base_files.get(rel, {}).get("executable", False)
            new_mode = remote.get(rel, {}).get("executable", False)
            if base == new and old_mode == new_mode:
                continue
            try:
                target = map_remote(name, rel)
                if target is None:
                    # These files belong to the publisher; manual edits cannot be
                    # silently discarded by regenerating them.
                    raise ValueError("Remote edit of generated synchronization metadata")
                path = safe_path(root, target)
                if target not in observed:
                    observed[target] = path.read_bytes() if path.is_file() else None
                local = virtual.get(target, observed[target])
                merged = merge_bytes(base, local, new)
                if merged != local and any(fnmatch.fnmatch(target, p) for p in protected):
                    raise ValueError("Frozen experiment/template source: use a new versioned namespace")
                if merged is not None:
                    scan_secret(merged, target)
                virtual[target] = merged
                local_mode = bool(path.stat().st_mode & 0o111) if path.exists() else False
                modes[target] = new_mode if old_mode != new_mode else local_mode
                imports.append({"remote": name, "path": target, "deleted": merged is None})
            except ValueError as exc:
                conflicts.append({"remote": name, "path": rel, "reason": str(exc)})
    if conflicts:
        atomic_json(state_dir / "conflicts.json", {"time": now(), "files": conflicts,
                     "action": "No workspace imports were applied. Resolve from local files, remote checkouts and private base blobs, then rerun."})
        raise ValueError(f"{len(conflicts)} synchronization conflict(s); see private conflicts.json")
    return virtual, observed, modes, imports


def apply_imports(root, state_dir, planned, observed, modes):
    # Validate the entire plan before changing anything, then keep recoverable originals.
    for rel, old in observed.items():
        path = safe_path(root, rel)
        if (path.read_bytes() if path.is_file() else None) != old:
            raise ValueError("Workspace changed during reconciliation; retry when edits settle")
    changed = [rel for rel in planned if planned[rel] != observed[rel] or
               (planned[rel] is not None and bool((root / rel).stat().st_mode & 0o111) != modes[rel])]
    if not changed:
        return
    backup = state_dir / "imports" / (str(time.time_ns()))
    backup.mkdir(parents=True)
    for rel in changed:
        if observed[rel] is not None:
            out = safe_path(backup, rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(observed[rel])
    atomic_json(backup / "import-plan.json", {"files": changed,
                 "previously_absent": [r for r in changed if observed[r] is None]})
    for rel in changed:
        path = safe_path(root, rel)
        if planned[rel] is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".sync-tmp")
            tmp.write_bytes(planned[rel])
            tmp.chmod(0o755 if modes[rel] else 0o644)
            tmp.replace(path)


def workspace_fingerprint(root):
    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")
    result = hashlib.sha256()
    for rel in sorted(set(n.decode() for n in names if n)):
        if include(Path(rel)):
            path = safe_path(root, rel)
            if path.is_file():
                result.update(rel.encode() + b"\0" + path.read_bytes())
                result.update(b"x" if path.stat().st_mode & 0o111 else b"-")
    return result.hexdigest()


def clean_checkout(repo):
    if git(repo, "status", "--porcelain").strip():
        raise ValueError("Publication checkout has uncommitted changes: " + str(repo))


def refresh(repo, branch, credential):
    clean_checkout(repo)
    if git(repo, "branch", "--show-current").decode().strip() != branch:
        raise ValueError("Unexpected publication branch")
    git(repo, "fetch", "origin", branch, credential=credential)
    git(repo, "merge", "--ff-only", "origin/" + branch)
    return git(repo, "rev-parse", "HEAD").decode().strip()


def replace_tree(repo, source):
    previous = set(tracked_files(repo))
    current = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    for rel in previous - current:
        safe_path(repo, rel).unlink()
    for rel in sorted(current):
        target = safe_path(repo, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, target)


def commit_push(repo, branch, credential, message):
    git(repo, "add", "--all")
    if git(repo, "diff", "--cached", "--name-only").strip():
        git(repo, "commit", "-m", message)
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    git(repo, "push", "origin", "HEAD:" + branch, credential=credential)
    remote = git(repo, "ls-remote", "origin", "refs/heads/" + branch,
                 credential=credential).decode().split()[0]
    if head != remote:
        raise ValueError("Published commit verification failed")
    return head


def check_remote_url(repo, expected):
    # Avoid sending an explicitly scoped saved credential to a substituted remote.
    if git(repo, "remote", "get-url", "origin").decode().strip().removesuffix(".git") != expected.removesuffix(".git"):
        raise ValueError("Publication remote does not match the configured destination")


def synchronize(args):
    config = json.loads(args.config.read_text())
    state_dir = Path(config.get("sync_state", Path.home() / ".local/share/shiftwm/sync")).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    lock = (state_dir / "sync.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {"status": "already_running"}
    state_file = state_dir / "state.json"
    repos = {"github": Path(config["github_checkout"]).expanduser(),
             "overleaf": Path(config.get("overleaf_checkout", Path.home() / ".local/share/shiftwm/overleaf/repo")).expanduser()}
    pages = Path(config["pages_checkout"]).expanduser()
    credentials = {"github": config["github_credential_file"], "overleaf": config["credential_file"]}
    github_url = "https://github.com/" + config["github_repository"].removesuffix(".git") + ".git"
    check_remote_url(repos["github"], github_url)
    check_remote_url(repos["overleaf"], config["overleaf_git"])
    check_remote_url(pages, github_url)
    if args.initialize:
        if state_file.exists():
            raise ValueError("Already initialized; refusing to discard merge bases")
        state = {"version": 1, "initialized_at": now(), "bases": {}, "fingerprint": None}
        for name, repo in repos.items():
            clean_checkout(repo)
            state["bases"][name] = {"commit": git(repo, "rev-parse", "HEAD").decode().strip(),
                                    "files": put_blobs(state_dir, tracked_files(repo))}
        atomic_json(state_file, state)
        return {"status": "initialized", "bases": {n: b["commit"] for n, b in state["bases"].items()}}
    if not state_file.exists():
        raise ValueError("Initialize from the last published clean checkouts first")
    state = json.loads(state_file.read_text())
    fingerprint = workspace_fingerprint(ROOT)
    if args.watch_cycle:
        time.sleep(args.settle_seconds)
        if workspace_fingerprint(ROOT) != fingerprint:
            return {"status": "deferred_active_edits"}
    heads = {name: refresh(repo, "main", credentials[name]) for name, repo in repos.items()}
    if fingerprint == state.get("fingerprint") and all(heads[n] == state["bases"][n]["commit"] for n in repos) and not state.get("pending"):
        return {"status": "up_to_date", "checked_at": now()}
    remote_files = {name: tracked_files(repo) for name, repo in repos.items()}
    planned, observed, modes, imports = plan_imports(ROOT, state_dir, state["bases"], remote_files,
                         config.get("protected_import_globs", DEFAULT_PROTECTED))
    apply_imports(ROOT, state_dir, planned, observed, modes)
    # Imported remote versions become the new base immediately. A later build or
    # push failure is retryable without losing imports or applying them twice.
    for name in repos:
        state["bases"][name] = {"commit": heads[name], "files": put_blobs(state_dir, remote_files[name])}
    state["pending"] = True
    state["last_imports"] = imports
    atomic_json(state_file, state)
    (state_dir / "conflicts.json").unlink(missing_ok=True)

    # Build from reconciled source. Keep the shell build's own manuscript lock.
    run(["bash", "paper/build.sh"], ROOT)
    run([sys.executable, "site/publish.py"], ROOT)
    # The exact remote SHA is checked again inside the standalone manuscript sync.
    run([sys.executable, "scripts/publishing/sync_overleaf.py", "--config", str(args.config),
         "--push", "--reconciled-remote-commit", heads["overleaf"]], ROOT)
    state["bases"]["overleaf"] = {"commit": git(repos["overleaf"], "rev-parse", "HEAD").decode().strip(),
                                  "files": put_blobs(state_dir, tracked_files(repos["overleaf"]))}
    atomic_json(state_file, state)

    with tempfile.TemporaryDirectory(prefix="shiftwm-public-") as temporary:
        snapshot = Path(temporary) / "snapshot"
        run([sys.executable, "scripts/publishing/prepare_public_snapshot.py", "--source", str(ROOT),
             "--output", str(snapshot)], ROOT)
        replace_tree(repos["github"], snapshot)
    github_head = commit_push(repos["github"], "main", credentials["github"],
                              "Sync research code, manuscript and project demo")
    state["bases"]["github"] = {"commit": github_head, "files": put_blobs(state_dir, tracked_files(repos["github"]))}
    atomic_json(state_file, state)
    refresh(pages, "gh-pages", credentials["github"])
    replace_tree(pages, ROOT / "site/export")
    pages_head = commit_push(pages, "gh-pages", credentials["github"], "Update the ShiftWM project page")
    state.update({"pending": False, "fingerprint": workspace_fingerprint(ROOT),
                  "last_success": now(), "pages_commit": pages_head})
    atomic_json(state_file, state)
    receipt = {"status": "synchronized", "time": state["last_success"], "imports": imports,
               "github_commit": github_head, "overleaf_commit": state["bases"]["overleaf"]["commit"],
               "pages_commit": pages_head, "validation": "Paper build, independent Overleaf bundle compile/text parity, public secret scan, remote commit verification"}
    atomic_json(state_dir / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/shiftwm/publishing.json")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--watch-cycle", action="store_true")
    parser.add_argument("--settle-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        print(json.dumps(synchronize(args)))
    except Exception as exc:
        # Error text generated here includes paths/reasons, never credential values.
        print(json.dumps({"status": "stopped", "reason": str(exc), "time": now()}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
