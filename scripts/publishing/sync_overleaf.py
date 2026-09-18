#!/usr/bin/env python3
"""Compile and sync the manuscript through an external, credential-free checkout.

Connection metadata lives in ~/.config/shiftwm/publishing.json. Authentication
uses its credential_file through Git's credential-store; secrets never enter
the paper, command arguments, repository URLs, or this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def run(command, cwd, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command[0]} (exit {result.returncode})\n"
                           + (result.stdout + result.stderr)[-3000:])
    return result.stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def managed_path(repo, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe managed manuscript path")
    destination = repo / path
    if not destination.resolve().is_relative_to(repo.resolve()):
        raise ValueError("Managed manuscript path escapes its checkout")
    return destination


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path.home() / ".config/shiftwm/publishing.json")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--reconciled-remote-commit",
                        help="Sync orchestrator only: exact remote commit already merged into the workspace")
    parser.add_argument("--adopt-existing-main", action="store_true",
                        help="First sync: replace an existing main.tex; its Git history is retained")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    remote_url = urlsplit(config["overleaf_git"])
    if remote_url.scheme != "https" or remote_url.hostname != "git.overleaf.com" or remote_url.username:
        raise ValueError("Use a credential-free HTTPS Overleaf Git URL")
    credential = Path(config["credential_file"]).expanduser().resolve()
    if credential.stat().st_mode & 0o077:
        raise ValueError("Credential file must not be accessible to group or others")
    repo = Path(config.get("overleaf_checkout", Path.home() / ".local/share/shiftwm/overleaf/repo"))
    paper = ROOT / "paper"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    git = ["git", "-c", "credential.helper=", "-c",
           "credential.helper=store --file=" + str(credential)]
    if not (repo / ".git").is_dir():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(git + ["clone", config["overleaf_git"], str(repo)], ROOT, env)
    run(git + ["fetch", "origin"], repo, env)
    if run(["git", "status", "--porcelain"], repo):
        raise ValueError("Overleaf checkout has local changes; preserve/review them before syncing")
    branch = run(["git", "branch", "--show-current"], repo)
    run(["git", "merge", "--ff-only", "origin/" + branch], repo)
    before = run(["git", "rev-parse", "HEAD"], repo)
    reconciled = args.reconciled_remote_commit == before
    if args.reconciled_remote_commit and not reconciled:
        raise ValueError("Overleaf changed after reconciliation; fetch and merge the new edits first")
    manifest_path = repo / ".shiftwm-sync-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    for name in previous["files"]:
        managed_path(repo, name)
    edited = [name for name, digest in previous["files"].items()
              if not (repo / name).is_file() or sha(repo / name) != digest]
    if edited and not reconciled:
        raise ValueError("Overleaf editor changes require merging into local paper first: " + ", ".join(edited))

    # Recorder discovers the exact local assets required by the current paper.
    scan_env = dict(os.environ, TEXINPUTS=str(paper / "template/official/iclr2027") + "//:")
    run(["pdflatex", "-recorder", "-interaction=nonstopmode", "-halt-on-error",
         "-output-directory=build", "main.tex"], paper, scan_env)
    sources = {}
    for line in (paper / "build/main.fls").read_text().splitlines():
        if not line.startswith("INPUT "):
            continue
        path = Path(line[6:])
        path = (paper / path).resolve() if not path.is_absolute() else path.resolve()
        if path.is_relative_to(paper) and path.is_file():
            relative = path.relative_to(paper)
            if relative.parts[0] not in ("build", "archive_tta", "template"):
                sources[relative.as_posix()] = path
    sources.update({"main.tex": paper / "main.tex", "references.bib": paper / "references.bib"})
    for path in (paper / "template/official/iclr2027").iterdir():
        if path.suffix in (".sty", ".bst"):
            sources[path.name] = path
    for name, source in sources.items():
        target = managed_path(repo, name)
        if target.exists() and name not in previous["files"]:
            if not reconciled and not (args.adopt_existing_main and name == "main.tex") and sha(target) != sha(source):
                raise ValueError("Refusing to replace an unmanaged Overleaf file: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Retire only files explicitly managed by the previous synchronization.
    for name in previous["files"].keys() - sources.keys():
        (repo / name).unlink()
    manifest = {"format_version": 1, "files": {name: sha(repo / name) for name in sorted(sources)},
                "source_pdf_sha256": sha(paper / "world_model_draft.pdf"),
                "scope": "Manuscript source and required assets; no credentials or experimental data"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Compile the standalone upload bundle using only its own assets and TeX Live.
    compile_env = dict(os.environ, TEXINPUTS=".:", BIBINPUTS=".:", BSTINPUTS=".:")
    proof_dir = ROOT / "artifacts/publishing/overleaf-proof"
    proof_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shiftwm-overleaf-") as output:
        latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory=" + output, "main.tex"]
        run(latex, repo, compile_env)
        bib_env = dict(compile_env, BIBINPUTS=str(repo) + ":", BSTINPUTS=str(repo) + ":")
        run(["bibtex", "main"], output, bib_env)
        run(latex, repo, compile_env)
        run(latex, repo, compile_env)
        original = run(["pdftotext", "-layout", str(paper / "world_model_draft.pdf"), "-"], ROOT)
        uploaded = run(["pdftotext", "-layout", str(Path(output) / "main.pdf"), "-"], ROOT)
        if original != uploaded:
            raise ValueError("Standalone Overleaf bundle text differs from the reviewed local PDF")
        shutil.copyfile(Path(output) / "main.pdf", proof_dir / "paper.pdf")
        shutil.copyfile(Path(output) / "main.log", proof_dir / "compile.log")
    run(["git", "add", "--all"], repo)
    if run(["git", "diff", "--cached", "--name-only"], repo):
        # Preserve the user's configured identity; use the source repository if unset.
        for key in ("user.name", "user.email"):
            current = subprocess.run(["git", "config", "--get", key], cwd=repo, capture_output=True, text=True)
            if not current.stdout.strip():
                inherited = subprocess.run(["git", "config", "--get", key], cwd=ROOT,
                                           capture_output=True, text=True)
                value = inherited.stdout.strip()
                if not value:
                    field = "%an" if key == "user.name" else "%ae"
                    value = run(["git", "log", "-1", "--format=" + field], ROOT)
                if not value:
                    raise ValueError("A Git author identity is required")
                run(["git", "config", key, value], repo)
        run(["git", "commit", "-m", "Update ShiftWM manuscript with audited real-video results"], repo)
    commit = run(["git", "rev-parse", "HEAD"], repo)
    if args.push:
        # A concurrent editor commit causes a normal non-fast-forward rejection.
        run(git + ["push", "origin", "HEAD:" + branch], repo, env)
        remote = run(git + ["ls-remote", "origin", "refs/heads/" + branch], repo, env).split()[0]
        if remote != commit:
            raise ValueError("Remote commit verification failed")
    receipt = {"status": "pushed" if args.push else "prepared", "commit": commit,
               "previous_remote_commit": before, "files": len(sources),
               "standalone_compile": "passed", "pdf_text_parity": "exact",
               "source_pdf_sha256": manifest["source_pdf_sha256"]}
    (proof_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
