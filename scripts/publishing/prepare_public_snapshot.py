#!/usr/bin/env python3
"""Create a secret-audited source snapshot without private Git history or data.

This writes to a separate, new directory and never mutates research artifacts.
It does not push or upload anything. Public data imagery is retained only in
attributed paper/site figures; raw datasets and trained weights are excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT_FILES = {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "REPRODUCING.md",
              "pyproject.toml", "requirements.lock.txt", "CITATION.cff"}
CODE_ROOTS = {"src", "scripts", "tests", "configs", "demo", "docs", "deployment", ".github"}
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".cache",
              "build", "proof", "proofs", "main_review", "comparison_review",
              "review", "comparison_proof", "design", "archive_tta", "node_modules", "outputs", "runtime",
              "bundle", "checkpoints", "history"}
TEXT_EXTENSIONS = {".py", ".sh", ".slurm", ".json", ".md", ".txt", ".toml",
                   ".yaml", ".yml", ".tex", ".bib", ".sty", ".bst", ".cls",
                   ".html", ".css", ".js", ".svg", ".csv", ".cff", ".gitattributes"}
TOKEN_RULES = {
    "overleaf_token": re.compile(rb"olp_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    "hf_token": re.compile(rb"hf_[A-Za-z0-9]{25,}"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "credential_url": re.compile(rb"https?://[^\s/@:]{1,100}:[^\s/@]{5,200}@"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "google_api_key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "slack_token": re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"),
}


def include(path):
    parts = path.parts
    if any(p in SKIP_PARTS or p.endswith(".egg-info") for p in parts):
        return False
    if path.name.startswith(".env") or path.suffix in {".log", ".aux", ".out", ".fls", ".fdb_latexmk", ".synctex", ".zip", ".pt", ".pth", ".bin", ".safetensors", ".npz", ".npy", ".parquet"}:
        return False
    if len(parts) == 1:
        return path.name in ROOT_FILES
    top = parts[0]
    if top in CODE_ROOTS:
        return path.suffix in TEXT_EXTENSIONS or path.name in {"LICENSE", ".gitignore", ".gitattributes"}
    if top == "environments":
        return len(parts) == 3 and (path.suffix in {".md", ".txt", ".sh", ".json"} or path.name == ".gitignore")
    if top == "site":
        if str(path) in {"site/status.json", "site/artifact-manifest.json", "site/assets/protocol.md", "site/assets/demo-readme.md", "site/assets/upstream-revisions.json"}:
            return False
        return path.suffix in TEXT_EXTENSIONS | {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".ico"} or path.name == ".nojekyll"
    if top == "paper":
        if path.name == "proposal.pdf":
            return False
        if "assets" in parts or "split_assets" in parts:
            return path.suffix in TEXT_EXTENSIONS | {".png", ".jpg", ".pdf"} or path.name.endswith("LICENSE")
        return path.suffix in TEXT_EXTENSIONS | {".pdf"} or path.name in {"LICENSE", "Makefile"}
    if top == "references":
        return len(parts) == 2 and path.suffix == ".json" and (path.name.startswith("world_") or path.name == "real_dinov2_sources.json")
    if top == "reports":
        if path.suffix not in {".md", ".json"}:
            return False
        lower = str(path).lower()
        if any(word in lower for word in ("allocation", "slurm", "compute_audit", "gpu_summary", "gpu_start", "publication", "figure_skill", "submission_status", "probe_jobs", "real_surgical_sources", "source_fetch")):
            return False
        return True
    if top == "artifacts":
        return parts[1] == "model_cards" and path.suffix == ".md"
    if top == "results":
        return parts[1] == "real_video" and path.name == "results.json"
    return False


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.output.resolve()
    if destination.exists():
        raise SystemExit("Output already exists; use a new directory to preserve old snapshots")
    if source == destination or source in destination.parents:
        raise SystemExit("Public snapshot must be outside the scientific workspace")
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=source).decode().split("\0")
    candidates = {Path(p) for p in paths if p}
    result_root = source / "results" / "real_video"
    if result_root.exists():
        candidates.update(p.relative_to(source) for p in result_root.rglob("results.json"))
    records, findings = [], []
    for relative in sorted(candidates):
        original = source / relative
        if not include(relative) or not original.is_file():
            continue
        if original.is_symlink():
            raise SystemExit(f"Unexpected symlink: {relative}")
        data = original.read_bytes()
        if len(data) > 20_000_000:
            raise SystemExit(f"Unexpected large public file: {relative}")
        for rule, pattern in TOKEN_RULES.items():
            for match in pattern.finditer(data):
                findings.append({"path": str(relative), "line": data[:match.start()].count(b"\n") + 1, "rule": rule})
        records.append({"path": str(relative), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                        "executable": bool(original.stat().st_mode & 0o111)})
    if findings:
        print(json.dumps({"status": "blocked", "findings": findings}, indent=2))
        raise SystemExit(1)
    destination.mkdir(parents=True)
    for row in records:
        original, target = source / row["path"], destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
            raise SystemExit(f"Source changed during snapshot: {row['path']}")
    (destination / ".gitignore").write_text(""".venv/
.cache/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.env
.env.*
*.pem
*.key
external/
data/
runs/
logs/
checkpoints/
artifacts/releases/
paper/build/
demo/bundle/
demo/checkpoints/
demo/runtime/
demo/outputs/
dist/
build/
""")
    manifest = {"status": "secret_scan_passed", "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
                "source_was_dirty": True, "history_included": False, "file_count": len(records),
                "total_bytes": sum(x["bytes"] for x in records), "secret_findings": [], "files": records,
                "scope": "Curated source snapshot; raw data, model weights, environment installs and private configuration excluded."}
    (destination / "PUBLIC_SNAPSHOT.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
