#!/usr/bin/env python3
"""Fetch exact upstream sources without overwriting local modifications."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--check", action="store_true", help="Verify existing checkouts without modifying them")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for source in json.loads((root / "references/world_upstream_revisions.json").read_text()):
    directory = root / "external" / source["name"]
    if not directory.exists():
        if args.check:
            raise SystemExit(f"Missing checkout: {directory}")
        directory.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", source["remote"], str(directory)], check=True)
    revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise SystemExit(f"Upstream checkout has local modifications; preserve/review them: {directory}")
    if revision != source["revision"]:
        if args.check:
            raise SystemExit(f"Revision mismatch: {directory}: {revision}")
        subprocess.run(["git", "-C", str(directory), "fetch", "origin", source["revision"]], check=True)
        subprocess.run(["git", "-C", str(directory), "checkout", "--detach", source["revision"]], check=True)
    print(json.dumps({"name": source["name"], "revision": source["revision"], "status": "verified"}))
