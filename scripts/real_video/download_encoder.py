#!/usr/bin/env python3
"""Restore the exact public encoder used by the completed DROID campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def verified(path, row):
    return (path.is_file() and path.stat().st_size == row["bytes"]
            and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", default="references/real_dinov2_sources.json")
    parser.add_argument("--output", default="data/pretrained/dinov2-small")
    args = parser.parse_args()
    source = Path(args.manifest).read_bytes()
    manifest = json.loads(source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for row in manifest["files"]:
        target = output / row["file"]
        if Path(row["file"]).name != row["file"]:
            raise ValueError("Encoder manifest requires flat filenames")
        if not verified(target, row):
            partial = target.with_name(target.name + ".partial")
            url = f'https://huggingface.co/{manifest["repo"]}/resolve/{manifest["revision"]}/{row["file"]}'
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                while chunk := response.read(1 << 20):
                    stream.write(chunk)
            if not verified(partial, row):
                raise ValueError(f"Encoder checksum mismatch: {row['file']}")
            partial.replace(target)
        print(f"Verified {row['file']}")
    (output / "provenance.json").write_bytes(source)


if __name__ == "__main__":
    main()
