#!/usr/bin/env python3
"""Upload and verify an audited inference archive as a GitHub release asset."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--credential-file", required=True, type=Path)
    parser.add_argument("--target-commit", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.credential_file.stat().st_mode & 0o077:
        raise ValueError("Credential store must be private (0600)")
    credential = next(urlparse(line) for line in args.credential_file.read_text().splitlines()
                      if urlparse(line).hostname == "github.com")
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + unquote(credential.password),
                            "Accept": "application/vnd.github+json",
                            "X-GitHub-Api-Version": "2022-11-28",
                            "User-Agent": "ShiftWM-checkpoint-release"})
    repository = "aj-das-research/WM-ICLR"
    api = f"https://api.github.com/repos/{repository}"
    tag = "real-droid-v1"
    names = ["shiftwm-real-droid-v1.tar.gz", "README.md", "MODEL_LICENSE.md",
             "PUBLICATION_MANIFEST.json", "PUBLICATION_VERIFICATION.json", "SHA256SUMS"]
    files = {name: args.directory / name for name in names}
    expected = {name: sha(path) for name, path in files.items()}
    checksums = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
                 for line in files["SHA256SUMS"].read_text().splitlines()}
    if any(expected[name] != value for name, value in checksums.items()):
        raise ValueError("Frozen release asset checksum mismatch")
    audit = json.loads(files["PUBLICATION_MANIFEST.json"].read_text())
    verification = json.loads(files["PUBLICATION_VERIFICATION.json"].read_text())
    if audit["status"] != "verified" or audit["secret_findings"] or verification["status"] != "passed":
        raise ValueError("Release audit incomplete")
    body = """Offline research checkpoints for the completed real-DROID subset study.

- Twelve predictors (four methods × three seeds), one shared frozen DINOv2-small encoder, full inference source, model cards, licenses, hashes and provenance.
- Every run completed 30 training epochs. Validation selected epoch 1 for ten checkpoints and epoch 2 for two; these are validation-selected weights, not epoch-30 weights.
- Twelve optional residual-calibration wrappers were fitted using training episodes only. They add no newly trained neural weights. Their comparison is validation development evidence, not a new held-out test result.
- Base and calibrated packages passed twelve-model isolated CPU offline loading/parity checks. Source videos, full datasets, optimizer states and credentials are excluded.

**Measured limits:** primary five-block real-video MSE is 0.20% lower than Framewise (paired interval includes zero) and 2.94% lower than persistence. Ten-block point results are worse than Framewise. Calibration improves validation MSE by 0.779% at h5 and 0.154% at h10 relative to also-calibrated Framewise; only the former exploratory interval excludes zero. No state-of-the-art, physical robot-control or clinical performance is established.

These models predict 1,536-dimensional frozen visual features, **not RGB video**. The study is a fixed 1,126-recording subset, not the full DROID policy benchmark. Read model cards before reuse. Project predictor weights/wrappers use MIT; unchanged DINO, LeWM and DROID components retain Apache 2.0, MIT and CC BY 4.0 respectively.

Download the archive and verify it with `SHA256SUMS`. The archive includes the exact inference and calibration source; the release tag anchors the initial published source snapshot. This is a research prerelease.
"""
    response = session.get(api + "/releases/tags/" + tag, timeout=30)
    if response.status_code == 404:
        response = session.post(api + "/releases", json={"tag_name": tag,
            "target_commitish": args.target_commit, "name": "Real DROID v1: offline predictors and development calibration",
            "body": body, "draft": True, "prerelease": True}, timeout=30)
        response.raise_for_status()
        release = response.json()
    else:
        response.raise_for_status()
        release = response.json()
    known = {row["name"]: row for row in release["assets"]}
    uploaded = []
    for name in names:
        path = files[name]
        if name in known:
            asset = known[name]
        else:
            if not release["draft"]:
                raise ValueError("A public release is immutable; a required asset is missing")
            upload_url = release["upload_url"].split("{", 1)[0]
            if urlparse(upload_url).hostname != "uploads.github.com":
                raise ValueError("Unexpected release upload host")
            content_type = "application/gzip" if name.endswith(".gz") else "application/octet-stream"
            with path.open("rb") as stream:
                response = session.post(upload_url, params={"name": name}, data=stream,
                                        headers={"Content-Type": content_type}, timeout=(30, 600))
            response.raise_for_status()
            asset = response.json()
        if asset["size"] != path.stat().st_size or asset.get("digest") != "sha256:" + expected[name]:
            raise ValueError("GitHub asset digest/size mismatch: " + name)
        uploaded.append({"name": name, "id": asset["id"], "bytes": asset["size"],
                         "sha256": expected[name], "url": asset["browser_download_url"],
                         "server_digest_verified": True})
        print(json.dumps({"uploaded": name, "bytes": asset["size"], "sha256_verified": True}), flush=True)
    if release["draft"]:
        response = session.patch(api + "/releases/" + str(release["id"]),
                                 json={"draft": False, "prerelease": True}, timeout=30)
        response.raise_for_status()
        published = response.json()
    else:
        published = release
    # Publishing replaces draft-only untagged asset URLs with the release tag.
    published_assets = {asset["id"]: asset for asset in published["assets"]}
    for asset in uploaded:
        asset["url"] = published_assets[asset["id"]]["browser_download_url"]
    # Verify the public downloads without authentication, including the full archive.
    for asset in uploaded:
        digest = hashlib.sha256()
        size = 0
        with requests.get(asset["url"], stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(4 << 20):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != asset["sha256"] or size != asset["bytes"]:
            raise ValueError("Public download mismatch: " + asset["name"])
        asset["public_download_verified"] = True
    record = {"status": "published_verified", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "release_url": published["html_url"], "release_id": published["id"],
              "tag": tag, "target_commit": args.target_commit, "prerelease": True,
              "assets": uploaded, "base_predictors": 12, "calibration_wrappers": 12,
              "encoder": "facebook/dinov2-small", "secret_findings": 0,
              "offline_parity": verification, "scope": "Offline latent-feature forecasting; calibration is development-only."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "release_url": record["release_url"], "assets": len(uploaded)}), flush=True)


if __name__ == "__main__":
    main()
