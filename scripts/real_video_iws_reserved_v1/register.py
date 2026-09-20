#!/usr/bin/env python3
"""Prepare or freeze reserved identities without opening reserved payloads."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws_reserved.protocol import (
    REGISTRATION_PATH, checked_registration, immutable_json, prepare_registration, sha)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("candidate", "freeze", "check"))
    args = parser.parse_args()
    candidate = ROOT / "reports/real_video_iws_reserved_v1/registration_candidate.json"
    if args.command == "candidate":
        registry = prepare_registration(ROOT)
        # Candidates are never an authorization for raw data access.
        if candidate.exists():
            old = json.loads(candidate.read_text())
            registry["created_utc"] = old["created_utc"]
        immutable_json(registry, candidate)
        print(json.dumps({"status": "candidate_created_no_payload_access", "sha256": sha(candidate), "runs": 36}))
    elif args.command == "freeze":
        expected = json.loads(candidate.read_text())
        actual = prepare_registration(ROOT)
        actual["created_utc"] = expected["created_utc"]
        if actual != expected:
            raise ValueError("Candidate source changed before freeze; review a new candidate first")
        immutable_json(expected, ROOT / REGISTRATION_PATH)
        checked_registration(ROOT, require_review=False)
        print(json.dumps({"status": "registered_awaiting_independent_preaccess_receipt", "sha256": sha(ROOT / REGISTRATION_PATH)}))
    else:
        registry = checked_registration(ROOT)
        print(json.dumps({"status": "independent_preaccess_gate_passed", "sha256": sha(ROOT / REGISTRATION_PATH), "runs": len(registry["runs"])}))


if __name__ == "__main__":
    main()
