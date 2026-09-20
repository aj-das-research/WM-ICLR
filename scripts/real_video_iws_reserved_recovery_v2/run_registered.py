#!/usr/bin/env python3
"""Run one immutable roster entry in a fresh CPU inference process."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video_iws_reserved_recovery.protocol import checked_registration


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    registry = checked_registration(ROOT)
    if not 0 <= args.index < len(registry["runs"]):
        raise ValueError("Array index is outside the fixed reserved roster")
    row = registry["runs"][args.index]
    script = ROOT / "scripts/real_video_iws_reserved_recovery_v2/evaluate.py"
    os.execv(sys.executable, [sys.executable, "-I", str(script), "--name", row["name"]])


if __name__ == "__main__":
    main()
