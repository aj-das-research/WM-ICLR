#!/usr/bin/env python3
"""Refresh presentation tables and plots before the publisher takes its fingerprint.

This operational wrapper never trains or evaluates a model. Each reviewed
renderer owns its numerical gate. Missing expected programs and invalid evidence
fail the service preflight; pending experiments remain the renderer's explicit
pending state. The build lock prevents compilation during a partial refresh.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RENDERERS = (
    ("paper/scripts/render_current_real_scorecards.py", "--if-ready"),
    ("paper/scripts/render_simulator_tables.py", "--refresh-source-pack"),
    ("paper/scripts/render_iws_unbounded_results.py", "--if-ready"),
    ("paper/scripts/render_iws_variant_forecasts.py", "--if-ready"),
    ("paper/scripts/render_iws_compact_evidence.py",),
    ("paper/scripts/render_iws_predictor_resources.py",),
    ("paper/scripts/render_iws_reserved_evidence.py", "--if-ready"),
    ("paper/scripts/render_iws_bound_diagnostic.py", "--if-ready"),
    ("paper/scripts/render_external_dinowm.py", "--if-ready"),
    ("paper/scripts/render_editorial_tables.py",),
    ("scripts/extensions_diagnostics_20260920/render_tables.py", "--if-ready"),
    ("paper/scripts/render_comparison_inventory.py",),
)
REGISTRATIONS = (
    "configs/real_video_iws/training_registration_v1.json",
    "configs/real_video_iws_unbounded/registration_v1.json",
    "reports/real_video_iws/recovery/common_cpu_v1/registration.json",
    "configs/real_video_iws_reserved_v1/registration.json",
    "configs/real_video_iws_reserved_recovery_v2/registration.json",
)
RECEIPT = "reports/evidence/comparison_presentation_refresh.json"
OUTPUT_DIRS = ("paper/generated/benchmark_scorecards", "paper/generated/simulator_tables",
               "paper/generated/iws_variant_forecasts", "paper/generated/iws_compact_evidence",
               "paper/generated/iws_resources", "paper/generated/iws_reserved_evidence",
               "paper/generated/extensions_completed", "paper/generated/iws_bound_diagnostic",
               "paper/generated/external_dinowm")
SOURCE_PACK_DIRS = ("paper/table_sources/simulator_tables", "paper/figure_sources/current_real_scorecards",
                    "paper/figure_sources/iws_compact_evidence", "paper/table_sources/iws_predictor_resources_v1",
                    "paper/figure_sources/iws_reserved_evidence")
COMPONENT_TEX = "paper/generated/experiment_alignment/unbounded_component.tex"
COMPONENT_JSON = "paper/generated/experiment_alignment/unbounded_component.json"
VARIANT_PLOT = "paper/scripts/render_iws_variant_forecasts.py"
EDITORIAL_OUTPUTS = ("paper/generated/editorial/spatial_main_table.tex",
                     "paper/generated/editorial/context_main_table.tex",
                     "paper/generated/editorial/main_tables_evidence.json")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path leaves the workspace: " + relative)
    return path


def frozen_bindings(root, registrations=REGISTRATIONS):
    """Check pinned files, including recovery operations; read no dataset arrays."""
    bindings = {}
    counts = {}
    for relative in registrations:
        path = local(root, relative)
        data = json.loads(path.read_text())
        bindings[relative] = sha(path)
        count = 0
        for field in ("dependencies", "frozen_scientific_dependencies", "operational_dependencies"):
            for name, expected in data.get(field, {}).items():
                if name in bindings and bindings[name] != expected:
                    raise ValueError("Conflicting frozen binding: " + name)
                actual = sha(local(root, name))
                if actual != expected:
                    raise ValueError("Frozen dependency changed: " + name)
                bindings[name] = actual
                count += 1
        if not count:
            raise ValueError("Registration has no checked dependencies: " + relative)
        counts[relative] = count
    return bindings, counts


def atomic_json(path, value):
    content = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".comparison-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def refresh(root=ROOT, renderers=RENDERERS, registrations=REGISTRATIONS, check_only=False):
    root = Path(root).resolve()
    missing = [row[0] for row in renderers if not local(root, row[0]).is_file()]
    if missing:
        raise FileNotFoundError("Presentation refresh requires all reviewed renderers; missing: " + ", ".join(missing))
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("Presentation interpreter is absent: " + str(python))
    lock_dir = root / "runs/publishing"
    lock_dir.mkdir(parents=True, exist_ok=True)
    build_dir = root / "paper/build"
    build_dir.mkdir(parents=True, exist_ok=True)
    # Stable order: operational lock, then the same lock used by paper/build.sh.
    with (lock_dir / "comparison_presentations.lock").open("a") as own_lock, \
            (build_dir / ".build.lock").open("a") as build_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX)
        fcntl.flock(build_lock, fcntl.LOCK_EX)
        before, counts = frozen_bindings(root, registrations)
        sources = {row[0]: sha(local(root, row[0])) for row in renderers}
        if set(sources) & set(before):
            raise ValueError("A presentation renderer is registered scientific code")
        if check_only:
            return {"status": "preflight_passed", "renderers": list(sources), "registration_counts": counts}
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        gate_states = {}
        for row in renderers:
            result = subprocess.run([str(python), "-B", str(local(root, row[0])), *row[1:]],
                                    cwd=root, env=env, text=True, capture_output=True, timeout=300)
            if result.returncode:
                detail = (result.stderr or result.stdout)[-4000:]
                raise RuntimeError(f"Presentation renderer failed ({result.returncode}): {row[0]}\n{detail}")
            try:
                report = json.loads(result.stdout.strip().splitlines()[-1])
                status = report["status"]
            except (ValueError, IndexError, KeyError, TypeError) as error:
                raise ValueError("Renderer omitted its structured completion status: " + row[0]) from error
            if not isinstance(status, str) or not status or status in ("failed", "error", "invalid"):
                raise ValueError("Renderer did not report success/pending: " + row[0])
            # Exclude write/no-write differences so repeat refreshes stay stable.
            gate_states[row[0]] = "pending" if status == "pending" else "completed"
        if gate_states.get(VARIANT_PLOT) == "completed":
            # Refresh the real pack before the component table, then the plot.
            # A finalizer arriving mid-cycle cannot produce a plot whose caption
            # references a table that the same cycle has not yet validated.
            table = local(root, COMPONENT_TEX)
            record = json.loads(local(root, COMPONENT_JSON).read_text())
            if (record.get("status") != "complete_validated_exploratory_development"
                    or record.get("official_validation_payloads_read") != 0
                    or sha(table) != record.get("tex_sha256")
                    or r"\label{tab:iws-unbounded-component}" not in table.read_text()):
                raise ValueError("Completed variant plot lacks its validated paired-comparison table")
        after, _ = frozen_bindings(root, registrations)
        if before != after or any(sha(local(root, path)) != digest for path, digest in sources.items()):
            raise ValueError("Source changed during presentation refresh; publication must retry")
        outputs = {str(path.relative_to(root)): sha(path)
                   for directory in OUTPUT_DIRS for path in sorted((root / directory).rglob("*"))
                   if path.is_file() and not path.name.startswith(".")}
        outputs.update({name: sha(local(root, name)) for name in (COMPONENT_TEX, COMPONENT_JSON, *EDITORIAL_OUTPUTS)
                        if local(root, name).is_file()})
        source_packs = {str(path.relative_to(root)): sha(path)
                        for directory in SOURCE_PACK_DIRS for path in sorted((root / directory).rglob("*"))
                        if path.is_file() and not path.name.startswith(".")}
        if not outputs:
            raise ValueError("Presentation renderers produced no scorecard artifacts")
        value = {"schema": "shiftwm_comparison_presentation_refresh_v1", "status": "passed",
                 "scope": "Presentation only; renderer gates determine complete or pending scientific scope",
                 "commands": [list(row) for row in renderers], "renderers": sources,
                 "renderer_gate_states": gate_states,
                 "wrapper_sha256": sha(Path(__file__)), "frozen_bindings": before,
                 "registration_counts": counts, "outputs": outputs,
                 "public_source_packs": source_packs,
                 "training_or_evaluation_invoked": False, "publishing_invoked": False,
                 "visual_review": "Separate source-bound numerical and manuscript layout reviews required"}
        changed = atomic_json(root / RECEIPT, value)
        return {"status": "refreshed" if changed else "unchanged_verified", "receipt": RECEIPT,
                "output_files": len(outputs), "frozen_files_checked": len(before)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check all expected programs and frozen bindings without rendering")
    args = parser.parse_args()
    try:
        print(json.dumps(refresh(check_only=args.check_only), sort_keys=True))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
