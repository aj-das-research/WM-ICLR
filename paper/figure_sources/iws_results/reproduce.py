#!/usr/bin/env python3
"""Replay the completed public IWS plot payload without private research data.

This reproduces a published visualization, not its scientific evaluations.
The exact pinned renderer is imported only for plot_payload and make_figure.
No finalizer, reporting helper, model, checkpoint, or feature cache is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
RENDERER_KEY = "paper/scripts/render_iws_results.py"
RENDERER_SHA256 = "c762da602f874336891628bb13ed55815c3f01145f684806ef2e3609f4655671"
BRIEF_KEY = "paper/figure_sources/iws_results/brief.md"
EXPECTED_EXPORTS = {"forecast_transfer.pdf", "forecast_transfer.svg",
                    "forecast_transfer.png", "forecast_transfer_figure.tex"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def pixels(source):
    import numpy as np
    from PIL import Image
    with Image.open(source) as image:
        return np.asarray(image.convert("RGBA")).copy()


def pdf_pixels(path):
    process = subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-singlefile",
                              "-r", "144", "-png", str(path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return pixels(BytesIO(process.stdout))


def replay(evidence, renderer, brief, output, if_ready=False):
    evidence, renderer, brief, output = map(lambda p: Path(p).resolve(),
                                           (evidence, renderer, brief, output))
    if not evidence.is_file():
        if if_ready:
            return {"status": "pending", "reason": "Finalized public plot evidence absent",
                    "outputs_written": False}
        raise ValueError("Completed forecast_transfer.json is required; no substitute data are used")
    if output.exists():
        raise ValueError("Choose a new separate output directory; existing artifacts are never replaced")
    if any(p.is_relative_to(output) for p in (evidence, renderer, brief)):
        raise ValueError("Output cannot contain or replace replay inputs")
    if shutil.which("pdftoppm") is None:
        raise ValueError("Poppler pdftoppm is required for actual PDF pixel comparison")
    record = json.loads(evidence.read_text())
    content = record["bound_payload"]
    if (content.get("schema") != "shiftwm_iws_forecast_figure_v1"
            or record.get("fingerprint") != canonical_hash(content)):
        raise ValueError("Public plot payload fingerprint/schema differs")
    if (sha(renderer) != RENDERER_SHA256
            or content["renderer_dependencies_sha256"].get(RENDERER_KEY) != RENDERER_SHA256
            or content["renderer_dependencies_sha256"].get(BRIEF_KEY) != sha(brief)):
        raise ValueError("Renderer or brief differs from the exact finalized source snapshot")
    if set(record.get("outputs_sha256", {})) != EXPECTED_EXPORTS:
        raise ValueError("Authoritative plot export set is incomplete")
    source_hashes = {str(evidence): sha(evidence), str(renderer): sha(renderer), str(brief): sha(brief)}
    for name, expected in record["outputs_sha256"].items():
        path = evidence.parent / name
        if sha(path) != expected:
            raise ValueError("Authoritative export changed: " + name)
        source_hashes[str(path)] = expected
    development = content["full_validated_development"]
    tasks = ("pusht", "bimanual_box", "bimanual_rope")
    modes = ("autoregressive", "anchored_additive", "bounded_spatial_mix")
    expected_runs = {(task, mode, seed) for task in tasks for mode in modes for seed in (0, 1, 2)}
    runs = development.get("runs", [])
    if (development.get("status") != "complete_validated_development"
            or development.get("scope") != "internal_development"
            or development.get("finalization_sha256") != content.get("finalization_sha256")
            or len(runs) != 27
            or {(r.get("task"), r.get("mode"), r.get("seed")) for r in runs} != expected_runs):
        raise ValueError("Payload does not attest the complete declared development campaign")

    attempts = []
    def denied(*args, **kwargs):
        attempts.append("network")
        raise RuntimeError("Public plot replay is offline")
    for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                        (socket, "create_connection"), (socket, "getaddrinfo")):
        setattr(owner, name, denied)
    spec = importlib.util.spec_from_file_location("_pinned_public_iws_plot", renderer)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    # These two functions use only the supplied JSON and plotting libraries.
    # Deliberately never call load_evidence(), render(), or the frozen reporter.
    payload = plot.plot_payload(development)
    if payload != content["numerical_payload"] or payload["plotted_means"] != 708:
        raise ValueError("Public numerical payload differs from its complete curve reconstruction")
    font = Path(plot.font_manager.findfont(
        plot.font_manager.FontProperties(family="Liberation Sans"), fallback_to_default=False))
    runtime = content["runtime"]
    if (plot.matplotlib.__version__ != runtime["matplotlib"]
            or plot.np.__version__ != runtime["numpy"]
            or sha(font) != runtime["font_sha256"]):
        raise ValueError("Use the recorded Matplotlib/NumPy versions and exact Liberation Sans font")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".iws-public-replay-", dir=output.parent))
    try:
        fig, geometry = plot.make_figure(payload)
        try:
            for extension in ("pdf", "svg", "png"):
                fig.savefig(stage / f"forecast_transfer.{extension}", dpi=300)
        finally:
            plot.plt.close(fig)
        (stage / "forecast_transfer_figure.tex").write_text(content["include"])
        comparison = {}
        for name, loader in (("png", pixels), ("pdf", pdf_pixels)):
            original = loader(evidence.parent / f"forecast_transfer.{name}")
            reproduced = loader(stage / f"forecast_transfer.{name}")
            if original.shape != reproduced.shape or not plot.np.array_equal(original, reproduced):
                raise ValueError(f"Reproduced {name.upper()} pixels differ from the authoritative export")
            comparison[name] = {"equal": True, "shape": list(original.shape),
                                "rgba_pixels_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                                "maximum_absolute_difference": 0}
        if attempts or any(sha(path) != expected for path, expected in source_hashes.items()):
            raise ValueError("Network attempted or public inputs changed during replay")
        manifest = {
            "schema": "shiftwm_iws_public_plot_replay_v1", "status": "passed_exact_pixel_replay",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Public figure reconstruction only; scientific checkpoint/window evaluation is not rerun.",
            "replay_source_sha256": sha(__file__), "renderer_sha256": RENDERER_SHA256,
            "input_files_sha256": source_hashes, "finalization_sha256": content["finalization_sha256"],
            "numerical_payload_sha256": canonical_hash(payload), "plotted_means": 708,
            "methods_per_task": 4, "tasks": list(tasks), "geometry_checks": geometry,
            "pixel_comparison": comparison, "pdf_rasterizer": "pdftoppm; first page; 144 dpi; RGBA",
            "metadata_policy": "PDF dates and SVG internal IDs may differ; byte identity is not asserted.",
            "private_checkpoints_or_caches_opened": False, "scientific_evaluation_recomputed": False,
            "network_attempts": 0, "outputs_sha256": {p.name: sha(p) for p in stage.iterdir()},
        }
        # A success manifest appears only after genuine final evidence and pixel parity.
        (stage / "replay_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise ValueError("Output appeared during replay; refusing replacement")
        os.rename(stage, output)
        return {"status": manifest["status"], "output": str(output), "plotted_means": 708,
                "outputs_written": True}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "paper/generated/experiment_alignment/forecast_transfer.json")
    parser.add_argument("--renderer", type=Path, default=ROOT / RENDERER_KEY)
    parser.add_argument("--brief", type=Path, default=ROOT / BRIEF_KEY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    print(json.dumps(replay(args.evidence, args.renderer, args.brief, args.output, args.if_ready)))
