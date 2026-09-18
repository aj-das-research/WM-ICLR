#!/usr/bin/env python3
"""Prepare a public static snapshot, or package the committed snapshot for Pages."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FILES = ["index.html", "styles.css", "app.js", "real-results.json", "publication-manifest.json"]
ASSETS = ["paper.pdf", "method.svg", "recorded-droid.mp4", "recorded-droid-poster.png", "video-provenance.json", "real-results.md", "real-protocol.md", "real-interpretation.md", "real-comparison.svg", "DROID-LICENSE.txt"]
NAMES = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics", "factorized": "ShiftWM (ours)", "action_free": "Action-free", "persistence": "Persistence", "constant_velocity": "Constant feature velocity"}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def refresh():
    report_path = ROOT / "reports/real_droid_results.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "completed" or report["completed_runs"] != 12 or report["completed_evaluations"] != 48:
        raise ValueError("Public results require the completed 12-run / 48-evaluation study")
    output = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": digest(report_path), "completed_runs": 12, "completed_evaluations": 48, "episodes": 1126, "split": {"train":851,"validation":143,"test":132}, "populations": []}
    for p in report["populations"]:
        a = p["aggregate"]
        if a["status"] != "completed": raise ValueError("Incomplete population")
        metric = f"h{a['horizon']}_standardized_mse"
        pair = {r["reference"]: r for r in a["paired_comparisons"] if r["metric"] == metric}
        rows = []
        for mode, name in NAMES.items():
            learned = a["methods"].get(mode)
            mse = learned[metric]["mean"] if learned else a["fixed_support_baselines"][mode][metric]
            rows.append({"mode": mode, "name": name, "mse": mse, "seed_sd": learned[metric]["seed_sd"] if learned else None})
        ours = next(r["mse"] for r in rows if r["mode"] == "factorized")
        comparisons = {}
        for reference in ["framewise", "persistence"]:
            ref = next(r["mse"] for r in rows if r["mode"] == reference)
            comparisons[reference] = {"relative_reduction_percent": 100*(ref-ours)/ref, "mse_difference": pair[reference]["mean_difference"], "ci95": pair[reference]["ci95"]}
        output["populations"].append({"camera": a["camera"], "horizon": a["horizon"], "kind": a["population"], "episodes": pair["framewise"]["episode_count"], "sessions": pair["framewise"]["session_count"], "rows": rows, "comparisons": comparisons})
    (HERE / "real-results.json").write_text(json.dumps(output, indent=2) + "\n")
    preview = "data/real_video/droid_selected/processed/preview/"
    mapping = {
        "paper/world_model_draft.pdf": "paper.pdf",
        "paper/figures/method.svg": "method.svg",
        preview + "recorded_three_camera_episode.mp4": "recorded-droid.mp4",
        preview + "provenance.json": "video-provenance.json",
        "reports/real_droid_results.md": "real-results.md",
        "reports/real_droid_protocol.md": "real-protocol.md",
        "reports/real_droid_interpretation.md": "real-interpretation.md",
        "paper/generated/real_video/comparison_recorded_droid.svg": "real-comparison.svg",
        "data/real_video/droid_selected/raw/1.0.0/CC-BY-4.0": "DROID-LICENSE.txt",
    }
    (HERE / "assets").mkdir(exist_ok=True)
    for source, name in mapping.items():
        path = ROOT / source
        if not path.is_file() and name == "DROID-LICENSE.txt":
            alternatives = list((ROOT / "data/real_video/droid_selected/raw").rglob("*LICENSE*"))
            if not alternatives: raise FileNotFoundError("DROID publisher license missing")
            path = alternatives[0]
        shutil.copy2(path, HERE / "assets" / name)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(HERE/"assets/recorded-droid.mp4"), "-frames:v", "1", str(HERE/"assets/recorded-droid-poster.png")], check=True)
    manifest = {"poster_derivation": "First decoded frame of the attributed recorded video, without resizing or overlays.", "generated_at_utc": output["generated_at_utc"], "repository": "https://github.com/aj-das-research/WM-ICLR", "source_report_sha256": output["source_sha256"], "scope": "Static recorded-video playback and source-derived aggregate results; no online model inference.", "files": {"assets/" + n: {"bytes": (HERE/"assets"/n).stat().st_size, "sha256": digest(HERE/"assets"/n)} for n in ASSETS}}
    manifest["files"]["real-results.json"] = {"bytes": (HERE/"real-results.json").stat().st_size, "sha256": digest(HERE/"real-results.json")}
    (HERE / "publication-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

def package(output):
    output = output.resolve()
    if output == HERE: raise ValueError("Use a separate export directory")
    manifest = json.loads((HERE/"publication-manifest.json").read_text())
    for rel, record in manifest["files"].items():
        if digest(HERE/rel) != record["sha256"]: raise ValueError(f"Snapshot mismatch: {rel}; refresh locally first")
    if output == HERE / "export" and output.exists():
        shutil.rmtree(output)  # This one path is documented generated output.
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    for name in FILES: shutil.copy2(HERE/name, output/name)
    for name in ASSETS: shutil.copy2(HERE/"assets"/name, output/"assets"/name)
    (output/".nojekyll").write_text("")
    print(json.dumps({"output":str(output),"files":len(FILES)+len(ASSETS)+1,"bytes":sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-only", action="store_true", help="Package committed assets without cluster datasets/checkpoints")
    parser.add_argument("--output", type=Path, default=HERE/"export")
    args = parser.parse_args()
    if not args.snapshot_only: refresh()
    package(args.output)
