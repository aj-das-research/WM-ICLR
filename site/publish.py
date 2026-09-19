#!/usr/bin/env python3
"""Prepare a public static snapshot, or package the committed snapshot for Pages."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FILES = ["index.html", "styles.css", "app.js", "real-results.json", "showcase.json", "fresh-results.json", "demo-config.json", "publication-manifest.json"]
ASSETS = ["paper.pdf", "method.svg", "recorded-droid.mp4", "recorded-droid-poster.png", "video-provenance.json", "real-results.md", "real-protocol.md", "real-interpretation.md", "real-comparison.svg", "DROID-LICENSE.txt"]
ASSETS += ["spatial_task.svg", "spatial_qualitative.svg", "spatial_versions_comparison.svg"]
ASSETS += ["spatial_architecture_main.svg", "anchoring_teaser.svg"]
NAMES = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics", "factorized": "Historical context model", "action_free": "Action-free", "persistence": "Persistence", "constant_velocity": "Constant feature velocity"}
OUTLINED_FIGURES = {
    "paper/generated/editorial/task_story.pdf": "spatial_task.svg",
    "paper/generated/editorial/architecture_visual_design.pdf": "spatial_architecture_main.svg",
    "paper/generated/editorial/teaser_camera_story.pdf": "anchoring_teaser.svg",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def showcase_media():
    data = json.loads((HERE / "showcase.json").read_text())
    media = data["media"]
    if len(media) != len(set(media)) or any(not p.startswith("assets/sim-") or Path(p).name != p.removeprefix("assets/") or not p.endswith(".png") for p in media):
        raise ValueError("Invalid showcase media paths")
    return media

def refresh_web_figures(destination):
    """Derive self-contained, font-independent web SVGs from reviewed PDFs.

    Keep the canonical editable SVGs in paper/ unchanged. Validate every staged
    conversion before replacing any of the three public files.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    with tempfile.TemporaryDirectory(prefix=".outlined-", dir=destination) as staging:
        staging = Path(staging)
        for relative, public_name in OUTLINED_FIGURES.items():
            source = ROOT / relative
            source_hash = digest(source)
            info = subprocess.run(["pdfinfo", str(source)], check=True, capture_output=True, text=True).stdout
            pages = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
            size = re.search(r"^Page size:\s+([\d.]+) x ([\d.]+) pts", info, re.MULTILINE)
            if pages is None or int(pages[1]) != 1 or size is None:
                raise ValueError("Expected a single-page reviewed figure: " + relative)
            expected = [0.0, 0.0, float(size[1]), float(size[2])]
            target = staging / public_name
            subprocess.run(["pdftocairo", "-svg", "-f", "1", "-l", "1", str(source), str(target)], check=True)
            tree = ET.parse(target).getroot()
            viewbox = [float(v) for v in tree.attrib.get("viewBox", "").split()]
            if len(viewbox) != 4 or any(not math.isfinite(v) or abs(v-e) > 1e-5 for v, e in zip(viewbox, expected)):
                raise ValueError("Web SVG changed reviewed PDF dimensions: " + relative)
            ids = [node.attrib["id"] for node in tree.iter() if "id" in node.attrib]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate web SVG IDs: " + public_name)
            for node in tree.iter():
                if node.tag.rsplit("}", 1)[-1] in {"text", "foreignObject", "script", "font", "font-face"}:
                    raise ValueError("Web SVG must contain outlined glyphs, not live text: " + public_name)
                for key, value in node.attrib.items():
                    if key.rsplit("}", 1)[-1] == "href":
                        if value.startswith("#") and value[1:] in ids:
                            continue
                        if value.startswith("data:image/png;base64,"):
                            continue
                        raise ValueError("Unresolved or external SVG reference: " + public_name)
                    for reference in re.findall(r"url\(#([^)]*)\)", value):
                        if reference not in ids:
                            raise ValueError("Unresolved SVG geometry reference: " + public_name)
            if digest(source) != source_hash:
                raise ValueError("Reviewed PDF changed during web conversion: " + relative)
            records["assets/" + public_name] = {
                "source_pdf": relative, "source_pdf_sha256": source_hash,
                "svg_sha256": digest(target), "viewbox_points": viewbox,
                "conversion": "pdftocairo -svg; embedded glyph outlines and source images; no external fonts",
            }
        for public_name in OUTLINED_FIGURES.values():
            (staging / public_name).replace(destination / public_name)
    return records

def refresh():
    interpreter = sys.executable if all(importlib.util.find_spec(m) for m in ("numpy", "PIL")) else str(ROOT / ".venv/bin/python")
    subprocess.run([interpreter, str(HERE / "prepare_showcase.py")], check=True)
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
    fresh_path = ROOT / "reports/real_droid_fresh_evaluation_results.json"
    fresh = json.loads(fresh_path.read_text())
    if fresh["status"] != "completed":
        raise ValueError("Fresh holdout results must be complete")
    primary = fresh["primary_comparison"]
    if primary["method"] != "calibrated/factorized" or primary["reference"] != "calibrated/framewise":
        raise ValueError("Unexpected fresh primary comparison")
    population = fresh["populations"][fresh["primary_population"]]
    fresh_output = {"schema_version": 1, "status": "completed", "source_sha256": digest(fresh_path),
                    "population": fresh["primary_population"], "primary_comparison": primary,
                    "ours_mse": population["methods"][primary["method"]][primary["metric"]]["mean"],
                    "framewise_mse": population["methods"][primary["reference"]][primary["metric"]]["mean"],
                    "selection": fresh["selection"], "uncertainty": fresh["uncertainty"], "limitations": fresh["limitations"]}
    (HERE / "fresh-results.json").write_text(json.dumps(fresh_output, indent=2) + "\n")
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
        "paper/generated/real_video/spatial_qualitative.svg": "spatial_qualitative.svg",
        "paper/generated/real_video/spatial_versions_comparison.svg": "spatial_versions_comparison.svg",
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
    outlined_figures = refresh_web_figures(HERE / "assets")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(HERE/"assets/recorded-droid.mp4"), "-frames:v", "1", str(HERE/"assets/recorded-droid-poster.png")], check=True)
    manifest = {"poster_derivation": "First decoded frame of the attributed recorded video, without resizing or overlays.", "generated_at_utc": output["generated_at_utc"], "repository": "https://github.com/aj-das-research/WM-ICLR", "source_report_sha256": output["source_sha256"], "scope": "Recorded simulation rollout explorer, DROID video playback, source-derived forecast comparisons, and released checkpoint links. Separately hosted live inference is connected only after verification.", "files": {"assets/" + n: {"bytes": (HERE/"assets"/n).stat().st_size, "sha256": digest(HERE/"assets"/n)} for n in ASSETS}}
    manifest["outlined_web_figures"] = outlined_figures
    for rel in ["real-results.json", "showcase.json", "fresh-results.json", "demo-config.json", *showcase_media()]:
        manifest["files"][rel] = {"bytes": (HERE / rel).stat().st_size, "sha256": digest(HERE / rel)}
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
    for rel in showcase_media(): shutil.copy2(HERE / rel, output / rel)
    (output/".nojekyll").write_text("")
    print(json.dumps({"output":str(output),"files":len(FILES)+len(ASSETS)+len(showcase_media())+1,"bytes":sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-only", action="store_true", help="Package committed assets without cluster datasets/checkpoints")
    parser.add_argument("--output", type=Path, default=HERE/"export")
    args = parser.parse_args()
    if not args.snapshot_only: refresh()
    package(args.output)
