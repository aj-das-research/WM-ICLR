#!/usr/bin/env python3
"""Compact qualitative derivative of the hash-pinned, already verified replay pack.

No new model inference, case selection, scientific data or recorded image edits.
Run with .venv/bin/python paper/scripts/render_editorial_qualitative.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "paper/figure_sources/spatial_qualitative"
OUT = ROOT / "paper/generated/editorial"
STYLE = ROOT / "paper/design/editorial_style.json"
EXPECTED_REPLAY = "12d805f0692f9cd0068f7eb392cf9be996dd778f1fff00f5c1ee4273cfffac20"
WIDTH, HEIGHT = 5.5, 2.8
MODES = ("autoregressive", "transport")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verified_pack():
    replay = read(PACK / "replay.json")
    manifest = read(PACK / "manifest.json")
    if sha(PACK / "replay.json") != EXPECTED_REPLAY:
        raise ValueError("The reviewed replay changed")
    if manifest["replay_sha256"] != EXPECTED_REPLAY:
        raise ValueError("Replay manifest mismatch")
    for name, record in manifest["files"].items():
        if Path(name).name != name or sha(PACK / name) != record["sha256"]:
            raise ValueError("Portable source mismatch: " + name)
    if sha(PACK / "replay_arrays.npz") != replay["arrays_sha256"]:
        raise ValueError("Numeric pack changed")
    with np.load(PACK / "replay_arrays.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if len(replay["cases"]) != 3 or replay["population"]["episodes"] != 141:
        raise ValueError("Unexpected original three-case population")
    maps = {}
    checks = []
    for case in replay["cases"]:
        prefix = case["prefix"]
        if len(case["replay_checks"]) != 6:
            raise ValueError("Incomplete original replay")
        for entry, index in zip(case["frame_exports"], (0, 1, 2, 12), strict=True):
            pixels = np.asarray(Image.open(PACK / entry["path"]).convert("RGB"))
            if not np.array_equal(pixels, arrays[prefix + "_images"][index]):
                raise ValueError("Recorded image pixels differ")
            if sha(PACK / entry["path"]) != entry["file_sha256"]:
                raise ValueError("Recorded image hash differs")
            if int(arrays[prefix + "_frame_indices"][index]) != entry["native_frame_index"]:
                raise ValueError("Recorded image temporal index differs")
        for mode in MODES:
            errors = np.stack([arrays[f"{prefix}_{mode}_s{s}_patch_errors"] for s in (0, 1, 2)])
            if errors.shape != (3, 10, 4, 4) or not np.isfinite(errors).all() or (errors < 0).any():
                raise ValueError("Invalid measured error grid")
            native = np.stack([arrays[f"{prefix}_{mode}_s{s}_native_mse"] for s in (0, 1, 2)])
            np.testing.assert_allclose(errors.mean((-1, -2)), native, rtol=2e-6, atol=2e-7)
            # Preserve the original full-gallery FP32 aggregation order.
            maps[prefix, mode] = errors[:, 9].mean(0)
        ar = float(maps[prefix, "autoregressive"].mean())
        ours = float(maps[prefix, "transport"].mean())
        gain = 100 * (ar - ours) / ar
        np.testing.assert_allclose(gain, case["first_window_gain_percent"], rtol=2e-6, atol=1e-5)
        checks.append({"prefix": prefix, "episode_id": case["episode_id"],
                       "episode_gain_percent": case["gain_percent"],
                       "shown_window_gain_percent": case["first_window_gain_percent"],
                       "shown_AR_MSE": ar, "shown_ours_MSE": ours,
                       "gain_recomputed_from_maps": gain})
    return replay, arrays, maps, checks


def initialize_style():
    tokens = read(STYLE)
    plt.rcParams.update({"font.family": tokens["figure_font_family"], "font.size": 8.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "image.composite_image": False})
    return tokens["colors"]


def axis(fig, x, y, width, height):
    return fig.add_axes([x / WIDTH, y / HEIGHT, width / WIDTH, height / HEIGHT])


def text(fig, x, y, value, size=8.5, color="#243447", weight="normal", ha="left"):
    return fig.text(x / WIDTH, y / HEIGHT, value, fontsize=size, color=color,
                    weight=weight, ha=ha, va="center")


def sketches(arrays, maps):
    """Three different readings: case rows, side-by-side cards, evidence bands."""
    OUT.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT / "qualitative_layout_sketches.pdf") as pdf:
        for layout, name in enumerate(("A · Case-aligned evidence rows", "B · Parallel case cards",
                                       "C · Scenes above a shared diagnostic band")):
            fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=150, facecolor="white")
            text(fig, .10, 2.64, name, 9.5, weight="bold")
            placements = []
            for case_id in (0, 2):
                row = case_id // 2
                if layout == 0:
                    y = 1.48 - row * 1.03
                    placements.extend([(case_id, "obs", .10, y, 1.34, .75),
                                       (case_id, "target", 1.58, y, 1.34, .75),
                                       (case_id, "autoregressive", 3.25, y + .06, .63, .63),
                                       (case_id, "transport", 4.27, y + .06, .63, .63)])
                elif layout == 1:
                    x = .10 + row * 2.75
                    placements.extend([(case_id, "obs", x, 1.35, 1.15, .65),
                                       (case_id, "target", x + 1.25, 1.35, 1.15, .65),
                                       (case_id, "autoregressive", x + .25, .36, .67, .67),
                                       (case_id, "transport", x + 1.47, .36, .67, .67)])
                else:
                    x = .10 + row * 2.75
                    placements.extend([(case_id, "obs", x, 1.51, 1.18, .66),
                                       (case_id, "target", x + 1.26, 1.51, 1.18, .66),
                                       (case_id, "autoregressive", x + .04, .26, .86, .86),
                                       (case_id, "transport", x + 1.41, .26, .86, .86)])
            vmax = max(float(v.max()) for v in maps.values())
            for case_id, kind, x, y, w, h in placements:
                ax = axis(fig, x, y, w, h)
                prefix = f"case{case_id}"
                if kind in ("obs", "target"):
                    ax.imshow(arrays[prefix + "_images"][2 if kind == "obs" else 12], interpolation="none")
                else:
                    ax.imshow(maps[prefix, kind], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
                ax.axis("off")
                label = {"obs": "Observed", "target": "Withheld", "autoregressive": "AR", "transport": "Ours-5"}[kind]
                text(fig, x + w / 2, y - .12, label, 8, ha="center")
            pdf.savefig(fig)
            fig.savefig(OUT / f"qualitative_sketch_{'ABC'[layout]}.png", dpi=150)
            plt.close(fig)


def geometry(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bad, labels = [], []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        box = obj.get_window_extent(renderer)
        if obj.get_fontsize() < 8:
            bad.append("Font below 8pt: " + obj.get_text())
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
            bad.append("Clipped text: " + obj.get_text())
        if obj in fig.texts:
            labels.append((obj, box))
    for i, (obj, box) in enumerate(labels):
        for other, otherbox in labels[i + 1:]:
            if box.overlaps(otherbox):
                bad.append("Overlapping labels: " + obj.get_text() + " / " + other.get_text())
    if bad:
        raise ValueError("; ".join(bad))
    return {"status": "passed", "issues": [], "min_effective_font_pt": 8,
            "width_inches": WIDTH, "height_inches": HEIGHT}


def render(replay, arrays, maps, checks, colors):
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=180, facecolor="white")
    # Fixed-column geometry creates the case → observed target → model-error reading.
    for x, label in ((.72, "Last observed · f10"), (2.14, "Withheld · f60"),
                     (3.48, "AR"), (4.43, "Ours-5")):
        text(fig, x, 2.66, label, 8.5, color=colors["ours"] if label == "Ours-5" else colors["ink"], ha="center")
    vmin = min(float(value.min()) for value in maps.values())
    vmax = max(float(value.max()) for value in maps.values())
    if vmin <= 0:
        raise ValueError("A shared logarithmic scale requires positive measured values, without an invented floor")
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=False)
    displayed = []
    for row, index in enumerate((0, 2)):
        case = replay["cases"][index]
        prefix = case["prefix"]
        top = 2.39 - row * 1.23
        text(fig, .07, top, ("a  Largest episode gain" if row == 0 else "b  Largest regression"),
             9, weight="bold")
        text(fig, 4.87, top, f"Episode {case['gain_percent']:+.1f}% · shown {case['first_window_gain_percent']:+.1f}%",
             8, color=colors["secondary"], ha="right")
        bottom = top - .91
        for x, frame in ((.07, 2), (1.49, 12)):
            ax = axis(fig, x, bottom, 1.30, .73125)
            ax.imshow(arrays[prefix + "_images"][frame], interpolation="none")
            ax.axis("off")
        for col, mode in enumerate(MODES):
            ax = axis(fig, 3.10 + col * .95, bottom, .76, .76)
            last_image = ax.imshow(maps[prefix, mode], cmap="magma", norm=norm, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", alpha=.28, linewidth=.45)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_edgecolor(colors["grid"]); spine.set_linewidth(.55)
            text(fig, 3.48 + col * .95, bottom - .12, f"{maps[prefix, mode].mean():.4f}", 8,
                 color=colors["secondary"], ha="center")
        displayed.append(checks[index])
    # Exact common data bounds use all six appendix maps; logarithmic colors expose both scales.
    bar = fig.colorbar(last_image, cax=axis(fig, 4.87, .48, .065, 1.59))
    bar.set_ticks([vmin, .1, vmax]); bar.ax.set_yticklabels([f"{vmin:.4f}", "0.1", f"{vmax:.3f}"])
    bar.ax.minorticks_off()
    bar.ax.tick_params(length=2, pad=2, labelsize=8)
    bar.outline.set_linewidth(.5)
    fig.text(5.44 / WIDTH, 1.25 / HEIGHT, "MSE ↓ · log scale", fontsize=8,
             color=colors["secondary"], rotation=90, ha="center", va="center")
    audit = geometry(fig)
    stem = OUT / "qualitative_main"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, facecolor="white")
    plt.close(fig)
    for name, extra, dpi in (("paper_size", [], 120), ("enlarged", [], 300), ("grayscale", ["-gray"], 120)):
        subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", str(dpi), *extra,
                        str(stem.with_suffix(".pdf")), str(OUT / ("qualitative_" + name))],
                       check=True, capture_output=True)
    caption = (r"\textbf{Selected real-video gain and failure.} "
               r"Prespecified extremes among 141 DROID development episodes; each displays its first eligible window. "
               r"RGB frames are recorded observations and targets, not predictions. "
               r"Maps show native standardized $h=10$ feature MSE averaged over three seeds, with one unclipped logarithmic scale. "
               r"AR is matched autoregression; Ours-5 is bounded mixing. "
               r"Positive percentages mean lower error: `episode' averages its windows, while `shown' denotes this window. "
               r"The complete appendix gallery retains the median and selection record. Images: DROID (CC BY 4.0).")
    (OUT / "qualitative_caption.tex").write_text(caption + "\n")
    (OUT / "qualitative_figure.tex").write_text("\\begin{figure}[!htb]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{generated/editorial/qualitative_main.pdf}\n"
        "\\caption{" + caption + "}\n\\label{fig:editorial-qualitative}\n\\end{figure}\n")
    proof_dir = OUT / "qualitative_proof"
    proof_dir.mkdir(exist_ok=True)
    proof = (r"\documentclass{article}" "\n" r"\usepackage{iclr2027_conference,times,graphicx,amsmath,hyperref}" "\n"
             r"\begin{document}\begin{figure}[!htb]\centering" "\n"
             r"\includegraphics[width=\linewidth]{../qualitative_main.pdf}" "\n"
             r"\caption{" + caption + r"}\end{figure}\end{document}" "\n")
    (proof_dir / "proof.tex").write_text(proof)
    env = dict(os.environ, TEXINPUTS=str(ROOT / "paper/template/official/iclr2027") + "//:" + os.environ.get("TEXINPUTS", ""))
    for i in range(2):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "proof.tex"],
                                cwd=proof_dir, env=env, capture_output=True, text=True)
        (proof_dir / f"pass{i+1}.stdout").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError("Qualitative official-width proof failed")
    log = (proof_dir / "proof.log").read_text()
    warnings = [line for line in log.splitlines() if "Overfull" in line or "Underfull" in line or "Float too large" in line]
    if warnings:
        raise ValueError("Qualitative proof layout warnings: " + repr(warnings))
    subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "120", str(proof_dir / "proof.pdf"),
                    str(proof_dir / "page")], check=True, capture_output=True)
    evidence = {"status": "numeric_and_geometry_passed_visual_review_pending",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "replay_sha256": EXPECTED_REPLAY, "style_sha256": sha(STYLE),
                "source_sha256": sha(__file__), "selection": "Existing prespecified best and worst of original three-case pack; no reselection",
                "recorded_pixel_check": "All twelve PNGs are bit-identical to the corresponding stored decoded RGB arrays",
                "shown_observed_frame_index": 10, "withheld_frame_index": 60,
                "omitted_support_display": "Model still receives frames 0, 5, 10; only last observed frame is displayed",
                "coordinate_space": "Native shared-channel standardized 4x4 features",
                "common_vmin": vmin, "common_vmax": vmax, "color_normalization": "LogNorm; positive measured minimum, no artificial floor, clip=False",
                "scale_population": "All six maps of original three-case gallery",
                "displayed_cases": displayed, "all_recomputed_case_values": checks,
                "geometry": audit, "proof_warnings": warnings,
                "scientific_scope": "Existing replay re-render only; no fresh inference, no independent-test result, no causal/RGB reconstruction claim",
                "input_files": {str(p.relative_to(ROOT)): sha(p) for p in sorted(PACK.iterdir()) if p.is_file()},
                "outputs": {str(p.relative_to(ROOT)): sha(p) for p in (stem.with_suffix(".pdf"), stem.with_suffix(".svg"), stem.with_suffix(".png"),
                            OUT / "qualitative_caption.tex", OUT / "qualitative_figure.tex", proof_dir / "proof.pdf")}}
    (OUT / "qualitative_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketches-only", action="store_true")
    args = parser.parse_args()
    replay, arrays, maps, checks = verified_pack()
    colors = initialize_style()
    sketches(arrays, maps)
    if not args.sketches_only:
        render(replay, arrays, maps, checks, colors)
    print("Saved compact qualitative " + ("layout sketches" if args.sketches_only else "figure, source checks and official-width proof"))


if __name__ == "__main__":
    main()
