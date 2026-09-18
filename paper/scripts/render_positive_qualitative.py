#!/usr/bin/env python3
"""Explain every observed ShiftWM-only success versus Framewise at seed zero.

Goal contours are extracted from the saved goal pixels for annotation only.
Physical scores remain the original simulator measurements. No replay,
interpolated state, segmentation-derived physical score or mechanism claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("positive_qualitative_source", Path(__file__).with_name("render_qualitative.py"))
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
OUT = ROOT / "paper/generated/qualitative"
PURPLE = "#a14ea4"
INK, GREEN, RED, BLUE, MUTED = source.INK, source.GREEN, source.RED, source.BLUE, source.MUTED


def positive_cases(examples):
    return sorted([ex for ex in examples if ex["category"] == "ours_only"],
                  key=lambda ex: (ex["environment"], ex["trajectory_id"], ex["observation_id"]))


def matched_frame(example):
    """Last jointly observed pre-stop time, never a synthesized baseline state."""
    a, b = (example["methods"][m] for m in ("factorized", "framewise"))
    common = set(a["frame_steps"]) & set(b["frame_steps"])
    common = [t for t in common if t <= a["record"]["native_steps"]]
    if not common:
        raise ValueError("No actually recorded shared comparison time")
    time = max(common)
    return time, {m: example["methods"][m]["frame_steps"].index(time)
                  for m in ("factorized", "framewise")}


def goal_mask(goal, environment):
    """Display-only silhouettes for the known observed development palette.

    PushT: gray block + blue pusher, excluding the green persistent marker.
    Reacher: orange arm, excluding the dark background. Sources are untouched.
    """
    if goal.shape != (224, 224, 3) or goal.dtype != np.uint8:
        raise ValueError("Expected original 224x224 uint8 goal pixels")
    r, g, b = goal.astype(np.int16).transpose(2, 0, 1)
    if environment == "pusht":
        block = (r > 70) & (r < 175) & (r >= g) & (r - g < 22) & (g >= b) & (g - b < 18)
        pusher = (b - r > 35) & (b - g > 25) & (b > 110)
        mask = block | pusher
        if block.sum() < 100 or pusher.sum() < 20:
            raise ValueError("Goal block/pusher cannot be reliably annotated with the recorded palette")
    elif environment == "reacher":
        mask = (r > 150) & (r - g > 30) & (g - b > 30)
    else:
        raise ValueError("Unknown environment")
    if not 30 < mask.sum() < 5000:
        raise ValueError("Display-only goal silhouette is empty or implausible")
    return mask


def detail_crop(example, indices):
    """Common square detail contains every displayed arm, with 12px padding."""
    if example["environment"] != "reacher":
        return None
    images = [example["methods"]["factorized"]["goal"]]
    for mode in ("factorized", "framewise"):
        item = example["methods"][mode]
        images += [item["frames"][indices[mode]], item["frames"][-1]]
    union = np.logical_or.reduce([goal_mask(pixels, "reacher") for pixels in images])
    ys, xs = np.where(union)
    side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) + 25)
    if side > 224:
        return (0, 0, 224, 224)
    left = int(np.clip(np.floor((xs.max() + xs.min() + 1 - side) / 2), 0, 224 - side))
    top = int(np.clip(np.floor((ys.max() + ys.min() + 1 - side) / 2), 0, 224 - side))
    return left, top, left + side, top + side


def image_panel(fig, bounds, pixels, title, mask=None, color="#cad4df", crop=None):
    ax = fig.add_axes(bounds)
    ax.imshow(pixels, interpolation="nearest")
    if mask is not None:
        # Explicit pixel-center coordinates match imshow's y-down observation
        # axes; origin='upper' here would reflect the contour a second time.
        ax.contour(np.arange(224), np.arange(224), mask.astype(float), levels=[.5],
                   colors=[PURPLE], linewidths=.75, linestyles="dashed")
    ax.set_xlim(-.5, 223.5); ax.set_ylim(223.5, -.5)
    if crop is not None:
        left, top, right, bottom = crop
        ax.set_xlim(left - .5, right - .5); ax.set_ylim(bottom - .5, top - .5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color); spine.set_linewidth(1.25 if color in (GREEN, RED) else .6)
    ax.set_title(title, fontsize=7.2, pad=4, color=INK)
    return ax


def draw_example(fig, example, bottom, height, letter):
    env = example["environment"]
    name = "PushT" if env == "pusht" else "Reacher"
    time, indices = matched_frame(example)
    goal = example["methods"]["factorized"]["goal"]
    mask = goal_mask(goal, env)
    crop = detail_crop(example, indices)
    def text(x, y, value, **kw):
        fig.text(x, bottom + height * y, value, **kw)
    takeaway = {2031024: "block and pusher reach the target", 2031004: "the folded pose is reached",
                2031008: "a close-looking pose can still miss", 2031011: "the forearm matches the goal"}[example["seed"]]
    text(.025, .98, f"{letter}   {name} · {takeaway}", fontsize=9.4,
         weight="bold", va="top")
    text(.025, .91, f"Episode {example['seed']} · same goal/support · 50-action allowance", fontsize=7.2, color=MUTED, va="top")
    if crop is not None:
        # The same per-case crop is used for the goal and every method/time.
        # An overview keeps the scene location visible alongside the detail.
        overview = image_panel(fig, [.89, bottom + height * .85, .085, height * .12], goal, "")
        left, top, right, lower = crop
        overview.add_patch(Rectangle((left, top), right - left, lower - top,
                                     fill=False, edgecolor=PURPLE, lw=.8))
        text(.87, .87, f"{224 / (right-left):.2f}×\ndetail", fontsize=6.5, ha="right", va="center", color=MUTED)
    xs = [.25, .505, .76]
    for i, mode in enumerate(("factorized", "framewise")):
        item = example["methods"][mode]
        record = item["record"]
        y = .79 - i * .345
        cell_h = .236
        color = GREEN if record["success"] else RED
        text(.025, y - .035, source.LABELS[mode], fontsize=8.4, weight="bold", va="top",
             color=BLUE if mode == "factorized" else INK)
        text(.025, y - .116, "Goal reached" if record["success"] else "Goal missed", fontsize=7.8,
             weight="bold", color=color, va="top")
        text(.025, y - .186, f"t = {record['native_steps']} / 50", fontsize=7.4, color=MUTED, va="top")
        images = [goal, item["frames"][indices[mode]], item["frames"][-1]]
        titles = ["Goal image", f"Matched time: t={time}", f"Actual end: t={record['native_steps']}"]
        for j, (x, pixels, title) in enumerate(zip(xs, images, titles)):
            image_panel(fig, [x, bottom + height * (y - cell_h), .21, height * cell_h],
                        pixels, title, mask=None if j == 0 else mask,
                        color=color if j == 2 else "#cad4df", crop=crop)
        text(.25, y - cell_h - .027, source.terminal_summary(env, record), fontsize=7.1,
             color=GREEN if mode == "factorized" else MUTED, va="top",
             weight="bold" if mode == "factorized" else "normal")
    # A native-action ruler shows early success versus exhausting the allowance.
    ax = fig.add_axes([.25, bottom + height * .038, .71, height * .067])
    ax.set_xlim(-1, 51); ax.set_ylim(-.8, 1.8); ax.set_axis_off()
    for i, mode in enumerate(("factorized", "framewise")):
        row = example["methods"][mode]["record"]
        y = 1 - i
        ax.plot([0, 50], [y, y], color="#e5eaf0", lw=4, solid_capstyle="butt")
        ax.plot([0, 10], [y, y], color="#aebac7", lw=4, solid_capstyle="butt")
        ax.plot([10, row["native_steps"]], [y, y], color=GREEN if i == 0 else RED,
                lw=4, solid_capstyle="butt")
        ax.scatter([row["native_steps"]], [y], s=18, color=GREEN if i == 0 else RED, zorder=3,
                   marker="o" if i == 0 else "x")
    text(.025, .08, "Actions used", fontsize=7.1, va="center", color=MUTED)
    # Keep native-action annotations registered to the padded ruler limits.
    for action, label in ((0, "0"), (10, "10: support ends"), (50, "50: limit")):
        x = .25 + .71 * (action + 1) / 52
        text(x, .011, label, fontsize=6.5, color=MUTED, va="top",
             ha="right" if action == 50 else "center")
    return {"environment": env, "trajectory_id": example["trajectory_id"],
            "seed": example["seed"], "matched_native_time": time, "matched_frame_indices": indices,
            "goal_mask_pixels": int(mask.sum()),
            "goal_mask_sha256": __import__("hashlib").sha256(mask.tobytes()).hexdigest(),
            "display_crop_xyxy": list(crop) if crop is not None else [0, 0, 224, 224],
            "detail_magnification": 224 / (crop[2] - crop[0]) if crop is not None else 1.0,
            "records": {mode: example["methods"][mode]["record"] for mode in source.MODES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-needed", action="store_true")
    args = parser.parse_args()
    ledger = OUT / "positive_evidence.json"
    files = [OUT / f"positive_{i}.{ext}" for i in (1, 2) for ext in ("pdf", "svg", "png")]
    combined = ROOT / "artifacts/qualitative/positive_cases.pdf"
    files.append(combined)
    if args.if_needed and ledger.exists() and all(p.exists() for p in files):
        old = json.loads(ledger.read_text())
        if old.get("source_sha256") and all((ROOT / p).exists() and source.sha(ROOT / p) == h
                                            for p, h in old["source_sha256"].items()):
            print(json.dumps({"status": "reused_identical_positive_cases", "examples": len(old["selected"])}))
            return
    examples, dependencies = source.collect()
    selected = positive_cases(examples)
    if len(selected) != 4 or any(ex["methods"]["single"]["record"]["success"] for ex in selected):
        raise ValueError("Current declared four-case display needs an explicit update for changed outcomes")
    records = []
    for page in range(2):
        fig = plt.figure(figsize=(5.5, 6.4), facecolor="white")
        for panel, ex in enumerate(selected[page * 2:page * 2 + 2]):
            records.append(draw_example(fig, ex, .515 if panel == 0 else .033, .452,
                                        chr(ord('A') + page * 2 + panel)))
        fig.add_artist(Line2D([.025, .98], [.505, .505], transform=fig.transFigure,
                             lw=.7, color="#d4dfe8"))
        fig.add_artist(Line2D([.03, .08], [.012, .012], transform=fig.transFigure,
                             lw=1.2, ls="--", color=PURPLE))
        fig.text(.095, .012, "Goal silhouette from input pixels · observed development episodes · positive subset",
                 fontsize=6.4, va="center", color=MUTED)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"positive_{page + 1}.{ext}", dpi=240, facecolor="white")
        plt.close(fig)
    combined.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdfunite", str(OUT / "positive_1.pdf"), str(OUT / "positive_2.pdf"),
                    str(combined)], check=True)
    dependencies[str(Path(__file__).relative_to(ROOT))] = source.sha(__file__)
    packet = {"status": "source_validated", "selection": "All eligible ShiftWM-success/Framewise-failure tasks in the 64-task recorded development gallery; Shared context also fails in all four",
              "scope": "posthoc positive-case explanation at training seed0; not a success-rate estimate or context-causality result",
              "image_policy": "Original RGB observations. PushT full frame; every Reacher goal/time/method uses the same per-case square crop containing all displayed orange arm pixels plus 12px padding, located by an overview. Dashed display-only silhouette comes from the common goal image.",
              "time_policy": "Use latest native time recorded for both methods before ShiftWM stopping; no interpolated baseline frame",
              "mask_policy": "Display-only palette segmentation; never used for evaluation, training, physical-error annotation or method selection",
              "selected": records, "source_sha256": dependencies}
    ledger.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "source_validated", "positive_examples": len(selected)}))


if __name__ == "__main__":
    main()
