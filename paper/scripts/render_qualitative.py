#!/usr/bin/env python3
"""Source-validated observed rollouts, paired paper panels and all-task gallery.

No new inference, synthetic observations or interpolated states. Reuses the
development reporter's checkpoint, task, protocol and support validation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("qualitative_contract", ROOT / "scripts/summarize_goal_intervention.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
OUT = ROOT / "paper/generated/qualitative"
MODES = ("factorized", "framewise", "single")
LABELS = {"factorized": "ShiftWM (ours)", "framewise": "Framewise", "single": "Shared context"}
INK, MUTED, GREEN, RED, BLUE = "#243447", "#65758a", "#166534", "#a14032", "#0072b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": INK})


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frame_steps(record, frames):
    """Recover actual native times, including early success in a partial block."""
    blocks = record["executed_action_blocks"]
    if len(frames) != len(blocks) + 1:
        raise ValueError("Saved frame count differs from executed action blocks")
    for index, block in enumerate(blocks):
        values = np.asarray(block)
        if (values.ndim != 1 or values.dtype.kind not in "fiu"
                or not 0 < values.size <= 10 or values.size % 2
                or not np.isfinite(values).all()):
            raise ValueError("Malformed executed native actions")
        if values.size != 10 and (index != len(blocks) - 1 or record["success"] != 1):
            raise ValueError("A partial action block requires terminal success")
    if (not isinstance(record["native_steps"], (int, np.integer))
            or isinstance(record["native_steps"], (bool, np.bool_))):
        raise ValueError("Native action time must be an integer")
    if record["num_replans"] != len(blocks):
        raise ValueError("Saved action blocks differ from the number of replans")
    initial = record["native_steps"] - sum(len(block) // 2 for block in blocks)
    if record["policy_eligible"] and initial != 10:
        raise ValueError("Eligible trajectory does not start after ten support actions")
    if not record["policy_eligible"] and (blocks or not 0 < initial <= 10):
        raise ValueError("Support-only success contains planned actions")
    steps = [initial]
    for block in blocks:
        steps.append(steps[-1] + len(block) // 2)
    if steps[-1] != record["native_steps"] or steps[-1] > 50:
        raise ValueError("Frame timeline violates native budget")
    return steps


def category(records):
    ours, base = records["factorized"], records["framewise"]
    if not ours["policy_eligible"]:
        return "support_success"
    return {(1, 0): "ours_only", (0, 1): "baseline_only",
            (0, 0): "neither", (1, 1): "both"}[(ours["success"], base["success"])]


CATEGORY_LABEL = {"ours_only": "ShiftWM succeeds; Framewise fails",
                  "baseline_only": "Framewise succeeds; ShiftWM fails",
                  "neither": "Neither reaches the scored goal",
                  "both": "Both reach the scored goal",
                  "support_success": "Success during shared support"}


def validated_dependencies(root, environment, expected):
    """Pin every live input consulted by the reused reporting validators.

    The expectation builder already hashed each large checkpoint. Reuse those
    verified digests rather than scanning model weights a second time.
    """
    root = Path(root).resolve()
    data_name = "pusht_relative" if environment == "pusht" else "reacher"
    sources = {f"data/world/{data_name}/manifest.json": expected["data_manifest_sha256"]}
    for source in expected["sources"].values():
        package = Path(source["path"])
        stats = Path(source["config"]["provenance"]["action_stats"])
        if not stats.is_absolute():
            stats = root / stats
        files = {"model_sha256": package / "model.pt", "config_sha256": package / "config.json",
                 "run_config_sha256": package.parent / "run_config.json",
                 "training_summary_sha256": package.parent / "training_summary.json",
                 "metrics_sha256": package.parent / "metrics.jsonl", "action_stats_sha256": stats}
        for key, path in files.items():
            path = path.resolve()
            name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            digest = source["hashes"][key]
            if name in sources and sources[name] != digest:
                raise ValueError("Shared qualitative dependency has inconsistent source hashes")
            sources[name] = digest
    for name in ("paper/scripts/render_qualitative.py", "scripts/summarize_goal_intervention.py",
                 "scripts/evaluate_goal_calibration_intervention.py", "src/shiftwm/evaluate.py",
                 "src/shiftwm/generate.py", "src/shiftwm/data.py", "src/shiftwm/model.py",
                 "src/shiftwm/checkpoint.py", "src/shiftwm/upstream.py"):
        sources[name] = sha(root / name)
    return sources


def select_examples(examples):
    """First lexicographic task in every fixed, explicitly declared stratum."""
    selected = []
    for env in ("pusht", "reacher"):
        strata = ("ours_only", "baseline_only", "neither", "support_success" if env == "pusht" else "both")
        for stratum in strata:
            candidates = [ex for ex in examples if ex["environment"] == env and ex["category"] == stratum]
            if not candidates:
                raise ValueError(f"No observed example in declared stratum {env}/{stratum}")
            selected.append(min(candidates, key=lambda ex: (ex["trajectory_id"], ex["observation_id"])))
    return selected


def collect(root=ROOT):
    examples, sources = [], {}
    for env in ("pusht", "reacher"):
        expected = contract.build_expectations(root, env)
        sources.update(validated_dependencies(root, env, expected))
        maps = {}
        paths = {}
        for mode in MODES:
            path = root / "results/development_official_budget" / f"{env}_{mode}_s0/planning_development.json"
            record = json.loads(path.read_text())
            if record.get("status") != "complete":
                raise ValueError("Qualitative source must be a completed evaluation")
            contract.validate_regular(record, env, mode, expected)
            maps[mode] = contract.validate_records(record, expected)
            paths[mode] = path
            sources[str(path.relative_to(root))] = sha(path)
        for mode in MODES[1:]:
            contract.validate_shared_support(maps["factorized"], maps[mode], env)
        for key in sorted(expected["keys"]):
            records = {mode: maps[mode][key] for mode in MODES}
            item = {"environment": env, "trajectory_id": key[0], "observation_id": key[1],
                    "seed": records["factorized"]["seed"], "category": category(records),
                    "methods": {}, "id": f"{env}-{key[0]}-o{key[1]}"}
            reference_goal = reference_start = None
            for mode in MODES:
                video = paths[mode].parent / "videos" / f"{key[0]}-o{key[1]}.npz"
                with np.load(video, allow_pickle=False) as saved:
                    frames, goal = saved["frames"].copy(), saved["goal_image"].copy()
                if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] < 1
                        or frames.shape[1:] != (224, 224, 3)
                        or goal.dtype != np.uint8 or goal.shape != (224, 224, 3)):
                    raise ValueError("Unexpected saved RGB dimensions or dtype")
                if reference_goal is None:
                    reference_goal, reference_start = goal, frames[0]
                elif not np.array_equal(goal, reference_goal) or not np.array_equal(frames[0], reference_start):
                    raise ValueError("Paired methods have different goal/start pixels")
                times = frame_steps(records[mode], frames)
                video_name = str(video.relative_to(root))
                sources[video_name] = sha(video)
                item["methods"][mode] = {"record": records[mode], "source": str(paths[mode].relative_to(root)),
                                          "video": video_name, "frame_steps": times,
                                          "frames": frames, "goal": goal}
            examples.append(item)
    return examples, sources


def terminal_summary(env, row):
    if env == "pusht":
        return (f"Block {row['block_translation_error_px']:.1f} px · "
                f"angle {row['block_angle_error_rad']:.3f} rad · "
                f"pusher {row['agent_position_error_px']:.1f} px")
    return f"Joint-angle distance {row['final_distance']:.3f} rad"


def draw_case(fig, example, bottom=0, height=1, methods=("factorized", "framewise"), full=False):
    """Identical full-frame images and scales, labels outside observations."""
    env = example["environment"]
    title = "PushT" if env == "pusht" else "Reacher"
    def text(x, y, label, **kwargs):
        return fig.text(x, bottom + y * height, label, **kwargs)
    text(.025, .969, title + "  |  " + CATEGORY_LABEL[example["category"]],
         fontsize=9, fontweight="bold", va="top")
    text(.025, .897, f"Development · episode {example['seed']} · training seed 0 · 50-action limit",
         fontsize=7.0, color=MUTED, va="top")
    n = len(methods)
    row_h = .72 / n
    # Five visual columns in gallery: whole recorded timeline is a separate view.
    xs = [.225, .42, .615, .81]
    for index, mode in enumerate(methods):
        item = example["methods"][mode]
        row, frames, steps = item["record"], item["frames"], item["frame_steps"]
        y = .79 - index * row_h
        image_h = row_h * .69
        color = GREEN if row["success"] else RED
        text(.025, y - .027, LABELS[mode], fontsize=8.0, fontweight="bold", va="top",
             color=BLUE if mode == "factorized" else INK)
        status = "Support success" if not row["policy_eligible"] else ("Success" if row["success"] else "Failure")
        text(.025, y - .112, status, fontsize=7.8, fontweight="bold", color=color, va="top")
        text(.025, y - .19, f"{row['native_steps']} native actions", fontsize=7.0, va="top", color=MUTED)
        first = min(1, len(frames) - 1)
        images = [item["goal"], frames[0], frames[first], frames[-1]]
        labels = ["Goal image", f"Support: t={steps[0]}",
                  f"First move: t={steps[first]}" if first else "No planned move",
                  f"Final: t={steps[-1]}"]
        for j, (x, pixels, label) in enumerate(zip(xs, images, labels)):
            ax = fig.add_axes([x, bottom + (y - image_h) * height, .17, image_h * height])
            ax.imshow(pixels, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.6 if j == 3 else .65)
                spine.set_edgecolor(color if j == 3 else "#ccd4dd")
            ax.set_title(label, fontsize=7.0, pad=3, color=INK)
        text(.225, y - image_h - .024, terminal_summary(env, row), fontsize=7.0, color=MUTED, va="top")
    text(.025, .025, "Observed simulator frames  •  shared start and goal  •  no interpolated or predicted images",
         fontsize=6.5, color=MUTED, va="bottom")


def gallery_page(examples):
    cards = []
    # Lead with every positive discordant case, while retaining all other tasks
    # and explicit filtering. The presentation order is outcome-selected.
    ordered = sorted(examples, key=lambda ex: (ex["category"] != "ours_only",
                                               ex["environment"], ex["trajectory_id"], ex["observation_id"]))
    for ex in ordered:
        rows = []
        for mode in MODES:
            item = ex["methods"][mode]
            row = item["record"]
            # Actual frame index maps to its recorded executed-action time.
            src = f"assets/{ex['id']}-{mode}.png"
            status = "support success" if not row["policy_eligible"] else "success" if row["success"] else "failure"
            rows.append(f'<h3>{LABELS[mode]} <span class="{"success" if row["success"] else "failure"}">{status}</span></h3>'
                        f'<p>{html.escape(terminal_summary(ex["environment"], row))}; '
                        f'{row["native_steps"]} native actions; {row["num_replans"]} replans.</p>'
                        f'<img loading="lazy" src="{src}" alt="All saved simulator frames with exact native action times for {LABELS[mode]}"/>')
        expanded = " open" if not cards else ""
        cards.append(f'<details{expanded} data-env="{ex["environment"]}" data-category="{ex["category"]}" data-seed="{ex["seed"]}">'
                     f'<summary>{ex["environment"].title()} · {ex["seed"]} · {CATEGORY_LABEL[ex["category"]]}</summary>'
                     + "".join(rows) + '</details>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ShiftWM: matched qualitative rollouts</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 22px;color:#243447;background:#f7f9fb}
h1{font-size:32px;line-height:1.15}p{max-width:90ch}h3{margin-bottom:3px}h3+p{margin-top:0;font-size:14px}
details{background:white;border:1px solid #dce3eb;border-radius:8px;padding:16px;margin:12px 0}summary{cursor:pointer;font-weight:650}
img{max-width:100%;height:auto;background:white}select,input{font:inherit;padding:8px;margin:4px 16px 4px 0;border:1px solid #bcc8d5;border-radius:4px}
.success{color:#166534}.failure{color:#a14032}h3 span{font-size:14px;margin-left:15px}aside{border-left:4px solid #0072b2;padding:3px 18px;background:#edf4f8}
[hidden]{display:none}a{color:#006898}footer{font-size:13px;color:#65758a;margin-top:28px}
</style><h1>Where ShiftWM reaches the goal</h1>
<p><strong>Positive cases first:</strong> four tasks where ShiftWM succeeds and both learned controls fail.
These selected examples explain observed behavior; all 64 tasks remain available below, including counterexamples.</p>
<p><a href="positive_cases.pdf">Annotated positive-case figures</a> · <a href="qualitative_examples.pdf">Balanced examples</a></p>
<p>All 64 recorded development tasks, three methods, training seed 0. Expand an episode to compare every saved frame.
Times count native environment actions, including the common support. An early success stops the episode; blank space is not a missing prediction.</p>
<aside><p><strong>Reading PushT:</strong> the target is the gray block and blue pusher configuration in the goal image.
The green T is a persistent marker from the upstream renderer, not this episode’s scored goal.
Success requires the pusher and block pose to satisfy the original tolerances; a well-placed block alone can still fail.</p>
<p><strong>Reading Reacher:</strong> the goal image specifies the desired arm pose. Scores use the upstream joint-angle criterion.
All displayed images are actual simulator observations. The world model predicts latents; these are not decoded predictions.</p></aside>
<p>Examples illustrate behavior and do not estimate a general win rate. The balanced paper panels use the first episode in each declared outcome stratum;
the positive panels include all four ShiftWM-only successes. This gallery includes every task, including support-only successes and failures.</p>
<label>Environment <select id="env"><option value="">Both</option><option value="pusht">PushT</option><option value="reacher">Reacher</option></select></label>
<label>Outcome versus Framewise <select id="category"><option value="">All outcomes</option><option value="ours_only">ShiftWM only</option>
<option value="baseline_only">Framewise only</option><option value="both">Both succeed</option><option value="neither">Neither succeeds</option>
<option value="support_success">Support-only success</option></select></label>
<label>Episode <input id="seed" placeholder="e.g. 2031000" type="search"></label><p id="count" aria-live="polite"></p>
''' + "".join(cards) + '''<footer>Local evidence artifact. <a href="evidence.json">Source hashes, records, selection rule and exact frame times</a>.
Same full-frame display scale; no scene retouching or generative imagery. Development protocol: 300 candidates, 30 CEM iterations, 30 elites,
horizon 5, 50 native actions including up to 10 support actions. Methods share initial support, goals and planner seed.</footer>
<script>const selectors=['env','category','seed'].map(id=>document.getElementById(id));function filter(){let n=0;
document.querySelectorAll('details').forEach(d=>{let ok=(!selectors[0].value||d.dataset.env===selectors[0].value)&&
(!selectors[1].value||d.dataset.category===selectors[1].value)&&d.dataset.seed.includes(selectors[2].value);d.hidden=!ok;n+=ok});
document.getElementById('count').textContent=n+' of 64 episodes shown';}selectors.forEach(s=>s.addEventListener('input',filter));filter();</script></html>'''


def timeline(example, mode, path):
    item = example["methods"][mode]
    frames, steps = item["frames"], item["frame_steps"]
    # Every sequence uses the same cell size. Never stretch short successful runs.
    fig, axes = plt.subplots(1, 10, figsize=(12, 1.45))
    fig.subplots_adjust(left=.007, right=.993, bottom=.03, top=.82, wspace=.06)
    images = [item["goal"], *frames]
    labels = ["Goal", *(f"t = {t}" for t in steps)]
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i < len(images):
            ax.imshow(images[i], interpolation="nearest")
            ax.set_title(labels[i], fontsize=9, color=INK)
        elif i == len(images):
            ax.text(.5, .5, "Episode\nended", ha="center", va="center", fontsize=9, color=MUTED)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def clean_example(example):
    return {**{k: v for k, v in example.items() if k != "methods"},
            "methods": {mode: {k: v for k, v in item.items() if k not in {"frames", "goal"}}
                        for mode, item in example["methods"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-needed", action="store_true", help="Reuse only if every pinned source and output exists unchanged")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gallery = ROOT / "artifacts/qualitative"
    (gallery / "assets").mkdir(parents=True, exist_ok=True)
    ledger = OUT / "evidence.json"
    if args.if_needed and ledger.exists():
        prior = json.loads(ledger.read_text())
        files = [OUT / f"{env}_{kind}.{ext}" for env in ("pusht", "reacher")
                 for kind in ("contrasts", "controls") for ext in ("pdf", "svg", "png")]
        files += [gallery / "index.html", gallery / "evidence.json"]
        files += [gallery / "assets" / f"{ex['id']}-{mode}.png"
                  for ex in prior.get("examples", []) for mode in MODES]
        if (len(prior.get("examples", [])) == 64 and prior.get("source_sha256")
                and all(path.is_file() for path in files)
                and all((ROOT / path).is_file() and sha(ROOT / path) == digest
                        for path, digest in prior["source_sha256"].items())):
            print(json.dumps({"status": "reused_identical_qualitative_sources", "examples": 64}))
            return
    examples, sources = collect()
    selected = select_examples(examples)
    for i in range(0, len(selected), 2):
        group = selected[i:i + 2]
        name = f"{group[0]['environment']}_{'contrasts' if i % 4 == 0 else 'controls'}"
        fig = plt.figure(figsize=(5.5, 5.6), facecolor="white")
        for panel, ex in enumerate(group):
            draw_case(fig, ex, bottom=.5 * (1 - panel), height=.49)
        fig.add_artist(plt.Line2D([.025, .98], [.505, .505], transform=fig.transFigure,
                                 lw=.7, color="#d6dfe8"))
        for extension in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"{name}.{extension}", dpi=240, facecolor="white")
        plt.close(fig)
    for ex in examples:
        for mode in MODES:
            timeline(ex, mode, gallery / "assets" / f"{ex['id']}-{mode}.png")
    sources[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    sources["scripts/summarize_goal_intervention.py"] = sha(ROOT / "scripts/summarize_goal_intervention.py")
    sources["src/shiftwm/evaluate.py"] = sha(ROOT / "src/shiftwm/evaluate.py")
    sources["src/shiftwm/generate.py"] = sha(ROOT / "src/shiftwm/generate.py")
    evidence = {"status": "source_validated", "scope": "development_only_fixed_training_seed0",
                "selection_rule": "First lexicographic episode per environment/outcome stratum versus Framewise; gallery includes all 64 tasks",
                "paper_strata": {"pusht": ["ours_only", "baseline_only", "neither", "support_success"],
                                  "reacher": ["ours_only", "baseline_only", "neither", "both"]},
                "pixel_policy": "Observed uint8 RGB; no crop, retouch, interpolation between states or generative content",
                "time_policy": "Post-support first frame plus one frame per executed action block; actual partial block lengths",
                "pusht_marker_notice": "Green T is the upstream renderer's persistent goal_pose marker; task scoring uses episode goal_state",
                "selected": [clean_example(ex) for ex in selected],
                "examples": [clean_example(ex) for ex in examples], "source_sha256": sources}
    for path in (OUT / "evidence.json", gallery / "evidence.json"):
        path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    (gallery / "index.html").write_text(gallery_page(examples))
    print(json.dumps({"examples": len(examples), "matched_methods": len(MODES),
                      "paper_examples": len(selected), "gallery": str(gallery / "index.html")}))


if __name__ == "__main__":
    main()
