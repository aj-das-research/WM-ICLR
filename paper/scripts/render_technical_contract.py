#!/usr/bin/env python3
"""Editable code-derived inference comparison; no quantitative/causal claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/generated/qualitative"
ARCHIVE = ROOT / "results/development_official_budget/reacher_factorized_s0/videos/development-s2031004-d1-o1.npz"
INK, MUTED = "#243447", "#65758a"
OBS, DYN, GOAL = "#0072b2", "#b75b16", "#94639c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in
            (ARCHIVE, ROOT / "src/shiftwm/model.py", ROOT / "src/shiftwm/evaluate.py", Path(__file__))}


def cache_current():
    ledger = OUT / "technical_contract.json"
    if not ledger.is_file():
        return False
    try:
        evidence = json.loads(ledger.read_text())
        if evidence.get("source_sha256") != source_hashes():
            return False
        return all((OUT / filename).is_file() and digest(OUT / filename) == value
                   for filename, value in evidence["output_sha256"].items()) and set(evidence["output_sha256"]) == {
                       "technical_contract.pdf", "technical_contract.svg", "technical_contract.png"}
    except (KeyError, OSError, ValueError, TypeError):
        return False


def render():
    with np.load(ARCHIVE, allow_pickle=False) as data:
        support, goal = data["frames"][0].copy(), data["goal_image"].copy()
    assert support.shape == goal.shape == (224, 224, 3)
    assert support.dtype == goal.dtype == np.uint8
    fig = plt.figure(figsize=(5.5, 2.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 5.5), ylim=(0, 2.4))
    ax.set_axis_off()
    texts, edges = [], []

    def text(x, y, label, size=8, color=INK, ha="center", **kwargs):
        texts.append(label)
        return ax.text(x, y, label, fontsize=size, color=color, ha=ha, va="center", **kwargs)

    def box(x, y, w, h, label=None, color=INK, fill="white", size=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.045",
                          linewidth=.85, edgecolor=color, facecolor=fill, zorder=4)
        ax.add_patch(p)
        if label is not None:
            text(x + w / 2, y + h / 2, label, size=size, color=color, zorder=5)
        return p

    def arrow(name, points, color=INK, head=True, lw=.8):
        # One continuous path per edge; labels never obscure a connector.
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        edge = FancyArrowPatch(path=path, arrowstyle="-|>" if head else "-",
                               mutation_scale=5.4, linewidth=lw, color=color,
                               joinstyle="round", capstyle="round", zorder=2)
        ax.add_patch(edge)
        edges.append({"id": name, "points_inches": points, "color": color, "arrow": head})
        return edge

    def tensors(x, y, n=3, color=OBS, w=.043, h=.14):
        for i in range(n):
            ax.add_patch(Rectangle((x + i * .065, y), w, h, edgecolor=color,
                                   facecolor=color, linewidth=.6, alpha=.75, zorder=5))

    # A real last-support view plus real goal illustrate the shared input objects.
    # They are never presented as all three support frames or predicted images.
    for x, pixels, label in ((.07, support, "Last obs."), (.52, goal, "Goal")):
        image_axes = fig.add_axes([x / 5.5, 1.94 / 2.4, .35 / 5.5, .35 / 2.4])
        image_axes.imshow(pixels, interpolation="nearest")
        image_axes.set_xticks([]); image_axes.set_yticks([])
        for spine in image_axes.spines.values():
            spine.set_color("#aab7c4"); spine.set_linewidth(.6)
        text(x if label == "Last obs." else x + .175, 1.875, label, 6.4, MUTED,
             ha="left" if label == "Last obs." else "center")
    arrow("RGB_to_fixed_encoder", [(.9, 2.115), (1.07, 2.115)])
    box(1.07, 1.965, .51, .30, "$E_0$", size=10)
    text(1.325, 2.335, "Frozen", 7, MUTED)
    arrow("encoder_to_feature_objects", [(1.60, 2.115), (1.84, 2.115)])
    tensors(1.90, 2.045)
    tensors(2.23, 2.045, n=1, color=GOAL)
    text(2.08, 2.28, r"$u_{0:2},\ u_g$", 9)
    text(2.08, 1.87, "$H=3$", 7.7, MUTED)
    tensors(2.64, 2.045, n=2, color=DYN, h=.14)
    text(2.72, 2.28, "$a_{0:1}$", 9, DYN)
    text(2.72, 1.87, "executed", 7, MUTED)
    text(3.43, 2.12, "Same inputs", 9, weight="bold")
    text(5.40, 2.13, "Temporal predictor\nsame CEM · separate weights", 7.0, ha="right")
    ax.plot([.06, 5.42], [1.79, 1.79], color="#d7e0e7", lw=.65)

    # Gray grouping means common architecture/protocol, not shared learned weights.
    ax.add_patch(FancyBboxPatch((4.29, .53), 1.13, 1.12,
                 boxstyle="round,pad=0.025,rounding_size=0.06", facecolor="#f0f3f6",
                 edgecolor="#d5dfe7", linewidth=.7, zorder=0))
    for y in (1.30, .70):
        box(4.40, y, .35, .29, r"$P_\theta$", fill="#f8fafc", size=8.4)
        box(5.00, y, .33, .29, "CEM", fill="#f8fafc", size=6.9)
        arrow("predictor_to_search_" + str(y), [(4.77, y + .145), (4.98, y + .145)])

    # Framewise still retains the trained temporal predictor and action embedding.
    text(.08, 1.60, "Framewise", 8.6, ha="left", weight="bold")
    tensors(.27, 1.33); tensors(.53, 1.33, n=1, color=GOAL)
    text(.42, 1.25, "$u_{0:2},u_g$", 7.5)
    arrow("framewise_input", [(.67, 1.43), (.96, 1.43)])
    box(.98, 1.285, .79, .29, r"$\mathrm{MLP}\!\circ\!\mathrm{LN}$", size=7.5)
    arrow("framewise_MLP_to_residual", [(1.80, 1.43), (2.07, 1.43)])
    ax.add_patch(Circle((2.17, 1.43), .085, facecolor="white", edgecolor=INK, lw=.8, zorder=4))
    text(2.17, 1.43, "+", 10, zorder=5)
    arrow("framewise_identity_skip", [(.86, 1.43), (.86, 1.69), (2.17, 1.69), (2.17, 1.53)])
    arrow("framewise_corrected_history", [(2.27, 1.43), (4.38, 1.43)], color=OBS)
    text(2.78, 1.52, "$z_{0:2}$", 8, OBS)
    arrow("framewise_corrected_goal", [(2.36, 1.43), (2.36, 1.72), (5.165, 1.72), (5.165, 1.61)], color=GOAL)
    text(3.53, 1.64, "$z_g$", 7.5, GOAL)
    text(3.72, 1.155, "$E_a(a)$", 8.3, DYN)
    arrow("framewise_candidate_action_embeddings", [(3.98, 1.16), (4.57, 1.16), (4.57, 1.28)], color=DYN)

    # Proposed path: observation context does not see the goal; dynamics context
    # uses only calibrated support and two past executed action blocks.
    text(.08, 1.06, "ShiftWM\n(ours)", 7.8, ha="left", color=OBS, weight="bold", linespacing=1.05)
    tensors(.29, .70)
    text(.39, .64, "$u_{0:2}$", 7.5, OBS)
    box(.88, .685, .54, .32, "$C_o$", color=OBS, fill="#edf6fb", size=9)
    text(1.15, 1.105, "mean / var", 7, OBS)
    arrow("history_statistics_to_observation_context", [(.59, .84), (.86, .84)], color=OBS)
    box(1.77, .685, .55, .32, r"FiLM$_o$", color=OBS, fill="#edf6fb", size=7.8)
    arrow("observation_context_conditions_calibration", [(1.44, .84), (1.75, .84)], color=OBS)
    text(1.60, .96, "$c_o$", 8, OBS)
    tensors(.50, .42); tensors(.76, .42, n=1, color=GOAL)
    text(.67, .33, "$u_{0:2},u_g$", 7.5)
    arrow("history_goal_enter_same_calibration", [(.91, .485), (1.94, .485), (1.94, .665)], color=OBS)
    arrow("corrected_history_to_dynamics_context", [(2.34, .84), (2.73, .84)], color=OBS)
    text(2.54, .75, "$z_{0:2}$", 7.7, OBS)
    arrow("corrected_history_to_temporal_predictor", [(2.50, .84), (2.50, 1.095), (4.57, 1.095), (4.57, 1.01)], color=OBS)
    box(2.75, .685, .55, .32, color=DYN, fill="#fff3e9")
    text(3.025, .935, r"$[z,\Delta z]$", 6.4, DYN, zorder=5)
    text(3.025, .790, "GRU", 7.6, DYN, zorder=5)
    text(3.02, .455, "$a_{0:1}$", 8, DYN)
    arrow("executed_past_into_dynamics_context", [(3.025, .54), (3.025, .665)], color=DYN)
    box(3.66, .685, .49, .32, r"FiLM$_d$", color=DYN, fill="#fff3e9", size=7.5)
    arrow("dynamics_context_conditions_actions", [(3.32, .84), (3.64, .84)], color=DYN)
    text(3.48, .96, "$c_d$", 8, DYN)
    text(3.90, .455, "$E_a(a)$", 8.2, DYN)
    arrow("candidate_embeddings_enter_FiLM", [(3.905, .54), (3.905, .665)], color=DYN)
    arrow("modulated_actions_to_predictor", [(4.17, .84), (4.38, .84)], color=DYN)
    arrow("corrected_goal_to_search", [(2.17, .665), (2.17, .26), (5.165, .26), (5.165, .68)], color=GOAL)
    text(2.35, .345, "$z_g$", 7.8, GOAL)
    text(2.76, .095, "Weights fixed · contexts fixed in CEM; recompute after each real block", 7.1, MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / ("technical_contract." + extension), dpi=300, facecolor="white")
    plt.close(fig)
    evidence = {"kind": "code_derived_inference_schematic", "size_inches": [5.5, 2.4],
                "quantitative_claim": None, "causal_claim": None,
                "source_sha256": source_hashes(),
                "output_sha256": {"technical_contract." + ext: digest(OUT / ("technical_contract." + ext))
                                  for ext in ("pdf", "svg", "png")},
                "input_thumbnails": {"trajectory_id": "development-s2031004-d1", "observation_id": 1,
                    "support": "saved frame0 at native time10; only last support observation shown",
                    "goal": "saved available shifted goal_image; no canonical target shown",
                    "transform": "full original uint8 RGB, no crop or retouch"},
                "semantics": {"common_region": "same predictor architecture/CEM procedure, separately learned method weights",
                    "Framewise": "per-image u+MLP(LN(u)); temporal/action-conditioned predictor retained",
                    "ShiftWM": "support statistics infer c_o; A_o corrects history and goal; GRU of corrected transitions and executed past infers c_d; c_d modulates action embeddings",
                    "goal": "enters calibration and CEM cost, never either context network",
                    "future_actions": "enter predictor/action FiLM, never context inference",
                    "action_embedding": "E_a(a) includes normalized executed past and candidate future actions in predictor history windows",
                    "context_updates": "fixed inside search, inferred again from new observed support after actual execution",
                    "omissions": "action normalization, internal GRU difference construction, CEM cost internals and offline training losses"},
                "labels": texts, "connections": edges}
    (OUT / "technical_contract.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": "rendered", "output": str(OUT / "technical_contract.pdf"),
                      "size_inches": [5.5, 2.4], "connections": len(edges)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-needed", action="store_true", help="Skip only when all source and output hashes match.")
    args = parser.parse_args()
    if args.if_needed and cache_current():
        print(json.dumps({"status": "unchanged", "output": str(OUT / "technical_contract.pdf")}))
    else:
        render()
