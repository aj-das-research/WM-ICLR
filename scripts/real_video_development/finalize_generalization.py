#!/usr/bin/env python3
"""Publish the complete 36-run validation study only after independent gates.

This is a new postprocessor. It never modifies the registered runner, protocol,
configurations or model files, and never opens original/fresh test payloads.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/real_video")]
import train
import evaluate
from shiftwm.real_video.data import RealVideoDataset

REGISTRY = ROOT / "configs/real_video_development/generalization_v1/registration.json"
RUNNER = ROOT / "scripts/real_video_development/generalization_campaign.py"
GENERATED = ROOT / "paper/generated/real_video/generalization"
SECTION = ROOT / "paper/sections/generalization_development.tex"
REPORT = ROOT / "reports/real_droid_generalization_finalization.json"
MODES = ("framewise", "constant_dynamics", "factorized", "action_free")
ARMS = ("slow", "decay", "compact")
NAMES = {"framewise": "Framewise", "constant_dynamics": "Constant dynamics",
         "factorized": r"\textbf{ShiftWM (ours)}", "action_free": "Action-free"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_registration(registration, root=ROOT):
    rows = registration["runs"]
    expected = {(a, m, s) for a in ARMS for m in MODES for s in range(3)}
    if (len(rows) != 36 or {(x["arm"], x["mode"], x["seed"]) for x in rows} != expected
            or len({x["name"] for x in rows}) != 36):
        raise ValueError("Campaign must contain exactly 36 unique matched runs")
    for relative, expected_hash in registration["dependencies"].items():
        if sha(root / relative) != expected_hash:
            raise ValueError("Registered dependency changed: " + relative)
    configurations = {}
    for row in rows:
        if sha(root / row["config"]) != row["config_sha256"]:
            raise ValueError("Registered configuration changed")
        configurations[row["name"]] = read(root / row["config"])
        config = configurations[row["name"]]
        if config.get("mode") != row["mode"] or config.get("seed") != row["seed"] or config.get("epochs") != 30:
            raise ValueError("Registered run configuration disagrees with its arm identity")
    for arm in ARMS:
        for seed in range(3):
            group = [configurations[r["name"]] for r in rows if (r["arm"], r["seed"]) == (arm, seed)]
            shared = [{k: v for k, v in c.items() if k not in ("mode", "output_dir")} for c in group]
            if any(c != shared[0] for c in shared):
                raise ValueError("An arm/seed comparison has unmatched training settings")
    return configurations


def require_complete_files(registration, configurations, root=ROOT):
    missing = []
    for row in registration["runs"]:
        directory = root / configurations[row["name"]]["output_dir"]
        for name in ("training_summary.json", "training_config.json", "development_receipt.json",
                     "best/model.pt", "last/model.pt", "validation_h5.json", "validation_h10.json"):
            if not (directory / name).is_file():
                missing.append(f"{row['name']}/{name}")
    if missing:
        raise ValueError(f"Incomplete campaign: {len(missing)} required files absent; first: {missing[0]}")


def check_evaluation(document, *, row, horizon, registry_hash, checkpoint_hash, expected_population):
    if (document.get("scope") != "validation development only" or document.get("arm") != row["arm"]
            or document.get("mode") != row["mode"] or document.get("seed") != row["seed"]
            or document.get("horizon") != horizon or document.get("registration_sha256") != registry_hash
            or document.get("checkpoint_sha256") != checkpoint_hash):
        raise ValueError("Validation result identity mismatch")
    result = document["result"]
    episodes = sorted(result["episodes"], key=lambda x: x["episode_id"])
    if [x["episode_id"] for x in episodes] != sorted(expected_population):
        raise ValueError("Validation population missing, duplicated, or includes another split")
    methods = {"model", "persistence", "constant_velocity", "reversed_future_actions"}
    metric_keys = None
    for episode in episodes:
        expected = expected_population[episode["episode_id"]]
        if (episode["session_id"] != expected["session_id"] or episode["window_starts"] != expected["starts"]
                or episode["windows"] != len(expected["starts"]) or set(episode["errors"]) != methods):
            raise ValueError("Validation session/window population mismatch")
        for errors in episode["errors"].values():
            if metric_keys is None:
                metric_keys = set(errors)
            if set(errors) != metric_keys or any(not math.isfinite(v) or v < 0 for v in errors.values()):
                raise ValueError("Invalid or unmatched evaluation metrics")
    if (result["episode_count"] != len(episodes)
            or result["session_count"] != len({x["session_id"] for x in episodes})
            or result["window_count"] != sum(x["windows"] for x in episodes)):
        raise ValueError("Incorrect evaluation population counts")
    if set(result["summary"]) != methods:
        raise ValueError("Incorrect evaluation summary methods")
    for method in methods:
        if set(result["summary"][method]) != metric_keys:
            raise ValueError("Incorrect evaluation summary metrics")
        for key in metric_keys:
            recomputed = math.fsum(x["errors"][method][key] for x in episodes) / len(episodes)
            if not math.isclose(recomputed, result["summary"][method][key], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("Episode-derived mean differs from reported summary")
    return episodes


def collect():
    registration = read(REGISTRY)
    configs = check_registration(registration)
    require_complete_files(registration, configs)
    registry_hash = sha(REGISTRY)
    manifest_path = ROOT / "data/features/droid_selected_v1/manifest.json"
    manifest = read(manifest_path)
    population = {}
    for horizon in (5, 10):
        population[horizon] = {}
        for episode in manifest["episodes"]:
            if episode["split"] != "val":
                continue
            count = episode["cameras"]["exterior_image_1_left"]["frames"]
            starts = list(range(0, count-(3+horizon)+1, 5))
            if starts:
                population[horizon][episode["episode_id"]] = {"session_id": episode["session_id"], "starts": starts}
    validation = RealVideoDataset(ROOT / "data/features/droid_selected_v1", "val", horizon=10, stride=5, verify=True)
    sample = validation[0]
    support = sample["features"][None, :3].float()
    actions = sample["actions"][None].float()
    development_sources = {str((ROOT / "data/features/droid_selected_v1" / x["cameras"]["exterior_image_1_left"]["file"]).relative_to(ROOT)):
                           x["cameras"]["exterior_image_1_left"]["sha256"] for x in manifest["episodes"] if x["split"] in ("train", "val")}
    sources = {**registration["dependencies"], str(REGISTRY.relative_to(ROOT)): registry_hash,
               str(Path(__file__).relative_to(ROOT)): sha(__file__), **development_sources}
    records = []
    for row in registration["runs"]:
        config = configs[row["name"]]
        directory = ROOT / config["output_dir"]
        stored_config = read(directory / "training_config.json")
        if train.scientific_config(stored_config) != train.scientific_config(config):
            raise ValueError("Trained scientific config differs from its registration")
        summary = train.validate_completed(directory)
        receipt = read(directory / "development_receipt.json")
        if (receipt.get("status") != "completed" or receipt.get("training") != summary
                or receipt.get("offline_cpu_reload_max_error") != 0.0
                or not math.isfinite(receipt.get("elapsed_seconds", -1)) or receipt.get("elapsed_seconds", -1) <= 0):
            raise ValueError("Invalid full-run completion or CPU reload receipt")
        checkpoint = directory / "best"
        checkpoint_hash = sha(checkpoint / "model.pt")
        model, state = train.load_package(checkpoint, "cpu")
        second, _ = train.load_package(checkpoint, "cpu")
        with torch.inference_mode():
            predicted = model.predict(support, actions[:, :2], actions[:, 2:])
            reloaded = second.predict(support, actions[:, :2], actions[:, 2:])
        if predicted.shape != (1, 10, 1536) or not torch.isfinite(predicted).all() or not torch.equal(predicted, reloaded):
            raise ValueError("Independent CPU weights-only reload parity failed")
        if model.config.mode != row["mode"] or state["epoch"] != summary["best_epoch"]:
            raise ValueError("Loaded checkpoint differs from recorded selection")
        record = {**row, "training": summary, "elapsed_seconds": receipt["elapsed_seconds"],
                  "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash,
                  "checkpoint_bytes": (checkpoint / "model.pt").stat().st_size,
                  "cpu_prediction_sha256": hashlib.sha256(predicted.numpy().tobytes()).hexdigest(), "evaluation": {}}
        for horizon in (5, 10):
            path = directory / f"validation_h{horizon}.json"
            if receipt.get("evaluations", {}).get(str(horizon)) != str(path.relative_to(ROOT)):
                raise ValueError("Completion receipt points to another evaluation")
            document = read(path)
            episodes = check_evaluation(document, row=row, horizon=horizon, registry_hash=registry_hash,
                                       checkpoint_hash=checkpoint_hash, expected_population=population[horizon])
            record["evaluation"][str(horizon)] = {"episodes": episodes, "summary": document["result"]["summary"]}
        for relative in ("training_config.json", "training_summary.json", "development_receipt.json", "metrics.jsonl",
                         "validation_h5.json", "validation_h10.json", "best/model.pt", "best/config.json", "best/package_manifest.json",
                         "last/model.pt", "last/config.json", "last/package_manifest.json"):
            source = directory / relative
            sources[str(source.relative_to(ROOT))] = sha(source)
        sources[row["config"]] = row["config_sha256"]
        records.append(record)
        del model, second, state
    check_registration(registration)
    for path, expected in sources.items():
        if sha(ROOT / path) != expected:
            raise ValueError("A source changed during finalization: " + path)
    return registration, records, sources, population


def aggregate(records):
    results = []
    for arm in ARMS:
        for horizon in (5, 10):
            metric = f"h{horizon}_standardized_mse"
            matrices, rows_by_mode = {}, {}
            sessions = None
            for mode in MODES:
                rows = sorted((r for r in records if r["arm"] == arm and r["mode"] == mode), key=lambda r: r["seed"])
                if [r["seed"] for r in rows] != [0, 1, 2]:
                    raise ValueError("Missing seed in matched aggregate")
                rows_by_mode[mode] = rows
                matrices[mode] = np.asarray([[e["errors"]["model"][metric] for e in r["evaluation"][str(horizon)]["episodes"]]
                                            for r in rows], dtype=np.float64)
                current_sessions = [e["session_id"] for e in rows[0]["evaluation"][str(horizon)]["episodes"]]
                if sessions is not None and sessions != current_sessions:
                    raise ValueError("Aggregate session ordering differs")
                sessions = current_sessions
            methods = {}
            reference = matrices["framewise"].mean()
            for mode in MODES:
                values = matrices[mode]
                per_seed = values.mean(1)
                pair = evaluate.crossed_session_bootstrap(values-matrices["framewise"], sessions,
                                                          draws=10000, seed=5197000+horizon)
                methods[mode] = {"mean": float(values.mean()), "seed_sd": float(per_seed.std(ddof=1)),
                                 "per_seed": per_seed.tolist(), "paired_difference": pair,
                                 "gain_percent": float(100*(reference-values.mean())/reference) if reference else None}
            results.append({"arm": arm, "horizon": horizon, "methods": methods,
                            "episodes": matrices["framewise"].shape[1], "sessions": len(set(sessions))})
    return results


def gain_tex(value):
    if value is None:
        return "undefined"
    value_text = f"{value:+.2f}"
    return r"\positivegain{" + value_text + "}" if value > 0 else value_text


def tables(records, aggregates):
    lookup = {(r["arm"], r["horizon"]): r for r in aggregates}
    comparison = [r"\begin{table}[p]\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{llrrrr}\toprule", r"Arm & Method & MSE@5 $\downarrow$ & MSE@10 $\downarrow$ & Gain@5 (\%) & Gain@10 (\%)\\\midrule"]
    costs = [r"\begin{table}[p]\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{llrrrr}\toprule", r"Arm & Method & Best epochs & Total (M) & Trainable (M) & Seconds/run\\\midrule"]
    for arm in ARMS:
        for mode in MODES:
            a, b = lookup[arm, 5]["methods"][mode], lookup[arm, 10]["methods"][mode]
            if mode == "factorized":
                comparison.append(r"\rowcolor{orange!9}")
                costs.append(r"\rowcolor{orange!9}")
            comparison.append(f"{arm.capitalize()} & {NAMES[mode]} & {a['mean']:.5f} & {b['mean']:.5f} & " +
                              ("reference & reference" if mode == "framewise" else f"{gain_tex(a['gain_percent'])} & {gain_tex(b['gain_percent'])}") + r"\\")
            rows = sorted((x for x in records if x["arm"] == arm and x["mode"] == mode), key=lambda x: x["seed"])
            counts = [x["training"]["parameter_counts"] for x in rows]
            if any(x != counts[0] for x in counts):
                raise ValueError("Parameter counts differ across seeds")
            epochs = "/".join(str(x["training"]["best_epoch"]) for x in rows)
            seconds = [x["elapsed_seconds"] for x in rows]
            costs.append(f"{arm.capitalize()} & {NAMES[mode]} & {epochs} & {counts[0]['total']/1e6:.3f} & {counts[0]['trainable']/1e6:.3f} & " +
                         f"{np.median(seconds):.0f} [{min(seconds):.0f}, {max(seconds):.0f}]" + r"\\")
        if arm != ARMS[-1]:
            comparison.append(r"\midrule")
            costs.append(r"\midrule")
    comparison += [r"\bottomrule\end{tabular}",
        r"\caption{\textbf{Complete matched generalization development study.} All 36 runs completed 30 epochs on original training data. Entries are three-seed, equal-episode validation means; MSE uses fixed train-standardized feature coordinates. Five blocks is the primary development horizon; ten blocks is extrapolation. Slow reduces learning rate, Decay increases weight decay, and Compact jointly reduces predictor/context capacity. Gains compare each method to Framewise within the same arm. \positivegain{Bold green} identifies positive point reductions, not statistical significance. Every arm and method is retained; these are validation results, not an additional held-out test.}",
        r"\label{tab:generalization-development}\end{table}"]
    costs += [r"\bottomrule\end{tabular}",
        r"\caption{\textbf{Reusable checkpoints and computation.} Best epochs are ordered by seeds 0/1/2 and selected by the shared all-five-query validation objective after full 30-epoch training. Parameter counts include frozen parameters; trainable counts exclude inactive modules. Seconds/run is the median [minimum, maximum] successful runner invocation, including training, validation and CPU reload verification; prior interrupted work, if any, is excluded. Different GPU partitions are used, so these times are operational evidence, not a hardware-matched speed comparison. All 36 selected packages pass strict CPU reload parity.}",
        r"\label{tab:generalization-checkpoints}\end{table}"]
    intervals = []
    for horizon in (5, 10):
        block = [r"\begin{table}[p]\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}",
            r"\begin{tabular}{llrrl}\toprule", r"Arm & Method & $\Delta$MSE & Gain (\%) & Paired 95\% interval\\\midrule"]
        for arm in ARMS:
            for mode in MODES[1:]:
                item = lookup[arm, horizon]["methods"][mode]
                pair = item["paired_difference"]
                if mode == "factorized":
                    block.append(r"\rowcolor{orange!9}")
                block.append(f"{arm.capitalize()} & {NAMES[mode]} & {pair['mean_difference']:+.6f} & {gain_tex(item['gain_percent'])} & " +
                             f"[{pair['ci95'][0]:+.6f}, {pair['ci95'][1]:+.6f}]" + r"\\")
            if arm != ARMS[-1]:
                block.append(r"\midrule")
        block += [r"\bottomrule\end{tabular}",
            r"\caption{\textbf{Matched validation uncertainty at " + str(horizon) + r" blocks.} Differences are method minus Framewise within each arm, in standardized feature MSE; negative favors the named method. Relative gains are $100(\mathrm{Framewise}-\mathrm{method})/\mathrm{Framewise}$. Intervals jointly resample recording sessions and the three training seeds with 10,000 draws. These exploratory development intervals are unadjusted for multiple comparisons; intervals crossing zero do not establish a difference. Positive point gains are colored independently of interval significance.}",
            r"\label{tab:generalization-ci-" + str(horizon) + r"}\end{table}"]
        intervals.extend(block)
    return {"comparison.tex": "\n".join(comparison)+"\n", "intervals.tex": "\n".join(intervals)+"\n", "checkpoints.tex": "\n".join(costs)+"\n"}


def proof(stage, files):
    for name, contents in files.items():
        (stage / name).write_text(contents)
    (stage / "proof.tex").write_text(r"""\documentclass{article}
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=5.5in,textheight=9in]{geometry}
\usepackage{booktabs,xcolor,colortbl,amsmath}
\definecolor{gainpositive}{HTML}{166534}
\newcommand{\positivegain}[1]{\textcolor{gainpositive}{\textbf{#1}}}
\begin{document}
\input{comparison.tex}\input{intervals.tex}\input{checkpoints.tex}
\end{document}
""")
    for _ in range(2):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "proof.tex"], cwd=stage, capture_output=True, text=True)
        if result.returncode:
            raise ValueError("Standalone table proof failed: " + result.stdout[-1500:])
    log = (stage / "proof.log").read_text()
    if "Overfull" in log or "undefined" in log or "multiply defined" in log:
        raise ValueError("Table proof has overflow or unresolved references")
    return {"status": "passed", "compile_passes": 2, "overflow_or_reference_warnings": 0,
            "proof_sha256": sha(stage / "proof.pdf"), "visual_review": "Pending inspection of completed result proof; automated layout checks passed"}


def finalize(build_paper=False):
    torch.set_num_threads(4)
    registration, records, sources, populations = collect()
    aggregates = aggregate(records)
    # Invoke the immutable registered summarizer, then check every primary
    # aggregate and interval against independently collected per-episode data.
    subprocess.run([sys.executable, str(RUNNER), "--summarize"], cwd=ROOT, check=True, capture_output=True, text=True)
    official = read(ROOT / "reports/real_droid_generalization_results.json")
    if official["status"] != "completed" or official["completed_runs"] != 36:
        raise ValueError("Frozen summarizer did not confirm all 36 runs")
    for ours in aggregates:
        reference = next(x for x in official["aggregate"] if (x["arm"], x["horizon"]) == (ours["arm"], ours["horizon"]))
        for mode in MODES:
            if not math.isclose(reference["means"][mode], ours["methods"][mode]["mean"], rel_tol=1e-12):
                raise ValueError("Independent and registered aggregates disagree")
        pair = ours["methods"]["factorized"]["paired_difference"]
        if not np.allclose(reference["paired_ours_minus_framewise"]["ci95"], pair["ci95"], atol=1e-14, rtol=1e-12):
            raise ValueError("Independent and registered paired intervals disagree")
    files = tables(records, aggregates)
    section = r"""\section{Matched generalization development on recorded video}
\label{app:generalization-development}
This registered development study investigates the early validation optimum in
the original recorded-video campaign. Three fixed interventions change either
the learning rate (Slow: $3\times10^{-5}$), weight decay (Decay: $0.1$), or the
predictor/context capacity (Compact: width 96, depth 2, context dimensions 16/64).
Each arm trains Framewise, Constant dynamics, ShiftWM (ours), and Action-free at
three seeds for all 30 epochs. The Compact intervention changes these capacity
settings jointly and does not isolate their separate effects.

All inputs, normalization and model selection use the original training and
validation partitions. Neither the original test nor the separately acquired
fresh holdout selects these revisions. Each checkpoint uses the common
all-five-query validation objective. Subsequent five- and ten-block forecasts
are summarized by averaging windows within episodes and then equally averaging
episodes and seeds. Development improvements, if present, require another
independent evaluation for a new confirmatory claim; ordinary optimization and
capacity controls are not presented as additional method novelty.

Table~\ref{tab:generalization-development} reports every matched arm and method.
Tables~\ref{tab:generalization-ci-5} and~\ref{tab:generalization-ci-10} provide
the session/seed paired intervals, including regressions and intervals crossing
zero. Table~\ref{tab:generalization-checkpoints} records the selected epochs,
parameter counts and measured invocation times. All 36 selected packages pass
strict weights-only CPU reload parity on original-validation causal inputs.
\input{generated/real_video/generalization/comparison.tex}
\input{generated/real_video/generalization/intervals.tex}
\input{generated/real_video/generalization/checkpoints.tex}
"""
    with tempfile.TemporaryDirectory(prefix="shiftwm-generalization-proof-") as name:
        stage = Path(name)
        review = proof(stage, files)
        for path, expected in sources.items():
            if sha(ROOT / path) != expected:
                raise ValueError("A source changed before publication")
        GENERATED.mkdir(parents=True, exist_ok=True)
        # The section appears last: a manuscript IfFileExists gate cannot see
        # incomplete generated tables on the first successful publication.
        for filename, contents in files.items():
            temporary = GENERATED / (filename + ".tmp")
            temporary.write_text(contents)
            temporary.replace(GENERATED / filename)
        shutil.copyfile(stage / "proof.pdf", GENERATED / "proof.pdf")
        shutil.copyfile(stage / "proof.log", GENERATED / "proof.log")
        train.atomic_json(review, GENERATED / "review.json")
        train.atomic_json({"sources": sources, "registration_sha256": sha(REGISTRY),
                           "scope": "original validation development only", "generation_script_sha256": sha(__file__)}, GENERATED / "sources.json")
        temporary = SECTION.with_suffix(".tex.tmp")
        temporary.write_text(section)
        temporary.replace(SECTION)
    public_records = []
    for record in records:
        record = dict(record)
        record.pop("evaluation")
        public_records.append(record)
    report = {"status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "completed_training_runs": 36, "completed_validation_evaluations": 72,
              "registered_checkpoint_parity_passes": 36, "independent_cpu_reload_parity_passes": 36,
              "scope": "original validation development only; no original or fresh test payloads read",
              "registration_sha256": sha(REGISTRY), "sources": sources, "runs": public_records,
              "aggregate": aggregates, "review": review,
              "manuscript_files": {str(p.relative_to(ROOT)): sha(p) for p in [SECTION, *sorted(GENERATED.glob("*.tex"))]}}
    train.atomic_json(report, REPORT)
    if build_paper:
        subprocess.run(["bash", "paper/build.sh"], cwd=ROOT, check=True)
        report["paper_build"] = {"status": "passed", "pdf_sha256": sha(ROOT / "paper/world_model_draft.pdf")}
        train.atomic_json(report, REPORT)
    print(json.dumps({"status": "completed", "runs": 36, "evaluations": 72, "paper_built": build_paper}))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-paper", action="store_true")
    args = parser.parse_args()
    lock = ROOT / "artifacts/development/generalization_finalization.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finalize(build_paper=args.build_paper)


if __name__ == "__main__":
    main()
