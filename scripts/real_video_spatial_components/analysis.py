"""Predeclared component effects and their paired session × seed interaction."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODES = ("anchored_additive", "bounded_additive", "unbounded_transport", "transport")
EDGES = (
    ("bounding_without_mixing", "bounded_additive", "anchored_additive"),
    ("bounding_with_mixing", "transport", "unbounded_transport"),
    ("mixing_without_bounding", "unbounded_transport", "anchored_additive"),
    ("mixing_with_bounding", "transport", "bounded_additive"),
)


def helper():
    spec = importlib.util.spec_from_file_location("component_analysis_private_ledger", ROOT / "scripts/real_video_spatial/validate_ledger.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def interaction(results, draws=10000, seed=173):
    """(Bounded mix − unbounded mix) − (bounded additive − additive).

    Negative values mean that bounding reduces error more with the mixing
    package. This is a signed absolute MSE interaction, not a percentage gain.
    Inputs must already pass the full checkpoint and per-window ledger gates.
    """
    h = helper()
    if len(results) != 12 or {(r["mode"], r["seed"]) for r in results} != {(m, s) for m in MODES for s in (0, 1, 2)}:
        raise ValueError("Interaction requires exactly four arms × three unique seeds")
    matrices = {}
    population = window_keys = None
    for mode in MODES:
        group = []
        for result in sorted((r for r in results if r["mode"] == mode), key=lambda r: r["seed"]):
            episodes = sorted(result["episodes"], key=lambda row: row["episode_id"])
            keys = [(r["episode_id"], r["session_id"]) for r in episodes]
            windows = sorted((r["episode_id"], r["session_id"], r["window_start"]) for r in result["windows"])
            if population is None:
                population, window_keys = keys, windows
            if (keys != population or windows != window_keys or len(set(keys)) != len(keys)
                    or len(set(windows)) != len(windows)):
                raise ValueError("Interaction populations must match without duplicates")
            group.append(np.stack([[h.vector(e[metric], metric) for metric in h.METRICS] for e in episodes]))
        matrices[mode] = np.stack(group)
    delta = ((matrices["transport"] - matrices["unbounded_transport"])
             - (matrices["bounded_additive"] - matrices["anchored_additive"]))
    sessions = np.array([session for _, session in population])
    clusters = [np.flatnonzero(sessions == session) for session in sorted(set(sessions))]
    if len(clusters) < 2 or type(draws) is not int or draws < 1:
        raise ValueError("Interaction needs at least two sessions and positive bootstrap draws")
    rng = np.random.default_rng(seed)
    distribution = np.empty((draws, len(h.METRICS), 10))
    for i in range(draws):
        selected_seeds = rng.integers(0, 3, size=3)
        selected_episodes = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), size=len(clusters))])
        distribution[i] = delta[selected_seeds][:, selected_episodes].mean((0, 1))
    ci = np.quantile(distribution, [.025, .975], axis=0)
    mean = delta.mean((0, 1))
    return {"formula": "(transport-unbounded_transport)-(bounded_additive-anchored_additive)",
            "direction": "negative means bounding reduces error more with the mixing package",
            "episode_count": len(population), "session_count": len(clusters), "seed_count": 3,
            "metrics": {metric: [{"horizon": t+1, "mean_difference": float(mean[j, t]),
                                    "ci95": ci[:, j, t].tolist()} for t in range(10)]
                        for j, metric in enumerate(h.METRICS)}}


def contrasts(results, draws=10000, seed=173):
    # This call checks all four-arm populations before any two-arm reporting.
    did = interaction(results, draws, seed)
    h = helper()
    effects, complete = [], {}
    common = {"bootstrap_draws": draws, "bootstrap_seed": seed,
              "bootstrap_units": "recording sessions and training seeds, paired",
              "scope": "follow-up original-validation development; unadjusted intervals"}
    for name, first, second in EDGES:
        comparison = h.paired_intervals(results, first, second, draws, seed)
        complete[name] = comparison
        for metric in ("native_mse", "original_2x2_mse"):
            for horizon in (5, 10):
                row = comparison["metrics"][metric][horizon-1]
                effects.append({**common, "contrast": name, "method": first, "comparator": second,
                    "metric": metric, "horizon": horizon, "method_mean": row["first_mean"],
                    "comparator_mean": row["second_mean"], "method_minus_comparator": row["mean_difference"],
                    "relative_error_reduction_percent": 100*(row["second_mean"]-row["first_mean"])/row["second_mean"] if row["second_mean"] > 0 else None,
                    "paired_95_percent_interval": row["ci95"],
                    "interval_includes_zero": row["ci95"][0] <= 0 <= row["ci95"][1]})
    for metric in ("native_mse", "original_2x2_mse"):
        for horizon in (5, 10):
            row = did["metrics"][metric][horizon-1]
            effects.append({**common, "contrast": "mixing_x_bounding_interaction", "formula": did["formula"],
                "direction": did["direction"], "metric": metric, "horizon": horizon,
                "difference_of_differences": row["mean_difference"], "paired_95_percent_interval": row["ci95"],
                "interval_includes_zero": row["ci95"][0] <= 0 <= row["ci95"][1]})
    return {"reported_effects": effects, "all_horizons": complete, "interaction_all_horizons": did}
