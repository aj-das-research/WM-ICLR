"""Fill paper tables from completed evaluation files in results/v2.

Cells without completed runs stay \\pend. A cell is filled only when all requested seeds of that
arm have finished; the number of seeds is recorded in tables/generated/provenance.json.
Paired bootstrap (10k resamples over episodes -- sessions for DROID) decides the dagger mark.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/v2"
TAB = ROOT / "paper/submission_folder/tables"
GEN = TAB / "generated"
PEND = r"\pend"
ARMS = [("persistence", "Persistence"), ("linear", "Linear extrapolation"), ("ar_tf", "AR-TF (DINO-WM-style)"),
        ("ar", "AR (rollout-trained)"), ("direct", "Direct (cross-attn)"), ("shiftwm", r"\ours{} (ours)")]
LEARNED = {"ar_tf", "ar", "direct", "shiftwm"}
provenance = {}


def load(dataset, arm, encoder="dinov2s", split="test", min_seeds=1):
    runs = sorted((RES / dataset / encoder / arm).glob(f"s*/eval_{split}.npz"))
    if arm not in LEARNED:
        runs = runs[:1]
    if len(runs) < (min_seeds if arm in LEARNED else 1):
        return None
    arrs = [np.load(r, allow_pickle=True) for r in runs]
    provenance[f"{dataset}/{encoder}/{arm}/{split}"] = [str(r.relative_to(ROOT)) for r in runs]
    return {"mse": np.stack([a["mse"] for a in arrs]).mean(0),          # [E, K] averaged over seeds
            "episodes": list(arrs[0]["episodes"]), "tasks": list(arrs[0]["tasks"]), "seeds": len(runs)}


def paired_ci(a, b, n=10000, seed=0):
    """95% percentile CI of mean(a-b) over episodes (a,b: per-episode values)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(boots, [2.5, 97.5])


def fmt(v, bold=False, under=False, dagger=False):
    s = f"{v:.3f}"
    if bold:
        s = r"\textbf{" + s + "}"
    if under:
        s = r"\underline{" + s + "}"
    return s + (r"$^\dagger$" if dagger else "")


IWS_TASKS = ("iws_pusht", "iws_box", "iws_rope")


def load_iws(arm, split="test", min_seeds=1):
    """Macro view over the three IWS tasks: per-episode rows concatenated, each task weighted equally."""
    parts = [load(t, arm, split=split, min_seeds=min_seeds) for t in IWS_TASKS]
    if any(p is None for p in parts):
        return None
    w = [np.full(len(p["mse"]), 1.0 / (len(IWS_TASKS) * len(p["mse"]))) for p in parts]
    m = np.concatenate([p["mse"] for p in parts])
    return {"mse": m, "weights": np.concatenate(w) * len(m), "episodes": sum((p["episodes"] for p in parts), []),
            "tasks": sum((p["tasks"] for p in parts), []), "seeds": min(p["seeds"] for p in parts)}


def column(dataset, reducer, split="test", min_seeds=1):
    vals = {}
    per_ep = {}
    for arm, _ in ARMS:
        ev = load_iws(arm, split, min_seeds) if dataset == "iws" else load(dataset, arm, split=split, min_seeds=min_seeds)
        if ev is not None:
            e = reducer(ev["mse"]) * ev.get("weights", 1.0)
            vals[arm], per_ep[arm] = float(e.mean()), e
    return vals, per_ep


def rank_marks(vals, per_ep):
    order = sorted(vals, key=vals.get)
    marks = {a: {} for a in vals}
    if "shiftwm" not in vals:           # no ranking marks until the proposed method is scored
        return marks
    if order:
        marks[order[0]]["bold"] = True
    if len(order) > 1:
        marks[order[1]]["under"] = True
    if "shiftwm" in vals:
        rivals = [a for a in vals if a != "shiftwm"]
        if rivals:
            best = min(rivals, key=vals.get)
            lo, hi = paired_ci(per_ep["shiftwm"], per_ep[best])
            if hi < 0:
                marks["shiftwm"]["dagger"] = True
    return marks


def main_table():
    cols = [("droid", lambda m: m.mean(1)), ("droid", lambda m: m[:, -1]), ("droid_cam2", lambda m: m.mean(1)),
            ("openh_hamlyn", lambda m: m.mean(1)), ("iws", lambda m: m.mean(1)), ("bridge", lambda m: m.mean(1)),
            ("fractal", lambda m: m.mean(1))]
    cells = {arm: [] for arm, _ in ARMS}
    for ds, red in cols:
        vals, per_ep = column(ds, red)
        marks = rank_marks(vals, per_ep)
        for arm, _ in ARMS:
            cells[arm].append(fmt(vals[arm], **marks[arm]) if arm in vals else PEND)
    rows = []
    for arm, label in ARMS:
        c = cells[arm]
        pre = r"\rowcolor{bestbg}" if arm == "shiftwm" else ""
        rows.append(f"{pre}{label} & {c[0]} & {c[1]} && {c[2]} & {c[3]} & {c[4]} & {c[5]} & {c[6]} \\\\")
    return "\n".join(rows)


def per_horizon_table():
    rows = []
    for arm, label in ARMS:
        if arm == "linear":
            continue
        ev = load("droid", arm)
        if ev is None:
            rows.append(label + " & " + " & ".join([PEND] * 10) + r" \\")
        else:
            rows.append(label + " & " + " & ".join(f"{v:.3f}" for v in ev["mse"].mean(0)) + r" \\")
    return "\n".join(rows)


def hamlyn_tasks_table():
    tasks = ["knot_tying", "needle_grasp_and_handover", "peg_transfer", "suturing_1", "suturing_2",
             "tissue_lifting", "tissue_retraction"]
    rows = []
    for arm, label in ARMS:
        if arm == "linear":
            continue
        ev = load("openh_hamlyn", arm)
        if ev is None:
            rows.append(label + " & " + " & ".join([PEND] * 7) + r" \\")
            continue
        t = np.array(ev["tasks"]); m = ev["mse"].mean(1)
        rows.append(label + " & " + " & ".join(f"{m[t == k].mean():.3f}" for k in tasks) + r" \\")
    return "\n".join(rows)


def region_table(dataset="droid", encoder="dinov2s"):
    """Moving / static error and skill vs persistence, from results/v2/analysis/regions (all finished seeds)."""
    f = RES / f"analysis/regions/{dataset}_{encoder}_K10.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    def agg(arm, m):
        runs = [np.array(v[m]) for k, v in r.items() if k.split("/")[0] == arm]
        return np.mean([x.mean() for x in runs]) if runs else None
    base = {m: agg("persistence", m) for m in ("all", "moving", "static")}
    rows, order = [], [("persistence", "Persistence"), ("ar_tf", "AR-TF (DINO-WM-style)"), ("ar", "AR (rollout-trained)"),
                       ("direct", "Direct (cross-attn)"), ("shiftwm", r"\ours{} (ours)")]
    vals = {a: {m: agg(a, m) for m in ("all", "moving", "static")} for a, _ in order}
    best = {m: min(v[m] for v in vals.values() if v[m] is not None) for m in ("all", "moving", "static")}
    for arm, label in order:
        v = vals[arm]
        if v["all"] is None:
            rows.append(label + " & " + " & ".join([PEND] * 4) + r" \\"); continue
        cell = lambda m: (r"\textbf{%.3f}" if abs(v[m] - best[m]) < 1e-9 else "%.3f") % v[m]
        skill = 100 * (1 - v["all"] / base["all"])
        pre = r"\rowcolor{bestbg}" if arm == "shiftwm" else ""
        rows.append(f"{pre}{label} & {cell('moving')} & {cell('static')} & {cell('all')} & {skill:.1f}\\% \\\\")
    return "\n".join(rows)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    rt = region_table()
    (GEN / "region_rows.tex").write_text((rt or "Persistence & \\pend & \\pend & \\pend & \\pend \\\\") + "\n")
    (GEN / "main_rows.tex").write_text(main_table() + "\n")
    (GEN / "per_horizon_rows.tex").write_text(per_horizon_table() + "\n")
    (GEN / "hamlyn_task_rows.tex").write_text(hamlyn_tasks_table() + "\n")
    (GEN / "provenance.json").write_text(json.dumps(provenance, indent=1))
    # Seed status for captions: which learned arms are complete (3/3) on each dataset.
    status = []
    for ds, name in (("droid", "DROID"), ("openh_hamlyn", "Hamlyn")):
        counts = {a: len(list((RES / ds / "dinov2s" / a).glob("s*/eval_test.npz"))) for a in sorted(LEARNED)}
        if any(c < 3 for c in counts.values()):
            status.append(name + ": " + ", ".join(f"{a.replace('_', '-')} {c}/3" for a, c in counts.items()))
    note = (r" \textcolor{mutedgray}{[Interim: seeds completed -- " + "; ".join(status) + ".]}") if status else ""
    (GEN / "seed_status.tex").write_text("\\def\\seedstatus{" + note + "}\n")
    print("tables written;", len(provenance), "result groups used")


if __name__ == "__main__":
    main()
