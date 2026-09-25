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


# Final results use the short-schedule recipe (results/v2s); fall back to the first recipe (results/v2) until ready.
ROOTS = [ROOT / "results/v2s", ROOT / "results/v2"]


def root_for(dataset, encoder="dinov2s", split="test"):
    """One recipe per dataset: the first root in which every learned arm has at least one finished run."""
    for base in ROOTS:
        if all(list((base / dataset / encoder / a).glob(f"s*/eval_{split}.npz")) for a in ("shiftwm", "direct", "ar")):
            return base
    return ROOTS[-1]


def load(dataset, arm, encoder="dinov2s", split="test", min_seeds=1):
    base = root_for(dataset, encoder, split)
    runs = sorted((base / dataset / encoder / arm).glob(f"s*/eval_{split}.npz"))
    if arm not in LEARNED and not runs:
        runs = sorted((ROOTS[-1] / dataset / encoder / arm).glob(f"s*/eval_{split}.npz"))
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


def _droid_sessions():
    """DROID episode id -> recording session (the resampling unit for DROID)."""
    f = ROOT / "data/v2/features/droid/dinov2s/manifest.json"
    if not f.exists():
        return {}
    return {r["id"]: r.get("session", r["id"]) for r in json.loads(f.read_text())["episodes"]}


def paired_ci_pct(dataset, n=10000, seed=0):
    """95% CI of the % error reduction of ShiftWM vs. the best learned baseline (seed-mean per-episode MSE).
    Resamples recording sessions for DROID (episodes elsewhere); returns (lo, hi) in % of the baseline mean."""
    sw = load(dataset, "shiftwm")
    base = [(a, load(dataset, a)) for a in ("ar_tf", "ar", "direct")]
    base = [(a, b) for a, b in base if b is not None]
    if sw is None or not base:
        return None
    _, b = min(base, key=lambda ab: ab[1]["mse"].mean())
    d = sw["mse"].mean(1) - b["mse"].mean(1)
    groups = [_droid_sessions().get(e, e) for e in sw["episodes"]] if dataset == "droid" else list(sw["episodes"])
    uniq = sorted(set(groups)); gi = np.array([uniq.index(g) for g in groups])
    sums, cnts = np.bincount(gi, d, len(uniq)), np.bincount(gi, None, len(uniq))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (n, len(uniq)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    m = b["mse"].mean()
    return -100 * hi / m, -100 * lo / m


def fmt(v, bold=False, under=False, dagger=False, ours=False):
    s = f"{v:.3f}"
    if bold:
        s = (r"\good{" if ours else r"\textbf{") + s + "}"
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
            ("fractal", lambda m: m.mean(1)), ("language_table", lambda m: m.mean(1))]
    cells = {arm: [] for arm, _ in ARMS}
    gains = []
    for ds, red in cols:
        vals, per_ep = column(ds, red)
        marks = rank_marks(vals, per_ep)
        for arm, _ in ARMS:
            cells[arm].append(fmt(vals[arm], ours=(arm == "shiftwm"), **marks[arm]) if arm in vals else PEND)
        # improvement of ShiftWM over the best learned baseline (green bold if positive)
        rivals = [vals[a] for a in ("ar_tf", "ar", "direct") if a in vals]
        if "shiftwm" in vals and rivals:
            g = 100 * (1 - vals["shiftwm"] / min(rivals))
            gains.append((r"\good{" if g > 0 else "") + f"{g:+.1f}\\%" + ("}" if g > 0 else ""))
        else:
            gains.append(PEND)
    rows = []
    for arm, label in ARMS:
        c = cells[arm]
        pre = r"\rowcolor{bestbg}" if arm == "shiftwm" else ""
        rows.append(f"{pre}{label} & " + " & ".join(c) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\textit{error reduction vs.\ best baseline} & " + " & ".join(gains) + r" \\")
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
        mk = r"\good{%.3f}" if arm == "shiftwm" else r"\textbf{%.3f}"
        cell = lambda m: (mk if abs(v[m] - best[m]) < 1e-9 else "%.3f") % v[m]
        skill = 100 * (1 - v["all"] / base["all"])
        pre = r"\rowcolor{bestbg}" if arm == "shiftwm" else ""
        sk = (r"\good{%.1f\%%}" if arm == "shiftwm" else "%.1f\%%") % skill
        rows.append(f"{pre}{label} & {cell('moving')} & {cell('static')} & {cell('all')} & {sk} \\\\")
    return "\n".join(rows)


def recipe_table():
    """Every training recipe / add-on we evaluated on DROID (VALIDATION MSE avg over horizons, seed 0).
    Recipes were compared on validation only; the test split is scored once for the final recipe (Table 1)."""
    def test(path):
        f = RES.parent / path / "summary.json"
        if not f.exists():
            return PEND
        return f"{json.loads(f.read_text())['results']['val']['mse_mean_h']:.3f}"
    recipes = [("base (16k steps)", "v2/droid/dinov2s/{a}/s0"), ("base, short (8k steps)", "v2s/droid/dinov2s/{a}/s0"),
               ("+ 2nd camera, EMA, dropout", "v2r2/droid/dinov2s/{a}/s0"),
               ("+ correlation features", "v2r2/droid/dinov2s/ablations/{a}_cv/s0")]
    arms = [("ar_tf", "AR-TF"), ("ar", "AR"), ("direct", "Direct"), ("shiftwm", r"\ours{}")]
    rows = [name + " & " + " & ".join(test(path.format(a=a)) for a, _ in arms) + r" \\" for name, path in recipes]
    return "\n".join(rows)


def external_table():
    """Skill vs persistence on identical DROID test windows, each model in its own feature space."""
    rows = []
    f = RES / "external/vjepa2ac/droid_test_summary.json"
    if f.exists():
        g = json.loads(f.read_text())["relative_gain_vs_persistence"]["mse"]
        lo, hi = g["mean_over_horizons_ci95_session_bootstrap"]
        rows.append(f"V-JEPA 2-AC (zero-shot, ViT-g, 1.3B) & V-JEPA 2 ViT-g & {100*g['mean_over_horizons']:.1f} [{100*lo:.1f}, {100*hi:.1f}] \\\\")
    for base, sub in (("vjepa2ac_plugin/finetune", "V-JEPA 2-AC fine-tuned on our split"),
                      ("vjepa2ac_plugin/finetune_shiftwm", r"V-JEPA 2-AC fine-tuned + \ours{} head")):
        sm = sorted((RES / "external" / base).glob("s*/test_summary.json"))
        if sm:
            gains = [json.loads(f.read_text()).get("relative_gain_vs_persistence", {}).get("mse", {}).get("mean_over_horizons")
                     for f in sm]
            gains = [x for x in gains if x is not None]
            v = float(np.mean(gains)) if gains else None
            sub = sub + (f" ({len(gains)} seeds)" if len(gains) > 1 else "")
            rows.append(f"{sub} & V-JEPA 2 ViT-g & {100*v:.1f} \\\\" if v is not None else f"{sub} & V-JEPA 2 ViT-g & {PEND} \\\\")
        else:
            rows.append(f"{sub} & V-JEPA 2 ViT-g & {PEND} \\\\")
    for arm, label in (("ar_tf", "AR-TF (DINO-WM-style)"), ("ar", "AR"), ("direct", "Direct"), ("shiftwm", r"\ours{} (ours)")):
        ev, base = load("droid", arm), load("droid", "persistence")
        if ev is None or base is None:
            rows.append(f"{label} & DINOv2-S & {PEND} \\\\"); continue
        rows.append(f"{label} & DINOv2-S & {100*(1-ev['mse'].mean()/base['mse'].mean()):.1f} \\\\")
    return "\n".join(rows)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "external_rows.tex").write_text(external_table() + "\n")
    (GEN / "recipe_rows.tex").write_text(recipe_table() + "\n")
    rt = region_table()
    (GEN / "region_rows.tex").write_text((rt or "Persistence & \\pend & \\pend & \\pend & \\pend \\\\") + "\n")
    (GEN / "main_rows.tex").write_text(main_table() + "\n")
    (GEN / "per_horizon_rows.tex").write_text(per_horizon_table() + "\n")
    (GEN / "hamlyn_task_rows.tex").write_text(hamlyn_tasks_table() + "\n")
    (GEN / "provenance.json").write_text(json.dumps(provenance, indent=1))
    # Seed status for captions: which learned arms are complete (3/3) on each dataset.
    status = []
    for ds, name in (("droid", "DROID"), ("openh_hamlyn", "Hamlyn")):
        counts = {a: len(list((root_for(ds) / ds / "dinov2s" / a).glob("s*/eval_test.npz"))) for a in sorted(LEARNED)}
        if any(c < 3 for c in counts.values()):
            status.append(name + ": " + ", ".join(f"{a.replace('_', '-')} {c}/3" for a, c in counts.items()))
    note = (r" \textcolor{mutedgray}{[Interim: seeds completed -- " + "; ".join(status) + ".]}") if status else ""
    (GEN / "seed_status.tex").write_text("\\def\\seedstatus{" + note + "}\n")
    print("tables written;", len(provenance), "result groups used")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- plug-in tables + text macros
def _vjepa(arm):
    base = RES / "external/vjepa2ac_plugin" / arm
    files = [base / "test_summary.json"] if arm == "zeroshot" else sorted(base.glob("s*/test_summary.json"))
    files = [f for f in files if f.exists()]
    if not files:
        return None
    ds = [json.loads(f.read_text()) for f in files]
    m = lambda k: float(np.mean([d["mean_over_horizons"][k] for d in ds]))
    return {"mse": m("model_mse"), "moving": m("model_mse_moving"), "static": m("model_mse_static"), "cos": m("model_cos"),
            "skill": 100 * float(np.mean([d["relative_gain_vs_persistence"]["mse"]["mean_over_horizons"] for d in ds])),
            "rank": float(np.mean([d.get("action_rank_acc") or np.nan for d in ds])), "seeds": len(ds),
            "pers": m("persistence_mse"), "pers_moving": m("persistence_mse_moving")}


def vjepa_rows():
    arms = [("zeroshot", "V-JEPA 2-AC, zero-shot"), ("finetune", "V-JEPA 2-AC, fine-tuned"),
            ("finetune_shiftwm", r"V-JEPA 2-AC, fine-tuned + \ours{} head")]
    vals = {a: _vjepa(a) for a, _ in arms}
    have = {a: v for a, v in vals.items() if v}
    best = {k: (min if k in ("mse", "moving", "static") else max)(v[k] for v in have.values()) for k in ("mse", "moving", "static", "skill")} if have else {}
    rows = []
    for a, label in arms:
        v = vals[a]
        if v is None:
            rows.append(label + " & " + " & ".join([PEND] * 4) + r" \\"); continue
        mk = r"\good{" if a == "finetune_shiftwm" else r"\textbf{"
        c = lambda k, f="%.3f": (mk + f % v[k] + "}") if abs(v[k] - best[k]) < 1e-12 and len(have) > 1 else f % v[k]
        pre = r"\rowcolor{bestbg}" if a == "finetune_shiftwm" else ""
        seeds = f" ({v['seeds']} seeds)" if v["seeds"] > 1 else ""
        rows.append(f"{pre}{label}{seeds} & {c('mse')} & {c('moving')} & {c('static')} & {c('skill', '%.1f')} \\\\")
    return "\n".join(rows), vals


def _dinowm(env):
    base = RES / "external/dinowm_plugin" / env
    out = {}
    for arm in ("dinowm", "dinowm_shiftwm"):
        ol = base / arm / "openloop.json"
        if not ol.exists():
            continue
        t = json.loads(ol.read_text()).get("teacher_forced", {})
        succ = None
        for d in sorted((base / arm).glob("plan_*_seed*/final.json")):
            s = json.loads(d.read_text().strip().splitlines()[-1]).get("final_eval/success_rate")
            succ = (succ or []) + [s] if s is not None else succ
        out[arm] = {"err": t.get("z_visual_err_pred"), "lpips": t.get("pred_img_lpips"), "ssim": t.get("pred_img_ssim"),
                    "succ": 100 * float(np.mean(succ)) if succ else None}
    return out


def dinowm_rows():
    rows, allv = [], {}
    for env, name in (("pusht", "PushT"), ("wall", "Wall")):
        v = _dinowm(env); allv[env] = v
        for arm, label in (("dinowm", "DINO-WM"), ("dinowm_shiftwm", r"DINO-WM + \ours{} head")):
            x = v.get(arm)
            f = lambda k, fmt="%.3f": PEND if not x or x.get(k) is None else fmt % x[k]
            pre = r"\rowcolor{bestbg}" if arm == "dinowm_shiftwm" else ""
            rows.append(f"{pre}{name} & {label} & {f('err')} & {f('ssim')} & {f('lpips')} & {f('succ', '%.1f')} \\\\")
    return "\n".join(rows), allv


def numbers_macros(vj, dw):
    """Every number quoted in the prose, recomputed from result files (\\pend if not available yet)."""
    M = {}
    def put(name, val, fmt="%.1f"):
        M[name] = PEND if val is None or (isinstance(val, float) and not np.isfinite(val)) else fmt % val
    def red(a, b):  # % reduction of a relative to b
        return None if a is None or b is None else 100 * (1 - a / b)
    def mean_mse(ds, arm):
        ev = load(ds, arm)
        return None if ev is None else float(ev["mse"].mean())
    for ds, tag in (("droid", "droid"), ("openh_hamlyn", "hamlyn"), ("language_table", "lt"), ("bridge", "bridge"), ("fractal", "rtone")):
        sw, di, ar, at, pe = (mean_mse(ds, a) for a in ("shiftwm", "direct", "ar", "ar_tf", "persistence"))
        put(tag + "VsDirect", red(sw, di)); put(tag + "VsAR", red(sw, ar)); put(tag + "VsARTF", red(sw, at))
        put(tag + "Skill", red(sw, pe)); put(tag + "SkillDirect", red(di, pe)); put(tag + "SkillAR", red(ar, pe))
        ev = load(ds, "shiftwm"); put(tag + "Seeds", ev["seeds"] if ev else None, "%d")
    # Paired 95% CI of ShiftWM vs. the best learned baseline, as % error reduction (resampling sessions for DROID).
    for ds, tag in (("droid", "droid"), ("language_table", "lt"), ("openh_hamlyn", "hamlyn")):
        ci = paired_ci_pct(ds)
        put(tag + "CILo", ci and ci[0]); put(tag + "CIHi", ci and ci[1])
    # Action-ranking accuracy (%): true future actions give lower error than another episode's actions (seed mean).
    for arm, tag in (("shiftwm", "Shift"), ("direct", "Direct"), ("ar", "AR"), ("ar_tf", "ARTF")):
        base = root_for("droid") / "droid/dinov2s" / arm
        v = [np.load(f)["rank_ok"].mean() for f in sorted(base.glob("s*/eval_test.npz")) if "rank_ok" in np.load(f).files]
        put("droidRank" + tag, 100 * float(np.mean(v)) if v else None)
    # V-JEPA 2-AC + head: mean gate on test at the selected checkpoint, and on validation at the end of fine-tuning.
    vs = sorted((RES / "external/vjepa2ac_plugin/finetune_shiftwm").glob("s*/test_summary.json"))
    gt = [json.loads(f.read_text())["mean_over_horizons"].get("gate_mean") for f in vs]
    gt = [x for x in gt if x is not None]
    put("vjepaGateTest", 100 * float(np.mean(gt)) if gt else None, "%.0f")
    ge = [json.loads((f.parent / "curve.json").read_text())["curve"][-1].get("val_gate_mean") for f in vs if (f.parent / "curve.json").exists()]
    ge = [x for x in ge if x is not None]
    put("vjepaGateEnd", 100 * float(np.mean(ge)) if ge else None, "%.0f")
    put("droidTestSessions", len(set(_droid_sessions().get(e, e) for e in (load("droid", "shiftwm") or {}).get("episodes", []))) or None, "%d")
    f = RES / "analysis/regions/droid_dinov2s_K10.json"
    if f.exists():
        r = json.loads(f.read_text())
        g = lambda arm, m: (np.mean([np.mean(v[m]) for k, v in r.items() if k.split("/")[0] == arm]) if any(k.split("/")[0] == arm for k in r) else None)
        put("droidMovingVsAR", red(g("shiftwm", "moving"), g("ar", "moving"))); put("droidMovingVsDirect", red(g("shiftwm", "moving"), g("direct", "moving")))
        put("droidStaticVsAR", red(g("shiftwm", "static"), g("ar", "static"))); put("droidStaticVsPers", red(g("shiftwm", "static"), g("persistence", "static")))
        # Caption note for the region table: which seeds it uses, and whether it matches Table 1's checkpoints.
        runs = {}
        for k in r:
            if k.split("/")[0] in LEARNED:
                runs.setdefault(k.split("/")[0], []).append(k.split("/")[1])
        ckpts = [f for a_ in runs for f in (ROOTS[0] / "droid/dinov2s" / a_).glob("s*/best.pt")]
        stale = bool(ckpts) and f.stat().st_mtime < max(c.stat().st_mtime for c in ckpts)
        n = {len(v) for v in runs.values()}
        if n == {3} and not stale:
            M["regionsNote"] = r" Mean over 3 training seeds, as in \cref{tab:main}."
        else:
            seeds = "; ".join(f"{dict(ar='AR', ar_tf='AR-TF', direct='Direct', shiftwm=chr(92) + 'ours{}').get(a_, a_)} {', '.join(sorted(v))}" for a_, v in sorted(runs.items()))
            M["regionsNote"] = (r" Training seeds: " + seeds + r" (\cref{tab:main}: mean over all seeds)"
                                + (r", computed on earlier checkpoints of the same recipe" if stale else "") + ".")
    zs, ft, ours = vj.get("zeroshot"), vj.get("finetune"), vj.get("finetune_shiftwm")
    put("vjepaSkillZS", zs and zs["skill"]); put("vjepaSkillFT", ft and ft["skill"]); put("vjepaSkillOurs", ours and ours["skill"])
    put("vjepaSkillGain", ours["skill"] - ft["skill"] if ours and ft else None)
    put("vjepaMSERed", red(ours and ours["mse"], ft and ft["mse"])); put("vjepaMovingRed", red(ours and ours["moving"], ft and ft["moving"]))
    fw = RES / "analysis/flowwarp/summary.json"
    if fw.exists():
        F = json.loads(fw.read_text())["methods"]; sw = mean_mse("droid", "shiftwm")
        put("flowVsWarp", red(sw, F["flow_extrap_bwd"]["mean_h"])); put("flowVsOracle", red(sw, F["oracle_flow"]["mean_h"]))
        put("flowWarpVsPers", red(F["flow_extrap_bwd"]["mean_h"], F["persistence"]["mean_h"]))
        put("flowPers", F["persistence"]["mean_h"], "%.3f"); put("flowExtrap", F["flow_extrap_bwd"]["mean_h"], "%.3f")
        put("flowOracle", F["oracle_flow"]["mean_h"], "%.3f"); put("droidSWmse", sw, "%.3f")
    fi = RES / "analysis/interpret/summary.json"
    I = json.loads(fi.read_text()) if fi.exists() else {}
    put("koMoving", I.get("knockout_increase_moving")); put("koStatic", I.get("knockout_increase_static"))
    put("koHighGate", I.get("knockout_increase_highgate")); put("steerRatio", I.get("steer_ratio_moving_over_static"))
    dec = I.get("gain_vs_direct_by_decile")
    put("decileMin", min(dec) if dec else None); put("decileMax", max(dec) if dec else None)
    for env in ("pusht", "wall"):
        b, o = dw.get(env, {}).get("dinowm"), dw.get(env, {}).get("dinowm_shiftwm")
        E = env.capitalize()
        put(f"dinowm{E}ErrRed", red(o and o["err"], b and b["err"]))
        r_ = red(o and o["err"], b and b["err"]); put(f"dinowm{E}ErrInc", -r_ if isinstance(r_, (int, float)) else r_)
        put(f"dinowm{E}SuccBase", b and b["succ"]); put(f"dinowm{E}SuccOurs", o and o["succ"])
    return "\n".join(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n"


def write_plugin_outputs():
    vr, vj = vjepa_rows(); dr, dw = dinowm_rows()
    (GEN / "vjepa_rows.tex").write_text(vr + "\n"); (GEN / "dinowm_rows.tex").write_text(dr + "\n")
    (GEN / "numbers.tex").write_text(numbers_macros(vj, dw))


if __name__ == "__main__":
    write_plugin_outputs()


# ----------------------------------------------------------------------------- planning table
def planning_rows():
    """Success (%) per env, mean over planner seeds 42/43/44 (training seed 0), from results/v2/planning."""
    arms = [("random", "Random actions (floor)"),
            ("lewm", "LeWM (released) \\citep{maes2026lewm}"), ("v2_ar_tf_s0", "AR-TF (DINO-WM-style)"),
            ("v2_ar_s0", "AR (rollout-trained)"), ("v2_direct_s0", "Direct"), ("v2_shiftwm_s0", r"\ours{}"),
            ("v2_shiftwm_ctr_s0", r"\ours{} + action-contrastive")]
    rows = []
    for key, label in arms:
        cells = []
        for env in ("pusht", "tworoom", "reacher"):
            fs = [f for f in (RES / "planning" / env / key).glob("4[234].json")]
            v = [json.loads(f.read_text())["success_rate"] for f in fs]
            cells.append(f"{np.mean(v):.1f}" if v else PEND)
        ts = [json.loads(f.read_text()).get("timing", {}).get("sec_per_plan_per_env_mean") for f in (RES / "planning/pusht" / key).glob("4[234].json")]
        ts = [t for t in ts if t]
        cells.append("--" if key == "random" else (f"{np.mean(ts):.2f}" if ts else PEND))
        pre = r"\rowcolor{bestbg}" if key == "v2_shiftwm_ctr_s0" else ""
        rows.append(f"{pre}{label} & " + " & ".join(cells) + r" \\")
        if key == "lewm":
            rows.append(r"\midrule")
    return "\n".join(rows) + "\n"


# ----------------------------------------------------------------------------- green highlighting of our wins
import re as _re

REFERENCE = ("Persistence", "Linear", "Decoder on true", "True future", "upper bound")


def _num(cell):
    if "pend" in cell:
        return None
    m = _re.findall(r"-?\d+\.\d+|-?\d+", _re.sub(r"\\[a-zA-Z]+|\$\^\\dagger\$", " ", cell).replace("\\%", ""))
    return float(m[0]) if m else None


def _wrap(cell):
    core = cell.strip()
    if core.startswith("\\good{") or "pend" in core:
        return cell
    core = _re.sub(r"^\\textbf\{(.*)\}$", r"\1", core)
    return " \\good{" + core + "} "


def highlight_rows(text, directions, ours_key=r"\ours", first_col=1, groups=None):
    """Wrap our cells in \\good{} where ours beats every non-reference competitor in that column.
    directions: list of 'min'/'max' per numeric column (starting at `first_col`); groups: row -> group label."""
    lines = text.rstrip("\n").split("\n")
    rows = [(i, l) for i, l in enumerate(lines) if "&" in l and not l.lstrip().startswith("\\midrule")]
    parsed = {i: [c for c in l.rstrip().rstrip("\\").split("&")] for i, l in rows}
    for i, l in rows:
        if ours_key not in l:
            continue
        cells = parsed[i]
        for j, d in enumerate(directions):
            c = first_col + j
            if c >= len(cells):
                break
            mine = _num(cells[c])
            if mine is None:
                continue
            rivals = []
            for i2, l2 in rows:
                if i2 == i or ours_key in l2 or any(k in l2 for k in REFERENCE) or "error reduction" in l2:
                    continue
                if groups and groups(l2) != groups(l):
                    continue
                v = _num(parsed[i2][c]) if c < len(parsed[i2]) else None
                if v is not None:
                    rivals.append(v)
            if rivals and ((d == "min" and mine < min(rivals)) or (d == "max" and mine > max(rivals))):
                cells[c] = _wrap(cells[c])
        lines[i] = "&".join(cells) + " \\\\"
    return "\n".join(lines) + "\n"


def highlight_recipe(text):
    """Recipe table: rows are recipes, the last column is ours; green where ours is the lowest in its row."""
    out = []
    for l in text.rstrip("\n").split("\n"):
        cells = l.rstrip().rstrip("\\").split("&")
        vals = [_num(c) for c in cells[1:]]
        if vals and vals[-1] is not None and all(v is None or vals[-1] < v for v in vals[:-1]) and any(v is not None for v in vals[:-1]):
            cells[-1] = _wrap(cells[-1])
        out.append("&".join(cells) + " \\\\")
    return "\n".join(out) + "\n"


PLAN_EXTRA_ARMS = [("random", "Random actions"), ("lewm", "LeWM (released)"), ("v2_ar_tf_s0", "AR-TF"),
                   ("v2_ar_s0", "AR"), ("v2_direct_s0", "Direct"), ("v2_shiftwm_s0", r"\ours{}"),
                   ("v2_shiftwm_ctr_s0", r"\ours{} + contrastive")]


def planning_extra_rows():
    """Supplementary planning metrics over the 150 episodes (planner seeds 42/43/44) of tab:planning:
    steps to first success, censored at the 50-step budget (mean), and terminal physical goal error (median).
    Success steps come from <seed>.json; the terminal error from <seed>.json if it was recorded there,
    else from the re-run <seed>_phys.json (same checkpoint, tasks and planner seed)."""
    rows = []
    for key, label in PLAN_EXTRA_ARMS:
        cells = []
        for env in ("pusht", "tworoom", "reacher"):
            d = RES / "planning" / env / key
            steps, errs = [], []
            for seed in (42, 43, 44):
                f = d / f"{seed}.json"
                if not f.exists():
                    steps = errs = None
                    break
                r = json.loads(f.read_text()); budget = r["protocol"]["eval"]["eval_budget"]
                steps += [e["success_step"] if e["success_step"] else budget for e in r["episodes"]]
                ep = r["episodes"]
                if any(e.get("terminal_goal_error") is None for e in ep) and (d / f"{seed}_phys.json").exists():
                    ep = json.loads((d / f"{seed}_phys.json").read_text())["episodes"]
                te = [e.get("terminal_goal_error") for e in ep]
                errs = None if errs is None or any(v is None for v in te) else errs + te
            cells.append(f"{np.mean(steps):.1f}" if steps else PEND)
            cells.append((f"{np.median(errs):.3f}" if env == "reacher" else f"{np.median(errs):.1f}") if errs else PEND)
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        if key == "lewm":
            rows.append(r"\midrule")
    return "\n".join(rows) + "\n"


def apply_highlights():
    (GEN / "planning_rows.tex").write_text(planning_rows())
    (GEN / "planning_extra_rows.tex").write_text(planning_extra_rows())
    specs = {"planning_rows.tex": ["max", "max", "max"], "per_horizon_rows.tex": ["min"] * 10, "hamlyn_task_rows.tex": ["min"] * 7,
             "external_rows.tex": (None, 2, ["max"]),
             "dinowm_rows.tex": (lambda l: l.split("&")[0].replace("\\rowcolor{bestbg}", "").strip(), 2, ["min", "max", "min", "max"]),
             "pixel_rows.tex": ["max", "max", "min", "max", "max", "min"], "probe_rows.tex": ["min", "max", "min", "max"]}
    for f, spec in specs.items():
        p = GEN / f
        if not p.exists():
            continue
        if isinstance(spec, tuple):
            groups, first, dirs = spec
            p.write_text(highlight_rows(p.read_text(), dirs, first_col=first, groups=groups))
        else:
            p.write_text(highlight_rows(p.read_text(), spec))
    if (GEN / "recipe_rows.tex").exists():
        (GEN / "recipe_rows.tex").write_text(highlight_recipe((GEN / "recipe_rows.tex").read_text()))


if __name__ == "__main__":
    apply_highlights()
