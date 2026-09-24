"""Language-Table split check: do test segments continue directly from training segments?

Open-X Language-Table episodes are hindsight-labelled segments of longer play; the release has no source-episode id,
so our split (salted hash of the segment id, scripts/v2/prepare_openx.py) is by segment. For every test segment we find
the training segment whose first frame is closest to the test segment's last frame, or whose last frame is closest to
the test segment's first frame (mean abs. RGB difference on a 56x56 average-pooled image), and compare it with the
typical one-step frame change inside an episode. Also checks whether RLDS-adjacent records (same shard, index i, i+1)
are temporally contiguous (end-effector position and frame difference vs. random pairs).
Writes results/v2/analysis/lt_split_check.json. CPU only.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/v2/frames/language_table"
m = json.loads((D / "manifest.json").read_text())
eps = m["episodes"]


def small(x):
    return x.reshape(56, 4, 56, 4, 3).mean((1, 3)).astype(np.float32)


F, L, P0, P1, step = {}, {}, {}, {}, []
for e in eps:
    with np.load(D / e["file"]) as z:
        im, pr = z["images"], z["proprio"]
        F[e["id"]], L[e["id"]] = small(im[0]).ravel(), small(im[-1]).ravel()
        P0[e["id"]], P1[e["id"]] = pr[0, :2], pr[-1, :2]
        step.append(np.abs(small(im[1]) - small(im[0])).mean())
tr = [e["id"] for e in eps if e["split"] == "train"]
te = [e["id"] for e in eps if e["split"] == "test"]
TF, TL = np.stack([F[i] for i in tr]), np.stack([L[i] for i in tr])
near = np.array([min(np.abs(TF - L[i]).mean(1).min(), np.abs(TL - F[i]).mean(1).min()) for i in te])
one_step = float(np.median(step))
# RLDS-adjacent records
ids = {e["id"] for e in eps}
pairs = []
for k in sorted(ids):
    shard, idx = k.split("__train")[1].split("_")
    nk = f"language_table__train{shard}_{int(idx) + 1:03d}"
    if nk in ids:
        pairs.append((k, nk))
rng = np.random.default_rng(0)
rand = [tuple(rng.choice(sorted(ids), 2, replace=False)) for _ in range(len(pairs))]
eff = lambda ps: [float(np.abs(P1[a] - P0[b]).max()) for a, b in ps]
img = lambda ps: [float(np.abs(L[a] - F[b]).mean()) for a, b in ps]
out = {"n_train": len(tr), "n_test": len(te), "one_step_frame_diff_median": one_step,
       "test_nearest_train_boundary_diff_percentiles_5_25_50_75": np.percentile(near, [5, 25, 50, 75]).tolist(),
       "test_segments_with_train_boundary_closer_than_one_step": int((near < one_step).sum()),
       "rlds_adjacent_pairs": len(pairs),
       "adjacent_effector_gap_median": float(np.median(eff(pairs))), "random_effector_gap_median": float(np.median(eff(rand))),
       "adjacent_frame_diff_median": float(np.median(img(pairs))), "random_frame_diff_median": float(np.median(img(rand)))}
f = ROOT / "results/v2/analysis/lt_split_check.json"
f.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
