"""Segmentation view of DINOv2 forecasts: does each method put the moving robot arm / tool / object in the right place?

Usage (GPU):  PYTHONPATH=src python scripts/v2/segments.py --dataset droid [--stride 2]
Outputs:      results/v2/analysis/segments/<dataset>/{summary.json, examples.npz, windows.npz}
Datasets without trained ShiftWM + Direct checkpoints are skipped with status "pending" in summary.json.

Pipeline (every rule is fixed in advance; nothing is hand-picked):
  1. Reference masks (pixel space), per held-out window (test split; the planning suites have no test split -> val),
     stride 2, all episodes (if a split has more than MAX_WINDOWS=1500 windows, an evenly spaced subset is used).
     Real video: Grounding DINO (IDEA-Research/grounding-dino-base) detects the dataset's text prompt (DATASETS below)
     on the observed frame t (last history frame); the highest-scoring box (up to two non-overlapping boxes for the
     two Hamlyn instruments) prompts SAM 2.1 (facebook/sam2.1-hiera-large, HF transformers video predictor), which
     segments frame t and tracks the mask through the true future frames t+1..t+K. Frames are segmented at their
     native aspect ratio. Each mask keeps its largest connected component plus components >= 25% of its area.
     Rendered simulators (plan_*): an exact colour threshold of the rendered agent/object (colour_mask), because a
     text prompt is unreliable on these renders.
     Masks are area-averaged onto the 16x16 DINOv2 patch grid (the encoder squashes the full frame to 224x224); a
     patch is foreground if coverage >= cov (lower for thin structures).
     QC: a window is kept only if the detection score >= 0.3, 3..128 foreground patches at t (>= 1 for the small
     sim agents), the track is never lost, and the area never changes > 3x between consecutive frames. Rejections
     are counted per reason.
  2. Forecast segmentation (feature space). Labels come from the observed frame only: every patch of a forecast
     Z_hat_k is labelled by 1-nearest-neighbour (cosine) transfer from the 256 standardised DINOv2 patches of frame t
     and the reference mask at t ("nn": foreground if its most similar foreground patch at t is more similar than its
     most similar background patch). nn was fixed in the DROID pilot as the labeller with the highest IoU on the TRUE
     future features (oracle) -- a criterion that involves no forecast; prototype labellers are reported too.
  3. IoU of the predicted mask with the tracked reference mask of the true future frame, per method (ShiftWM, Direct,
     AR, persistence, oracle = true future features), per horizon and averaged over horizons; 95% bootstrap CIs over
     episodes and paired ShiftWM-minus-baseline CIs. Also on "moving" windows (IoU(M_t, M_t+K) below the median).
  4. Placement: distance (display px and patches) between the centroid of the cells a forecast labels foreground and
     the coverage-weighted centroid of the tracked SAM mask at t+k (summary.json["placement"]; paired, windows where a
     compared method labels no cell are dropped at that k).
     Example windows (illustrative, not representative): among QC-passing moving windows with ShiftWM's k=K placement
     error below its median (and oracle error below its median), the two (distinct episodes) with the largest
     advantage min(Direct, AR) - ShiftWM.
     --stage placement / examples recompute from cached masks without changing the IoU entries of summary.json.
  Checkpoints: results/v2s/<ds>/dinov2s/<arm>/s* (final recipe) where present, else results/v2 (recorded).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from shiftwm.v2.models import V2WorldModel
from shiftwm.v2.train import FeatureSplit

ROOT = Path(__file__).resolve().parents[2]
RUNS = [ROOT / "results/v2s", ROOT / "results/v2"]   # final checkpoints first; results/v2 only for arms missing in v2s
OUT0 = ROOT / "results/v2/analysis/segments"
G = 16
SAM_ID, DET_ID = "facebook/sam2.1-hiera-large", "IDEA-Research/grounding-dino-base"
MIN_SCORE, MAX_JUMP, MIN_KEEP, MAX_WINDOWS = 0.3, 3.0, 0.25, 1500
ARMS = ("shiftwm", "direct", "ar")
REQUIRED = ("shiftwm", "direct")
LABELLERS = ("proto", "multi", "nn")
PRIMARY = "nn"          # fixed before the final runs (best oracle IoU in the DROID pilot); not tuned per method
TAU = 0.05              # soft foreground probability sigmoid((max cos fg - max cos bg) / TAU), for the figures only

# target name, text prompt (None -> colour mask), max objects, display aspect W/H (None = native frames), coverage
# threshold for a foreground patch, min foreground patches at t
DATASETS = {
    "droid": dict(target="robot arm", prompt="a robot arm.", objs=1, aspect=None, cov=0.5, min_fg=3),
    "openh_hamlyn": dict(target="instruments", prompt="a surgical instrument.", objs=2, aspect=848 / 480, cov=0.5, min_fg=3),
    "bridge": dict(target="robot arm", prompt="a robot arm.", objs=1, aspect=4 / 3, cov=0.5, min_fg=3),
    "fractal": dict(target="robot arm", prompt="a robot arm.", objs=1, aspect=1.25, cov=0.5, min_fg=3),
    "iws_pusht": dict(target="T block", prompt="a pink T-shaped block.", objs=1, aspect=4 / 3, cov=0.5, min_fg=3),
    "iws_box": dict(target="box", prompt="a box.", objs=1, aspect=4 / 3, cov=0.5, min_fg=3),
    "iws_rope": dict(target="rope", prompt="a rope.", objs=1, aspect=4 / 3, cov=0.3, min_fg=3),
    "plan_pusht": dict(target="T block + agent", prompt=None, objs=1, aspect=1.0, cov=0.5, min_fg=3),
    "plan_reacher": dict(target="arm", prompt=None, objs=1, aspect=1.0, cov=0.2, min_fg=1),
    "plan_tworoom": dict(target="agent", prompt=None, objs=1, aspect=1.0, cov=0.3, min_fg=1),
}
REASONS = ("ok", "no_detection", "fg_area_at_t", "lost_track", "area_jump")


def colour_mask(ds, img):
    """Exact masks for the rendered planning suites (uint8 [...,H,W,3] -> bool)."""
    x = img.astype(np.int16); r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    if ds == "plan_pusht":      # grey T block (low saturation, mid grey) + blue agent; the green goal T is excluded
        block = (mx - mn < 25) & (mx > 90) & (mx < 190)
        agent = (b > 150) & (b - r > 60) & (b - g > 30)
        return block | agent
    if ds == "plan_reacher":    # yellow/orange arm links on a blue background
        return (r > 170) & (g > 110) & (b < 150) & (r - b > 60)
    if ds == "plan_tworoom":    # red agent
        return (r > 150) & (g < 140) & (b < 140) & (r - g > 60)
    raise KeyError(ds)


def checkpoints(ds):
    out = {}
    for arm in ARMS:
        for base in RUNS:
            cks = sorted((base / ds / "dinov2s" / arm).glob("s*/best.pt"))
            if cks:
                out[arm] = cks[0]; break
    return out


def load(ck, dev):
    st = torch.load(ck, map_location=dev)
    m = V2WorldModel(st["config"]).to(dev).eval(); m.load_state_dict(st["model"])
    return m


def frames_for(ds, eid, aspect):
    """uint8 frames of one episode at native aspect ratio."""
    if ds == "droid":
        root = ROOT / "data/real_video/droid_selected/processed"
        man = json.loads((root / "manifest.json").read_text())
        row = next(r for r in man["episodes"] if r["episode_id"] == eid)
        with np.load(root / row["cameras"]["exterior_image_1_left"]["file"]) as z:
            return z["images"]
    with np.load(ROOT / "data/v2/frames" / ds / "episodes" / f"{eid}.npz") as z:
        imgs = z["images"]
    if aspect is None or abs(aspect - 1) < 1e-6:
        return imgs
    h = imgs.shape[1]; w = int(round(h * aspect / 2)) * 2
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 255)
    return t.round().byte().permute(0, 2, 3, 1).numpy()


def to_grid(mask):
    """[..., Hpx, Wpx] bool -> [..., 16, 16] area coverage."""
    m = torch.as_tensor(mask, dtype=torch.float32)
    lead = m.shape[:-2]
    m = m.reshape(-1, 1, *m.shape[-2:])
    up = F.interpolate(m, size=(G * 45, G * 40), mode="nearest")          # area average on a fine lattice
    return F.avg_pool2d(up, (45, 40)).reshape(*lead, G, G).numpy()


def clean(m):
    """Keep the largest connected component and any component with >= MIN_KEEP of its area (drops SAM specks)."""
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    area = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(area >= MIN_KEEP * area.max())[0])


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(u, 1e-6)


class Segmenter:
    def __init__(self, dev, prompt, objs):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2VideoModel, Sam2VideoProcessor
        self.dev, self.prompt, self.objs = dev, prompt, objs
        self.dp = AutoProcessor.from_pretrained(DET_ID)
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained(DET_ID).to(dev).eval()
        self.sp = Sam2VideoProcessor.from_pretrained(SAM_ID)
        self.sm = Sam2VideoModel.from_pretrained(SAM_ID).to(dev, dtype=torch.bfloat16).eval()

    @torch.no_grad()
    def detect(self, imgs, bs=16):
        """Top box(es) for the prompt per image -> (list of [objs x 4] boxes, score of the top box per image)."""
        boxes, scores = [], []
        for i in range(0, len(imgs), bs):
            im = list(imgs[i:i + bs])
            inp = self.dp(images=im, text=[self.prompt] * len(im), return_tensors="pt").to(self.dev)
            res = self.dp.post_process_grounded_object_detection(self.dm(**inp), inp.input_ids, threshold=0.0,
                                                                 text_threshold=0.0, target_sizes=[imgs.shape[1:3]] * len(im))
            for r in res:
                if len(r["scores"]) == 0:
                    boxes.append([[0.0, 0.0, 1.0, 1.0]]); scores.append(0.0); continue
                order = r["scores"].argsort(descending=True).tolist()
                chosen = [order[0]]
                for j in order[1:]:
                    if len(chosen) == self.objs:
                        break
                    if float(r["scores"][j]) >= MIN_SCORE and all(
                            box_iou(r["boxes"][j].tolist(), r["boxes"][c].tolist()) < 0.3 for c in chosen):
                        chosen.append(j)
                boxes.append([r["boxes"][j].tolist() for j in chosen]); scores.append(float(r["scores"][order[0]]))
        return boxes, np.array(scores)

    @torch.no_grad()
    def track(self, frames, boxes):
        """frames [T,H,W,3] uint8; box prompt(s) on frame 0; returns bool union masks [T,H,W]."""
        sess = self.sp.init_video_session(video=list(frames), inference_device=self.dev, processing_device=self.dev,
                                           video_storage_device=self.dev, dtype=torch.bfloat16)
        ids = list(range(1, len(boxes) + 1))
        self.sp.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=ids,
                                                input_boxes=[[list(map(float, b)) for b in boxes]])
        out = np.zeros(frames.shape[:3], bool)
        size = [[frames.shape[1], frames.shape[2]]]

        def post(o):
            return self.sp.post_process_masks([o.pred_masks], original_sizes=size, binarize=True)[0][:, 0].any(0).cpu().numpy()
        out[0] = post(self.sm(inference_session=sess, frame_idx=0))
        for o in self.sm.propagate_in_video_iterator(sess, start_frame_idx=0):
            out[o.frame_idx] = post(o)
        return np.stack([clean(m) for m in out])


def qc(areas_px, cov_t, cfg):
    n_t = int((cov_t >= cfg["cov"]).sum())
    if n_t < cfg["min_fg"] or n_t > 128:
        return 2
    if (areas_px[1:] == 0).any():
        return 3
    a = np.maximum(areas_px, 1).astype(float)
    if (np.maximum(a[1:] / a[:-1], a[:-1] / a[1:]) > MAX_JUMP).any():
        return 4
    return 0


def kmeans(x, k, iters=20):
    """Cosine k-means on rows of x [n,C] (deterministic farthest-point init)."""
    x = F.normalize(x, dim=-1)
    k = min(k, len(x))
    c = [x[(x @ F.normalize(x.mean(0), dim=0)).argmax()]]
    for _ in range(k - 1):
        c.append(x[(x @ torch.stack(c).T).max(1).values.argmin()])
    c = torch.stack(c)
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        c = torch.stack([F.normalize(x[a == j].mean(0), dim=0) if (a == j).any() else c[j] for j in range(k)])
    return c


def label(obs, cov, preds, thr):
    """obs [N,C] observed features, cov [N] coverage at t, preds [M,N,C] -> name -> score [M,N]; foreground if > 0."""
    fg, bg = cov >= thr, cov < 0.1
    o = F.normalize(obs, dim=-1); p = F.normalize(preds, dim=-1)
    pa, pb = F.normalize(obs[fg].mean(0), dim=0), F.normalize(obs[bg].mean(0), dim=0)
    s = {"proto": p @ pa - p @ pb}
    ca, cb = kmeans(obs[fg], 2), kmeans(obs[bg], 6)
    s["multi"] = (p @ ca.T).max(-1).values - (p @ cb.T).max(-1).values
    sim = p @ o.T                                                       # [M,N,N]
    s["nn"] = sim[..., fg].max(-1).values - sim[..., bg].max(-1).values
    return s


def iou(a, b):
    inter = (a & b).sum(-1); uni = (a | b).sum(-1)
    return np.where(uni > 0, inter / np.maximum(uni, 1), 1.0)


def boot_ci(per_ep, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.stack([per_ep[rng.integers(0, len(per_ep), len(per_ep))].mean(0) for _ in range(n)])
    return np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


def offsets(model, weights):
    c = model.config; r = c.window // 2
    oy, ox = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
    oy = oy.flatten().repeat(c.sources).to(weights); ox = ox.flatten().repeat(c.sources).to(weights)
    return torch.stack((weights @ ox, weights @ oy), -1)


def disp_size(ds, aspect):
    """(h, w) in pixels of the displayed (native-aspect) frame."""
    if ds == "droid":
        return 180, 320
    return 224, (224 if aspect is None or abs(aspect - 1) < 1e-6 else int(round(224 * aspect / 2)) * 2)


def centroids(lab, cov_k, h, w):
    """lab [M,K,N] bool predicted cells, cov_k [K,N] true coverage -> predicted [M,K,2], true [K,2] (x,y) in display px.
    Predicted centroid = mean of labelled cell centres (nan if none); true = coverage-weighted cell centres."""
    gy, gx = np.divmod(np.arange(G * G), G)
    cx, cy = (gx + 0.5) * w / G, (gy + 0.5) * h / G
    cnt = lab.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.stack(((lab * cx).sum(-1) / cnt, (lab * cy).sum(-1) / cnt), -1)
        wsum = cov_k.sum(-1)
        true = np.stack(((cov_k * cx).sum(-1) / wsum, (cov_k * cy).sum(-1) / wsum), -1)
    return pred, true


def boot_ci_nan(per_ep, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.stack([np.nanmean(per_ep[rng.integers(0, len(per_ep), len(per_ep))], 0) for _ in range(n)])
    return np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)


def window_table(data):
    ep_of = data.episode_of.cpu().numpy(); starts = data.starts.cpu().numpy()
    first = {e: starts[ep_of == e].min() for e in np.unique(ep_of)}
    return ep_of, np.array([starts[j] - first[ep_of[j]] for j in range(len(starts))])


def stage_masks(ds, cfg, data, sel, dev, out, stride):
    H, K = data.history, data.horizon
    seg = Segmenter(dev, cfg["prompt"], cfg["objs"]) if cfg["prompt"] else None
    rec = {k: [] for k in ("widx", "cov", "score", "box", "areas", "reason")}
    ep_of, rel = window_table(data)
    for n_e, e in enumerate(np.unique(ep_of[sel])):
        imgs = frames_for(ds, data.episodes[e]["id"], cfg["aspect"])
        js = sel[ep_of[sel] == e]
        if seg is not None:
            boxes, scores = seg.detect(imgs[rel[js] + H - 1])
        else:
            boxes, scores = [[[0.0, 0.0, 0.0, 0.0]]] * len(js), np.ones(len(js))
        for j, box, sc in zip(js, boxes, scores):
            s = rel[j]
            rec["widx"].append(j); rec["score"].append(sc); rec["box"].append((list(box) + [[0.0] * 4] * 2)[:2])
            if sc < MIN_SCORE:
                rec["cov"].append(np.zeros((K + 1, G, G), np.float16)); rec["areas"].append(np.zeros(K + 1, int))
                rec["reason"].append(1); continue
            clip = imgs[s + H - 1: s + H + K]
            m = seg.track(clip, box) if seg is not None else np.stack([clean(colour_mask(ds, f)) for f in clip])
            cv = to_grid(m); areas = m.reshape(K + 1, -1).sum(-1)
            rec["cov"].append(cv.astype(np.float16)); rec["areas"].append(areas); rec["reason"].append(qc(areas, cv[0], cfg))
        rs = np.array(rec["reason"])
        if n_e % 10 == 0:
            print(f"[{ds}] episode {n_e + 1}: {len(rs)} windows, kept {(rs == 0).sum()} "
                  + str({REASONS[i]: int((rs == i).sum()) for i in range(1, len(REASONS))}), flush=True)
    W = {k: np.array(v) for k, v in rec.items()}
    np.savez(out / "windows.npz", stride=stride, **W)


@torch.no_grad()
def stage_eval(ds, cfg, data, W, cks, dev, out, split, examples_only=False):
    H, K = data.history, data.horizon
    arms = [a for a in ARMS if a in cks]
    models = {a: load(cks[a], dev) for a in arms}
    keep = W["reason"] == 0
    widx = W["widx"][keep]; cov = W["cov"][keep].astype(np.float32)          # [n, K+1, 16, 16]
    rejected = {REASONS[i]: int((W["reason"] == i).sum()) for i in range(1, len(REASONS))}
    thr = cfg["cov"]
    names = ("persistence",) + tuple(arms) + ("oracle",)
    ious = {l: {n: [] for n in names} for l in LABELLERS}
    hd, wd = disp_size(ds, cfg["aspect"])
    place = {n: [] for n in names}; place_patch = {n: [] for n in names}
    for i in range(0, len(widx), 64):
        idx = torch.as_tensor(widx[i:i + 64], device=dev)
        h, pa, f, t = data.batch(idx)
        preds = {"persistence": h[:, -1:].expand(-1, K, -1, -1), "oracle": t}
        for a in arms:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds[a] = models[a](h, pa, f).float()
        for b in range(len(idx)):
            c = torch.as_tensor(cov[i + b].reshape(K + 1, -1), device=dev)
            gt = (c[1:] >= thr).cpu().numpy()                             # [K,N]
            stack = torch.stack([preds[n][b] for n in names])             # [M,K,N,C]
            s = label(h[b, -1], c[0], stack.reshape(-1, *stack.shape[2:]), thr)
            for l in LABELLERS:
                lab = (s[l] > 0).reshape(len(names), K, -1).cpu().numpy()
                for m, n in enumerate(names):
                    ious[l][n].append(iou(lab[m], gt))
                if l == PRIMARY:
                    ck = c[1:].cpu().numpy()
                    pc, tc = centroids(lab, ck, hd, wd)
                    pg, tg = centroids(lab, ck, G, G)
                    for m, n in enumerate(names):
                        place[n].append(np.linalg.norm(pc[m] - tc, axis=-1))
                        place_patch[n].append(np.linalg.norm(pg[m] - tg, axis=-1))
    ep = data.episode_of[torch.as_tensor(widx, device=dev)].cpu().numpy()
    ious = {l: {n: np.stack(v) for n, v in d.items()} for l, d in ious.items()}   # [n, K]
    oracle = {l: float(ious[l]["oracle"][:, K - 1].mean()) for l in LABELLERS}
    fg0 = (cov[:, 0] >= thr).reshape(len(cov), -1); fgK = (cov[:, K] >= thr).reshape(len(cov), -1)
    motion = 1 - iou(fg0, fgK)
    moving = motion >= np.median(motion)

    place = {n: np.stack(v) for n, v in place.items()}                         # [n, K] display px (nan: empty)
    place_patch = {n: np.stack(v) for n, v in place_patch.items()}
    valid = np.all(np.stack([np.isfinite(place[n]) for n in names if n != "oracle"]), 0)   # all compared non-empty

    def summarise_place(P, sel):
        eps = np.unique(ep[sel])
        Pm = {n: np.where(valid, P[n], np.nan) for n in names}
        per_ep = {n: np.stack([np.nanmean(Pm[n][sel & (ep == e)], 0) if np.isfinite(Pm[n][sel & (ep == e)]).any()
                               else np.full(K, np.nan) for e in eps]) for n in names}
        avg = {n: np.nanmean(v, 1) for n, v in per_ep.items()}
        res = {}
        for n in names:
            lo, hi = boot_ci_nan(per_ep[n]); alo, ahi = boot_ci_nan(avg[n][:, None])
            res[n] = {"mean": np.nanmean(Pm[n][sel], 0).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                      "avg": {"mean": float(np.nanmean(Pm[n][sel])), "lo": float(alo[0]), "hi": float(ahi[0])}}
            if n != "shiftwm":
                d = per_ep["shiftwm"] - per_ep[n]; dlo, dhi = boot_ci_nan(d)
                res[n]["diff_sw_minus"] = {"mean": np.nanmean(d, 0).tolist(), "lo": dlo.tolist(), "hi": dhi.tolist()}
                da = (avg["shiftwm"] - avg[n])[:, None]; dlo, dhi = boot_ci_nan(da)
                res[n]["avg"]["diff_sw_minus"] = {"mean": float(np.nanmean(da)), "lo": float(dlo[0]), "hi": float(dhi[0])}
        res["windows"] = int(sel.sum()); res["episodes"] = int(len(eps))
        res["valid_fraction"] = np.mean(valid[sel], 0).tolist()
        res["empty_fraction"] = {n: np.mean(~np.isfinite(P[n][sel]), 0).tolist() for n in names}
        return res

    placement = {"unit_px": f"pixels of the displayed {wd}x{hd} frame", "display_hw": [hd, wd], "labeller": PRIMARY,
                 "definition": "distance between the centroid of the patch cells a forecast labels as foreground and the "
                               "coverage-weighted centroid of the tracked SAM mask at t+k; windows where any compared "
                               "method labels no cell are excluded at that k (paired)",
                 "px": {"all": summarise_place(place, np.ones(len(widx), bool)), "moving": summarise_place(place, moving)},
                 "patches": {"all": summarise_place(place_patch, np.ones(len(widx), bool)),
                             "moving": summarise_place(place_patch, moving)}}

    def summarise(I, sel):
        eps = np.unique(ep[sel])
        per_ep = {n: np.stack([I[n][sel & (ep == e)].mean(0) for e in eps]) for n in names}
        avg = {n: v.mean(1) for n, v in per_ep.items()}                       # horizon-averaged, per episode
        res = {}
        for n in names:
            lo, hi = boot_ci(per_ep[n])
            res[n] = {"mean": I[n][sel].mean(0).tolist(), "ep_mean": per_ep[n].mean(0).tolist(),
                      "lo": lo.tolist(), "hi": hi.tolist()}
            alo, ahi = boot_ci(avg[n][:, None])
            res[n]["avg"] = {"mean": float(I[n][sel].mean()), "ep_mean": float(avg[n].mean()),
                             "lo": float(alo[0]), "hi": float(ahi[0])}
            if n != "shiftwm":
                d = per_ep["shiftwm"] - per_ep[n]; dlo, dhi = boot_ci(d)
                res[n]["diff_sw_minus"] = {"mean": d.mean(0).tolist(), "lo": dlo.tolist(), "hi": dhi.tolist()}
                da = (avg["shiftwm"] - avg[n])[:, None]; dlo, dhi = boot_ci(da)
                res[n]["avg"]["diff_sw_minus"] = {"mean": float(da.mean()), "lo": float(dlo[0]), "hi": float(dhi[0])}
        res["windows"] = int(sel.sum()); res["episodes"] = int(len(eps))
        return res

    allw = np.ones(len(widx), bool)
    summ = {"dataset": ds, "status": "done", "split": split, "target": cfg["target"],
            "segmenter": (f"{SAM_ID} video tracking from frame t, prompted by the top {DET_ID} box(es) for "
                          f"'{cfg['prompt']}' (max {cfg['objs']} objects)") if cfg["prompt"] else
                         "exact colour threshold of the rendered target (text prompt unreliable on these renders)",
            "checkpoints": {a: str(cks[a].relative_to(ROOT)) for a in arms},
            "checkpoint_mtime": {a: cks[a].stat().st_mtime for a in arms},
            "primary_labeller": PRIMARY, "oracle_iou_k_last_by_labeller": oracle, "horizon": K, "history": H,
            "coverage_threshold": thr, "window_stride": int(W["stride"]),
            "windows_total": int(len(data)), "windows_considered": int(len(W["widx"])), "windows_kept": int(len(widx)),
            "windows_rejected": rejected,
            "qc": {"min_detection_score": MIN_SCORE, "fg_patches_at_t": [cfg["min_fg"], 128], "max_area_jump": MAX_JUMP,
                   "component_keep_fraction": MIN_KEEP},
            "median_motion": float(np.median(motion)),
            "results": {l: {"all": summarise(ious[l], allw), "moving": summarise(ious[l], moving)} for l in LABELLERS}}
    np.savez_compressed(out / "perwindow.npz", widx=widx, episode=ep, motion=motion, moving=moving, names=np.array(names),
                        **{f"iou_{n}": ious[PRIMARY][n] for n in names}, **{f"place_{n}": place[n] for n in names})
    if examples_only:           # keep the published IoU aggregates untouched; add/refresh the placement block only
        old = json.loads((out / "summary.json").read_text())
        assert old["windows_kept"] == summ["windows_kept"]
        old["placement"] = placement
        summ = old
    else:
        summ["placement"] = placement
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    pm = placement["px"]["moving"]
    print(f"[{ds}] placement error (px, moving):", {n: round(pm[n]["avg"]["mean"], 2) for n in names},
          {n: [round(x, 2) for x in (pm[n]["avg"]["diff_sw_minus"]["mean"], pm[n]["avg"]["diff_sw_minus"]["lo"],
                                     pm[n]["avg"]["diff_sw_minus"]["hi"])] for n in names if n != "shiftwm"})
    for sub in ("all", "moving"):
        r = summ["results"][PRIMARY][sub]
        print(f"[{ds}] {sub}: {r['windows']} windows / {r['episodes']} episodes")
        for n in names:
            d = r[n]["avg"].get("diff_sw_minus")
            print(f"  {n:12s}", " ".join(f"k{k}={r[n]['mean'][k - 1]:.3f}" for k in (1, 5, K)),
                  f"avg={r[n]['avg']['mean']:.3f}", f"SW-this avg={d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]" if d else "")
    print(f"[{ds}] rejected {rejected}; kept {len(widx)} of {len(W['widx'])} considered ({len(data)} total)")

    # examples (illustrative, NOT representative; averages are in summary.json): among QC-passing moving windows where
    # ShiftWM's k=K placement error is below its median on moving windows and the labeller applied to the TRUE future
    # features is at least median-accurate (method-independent sanity check), the windows with the largest placement
    # advantage min(err_Direct, err_AR) - err_ShiftWM at k=K, at most one per episode
    I = ious[PRIMARY]
    e_sw = place["shiftwm"][:, K - 1]
    rival = np.min(np.stack([place[a][:, K - 1] for a in arms if a != "shiftwm"]), 0)
    ok = moving & valid[:, K - 1]
    med = np.median(e_sw[ok])
    e_or = place["oracle"][:, K - 1]            # sanity (method-independent): the labeller localises the TRUE features
    ok &= np.isfinite(e_or) & (e_or <= np.nanmedian(e_or[ok]))
    adv = np.where(ok & (e_sw <= med), rival - e_sw, -np.inf)
    ex, seen = [], set()
    for w in np.argsort(-adv, kind="stable"):
        if not np.isfinite(adv[w]) or len(ex) == 2:
            break
        if ep[w] not in seen:
            ex.append(int(w)); seen.add(ep[w])
    print(f"[{ds}] examples (largest placement advantage; k=K error px / IoU):",
          [(data.episodes[ep[w]]["id"], {n: (round(float(place[n][w, K - 1]), 1), round(float(I[n][w, K - 1]), 3))
                                         for n in names}) for w in ex])
    idx = torch.as_tensor(widx[ex], device=dev)
    h, pa, f, t = data.batch(idx)
    preds = {"persistence": h[:, -1:].expand(-1, K, -1, -1), "oracle": t}
    for a in arms:
        preds[a] = models[a](h, pa, f).float()
    _, det = models["shiftwm"](h, pa, f, return_details=True)
    off = offsets(models["shiftwm"], det["weights"]).cpu().numpy()          # [B,K,N,2] (dx,dy) expected source offset
    gate = det["gate"][..., 0].cpu().numpy()
    ep_of, rel = window_table(data)
    seg = Segmenter(dev, cfg["prompt"], cfg["objs"]) if cfg["prompt"] else None
    E = {"episode": [], "start": [], "frames": [], "masks": [], "score": [], "iou": [], "motion": []}
    for b, w in enumerate(ex):
        j = int(widx[w]); e = ep_of[j]; s = int(rel[j])
        imgs = frames_for(ds, data.episodes[e]["id"], cfg["aspect"])
        clip = imgs[s + H - 1:s + H + K]
        if seg is not None:
            masks = seg.track(clip, [bb for bb in W["box"][keep][w].tolist() if bb[2] > bb[0]])
        else:
            masks = np.stack([clean(colour_mask(ds, fr)) for fr in clip])
        c = torch.as_tensor(cov[w].reshape(K + 1, -1), device=dev)
        stack = torch.stack([preds[n][b] for n in names])
        sc = label(h[b, -1], c[0], stack.reshape(-1, *stack.shape[2:]), thr)[PRIMARY].reshape(len(names), K, -1)
        E["episode"].append(data.episodes[e]["id"]); E["start"].append(s)
        E["frames"].append(imgs[s:s + H + K]); E["masks"].append(masks); E["score"].append(sc.cpu().numpy())
        E["iou"].append(np.stack([ious[PRIMARY][n][w] for n in names])); E["motion"].append(motion[w])
    np.savez_compressed(out / "examples.npz", names=np.array(names), history=H, offsets=off, gate=gate, tau=TAU,
                        advantage=np.array([adv[w] for w in ex]), rule="largest_placement_advantage", place=np.array([[place[n][w] for n in names] for w in ex]),
                        display_hw=np.array([hd, wd]), dataset=ds, target=cfg["target"],
                        **{k: np.array(v) for k, v in E.items()})
    print(f"[{ds}] wrote {out}")


class ImageSegmenter:
    """Grounding DINO box(es) + SAM 2.1 image predictor on single images (used on DECODED forecasts)."""

    def __init__(self, dev, prompt, objs):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.dev, self.prompt, self.objs = dev, prompt, objs
        self.dp = AutoProcessor.from_pretrained(DET_ID)
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained(DET_ID).to(dev).eval()
        self.sp = Sam2Processor.from_pretrained(SAM_ID)
        self.sm = Sam2Model.from_pretrained(SAM_ID).to(dev).eval()
        self.det = Segmenter.detect.__get__(self)

    @torch.no_grad()
    def segment(self, imgs, bs=8):
        """uint8 [n,H,W,3] -> bool masks [n,H,W] (empty where the detection score < MIN_SCORE), scores [n]."""
        boxes, scores = self.det(imgs)
        out = np.zeros(imgs.shape[:3], bool)
        for i in range(0, len(imgs), bs):
            sl = range(i, min(i + bs, len(imgs)))
            bx = [(list(boxes[j]) * self.objs)[:self.objs] for j in sl]      # pad by repeating (union unaffected)
            inp = self.sp(images=[imgs[j] for j in sl], input_boxes=bx, return_tensors="pt").to(self.dev)
            o = self.sm(**inp, multimask_output=False)
            ms = self.sp.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])
            for j, m in zip(sl, ms):
                if scores[j] >= MIN_SCORE:
                    out[j] = clean(m[:, 0].any(0).numpy())
        return out, scores


def to_display(rgb, hd, wd):
    """decoded float [B,3,224,224] in [0,1] -> uint8 [B,hd,wd,3] at the native display aspect (bicubic)."""
    x = F.interpolate(rgb.float(), size=(hd, wd), mode="bicubic", align_corners=False).clamp(0, 1)
    return (x * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def stage_decoded(ds, cfg, data, W, cks, dev, out):
    """Decode each method's k=K forecast to RGB, segment the target in the DECODED image with one fixed rule for every
    method (top Grounding-DINO box(es) for the dataset prompt on that decoded image -> SAM 2.1), and compare with the
    tracked SAM mask of the true frame t+K (patch-level IoU at the dataset's coverage threshold, and centroid distance in
    display px). "decoded_truth" = the decoder applied to the TRUE future features (reference)."""
    from shiftwm.v2 import analysis as A
    H, K = data.history, data.horizon
    arms = [a for a in ARMS if a in cks]
    models = {a: load(cks[a], dev) for a in arms}
    decoder = A.load_decoder(A.decoder_path(ds), dev)
    seg = ImageSegmenter(dev, cfg["prompt"], cfg["objs"])
    keep = W["reason"] == 0
    widx = W["widx"][keep]; cov = W["cov"][keep].astype(np.float32)
    thr = cfg["cov"]; hd, wd = disp_size(ds, cfg["aspect"])
    names = tuple(arms) + ("decoded_truth",)
    cx, cy = [v.ravel() for v in np.meshgrid((np.arange(G) + 0.5) * wd / G, (np.arange(G) + 0.5) * hd / G)]
    res = {n: {"iou": [], "place": [], "detected": []} for n in names}
    for i in range(0, len(widx), 16):
        idx = torch.as_tensor(widx[i:i + 16], device=dev)
        h, pa, f, t = data.batch(idx)
        zs = {"decoded_truth": t[:, K - 1]}
        for a in arms:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zs[a] = models[a](h, pa, f).float()[:, K - 1]
        ck = cov[i:i + len(idx), K].reshape(len(idx), -1)
        gt = ck >= thr
        tcx = (ck * cx).sum(1) / ck.sum(1); tcy = (ck * cy).sum(1) / ck.sum(1)
        for n in names:
            imgs = to_display(A.decode(decoder, zs[n]), hd, wd)
            m, sc = seg.segment(imgs)
            pg = to_grid(m).reshape(len(idx), -1) >= thr
            res[n]["iou"].append(iou(pg, gt)); res[n]["detected"].append(sc >= MIN_SCORE)
            pl = np.full(len(idx), np.nan)
            for b in range(len(idx)):
                if m[b].any():
                    yy, xx = np.nonzero(m[b]); pl[b] = np.hypot(xx.mean() - tcx[b], yy.mean() - tcy[b])
            res[n]["place"].append(pl)
        if i % 160 == 0:
            print(f"[{ds}] decoded {i + len(idx)}/{len(widx)} (peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)",
                  flush=True)
    res = {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in res.items()}
    ep = data.episode_of[torch.as_tensor(widx, device=dev)].cpu().numpy()
    fg0 = (cov[:, 0] >= thr).reshape(len(cov), -1); fgK = (cov[:, K] >= thr).reshape(len(cov), -1)
    motion = 1 - iou(fg0, fgK); moving = motion >= np.median(motion)

    def summ(key, sel, fill=None):
        eps = np.unique(ep[sel])
        vals = {n: (np.where(np.isfinite(res[n][key]), res[n][key], fill) if fill is not None else res[n][key]) for n in names}
        per_ep = {n: np.array([np.nanmean(vals[n][sel & (ep == e)]) for e in eps]) for n in names}
        o = {}
        for n in names:
            lo, hi = boot_ci_nan(per_ep[n][:, None])
            o[n] = {"mean": float(np.nanmean(vals[n][sel])), "lo": float(lo[0]), "hi": float(hi[0])}
            if n != "shiftwm":
                d = (per_ep["shiftwm"] - per_ep[n])[:, None]; dlo, dhi = boot_ci_nan(d)
                o[n]["diff_sw_minus"] = {"mean": float(np.nanmean(d)), "lo": float(dlo[0]), "hi": float(dhi[0])}
        o["windows"] = int(sel.sum()); o["episodes"] = int(len(eps))
        return o

    both = np.all(np.stack([np.isfinite(res[n]["place"]) for n in arms]), 0)
    D = {"rule": f"top {DET_ID} box(es) for '{cfg['prompt']}' on each decoded image -> {SAM_ID} image predictor; "
                 f"same rule for every method; empty if the detection score < {MIN_SCORE}",
         "k": K, "display_hw": [hd, wd], "decoder": str(A.decoder_path(ds).relative_to(ROOT)),
         "detection_rate": {n: float(res[n]["detected"].mean()) for n in names},
         "iou": {sub: summ("iou", sel) for sub, sel in (("all", np.ones(len(widx), bool)), ("moving", moving))},
         "place_px": {sub: summ("place", sel & both) for sub, sel in (("all", np.ones(len(widx), bool)), ("moving", moving))},
         "place_note": "centroid distance, windows where every method's decoded image yields a segment (paired)"}
    S = json.loads((out / "summary.json").read_text()); S["decoded_segment"] = D
    D["example_rule"] = ("moving windows with ShiftWM decoded IoU >= 0.6 or placement <= 25th pct, and both Direct and AR "
                         "IoU <= 0.3 or placement >= 2x ShiftWM; ranked by IoU margin over the better baseline; distinct "
                         "episodes; top 2 (+3rd if margin >= 90% of the 2nd)")
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(f"[{ds}] decoded-segment IoU (moving):", {n: round(D["iou"]["moving"][n]["mean"], 3) for n in names},
          {n: [round(D["iou"]["moving"][n]["diff_sw_minus"][k], 3) for k in ("mean", "lo", "hi")] for n in names if n != "shiftwm"},
          "detection", {n: round(v, 3) for n, v in D["detection_rate"].items()})
    np.savez_compressed(out / "decoded_perwindow.npz", widx=widx, episode=ep, moving=moving, names=np.array(names),
                        **{f"{k}_{n}": res[n][k] for n in names for k in ("iou", "place", "detected")})
    # examples (illustrative, not representative): QC-passing moving windows where ShiftWM's decoded segment matches the
    # target (IoU >= 0.6 or placement <= 25th pct of ShiftWM's placement on moving windows) AND both Direct and AR fail
    # clearly (each: IoU <= 0.3 or placement >= 2x ShiftWM's); ranked by ShiftWM IoU - max(Direct, AR) IoU; distinct
    # episodes; top 2 (+ a 3rd if its margin is >= 90% of the 2nd's). Fallback: largest margins if none qualify.
    sw_i, sw_p = res["shiftwm"]["iou"], res["shiftwm"]["place"]
    p25 = np.nanpercentile(sw_p[moving], 25)
    good = (sw_i >= 0.6) | (np.nan_to_num(sw_p, nan=np.inf) <= p25)
    fail = np.ones(len(widx), bool)
    for a in arms:
        if a == "shiftwm":
            continue
        pa_ = np.nan_to_num(res[a]["place"], nan=np.inf)
        fail &= (res[a]["iou"] <= 0.3) | (pa_ >= 2 * np.nan_to_num(sw_p, nan=np.inf))
    rival = np.max(np.stack([res[a]["iou"] for a in arms if a != "shiftwm"]), 0)
    margin = sw_i - rival
    cand = moving & good & fail
    thresholds_met = bool(cand.any())
    adv = np.where(cand if thresholds_met else moving, margin, -np.inf)
    ex, seen = [], set()
    for w in np.argsort(-adv, kind="stable"):
        if not np.isfinite(adv[w]) or len(ex) == 3:
            break
        if ep[w] in seen:
            continue
        if len(ex) == 2 and adv[w] < 0.9 * adv[ex[1]]:
            break
        ex.append(int(w)); seen.add(ep[w])
    print(f"[{ds}] decoded example candidates meeting thresholds: {int(cand.sum())} (thresholds met: {thresholds_met})")
    ep_of, rel = window_table(data)
    trk = Segmenter(dev, cfg["prompt"], cfg["objs"])
    E = {k: [] for k in ("episode", "start", "obs", "fut", "mask_t", "mask_true", "decoded", "seg", "iou", "place")}
    idx = torch.as_tensor(widx[ex], device=dev)
    h, pa, f, t = data.batch(idx)
    zs = {"decoded_truth": t[:, K - 1]}
    for a in arms:
        zs[a] = models[a](h, pa, f).float()[:, K - 1]
    dec = {n: to_display(A.decode(decoder, zs[n]), hd, wd) for n in names}
    segs = {n: seg.segment(dec[n])[0] for n in names}
    for b, w in enumerate(ex):
        j = int(widx[w]); e = ep_of[j]; s0 = int(rel[j])
        imgs = frames_for(ds, data.episodes[e]["id"], cfg["aspect"])
        clip = imgs[s0 + H - 1:s0 + H + K]
        masks = trk.track(clip, [bb for bb in W["box"][keep][w].tolist() if bb[2] > bb[0]])
        E["episode"].append(data.episodes[e]["id"]); E["start"].append(s0)
        E["obs"].append(clip[0]); E["fut"].append(clip[K]); E["mask_t"].append(masks[0]); E["mask_true"].append(masks[K])
        E["decoded"].append(np.stack([dec[n][b] for n in names])); E["seg"].append(np.stack([segs[n][b] for n in names]))
        E["iou"].append([float(res[n]["iou"][w]) for n in names]); E["place"].append([float(res[n]["place"][w]) for n in names])
    print(f"[{ds}] decoded examples:", [(E["episode"][b], dict(zip(names, np.round(E["iou"][b], 3)))) for b in range(len(ex))])
    np.savez_compressed(out / "decoded_examples.npz", names=np.array(names), k=K, rule="decoded: ShiftWM good & both baselines fail; ranked by margin", thresholds_met=thresholds_met,
                        n_candidates=int(cand.sum()),
                        **{k: np.array(v) for k, v in E.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="droid", choices=tuple(DATASETS))
    ap.add_argument("--stage", default="all", choices=("all", "masks", "eval", "examples", "placement", "decoded"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-episodes", type=int, default=None, help="debug only")
    ap.add_argument("--out", default=None, help="override output dir (debug)")
    ap.add_argument("--max-gb", type=float, default=16.0, help="GPU memory cap (shared GPU)")
    ap.add_argument("--skip-if-current", action="store_true",
                    help="do nothing if summary.json is 'done' for the same checkpoints (paths and mtimes)")
    args = ap.parse_args()
    ds, cfg = args.dataset, DATASETS[args.dataset]
    out = Path(args.out) if args.out else OUT0 / ds
    out.mkdir(parents=True, exist_ok=True)
    cks = checkpoints(ds)
    if not all(a in cks for a in REQUIRED):
        (out / "summary.json").write_text(json.dumps({"dataset": ds, "status": "pending", "reason": "missing checkpoints: "
                                                      + ", ".join(a for a in REQUIRED if a not in cks)}, indent=1))
        print(f"[{ds}] pending: missing checkpoints"); return
    if args.skip_if_current and (out / "summary.json").exists():
        old = json.loads((out / "summary.json").read_text())
        if old.get("status") == "done" and old.get("checkpoints") == {a: str(c.relative_to(ROOT)) for a, c in cks.items()} \
                and all(abs(old["checkpoint_mtime"][a] - c.stat().st_mtime) < 1 for a, c in cks.items()):
            print(f"[{ds}] up to date"); return
    dev = "cuda"
    # the GPU is shared with training jobs (srun --overlap): hard cap on this process's memory
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gb * 2 ** 30 / total))
    feat = ROOT / "data/v2/features" / ds / "dinov2s"
    man = json.loads((feat / "manifest.json").read_text())
    split = "test" if any(r["split"] == "test" for r in man["episodes"]) else "val"
    run_cfg = json.loads((cks["shiftwm"].parent / "config.json").read_text())
    stats = json.loads((feat / "stats.json").read_text())
    data = FeatureSplit(feat, split, run_cfg["history"], run_cfg["horizon"], dev, stats, stride=args.stride,
                        tasks=run_cfg.get("tasks"), max_episodes=args.max_episodes)
    n = len(data)
    sel = np.arange(n) if n <= MAX_WINDOWS else np.unique(np.linspace(0, n - 1, MAX_WINDOWS).round().astype(int))
    if args.stage in ("all", "masks"):
        stage_masks(ds, cfg, data, sel, dev, out, args.stride)
    if args.stage == "decoded":
        stage_decoded(ds, cfg, data, dict(np.load(out / "windows.npz")), cks, dev, out)
        return
    if args.stage in ("all", "eval", "examples", "placement"):
        W = dict(np.load(out / "windows.npz"))
        assert int(W["stride"]) == args.stride
        stage_eval(ds, cfg, data, W, cks, dev, out, split, examples_only=args.stage in ("examples", "placement"))


if __name__ == "__main__":
    main()
