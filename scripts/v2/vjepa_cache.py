"""Cache frozen V-JEPA 2-AC ViT-g/16 encoder tokens for every DROID frame (primary camera, all splits).

Exactly the per-frame encoding of scripts/v2/eval_vjepa2ac.py (their forward_target): training-transform crop
(rows 0..180, cols 38..281 of 180x320 -> 256x256), frame duplicated into a 2-frame tubelet, bf16 autocast,
F.layer_norm over channels, stored fp16; per-episode batches of --encode-batch (default 64, as the evaluator) so
that cached tokens are bitwise what eval_vjepa2ac.py computes on the fly.

Output (--out, default data/v2/features/droid/vjepa2g):
  tokens.npy   [F,256,1408] float16  (np.lib.format memmap; 16x16 grid, row-major, 721 KB/frame)
  states.npy   [F,7] float32         state proxy per frame (eval_vjepa2ac.frame_states)
  actions.npy  [F,7] float32         poses_to_diffs(state_t, state_t+1) at index t; 0 on each episode's last frame
  index.json   episode table (id, session, split, offset, T) + provenance; "complete": true when finished
  progress.json  number of episodes done (resume point; the job can be re-submitted any time)
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import eval_vjepa2ac as ev  # noqa: E402

SPLITS = ("train", "val", "test")


def episode_table(data, camera, limit=None):
    manifest = json.loads((data / "manifest.json").read_text())
    rows, off = [], 0
    for split in SPLITS:
        for r in [r for r in manifest["episodes"] if r["split"] == split][:limit]:
            rec = r["cameras"][camera]
            rows.append({"id": r["episode_id"], "session": r["session_id"], "split": split, "offset": off,
                         "T": int(rec["frames"]), "file": rec["file"]})
            off += int(rec["frames"])
    return rows, off


def load_episode(data, row):
    with np.load(data / row["file"]) as z:
        imgs, act, fidx = z["images"], z["actions"], z["frame_indices"]
    T = len(imgs)
    assert T == row["T"] and act.shape == (T - 1, 35), (row["id"], imgs.shape, act.shape)
    assert np.array_equal(fidx - fidx[0], ev.NATIVE_PER_STEP * np.arange(T)), "unexpected frame clock"
    s = ev.frame_states(act)
    a = np.concatenate([ev.poses_to_diffs(s), np.zeros((1, 7))], 0)
    return imgs, s.astype(np.float32), a.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data/real_video/droid_selected/processed"))
    p.add_argument("--camera", default="exterior_image_1_left")
    p.add_argument("--out", default=str(ROOT / "data/v2/features/droid/vjepa2g"))
    p.add_argument("--repo", default=str(ROOT / "external/vjepa2"))
    p.add_argument("--checkpoint", default=str(ROOT / "data/pretrained/vjepa2_ac/vjepa2-ac-vitg.pt"))
    p.add_argument("--crop", choices=("train", "square"), default="train")
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--limit-episodes", type=int, default=None, help="smoke test: only the first N episodes of each split")
    p.add_argument("--dry-run", action="store_true", help="CPU: build the table/states only, no model")
    a = p.parse_args()
    t0 = time.time()
    data, out = Path(a.data), Path(a.out)
    rows, F_total = episode_table(data, a.camera, a.limit_episodes)
    print(f"[table] {len(rows)} episodes, {F_total} frames -> {F_total * 256 * 1408 * 2 / 1e9:.1f} GB fp16", flush=True)
    if a.dry_run:
        for r in rows[:3]:
            _, s, act = load_episode(data, r)
            print(r["id"], r["split"], r["T"], s[:2].round(3).tolist(), act[:1].round(4).tolist())
        return
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((ROOT / "references/vjepa2_ac_sources.json").read_text())
    index = {"episodes": rows, "frames": F_total, "grid": ev.GRID, "tokens": ev.GRID ** 2, "dim": 1408,
             "camera": a.camera, "crop": a.crop, "crop_box_ijhw_for_180x320": list(ev.crop_box(180, 320, a.crop)),
             "encode_batch": a.encode_batch, "dtype": "float16",
             "feature_space": "F.layer_norm(ViT-g/16 target encoder tokens), each frame duplicated into a 2-frame "
                              "tubelet (app/vjepa_droid/train.py::forward_target), bf16 autocast",
             "checkpoint": {"path": str(a.checkpoint), "sha256": ref["checkpoint"]["sha256"]},
             "repo_commit": ref["repo"]["commit"], "state_action": "eval_vjepa2ac.frame_states / poses_to_diffs; "
             "actions[t] = transition t->t+1, zero on each episode's last frame", "complete": False}
    idx_path, prog_path = out / "index.json", out / "progress.json"
    tok_path = out / "tokens.npy"
    done = 0
    if idx_path.exists() and tok_path.exists():
        old = json.loads(idx_path.read_text())
        same = old["episodes"] == rows and old["crop"] == a.crop
        if same and prog_path.exists():
            done = json.loads(prog_path.read_text())["done"]
            print(f"[resume] {done}/{len(rows)} episodes already cached", flush=True)
        elif not same:
            raise SystemExit(f"{out} holds a different cache (episode table/crop differ); remove it first")
    if done == 0 or not tok_path.exists():
        tokens = np.lib.format.open_memmap(tok_path, mode="w+", dtype=np.float16, shape=(F_total, 256, 1408))
        states = np.lib.format.open_memmap(out / "states.npy", mode="w+", dtype=np.float32, shape=(F_total, 7))
        actions = np.lib.format.open_memmap(out / "actions.npy", mode="w+", dtype=np.float32, shape=(F_total, 7))
        done = 0
    else:
        tokens = np.load(tok_path, mmap_mode="r+")
        states = np.load(out / "states.npy", mmap_mode="r+")
        actions = np.load(out / "actions.npy", mmap_mode="r+")
        assert tokens.shape == (F_total, 256, 1408)
    idx_path.write_text(json.dumps(index, indent=1))
    if done == len(rows):
        print("[cache] already complete", flush=True)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        dev = "cuda"
        encoder, predictor, info = ev.load_model(a.repo, a.checkpoint, dev)
        del predictor
        torch.cuda.empty_cache()
        print(f"[model] encoder loaded in {time.time() - t0:.0f}s (embed {encoder.embed_dim})", flush=True)
        pool = ThreadPoolExecutor(8)
        todo = rows[done:]
        futs = {i: pool.submit(load_episode, data, todo[i]) for i in range(min(16, len(todo)))}
        t1, nf = time.time(), 0
        for i, r in enumerate(todo):
            imgs, s, act = futs.pop(i).result()
            if i + 16 < len(todo):
                futs[i + 16] = pool.submit(load_episode, data, todo[i + 16])
            x = ev.preprocess(imgs, a.crop, dev)
            z = ev.encode_frames(encoder, x, a.encode_batch)          # [T,256,1408] fp16, layer-normed
            assert z.shape == (r["T"], 256, 1408), z.shape
            o, T = r["offset"], r["T"]
            tokens[o:o + T] = z.cpu().numpy()
            states[o:o + T] = s
            actions[o:o + T] = act
            nf += T
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                tokens.flush(); states.flush(); actions.flush()
                prog_path.write_text(json.dumps({"done": done + i + 1}))
                el = time.time() - t1
                print(f"[encode] {done + i + 1}/{len(rows)} episodes, {nf} frames, {nf / el:.0f} frames/s, "
                      f"ETA {(F_total - o - T) / max(nf / el, 1e-9) / 60:.1f} min", flush=True)
        pool.shutdown()
    # sanity: layer-normed tokens (mean 0 / var 1 per token), a few frames per split
    chk = {}
    for split in SPLITS:
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        z = torch.from_numpy(np.asarray(tokens[rs[0]["offset"]:rs[0]["offset"] + 4])).float()
        chk[split] = {"token_mean_abs": float(z.mean(-1).abs().max()), "token_var_mean": float(z.var(-1, unbiased=False).mean()),
                      "sha256_first_frame": hashlib.sha256(np.asarray(tokens[rs[0]["offset"]]).tobytes()).hexdigest()}
    index["complete"] = True
    index["sanity"] = chk
    index["runtime_s_last_job"] = time.time() - t0
    idx_path.write_text(json.dumps(index, indent=1))
    print(json.dumps(chk), flush=True)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
