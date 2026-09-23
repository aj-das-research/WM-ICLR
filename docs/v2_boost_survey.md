# ShiftWM-v2 boost survey: techniques to borrow (2026-09-23)

Scope: published techniques that could lower held-out DROID feature MSE (and help planning), need at most 1 day of
coding, and fit in the remaining ~60 H200 GPU-h (access ends 2026-09-26 17:00 +04). The repo was not modified
except for this file. Diagnostics were run on CPU from `/tmp/.../scratchpad/{oracle,diag2}.py` using the existing caches and checkpoints.

## 0. New diagnostics (these change the priorities)

**D1. Every learned arm overfits early.** Val MSE (mean over horizons) from the existing logs:

| arm | 2k | 4k | 8k | 12k | 16k |
|---|---|---|---|---|---|
| shiftwm s0 | 0.2032 | **0.2000** | 0.2088 | 0.2178 | 0.2219 |
| direct s0 | 0.2191 | **0.2144** | 0.2176 | 0.2250 | 0.2285 |
| ar s0 | 0.2238 | **0.2236** | 0.2344 | 0.2438 | 0.2471 |

Training loss falls to about 0.12 over the same period. DROID train is only **851 episodes, 47k frames (about 4.4 h at 3 Hz)**.
V-JEPA 2-AC used about 62 h of DROID. The best checkpoint comes at about 4 epochs, while the cosine LR is still near its peak.
**At 22M parameters the model is limited by data and variance, not by capacity.** Three consequences:
(a) regularisation, data and averaging are the most reliable levers; (b) extra head capacity is a risk;
(c) runs can be cut to about 6k steps, roughly 35–40% of the current cost, which roughly doubles the number of experiments the budget allows.

**D2. The 7×7 window is not the main bottleneck.** An oracle picked the best *single* source patch from the 3 observed frames using the target itself,
so it is an optimistic bound for hard copying. Test subset: 40 episodes, stride 4. "Moving" means the per-patch true change exceeds 0.5.

| candidate set | all (mean k) | moving (mean k) | moving k=10 |
|---|---|---|---|
| 3×3 | 0.172 | 0.613 | 0.695 |
| **7×7 (current)** | 0.157 | 0.523 | 0.588 |
| 7×7 ∪ 7×7 dilation 3 | 0.155 | 0.517 | 0.579 |
| 15×15 | 0.153 | 0.505 | 0.564 |
| global | 0.153 | 0.503 | 0.561 |
| persistence | 0.275 | 1.161 | 1.235 |

Offset from the nearest-neighbour source on moving patches: median 2 patches, 90th percentile 5 at k=10; 21% of patches lie outside ±3.
Even with the oracle, though, going from 7×7 to global only lowers overall MSE by 2.5% and moving-patch MSE by 4%.
**Enlarging the window, multi-scale transport or deformable offsets can recover about 0.5–2% at best.** The rest of the gap is in choosing the right source
(ShiftWM 0.194 vs oracle copy 0.157) and in content that is genuinely new or uncertain.

**D3. Checkpoints on the same subset** (all / moving / k=1 / k=10):
ShiftWM s0 0.1938/0.683/0.098/0.247; Direct s0 0.2095/0.731; persistence 0.2750/1.183.
- **Averaging the 2 ShiftWM seeds gives 0.1889 (−2.6%).** For Direct it gives −2.9%. Variance is a real error component.
- Transport weight on the 7×7 border ring is 0.05 overall and 0.18 on moving patches. Gate mean is 0.57 overall and 0.85 on moving patches.
- Least-squares scale of the predicted change (Ẑ−Z0) against the true change is **0.887**, below 1. The model slightly over-predicts motion magnitude, which also points to variance or overfitting.

## 1. Candidate techniques

"Effect" refers to our failure modes: L = large motion beyond the window, O = disocclusion or new content, B = blur, V = variance/overfitting.
All repositories were checked (HTTP 200) on 2026-09-23.

| # | Technique (paper, venue, arXiv) | Code | What to borrow | Effect | Implementation sketch | Cost | Risk |
|---|---|---|---|---|---|---|---|
| 1 | **Weight EMA** (Morales-Brotons et al., TMLR 2024, 2411.18704). Standard in DINO-WM, TD-MPC2 and diffusion models | [lucidrains/ema-pytorch](https://github.com/lucidrains/ema-pytorch) | EMA of weights with decay 0.999, evaluated and saved as `best.pt` | V; captures part of the measured 2.6% ensemble gain | `train.py` ~l.182: `ema=torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))`; update after `opt.step()`; `evaluate(ema.module)` | ~0 | very low |
| 2 | **Short annealed schedule plus dropout** (V-JEPA 2-AC anneal phase; OpenSTL's finding that recurrent-free models need regularisation at small data) | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), [chengtan9907/OpenSTL](https://github.com/chengtan9907/OpenSTL) | `steps` 6000 (cosine reaches 0 near the current overfitting point), `dropout` 0.1, weight decay 0.05→0.1, plus 10–15% memory-token dropout (random tokens of `memory` masked in cross-attention) | V | config only for steps, dropout and wd; token drop is about 10 lines in `decode()` (l.163) via an attention mask | saves compute | low |
| 3 | **More views and data augmentation** (DINO-world dataset ablation: Cityscapes-only 45.6 vs 53.2 with large data; V-JEPA 2-AC uses random resized crops on DROID) | vjepa2 `configs/train/vitg16/droid-256px-8f.yaml` (`random_resize_aspect_ratio [0.75,1.35]`) | (a) **train on cam1 and cam2 train episodes together**: the `droid_cam2` cache already exists with the same 851/143/132 split, which doubles the data at zero extraction cost, and eval stays cam1; (b) optional offline random-resized-crop feature caches (one fixed crop per episode) from `processed/cameras/*` with `extract.encode` | V (largest expected effect) | `FeatureSplit` (l.25): accept a list of roots and concatenate `features/actions/starts` with offsets, standardising with **cam1** `stats.json` (as in camera transfer). Crops need about 15 lines in `extract.py` (crop before `F.interpolate`, l.52) | (a) 0 GPU-h prep; (b) about 0.3 GPU-h extraction per crop set, CPU decode | low for (a); medium for (b), since crops shift the patch grid relative to training statistics |
| 4 | **Explicit correlation (cost-volume) features** (F2MF: Šarić et al., "Dense Semantic Forecasting in Video by Joint Regression of Features and Feature Motion", IEEE TNNLS 2021, arXiv 2101.10777; CVPR 2020 "Warp to the Future"). RAFT (ECCV 2020, 2003.12039) ablation: lookup radius 0 gives 3.41 EPE vs 1.63 | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) (`core/corr.py`); no official F2MF repository found | Cosine similarity between each patch at t and its 9×9 neighbourhood at t−1 (and t−1 vs t−2): 81×2 channels → Linear → added to the memory tokens of frames t−1 and t. In F2MF this gave **+0.8/+2.3 mIoU** (short/mid-term) to the motion head and +1.1–2.5 to the blended model. It was the most consistent gain in their ablation and helped the direct (F2F) head too | L, source selection | `encode_memory` (l.145): reuse the `unfold` from `transport` on L2-normalised `self.inp(hist)` projections; `x = x + self.corr_in(cv)`. About 25 lines | +2% FLOPs | low. Give the same input to **Direct** for a fair control |
| 5 | **RAFT-style iterative transport refinement with a recentred (deformable) window** (RAFT; GMFlow CVPR 2022, 2111.13680, where 1 refinement beats 31 RAFT iterations; Deformable DETR ICLR 2021, 2010.04159 for offsets predicted from queries) | RAFT; [haofeixu/gmflow](https://github.com/haofeixu/gmflow); [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) | 2–3 iterations with a shared update. Iteration i: expected offset `o_i = Σ π·δ` from the current weights; update `h ← h + MLP([h, proj(moved_i), corr stats(entropy, max)])` (weight-tied, zero-initialised output); new query from `h`; new 7×7 window **centred at p+o_i**, sampled with `F.grid_sample` (bilinear) from the S=3 source grids. Loss on each iteration with RAFT/DMVFN weights γ^(N−i), γ=0.8 | L (reach grows with iterations), selection; mild B | Replace `transport()` (l.174) with a gather at arbitrary centres: build `[B·K, S, 49]` sample grids `p+o+δ` and gather keys and values with `grid_sample` on `[B·S, D, 16, 16]`. About 60–80 lines. Keep the current l.181–206 path as iteration 0 | +10–20% step time | medium: D2 caps the reach gain; the benefit has to come from better selection |
| 6 | **Dilated / multi-scale window union** (DiNAT, 2209.15001; NATTEN; DMVFN CVPR 2023, 2303.09875: multi-scale [4,2,1] beats single scale, t+5 MS-SSIM 83.45 vs 80.93; PWC-Net coarse-to-fine) | [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN), [hzwer/CVPR2023-DMVFN](https://github.com/hzwer/CVPR2023-DMVFN) | Joint softmax over {7×7 d=1} ∪ {7×7 d=3} (98 candidates per source). NATTEN is unnecessary: at 16×16, `F.unfold(..., dilation=3, padding=9)` or even global (768 keys) is cheap | L | `unfold` (l.185–189): concatenate a second unfold with `dilation=3, padding=3*r` and deduplicate the centre (its valid mask via the same call). Add a learned relative-position bias per candidate so the locality prior is kept (global attention without it tends to blur) | +1× transport cost (negligible) | low, but the oracle bound says ≤0.3% overall and ≤1.5% on moving patches. Mainly useful as a reviewer-facing ablation ("large motion is not the limit") |
| 7 | **Test-time seed ensembling / post-hoc delta calibration** | n/a | Average of 2–3 seeds (measured −2.6%); per-horizon scalar on (Ẑ−Z0) fitted on **val** (measured optimum 0.887) | V | eval-only script | 0 train; K× inference | none, but it must be applied to **all** arms, or it is an unfair comparison. Report it as a secondary row |
| 8 | **TAU differential-divergence regulariser** (Tan et al., CVPR 2023, 2206.12126: MMNIST MSE 21.1→19.8 (−6%)) | OpenSTL | KL between softmax(ΔẐ/τ) and softmax(ΔZ/τ) over (N,C) for consecutive-step differences, τ=0.1, weight 0.1 | B, moving patches | `loss_fn` (l.85): 6 lines | ~0 | medium: evidence is only on pixel toy data; in standardised feature space the softmax over N·C is dominated by outliers |
| 9 | **TF + short-rollout loss mix for AR baselines** (V-JEPA 2-AC, 2506.09985; `app/vjepa_droid/train.py`: `loss = jloss(TF) + sloss(auto_steps=2)`, L1 on layer-normed features) | vjepa2 | Only for the **baselines**: `ar_tf` plus a 2-step rollout term. ShiftWM is a direct multi-step model, so TF, rollout and scheduled sampling do not apply to it | makes AR/DINO-WM baselines stronger (reviewer point #2) | `loss_fn`: call `_rollout` with teacher forcing for all steps, plus a free-running call of 2 steps | +30% for AR arms | none for ShiftWM |
| 10 | **Loss swap** (SmoothL1 β=0.1 / L1 / cosine: DINO-Foresight NeurIPS 2025, 2412.11673, Table 10 "L1, MSE, SmoothL1, SmoothL1+Cosine comparable"; DINO-world uses SmoothL1) | [Sta8is/DINO-Foresight](https://github.com/Sta8is/DINO-Foresight) | not recommended | none for MSE | — | — | the reported metric is MSE and MSE is its minimiser. FlowWM reports that ℓ1 degraded results |
| 11 | **Stochastic heads** (FlowWM, 2606.29059, [facebookresearch/Flow-World-Models](https://github.com/facebookresearch/Flow-World-Models); VFMF 2512.11225; MCVD NeurIPS 2022, [voletiv/mcvd-pytorch](https://github.com/voletiv/mcvd-pytorch)); multi-hypothesis winner-takes-all | — | not recommended for this deadline | would *raise* single-sample MSE | — | 50 ODE steps per sample | FlowWM's gains are best-of-N or perception metrics, not mean error; B is MSE-optimal shrinkage, not a bug |
| 12 | **Multi-layer DINO features / DINOv2-B / DINOv3 targets** (DINO-Foresight layers 3, 6, 9, 12 + PCA) | DINO-Foresight | ablation only | changes the target space, so MSE is not comparable across encoders | caches already exist (`droid/dinov2b`, `droid/dinov3s`) | 1 run each | reviewers may read an encoder swap as "tuning the metric". Keep it as the planned encoder ablation, not the headline |

Also checked and not prioritised: PredRNN ([thuml/predrnn-pytorch](https://github.com/thuml/predrnn-pytorch); recurrent and slow),
softmax splatting ([sniklaus/softmax-splatting](https://github.com/sniklaus/softmax-splatting); forward warping would help disocclusion ordering, but D2 shows copy-reachability is not the limit),
RIFE ([hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)), TD-MPC2 latent consistency ([nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2); multi-step consistency for AR latents, not relevant to a direct predictor),
PLDM ([vladisai/PLDM](https://github.com/vladisai/PLDM)), DINO-WM ([gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)), and DreamerV3 ([danijar/dreamerv3](https://github.com/danijar/dreamerv3)).
DINO-world (2507.19468; no official code found) offers variable-Δt training (sampling time deltas),
but DROID actions are tied to a 0.33 s step, so resampling Δt would change the meaning of the actions.

## 2. Top 5 ranking (expected held-out gain × evidence ÷ risk)

1. **More data: train on cam1+cam2 (#3a)**, applied to all arms. Most direct fix for D1. Expected −3 to −8% (uncertain; the views are correlated). About 1 h of coding.
2. **EMA plus short annealed schedule plus dropout / memory-token dropout (#1+#2)**. Expected −2 to −4%, and it cuts cost per run by about 60%. About 1 h of coding.
3. **Cost-volume features into memory (#4)**. The comparable dense-feature-forecasting setting (F2MF) shows consistent gains, largest at mid-term. Expected −1 to −3% overall, more on moving patches. About 2 h.
4. **Iterative recentred transport refinement ×2 (#5)**. Strong evidence in flow estimation, unproven for forecasting. Expected −1 to −3% on moving patches. About 5 h. It also makes a good method-section story ("transport as iterative correspondence").
5. **Dilated window union plus relative-position bias (#6)**. Cheap; small gain (≤1.5% moving by the oracle bound). Mainly to show the large-motion case is handled.
   (Seed ensembling, #7, gives a certain −2.6% but is an evaluation protocol, not a method change.)

## 3. The two combinations to try first

Current budget: at 6k steps a run takes about 30–45 min on one H200. Plan about 20–25 GPU-h in total and keep more than 30 GPU-h in reserve for the P0 queue.

**Combo A: "regularised, more data" (applies to ShiftWM, Direct and AR alike; run first).** `droid_base.json` diff:
```json
"train_roots": ["data/v2/features/droid/dinov2s", "data/v2/features/droid_cam2/dinov2s"],
"stats_root": "data/v2/features/droid/dinov2s",
"steps": 8000, "warmup": 400, "lr": 3e-4, "weight_decay": 0.1, "ema": 0.999, "eval_every": 1000,
"model": {"dropout": 0.1, "extra": {"mem_token_drop": 0.15}}
```
Screening: ShiftWM ×{A-full, A without cam2, A without EMA}, 1 seed each, about 2.5 GPU-h. Then Direct and AR with A-full.
Decision rule: keep A if val MSE is at least 0.004 below 0.200.

**Combo B: "correspondence-aware transport" (ShiftWM-specific, on top of A).**
```json
"model": {"window": 7, "window_dilations": [1, 3], "rel_pos_bias": true,
          "cost_volume": {"radius": 4, "pairs": 2},
          "transport_iters": 2, "iter_loss_gamma": 0.8, "recenter": true}
```
Ablation ladder, 1 seed each at 6–8k steps (about 4 GPU-h): A → A+cost volume → A+cost volume+iters2 → +dilation.
Give the cost volume to Direct as well (1 run), so the gain cannot be attributed to the correlation input alone.
Decision rule: adopt a component if it improves val mean-over-horizon MSE by at least 0.002 and moving-patch MSE by at least 1%.
Then run the chosen configuration for 3 seeds against 3 seeds each of Direct and AR under Combo A: about 9 runs, about 8–10 GPU-h.

Optionally, if Combo A shows data is the binding constraint (cam2 gives more than 3%), extract one random-resized-crop cache (#3b)
for cam1 and cam2 train episodes (scale [0.7, 1], aspect [0.75, 1.35], fixed per episode; about 1 GPU-h plus CPU decode) and rerun the best configuration.

## 4. Planning-specific notes

- EMA, ensembles and more data smooth the CEM cost landscape. Check the action-sensitivity metrics (`rank_acc`), since anchoring already lowers them.
- Keep `contrastive_weight` for the planning envs. The cost volume does not see actions, so it should not reduce action sensitivity.
  Iterative refinement conditions on `hidden`, which carries the AdaLN action prefix, so action dependence goes through it.
- Borrow #9 (V-JEPA 2-AC TF + 2-step rollout) for the AR/DINO-WM planning baselines to pre-empt the "weak baseline" critique.

## Sources
RAFT https://arxiv.org/abs/2003.12039 · GMFlow https://arxiv.org/abs/2111.13680 · Deformable DETR https://arxiv.org/abs/2010.04159 ·
DiNAT https://arxiv.org/abs/2209.15001 · DMVFN https://arxiv.org/abs/2303.09875 · TAU https://arxiv.org/abs/2206.12126 ·
OpenSTL https://arxiv.org/abs/2306.11249 · F2MF https://arxiv.org/abs/2101.10777 · Warp to the Future (CVPR 2020)
https://openaccess.thecvf.com/content_CVPR_2020/html/Saric_Warp_to_the_Future_Joint_Forecasting_of_Features_and_Feature_CVPR_2020_paper.html ·
DINO-Foresight https://arxiv.org/abs/2412.11673 · DINO-world https://arxiv.org/abs/2507.19468 · V-JEPA 2 https://arxiv.org/abs/2506.09985 ·
DINO-WM https://arxiv.org/abs/2411.04983 · FlowWM https://arxiv.org/abs/2606.29059 · VFMF https://arxiv.org/abs/2512.11225 ·
EMA https://arxiv.org/abs/2411.18704
