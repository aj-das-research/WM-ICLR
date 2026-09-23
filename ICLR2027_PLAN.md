# ShiftWM → ICLR 2027: complete submission plan

Created 2026-09-23 17:40 (+04). Owner: Abhijit Das. Compute: Jebel H200 (partition `h200`,
`--comment="h200test2609"`, ≤2 GPUs, ≤2 jobs, ≤8 h/job, **access ends 2026-09-26 17:00 +04**).
Overleaf: project `6aadbb24b37acd9be4eed157` → `submission_folder/` + root `main.tex`.

Status legend: ☐ todo · ◐ running · ☑ done · ✗ dropped (with reason)

---

## 0. Diagnosis (condensed from the internal area-chair review, 2026-09-23)

Current verdict if submitted as-is: **reject (≈3/10)**. Root causes:

| # | Problem | Evidence in current draft | Fix in this plan |
|---|---|---|---|
| 1 | **No held-out result for the proposed model.** Every DROID number is on the 141-episode validation split used for model/checkpoint selection. | `main_discussion.tex:3-4` | §3 E1: 132-episode DROID test split (never used by spatial model) + fresh sessions |
| 2 | **Weak/unfaithful baselines.** Only external baseline (adapted DINO-WM) loses to *persistence*. No V-JEPA 2-AC, LeWM, DINO-Foresight-style, cross-patch control. | Table 1 | §3 E2: faithful baselines at native tokens, tuned until ≥ persistence |
| 3 | **Toy scale.** 4×4 pooled DINOv2-S grid (16 tokens), d=96, ~1M params, 1,126 episodes, one camera. | `real_video_spatial/features.py:100-107` | §2 v2 architecture at native 16×16 tokens; encoder ablation (DINOv2-S/B, DINOv3) |
| 4 | **Method definition contradicted by evidence.** tanh bound inert on DROID (0.07%), harmful on IWS (7–12%). | Tables 1, IWS | v2 removes tanh → learned per-channel scale |
| 5 | **Control failure.** Reacher 27% vs AR 80%; PushT ~11% for all arms (LeWM reports far higher) — planner budget cripples everyone; anchoring lowers action sensitivity (ρ 0.64→0.42). | `generated/iclr_review_v1/action_ranking_table.tex` | §3 E4 official planning budgets + action-contrastive loss + multi-source memory |
| 6 | **Confound:** temporal predictor is per-patch (no cross-patch path), so "mixing helps" ≈ "any spatial interaction helps". | `model.py:118-120` | Cross-patch-attention capacity-matched control; v2 predictor has spatial attention |
| 7 | **Metrics too narrow**: endpoint feature MSE only. | — | §4 full metric suite |
| 8 | **Presentation**: "Research draft" header, 83 pp, appendices A–Y, ~40% historical material, 22 refs, no related-work section, hedged prose, PushT-render teaser. | `main.tex:32-37` | §6 rewrite into `submission_folder/` |

Novelty risk: copy/transform predictors (CDNA/DNA, SNA/Ebert 2017, DFN), DDP-WM (2602.01780),
DINO-world (2507.19468), drift-resistant anchored NWM (2605.24761). **Positioning**: ShiftWM =
*action-prefix-conditioned transport of a fixed observed feature memory in frozen-foundation-feature
space*, with the transport field as an interpretable, measurable motion estimate, and an explicit
study of the **open-loop accuracy ↔ action-sensitivity trade-off** that anchoring induces (and a fix).

---

## 1. Claims we will aim to support (and falsification criteria)

- **C1 (forecasting):** At native resolution, ShiftWM-v2 lowers multi-step feature error vs matched AR,
  direct-additive and DINO-WM-style predictors on **held-out** DROID test, IWS reserved handles, and
  Open-H surgical (Hamlyn dVRK). *Falsified if* the paired 95% CI vs best matched baseline includes 0 on
  ≥2 of 3 datasets → paper reframed as analysis.
- **C2 (mechanism):** The learned transport field tracks real motion (agreement with optical flow /
  end-effector displacement) and gains concentrate where motion is present.
- **C3 (control):** With action-contrastive training, ShiftWM-v2 matches or beats AR planning success at
  official budgets on PushT / TwoRoom / Reacher. *If not*, report the trade-off honestly as a finding.
- **C4 (efficiency):** Params / FLOPs / latency vs DINO-WM, LeWM, V-JEPA 2-AC.

All numbers in the paper come from completed, logged runs. Negative results are kept (appendix if secondary).

---

## 2. Architecture: ShiftWM-v2 (stays faithful to core idea)

| Change | Rationale | Ablation arm |
|---|---|---|
| Native **16×16** DINO patch tokens (224 px) instead of 4×4 pooled | Real motion is expressible; answers "toy" critique | 4×4 pooled (v1) |
| **Local-window transport** (k×k neighbourhood, k=7 default; softmax over window + identity bias) | Linear cost in tokens; motion-field prior | global transport; k∈{3,5,7,11} |
| **Multi-source memory**: mix from all observed grids (t−2..t) | long motions, occlusion | last-frame only |
| **Unbounded correction with learned per-channel scale** (replace tanh) | fixes issue #4 | tanh bound (v1) |
| **Spatio-temporal predictor** (factorized space/time attention over tokens × history, action tokens via AdaLN) | removes per-patch confound | per-patch (v1) |
| Drop context/FiLM module | zero measured benefit, −14% params | with context |
| **Action-contrastive auxiliary loss** (true vs. shuffled action prefix must yield distinguishable predictions; InfoNCE on predicted-future vs target) | restore action sensitivity for planning | λ=0 |
| Horizon-aware gate/temperature | learned schedule for leaving identity | fixed |

Model sizes: S (d=192, depth 6, ~6M), B (d=384, depth 8, ~20M) — comparable to LeWM (15M) and DINO-WM (19M).

---

## 3. Experiments (priority, est. H200-hours; several small runs are packed per GPU)

Throughput assumption: models ≤20M params on cached features are dataloader-bound; pack 3–6 runs per
H200 (141 GB), data cached in node-local `/tmp` (861 GB free). Budget ≈ 2 GPUs × ~68 h ≈ 136 GPU-h.

### E0 Infrastructure (Sep 23) — P0
- ◐ uv env (Py3.11, torch 2.6 cu124) ☑; GPU smoke job 6611 ☑ (H200 NVL, 260 TFLOPS bf16, compute nodes have egress)
- ◐ Downloads: DINOv2-S ☑, IWS 2.8 GB, DROID 24 shards 22 GB, LeWM PushT/Reacher/TwoRoom data + ckpts, Open-H Hamlyn 24 GB
- ☐ Native-resolution feature caches (DINOv2-S 16×16; DINOv2-B; DINOv3-S/16 if gated access OK) for DROID/IWS/Hamlyn — GPU job, ~3 GPU-h
- ☐ Generic multi-dataset training harness (`src/shiftwm/v2/`): one trainer, all arms, packed multi-run launcher, per-horizon metric logging, resumable at epoch boundaries (8 h wall limit)
- ☐ Unit tests: causality (no future leakage), shape, identity-at-init, resume parity

### E1 Held-out DROID forecasting — P0 (~15 GPU-h)
- Splits: train 851 / val 143 (selection) / **test 132 (report)** — session-disjoint, exist in inventory.
- Arms × 3 seeds: Persistence, Linear-extrap, AR, Direct-additive, DINO-WM-style ViT predictor (native tokens, tuned), Cross-patch attention control (capacity-matched), ShiftWM-v1 (4×4, for continuity), **ShiftWM-v2-S**, v2-B.
- Horizons 1…10 (and 20 if episode lengths permit). Paired bootstrap CIs over episodes; Holm correction across primary contrasts.
- Stretch: extra fresh DROID sessions (download ~300 more episodes) — only if time permits.

### E2 Strong external baselines — P0/P1 (~25 GPU-h)
- **DINO-WM** predictor re-implemented faithfully (ViT causal predictor, frame-level causal mask, native 16×16 patches, proprio optional) + lr/depth sweep until ≥ persistence — P0.
- **LeWM** predictor (vendored) trained on same features — P0.
- **V-JEPA 2-AC** (public ViT-g + AC predictor, trained on DROID): zero-shot forecasting in its own feature space on our DROID test episodes, reported with its own persistence/AR reference; plus ShiftWM-v2 head trained on V-JEPA 2 ViT-L features (encoder swap) — P1. *Caveat to report: V-JEPA 2-AC trained on DROID; overlap with our test sessions cannot be ruled out.*
- DINO-world-style cross-attention predictor (attend to all past tokens) — P1 (same harness).
- ✗ Cosmos / Ctrl-World / Genie / DreamerV3 / TD-MPC2: out of budget; justified in related work (pixel generators / online RL).

### E3 Medical: Open-H-Embodiment Hamlyn dVRK (7 surgical tasks, kinematics) — P0 (~12 GPU-h)
- Tasks: suturing_1/2, knot_tying, needle_grasp_and_handover, peg_transfer, tissue_lifting, tissue_retraction. Episode-disjoint split per task (70/15/15), frozen before evaluation.
- Same arms as E1 (3 seeds for top-4 arms). Metrics: feature MSE/cos per horizon, instrument-motion-weighted error, kinematics probe error.
- Stretch: Endoscopy/cuhk (9.5 GB) as second medical set.

### E4 Planning at official protocols — P0/P1 (~35 GPU-h)
- stable-worldmodel / LeWM suite: **PushT, TwoRoom, Reacher** (Cube: 46 GB data — stretch).
- Sanity first: reproduce released LeWM checkpoint success at official CEM budget (DINO-WM: 300 samples × 30 iters).
- Train AR, DINO-WM-style, ShiftWM-v2 (±action-contrastive) on each env; 50 eval episodes × 3 seeds; report success rate + planning time.

### E5 IWS (PushT/Box/Rope single-image, RLA-WM handles) — P1 (~8 GPU-h)
- Re-run v2 vs AR vs direct vs v1-no-tanh at native tokens on the reserved handles; RGB metrics via existing shared decoder (LPIPS/PSNR/SSIM).

### E6 Analysis — P1 (~8 GPU-h)
- Action sensitivity: Δerror under shuffled/swapped action prefixes; true-vs-counterfactual ranking (Spearman ρ, top-1).
- Transport field vs optical flow (RAFT/Farnebäck on RGB, patch-level EPE/cosine) and vs end-effector displacement (DROID proprio).
- End-effector / kinematics linear probes on predicted features (DROID, Hamlyn).
- Error vs horizon curves; gain vs per-patch motion magnitude.
- Efficiency: params, FLOPs/step (fvcore/torch profiler), latency, peak memory, train GPU-h.
- Encoder ablation: DINOv2-S vs DINOv2-B vs DINOv3-S/B (if access) on DROID.

### E7 Pixel decoding (for visuals + LPIPS/PSNR/SSIM/FVD on DROID) — P2 (~8 GPU-h)
- Train one shared feature→RGB decoder on DROID train features; decode all arms' forecasts.

### GPU schedule (2 lanes, each ≤8 h jobs, chained with `--dependency`)
| Window (+04) | Lane A | Lane B |
|---|---|---|
| Sep 23 eve–night | feature extraction (DROID/IWS/Hamlyn) | harness smoke + LeWM planning sanity |
| Sep 24 | E1 DROID sweep (packed) | E3 Hamlyn (packed) + E2 DINO-WM tuning |
| Sep 25 | E1 seeds / v2-B / encoder ablation | E4 planning (PushT, TwoRoom, Reacher) |
| Sep 26 until ~13:00 | E2 V-JEPA 2-AC, E6 analysis, E7 decoder | E4 seeds, E5 IWS |
| Sep 26 13:00–17:00 | **backup**: results, logs, selected checkpoints → GitHub release / local Mac; remove data per notice | — |

---

## 4. Metrics we will report (matching related work)

| Family | Metrics | Who reports them |
|---|---|---|
| Feature forecasting | MSE (standardized + raw), cosine, per-horizon curves, endpoint & mean-over-horizon, episode win rate | DINO-Foresight, DINO-world, RLA-WM |
| Pixel (decoded) | PSNR, SSIM, LPIPS, FVD (DROID/IWS) | NWM, IRASim, Cosmos, Endora/SurgWM |
| Planning | success rate, planning time/episode, CEM budget | DINO-WM, LeWM, PLDM, DDP-WM |
| Physical grounding | linear-probe EE pose / kinematics error; LeWM-style probe MSE & Pearson r | LeWM, V-JEPA 2 |
| Action sensitivity | shuffled-action Δerror, counterfactual ranking ρ | Delta-JEPA, 2606.07687 |
| Efficiency | params (trainable/frozen), FLOPs/step, latency, memory, GPU-h | LeWM, DDP-WM |
| Statistics | ≥3 seeds, mean±std, paired bootstrap 95% CI, Holm-adjusted primary contrasts | — |

---

## 5. Related work to add (target ≥45 refs)
Feature/JEPA WMs: DINO-WM, LeWM, LeJEPA, PLDM, V-JEPA 2(-AC), DINO-world, DINO-Foresight, DDP-WM,
Delta-JEPA, FlowWM, VFMF, RLA-WM, "Reconstruction or Semantics?", Nano World Models.
Generative WMs: NWM, Cosmos (+Policy, Predict2.5), Ctrl-World, IRASim, UniSim, Genie 1/3, IWS.
Model-based RL: PlaNet, DreamerV3, TD-MPC2, IRIS, DIAMOND.
Transport/warp/copy: CDNA/DNA, SNA (Ebert 2017), DFN, Vondrick&Torralba 2017, SAVP, MCVD, FlowDreamer,
AHEAD, anchored-NWM (2605.24761); slots: SlotFormer, SOLD.
Benchmarks/eval: DROID, Bridge V2, Open-X, stable-worldmodel, WorldScore, WorldModelBench, EWMBench, Physics-IQ.
Medical: Open-H-Embodiment, Cosmos-H-Surgical, SurgWM, Endora, SutureBot, JIGSAWS, Cholec80/CholecT50, SurgVista.
Encoders: DINOv2, DINOv3, SigLIP2, Web-DINO.
(Every entry verified against arXiv/venue before it enters `references.bib`.)

---

## 6. Paper & presentation (Overleaf `submission_folder/`)

Layout:
```
main.tex                      # root: compiles submission_folder (main + appendix in one PDF)
submission_folder/
  iclr2027_conference.{sty,bst}, fancyhdr.sty, natbib.sty, math_commands.tex
  sections/{01_intro,02_related,03_method,04_experiments,05_analysis,06_conclusion}.tex
  appendix/{A_impl,B_datasets,C_baselines,D_full_results,E_planning,F_medical,G_analysis,H_qualitative,I_stats,J_repro}.tex
  figures/  tables/  references.bib
<existing draft files untouched at root, renamed entry: draft_v1_main.tex>
```
Main text (9 pp strict): 1 Intro (+teaser Fig 1) · 2 Related work · 3 Method (Fig 2 architecture) ·
4 Experiments: setup, Table 1 forecasting (DROID-test / Hamlyn / IWS × all methods), Table 2 planning,
Fig 3 error-vs-horizon · 5 Analysis: Fig 4 transport-field vs flow, Fig 5 action sensitivity, ablation table ·
6 Conclusion + limitations · Reproducibility, Ethics, **ICLR AI-use statement** (template format).
Appendix target 20–30 pp; historical context-model material removed (kept in repo/tech report).

Figures (vector PDF, colour-blind-safe palette, consistent typography, light/grayscale checked):
- **Teaser**: real DROID frame + surgical frame; learned transport arrows over patches; AR drift vs ShiftWM; headline held-out numbers.
- **Architecture**: 3 blocks (observed memory → action-prefix spatio-temporal predictor → local transport + gate + scaled correction), clean TikZ/vector.
- Error-vs-horizon curves (all datasets, CI bands); planning success bars; action-sensitivity scatter; transport-vs-flow qualitative grid; decoded rollouts strip; efficiency Pareto (FLOPs vs error).

Writing rules: one name ("ShiftWM"), confident but accurate wording, no "draft" header, no ops jargon in main text.

---

## 7. Risks & mitigations
- **GPU window closes Sep 26 17:00** → P0 first; checkpoint every epoch; back up results/ckpts by 13:00 Sep 26.
- v2 doesn't beat baselines → report honestly; reframe around the accuracy↔action-sensitivity trade-off (C3) which is itself a publishable analysis.
- Other users occupy h200 (2 nodes drained) → 1-GPU jobs start sooner than 2-GPU; pack runs.
- DINOv3 gated → fall back to DINOv2-B + V-JEPA 2 ViT-L as encoder ablation.
- ICLR 2027 deadline date: confirm (not yet announced in repo) — writing continues after GPU window.

## 8. Log
- 2026-09-23 17:10 cluster survey; 17:25 related-work survey; 17:35 area-chair review; env + downloads started; smoke job 6611 OK.
