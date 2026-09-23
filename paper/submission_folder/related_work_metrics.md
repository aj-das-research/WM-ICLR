# Closest methods: benchmarks and exact metrics they report

Checked 2026-09-23 against the arXiv HTML versions (table captions and headers, extracted programmatically) and, where noted, earlier project audits (`reports/metric_literature_audit_2026-09-21.md`). This guides the layout of our results tables. It is **not** a source of numbers to copy: every comparable number must be re-measured under our protocol (same encoder, same target coordinates, same splits).

Legend: SR = planning/task success rate; CD = Chamfer distance; FVD/FID = Fréchet video/image distance; δ1 = depth threshold accuracy; AbsRel = absolute relative depth error; ATE/RPE = absolute trajectory / relative pose error.

## A. Latent / foundation-feature world models (closest to ShiftWM)

| Method (bib key) | Benchmarks | Exact metrics reported | Where | Does it report raw feature-forecast error? |
|---|---|---|---|---|
| DINO-WM (`zhou2025dinowm`) | PointMaze, Reach, PushT, Wall (SR); Rope, Granular (CD) | SR over 50 start/goal pairs; CD over 10 instances; LPIPS and SSIM on decoded predictions (PushT, Wall, Rope, Granular); inference and planning time; CEM vs. GD planner | Tab. 1, 2, 3, 4, 8, 9, 10 | No. Latent error is used only as the planning cost |
| LeWM (`maes2026lewm`) | Two-Room, Reacher, PushT, OGBench-Cube | SR (mean ± std over 3 training seeds, 50 trajectories); planning time (reported as 48× faster than DINO-WM); linear/MLP physical probes (MSE, Pearson r); violation-of-expectation surprise; latent straightness (cosine between consecutive latent velocities) | Tab. 1, 3–10, Fig. 3 | No |
| Fast LeWM (`gao2026fastlewm`) | Same four as LeWM | SR; dynamics time and full CEM time under a fixed budget (RTX 4090); physical probes on PushT; open-loop latent loss (per project audit) | Tab. 1–4 | Latent loss only as a diagnostic |
| PLDM (`sobal2025pldm`) | Two-Rooms, Diverse PointMaze, Ant-U-Maze | SR (mean ± std over 3–10 seeds); time per episode against replanning frequency; Welch t-tests across seeds and datasets | Tab. 4–8 | No |
| JEPA-WMs (`terver2026jepawms`) | Maze, Wall, PushT, Metaworld Reach/Reach-Wall, Robocasa Place/Reach, DROID-like recorded data | SR for several planners, with variance over 3 seeds; Spearman correlation between success and validation metrics (visual-embedding prediction error, proprioceptive decoding error); offline action error and Action Score on recordings (per project audit) | Tab. 2, 11–16 | Yes, but only as a validation metric correlated with SR |
| V-JEPA 2-AC (`assran2025vjepa2`) | Real Franka arms in two labs: reach, grasp, reach-with-object, pick-and-place (cup, box) | Real-robot success rate; planning time per step and CEM settings (samples, iterations, horizon) compared with Cosmos | Tab. 2, 3 | No |
| DINO-world (`baldassarre2025dinoworld`) | VSPW, Cityscapes (segmentation), KITTI (depth); IntPhys, GRASP, InfLevel; PushT, Wall, PointMaze | Linear-head mIoU and RMSE at present, short (~0.2 s) and mid (~0.5 s) horizons; "Copy Last" baseline; intuitive-physics relative accuracy; planning SR over 512 episodes | Tab. 1, 2, 4 | No. Features are scored through frozen downstream heads |
| DINO-Foresight (`karypidis2025dinoforesight`) | Cityscapes (plus nuScenes in appendix) | Semantic segmentation mIoU (ALL, movable objects MO); instance AP50/AP; depth δ1/AbsRel; surface-normal mean angular error and % within 11.25°; short and mid horizons | Tab. 1, 2, 6, 7 | No (L1/MSE/SmoothL1 appear only as training-loss ablations) |
| VFMF (`boduljak2025vfmf`) | Cityscapes, Kubric | Segmentation mIoU (All, Mov.), depth d1/AbsRel, normals a3/MeanAE, with rollouts of 9–11 frames | Tab. 1, 6, 7 | No |
| FlowWM (`porcher2026flowwm`) | FuturePerception (Waymo) | Future object-detection AP_L(3), AP_L(6); depth RMSE, δ1, δ2, δ3; DINO-WM (DINOv3) used as a baseline | Tab. 2 | No |
| DDP-WM (`yin2026ddpwm`) | PointMaze, PushT, Wall (SR); Rope, Granular (CD) | SR/CD against IRIS, DreamerV3, DINO-WM; FLOPs per step; throughput (samples/s); MPC loop time; 5-step open-loop pixel error (count of erroneous pixels); mask IoU/precision/recall | Tab. 1–8 | No (pixel-count error on decoded frames) |
| RLA-WM (`zhang2026rlawm`) | ManiSkill (Panda, XArm, UR10), IWS (Box, PushT, Rope) | Final-frame LPIPS, SSIM, **DINO-token L1**; inference FLOPs; latent-action BC SR; world-model RL SR over 1,500 episodes | Tab. 1, 2, 3, A1, A2 | **Yes (DINO L1)**, but on masked DINOv3 features |
| Delta-JEPA (`zhang2026deltajepa`) | Two-Room, Reacher, PushT, OGB-Cube | Planning SR (mean ± std) against PLDM and LeWM; linear/MLP probes of state and state delta (MSE, r) | Tab. 1–11 | No |
| Nano World Models (`huang2026nanowm`) | RT-1 fractal, PointMaze, Wall, PushT | PSNR, SSIM, LPIPS, FID; PushT goal-conditioned SR across latent spaces; latent MSE and cosine distance for GT, zero and random actions; action-embedding RMS | Tab. 1–7 | **Yes (latent MSE and cosine, with a GT/zero/random-action control)** |
| Reconstruction or Semantics? (`nilaksh2026reconstruction`) | Bridge V2, World Arena | PSNR, SSIM, LPIPS, t-LPIPS, FID, FVD, each against its encoder's reconstruction ceiling; VLA SR scored by a VLM; inverse-dynamics Pearson r at k=1,4; success-probe accuracy; 95% bootstrap CIs | Tab. 1, 8–15 | Indirectly (inverse-dynamics recovery from rollouts) |
| Action-relevant latents (`yeom2026actionrelevant`) | LIBERO, MetaWorld, CALVIN | Inverse-dynamics action R²; robustness under 7 visual perturbations; temporal cosine collapse | Tab. 1–13 | No |

## B. Generative / pixel world models (complementary; not compute-matched)

| Method | Benchmarks | Exact metrics reported | Where |
|---|---|---|---|
| NWM (`bar2025nwm`) | RECON, HuRoN, SCAND, TartanDrive, Go Stanford | LPIPS, DreamSim, PSNR at 4 s; FID/FVD (FVD compared with DIAMOND); navigation ATE/RPE; runtime | Tab. 2, 4, 5, 7, 8; Fig. 6 |
| Anchored NWM (`luan2026anchorednwm`) | RECON, SCAND, HuRoN, TartanDrive | LPIPS, FID and Sampson (epipolar) error at horizons 1–16 s; relative gain over NWM; MEt3R; ATE with CEM | Tab. 1–7 |
| Ctrl-World (`guo2026ctrlworld`) | DROID (third-person and wrist views), real robot | PSNR, SSIM, LPIPS, FID, FVD for 10-s rollouts; instruction following and SR, real vs. imagined; policy improvement SR | Tab. 1–7 |
| IRASim (`zhu2025irasim`) | RT-1, Bridge, Language-Table, RoboNet; PushT; real robot | PSNR, SSIM, Latent L2, FID, FVD (Latent L2 and PSNR are primary); simulator vs. model SR correlation; PushT IoU | Tab. 1–6 |
| IWS (`wang2026iws`) | IWS PushT, Box, Rope, and others | RGB MSE, LPIPS, FID, PSNR, SSIM, UIQI, FVD over 192-step predictions; policy scores | Tab. I |

## C. Surgical world models

| Method | Data | Exact metrics reported | Where |
|---|---|---|---|
| Open-H-Embodiment / Cosmos-H-Surgical-Simulator (`openh2026embodiment`) | Open-H surgical mixture (9 platforms, incl. Hamlyn dVRK) | Per-frame pixel **L1** (in [0,1]) and **SSIM** for 72-frame (6-chunk) open-loop replays of held-out recorded actions; GR00T-H task SR | Results text, Tab. S4 |
| Cosmos-H-Surgical (`he2025cosmoshsurgical`) | SATA; real dVRK trajectories | FVD plus VBench dynamic degree, imaging quality and overall consistency; policy SR after few-shot fine-tuning | Tab. 1, 2 |
| SurgVista (`pan2026surgvista`) | SurgWorld-Bench | Instrument and tissue point-tracking AJ and δ_avg^vis; PSNR, SSIM, LPIPS; temporal smoothness; short and long horizons | Tab. 1 |
| SurgWM (`koju2025surgwm`) | SurgToolLoc-2022 | PSNR, SSIM, FVD_10 | Tab. 2 |
| SutureBot (`haworth2025suturebot`) | dVRK suturing | Subtask SR; insertion-point error (mm); time | Tab. 1–4 |

## Implications for our tables

1. **Primary forecasting table (DROID-test / Hamlyn / IWS):** the only close methods that score raw feature forecasts are RLA-WM (DINO L1), Nano WM (latent MSE and cosine) and JEPA-WMs (embedding error as a validation metric). Report **standardized MSE, raw L1 and cosine distance** per horizon, with endpoint and mean-over-horizon columns, and a **Copy-Last/persistence** row, as DINO-world does.
2. **Downstream-probe metrics** are how DINO-Foresight, DINO-world, VFMF and FlowWM validate feature forecasts. Our equivalent is end-effector and kinematics linear probes (MSE, Pearson r, as LeWM and Delta-JEPA report). Report the probe on ground-truth features as a ceiling, following Reconstruction-or-Semantics.
3. **Planning table:** SR as mean ± std over 3 seeds and 50 episodes (LeWM, Delta-JEPA protocol), stated CEM budget, and planning time. Use CD only if Rope/Granular are added.
4. **Pixel metrics (decoded):** LPIPS and SSIM, plus PSNR and FVD, are the common set (DINO-WM, RLA-WM, Ctrl-World, IRASim, NWM). For Hamlyn, the directly comparable Open-H protocol is **pixel L1 + SSIM on open-loop replays of recorded actions**.
5. **Efficiency:** FLOPs per step (RLA-WM, DDP-WM), throughput and MPC loop time (DDP-WM), planning time (LeWM, V-JEPA 2-AC).
6. **Action sensitivity:** Nano WM's GT/zero/random-action latent-error control and Delta-JEPA's state-delta probes are the published precedents for our shuffled-action diagnostic.
