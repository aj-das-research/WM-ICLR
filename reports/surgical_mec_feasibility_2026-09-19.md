# MEC Lab surgical world-model feasibility — 19 September 2026

This is a primary-source release audit, not a reproduction or new experimental result. No datasets, model weights, environments or jobs were installed. Public metadata, small source files, GitHub HEAD commits and Hugging Face revisions are recorded in `reports/evidence/surgical_mec_feasibility_2026-09-19.json`.

## Recommendation

**SWoMo is the strongest direct MEC Lab match, but its currently verified release is best used as a surgical video renderer/representation source, not assumed to be a ready-made agent benchmark.** For the five-day project, place compact adaptation and closed-loop evaluation in a simulator with a verified action/step API; make a MEC Lab rendering/appearance study a bounded additional experiment. Reproducing an entire new surgical diffusion model is not the critical path. This is an engineering recommendation from the release audit, not a measured runtime prediction.

Our contribution must be narrower than separating appearance and dynamics: SWoMo already does this explicitly. A possible contribution is **compact online adaptation of action-conditioned prediction under visual and physical shifts**, tested against a nominally matched learned constant-context model and frozen predictor. Our existing follow-up does not yet demonstrate consistent inferred-context benefit; a surgical application cannot by itself resolve that scientific weakness.

## Exact relevant releases

| Project | Paper / public artifact dates | Released assets and role | License status |
|---|---|---|---|
| [SWoMo](https://github.com/MECLabTUDA/SWoMo) | [Paper](https://arxiv.org/abs/2605.16530): 15 May 2026, revised20 May. Author project reports MICCAI2026 Spotlight. GitHub last push28 Aug2026; HF last update7 July2026. | Diffusion/graph training and inference, image/video/graph conditioning, Cataract-1K checkpoint stack and processed real/simulated videos. | HF card declares CC-BY4.0. Inspected GitHub root has no license file and GitHub API identifies none; do not assume the HF label licenses all code/dependencies. |
| [SG2VID](https://github.com/MECLabTUDA/SG2VID) | [Paper](https://arxiv.org/abs/2506.03082): 3 June2025, revised13 June; MICCAI2025 Oral. HF update28 Aug2025; GitHub last push11 March2026. | Scene-graph-conditioned video synthesis; checkpoints and processed CATARACTS, Cataract-1K and Cholec80. Useful controlled-rendering predecessor, without a demonstrated released closed-loop policy API. | GitHub and HF: CC-BY4.0. |
| [SurGrID](https://github.com/MECLabTUDA/SurGrID) | IPCAI2025; International Journal of Computer Assisted Radiology and Surgery20(7):1421–1429. HF update11 June2025; GitHub last push11 March2026. | Scene-graph-to-image diffusion, interactive graph edits, pretrained components and processed CaDISv2. Single-image synthesis rather than learned temporal action dynamics. | GitHub and HF: CC-BY4.0. |
| [IntrekSAM](https://github.com/MECLabTUDA/IntrekSAM) | GitHub last push2 September2026. | SAM2-based video annotation tooling; supports data preparation, not a world model or agent. | MIT. |

Repository creation/push and HF creation/update dates are metadata dates, not proven dates when artifacts first became publicly downloadable. Exact commits and HF revisions are pinned in the evidence JSON.

## What SWoMo does, and what it does not establish

The [paper](https://arxiv.org/html/2605.16530v2) defines explicit tool-motion actions, a Godot rule-based transition function, scene graphs, then diffusion-based visual rendering. It evaluates synthesis/conditioning, unusual tool motions, cross-dataset style transfer and phase-recognition augmentation on cataract datasets. Training uses four A40 GPUs and128×128 sequences of16frames; this is not evidence of closed-loop robot control or a pretrained compact latent controller.

The inspected108-file GitHub tree contains graph/diffusion modules, training scripts and samplers, but no Godot project, GDScript, simulator meshes or interactive reset/step action API. None appeared in the HF inventory either. This bounded audit does not prove that no separate simulator release exists; it means we have not verified one and cannot schedule end-to-end SWoMo agent experiments as already reproducible. The [inference configuration](https://github.com/MECLabTUDA/SWoMo/blob/c99eba31131a8390268d870266e7e2ca225086b2/configs/inference/inference_img_graph_vid_cataracts.yaml) consumes a whole sequence of conditioning graphs and simulated frames, plus the initial image. Those future conditions are legitimate rendering inputs but cannot be supplied from held-out observed futures in an online prediction benchmark.

Real/simulated paired videos can support appearance-transfer evaluation. Recovered visual motion or a surgical phase label is not equivalent to a synchronized executable robot command. A deployable action-conditioned learner needs actual controls and aligned transitions, or a simulator where we choose and execute those controls.

## Download and compute reality

The following decimal-GB sizes are summed from public HF file metadata at the pinned revisions, without downloading archives. They include all variants and optimizer states where present.

| Release | Data files | All checkpoint files | Practical selection |
|---|---:|---:|---|
| [SWoMo HF](https://huggingface.co/SsharvienKumar/SWoMo/tree/814e54aeca90a8b2454b059c794aa4d28cd6790f) |95.679GB|44.813GB|Five inference component files sum8.029GB: one UNet, one ControlNet, two graph encoders, one VAE. Additional upstream text/tokenizer/config assets still needed; skip optimizer states. |
| [SG2VID HF](https://huggingface.co/SsharvienKumar/SG2VID/tree/869a3efbad4f505818a2094bbbfb831deff0697f) |52.018GB|96.834GB|Select one dataset and one conditioning variant. The CATARACTS processed subset totals about2.530GB, versus all three datasets. |
| [SurGrID HF](https://huggingface.co/SsharvienKumar/SurGrID/tree/f3d4a28fc48574571116e11da68ab4c1c459b91d) |3.061GB|3.784GB|Smallest MEC Lab artifact, but only image synthesis. |

SWoMo's public model endpoint is ungated. Real16fps segmentation masks and extra phase/tracking annotations are request-only according to the [author model card](https://huggingface.co/SsharvienKumar/SWoMo); its public masks_real entry is a readme, not the annotations. These missing labels affect reproducing every downstream result. Public archive availability also does not erase original dataset/dependency terms.

Our three independent RTX5000Ada32GB allocations are not equivalent to four jointly allocated A40s. Full diffusion retraining has no verified five-day fit. The released training config has a1,000,000-step maximum, which is a configuration ceiling, not an observed training duration. Inference uses50denoising steps per16frame clip. The8GB component files are storage size, not measured VRAM; single-GPU inference and adapter tuning remain plausible but unverified until memory/throughput are measured. Current scientific campaigns should retain their resources while one bounded allocation establishes the optional rendering path.

## A concrete five-day research scope

1. **Day1: establish runnable contracts.** Pin the selected simulator, recover real action conventions, generate and verify complete state/action/RGB trajectories, and reproduce a fixed controller baseline. For MEC Lab assets, test one released checkpoint variant and one small processed dataset subset; do not begin by fetching all archives. If the Godot simulator remains unavailable, restrict SWoMo to conditional rendering.
2. **Days2–3: compact-model study.** Reuse a frozen visual representation and train a small predictor/context module. Compare inferred and learned constant context, history-only/action-shuffled controls, and frozen predictor under independently specified visual and physical shifts. Run separate seeds on independent allocations; report actual optimization and inference cost.
3. **Day4: evaluation.** Use a freshly reserved final set of initial states and shifts. Separate conditional prediction, action sensitivity and closed-loop success. Rendering quality is an auxiliary measurement, not a proxy for control. No future scene graphs or recorded future frames enter the policy.
4. **Day5: analysis and release.** Preserve failures and paired uncertainty, finalize a compact checkpoint with documented encoder/actions/normalization and an offline simulator demo, and write limitations. Advancement requires an improvement beyond the matched constant-context control; success in one surgical task does not establish generality or clinical performance.

This schedule is conditional on measured data/simulator throughput and is not an acceptance or completion guarantee. A new surgical extension should strengthen a demonstrated method, not substitute medical relevance for a missing mechanism result. The parallel simulator audit covers LapGym/SurRoL; prior `reports/medical_world_fit.md` separately audits Open-H motor-action data and CathSim. No large model, dataset or simulator installation was performed here.
