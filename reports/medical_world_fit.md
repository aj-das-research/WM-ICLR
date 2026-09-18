# Small world models in healthcare: feasibility study

Checked 2026-09-18. This is a source audit and proposal, not a reproduction result. No datasets, weights, environments, or training jobs were installed or run for this study.

## Recommendation

Medical world modeling is possible with actual action-conditioned public data. It should be an optional extension to a compact natural-control world-model paper, rather than a prerequisite for the five-day result. The most practical choices are one Open-H endoscopy subset for offline prediction and one CathSim task for closed-loop planning. Those answer different questions: offline action-conditioned prediction does not establish control success, while a medical simulator does not establish clinical effectiveness.

Static radiographs, histopathology patches, or independently sampled ultrasound frames do not supply the action/transition information needed for a controllable world model. Artificially ordering those images would not fix this. Medical video can support temporal prediction, but annotated surgical action categories should not automatically be treated as robot control commands or causal interventions.

## 1. Open-H: genuine recorded observations and motor actions

[Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) provides video paired with kinematics in LeRobot format. The hosting card lists CC-BY-4.0 and approximately 4.54 TB; the Hugging Face API returned `gated: false`, `private: false`, revision `e29dda7cabf2a2626634c7822db27695553ae523`. Download a named subset, not the entire collection. The [official contribution repository](https://github.com/open-h/open-h-embodiment) now targets v2; the released v1 data use LeRobot v2.1, so code and data formats must be pinned separately.

A concrete small candidate is the CUHK stomach-phantom endoscopy subset:

`Endoscopy/cuhk/openh_dataset_full/find_greater_curvature`

Its [dataset README](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/blob/main/Endoscopy/cuhk/openh_dataset_full/find_greater_curvature/README.md) and [metadata](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/blob/main/Endoscopy/cuhk/openh_dataset_full/find_greater_curvature/meta/info.json), fetched directly from the public resolve endpoints, specify:

- 462 episodes, 107,488 frames, 20 FPS, 640×480 RGB.
- Two continuous motor-speed commands, `m1_spd` and `m2_spd`.
- A 13-dimensional observation state including tracked position/orientation, motor position, and illumination `light_val`.
- Phantom navigation rather than patient treatment, with no recovery demonstrations.
- The released split is entirely training: `train: 0:462`, empty validation/test. We must create and publish episode-level splits and investigate session/operator grouping before evaluation.

The source describes a broader six-task collection containing 2,158 trajectories; that total is not the size of this one directory. Its timestamp audit explicitly does not establish all video/kinematics synchronization. Physical response delay and recording delay can be confounded; audit samples before attributing predictive errors to dynamics.

**Proposed use:** train or fine-tune a compact latent predictor from image history and true motor actions; assess multi-step representation/state prediction and action sensitivity on held-out episodes/tasks. Illumination metadata may help test appearance robustness. An action-shuffling and history-only baseline is essential: a model that ignores actions is not useful for planning. This dataset alone cannot measure online policy success or prove the outcomes of unobserved actions. A small subset's total byte size and loader throughput have not yet been measured.

**Reusable artifact:** endoscopy-specific encoder/predictor weights, action normalization, split manifest, and a clearly scoped model card. A natural-control LeWorldModel checkpoint cannot simply be assumed to transfer to endoscopy.

## 2. CathSim: controllable medical simulation

[CathSim](https://github.com/airvlab/cathsim) is an established endovascular navigation simulator using MuJoCo/dm_control with a Gymnasium interface. Its quickstart exposes step/reset, selectable vessel phantom/target, and optional image observations at configurable resolution. It includes training and trajectory recording code. Source is publicly accessible; the repository restricts use to non-commercial purposes, and its [terms](https://raw.githubusercontent.com/airvlab/cathsim/main/TERMS.md) prohibit direct application of trained agents to humans.

**Proposed use:** collect image/action trajectories and train a LeWorldModel-sized model, then evaluate model-predictive control to known navigation goals. Held-out vessel layouts and controlled camera/physical changes could assess transfer, but each perturbation needs an implemented, documented meaning. Use success, path length, interaction count, and available contact/force diagnostics rather than claim clinical safety.

**Why it is promising:** unlike offline clinical video, the simulator can test alternative actions from controlled initial conditions. **Five-day risk:** new environment integration, headless rendering, and meaningful trajectory coverage could consume the schedule. No pretrained LeWorldModel-compatible CathSim checkpoint was verified. A four-hour installation/data-quality gate is appropriate; count CPU simulation and GPU rendering time in the budget.

## 3. LapGym/SOFA: richer tissue dynamics, greater setup risk

[sofa_env](https://github.com/ScheiklP/sofa_env) supplies image-based Gym environments for surgical tasks such as tissue manipulation, grasp/lift/touch, cutting, and threading, under an MIT license. The [LapGym JMLR paper](https://jmlr.org/papers/volume24/23-0207/23-0207.pdf) establishes the benchmark, while the [API documents headless rendering](https://scheiklp.github.io/sofa_env/sofa_env.html). Installation uses SOFA binaries and a matching Python setup.

**Proposed use:** choose one spatial/manipulation task rather than train across the suite. Deformation makes it scientifically richer than a rigid-object extension. **Five-day risk:** SOFA/runtime integration and simulation throughput are unmeasured here; do not schedule it alongside CathSim and Open-H as three required medical benchmarks.

## Actual released medical world models, and why they are not the default base

[Cosmos-H-Surgical-Simulator](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator) has [public pretrained weights](https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator) and action-conditioned future-video generation based on Cosmos-Predict2.5-2B. The weights API is ungated, revision `32be28bac3d47aff3ab7fc1ff2f021c8b31a2ef3`. This is a substantial video diffusion model, not a compact latent predictor. Its model card reports A100 testing; we have not verified training or inference on our GPUs.

The newer [Cosmos-H-Dreams](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams), released July 23, 2026, provides a [2B streaming surgical world-model checkpoint](https://huggingface.co/nvidia/Cosmos-H-Dreams), specialized to dVRK tabletop suturing. It is a few-step causal student derived from the surgical simulator. Its code explicitly requires at least 12 GB VRAM, NVIDIA R580+ drivers/CUDA 13, and Python 3.12. Our previously measured driver is 570.195.03: memory may be sufficient, but the documented runtime does not match. Compatibility work remains unverified; do not make this the critical path. A generated simulator should not be its own sole evaluator of our policy.

These releases show that action-conditioned medical world models are a real active direction. They also mean that simply training a medical world model or distilling a surgical simulator is not sufficient novelty.

Two additional recent papers deserve awareness but were not verified as usable foundations in this audit:

- [Action-Conditioned World Model for Goal Plane Probe Guidance in Robotic Ultrasound](https://arxiv.org/abs/2607.21918), July 2026: a self-collected ultrasound dataset and closed-loop probe-guidance work. No official public code/data release was verified through this search.
- [EndoWAM](https://arxiv.org/abs/2608.01221), August 2026: grounded endoscopic world/action modeling and EndoMotion data. A usable official code/checkpoint/data endpoint was not established here; do not list it as reproducible merely because a paper exists.

## Scope appropriate to the user’s resources

For three independent 32 GB workers and five days, make compact natural-control models and closed-loop planning the main result. If those baselines reproduce, use either a small Open-H subset for an additional domain test or CathSim for a second control environment. Reserve roughly 20–30 GPU-hours for this optional medical extension only after throughput measurement; this is a planning estimate, not a verified requirement.

The reusable release should be explicit about scope: small encoder/dynamics checkpoints, their action conventions, normalization, training environments, and planning API. A common architecture can transfer through adaptation; one set of weights should not be advertised as universal across natural scenes, surgical tools, and ultrasound without evidence.

