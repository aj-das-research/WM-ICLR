# Surgical control feasibility — 19 September 2026

## Recommendation

Use **LapGym's `TissueManipulationEnv`, in its tissue-aligned two-dimensional workspace**, for one bounded surgical simulation extension. The question is whether a short, shared interaction history helps a compact action-conditioned world model position a deformable tissue target under a new camera appearance and an unknown instrument-control gain. This provides actual executed controls, deformable-object responses and task success. It should be described as simulated tissue positioning, not clinical surgery, patient validation, autonomous operating-room performance, or evidence that SWoMo's generated videos constitute an executable robot environment.

This is a conditional feasibility recommendation based on inspected source code. **No simulator installation, training, memory profiling, dataset download or experiment was performed for this assessment.** A five-day completion estimate is therefore a work budget, not a measured guarantee.

## Pinned implementation and verified facts

The official [LapGym paper, JMLR 2023](https://www.jmlr.org/papers/v24/23-0207.html) links the established surgical learning framework. `git ls-remote` verified these current branch heads:

- `ScheiklP/sofa_env`: **85bf7e05dd088b824794dda0046679df13b13e6e**.
- `ScheiklP/sofa_zoo`: **f5386cc981e9abc8b3e32bdf2836404836bc4464**.

| Property | Verified primary evidence |
|---|---|
| Code permission | Both repositories have MIT licenses: [environment license](https://github.com/ScheiklP/sofa_env/blob/85bf7e05dd088b824794dda0046679df13b13e6e/LICENSE), [baseline license](https://github.com/ScheiklP/sofa_zoo/blob/f5386cc981e9abc8b3e32bdf2836404836bc4464/LICENSE). Preserve separate SOFA/dependency notices. |
| Installation | The [installer](https://github.com/ScheiklP/sofa_env/blob/85bf7e05dd088b824794dda0046679df13b13e6e/setup.py) explicitly requires Python 3.10 and x86_64 for its automated SOFA v24.06 binary installation. Use a separate environment, leaving our running Python 3.11 study unchanged. Headless graphics and system-library compatibility still need an actual run. |
| Executable actions | [Workspace definition](https://github.com/ScheiklP/sofa_env/blob/85bf7e05dd088b824794dda0046679df13b13e6e/sofa_env/scenes/tissue_manipulation/sofa_robot_functions.py) defines `TISSUE_ALIGNED` with two actions. This matches our existing five-by-two grouped action shape; it does not make existing PushT/Reacher weights surgical checkpoints. |
| Observations and success | The [task implementation](https://github.com/ScheiklP/sofa_env/blob/85bf7e05dd088b824794dda0046679df13b13e6e/sofa_env/scenes/tissue_manipulation/tissue_manipulation_env.py) supports RGB, state and RGBD modes, headless rendering, and a default target-position threshold of 0.002 m. Success is tissue-target to desired-target distance; deformation and motion diagnostics are exposed. |
| Actual deformation | The [scene](https://github.com/ScheiklP/sofa_env/blob/85bf7e05dd088b824794dda0046679df13b13e6e/sofa_env/scenes/tissue_manipulation/scene_description.py) constructs deformable tissue, a pre-attached gripper and configurable target/grasp locations. This is a positioning task, not a new cutting or suturing simulator. |
| Established task settings | The [published PPO task configuration](https://github.com/ScheiklP/sofa_zoo/blob/f5386cc981e9abc8b3e32bdf2836404836bc4464/sofa_zoo/envs/tissue_manipulation/ppo.py) uses 64×64 RGB, 0.1-second simulation steps, 10 mm/s maximum gripper velocity, 2 mm success tolerance, and a 500-call episode cap. |
| Full reproduction cost | The [baseline experiment configuration](https://github.com/ScheiklP/sofa_zoo/blob/f5386cc981e9abc8b3e32bdf2836404836bc4464/sofa_zoo/common/lapgym_experiment_parameters.py) specifies 10 million training steps and eight parallel environments. Reproducing that entire PPO study is outside this proposed five-day extension. |

I found training code, but did not verify a downloadable policy checkpoint for this exact task in `sofa_zoo`. The proposal therefore does not depend on an unverified pretrained surgical controller.

## Bounded experiment

**Proposed task:** RGB-only tissue-target positioning, with the native success tolerance retained. Keep the existing compact model family and shared frozen visual coordinates across methods. Train a surgical-domain donor/predictor and matched adapters using the same generated transitions; do not relabel a PushT/Reacher checkpoint as trained for surgery. If the existing visual features cannot distinguish relevant tissue/target motion, representation training must be addressed before interpreting adaptation.

Use four prespecified evaluation cells: nominal appearance/nominal actuation, appearance shift only, actuation shift only, and their held-out combination. Appearance should be an image transform applied equally to current and goal observations. A bounded, hidden action-gain wrapper provides an explicitly artificial instrument-calibration shift; log both commanded and executed actions and keep the executed mapping hidden from the controller. Choose train/development/test gain ranges before measuring held-out success. A stiffness experiment is a later option only after checking that the changed parameter produces a measurable, stable action-response change; changing a material number is insufficient evidence of a meaningful dynamics intervention.

Generate a modest local transition dataset with scripted workspace motions and bounded exploratory actions. Keep collection-only simulator state separate from RGB policy inputs. Store full actions, render observations, reset seeds, parameters and physical state for scoring and reproducibility. Fit action normalization on training trajectories only; split by initial scene/trajectory, not overlapping image windows.

The goal adapter is a real implementation requirement: an image-goal controller needs a reachable reference image consistent with the task's native tissue-target position. Construct the reference in a separate identically seeded simulator using a recorded reachable trajectory, while retaining the same visible target definition. Do not substitute a fabricated target image or pass hidden tissue coordinates to the learned controller. Verify reset/replay and the native success criterion before scheduling expensive comparisons.

| Minimum comparison | Requirement |
|---|---|
| Methods | Framewise baseline; matched model trained with constant dynamics context; inferred-context method (ours). Same representation, data, predictor capacity, objective and selection rule where applicable. |
| Training repetitions | Three seeds per method, all selected by the same held-out recursive prediction metric. Nine compact runs total. |
| Forecasting | Every method on identical held-out histories and commanded actions; report one-step and recursive errors in common coordinates, with physical target error where a shared diagnostic head is justified. Include action-shuffled and zero-action checks for action dependence. |
| Closed-loop evaluation | Four cells × 16 paired initializations × three seeds × three methods = **576 episodes**. The 16 episodes are a minimum bounded study, with wide paired confidence intervals; enlarge only if measured throughput permits. |
| Interaction budget | Proposed **100 native calls**, including the same initial ten support actions for all methods. This is an explicitly budget-capped LapGym-derived experiment, not the original 500-call benchmark reproduction. Same CEM candidates/iterations/horizon across methods; any smaller budget is declared in advance. |
| Primary metric | Native tissue-position success, with paired uncertainty by episode and training seed. Report all four cells, not only the best shift. |
| Secondary metrics | Terminal target distance in mm, steps used, deformation diagnostic, invalid actions/simulation failures, wall time, and peak allocated GPU memory. Do not interpret simulator deformation as validated human tissue injury. |
| Mechanism controls | Matched trained constant-context control; separately labeled inference-time context permutation/zeroing, with their distribution-shift limitation. No future frames or goal image in context inference. |

## Five-day resource envelope and stop conditions

The available scheduling pattern is two ongoing workstation jobs plus one additional GPU-partition slot, with RTX 5000 Ada 32 GB devices observed in previous allocations. This verifies the hardware available to us, **not this new simulator/model's memory fit or speed**. Keep the existing studies running; use one spare GPU and modest CPU workers for this extension, then exploit released capacity only after allocation is actually granted.

- **Day 1:** isolated Python/SOFA setup, a complete deterministic episode/replay, correct goal construction and scoring, timed generation/planning, and a fixed small dataset. Reserve at most six hours for installation/integration uncertainty. Continue only if headless rendering and physical task response work reproducibly.
- **Day 2:** common compact surgical donor/features and matched seed-zero models; establish meaningful nominal prediction and control relative to action-shuffled/random controls. This is a model-capability check, not evidence of a positive adaptation result.
- **Days 3–4:** complete matched three-seed training and the prespecified evaluation matrix. Prefer cached visual features for training. A provisional envelope is 12–24 GPU-hours for compact training and 24–48 GPU-hours for evaluation; these are estimates to replace with Day-1 measurements, not claims of verified runtime.
- **Day 5:** independently validate source hashes, paired support/goals/actions and tables, render positive/negative examples, document the simulator-only limitations, and release local surgical weights/configs if training succeeded.

If installation, goal construction or nominal capability fails its Day-1/Day-2 gate, retain a clearly labeled feasibility/negative finding rather than spending all remaining time on a new diffusion stack or presenting generated surgical video as closed-loop control. An extension alone does not establish ICLR-level novelty or acceptance.

## Why the alternatives are not the first experiment

**MECLabTUDA SWoMo** is directly relevant related work and a possible later appearance generator: its [official code](https://github.com/MECLabTUDA/SWoMo) describes a neuro-symbolic, scene-graph/video-conditioned surgical diffusion pipeline and five training stages. Its [Hugging Face release](https://huggingface.co/SsharvienKumar/SWoMo) declares CC-BY-4.0 for the hosted release, while no separate GitHub code license was found. Processed videos/graphs and checkpoints are linked; manually annotated real segmentations and additional labels require author contact. These facts do not establish an available robot-action `step(a)` environment. Avoid adding diffusion retraining to this five-day control extension.

**SurRoL NeedlePick** is the fallback if SOFA installation fails. The official original [main branch](https://github.com/med-air/SurRoL/tree/a903fa6de40b41e1cad12ddd2d4a6758ec78e817) is MIT and provides PyBullet simulation, a five-dimensional position/yaw/jaw action and a scripted oracle. Its [demonstration generator](https://github.com/med-air/SurRoL/blob/a903fa6de40b41e1cad12ddd2d4a6758ec78e817/surrol/data/README.md) defaults to 100 resets. The default repository branch is now the larger `SR-VPPV` framework, so pin the intended version. Legacy Gym/Python assumptions, forced EGL initialization and action-shape changes add integration risk; the original TensorFlow/Baselines training stack is not needed merely to use the simulator.

**GAS/Dreamer surgical grasping** has a [primary RSS 2024 paper](https://arxiv.org/abs/2405.17940) and [official implementation](https://github.com/linhongbin/gas), including linked policy checkpoints. Its documented Python 3.9, TensorFlow 2.9, CUDA 11.3/cuDNN 8.2 environment and multiple submodules introduce a second learning stack. Checkpoint links are SharePoint rather than a verified local release, and I found no top-level code license. It is useful related work, but a risky first transfer dependency here.

**SofaGym** provides a [generic SOFA/Gym interface](https://github.com/SofaDefrost/SofaGym); its documented SOFA 23.06, Gym 0.21/SB3 1.7 and optional plugin stack is less directly aligned than LapGym's existing surgical task. Prefer the already implemented tissue scene.

**JIGSAWS** supplies synchronized video and measured manipulator kinematics, but the [official access page](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) requires an academic-use form and advertises an alternative download by email. Kinematics/gesture labels are not automatically executable robot command labels, and offline data alone supplies no new-action simulator. Do not make this external-access path a dependency of a five-day experiment. Similarly, surgical video datasets with action-recognition classes are not interchangeable with continuous control datasets.
