# Surgical extension implementation, 19 September 2026

The LapGym **TissueManipulationEnv** is installed and runs headlessly in an
isolated Python 3.10.21 / SOFA 24.06 environment. This is real deformable-tissue
simulation with executed 2D instrument commands, not generated surgical video.
No learned surgical policy result or surgical checkpoint is claimed yet.

## Completed

- Pinned upstream code: `ScheiklP/sofa_env` at
  `85bf7e05dd088b824794dda0046679df13b13e6e` (MIT). Official SOFA binary archive
  SHA-256 is `9d515e2f25f657c744821be8a5361e22803c18947b33af7a0b357c259202236a`.
  SOFA keeps its separate LGPL license. Executable setup and locked Python
  dependencies are under `environments/surgery`.
- Pyglet/EGL rendering at 128×128. Subsequent cross-node verification established
  that workstation runs use NVIDIA EGL despite `LIBGL_ALWAYS_SOFTWARE=1`;
  the original collection requested CPU resources only, but its renderer was
  not established as software-only. On `ws-l1-002`, initialized in 2.72 s
  after warmup and measured 92.57 native controls/s on the 30-control profile.
  Full independent reference replay had exactly equal physical states and RGB,
  with a native goal distance of zero. Profile timing is a small measurement,
  not the complete collection throughput.
- Full collection, Slurm **200059**: **384 episodes × 200 native controls =
  76,800 transitions**, collected in **214.0 s**, with eight CPU workers and no
  GPU request. Splits: 192 training, 48 validation, 48 development, 96 test.
- All **384 episodes / 128 matched gain groups** passed file-hash, finite-value,
  shape/dtype, grouped/native RGB/action, hidden-gain clipping, native-success,
  and disjoint-seed checks. Model files contain only RGB and commanded actions.
  Privileged states and supplied/realized controls are separate audit files.
- Gains 1.0 / 0.75 / 1.25 use identical initial states and commanded exploration
  per matched seed. Their measured endpoint differences from nominal have
  median **7.14 mm**, range **0.274–27.934 mm**.
- There were **zero unstable-deformation flags**, and **1,265 / 76,800 native
  controls** encountered a workspace constraint. These are retained, explicitly
  recorded forecasting transitions. They are excluded from planning-reference
  trajectories under the prespecified criterion.
- A JSON-line subprocess API bridges the Python 3.10 simulator to the existing
  Torch environment. All ten real-runtime RPC checks passed, including binary
  RGB framing, clean JSON stdout, exact budget stopping, native goal assignment,
  initial-task eligibility, errors, and graceful shutdown.

## Planning registration

Before any new model outcome is inspected, reference tasks are selected by this
rule: the native-200 endpoint is at least 4 mm from the initial tissue point and
all 200 recorded controls are valid and stable. Every excluded episode is
recorded with its reason. This leaves:

| Split | Gain 1.0 | Gain 0.75 | Gain 1.25 | Eligible | Excluded |
|---|---:|---:|---:|---:|---:|
| Development | 12 | 14 | 9 | 35 | 13 |
| Test | 27 | 30 | 20 | 77 | 19 |

The native desired target is set to the recorded endpoint, and an independent
simulator replays the full command sequence. Every tissue/gripper state is
checked against the original audit; all RGB pixels outside the old/new visible
target disks must remain byte-identical; the marker must remain at the same
position; and the resulting real reference frame must satisfy native success.
Actual initial/goal PNGs, commands, and hashes are stored alongside each record.
The registry manifests appear only after all selected references pass.

The evaluation budget is 200 native controls, including ten shared support
controls. Native success is XZ tissue-target distance <= 2 mm. The reachable
reference distribution differs from LapGym's original random-target sampler;
this is a declared image-goal extension, not a full original PPO reproduction.

## Evidence and practical limits

- `data/extensions/surgery_v1/manifest.json`: complete forecasting dataset.
- `reports/surgery/dataset_validation.json`: complete data audit.
- `reports/surgery/slurm-profile/verified-200059/profile.json`: allocated-node
  headless/replay measurement.
- `reports/surgery/rpc/validation.json`: live process-boundary tests.
- `data/extensions/surgery_v1/planning_goals/selection.json`: prespecified
  selected/excluded reference tasks, without any model outcome selection.
- `reports/surgery/implementation_manifest.json`: versions, licenses and source
  hashes. Original code and main `.venv` were not modified.

SOFA emits upstream topology-handler, redundant-mass, degenerate-mesh and
rigid-compliance diagnostics at initialization. Full logs are retained; these
have not been silently suppressed or called proof of clinical fidelity. The
recorded simulation is finite, responds to actions/gain changes, satisfies the
native metric, and passes deterministic replay checks. A first cold profile had
a 0.0143 mm state discrepancy; the fresh post-initialization profiles are exact.

Remaining scientific work is shared representation/model training, matched
baselines, held-out forecasting, closed-loop planning, and confidence intervals.
The extension itself is not evidence that ShiftWM outperforms a baseline or
that simulated positioning transfers to clinical surgery.

## Cross-node integration correction

The login host uses Mesa llvmpipe (LLVM 20.1.2 / Mesa 25.2.8), whereas allocated
`ws-l1-006` uses NVIDIA RTX 5000 Ada / driver 570.195.03. Pyglet chooses an EGL
device explicitly; `LIBGL_ALWAYS_SOFTWARE=1` does not force its NVIDIA device
onto Mesa. The first development reference had **exactly identical physical
state**, but 4,196 pixels differed on the login renderer (mean absolute channel
difference 0.104675, maximum 164). Repeated resets did not remove this mismatch.

Both real surgical evaluator integration cases (recorded-action replay and
random control) passed on the allocated NVIDIA renderer, including exact initial
RGB, native budget accounting and reference reachability. Future evaluation
must use this rendering stack and an allocated GPU; no pixel tolerance was
introduced. Evidence lives in `reports/extension_evaluation_contract` and
`reports/extension_evaluation_surgery_allocated_contract_test.log`.

The frozen generic trainer required `environment`, while the collector emitted
`env`. The audited alias migration adds only `environment: surgery`, preserves
all original metadata, reruns the complete data audit and train-only action
statistics, and rebinds feature-cache metadata. All 1,536 data/cache/signature
files and 451 goal files were verified unchanged, with identical normalization
numbers and encoder weights. The old/new hashes and original copies are in
`reports/surgery/manifest_alias_migration/receipt.json`.
