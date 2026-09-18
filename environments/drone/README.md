# Visual drone extension runtime

This directory isolates a **fixed external-camera planar drone goal-reaching**
task. It does not implement onboard obstacle navigation or a learned flight
stabilizer. `CtrlAviary` applies actual PyBullet CF2X quadcopter physics; the
unchanged upstream `DSLPIDControl` stabilizes attitude and altitude for all
methods. The learned planner sees RGB, previous commanded actions, and an
observed goal image. Low-level PID state access is part of the common simulator
interface, never a privileged learned-policy input.

## Provenance and installation

Official repository: <https://github.com/learnsyslab/gym-pybullet-drones>
(the former `utiasDSL` URL redirects here). Revision:
`7ebad1ecabd28a7000add2d05f888aa2e837c2cc`, 2026-09-06. MIT license, unchanged
at `external/gym-pybullet-drones/LICENSE`. Cite Panerati et al., IROS 2021,
*Learning to Fly—A Gym Environment with PyBullet Physics for Reinforcement
Learning of Multi-agent Quadcopter Control*, DOI 10.1109/IROS51168.2021.9635857.

The complete latest upstream package declares Python 3.12 and includes learning
and GUI dependencies. This project uses the unchanged control/physics source
subset with a separately validated Python 3.11 runtime and pinned minimal
dependencies. This is a local compatibility configuration, not a claim that
upstream supports its entire package on Python 3.11. PyBullet 3.2.7 has a wheel
for this interpreter. No dependency or code in the existing project `.venv`
was changed.

From the project root, with `uv` available:

```bash
uv venv --python 3.11 environments/drone/.venv
uv pip install --python environments/drone/.venv/bin/python -r environments/drone/requirements.lock.txt
export PYTHONPATH="$PWD/src:$PWD/external/gym-pybullet-drones"
environments/drone/.venv/bin/python -m pytest scripts/extensions/test_drone.py -q
sbatch scripts/extensions/collect_drone.slurm
```

The installed local runtime also has a `.pth` containing those two source roots;
the explicit `PYTHONPATH` makes the same instructions portable.

## Task and data contract

- Images: canonical 128×128 RGB from CPU PyBullet TinyRenderer.
- Native commands: two normalized XY velocity requests in `[-1,1]`, at 10 Hz.
- Low-level controller: upstream PID at 120 Hz; physics at 240 Hz; desired
  altitude 0.65 m. Nominal XY velocity scale is 0.18 m/s.
- Hidden command-calibration gains: 1.0, 0.75, 1.25. Gain 1.5 is reserved in
  the adapter, but is **not collected or evaluated in version 1**. This shift
  changes the requested velocity before PID; it does not claim altered mass,
  wind, motor thrust, or aerodynamics.
- Training trajectories: 200 native commands, grouped in blocks of five into
  40 transitions, with 41 RGB images and actions of shape `[40,10]`.
- Dataset: 192 train, 48 validation, 48 development, and 96 fresh test episodes,
  balanced across three gains. Initial-state and collector seeds may be paired
  across gains within a split; seed families never cross split boundaries.
- Final 20 commands hold zero desired velocity. The goal is this **actually
  observed settled endpoint**, stored with its genuine image and simulator
  state. The future endpoint is a task goal only, never a temporal context
  observation. Easy/initially solved episodes must be reported or excluded by
  an explicit protocol before model comparison.
- Success requires XY error <0.04 m, speed <0.06 m/s, altitude error <0.05 m,
  and no crash/workspace escape. These are newly specified task criteria,
  **not a reproduction of an upstream benchmark success definition**.

Model episode NPZ files contain only `images` and commanded `actions`. Separate
`audit/` NPZ files contain native images, commanded and gain-scaled requests,
all actually applied rotor RPMs, physical states, and goal provenance. Shared
photometric appearance transforms are applied by the common data loader.

Run the full audit after collection:

```bash
environments/drone/.venv/bin/python scripts/extensions/audit_drone_data.py
```

## Validation and limits

Six adapter/data invariants passed before collection: exact repeated RGB/state,
monotonic physical response to hidden gain, grouped/native alignment, reachable
goal identity, no privileged fields in model NPZs, invalid-action rejection,
and verified resumability/corruption rejection. Three full-length physical
episodes (one per gain) completed in 5.66 s with three CPU workers on the login
host. Full-dataset throughput is reported separately after its Slurm job.

This is a compact new simulation transfer task built on established physics
and control code. It is not real-flight validation, and dataset completion is
not evidence of a learned planning improvement. Matched baselines, constant
context controls, independent seeds, and held-out conditions remain necessary.

## Isolated physics RPC for a GPU planner

Use this launcher from the project root to avoid mixing simulator packages into
the active model-training environment:

```bash
environments/drone/.venv/bin/python scripts/extensions/serve_drone.py
```

Send one JSON object per stdin line; stdout returns one JSON object per line.
All simulator import/diagnostic output goes to stderr, including C-level prints.
Supported requests:

```json
{"op":"reset","seed":3053000,"dynamics_id":0,"image_size":128,"max_steps":200}
{"op":"step","actions":[[0.3,0.2],[0.1,0.0]],"stop_on_success":true}
{"op":"close"}
```

`reset` also accepts a finite recorded 20-component `goal_state`, used exclusively
for scoring. Optional request `id` is echoed. Returned images contain `shape`,
`dtype: "uint8"`, and base64-encoded contiguous RGB bytes. `step` accepts one to
five commands, stops at success (if requested), crash, escape, or time limit,
and returns actually executed commands, per-step metrics, last RGB, current
metrics, termination/truncation flags, and cumulative native-step count. Invalid
requests return `ok:false` with a structured error. No simulator state or gain
label is returned to the learned planner.

Before any trained-model comparison, planning eligibility was fixed to initial
XY goal distance **at least 0.08 m** and initially unsolved. All 384 trajectories
remain in the dataset; excluded planning-task counts must be reported. The data
audit reports the full distance distribution without using model outcomes.
