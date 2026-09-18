# Drone extension: completed implementation and data collection

The new domain is **fixed external-camera visual planar drone goal-reaching**,
built on the unchanged CF2X physics and DSL PID implementation from
[gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones), commit
`7ebad1ecabd28a7000add2d05f888aa2e837c2cc` (MIT). It is not onboard navigation
or a real-flight experiment. No learned-model result is implied by collection.

| Item | Verified state |
|---|---|
| Simulator | Headless PyBullet TinyRenderer, real upstream physics and PID |
| Action | 2D commanded XY velocity; hidden request gain 1.0 / 0.75 / 1.25 |
| Observation | 128×128 canonical RGB; appearance shifts applied by shared loader |
| Dataset | 192 train +48 validation +48 development +96 fresh test episodes |
| Native data | 76,800 commands; 77,184 observed images/states |
| Grouped model data | 15,360 transitions; blocks of five controls, ten action values |
| Dataset size | 66,323,329 bytes for model and audit NPZs |
| Collection | Slurm 200056, 8 CPU, zero GPU, 144.15 s recorded runtime |
| Full audit | All 384 episodes passed; all hashes, RGB/action alignment, state isolation, goals and seed boundaries |
| Isolated runtime | Python 3.11 + locked control dependencies, no current environment mutation |
| Physics RPC | Implemented and tested, isolated Python process with clean JSON stdout |

Model NPZ files contain only observed RGB and commanded actions. Native images,
gain-scaled requests, all low-level RPMs, and physical states are stored in
separate audit NPZs. The native state supports the common low-level PID,
collector, and metric calculations; it never enters learned-policy inputs.

Each goal is a genuinely observed final frame/state after 20 settling commands.
Goal references do not become future context. Prespecified planning eligibility
is initial XY distance at least 8 cm and initially unsolved, established before
model comparison. All data remain retained. There are **40/48 eligible
development and 87/96 eligible test episodes**. The test set has 29 eligible
episodes per gain and the same 29 eligible initial seeds across all three gains.
The planning budget is 200 native controls including ten support controls.

New task success criteria require XY error <4 cm, speed <6 cm/s, altitude error
<5 cm, and no crash or workspace escape. These are new protocol definitions,
not claimed as an existing benchmark's canonical thresholds. The dynamics
intervention changes command calibration before PID; it does not claim altered
wind, motor thrust, mass, or aerodynamics.

Source and artifacts:

- `src/shiftwm/extensions/drone.py`: adapter and explicit physics/control contract.
- `scripts/extensions/collect_drone.py`: whole-episode collection and audit export.
- `scripts/extensions/serve_drone.py`: JSON-line physics server for an isolated GPU planner.
- `data/extensions/drone_v1/manifest.json`: immutable collected manifest.
- `reports/evidence/drone_data_validation.json`: all-episode verification.
- `reports/evidence/drone_planning_eligibility.json`: before-model selection and goal/action references.
- `environments/drone/README.md`: installation, task, data, and RPC interface.

Six adapter invariants and the RPC integration test passed, including exact
seed/action replay, physical gain response, actual goal identity, action
validation, corruption rejection, and correct episode/success stopping. The
next scientific deliverables are matched learned-model comparisons, independent
training seeds, held-out appearance/dynamics results, and checkpoint release
records. A completed data pipeline alone does not establish improvement.
