# LapGym tissue-positioning extension

This isolated Python 3.10 environment executes **simulated deformable-tissue
positioning** through the MIT-licensed [LapGym environment](https://github.com/ScheiklP/sofa_env)
at commit `85bf7e05dd088b824794dda0046679df13b13e6e`. SOFA v24.06 has its own
LGPL notice in the downloaded distribution; it is not relicensed by this project.
Python dependencies are pinned in `requirements.lock.txt`. The project Python
3.11 environment and frozen original experiment code are unchanged.

```bash
environments/surgery/setup.sh
environments/surgery/run_python.sh scripts/extensions/profile_surgery.py
environments/surgery/run_python.sh scripts/extensions/collect_surgery.py --workers 4
```

The setup uses the official 291 MB Linux SOFA binary archive, verifies SHA-256,
and installs CPU Open3D because upstream imports it for I/O. No `sudo` or global
package changes are required. Headless rendering uses Pyglet/EGL. **The current
dataset's reproducible rendering stack is NVIDIA RTX 5000 Ada / driver 570.195.03,
requiring an allocated GPU for matching evaluation.** Pyglet explicitly selects
an EGL device; `LIBGL_ALWAYS_SOFTWARE=1` does not force NVIDIA EGL onto Mesa.
The login host's Mesa renderer has exact physical states but different raster
pixels, so it cannot substitute for this dataset's recorded visual references.
The launcher now explicitly pins
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
and fails clearly if it is absent. RPC reset responses and stderr logs include
the actual OpenGL vendor, renderer, driver/version string and host. The pinned
runtime passed a fresh allocated-node reference replay with unchanged goal RGB.
Real image
rendering, physical response, and replay checks are recorded by the profile
script; dependency imports alone do not establish runtime viability.

## Exact simulator contract

- Native timestep 0.1 s, frame skip 1, maximum gripper velocity 10 mm/s per axis.
- Two commanded normalized controls in [-1, 1]. The hidden gain is applied and
  clipped before passing the control to upstream. The resulting supplied action
  is recorded separately from actual workspace-clipped gripper displacement.
- Native success is Euclidean **XZ tissue-point to desired-target distance
  <= 0.002 m**. The learned model/policy receives neither these coordinates nor
  the hidden gain. State is allowed only for collection/auditing/scoring.
- Rendered RGB 128×128, canonical pixels stored. Photometric transforms belong
  in the common paired-view loader; current and image-goal views get the same
  transform at evaluation.
- Fixed-length forecasting collection deliberately continues after a target
  success. Policy evaluation must implement its declared success stopping rule.

## Dataset and splits

`collect_surgery.py` creates 192 train / 48 validation / 48 development / 96 test
episodes by default (384 total), each 200 native controls. Gains are 1.0, 0.75,
1.25; the same exploration seed is paired across gains **within** each split.
Seeds never cross splits. Each model-facing episode stores only:

```
images: uint8 [41, 128, 128, 3]
actions: float32 [40, 10]  # 5 chronological commanded 2D controls per image interval
```

`audit/*.npz` separately retains all 201 native RGB frames, commands, supplied
controls after gain/clipping, actual gripper displacement, tissue/target poses,
native success and distance, action validity, and deformation stability. The
model-facing loader must never load these privileged arrays. Per-episode JSON
sidecars pin configuration and file hashes. The manifest is written only after
the requested collection completes, preventing partial-data training by accident.

Forecast collection retains LapGym's native random target sampler and **does
not supply a reference image goal**. Planning uses the separate validated
reachable-reference protocol below, rather than silently treating a random
trajectory endpoint as success for an unrelated desired target.

## Reachable image-goal protocol

`profile_surgery.py` records a real exploration trajectory and its actual final
tissue target position, resets a fresh simulator with the same seed, assigns that
position to the native desired target, and replays the recorded commands. This
produces a real goal image with the same visible marker as the reset observation.
It checks that changing the desired marker leaves physical response unchanged,
and that a third fresh replay reproduces the goal-marked RGB sequence exactly.
The final reference frame must satisfy the native 2 mm criterion. This is a
**reachable-endpoint goal distribution**, not an exact reproduction of the
original randomly sampled LapGym target distribution. The reference is never
input to dynamics-context inference.

Reports include actual timings and failures; these are engineering validation,
not learned-policy performance or clinical surgical evidence.

## Torch-to-SOFA process boundary

Run `environments/surgery/run_python.sh scripts/extensions/serve_surgery.py` as
a persistent subprocess. JSON lines on stdin/stdout are the protocol; native
SOFA logging goes to stderr. A reset request contains `op: "reset"`, `seed`,
`dynamics_id` (0/1/2), `image_size`, `goal_state` (recorded XYZ endpoint), and
`max_steps: 200`. A step request contains `op: "step"`, `actions` (one to five
commanded 2D controls), and `stop_on_success: true`. Responses contain raw RGB
bytes as base64, shape/dtype, actually attempted commanded controls, per-native
step metrics, and termination/truncation flags. Invalid workspace moves,
unstable deformation, native success, and budget exhaustion stop evaluation.
An already successful initial task returns zero executed controls. Close with
`op: "close"`. A failed request returns `ok: false` with its error.

The metrics are privileged diagnostics for the evaluation harness. Feed only
the RGB observations and recorded commanded controls to the learned policy.
`test_surgery_rpc.py` exercises this boundary against the real simulator.

## Full audit and registered planning goals

`validate_surgery_data.py` verifies every data/audit hash, array alignment,
grouped commands, clipped hidden-gain mapping, native metric computation,
whole-episode splits, and matched initial states/commands across gains.

`prepare_surgery_goals.py` registers all development/test episodes whose
**native-200 endpoint** is at least 4 mm from the initial tissue point and whose
entire 200-control prefix contains valid, stable motion. It records every
exclusion before replay, with no access to model results. For each eligible
episode, a fresh simulator replays all recorded controls with the endpoint as
the native desired target. Every tissue/gripper state is compared to the
original episode. Pixels outside the original/new marker disks must match
exactly, and the desired target must stay constant throughout. Actual initial
and goal PNGs, commanded controls, checksums, and per-episode verification are
saved under `planning_goals/{development,test}`. The common evaluation budget
is 200 native controls **including ten shared support controls**.

For compatibility with the frozen generic trainer, run
`scripts/extensions/normalize_surgery_manifest.py` in the main Python environment
after initial dataset preparation/caching. It performs the recorded `env` →
additional `environment` alias migration, verifies all unchanged payloads and
weights, recomputes training-only statistics, and preserves old metadata plus an
explicit migration receipt. It changes no RGB, commands, goals, or feature values.
