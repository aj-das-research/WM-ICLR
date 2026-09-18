# CEM action-range audit — 2026-09-18

**The suspected ±1 z-score hard cap is absent.** The imported CEM solver uses the configured `Box` for its dimensions and type, without reading its low/high limits during sampling or elite updates. The current protocol correctly states that clipping occurs only after conversion to physical actions.

The source hashes, exact CPU fixture, results, and reproduction command are in [the evidence JSON](evidence/cem_action_range_audit_2026-09-18.json). This audit makes no evaluator, model, dataset, protocol, or running-process changes. The fixture runs on CPU with a toy cost, without loading a learned checkpoint or rolling out an environment.

## Code trace

| Location | Actual behavior |
|---|---|
| `src/shiftwm/evaluate.py:271–274` | Configures CEM with `Box(-1,1)` and grouped-action dimensions. |
| `external/stable-worldmodel/stable_worldmodel/planning/solver/cem.py:59–71` | Stores that object; uses its shape and checks its type. No bound conversion or clipping. |
| `cem.py:97–120` | Starts with unit sampling standard deviation; zero initial mean or retained warm-start mean plus a zero tail. The variable is named `var`, but it multiplies Gaussian samples as a standard deviation. |
| `cem.py:194–215` | Samples unbounded Gaussian proposals, scales/shifts them, and replaces proposal 0 with the current mean. Passes proposals directly to the cost. |
| `cem.py:244–253,280` | Updates the mean and population standard deviation from elites, without clipping. Returns the optimized mean. |
| `planning/solver/utils.py:42–64` | Preserves the supplied warm-start prefix and zero-pads missing steps for this non-actor cost. No clipping. |
| `src/shiftwm/evaluate.py:140–152` | Converts `native = action_mean + action_std * z`, then clips native controls. Uses those native actions in model rollouts. |
| `src/shiftwm/evaluate.py:291–296` | Applies the same conversion before execution; retains the remaining plan as unclipped z-score warm start. Random control samples uniformly over native bounds. |
| `src/shiftwm/generate.py:121–124` | Relative PushT and Reacher native bounds are `[-1,1]` per coordinate. |
| `src/shiftwm/model.py:143–148,192–196` | Normalizes the supplied native actions before action embedding, so this is not an extra scale multiplication inside the predictor. |

The actually imported solver is the pinned external checkout at commit `4821c8e6a3f0f83b7e6a80da3a757e026ea9026b`. Its `cem.py` SHA256 is `c66625d42cf02f912fc4c99d5874c57aba686d899e510d91630df210ece2ee89`. All relevant file hashes are recorded in the JSON. This verifies the current source/import path; it does not independently replay every historical process.

## Deterministic CPU verification

The fixture uses the same 300 samples, 30 iterations, 30 elites, horizon 5, and action block 5, with seed 123. A toy quadratic encourages the first coordinate at each horizon step to approach `z=3`.

- Initial proposals range from **−4.3686 to +3.6863**; **31.57%** of coordinates exceed one in absolute value.
- Returned optimized coordinates are **[3.0013, 2.9993, 2.9999, 2.7425, 2.8925]**. These are toy optimization outputs, not model performance.
- Replacing only `Box(-1,1)` with `Box(-100,100)` produces **bitwise identical** solver output.
- A warm-start prefix of `z=3` is preserved, with the last missing step zero-padded.
- Using the actual released PushT normalization, `z=3` maps to native **[0.61759, 0.62711]**; `z=-6` and `z=+6` map to native **−1** and **+1**, respectively.

Thus the full native action interval is representable. Reaching its endpoints requires roughly ±4.8 z-scores with the current PushT statistics. Finite CEM sampling is not guaranteed to discover a useful sequence anywhere in that interval.

## Proven distribution difference, unproven failure mechanism

For the random Gaussian component of the first cold-start CEM iteration:

| Quantity | PushT x | PushT y | Native-uniform random |
|---|---:|---:|---:|
| Initial native standard deviation before clipping | 0.20847 | 0.20675 | 0.57735 |
| Central 95% interval | [−0.4164, +0.4008] | [−0.3984, +0.4121] | [−0.95, +0.95] |
| Probability of `|a| > 0.5` | 1.654% | 1.565% | 50% |

CEM proposal 0 is the deterministic current mean, not a Gaussian sample. Subsequent proposal means/scales change with elites; warm starts also change the mean on later replans. Consequently these initial-distribution quantities must not be attributed to every proposed or executed action.

The narrower initial distribution is a verified property and a plausible finite-search limitation. It is **not** a demonstrated cause of low PushT success. There is no evidence here that changing proposal scale repairs planning, improves the learned cost, or changes the relative method ranking. Random and world-model policies differ in both distribution and optimization, so their outcome contrast does not isolate this mechanism. Any new scale study would need a separately identified development diagnostic; this audit changes no ongoing comparison.

## Reproduce

From the project root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -c 'import json; a=json.load(open("reports/evidence/cem_action_range_audit_2026-09-18.json")); n={}; exec(a["cpu_fixture"]["code"],n); print(json.dumps(n["fixture_result"],indent=2))'
```

The fixture code itself is embedded and hashed in the JSON. It uses source and normalization metadata only, without a model tensor load, GPU allocation, or result-file mutation.
