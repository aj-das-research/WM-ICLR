# Fresh real-DROID confirmatory comparison

Completed all 96 evaluations; no new training or fresh-data selection.

Primary calibrated ShiftWM (ours) vs calibrated Framewise: +0.742% error reduction; MSE difference -0.001103, paired 95% CI [-0.002145, -0.000290].

Negative MSE differences favor the first method. Positive reduction percentages favor ours. The primary endpoint is h5, not a mean across five frames.

## exterior_image_1_left/h5

65 eligible episodes, 52 sessions, 866 windows.

| Method | Original MSE | Calibrated MSE |
|---|---:|---:|
| Framewise | 0.148953 | 0.148757 |
| Constant dynamics | 0.148876 | 0.148695 |
| ShiftWM (ours) | 0.148514 | 0.147654 |
| Action-free | 0.149175 | 0.149096 |
| Persistence | 0.151979 | Same fixed baseline |
| Constant Velocity | 1.661833 | Same fixed baseline |

| Calibrated ours versus | Error reduction | Paired MSE difference 95% CI |
|---|---:|---:|
| calibrated/framewise | +0.742% | [-0.002145, -0.000290] |
| calibrated/constant_dynamics | +0.701% | [-0.002063, -0.000239] |
| calibrated/action_free | +0.968% | [-0.002701, -0.000410] |
| persistence | +2.846% | [-0.006816, -0.002035] |
| constant_velocity | +91.115% | [-1.703604, -1.331613] |

## exterior_image_1_left/h10

64 eligible episodes, 52 sessions, 801 windows.

| Method | Original MSE | Calibrated MSE |
|---|---:|---:|
| Framewise | 0.208396 | 0.207555 |
| Constant dynamics | 0.208241 | 0.207461 |
| ShiftWM (ours) | 0.210635 | 0.206815 |
| Action-free | 0.207603 | 0.207252 |
| Persistence | 0.210400 | Same fixed baseline |
| Constant Velocity | 6.011535 | Same fixed baseline |

| Calibrated ours versus | Error reduction | Paired MSE difference 95% CI |
|---|---:|---:|
| calibrated/framewise | +0.356% | [-0.002391, +0.000907] |
| calibrated/constant_dynamics | +0.311% | [-0.002265, +0.001007] |
| calibrated/action_free | +0.211% | [-0.002881, +0.002142] |
| persistence | +1.704% | [-0.009752, +0.002582] |
| constant_velocity | +96.560% | [-6.512552, -5.124078] |

## exterior_image_2_left/h5

65 eligible episodes, 52 sessions, 866 windows.

| Method | Original MSE | Calibrated MSE |
|---|---:|---:|
| Framewise | 0.185354 | 0.185222 |
| Constant dynamics | 0.185292 | 0.185169 |
| ShiftWM (ours) | 0.185000 | 0.184255 |
| Action-free | 0.185950 | 0.185900 |
| Persistence | 0.190841 | Same fixed baseline |
| Constant Velocity | 1.753509 | Same fixed baseline |

| Calibrated ours versus | Error reduction | Paired MSE difference 95% CI |
|---|---:|---:|
| calibrated/framewise | +0.522% | [-0.001971, -0.000174] |
| calibrated/constant_dynamics | +0.494% | [-0.001908, -0.000119] |
| calibrated/action_free | +0.885% | [-0.003148, -0.000448] |
| persistence | +3.451% | [-0.010103, -0.003590] |
| constant_velocity | +89.492% | [-1.831769, -1.340744] |

## exterior_image_2_left/h10

64 eligible episodes, 52 sessions, 801 windows.

| Method | Original MSE | Calibrated MSE |
|---|---:|---:|
| Framewise | 0.220646 | 0.219935 |
| Constant dynamics | 0.220554 | 0.219888 |
| ShiftWM (ours) | 0.222992 | 0.219325 |
| Action-free | 0.220577 | 0.220275 |
| Persistence | 0.226667 | Same fixed baseline |
| Constant Velocity | 6.012756 | Same fixed baseline |

| Calibrated ours versus | Error reduction | Paired MSE difference 95% CI |
|---|---:|---:|
| calibrated/framewise | +0.277% | [-0.002464, +0.001283] |
| calibrated/constant_dynamics | +0.256% | [-0.002382, +0.001290] |
| calibrated/action_free | +0.431% | [-0.003780, +0.001869] |
| persistence | +3.239% | [-0.014428, -0.000969] |
| constant_velocity | +96.352% | [-6.771271, -4.937745] |

Secondary intervals are descriptive and have no multiplicity adjustment. All metrics, seed values, comparisons, action reversal diagnostics and source hashes are in the accompanying JSON.

Small session-disjoint subset; not proven scene/object disjoint; observational feature forecasting, not physical control or novel calibration

Evaluation freeze SHA256: `5274eebfdbe441a0ef15a50e277cd0538972333d1b992e18cef9fe2319c0bf8d`
Result SHA256: `d1d39484ffc35bf0f4a5a4d8f9cc5058e5f0f4ff54964bc486103567a7a255b5`
