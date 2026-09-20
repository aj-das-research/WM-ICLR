# Adapted DINO-WM: complete raw-coordinate DROID follow-up

posthoc raw-coordinate/recipe follow-up on DROID development with an adapted official DINO-WM baseline; not an official benchmark/SOTA reproduction; not a single-factor intervention.

All six raw-coordinate external runs completed100 epochs; all15 fixed internal runs completed30 epochs. Two training objectives are reported separately; both selected checkpoints with the same recursively evaluated H10 development metric.

| Method | Native h5 | Native h10 | Original2x2 h5 | Original2x2 h10 |
|---|---:|---:|---:|---:|
| Autoregressive | 0.149573 | 0.203950 | 0.148577 | 0.202916 |
| Additive anchor | 0.149606 | 0.200244 | 0.148705 | 0.198704 |
| ShiftWM (ours) | 0.145074 | 0.193141 | 0.147921 | 0.197766 |
| Context-off (ours, ablation) | 0.145143 | 0.193330 | 0.147635 | 0.197553 |
| Action-free (ours, ablation) | 0.149480 | 0.202249 | 0.152414 | 0.206188 |
| Adapted DINO-WM: raw shifted one-step (100 epochs) | 0.231163 | 0.380522 | 0.250955 | 0.416676 |
| Adapted DINO-WM: raw recursive H10 (100 epochs) | 0.243828 | 0.353244 | 0.268923 | 0.390894 |
| Persistence (same recorded anchor) | 0.164028 | 0.226129 | 0.163156 | 0.225622 |

Lower MSE is better. Native4x4 and original2x2 are distinct standardized coordinate systems; their magnitudes are not interchangeable.

## Primary descriptive comparisons

| Coordinate system | Comparator | Horizon | ShiftWM reduction | ShiftWM−external MSE [95% interval] |
|---|---|---:|---:|---:|
| native_mse | Adapted DINO-WM: raw shifted one-step (100 epochs) | 5 | +37.242% | -0.086088 [-0.096525, -0.076003] |
| native_mse | Adapted DINO-WM: raw shifted one-step (100 epochs) | 10 | +49.243% | -0.187381 [-0.204622, -0.170652] |
| original_2x2_mse | Adapted DINO-WM: raw shifted one-step (100 epochs) | 5 | +41.057% | -0.103034 [-0.117194, -0.089810] |
| original_2x2_mse | Adapted DINO-WM: raw shifted one-step (100 epochs) | 10 | +52.537% | -0.218910 [-0.242073, -0.196768] |
| native_mse | Adapted DINO-WM: raw recursive H10 (100 epochs) | 5 | +40.501% | -0.098754 [-0.110469, -0.087725] |
| native_mse | Adapted DINO-WM: raw recursive H10 (100 epochs) | 10 | +45.324% | -0.160103 [-0.176898, -0.143468] |
| original_2x2_mse | Adapted DINO-WM: raw recursive H10 (100 epochs) | 5 | +44.995% | -0.121002 [-0.137321, -0.105825] |
| original_2x2_mse | Adapted DINO-WM: raw recursive H10 (100 epochs) | 10 | +49.407% | -0.193128 [-0.217199, -0.170298] |

Negative ShiftWM−external differences favor ShiftWM. Point reductions and signed intervals retain adverse or inconclusive results. Intervals resample matched training seeds and recording sessions10,000 times, unadjusted.

All ten horizons, four metrics (including both persistence controls), seven method means, per-seed summaries and ten paired method comparisons remain in finalization.json. All21 primitive window/episode ledgers remain at their source-bound evaluation paths. No RGB quality or physical control outcome was measured; no causal or official-benchmark SOTA claim follows.
