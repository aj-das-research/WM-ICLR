# Adapted DINO-WM: complete DROID development comparison

posthoc development comparison with an adapted official DINO-WM baseline; not an official benchmark/SOTA reproduction.

All six external and15 internal runs completed30 epochs. Two training objectives are reported separately; both selected checkpoints with the same recursively evaluated H10 development metric.

| Method | Native h5 | Native h10 | Original2x2 h5 | Original2x2 h10 |
|---|---:|---:|---:|---:|
| Autoregressive | 0.149573 | 0.203950 | 0.148577 | 0.202916 |
| Additive anchor | 0.149606 | 0.200244 | 0.148705 | 0.198704 |
| ShiftWM (ours) | 0.145074 | 0.193141 | 0.147921 | 0.197766 |
| Context-off (ours, ablation) | 0.145143 | 0.193330 | 0.147635 | 0.197553 |
| Action-free (ours, ablation) | 0.149480 | 0.202249 | 0.152414 | 0.206188 |
| Adapted DINO-WM: official shifted one-step objective | 0.219267 | 0.355590 | 0.224146 | 0.369692 |
| Adapted DINO-WM: matched recursive H10 objective | 0.233920 | 0.331548 | 0.238623 | 0.339783 |
| Persistence (same recorded anchor) | 0.164028 | 0.226129 | 0.163156 | 0.225622 |

Lower MSE is better. Native4x4 and original2x2 are distinct standardized coordinate systems; their magnitudes are not interchangeable.

## Primary descriptive comparisons

| Coordinate system | Comparator | Horizon | ShiftWM reduction | ShiftWM−external MSE [95% interval] |
|---|---|---:|---:|---:|
| native_mse | Adapted DINO-WM: official shifted one-step objective | 5 | +33.837% | -0.074193 [-0.081406, -0.067144] |
| native_mse | Adapted DINO-WM: official shifted one-step objective | 10 | +45.684% | -0.162449 [-0.177366, -0.147795] |
| original_2x2_mse | Adapted DINO-WM: official shifted one-step objective | 5 | +34.007% | -0.076225 [-0.085821, -0.066967] |
| original_2x2_mse | Adapted DINO-WM: official shifted one-step objective | 10 | +46.505% | -0.171926 [-0.192165, -0.152256] |
| native_mse | Adapted DINO-WM: matched recursive H10 objective | 5 | +37.981% | -0.088846 [-0.098291, -0.079217] |
| native_mse | Adapted DINO-WM: matched recursive H10 objective | 10 | +41.746% | -0.138407 [-0.154174, -0.122164] |
| original_2x2_mse | Adapted DINO-WM: matched recursive H10 objective | 5 | +38.011% | -0.090702 [-0.102351, -0.079017] |
| original_2x2_mse | Adapted DINO-WM: matched recursive H10 objective | 10 | +41.796% | -0.142017 [-0.161950, -0.121868] |

Negative ShiftWM−external differences favor ShiftWM. Point reductions and signed intervals retain adverse or inconclusive results. Intervals resample matched training seeds and recording sessions10,000 times, unadjusted.

All ten horizons, four metrics (including both persistence controls), seven method means, per-seed summaries and ten paired method comparisons remain in finalization.json. All21 primitive window/episode ledgers remain at their source-bound evaluation paths. No RGB quality or physical control outcome was measured; no causal or official-benchmark SOTA claim follows.
