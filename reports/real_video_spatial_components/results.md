# Spatial component follow-up

Follow-up after revealed controls; original validation only. Mixing includes learned gate, identity bias and ~1.84% more active parameters. Features, not RGB or physical control. No fresh-test or SOTA claim.

All six new runs and six previously revealed controls completed 30 epochs. All six new inference packages passed exact relocated CPU parity.

| Mode | Native h5 | Native h10 | Original 2x2 h5 | Original 2x2 h10 |
|---|---:|---:|---:|---:|
| anchored_additive | 0.149606 | 0.200244 | 0.148705 | 0.198704 |
| bounded_additive | 0.149347 | 0.200010 | 0.148430 | 0.198261 |
| unbounded_transport | 0.145037 | 0.193274 | 0.147812 | 0.197706 |
| transport | 0.145074 | 0.193141 | 0.147921 | 0.197766 |

Endpoint errors: average windows within episode, then equal episodes and training seeds. Selection remains window-weighted mean across all ten query steps.

| Contrast | Metric | Horizon | Signed difference [paired 95% interval] |
|---|---|---:|---|
| bounding_without_mixing | native_mse | 5 | -0.000259 [-0.000535, +0.000017] |
| bounding_without_mixing | native_mse | 10 | -0.000234 [-0.000649, +0.000160] |
| bounding_without_mixing | original_2x2_mse | 5 | -0.000275 [-0.000533, -0.000009] |
| bounding_without_mixing | original_2x2_mse | 10 | -0.000443 [-0.000907, -0.000009] |
| bounding_with_mixing | native_mse | 5 | +0.000037 [-0.000309, +0.000387] |
| bounding_with_mixing | native_mse | 10 | -0.000133 [-0.000606, +0.000263] |
| bounding_with_mixing | original_2x2_mse | 5 | +0.000109 [-0.000287, +0.000559] |
| bounding_with_mixing | original_2x2_mse | 10 | +0.000060 [-0.000449, +0.000652] |
| mixing_without_bounding | native_mse | 5 | -0.004569 [-0.005870, -0.002917] |
| mixing_without_bounding | native_mse | 10 | -0.006970 [-0.008666, -0.005013] |
| mixing_without_bounding | original_2x2_mse | 5 | -0.000893 [-0.002359, +0.001162] |
| mixing_without_bounding | original_2x2_mse | 10 | -0.000998 [-0.002923, +0.001653] |
| mixing_with_bounding | native_mse | 5 | -0.004273 [-0.005650, -0.002542] |
| mixing_with_bounding | native_mse | 10 | -0.006868 [-0.008690, -0.004723] |
| mixing_with_bounding | original_2x2_mse | 5 | -0.000509 [-0.001935, +0.001618] |
| mixing_with_bounding | original_2x2_mse | 10 | -0.000495 [-0.002513, +0.002224] |
| mixing_x_bounding_interaction | native_mse | 5 | +0.000296 [-0.000115, +0.000778] |
| mixing_x_bounding_interaction | native_mse | 10 | +0.000101 [-0.000471, +0.000710] |
| mixing_x_bounding_interaction | original_2x2_mse | 5 | +0.000384 [-0.000061, +0.000938] |
| mixing_x_bounding_interaction | original_2x2_mse | 10 | +0.000503 [-0.000175, +0.001344] |

Negative edge differences favor the first named component arm. Negative interaction means bounding reduces error more with the mixing package. Ten thousand paired recording-session and seed bootstrap draws (seed 173); intervals are exploratory and unadjusted. No run, epoch, horizon or failed comparison was filtered by outcome.
