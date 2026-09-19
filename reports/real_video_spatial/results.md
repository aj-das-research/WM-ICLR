# Spatial architecture development results

Original validation only. All15models completed30epochs; all15relocated CPU inference packages passed exact prediction parity. No fresh test was used.

| Mode | Native h5 | Native h10 | Original2x2 h5 | Original2x2 h10 | Native h10 reduction vs autoregression |
|---|---:|---:|---:|---:|---:|
| autoregressive | 0.149573 | 0.203950 | 0.148577 | 0.202916 | +0.000% |
| anchored_additive | 0.149606 | 0.200244 | 0.148705 | 0.198704 | +1.817% |
| transport | 0.145074 | 0.193141 | 0.147921 | 0.197766 | +5.300% |
| context_off | 0.145143 | 0.193330 | 0.147635 | 0.197553 | +5.208% |
| action_free | 0.149480 | 0.202249 | 0.152414 | 0.206188 | +0.834% |

Window-then-episode averages, then equal training seeds. Cross-resolution metrics cannot be compared directly. Parameter counts, selected epochs, all per-episode horizon ledgers, registration, and parity proofs accompany this report. Transport and innovation bounds are not separately isolated by these five arms.

## Paired validation comparisons

| Metric | Horizon | Transport comparator | Relative reduction | Difference [paired95% interval] |
|---|---:|---|---:|---|
| native_mse | 5 | autoregressive | +3.008% | -0.004499 [-0.006132, -0.002491] |
| native_mse | 10 | autoregressive | +5.300% | -0.010809 [-0.012461, -0.009028] |
| original_2x2_mse | 5 | autoregressive | +0.441% | -0.000656 [-0.002499, +0.001918] |
| original_2x2_mse | 10 | autoregressive | +2.538% | -0.005150 [-0.007196, -0.002727] |
| native_mse | 5 | anchored_additive | +3.029% | -0.004532 [-0.005930, -0.002795] |
| native_mse | 10 | anchored_additive | +3.547% | -0.007102 [-0.008831, -0.005019] |
| original_2x2_mse | 5 | anchored_additive | +0.527% | -0.000784 [-0.002281, +0.001370] |
| original_2x2_mse | 10 | anchored_additive | +0.472% | -0.000938 [-0.002897, +0.001784] |
| native_mse | 5 | context_off | +0.048% | -0.000069 [-0.000633, +0.000478] |
| native_mse | 10 | context_off | +0.097% | -0.000188 [-0.000943, +0.000485] |
| original_2x2_mse | 5 | context_off | -0.194% | +0.000286 [-0.000386, +0.001016] |
| original_2x2_mse | 10 | context_off | -0.108% | +0.000213 [-0.000644, +0.001075] |
| native_mse | 5 | action_free | +2.948% | -0.004406 [-0.006551, -0.002149] |
| native_mse | 10 | action_free | +4.503% | -0.009107 [-0.011896, -0.006061] |
| original_2x2_mse | 5 | action_free | +2.948% | -0.004493 [-0.007086, -0.001778] |
| original_2x2_mse | 10 | action_free | +4.085% | -0.008422 [-0.011892, -0.004691] |

Intervals resample recording sessions and training seeds together in matched comparisons (10,000 draws). These are exploratory validation comparisons, unadjusted for multiple comparisons.
