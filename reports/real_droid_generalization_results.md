# Matched real-DROID generalization development

Status: completed; 36/36 complete 30-epoch runs.
Original training/validation only. Neither original nor fresh test data select these revisions.

| Arm | Horizon | Framewise | Constant dynamics | ShiftWM (ours) | Action-free | Ours vs Framewise | Paired difference interval |
|---|---:|---:|---:|---:|---:|---:|---|
| slow | 5 | 0.158478 | 0.158413 | 0.158020 | 0.159017 | +0.289% | [-0.000671, -0.000283] |
| slow | 10 | 0.216255 | 0.216054 | 0.215557 | 0.215645 | +0.323% | [-0.001144, -0.000339] |
| decay | 5 | 0.159068 | 0.158995 | 0.158392 | 0.159764 | +0.425% | [-0.001535, -0.000053] |
| decay | 10 | 0.216886 | 0.216728 | 0.219479 | 0.217418 | -1.196% | [-0.000748, +0.005568] |
| compact | 5 | 0.159643 | 0.159626 | 0.158884 | 0.160186 | +0.476% | [-0.001353, -0.000278] |
| compact | 10 | 0.219250 | 0.219233 | 0.220541 | 0.218768 | -0.589% | [-0.000945, +0.003425] |

All paired intervals are exploratory validation intervals, unadjusted for multiple comparisons. Negative MSE differences favor ours; positive percentage reductions favor ours. All completed arms are retained.
