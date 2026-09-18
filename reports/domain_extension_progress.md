# Domain extension development progress

2026-09-18T22:00:46.998823+00:00

Full training: 36/36; forecast evaluations: 36; closed-loop evaluations: 36.

These are development results. All success/failure outcomes are included. Full training does not by itself establish a control gain or state of the art.

| Domain | Predictor | Method | Seed | Training | Forecast MSE@5 | Planning successes |
|---|---|---|---:|---|---:|---:|
| drone | transformer | Framewise | 0 | 30/30 (completed) | 0.005617 | 1/8 |
| drone | transformer | Constant dynamics | 0 | 30/30 (completed) | 0.005701 | 1/8 |
| drone | transformer | ShiftWM (ours) | 0 | 30/30 (completed) | 0.005699 | 1/8 |
| drone | transformer | Framewise | 1 | 30/30 (completed) | 0.005599 | 2/8 |
| drone | transformer | Constant dynamics | 1 | 30/30 (completed) | 0.005654 | 1/8 |
| drone | transformer | ShiftWM (ours) | 1 | 30/30 (completed) | 0.005639 | 2/8 |
| drone | transformer | Framewise | 2 | 30/30 (completed) | 0.005735 | 1/8 |
| drone | transformer | Constant dynamics | 2 | 30/30 (completed) | 0.005800 | 2/8 |
| drone | transformer | ShiftWM (ours) | 2 | 30/30 (completed) | 0.005796 | 2/8 |
| drone | gru | Framewise | 0 | 30/30 (completed) | 0.006519 | 1/8 |
| drone | gru | Constant dynamics | 0 | 30/30 (completed) | 0.006524 | 1/8 |
| drone | gru | ShiftWM (ours) | 0 | 30/30 (completed) | 0.006458 | 1/8 |
| drone | gru | Framewise | 1 | 30/30 (completed) | 0.006560 | 3/8 |
| drone | gru | Constant dynamics | 1 | 30/30 (completed) | 0.006591 | 4/8 |
| drone | gru | ShiftWM (ours) | 1 | 30/30 (completed) | 0.006491 | 2/8 |
| drone | gru | Framewise | 2 | 30/30 (completed) | 0.006654 | 2/8 |
| drone | gru | Constant dynamics | 2 | 30/30 (completed) | 0.006658 | 2/8 |
| drone | gru | ShiftWM (ours) | 2 | 30/30 (completed) | 0.006561 | 1/8 |
| surgery | transformer | Framewise | 0 | 30/30 (completed) | 0.003652 | 0/8 |
| surgery | transformer | Constant dynamics | 0 | 30/30 (completed) | 0.003616 | 1/8 |
| surgery | transformer | ShiftWM (ours) | 0 | 30/30 (completed) | 0.003625 | 1/8 |
| surgery | transformer | Framewise | 1 | 30/30 (completed) | 0.003550 | 1/8 |
| surgery | transformer | Constant dynamics | 1 | 30/30 (completed) | 0.003530 | 1/8 |
| surgery | transformer | ShiftWM (ours) | 1 | 30/30 (completed) | 0.003541 | 0/8 |
| surgery | transformer | Framewise | 2 | 30/30 (completed) | 0.003562 | 1/8 |
| surgery | transformer | Constant dynamics | 2 | 30/30 (completed) | 0.003544 | 0/8 |
| surgery | transformer | ShiftWM (ours) | 2 | 30/30 (completed) | 0.003554 | 0/8 |
| surgery | gru | Framewise | 0 | 30/30 (completed) | 0.004383 | 0/8 |
| surgery | gru | Constant dynamics | 0 | 30/30 (completed) | 0.004406 | 0/8 |
| surgery | gru | ShiftWM (ours) | 0 | 30/30 (completed) | 0.004380 | 0/8 |
| surgery | gru | Framewise | 1 | 30/30 (completed) | 0.004406 | 1/8 |
| surgery | gru | Constant dynamics | 1 | 30/30 (completed) | 0.004416 | 1/8 |
| surgery | gru | ShiftWM (ours) | 1 | 30/30 (completed) | 0.004383 | 1/8 |
| surgery | gru | Framewise | 2 | 30/30 (completed) | 0.004427 | 1/8 |
| surgery | gru | Constant dynamics | 2 | 30/30 (completed) | 0.004429 | 1/8 |
| surgery | gru | ShiftWM (ours) | 2 | 30/30 (completed) | 0.004404 | 2/8 |

Paired comparisons (all three training seeds required):

| Domain | Predictor | Comparator | Success difference, pp | 95% interval, pp |
|---|---|---|---:|---|
| drone | transformer | Framewise | +4.17 | [-25.00, +33.44] |
| drone | transformer | Constant dynamics | +4.17 | [-25.00, +33.33] |
| drone | gru | Framewise | -8.33 | [-29.17, +12.50] |
| drone | gru | Constant dynamics | -12.50 | [-37.50, +12.50] |
| surgery | transformer | Framewise | -4.17 | [-25.00, +12.50] |
| surgery | transformer | Constant dynamics | -4.17 | [-20.83, +0.00] |
| surgery | gru | Framewise | +4.17 | [+0.00, +25.00] |
| surgery | gru | Constant dynamics | +4.17 | [+0.00, +25.00] |

Intervals use paired crossed resampling of training and physical task seeds. They are exploratory, unadjusted intervals; eight development tasks provide limited precision. The source JSON records result-file hashes and all individual run statuses.
