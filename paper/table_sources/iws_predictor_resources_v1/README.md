# Complete predictor-only IWS resource comparison

Median of three model-level medians, each based on 30 warmed full 59-offset calls; brackets are min/max of the three medians, not CI. Persistence has one zero-parameter case/task. Registered mean-of-medians is also retained. Memory columns use maximum recorded peak across the three cases.

Batch 1 FP32 raw-feature-input predictor only on fixed training episode 000011/frame 0; excludes encoder, loading, transfer and control.

CPU: Intel Xeon w7-2495X, 2 PyTorch threads, recorded logical affinity [8, 32]. GPU: RTX 5000 Ada; synchronized eager FP32 wall latency. CPU peak is whole-process RSS; GPU peak is PyTorch allocated bytes (not device-total memory).

| Task | Method | Trainable / total | CPU ms [seed range] | CUDA ms [seed range] | CPU RSS peak MiB | CUDA allocated peak MiB |
|---|---|---:|---:|---:|---:|---:|
| pusht | autoregressive | 851,040 / 869,569 | 125.810 [124.290, 125.956] | 67.640 [67.259, 68.218] | 516.62 | 20.03 |
| pusht | anchored_additive | 851,040 / 869,569 | 108.109 [107.331, 108.203] | 56.962 [56.343, 57.195] | 516.22 | 20.03 |
| pusht | bounded_spatial_mix | 869,569 / 869,569 | 115.735 [114.572, 117.608] | 65.472 [64.623, 65.487] | 516.96 | 20.03 |
| pusht | unbounded_spatial_mix | 869,569 / 869,569 | 115.678 [114.933, 115.690] | 64.523 [63.821, 64.526] | 516.82 | 20.03 |
| pusht | persistence | 0 / 0 | 0.042 [0.042, 0.042] | 0.015 [0.015, 0.015] | 387.08 | 1.41 |
| bimanual_box | autoregressive | 853,920 / 872,449 | 125.789 [125.384, 125.891] | 67.316 [66.872, 67.718] | 517.16 | 20.06 |
| bimanual_box | anchored_additive | 853,920 / 872,449 | 107.233 [107.199, 108.311] | 56.921 [56.365, 57.324] | 516.87 | 20.06 |
| bimanual_box | bounded_spatial_mix | 872,449 / 872,449 | 116.102 [114.734, 117.415] | 65.062 [64.968, 65.648] | 516.77 | 20.06 |
| bimanual_box | unbounded_spatial_mix | 872,449 / 872,449 | 115.124 [114.814, 115.130] | 63.778 [63.453, 64.651] | 517.97 | 20.06 |
| bimanual_box | persistence | 0 / 0 | 0.038 [0.038, 0.038] | 0.013 [0.013, 0.013] | 387.65 | 1.41 |
| bimanual_rope | autoregressive | 852,192 / 870,721 | 125.398 [125.145, 125.675] | 67.551 [67.270, 67.706] | 517.54 | 20.05 |
| bimanual_rope | anchored_additive | 852,192 / 870,721 | 108.372 [105.574, 108.478] | 56.175 [56.143, 56.823] | 516.24 | 20.05 |
| bimanual_rope | bounded_spatial_mix | 870,721 / 870,721 | 116.408 [115.704, 117.030] | 65.225 [64.976, 65.647] | 516.60 | 20.05 |
| bimanual_rope | unbounded_spatial_mix | 870,721 / 870,721 | 116.271 [113.290, 116.725] | 64.526 [64.203, 65.129] | 517.11 | 20.05 |
| bimanual_rope | persistence | 0 / 0 | 0.034 [0.034, 0.034] | 0.013 [0.013, 0.013] | 387.75 | 1.41 |

All 39 cases and 78 device rows passed. No unsupported rows were omitted. All 2,340 raw latency samples and both original mean-of-medians and display median-of-medians are retained in data.json. There is no CPU tensor-only peak claim or GPU accuracy/prefix-equivalence certificate.

ru_maxrss and instantaneous /proc VmRSS are different approximate kernel accounting counters; do not subtract them or treat them as isolated CPU tensor memory.

This pack is derived only from completed timing receipts; it contains no input feature arrays, command arrays, targets or weights. The bound reports retain timing/identity provenance.


## Reproduce the paper table without private inputs

Run `python paper/scripts/render_iws_predictor_resources.py --pack-dir paper/table_sources/iws_predictor_resources_v1 --output-dir /tmp/iws-resources-table` from a public checkout. The renderer needs only itself, `data.json`, and `manifest.json`; it rechecks every raw timing median, the exact 39-case grid, all displayed seed summaries, parameter counts, and memory aggregation. No model, dataset, GPU, or network is needed. Identical reruns preserve output bytes and modification times.

The paper input is `paper/generated/iws_resources/predictor_resources.tex`, label `tab:iws-predictor-resources`. Its six columns cover all 15 task/method rows at 9 pt. Parameter units are thousands with three decimals (exact integer counts); learned timing cells round to one decimal while the three persistence cells use three decimals. The JSON retains full precision, all 2,340 raw samples, GPU reserved/incremental peaks, and the registered mean-of-seed-medians alongside the displayed median-of-seed-medians. GPU resource timing is not evidence that the GPU predictor satisfies accuracy or prefix-equivalence checks.
