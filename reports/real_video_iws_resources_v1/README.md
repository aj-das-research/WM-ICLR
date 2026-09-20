# Matched IWS predictor-resource measurements, v1

This is a separate resource-only protocol. It loads the unchanged selected inference exports for all 27 original predictors and all nine no-tanh ablations. Every call takes the same previously exported internal-training episode 000011/frame-0 feature grid for its task and the same 60 native command rows. No original dataset, cache, evaluation target, or reserved payload is opened. Persistence ignores commands and materializes the same [1,59,6144] output shape; it has one row per task, not three fictitious training seeds.

`configs/real_video_iws_resources_v1/protocol.json` fixes the comparison before measurement. Registration binds the complete 36-model grid, three persistence cases, input-array identity, every immutable bundle payload, profiling source/config and randomized case order. The frozen experiment sources, model weights, normalizations and existing releases remain unchanged. This work performs no training, model selection, benchmark-accuracy calculation, publication or paper editing.

Each case runs in a fresh isolated Python process, CPU first and CUDA second. Both use FP32, two intra-op threads, one inter-op thread, no autocast and no TF32. Input/model loading and transfers happen before timing. Ten full-forecast warmups precede 30 timed calls. CUDA wall latency synchronizes immediately before the timer and after the forecast. All samples are retained, including slow calls; per-case median, mean, sample SD and percentiles are descriptive timing summaries. Task/method summaries average the three model medians and show their range, without calling this uncertainty over tasks.

CUDA memory is the absolute peak PyTorch allocation/reservation during the timed calls and incremental allocated peak above resident model/input tensors. Context/driver/non-PyTorch memory is excluded. CPU memory is the fresh-process RSS high-water through the CPU phase, including Python, libraries and model loading. It is not a clean tensor-allocation measurement and is not directly comparable to CUDA allocator bytes. Finite-output validation follows peak capture, avoiding validation tensors contaminating the peak.

These figures measure the complete exported predictor's 59-offset feature forecast, including its internal normalization and input validation. They do not measure DINO encoding, image loading, host-device transfer, planning/control or end-to-end robot latency. Results apply to the recorded hardware/software, batch size and fixed input only. Unsupported CUDA is explicitly recorded and prevents the combined protocol receiving passed status.

Commands (registration once, before measurements):

```
.venv/bin/python -B -m unittest discover -s scripts/real_video_iws_resources_v1 -p 'test_*.py'
.venv/bin/python -B scripts/real_video_iws_resources_v1/profile.py register
sbatch scripts/real_video_iws_resources_v1/run.slurm
```

One Slurm GPU is requested, with two allocated logical CPUs and a 30-minute safety limit. Each job writes immutable case records and a summary beneath `reports/real_video_iws_resources_v1/job_<id>/`. A summary is complete only after all 39 cases finish, all 78 device rows are supported, and source and bundle checksums revalidate. No automatic manuscript/site ingestion is attached.

CUDA latency is a resource-only measurement. It does not certify CPU/GPU output parity or resolve the previously observed AR prefix-equivalence differences; those scientific evaluations remain bound to their separate validated common-CPU protocol.

CPU affinity is recorded per case; two PyTorch threads do not imply two distinct physical cores. The Linux `ru_maxrss` high-water and `/proc` instantaneous RSS are separate approximate counters and can differ slightly in either direction. No CPU incremental-allocation estimate is formed by subtracting them.
