# Shared IWS RGB readout, v1

This is a new auxiliary decoder experiment after the existing feature-space and
reserved evaluations. It does not retrospectively preregister RGB outcomes,
reselect any forecasting checkpoint, or replace a primary endpoint.

## Exact fit contract

Three separately trained decoders: PushT, Box, Rope, each seed **173**. Each
decoder is then held fixed for all four learned predictors, all three predictor
seeds, and feature persistence. Uncertainty is conditional on this single
decoder fit per task; it does not estimate variability over decoder seeds.

Input is the existing frozen raw DINOv2-small feature vector, **384×4×4=6,144**,
channel-major, standardized by the existing training-only per-channel mean/std.
No encoder is trained or rerun. The decoder is the unchanged official DINO-WM
`TransposedConvDecoder` implementation at commit
`0a9492fa12044b852ae9e001cc74604b79c8bb0c`, source SHA
`c1327257b0f964edb60c66ff528d6c9d24ae5d43322a13ef72eacd716374b0ac`.
Constructor: full-vector input 6144, depth32, five transposed convolutions,
kernel5/stride3 and final bilinear resize224×224. The first linear layer mixes
the complete input vector; this is not a new spatial feature predictor and is
not a claim to reproduce the published DINO-WM decoder recipe/results.
Official code: https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/decoder/transposed_conv.py

RLA-WM's local `DinoToImageDecoderV1` was also inspected and source-bound for
provenance. Its pretrained DINOv3 dense-grid readout is not compatible with the
present DINOv2 pooled4×4 representation. No RLA decoder weights are reused.

Use **every internal training trajectory** at outcome-independent native indices
0,5,10,… and the same rule over every internal development trajectory. Sequential
video decoding is reused unchanged. Full decoded RGB hashes must equal the
original feature-extraction receipts before frame selection. Targets: uint8 RGB,
float32/255, CPU torch bilinear resize224×224 with antialias=True and
align_corners=False, round(value×255), uint8; training converts these targets to
[0,1]. This explicitly quantized resized target is not an unchanged native image.
No crop, augmentation, foreground mask, future data, or reserved episode is used
to train/select the decoder. Full future offsets may later be decoded by the
same fixed readout, under a separate reviewed evaluator.

Fit30 complete epochs, batch32, FP32, Adam1e-4, betas(.9,.999), epsilon1e-8,
weight_decay0, TF32/autocast off, no scheduler, no early stopping. Loss is
unclipped RGB MSE with equal sampled-frame weight. Checkpoint selection uses
minimum **unclipped** RGB MSE, equal selected frames within development
trajectory and then equal trajectories; exact ties choose the earliest epoch.
The chosen checkpoint's validation must reproduce exactly before completion.
Clipped [0,1] reconstruction MSE is recorded separately. Model/optimizer/RNG,
history and best weights are stored in one atomic epoch checkpoint, so a requeue
does not repeat a committed epoch. GPU device identity is recorded; exact
cross-device bitwise reproducibility is not claimed.

## Freeze, review, and execution

No GPU runs or data preparation are launched by these scripts automatically.
The parent owns scheduler submission. Registration reads metadata/source hashes
only; all later commands require detached `source_review.json` with
`status=passed` and the exact `registration_sha256`.

```bash
.venv/bin/python -m pytest -q scripts/metrics_completion_v1/rgb_test.py
.venv/bin/python scripts/metrics_completion_v1/rgb_register.py
# After independent review only; one allocation pipelines cache→profile→fit.
sbatch --array=0-2%1 scripts/metrics_completion_v1/rgb_run.slurm
# Array index0=pusht,1=bimanual_box,2=bimanual_rope. Parent may place separate
# indices on allowed partitions while respecting the existing global GPU quota.
```

Manual stages, still requiring the same source-review gate:
```bash
.venv/bin/python scripts/metrics_completion_v1/rgb_train.py --task pusht --stage cache
.venv/bin/python scripts/metrics_completion_v1/rgb_train.py --task pusht --stage profile
.venv/bin/python scripts/metrics_completion_v1/rgb_train.py --task pusht --stage train
.venv/bin/python scripts/metrics_completion_v1/rgb_finalize.py --if-ready
```

Each scheduler allocation:1GPU,8CPU,24GiB,8h; epoch-boundary requeue after7h
including current invocation's cache/profile time, maximum8restarts. A profile
uses only the first256 lexicographic selected training frames,2warmups+20timed
updates, measures CUDA allocation peak and synchronized wall time. It does not
select an architecture, checkpoint or metric; weights are discarded and the fit
reseeds to173. Profile timing excludes IO and validation, so is not a completion
promise. Target+feature cache is approximately13GB total for all three tasks.
CPU cache work is in the same GPU allocation by the parent's budget decision.

All artifacts remain under `reports/metrics_completion_v1/rgb`. `targets/` and
`runs/` contain **local-only binary data**, not a public dataset or frame archive.
RLA/IWS dataset license remains unspecified; these outputs carry no new data
license grant. Do not add these binary trees to a publishing allowlist.

## Deliberate remaining evaluator work

`rgb_finalize.py` emits only after all3 full30-epoch fits pass. Its report is
ground-truth-feature reconstruction quality, **not forecast RGB accuracy**.

The subsequent evaluator must be frozen separately and bind these3 selected
decoder hashes plus all36 existing predictor hashes, exact reserved recovery-v2
CPU rowwise inference, all200 handles/task, all59 offsets and both window and
trajectory aggregation. Use the same task decoder for each predictor seed;
compare decoded predictions with real resized targets and include ground-truth
feature reconstruction as a readout limitation. The fixed qualitative case IDs
must come from `qualitative_closest_v1`, with observed/target/prediction roles and
native indices explicit. No outcome-based reselection or refitting is allowed.

Required RGB metrics: MSE in [0,1]²↓; PSNR with data_range1↑; SSIM with a frozen
window/kernel/channel convention↑; UIQI with a frozen window and zero-variance
convention↑; official LPIPS v0.1 VGG↓ after mapping [0,1] to[-1,1], with both
backbone and calibrated weights pinned. Neither LPIPS nor its weights is
installed by this training stage. These measurements remain **not run**, not0.

FID and FVD require a separate official feature implementation/weights,
preprocessing, sample count, clipping/temporal protocol, and finite-sample
limitations. They are not approximated by feature MSE or any proxy. Forecast
metric code and visual exports remain dependent work after the runnable fit
stage; no training command claims to fill those result cells yet.
