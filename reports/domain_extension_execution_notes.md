# Execution notes for the registered domain extension

These operational corrections preserve the registered scientific settings.

1. The first surgical training job (200069) failed during initialization because
   its data manifest used `env` where the shared trainer expected `environment`.
   No training updates occurred. A recorded migration added the compatible alias,
   verified 1,536 payload/signature files and 451 goal files unchanged, repeated
   the data audit and training-only action statistics, and rebound the feature
   cache manifest. Numeric normalization did not change. Job 200070 restarts the
   same 18 complete training configurations. See
   `reports/surgery/manifest_alias_migration/receipt.json` for old/new hashes.

2. Real simulator integration caught a drone RPC shutdown defect: it replied to
   `close` without leaving the request loop. The server now exits after replying.
   Both native-budget replay and random-control integration cases pass. This is
   a process-lifetime fix, with no physics, policy, dataset or scoring changes.

3. Surgical rendering needs the NVIDIA EGL stack. Although collection requested
   CPU resources and set `LIBGL_ALWAYS_SOFTWARE=1`, Pyglet selected NVIDIA EGL on
   the compute node. The login node used Mesa and produced different pixels at
   exactly identical physical states. Reproduction now explicitly pins NVIDIA
   EGL, reports the actual renderer, and uses GPU allocations. The exact-image
   check remains in place; no tolerance was loosened to pass an inconsistent
   renderer. Fresh allocated-node replay passes. See
   `reports/surgery/renderer_pin_migration/receipt.json`.

4. The published AdaJEPA reproduction is a separate track: its released model
   uses four-dimensional proprioception and its official goals/planning budget.
   It is not an input-matched comparison with ShiftWM. Its official checkpoint
   and data passed archive checks, exact action replay, an actual adaptation
   update, and exact parameter/buffer restoration. Its older Python runtime
   needs a pinned compatible auxiliary DINOv2 source import. The AdaJEPA source
   remains unchanged. See `reports/adajepa_baseline_2026-09-19.md`.

5. Validation disables autocast and uses float32 tensors, while the common
   trainer allows CUDA TF32 matrix kernels. The shorthand "FP32 validation"
   should not be read as a guarantee of full IEEE float32 mantissa arithmetic
   for every matrix multiplication. This setting is shared by all arms. Offline
   CPU and GPU kernel outputs need not be bitwise identical; release checks
   separately verify exact package tensors and bounded numerical inference.

No failed initialization, engineering replay test, validation loss or auxiliary
baseline reproduction is reported as a successful learned application result.
