# RLA-WM / IWS comparison preparation

Checked 19 September 2026. This is an acquisition and reproducibility audit, not a reproduced score or a claim of superiority.

The official [RLA-WM repository](https://github.com/mlzxy/rla-wm), pinned locally at `6f19048758699bf9a152eaed5ac6dbf1caa07c18`, provides checkpoints and recorded ALOHA manipulation data for PushT, rope routing and box manipulation. The [paper](https://arxiv.org/html/2605.07079v1) describes real IWS recordings and an official validation split. The separate [public dataset](https://huggingface.co/datasets/xyzhang368/RLA-WM) revision is `34bd8a8cbf3fa68e09ebd69aa35cb673279f4fc2`.

Acquisition is registered in `configs/real_video_development/iws_acquisition_v1.json`: five files, 2,765,690,664 bytes, including the 2.765 GB archive, all three upstream evaluation-handle files and its README. The downloader verifies archive SHA256 / small-file Git blob identity, rejects unsafe archive paths and preserves upstream contents. It performs no image decoding, selection or evaluation. Raw data stay local.

## Reproduction issues found before spending GPU time

- The official predictor imports `src.trainers.dino_latent_action_flow_trainer.DinoLatentActionFlowTrainer`, but this pinned checkout contains `src/trainers/rla_wm_trainer.py` and class `RLAWMTrainer`. A compatibility correction needs explicit validation against checkpoint configuration.
- The paper describes 60-step IWS evaluation, while a released predictor comment says 120. The downloaded official handle files resolve this discrepancy: all three contain 200 horizon 60 windows over 10 validation trajectories. Executable rollout code reads the horizon from each handle, so the 120-step comment is stale. Keep the actual 60-step contract.
- The required DINOv3-L weights are manually gated. The official weight URL returned HTTP401 without authentication. Current DINOv2-S features are not compatible with the released DINOv3-L models. The upstream extractor silently falls back to DINOv2-S on loading errors; any real reproduction must disable that fallback and verify encoder identity.
- Python3.10/CUDA12.8 and a large optional dependency set are upstream requirements. Use an isolated environment; do not modify the running DROID environment. Documentation recommends roughly 200 GB RAM for full training, beyond the current per-account allocation. Inference requirements need measurement.
- Required PushT predictor/encoder/decoder files alone total approximately 3.56 GB, plus DINOv3-L. These checkpoint files have not been downloaded; dataset acquisition completed independently.

## Next executable comparison

First retain the official real-video split and verify action/time/handle semantics. Once authorized DINOv3-L access is available, reproduce the unmodified model with explicit compatibility repairs, fixed sampling seeds and documented horizon. Then compare a frozen adaptation wrapper against the same base checkpoint, using equal information and inference budgets. A DINOv2-trained model on IWS would be a separate experiment, not a reproduction of RLA-WM. No upstream qualitative output should be labeled as ShiftWM output.

Residual latent prediction and learned dynamics corrections are established mechanisms; see [RLA-WM](https://arxiv.org/abs/2605.07079) and [ReDRAW](https://arxiv.org/abs/2504.02252). New architectural claims require matched evidence beyond those ingredients.

## Completed local acquisition

CPU job 200194 downloaded and verified all five assets, then safely extracted 17,771 entries (2,751,920,364 bytes). Metadata auditing found 3,489 training and 60 validation trajectories across six real manipulation tasks, with nonoverlapping official trajectory IDs. All three released evaluation files have 200 windows over 10 validation episodes at horizon 60. These metadata do not establish independent sessions or scenes. The reproducible metadata audit opens no videos: `scripts/real_video_development/audit_iws.py`; evidence is `reports/evidence/iws_metadata_audit.json`.

The first frame of the first official PushT training trajectory (`000010`) was inspected locally: it is actual two-arm tabletop footage. Its accompanying foreground-mask frame is entirely white, so these filenames must not be assumed to supply object annotations. Stored video dimensions are 640×480 and the container reports 200 frames at 30 fps; physical timing requires a separate conversion audit. This image inspection is separate from the metadata-only split audit.

## Completed temporal/command audit

The [independent temporal audit](iws_temporal_semantics_review.md) verified all 3,489 training metadata records and 600 official windows without decoding validation images. The nominal horizon60 pairs images at s and s+59 with60command rows; the official four15-row chunks have an unresolved boundary convention. Preserve that released evaluator for reproduction and disclose the ambiguity. Four published summary discrepancies remain recorded. Official RLA-WM uses one initial image, which must be matched when comparing information budgets.
