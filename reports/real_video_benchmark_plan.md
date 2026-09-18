# Real footage: verified options and execution choice

Updated 19 September 2026. Real data are available; the earlier drone and
LapGym experiments are simulations and must remain labeled accordingly.

| Source | Actual material | Useful supervision | Status / decision |
|---|---|---|---|
| [DROID](https://droid-dataset.github.io/) | Real robot manipulation across physical settings | Three camera streams, commands and robot state | **Primary real-data experiment.** Both the 100-episode ingestion release and the fixed 1,126-episode selection from the full release are downloaded, checksum-verified and audited. Full training and evaluation are running on three GPUs. |
| [Open-H CUHK endoscopy](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) | Actual camera footage in a physical stomach phantom | Two motor commands and tracked state | One original MP4+Parquet downloaded/decoded. Full named subset: 462 episodes, 2.23 GB. Real recording, not patients. Optional second action-conditioned domain; no training yet. |
| [Navigation World Models / GNM data](https://github.com/facebookresearch/nwm) | Real ground-robot navigation recordings | Images, trajectories and odometry-derived navigation conditions | Official code/weights and preprocessing exist. Some high-resolution SACSoN/HuRoN data request-only. A strong separate navigation direction; not the current required download. |
| [EuRoC MAV](https://projects.asl.ethz.ch/datasets/euroc-mav/) | Real onboard drone stereo images | IMU, pose and calibration | Genuine drone benchmark, primarily visual-inertial estimation. No motor command channel verified, so not interchangeable with action-conditioned control training. |
| [Cholec80](https://camma.unistra.fr/datasets/) | Real patient surgical videos | Phases and tool-presence labels | Current official 74.917 GB ZIP anonymously accessible; full payload not downloaded. Useful temporal prediction/anticipation; labels are not robot controls. |
| [SWoMo / processed Cataract-1K](https://github.com/MECLabTUDA/SWoMo) | Separate real patient and simulated video archives | Graph conditions, real/simulated pairs and generative weights | Real frames: 38.069 GB publicly byte-accessible. Real manual masks request-only; reconstructed motion is not verified native commands. |

The [surgical access audit](real_surgical_video_benchmarks.md) records nine
datasets, actual HTTP outcomes, restrictions and primary sources. The
[Open-H sample audit](real_openh_video_audit.md) identifies the genuine footage
and checks video/action alignment while retaining its timing limitations.

The immediate path is DROID real-video forecasting with a frozen DINOv2
representation and compact learned action-conditioned predictors, using the
existing LeWM transformer implementation. Read the prespecified
[protocol](real_droid_protocol.md) for session separation, models, metrics and
the distinction between the DROID subset study and a complete benchmark.
The sample release is an ingestion audit; model training uses the fixed larger
selection. The [live progress report](real_video_progress.md) records the
12 registered 30-epoch runs and 48 evaluation populations. The final results
report and portable local checkpoint release are generated only after all
runs, evaluations and offline reload checks pass. No positive real-data
result is assumed in this dataset-selection plan.
