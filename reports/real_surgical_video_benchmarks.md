# Real surgical video: source and executable-access audit

Checked 19 September 2026, Dubai time. This audit fetched public documentation,
metadata and bounded file prefixes. It did not download a complete dataset,
submit a registration form, contact anyone, train a model or run a checkpoint.
The machine-readable ledger is
[`evidence/real_surgical_video_benchmarks.json`](evidence/real_surgical_video_benchmarks.json);
raw HTTP outcomes are in
[`evidence/real_surgical_access_probe.json`](evidence/real_surgical_access_probe.json).
Reported GB are decimal unless the publisher's units are reproduced explicitly.

## Recommended choices

1. **For a feasible real patient-video extension now: SWoMo's processed
   Cataract-1K or Cholec80.** Anonymous range requests returned actual ZIP bytes
   from their official download endpoints. Both support held-out future-video
   prediction. Neither supplies verified native robot commands, so these would
   be visual prediction/representation experiments, not surgical robot-control
   evaluations. The direct endpoints are linked by the
   [SWoMo authors](https://github.com/MECLabTUDA/SWoMo) and
   [CAMMA](https://camma.unistra.fr/datasets/).
2. **For motion-conditioned real surgical robotics: SurgPose is the most
   relevant conditional candidate.** It has stereo video and measured dVRK
   Cartesian/joint states. Its official Zenodo listing is public, but the
   server's file/API requests timed out; its license also needs clarification
   from the record. Therefore executable download is **not yet verified**.
   JIGSAWS is another relevant option, with an explicit access request step.
   [SurgPose code/data description](https://github.com/zijianwu1231/SurgPose),
   [JIGSAWS official page](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/).
3. **Do not divert the immediate action-conditioned experiment to a
   segmentation-only benchmark.** EndoVis 2017/2018 and CholecT50 are useful
   complementary perception tasks, but their masks or semantic verbs are not
   actuator commands. Our immediate DROID experiment can address the real
   robot-action requirement while the surgical datasets support a separate,
   correctly named visual-prediction study. This is a design recommendation,
   not a claim of measured performance.

## Dataset comparison

| Dataset | Actual recorded material | Available signal | Official access observed | Size | Fit for this project |
|---|---|---|---|---|---|
| **SWoMo processed Cataract-1K** | Real patient cataract frames, separately stored simulated frames | Scene graphs; simulated masks; official splits. Real manual masks require author request | **Anonymous ZIP byte access verified** through the authors' HF repository | Real frames **38.069 GB**; all listed dataset files **95.679 GB** | Best immediately accessible MEC-Lab-linked real surgical video and related code; distinguish inferred scene motion from commanded action |
| **Cholec80** | 80 patient cholecystectomy videos, 25 fps | Phase at 25 fps; tool presence at 1 fps; no control channel described | **Anonymous ZIP byte access verified** on current official S3 link | **74.917 GB** verified Content-Length | Future-video or phase anticipation; no real closed-loop control claim |
| **CholecT50** | 50 patient laparoscopic videos; 1 fps labeled frames | 100 instrument–verb–target categories, phases, boxes for five videos | Full data: request form. Public challenge-validation ZIP: **anonymous bytes verified** | Validation **322.636 MB**, five clips/~1.1K frames; full archive size unverified | Semantic anticipation. The small public validation release alone is not a complete independent training benchmark |
| **SurgPose** | 34 stereo bench-top dVRK instrument sequences; real cameras and robot, not patient operations | Keypoints, stereo calibration, measured Cartesian/joint states, opening angle | Public official Zenodo listing; **server archive request timed out** | Publisher lists **23.1 GB** total; first sequence ~451.6 MB | Strong motion-video candidate after access, license and synchronization checks |
| **JIGSAWS** | Real da Vinci stereo recordings of bench-top suturing, knot tying and needle passing | Synchronized positions/orientations/velocities/gripper state, gestures and skill ratings | Official form requests identity/institution/email; official page mentions downloader problems and email alternative | Not verified | Strong observational motion-video benchmark after authorized access; recorded states are not automatically commanded controls |
| **EndoVis 2017** | Real porcine da Vinci Xi stereo sequences, sampled at 2 Hz | Binary, instrument-part and instrument-type segmentation | Official Data page works; Downloads returned **403, permission denied** | Not verified | Perception/temporal segmentation; no released native action channel verified |
| **EndoVis 2018** | Real porcine robotic-surgery scenes | Scene/instrument masks; stereo imagery/calibration | Official Data page works; Downloads returned **403, permission denied** | Not verified | Complementary perception benchmark, currently gated from this server |
| **CATARACTS** | 50 real patient cataract surgeries | Tool-use labels; related CaDIS masks and CAT-SG scene graphs | Official IEEE DataPort page accessible; authenticated file access not attempted or verified | Publisher: videos **117.8 GB**; 2018 images **691.2 GB**; labels **18.55 MB** | Real visual/scene-graph prediction; larger and less immediately executable than processed Cataract-1K |
| **Original Cataract-1K** | 1,000 real patient cataract videos | Different subsets annotated for segmentation, phases and irregularities | Official Synapse folder metadata accessible; video payload access unverified | Publisher: videos **89.9 GB**, phase subset **3.87 GB**, segmentation subset **4.74 GB** | Clinical video prediction; no native actuator commands described |

Dataset sources: [CAMMA](https://camma.unistra.fr/datasets/),
[CholecT50 downloads](https://github.com/CAMMA-public/cholect50/blob/master/docs/README-Downloads.md),
[SurgPose record](https://zenodo.org/records/15278516),
[JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/),
[official EndoVis index](https://opencas.dkfz.de/endovis/datasetspublications/),
[CATARACTS DataPort](https://ieee-dataport.org/open-access/cataracts),
[Cataract-1K authors](https://github.com/Negin-Ghamsarian/Cataract-1K).

## MEC-Lab TU Darmstadt: what is actually reusable

**SWoMo is directly relevant prior work, not merely a dataset pointer.** It
already separates rule-based motion from diffusion-based appearance and uses
reconstructed real/simulated pairs. Its project identifies a MICCAI 2026
spotlight acceptance. Our work cannot claim that separation itself as new.
Its inverse-pairing kinematics are recovered from video masks/tracks, not
documented robot command logs.
[Project and methodological description](https://ssharvienkumar.github.io/SWoMo/).

The [official code](https://github.com/MECLabTUDA/SWoMo) contains training and
sampling entry points. The [public HF inventory](https://huggingface.co/SsharvienKumar/SWoMo/tree/main)
contains real checkpoint payloads: one diffusion UNet is 4.965 GB, a ControlNet
is 2.102 GB, and a video VAE is 334.6 MB. All listed weight tensor files total
16.787 GB; downloading every checkpoint directory would cost 44.813 GB because
it includes optimizer states and alternate variants. A 277.3 MB graph-encoder
checkpoint returned anonymous byte ranges. This establishes file availability,
**not successful inference or a compatible turnkey Diffusers pipeline**. Use
the authors' configurations, not HF's generic auto-generated usage example.
The inspected inventory is under `Cataract-1K`; CATARACTS configuration examples
do not prove that a separate CATARACTS-trained checkpoint is present.

The HF card declares CC BY 4.0. The original Cataract-1K README also declares
CC BY 4.0 for data, while its preprocessing repository has an MIT license.
No explicit top-level SWoMo code license was identified in the inspected tree;
weight/data licensing should not be silently applied to code. Real 16 fps
manual segmentations for 869 videos are offered on request, unlike the public
real frames, simulated masks and scene graphs.
[SWoMo release instructions](https://github.com/MECLabTUDA/SWoMo),
[HF model card](https://huggingface.co/SsharvienKumar/SWoMo).

**SDS_Playground is useful loader infrastructure with incomplete video support.**
It implements wrappers for CATARACTS, CaDISv2, CholecSeg8k, M2CAI Seg and
Cataract-1K. Its README still marks Cholec80/Cataract101 video support as TODO;
the Cataract-1K phase and irregularity implementations are also marked TODO.
It is not evidence of a ready video-world-model training pipeline.
[Repository](https://github.com/MECLabTUDA/SDS_Playground),
[Cataract-1K wrapper status](https://github.com/MECLabTUDA/SDS_Playground/blob/main/sds_playground/datasets/cataract1k/README.md).

## Access and licensing details that change the execution plan

- **Cholec80:** current official page declares CC BY-NC-SA 4.0 and directly
  links `https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip`.
  HEAD returned 200; `Range: bytes=0-63` returned 206 and a ZIP signature with
  total length 74,916,858,781 bytes. Older code's request-form instructions
  should not override this current official route. The whole archive's
  integrity is not yet checked. [Official page](https://camma.unistra.fr/datasets/).
- **CholecT50:** CC BY-NC-SA 4.0, with a request form for the full data. Its
  direct challenge-validation release works, but does not confer access to
  the other 45/full 50 videos. Official
  [Rendezvous](https://github.com/CAMMA-public/rendezvous) provides loaders,
  evaluation and pretrained recognition checkpoints; these are recognition
  baselines, not world-model checkpoints. The original-split checkpoint's
  HEAD request returned 200 and 68,959,695 bytes; inference remains untested.
- **SurgPose:** the official README warns that raw measured kinematics need
  compensation and that some keypoints/boxes contain errors. Extract video at
  30 fps to preserve label alignment. We found no readable license declaration
  in the retrieved Zenodo rights section or repository; do not infer CC BY
  solely because Zenodo labels the record open.
  [Code](https://github.com/zijianwu1231/SurgPose),
  [record](https://zenodo.org/records/15278516).
- **CATARACTS:** the official challenge page says noncommercial use, while the
  IEEE landing-page JSON-LD links CC BY 4.0. Preserve that discrepancy until
  the actual download terms are inspected. A public landing page is not proof
  of anonymous archive access.
  [Challenge terms](https://cataracts.grand-challenge.org/Download/),
  [DataPort](https://ieee-dataport.org/open-access/cataracts).
- **EndoVis:** do not use unofficial HF/Zenodo mirrors to circumvent official
  access or assume a mirror's license replaces the original terms. The
  [winning EndoVis 2017 implementation](https://github.com/ternaus/robot-surgery-segmentation)
  provides MIT code, model links and evaluation, but is archived and targets
  PyTorch 0.4. Its code license does not license the source images.

## What a valid real-video experiment could demonstrate

**Real visual prediction:** split by entire patient/video before clip extraction;
use only observed support to infer context, then forecast held-out future
features/images. Compare persistence, constant velocity, an action-free model,
and matched learned controls. Report per-video uncertainty and horizon-specific
prediction error. Future masks/graphs may define a separate explicitly
conditioned task, but must not leak into a history-only prediction baseline.

**Motion-conditioned prediction:** on a verified SurgPose/JIGSAWS release,
audit timestamps, units, offsets and which recorded variables are state versus
command. If future measured motion is supplied, name the experiment accordingly.
Do not call pose differences ground-truth robot commands or use future measured
states to imply real action selection capability. Stored future frames can
define offline goal-retrieval targets; they cannot answer unexecuted alternative
actions or establish closed-loop surgical success.

**Leakage control:** Cholec80, CholecT50 and Endoscapes share source operations.
Use the [official overlap mapping](https://github.com/CAMMA-public/camma_dataset_overlaps)
before claiming cross-dataset generalization. Likewise, processed SWoMo
Cataract-1K and the original Cataract-1K are not independent datasets. Keep
patients/videos and all processed derivatives together across train, validation
and test.

All proposed experiments above are recommendations, not completed measurements.
No protected download, registration or message requesting access was performed.
