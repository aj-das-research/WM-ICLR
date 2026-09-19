# Drone, tissue simulation and physical-phantom observations

These two PNGs are unchanged first frames from the first lexicographic training
trajectories in the project's registered simulation collections. The extraction
manifest pins raw data, collection manifests, simulator revisions and exact pixel
hashes. They are representative **inputs**, not predicted images or selected
successful outcomes.

* **Drone:** locally rendered observation of the adapted
  [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) task,
  revision `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`; credit Panerati and
  collaborators / the upstream project. The upstream simulator code is MIT.
* **Tissue manipulation:** locally rendered observation of LapGym's SOFA
  `TissueManipulationEnv`, from
  [sofa_env](https://github.com/ScheiklP/sofa_env), revision
  `85bf7e05dd088b824794dda0046679df13b13e6e`; credit Scheikl and collaborators.
  The upstream environment code is MIT; SOFA retains its own LGPL terms.

Simulator software licensing does not establish a general license for every
generated dataset or model. These are attributed project-generated research
illustrations. They are not real flight footage, patient data or a demonstration
of autonomous surgical safety.

The associated completed development studies use separately trained historical
context predictors. They do not establish that the current bounded spatial
decoder has been evaluated on these tasks. Preserve that scope distinction when
placing these images beside DROID and ongoing IWS transfer examples.

## Open-H: input illustration only

`openh_episode_000000_frame160.png` is an unchanged-resolution decoded frame
of the previously audited CUHK `find_greater_curvature` episode 0 from
[PhysicalAI-Robotics-Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment),
revision `e29dda7cabf2a2626634c7822db27695553ae523`. Credit the dataset authors,
CUHK and the hosting Open-H collection. The release declares
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This is recorded footage of a **physical stomach phantom**, not patient surgery
and not rendered simulation. The project audited one input episode; it trained
and evaluated no Open-H predictor. Label the thumbnail **Open-H · input only**.
Do not connect it to the DROID result route or present it as a completed benchmark.
Source byte hashes, exact RGB extraction, FFmpeg version and the original audit
are recorded in `asset_manifest.json` and `reports/real_openh_video_audit.json`.
Frame160 was selected for visual clarity after inspecting nine uniformly spaced
frames (0,32,64,96,128,160,192,224,256): it is the first of those showing the full
phantom target and surrounding folds. No model output or experimental outcome
entered this illustration-only choice. Original RGB pixels and aspect are intact.
