# Prespecified experiment registry

Status: planning only, 18 September 2026. No result exists. Freeze exact split/sample hashes, configurations and seeds before final evaluation. Public test labels belong exclusively to the evaluator. Scope follows the latest general-controller proposal in the manuscript.

| ID | Question / scope | Backbone and upstream | Required comparison | Output |
|---|---|---|---|---|
| R0 | Can official behavior be reproduced? | CLIP B/16, B/32; TDA and PTA/D2O; Quilt with Histopath-C; MLMP segmentation host | Published protocol versus local reproduction, record every difference | reproduction.csv |
| E1 | Does delayed deployment help natural classification? | CLIP B/16 and B/32; CIFAR10/100-C, ImageNet-A/R | Frozen, view ensemble, host, controller+host | natural.csv, per-sample predictions |
| E2 | Does it transfer to a medical domain? | Quilt; CRC-VAL-HE-7K clean and Histopath-C | Frozen, host, controller, LATTE in separately matched transductive setting | medical.csv |
| E3 | Does it transfer to another task? | MLMP-supported CLIP segmentation host; VOC clean/corruptions | Frozen, original adaptation, delayed deployment | segmentation.csv, image-level mIoU records |
| E4 | Which component explains any effect? | One natural and one dense pilot, then selected final conditions | Delay only, random gate, entropy gate, CAS, CAS+delay, full controller; match views and state size | ablation.csv |
| E5 | How does it behave as distributions change? | Class-correlated, abrupt shift, clean return, recurrence | Same stream and allowed information for every method | recovery.csv, immutable predictions, gate trace |
| E6 | Does the unlabeled score predict actual benefit? | Prespecified development streams, separate final diagnostic set | Proxy versus evaluator-only paired utility; confidently wrong anchors; minority classes | proxy.csv |
| E7 | What does deployment cost? | All core hosts | Same images, count candidate/reference forward passes, snapshots, extraction, I/O | cost.csv, peak VRAM, end-to-end wall time |
| S1 | Optional generative transfer | Small VLM from efficient_test_time_scaling; public VQA subset | Frozen, TTAdapt, controller+TTAdapt | vqa.csv; execute only after core completion |

Core uses three stream seeds; development and final image identities are disjoint. Keep all corruption versions of an image in one split. Medical intervals cannot be described as patient-level without patient IDs. Segmentation pixels are not independent statistical replicates: score and resample at image/stream level. Report classification accuracy, balanced accuracy, NLL/Brier/ECE; segmentation mIoU; uncertainty from paired block/stream resampling. VQA requires its official answer normalization and separate scorer.

Two tasks and three named pretrained checkpoints are the minimum planned generality evidence, not proof of universality. Medical classification is a new domain, not a third task. Do not run a Cartesian product of every method, model and dataset. Use full baselines on the primary configuration and prespecified transfer slices; record exclusions before seeing final labels.

Budget: 240 GPU-hours planned across three independent workers, with 120 theoretical hours of headroom. Day-1 throughput and memory pilots determine exact feasible counts. Preserve E3 if claiming cross-task transfer; if E3 fails, narrow the paper claim explicitly. Small-pilot success is not permission to choose only favorable final tests.

All numeric manuscript cells remain `--` until records exist. Plots are generated from these records; they are currently empty axes. Each run must record upstream commit, local patch hash, model revision, data manifest hash, config, stream seed/order, hardware, resource usage, start/end, and failure status. Failed runs remain in the registry.
