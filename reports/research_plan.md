# Five-day VLM test-time adaptation research and release plan

Prepared 18 September 2026. This is a research proposal, not a claim of achieved results or guaranteed acceptance. Hardware allocation is verified; benchmark reproduction, method effectiveness and runtimes are not yet measured.

**Recommended direction: delayed deployment of test-time updates**, a controller around existing adaptation methods. Working title: *Delayed Deployment of Test-Time Updates for Vision–Language Adaptation*. Classification (natural images and histopathology) and open-vocabulary segmentation are the core tasks; generative VQA is a stretch goal. A shared interface supports multiple model families, but task-specific unlabeled scoring and empirical validation remain necessary. We cannot promise any-model, any-task effectiveness.

This is the pre-experiment planning stage requested by the user. The official ICLR manuscript scaffold is compiled in `paper/proposal.pdf`; no model experiments have run. The experiment registry and code reuse plan specify the next implementation stage.

The project will run from `/home/abhijit.das/projects/TTA` on this cluster. Code, downloads, inference, adaptation, checkpoints, analysis, manuscript builds and release builds stay here. GitHub and Hugging Face are publication destinations; they are not assumed to provide additional paid compute.

## Conference timing

The [ICLR 2027 call](https://www.iclr.cc/Conferences/2027/CallForPapers) lists abstract submission on September 18, 2026 at 23:59 AoE and full submission on September 25 at 23:59 AoE: **September 19 and September 26 at 15:59 Dubai**, respectively. Five days from the audit ends September 23, leaving a short buffer before the full-paper deadline.

The [author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) require a genuine abstract, and freeze the author list at the abstract deadline. Existing submission status, human author identities, OpenReview profiles and reciprocal-reviewer eligibility are not established. These are real external dependencies. A plan-only abstract must not be presented as completed research. If we lack an honest submission-ready abstract in time, the five-day output remains a reproducible manuscript/project, not an eligible main-track submission for this cycle.

I can implement, run, analyze and draft the work. Human authors must verify and own its claims and submission. Record this workflow in the required [AI-use disclosure](https://iclr.cc/Conferences/2027/AIPolicyForAuthors). Prepare anonymous supplementary code separately from the public project release.

## Why this direction

| Candidate | Compute fit | Main research/reproduction risk | Decision |
|---|---|---|---|
| Delayed memory validation on frozen VLM features | Excellent; three 32 GB workers, reusable features | Novelty and whether unlabeled proxy utility predicts actual benefit | Primary hypothesis, with early stop criteria |
| Pathology-specific prompt/LoRA adaptation | Good | LATTE/Histopath-C already covers this directly; a small loss change is weak novelty | Medical evaluation and fallback analysis, not a standalone novelty claim |
| Small generative VLM test-time learning for VQA | Possible for a small model | Sampling/optimization cost, evaluator normalization, stronger September 2026 competition | Optional follow-up, outside critical path |
| 3D CT, detection or robotics | Hardware may support individual cases | Data preparation, volumetric memory, dependencies or simulators dominate five days | Survey now; defer experiments |

This is the best **feasibility/research-value tradeoff** I can presently justify, not a numerical estimate of acceptance probability. ICLR competitiveness requires a clear mechanism, strong current baselines, honest evaluation and repeatable gains. More benchmarks alone cannot compensate for an incremental method.

## Proposed mechanism and falsifiable hypothesis

**Hypothesis:** confidence at the time a sample enters a memory does not reliably measure that sample's benefit to later predictions. Testing a candidate update on subsequent unlabeled observations may reduce harmful accumulation under stream changes.

Inspected source establishes a concrete starting point: TDA writes the current sample before computing its cache-adjusted prediction (`external/TDA/tda_runner.py`); PTA similarly updates prototypes before prediction (`external/PTA/pta_runner.py`). This is a legitimate adapt-then-predict protocol, **not evidence of label leakage or a bug**. We will retain their official protocols for reproduction and separately evaluate a clearly named predict-then-update streaming protocol for all methods. Merely changing update order is not the proposed contribution.

General controller: snapshot the active state, let an existing host propose a candidate on past samples, freeze both states while scoring them on later distinct unlabeled samples, and deploy the candidate only for subsequent predictions if its paired proxy benefit exceeds a fixed threshold. The same interface covers caches, prompts and lightweight adapters; copying full model weights is not required. Segmentation requires inverse-aligned dense outputs and image-level score aggregation. See the manuscript for the general formulation.

Initial classification instantiation, deliberately small enough to falsify quickly:

1. Freeze the pretrained encoders and prompts. Maintain an active cache and a bounded candidate cache. Store image features, frozen soft predictions, timestamp and sample ID; never labels.
2. Predict using only currently active state, and log the output before any admission decision that uses this observation.
3. Evaluate candidate-cache residuals on subsequent, distinct sample IDs. A second weak view provides an independent *view*, not an independent model or a ground-truth label. Compare frozen-anchor cross-view predictive loss with and without the candidate block; exclude all self matches and repeated-image IDs.
4. Accumulate a rolling proxy-utility score over, initially, 32 observations. Admit a candidate block only if it improves that held-out-view proxy beyond a development-set threshold. Do not retroactively change recorded predictions. Abandon the proxy if it correlates poorly with measured utility on development labels.
5. Bound the cache's additive logit contribution, and expire stale evidence within a fixed memory budget. Compare this to plain FIFO, fixed decay, random admission, entropy admission and periodic resets. Add more machinery only if those controls fail to explain the effect.

For class logits `z0`, residual `r`, and fixed bound `b`, use `z = z0 + clip(r, -b, b)`. This supplies a simple deterministic statement: if the frozen top-two logit margin is greater than `2b`, the top class cannot change. This is a bound on prediction changes, **not a guarantee of lower error**. It can also block beneficial corrections; quantify that tradeoff. No distribution-free safety or conformal guarantee will be claimed for arbitrary unlabeled shifts.

The view-based proxy may reward frozen-model mistakes; views are correlated; abrupt shifts may make delayed evidence stale; minority classes may receive too little evidence. These are the specific failure cases to test, not details to hide. Test full versus marginal/conditional utility variants only on development data, with a small prespecified sweep.

## Novelty checks before committing

The [literature matrix](literature.md) includes direct overlap: PTA (prototypes), D²O (environment-aware debiasing), Ramen (mixed-domain sample selection), LTTA (long-tail imbalance), StatA (correlated/partial-class streams), DLAE and ReTA (cache reliability), and the 2025 TTA-VLM critical benchmark. ComMem, DANCE and the June 2026 controlled-update study also require full-text comparison even if their implementation cannot be verified.

The September 8 [CAS paper](https://arxiv.org/abs/2609.08367) already studies selective adaptation. Its repository is currently a release placeholder: a paper-based CAS reproduction, explicitly labeled as ours, is needed if official code remains unavailable. Delay-only and CAS-plus-delay controls are mandatory. Adjacent [RMemSafe](https://arxiv.org/abs/2605.14063) and [unsupervised TTA model selection](https://github.com/cygerts/unsupervisedtta) further rule out broad claims that reliability gating or label-free selection are new.

Claim to investigate: **prospective, distinct-sample proxy utility for deciding whether an arbitrary host update becomes active**, plus a transparent evaluation of harmful carryover and recovery. Do not claim the first reliable cache, first realistic stream, first medical TTA, first two-memory system, or first delayed update. A search without a match does not establish novelty. A more complete method-section audit and Day-1 diagnostics can reject this proposal.

## Data and model acquisition

All downloads are initiated from this cluster, with resumable transfer, provenance, provider checksums, local SHA-256 and pinned model revisions. The confirmed initial sources are in `references/download_sources.json`.

| Priority | Data/model | Acquisition and purpose |
|---|---|---|
| Core | CIFAR-10-C, CIFAR-100-C | Official Zenodo archives, 2.918 GB each. Download corruption archives only, not CIFAR-10-P. Natural corruption streams. |
| Core | ImageNet-A, ImageNet-R | Authors' public download links; no dependency on the original ImageNet training archive. Use correct 200-class mappings and original evaluation conventions. |
| Core | PASCAL VOC segmentation through MLMP | Verify official download and split manifest on Day 1; start with one CLIP segmentation host and clean plus selected corruptions. Reproduce upstream before wrapping its update state. |
| Core | CRC-VAL-HE-7K | Official Zenodo archive, 800 MB, 7,180 patches. Run Histopath-C's ten corruptions and clean evaluation. |
| Secondary | NCT-CRC-HE-100K | 11.69 GB normalized archive, only if needed for a separately declared development split or replication. Official 7K cohort has no patient overlap with 100K. Never tune on 7K evaluation labels. |
| Secondary | PathMNIST/DermaMNIST, official MedMNIST splits | A small additional medical domain if time permits; benchmark robustness only, not proof of clinical generalization. |
| Core | OpenAI CLIP ViT-B/16 and ViT-B/32 | Pin official pretrained weights, tokenizer, resize/crop and text templates; second backbone validates natural-domain transfer. |
| Core | QuiltNet-B-32 | Hugging Face `wisdomik/QuiltNet-B-32`; API verified ungated, revision recorded. Pathology backbone used by Histopath-C. |
| Secondary | BiomedCLIP | Hugging Face `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`; API verified ungated, revision recorded. Extra medical backbone. |

Access and actual downloaded bytes must be checked on Day 1. We do not assume MIMIC credentialing, full ImageNet licensing, gated CONCH access or MoBE expert downloads will complete within five days. Avoid those on the critical path. A downloadable model is not necessarily licensed for every redistribution: release adapters/state and download instructions when base weights cannot be republished.

Use shared NFS for durable checkpoints/manifests; check compute-node local scratch before using it for unpacking/batched reads. Avoid generating millions of tiny cached files. Store feature arrays by dataset, checkpoint hash, preprocessing hash and augmentation seed. They can be reused only for frozen-encoder methods. Never reuse frozen features for an encoder-updating baseline, and count extraction time in end-to-end costs.

## Top-down reproduction and evaluation

1. **Official behavior first:** reproduce zero-shot, TDA, StatA and one 2026 baseline (PTA or D²O) on a small public dataset using their stated configurations. Record reference versus reproduced accuracy, preprocessing and exact command. Aim for agreement within reported variation; investigate discrepancies above 0.5 percentage points rather than silently adjusting prompts.
2. **Medical anchor:** reproduce Quilt zero-shot and LATTE on CRC-VAL-HE-7K and at least one corruption before modifying the medical pipeline. LATTE uses transductive batches: compare under matched information access, or show it separately, not as an identically online baseline.
3. **Common evaluation harness:** retain official runs and add standardized runs with identical backbone, prompts, views, stream order, batch size, adaptation order and resource budget. Unit-test that labels cannot enter adaptation, state resets are correct, checkpoint/resume reproduces the stream, and future observations are never read early.
4. **Primary frozen-feature baselines:** zero-shot, matched-view zero-shot ensemble, TDA, StatA, PTA, D²O; add DOTA/SCA if integration is inexpensive. TPT/A-TPT and LATTE are matched-budget gradient baselines on a smaller prespecified subset. Ramen is useful for mixed domains but is a stretch comparator if setup overruns.
5. **Stream settings:** shuffled stationary; class-correlated/long-tailed; abrupt corruption shift; return to clean; recurring corruption. Domain boundaries and class labels are available only to the evaluator. No oracle resets for the proposed method. When clean/corrupted copies share an original image, keep all copies within the same development/test split, explicitly identify repeated-image streams, and add a disjoint-image recurrence control.
6. **Metrics:** top-1 and balanced accuracy, negative log-likelihood, Brier score, ECE with fixed binning, worst-domain/segment accuracy, degradation relative to the same frozen model, recovery after a change, milliseconds/image, encoder forwards, backward steps, peak VRAM and memory size. For multi-label extensions use per-label AUROC/AUPRC rather than a multiclass softmax, but these are optional.
7. **Statistics:** three prespecified stream seeds on all core settings, five on the small stress test if affordable; report paired effects and intervals. Use block/stream bootstrap for dependent observations; patient/slide clustering only when metadata supports it. Do not claim patient-level intervals from unidentified tiles.
8. **Ablations:** delay only; proxy admission only; bound only; full method; random gate; entropy gate; FIFO/decay; fixed reset. Match memory and encoder-view budgets. Hold hyperparameters fixed across final datasets after development selection.

Natural core targets: all 15 standard corruption types at severity 5 on CIFAR-10/100-C, plus full ImageNet-A/R once available. Other severities become secondary unless throughput supports them. Medical core: full 7K clean and ten Histopath-C corruptions. Treat these as two benchmark families, not eleven independent clinical datasets. Reserve development image IDs before generating corrupted streams. Add a second medical dataset if the primary claim spans medical domains.

## Five-day schedule and compute budget

The verified limits allow two `ws-ia` workers and one `gpu` worker. Use 4 CPUs and 24 GB host RAM per worker initially; this fits aggregate QoS. Use 7 h 45 min jobs on `gpu`, 23 h jobs on `ws-ia`, and checkpoint regularly. Queued arrays must use at most two concurrent workstation tasks. Do not request eight GPUs or multi-node DDP.

| Time from start | Work and measurable exit condition | Planned GPU-hours |
|---|---|---:|
| Day 1, 0–24 h | Download/checksum public data and models; isolated environments; reproduction and 1,000-image throughput pilots; full-text novelty check; diagnose harmful memory carryover. Decide whether an honest abstract is ready before its deadline. | 30 |
| Day 2, 24–48 h | Implement minimal candidate/active memory and resumable stream harness; correctness tests; pilot ablations on development data; freeze method and hyperparameters. | 45 |
| Day 3, 48–72 h | Three workers run natural baseline grid, medical baseline grid, and segmentation/proposed-method grid; log full predictions and costs. | 65 |
| Day 4, 72–96 h | Additional seeds/backbone; recurrence controls; uncertainty estimates; rerun discrepant results; generate all tables/plots directly from logs. Draft full paper. | 75 |
| Day 5, 96–120 h | Reproduction from a fresh environment, checkpoint round-trip, paper/appendix audit, local demo and project-page builds, GitHub/HF release package and model/data cards. | 25 |
| Total | 240 GPU-hours of scheduled work, plus 120 theoretical GPU-hours of headroom | **240** |

These are allocation budgets, not measured runtimes. Estimate actual demand after the pilot as `images × views × measured encoder seconds / 3600`, plus serial adaptation and data I/O. Measure gradient baselines separately. If the revised total exceeds 260 GPU-hours, drop optional generative/3D work and extra dense settings, optional datasets, extra backbones and expensive baselines' secondary settings before reducing seeds on the main claims. Never split one online stream across workers without carrying its exact state.

**Decision gate by end of Day 1:** a reproducible carryover failure beyond ordinary frozen-model shift, and a plausible gap relative to the closest 2026 papers. Otherwise redesign or produce a reproduction/diagnostic report.

**Decision gate by end of Day 2:** on held-out development streams, a useful proxy-utility signal, improvement beyond FIFO/entropy/reset controls, and evidence in both a natural and a medical pilot. Suggested engineering target: at least 1 percentage point balanced-accuracy gain over the strongest matched baseline on two stress settings, without more than 0.5-point loss on stationary clean data. These thresholds are planning criteria, not promised outcomes and not a license to select favorable final benchmarks.

**Submission gate on Day 5:** meaningful paired effects on prespecified final tests, full controls and current baselines, defensible novelty, no unexplained reproduction discrepancies, correct citations and an audited manuscript. If this fails, preserve the negative findings and code rather than write an unsupported success narrative.

## Paper and release deliverables

The paper will center on one research question, one minimal mechanism, its limitations and evidence. Planned figures: harmful-update diagnostic; method timeline; accuracy/calibration/cost comparison; shift-and-recovery traces; ablation panel. Planned tables: main natural results, medical results, and matched-budget costs. The appendix contains per-domain results, all seeds, data/protocol details, environment manifests and exact commands. No placeholder numeric results enter the manuscript.

Repository target layout: `src/`, `configs/`, `tests/`, `scripts/slurm/`, `data/manifests/`, `results/`, `checkpoints/`, `paper/`, `demo/`, `site/`. Raw datasets and licensed base checkpoints stay out of Git. Keep clean patches/adapters around pinned upstream checkouts; do not rewrite upstream history. Several downloaded repositories lack a top-level license; use them for inspection/reproduction and avoid copying their code into the public release without appropriate permission. TDA and Histopath-C provide MIT-licensed starting points.

Checkpoint contents: active/candidate memory, counters, RNG state, stream cursor/ID ordering, frozen-model revision and config hashes; optimizer state only for learned adapters. A gradient-free method need not invent a trained model checkpoint—publish its state, configurations and model card accurately.

Demo: a local Gradio app showing zero-shot versus adapted predictions along a short image stream, reset/resume and memory-admission diagnostics. Provide a lightweight CPU replay mode for Hugging Face Spaces; it must be explicitly labeled as replay. Live inference on a CPU Space should use a small model and measured limits, with no assumption of free HF GPU access. Keep medical examples clearly research benchmarks.

Project page: static, built and previewed on the cluster, with paper, method, measured plots, demo and reproducibility commands. GitHub/HF release: versioned source, requirements locks, manifest/checksum files, third-party notices, checkpoint/adaptor card and a runnable Space directory. Public publishing requires an authenticated destination owned by the user; no GitHub/HF write access or namespace has yet been verified. Build everything locally before resolving that final dependency. Use anonymous code/materials for review and publish identifying assets in accordance with the conference's anonymity rules.

## Status at the end of the planning audit

Completed: cluster inventory, per-user QoS discovery, three concurrent hardware probes, current literature/repository audit, eleven pinned upstream implementation checkouts plus one code-release placeholder, official data/model source checks, this execution/release plan, accepted-paper analysis, experiment registry, code reuse plan, and compiled official ICLR 2027 paper scaffold. No baseline scores, measured model throughput, trained adapters, paper results or live public deployments exist yet. Those are the scheduled execution deliverables, not completed work.
