# What accepted papers imply for our proposal

This is an analysis of contributions and experimental evidence, **not an explanation of confidential acceptance decisions**. OpenReview's browser challenge prevented access to several discussion pages. I do not infer reviewer opinions or numerical acceptance odds from a paper's venue. PDFs and extracted text that could be downloaded are under `references/papers/`; access failures are recorded in its manifest.

| Accepted example | Contribution and evidence inspected | What we should learn |
|---|---|---|
| [DynaPrompt, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8270bf9237b7d2c9a8dfce5488f000a4-Abstract-Conference.html) | Dynamic selection, addition and eviction of prompts address online prompt collapse; evaluates 14 datasets. Main contribution is the adaptive mechanism and supporting experiments. | A reusable update mechanism can be meaningful without training a huge model. Selection and memory alone are already established. Compare to it rather than rediscover it. |
| [A-TPT, ICLR 2026](https://arxiv.org/abs/2510.26441) | Identifies limitations of average dispersion/orthogonality and substitutes max-min angular diversity. Includes geometry/gradient analyses, natural/medical calibration, multiple backbones, seeds and Pareto plots. | A focused objective change can be substantial when the failure mechanism is precise and the nearest alternatives are directly challenged. Calibration gains must be shown alongside accuracy. |
| [Efficient Test-Time Scaling, ICLR 2026](https://arxiv.org/abs/2510.03574) | Token-level augmentation aggregation and pseudo-label adaptation for small generative VLMs, nine benchmarks, multiple model scales, and component/compute analysis. | Small-model methods are viable. A cost-versus-quality argument needs matched inference budgets and careful answer evaluation. Generative VLMs are a distinct task family, not interchangeable with CLIP. |
| [Noisy TTA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf) | Formalizes unlabeled streams containing out-of-label-space examples; analyzes failure of existing adaptation and introduces an adaptive detector. Joint classification/detection evaluation spans clean and noisy streams. | A new deployment regime can support a contribution only if it changes the problem and is paired with convincing diagnosis and a response. “Realistic” by itself is not a novelty claim. |
| [PTA, ICML 2026](https://github.com/hzhxmu/PTA) | Confidence-weighted class prototypes anchored to text; 15 image and four 3D recognition benchmarks, speed/memory comparisons and component analysis. [Full paper](https://arxiv.org/abs/2604.21360) downloaded. ICML acceptance verified from author repository. | Strong breadth plus a simple, efficient algorithm can be persuasive. The relevant advantage is a measurable change in state representation and computation, not the number of modules. |
| [DOTA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/d24189d251aa453b6d2eb3421ec48bcc-Paper-Conference.pdf) | Gaussian distributional classifier built from soft zero-shot probabilities; Proposition 3.1 connects estimation to an EM step. Covariance/all-sample ablations probe that formulation. | Useful theory explains the actual update and its assumptions; it need not be a sweeping convergence guarantee. We must test the assumptions that matter, such as whether the utility proxy tracks benefit. |
| [ZERO, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e92cb6f981a2cacb2a710ecaa0d7b141-Abstract-Conference.html) | Analyzes marginal-entropy prompt adaptation and develops a zero-temperature aggregation alternative without backpropagation; includes matched-hardware cost evaluation. | Removing optimization can be a contribution if analysis explains why and controlled evidence backs it. The strongest cheap alternative must be in our comparison. |
| [TTA-VLM / Illusion of Progress, NeurIPS 2025 D&B](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b57ddd8726c217a6fef9a48ce3e09ffd-Abstract-Datasets_and_Benchmarks_Track.html) | Unified evaluation of episodic/online methods across 15 datasets, including calibration and stability. Distinct Datasets & Benchmarks track. | A diagnostic contribution can matter, but this broad critique already exists. A smaller repeat of its conclusions is not enough; our diagnostic must isolate a specific new mechanism. |

## Contribution standard for this project

There is no requirement that every accepted empirical paper introduce a new architecture, optimization algorithm **and** theorem. The consistent pattern in this sample is: a precise limitation, a mechanism that addresses it, comparison against the closest alternative, and evidence that distinguishes the mechanism from extra compute or tuning. This is an inference from these papers, not an official acceptance formula.

For us, the proposed primary contribution is **delayed deployment of an adaptation update**, tested as a controller over existing methods. The current draft asks whether improvements on later, disjoint unlabeled probes are more useful than immediate confidence for deployment decisions. Classification caches are one implementation, not the definition of the method.

The crucial latest comparator is [CAS / Selective Adaptation, ECCV 2026, September 8 preprint](https://arxiv.org/abs/2609.08367), with a [public repository](https://github.com/sirujiang/selective-adaptation) whose implementation is still announced as coming soon. It already decides whether to adapt based on cross-augmentation similarity. Therefore “skip harmful updates” and “use consistency to decide” are **not** sufficient distinctions. Our candidate must show an additional benefit from separation of update-generation data, prospective evaluation data and deployment time. Compare at equal encoder-forward budgets and include delay-only and CAS-plus-delay controls.

Also compare the assumptions to [unsupervised TTA model selection](https://github.com/cygerts/unsupervisedtta) and [reliability-gated source anchoring](https://arxiv.org/abs/2605.14063). A frozen model is not always a trustworthy anchor. These adjacent works constrain any claim of task-independent correctness.

## Generality: design interface versus demonstrated evidence

The controller needs `predict`, `propose_update`, `snapshot/restore`, and a bounded, label-free `score_pair` on aligned outputs. It does not require a particular transformer, model size or label vocabulary. The **scorer and state adapter are task-specific**; the controller can remain identical.

| Task family | Existing host code | Candidate state | Label-free probe | Final evaluator |
|---|---|---|---|---|
| Natural and medical classification | TDA, PTA, Histopath-C/Quilt | Cache/prototypes, or lightweight prompt state | Bounded cross-view prediction divergence relative to active/frozen anchors | Accuracy, balanced accuracy, NLL, Brier, ECE |
| Open-vocabulary segmentation | MLMP | Selected prompt/normalization parameters with exact snapshot/restore | Pixelwise divergence after inverse-aligning weak geometric views; aggregate per image | mIoU, per-class IoU, corruption/recovery curves |
| Generative VQA, stretch | Efficient Test-Time Scaling | Small prompt/LoRA state if supported by host | Same candidate response teacher-forced under matched contexts; normalized token/answer consistency | Official task metric and answer normalization, latency |

Token scores do not automatically support object detection, variable-length outputs or arbitrary black-box APIs. If an API cannot preserve/restore adaptation state or supply comparable outputs, it is outside the initial interface. We will claim model/adapter portability only for tested instances, not “all models, all tasks.”

## Theory that would actually help

1. Define the controller's information flow precisely: proposed state is constructed from past data; both candidate and reference are frozen during a probe block; deployment decisions affect only later predictions. This supports a no-lookahead argument and executable invariants.
2. A bounded paired utility score permits concentration analysis under explicit assumptions about prospective observations. If repeated candidates are selected adaptively, fixed-sample bounds cannot be recycled without accounting for selection and repeated testing. A conditional martingale or alpha-spending treatment is a possible analysis target, not a completed guarantee.
3. Proxy improvement is not ground-truth risk improvement. A transfer bound requires an explicit proxy-to-risk discrepancy assumption. We should estimate and expose this discrepancy on held-out development labels, then report failure cases, rather than hide it inside an unrealistic theorem.
4. For the classification implementation, an additive logit bound provides a simple margin-preservation lemma. It is useful for implementation tests, but elementary and **not sufficient theoretical novelty by itself**.

Universal label-free accuracy improvement is impossible without assumptions: two deployments can present exactly the same unlabeled inputs but different correct labels. An unlabeled decision rule cannot distinguish them, and a changed prediction can help one and hurt the other. We should present scope honestly and pursue evidence-backed portability, not an impossible promise.

## Paper strength gates

- Establish that delayed evaluation predicts useful *future* deployment changes, beyond confidence/CAS, random acceptance and equal-budget delayed updates.
- Show gains with at least two different adaptation-state families, two task families and multiple pretrained backbones if making broad controller claims.
- Include stable/clean streams where adapting can be unnecessary, severe shifts where the frozen anchor is wrong, recurrence, imbalance and the cost of extra probes.
- Make the negative result actionable: if prospective utility is unreliable or the nearest comparator matches it, change the claim or abandon the method. Do not dress an extra gate plus a trivial lemma as established ICLR-level novelty.

The five-day advantage comes from frozen-feature reuse, small state snapshots, published benchmark pipelines and a narrow mechanism. It does not come from avoiding strong baselines, reducing statistical rigor or launching an infeasible all-model/all-task grid.
