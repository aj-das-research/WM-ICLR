# ICLR reviewer audit — 22 September 2026

Scope: local manuscript, frozen experiment registrations, completion receipts,
primitive evaluation records and current scheduler. This is an author-side
critical audit, not an independent conference review or a prediction of acceptance.

## What is already done

- Current DROID spatial architecture: 21 full 30-epoch runs, internal controls,
  component interactions, native/pooled forecast errors and complementary metrics.
- IWS PushT/Box/Rope: 36 selected predictors and complete development/reserved
  evaluations (600 reserved windows from 30 trajectories), four feature metrics.
- Adapted DINO-WM: six 30-epoch and six raw-coordinate 100-epoch models completed.
  Both variants' weakness against persistence is disclosed; not official SOTA.
- Current spatial simulator campaign: 24 trained models, 2,400 complete executed
  trials. Existing draft incorrectly called these pending before this audit.
- Three shared RGB decoders: 30 epochs each and reconstruction finalization.
  The completed secondary endpoint study adds 45 rows / 9,000 image scores
  across RGB MSE, PSNR, SSIM, LPIPS and UIQI, with explicit numerical revision.
- Source-bound qualitative figures, checkpoint packages and a compiled manuscript.
- No active or pending jobs in the user's scheduler at initial inspection.

## Reviewer questions and actions

| Priority | Reviewer question | Evidence/action | Remaining limitation |
|---|---|---|---|
| High | Is fixed-observation reuse new? | Added DNA/CDNA (Finn et al., 2016) and SNA (Ebert et al., 2017); narrowed novelty to this feature-grid decoder and matched study. | No claim of inventing observation anchoring, advection or temporal skip connections. |
| Critical | Do feature gains improve planning? | Integrate completed current-model control results and success-by-budget curves. Diagnose all saved traces without excluding failures. | Reacher success is 27% versus 80% AR; a favorable control claim is unsupported. |
| Critical | Is the headline DROID gain a held-out result? | State development selection in abstract and main evaluation; preserve all inspected splits. | Needs a genuinely untouched session-disjoint population and frozen evaluation before access. Existing revealed tests cannot be relabeled fresh. |
| High | Why propose bounding when removing it wins on IWS? | Explicitly distinguish the fixed-observation mixing contribution from the optional bound. Preserve the original primary comparison and secondary ablation. | DROID differences are inconclusive; no universal benefit of bounding. |
| High | Are external baselines competent and fairly matched? | Retain both adapted objectives, raw-coordinate/full-budget sensitivity and persistence. | A poor adapted baseline cannot establish published-method superiority. Native-grid external reproduction remains an additional study. |
| High | Is mixing causal, or just extra capacity? | Completed three new 30-epoch patch-local additive controls with exactly matched active parameters. Native h10 advantage remains; pooled difference is inconclusive. | This controls parameter count, not all optimization or expressivity properties; fixed/permuted mixing is not tested. |
| High | Are effects robust to correlated windows and three seeds? | Reconstruct paired seed/episode uncertainty; add seed-specific and multiple-comparison sensitivity to control diagnosis. | Three seeds and ten reserved trajectories per IWS task limit generalization; post-hoc analyses remain secondary. |
| Medium | Do the feature improvements correspond to useful RGB? | Complete shared-decoder evaluation only with frozen trained readouts, identical targets and reconstruction/persistence references. | All five metrics completed. Pixel/perceptual rankings differ, and raw RGB persistence wins LPIPS. FID/FVD and full-clip validation remain unmeasured. |
| Medium | Are latency and compactness claims complete? | Integrate actual encoder-inclusive online planning costs, separate from predictor-only IWS timings. | FLOPs, complete training GPU-hours and cross-hardware matching remain incomplete; no universal efficiency claim. |
| High | Can the reader identify one contribution in nine pages? | Put the main negative planning result in main text, move historical work to the appendix, remove stale pending tables. | The decoder novelty is stated in introduction, method and figure caption; compiled-page checks accompany the final audit. |
| High | Is submission anonymous and reproducible? | Check actual active PDF/source paths, author-linked URLs, template, references, AI disclosure and final artifact consistency. | Human author review and conference upload remain external steps. |

## Submission logistics

Official ICLR 2027 author guidelines checked on 22 September 2026:
https://iclr.cc/Conferences/2027/AuthorGuidelines
Abstract deadline: 18 September 2026, 23:59 AoE. Full paper: 25 September,
23:59 AoE. Main text limit: nine pages, excluding references and permitted
statements; unlimited appendix. The user confirmed that the abstract was registered. Conference submission remains with the authors. The user subsequently authorized GitHub and Overleaf synchronization; its current status is recorded separately.

## Interpretation

The existing results support a bounded-scope forecasting study, not a demonstrated
control advance. Completing a measurement can answer a reviewer question with a
negative result. Further training cannot be promised to reverse it, and repeatedly
selecting changes on the revealed test would invalidate a confirmatory claim.

## Completed during this audit

- Integrated all 2,400 current-model planning trials; disclosed unfavorable control results in abstract and main text.
- Added per-seed planning robustness with Holm correction over eight declared contrasts.
- Executed 528 development candidate plans and scored 6,336 model/candidate pairs to examine action ranking.
- Completed the RGB endpoint evaluation and all three capacity-matched training runs.
- Promoted all four adapted DINO-WM recipes and four feature metrics to the main DROID table.
- User requested gated DINOv3 access; RLA-WM remains pending approval and a matched evaluation.
