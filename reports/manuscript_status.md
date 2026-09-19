# Manuscript and experiment handoff

Updated 2026-09-19 (Asia/Dubai). Research draft; no conference submission. GitHub, the project page, and Overleaf have been published. The fresh-session confirmation appendix, two complete tables and the technical qualitative figure are compiled and visually reviewed in a compiled manuscript snapshot.

## Current experiments and manuscript — 19 September, 12:02 UTC

Full IWS training is now submitted:27 registered runs, three tasks, three matched
learned arms and three seeds, each with30 complete epochs at H60. Arrays200640
(ws-ia, two concurrent jobs) and200645 (gpu, one concurrent job) execute the
frozen recipe. The live account permits three simultaneous GPUs; its many idle
nodes do not override these account limits. See `reports/current_results_and_gpu_status.md`
and `reports/real_video_iws/live_status.json` for the timestamped current queue.

The model/trainer/evaluator/finalizer and paper ingestion passed146 combined CPU
tests plus all-task cache metadata checks and full-shape allocated GPU profiles.
The final microbatch64 profile uses6.2–7.1GiB and projects20.4 GPU-hours for the
complete training/epoch-validation grid, excluding loading/checkpoint/final
evaluation overhead. This is a timing estimate, not new predictor evidence.

The rebuilt main-plus-appendix PDF contains69 pages. Main wording defines the
forecasting problem and relative feature-MSE gains precisely; the new appendix
matrix states task inputs, targets, metrics, controls and completion status.
DROID retains3.01%/5.30% h5/h10 gains versus matched AR. IWS and external SOTA
comparisons remain pending. Generated result tables accept only the validated
complete campaign, never partial training logs. Accepted figures are unchanged.

## Historical compact task and decoder diagnostics — 19 September, 10:59 UTC

Figures 6 and 7 are redesigned at **5.5 × 2.35 inches** each, with regular 8 pt
labels. They now share **page 15** with a short explanation, avoiding the split
and unused page area produced by the first integrated layout. The continuous
main-paper-plus-appendix PDF remains **67 pages**. Accepted Figures 1/2 are
unchanged; the historical camera inspection figure now appears on page 16.

Figure 6 uses the accepted layered feature objects, a compact recorded-command
strip and a clear one-way prediction-to-evaluator connection. All four original
DROID frames, the exact 2 past + 10 future action blocks and the evaluator-only
frame 60 boundary are retained. Its heatmap, linear 0–0.787 scale, MSE 0.1727 and
shown-window regression −0.305% are unchanged. The graphic is 13% shorter.

Figure 7 aligns the same three recorded cases and their six raw/gated source
weight maps beside one shared gate plot. All 96 map cells, 90 seedwise gate
values and 30 means/ranges are preserved. Each seed's gate is applied before
averaging effective weights. Direct case labels, distinct line styles and
markers support reading in color or grayscale. Images identify recorded
observations; the maps describe latent mixtures, not physical flow or a causal
explanation of prediction-error differences.

Reproducible sources are `paper/scripts/render_task_story.py` and
`paper/scripts/render_mixing_story.py`. Briefs, alternative layouts, image
identity receipts and actual-PDF proofs are under `paper/design/task_story/`
and `paper/design/mixing_story/`. Independent source, numeric and visual
reviews are in `reports/evidence/*_story_independent_review.json`.

No experiment or checkpoint changed during this figure revision. The live
scheduler check at 10:55 UTC listed **zero running or pending account jobs**.
The current results/gaps table is `reports/current_results_and_gpu_status.md`:
21 spatial models complete; 117 public predictors across all studies; 6 additional
component predictors local; real IWS caches complete but predictor training and
evaluation still pending. Broad SOTA superiority is not established.

## Historical visual story and architecture — 19 September, 10:34 UTC

Figures 1 and 2 now share recorded DROID observations, structured feature
objects and a brighter palette. Figure 1 is the user-approved teaser: it shows
the forecasting task, supplied action blocks, recursive versus fixed-reference
routes and all 141 development episodes (136 lower-error, five higher-error).
The source-linked 5.30% callout remains the ratio of population mean errors.

Figure 2 exposes the proposed decoder instead of hiding it in module boxes.
The fixed observed tensor and its mixed version have separate visible paths;
complementary patch gates weight them before their sum. A separate projected
correction is bounded by tanh and added at an explicit merge. The observed
support, past-only context, causal action-prefix GRU, pre-FiLM keys and all
28 source dependencies are preserved. The spatial encoder remains trainable;
only DINOv2 is frozen. Feature colors, gates and bounded-coordinate examples
are explanatory schematics, while the photographs are unchanged observations.

The rebuilt continuous main-paper-plus-appendix PDF remains **67 pages**.
Figures 1 and 2 occupy pages 2 and 3 at the official 5.5-inch width with regular
8 pt labels. Main comparison and qualitative figures remain on pages 5 and 6.
Sources are `paper/scripts/render_teaser_story.py` and
`paper/scripts/render_spatial_architecture_story.py`, with PDF, editable SVG,
PNG, briefs, provenance and independent checks beside the generated artifacts.
The project page uses outlined exports of these same reviewed PDFs.

This revision changes presentation only. Models, checkpoints, numerical
results, uncertainty and retained unfavorable outcomes are unchanged. Final
publication commits and public-download parity are recorded by the sync service.

## Historical figure refinement — 19 September, 10:03 UTC

The continuous main-paper-plus-appendix PDF is now **67 pages**, down from 71
after removing repeated display material. Main figures remain on pages 2, 3,
5 and 6; references occupy pages 6–7. The appendix guide and Appendix A begin
together on page 8. The official ICLR width, margins and body typography are
unchanged. Figure labels are regular 8 pt at the actual 5.5-inch display width.

| Figure | Change | Evidence retained |
|---|---|---|
| 1, p2 | Clear recursive route versus shared fixed-observation bank; compact aligned destinations | Same generated concept asset and source-linked 5.30% native h10 development callout |
| 2, p3 | Muted palette, filled feature tiles, clearer branching and a compact bounded-innovation strip replacing the large tanh curve | All 28 semantic edges, action-prefix conditioning, pre-FiLM keys, gate, anchor and correction |
| 3, p5 | Forecast profile beside horizontal paired confidence-interval rows | Same 50 forecast means and six paired 95% intervals; all arms and all 36 contrasts remain in tables |
| 4, p6 | Three compact qualitative rows plus signed error-gap curves | Prespecified best/median/worst, unchanged full RGB frames, six common-scale error maps, seed ranges and regressions |
| 6, p15 | Compact prediction and withheld-evaluation lanes with simple feature tiles | Exact input access, action-block counts and original measured median error map |
| 7, p15 | Unique raw-versus-gated source weights and gate traces | Same fixed three cases and query patch; per-seed gating before averaging; no causal or physical-flow claim |

The former duplicate spatial architecture, repeated qualitative gallery,
family-overview graphic and historical pictorial teaser are omitted from the
PDF. Their sources remain archived. The full historical numeric comparison,
all ablations, selection records, negative/null outcomes and complete replay
are retained. The new decoder diagnostic adds measured information instead of
repeating the main architecture. Figures 6 and 7 now share one page.

Independent checks reproduce the 50 means, six paired bootstrap intervals,
qualitative error maps and all effective mixing matrices. Actual-size color,
grayscale and integrated PDF inspections check labels, arrows and spacing.
Review receipts live in `reports/evidence/*refined*review.json`,
`qualitative_compact_independent_review.json` and
`mixing_diagnostics_independent_review.json`. This is a presentation revision;
no experiment, checkpoint, selection rule or measured outcome has changed.
Publication commits and anonymous-download checks are recorded by the private
sync service after the final reviewed build is uploaded.

## Historical single-method organization — 19 September, 08:29 UTC

The main paper now proposes **one observation-anchored spatial ShiftWM model**
(`transport`, historical Ours-5). The other spatial configurations are component
ablations; the earlier two-context model remains a distinct historical study.
Selection is based on development evidence, not a claim that all metrics improve:
the bounded arm has the lowest native h10 mean, while unbounded mixing has the
slightly lower h5 mean and native bounding increments remain inconclusive.

The single PDF remains 71 pages: main narrative 1–6, references 6–7, appendix
guide 8 and full evidence from 9. The four main figures are:

1. Observation-anchoring teaser, p2: original concept artwork with new, source-bound DROID result and schematic evidence routes.
2. Detailed spatial architecture, p3: observed support, causal action GRU, fixed pre-FiLM keys, spatial mixing, gate and bounded correction.
3. Forecast and paired-comparison panels, p5: one ShiftWM against matched controls.
4. Parallel qualitative cases, p6: largest gain and regression, unchanged recorded frames and the same unclipped feature-error scale.

The previous Figure 2 family overview moves to Appendix A (now Figure 9, p18).
The earlier complete context architecture and simulation teaser stay with their
historical study (Figures 15/18, p36/40), rather than implying the spatial model
produced those results. Table 1 has all eight endpoint rows and groups component
removals as ablations. Exact checkpoint IDs, all 36 spatial paired contrasts,
negative/null outcomes and historical primary endpoints are preserved.

Title, abstract, main prose, comparison labels and documentation now describe
one algorithm. The abstract states a 5.3% native ten-step endpoint error reduction
versus autoregression on DROID development data. It does not establish fresh
test, physical-control, cross-domain or state-of-the-art superiority. All 21
spatial/component runs completed 30 epochs and exact relocated reload checks;
15 spatial models are public and six component models remain local. These are
presentation changes, not new experiments. Independent source, numerical,
actual-width and integrated-paper reviews are recorded in reports/evidence.

The broader archive still contains 117 public predictors across five releases.
Real IWS caches are complete (1,804 trajectories /360,473 native frames);
training and evaluation there remain pending. Runtime synchronization receipts
record the exact GitHub, Overleaf and Pages commits separately.

## Historical manuscript organization — 19 September, 07:53 UTC

The paper is now a **single 71-page PDF with the main paper and appendix attached**.
The main narrative occupies pages 1–7 (references start on 7 and continue on 8),
with a linked appendix guide on 9 and complete evidence beginning on 10.
Four coherent main figures present the recorded-video task (p2), method family
(p3), spatial forecasts and mixing/bounding comparison (p5), and matched gain
and failure examples (p7). Tables 1/2 summarize native spatial errors and
original-context simulation gains; favorable point gains remain green and bold.

Full earlier architectures, all result tables, mixed outcomes and frozen
protocols remain in the grouped appendix. The original planning endpoint is
still primary for its original registered study. The abstract is unchanged
at 140 words. Official margins, body typography and style files are unchanged.
Each new figure has editable vector assets and a reproducible source; independent
numerical checks and actual-width reviews are recorded under `reports/evidence`.
Placement-only regenerations have exact rendered-pixel parity. Final combined
layout review/publication receipts document the final artifact separately.

The completed six-model component follow-up remains exploratory development:
all 30 epochs and exact relocated reloads; mixing improves native error with and
without bounding, while native bounding increments and all interactions remain
inconclusive. There are 117 public predictors; these additional 6 are local.
All three real IWS feature caches are now complete and validated: 1,804 trajectories,
360,473 native frames. Box/Rope full-package validators passed all 1,204 packages.
Reserved official-validation IDs remain excluded; predictor training/evaluation
on IWS remains pending. No GPU jobs were running at the cache completion check.

## Historical manuscript update — 07:23 UTC

The registration abstract is now 140 words in `paper/abstract.txt` and `paper/main.tex`, with one numerical development finding (5.3% relative MSE reduction). Independent factual audit passed. The 67-page PDF adds stable Ours-1–Ours-5 labels, a paired-interval comparison plot, and all16 spatial contrasts; actual integrated pages1/60/61/62 passed visual review. User's Overleaf title formatting edit was preserved. Current numerical evidence remains mixed across protocols; green bold entries mean favorable point estimates, not statistical significance. New IWS PushT cache is complete (600 recordings,119,887 frames); predictor results are not yet available. The six-model spatial component study is now complete (all30 epochs,20 effects,six exact relocated predictors): mixing improves native errors with or without bounding, while native bounding increments and all interactions remain inconclusive. Completed Ours-3/4 results are integrated and visually reviewed in Figure23/page62; Table41/page61 retains all20 component effects. Box/Rope caches are running as200465/200466 after independent84-test review; model training remains pending.

## Completed work

| Study | Completed work | Interpretation |
|---|---|---|
| Original PushT/Reacher | 30 original training runs and three-seed forecast/planning comparisons | Held-out forecast improvements, mixed planning; no established planning superiority |
| Dynamics-residual revision | Two 30-epoch development runs | No improvement over donor across both environments |
| Rollout/context revision | Eight 30-epoch development arms and planning evaluations | Registered promotion criterion not met |
| Drone/tissue simulator extensions | 36 original and six observation-gain runs, all 30 epochs, forecasting and planning | Positive and negative development outcomes retained; final tests remain unrun |
| Real DROID | 12 full 30-epoch runs and 48 evaluations | Small short-horizon gains; long-horizon regressions; separate from physical control |
| Real DROID development diagnosis | All 12 best models, three late checkpoints, train/validation motion and command-sensitivity probes | Real command dependence exists; late overfitting and excessive predicted motion remain |
| Matched residual calibration | 12 train-fitted scalar wrappers, h5/h10 validation comparisons, 24 exact offline/relocated reload checks | Modest validation improvement; no new test result or neural training |
| Fresh real-DROID confirmation | 96 evaluations of all original/calibrated variants, four modes, three seeds, two cameras and two horizons | Prespecified h5 gain of 0.742% over equally calibrated Framewise; interval excludes zero; h10 comparative gains remain inconclusive |

DROID ingestion downloaded and audited 1,126 actual robot episodes (22,017,792,821 raw download bytes), with 851 train / 143 validation / 132 test episodes separated by recording session. The model uses recorded commands and frozen DINO features; no synthetic image corruption is applied. Data are a prespecified subset, not the full published DROID benchmark.

## Real-video findings

| Population | Final-horizon MSE: ours | Framewise | Relative error reduction | Paired difference CI |
|---|---:|---:|---:|---|
| Camera 1, 5 blocks | 0.138198 | 0.138479 | +0.203% | [-0.001038, +0.000413] |
| Camera 1, 10 blocks | 0.193525 | 0.189895 | -1.912% | [-0.000387, +0.007493] |
| Camera 2, 5 blocks | 0.154190 | 0.154337 | +0.095% | [-0.000830, +0.000644] |
| Camera 2, 10 blocks | 0.222748 | 0.220020 | -1.239% | [-0.000741, +0.006195] |

All four final-horizon intervals versus Framewise include zero. Primary five-block error is 2.94% lower than persistence, but only 0.20% lower than Framewise. Early one-block differences have unadjusted intervals below zero; they are small and do not establish a broad advantage. All selected checkpoints come from validation, not test. Late overfitting and low sensitivity to reversed command order require explicit discussion.

## Separate validation development findings

The completed diagnostic reads training and validation recordings only. Substituting future commands from another recording session increases ours' validation h10 error by 5.17%, while the action-free control does not change. Reversal alone therefore understates action sensitivity. These perturbations are not physically executed counterfactuals.

All 12 frozen best models then received the same least-squares residual calibration, fitted on all 830 horizon-eligible training episodes and 7,721 windows. The remaining short training recordings remain in the audited manifest. All 851 train and 143 validation payloads, source files, protocol and checkpoints were verified before and after execution. No original test payload was opened.

| Validation comparison | h5 | h10 |
|---|---:|---:|
| Ours, before calibration | 0.158398 | 0.219507 |
| Ours, calibrated | 0.157704 | 0.215874 |
| Framewise, also calibrated | 0.158942 | 0.216206 |
| Relative reduction versus calibrated Framewise | +0.779% | +0.154% |
| Paired validation MSE-difference interval | [-0.002173, -0.000470] | [-0.001395, +0.001022] |

These are exploratory validation intervals, unadjusted for multiple comparisons; h10 is inconclusive. The original held-out table above remains authoritative for the completed test. Calibration is a standard control, not a newly established methodological contribution. Full results: [calibration report](real_droid_residual_calibration_results.md), [machine-readable results](real_droid_residual_calibration_results.json), and [reusable-wrapper model card](model_cards/real_droid_residual_calibration.md). Twelve compact configurations are in `configs/real_video_development/calibrations`; both original-location and portable-release relocation checks pass exactly on CPU. The paper source now includes a clearly labeled development paragraph, formula and generated table; the integrated table (Table 20 on page 45) was compiled and visually reviewed without overlap or overflow. The standalone Overleaf upload also compiled with exact PDF-text parity.

## Fresh-session confirmatory results

After the validation study, a separate acquisition registry selected twelve additional shards before download. Identity-only auditing excluded all original sessions across train/validation/test: 491 of 556 source episodes were excluded for session overlap, leaving 65 episodes from 52 new site/date sessions, without resampling. Episode and serialized-record overlap are zero. This establishes session separation, not scene/object separation.

A new method/evaluation freeze was recorded **before decoding fresh images**, pinning all twelve selected checkpoints and train-fitted scalars, original DINO encoder, original normalization, exact support/action window contract and analysis. CPU job 200168 completed decoding in 30 seconds; GPU job 200169 completed caching and all 96 evaluations with exit code 0 in 86 seconds. Five-block evaluation uses all 65 episodes (866 windows); ten blocks use 64 (801 windows), with the short episode retained in the manifest.

| Fresh comparison: calibrated ours vs calibrated Framewise | Ours MSE | Framewise MSE | Error reduction | Paired 95% MSE-difference CI |
|---|---:|---:|---:|---|
| Camera 1, 5 blocks — single primary | 0.147654 | 0.148757 | +0.742% | [-0.002145, -0.000290] |
| Camera 1, 10 blocks — secondary | 0.206815 | 0.207555 | +0.356% | [-0.002391, +0.000907] |
| Camera 2, 5 blocks — secondary | 0.184255 | 0.185222 | +0.522% | [-0.001971, -0.000174] |
| Camera 2, 10 blocks — secondary | 0.219325 | 0.219935 | +0.277% | [-0.002464, +0.001283] |

The single primary result supports a **modest** fresh-session improvement. Both calibrated h10 intervals cross zero. The original uncalibrated model's h10 regressions remain reported: -1.075% relative reduction on camera 1 and -1.063% on camera 2, both with intervals including zero. All original and calibrated methods, support-only baselines, seed values, action-reversal diagnostics and secondary intervals remain in the [full fresh report](real_droid_fresh_evaluation_results.md) and [JSON](real_droid_fresh_evaluation_results.json). Calibration remains a standard control, not methodological novelty; secondary intervals are descriptive and unadjusted for multiplicity.

The [independent verification](real_droid_fresh_evaluation_verification.json) rechecked every result hash and frozen dependency and independently recomputed endpoint arithmetic from all episode records. Evaluation freeze SHA256: `5274eebfdbe441a0ef15a50e277cd0538972333d1b992e18cef9fe2319c0bf8d`. Result SHA256: `d1d39484ffc35bf0f4a5a4d8f9cc5058e5f0f4ff54964bc486103567a7a255b5`. The paper source includes `sections/fresh_real_video_study.tex` and two source-checked generated tables; integrated compilation and visual review passed. Figure 20 (page 47) shows the exact correction, recorded inputs, matched errors, and all 65 episode outcomes; tables 21/22 and adjacent pages were also reviewed.

## Local artifacts and verification

- [Results](real_droid_results.md) and [machine-readable ledger](real_droid_results.json): all 12 runs and 48 evaluations validated.
- [Reusable release](../artifacts/releases/real_droid_v1/README.md): 12 validation-selected latent predictors, one shared encoder, 330,577,554 inventoried bytes, and 207 hashed files. All 12 isolated offline CPU prediction checks passed with exact parity. Original run folders retain optimizer/RNG states.
- [Simulator extension report](completed_extension_results.md): all 42 runs audited; the official AdaJEPA reproduction is reported separately because its inputs/checkpoints/budget differ.
- [Current paper](../paper/world_model_draft.pdf), real-data tables and source hashes are generated locally. The real-video comparison is integrated as Figure19 on page43; real-data tables18/19 appear on page44. Compiled pages were visually inspected.
- [Actual DROID footage](../data/real_video/droid_selected/processed/preview/recorded_three_camera_episode.mp4): playback assembled from recorded frames; display rate is not calibrated physical time.

## Important decisions and remaining work

- Keep original test outcomes and post-hoc development studies separate; preserve failures, uncertainty and method naming with `(ours)`.
- Frozen real-data training, feature, evaluation, configuration and protocol sources were not changed during the campaign.
- Real videos establish forecasting evidence on recorded observations. They do not establish physical closed-loop robot success, patient benefit, identified dynamics factors or state of the art.
- The next scientific priority is stronger generalization and action-dependent prediction. Use training/validation diagnostics to develop a revision and reserve fresh held-out sessions for any confirmatory claim; do not tune to this completed test.
- The fresh-session confirmation is complete and its results are now revealed. Preserve this frozen evaluation; do not tune further models or subgroups on it or reuse it as an untouched confirmatory population. Separately registered generalization development jobs 200170/200173 use original training/validation only and were registered before these outcomes were seen.
- Simulator extension final tests, matched independent published-method comparisons, and additional real domains remain unfinished. Open-H is only an audited physical-phantom sample; Cholec80/SWoMo/navigation sources were researched, not trained in this study.
- [GitHub](https://github.com/aj-das-research/WM-ICLR), [the live project page](https://aj-das-research.github.io/WM-ICLR/) and [Overleaf](https://www.overleaf.com/project/6aadbb24b37acd9be4eed157) are published. The calibration and fresh-confirmation appendices are synchronized; Figure 20 is synchronized. The generalization appendix, Figure 21 and support-reliability appendix are integrated, reviewed and published; later operational reports may await the next sync cycle. Actual independent edits made through both Git remotes were imported into the workspace and republished successfully; see `reports/evidence/publishing_sync_roundtrip.json` and `docs/PUBLISHING.md`. The five-minute user timer is enabled; remote source edits merge into the workspace and conflicting changes stop publication. Timer logs are private and readable at `~/.local/share/shiftwm/sync/timer.log`.
- [Reusable real-DROID checkpoints](https://github.com/aj-das-research/WM-ICLR/releases/tag/real-droid-v1) are public: the 245.7 MB archive contains 12 predictors, their encoder and 12 calibration wrappers. All six release assets passed anonymous download/hash checks, and all 24 model configurations passed exact offline parity. The generalization prerelease adds 36 verified predictors and the [simulator development prerelease](https://github.com/aj-das-research/WM-ICLR/releases/tag/simulator-development-v1) adds 42, and the ten-step prerelease adds 12, for **102 public trained predictors**. All42 simulator packages passed exact cached-feature and image-input source parity and all six assets passed anonymous checksum verification.
- The interactive project page is published and browser-verified: six matched simulation examples, complete result tabs, fresh-test evidence, checkpoint downloads and actual embedded CPU inference. The inference tunnel is temporary; static examples/results remain available if the server endpoint is offline. Credentials stay in private stores outside the repository. The user handles conference submission. Latest verified figure-skill revision is `436ea49b7c677210ed67cf1c44624db6ff6a3068`.

## Historical snapshot

The [previous manuscript status](history/manuscript_status_before_real_droid_completion.md) is preserved verbatim. Its in-progress job counts and page references are historical, not current.

## Completed generalization development

All 36 registered models completed 30 epochs: slower learning rate, stronger weight decay and smaller predictor capacity, each with four modes and three seeds. CPU finalizer `200177` validated all 72 evaluations and independently reloaded all 36 selected packages with exact prediction parity. Only original training/validation recordings were used. The completed fresh-session test remains unchanged.

Ours improves against matched Framewise in four of six arm/horizon comparisons: Slow h5 +0.289%, Slow h10 +0.323%, Decay h5 +0.425%, Compact h5 +0.476%. Their exploratory paired intervals exclude zero, without correction for multiple comparisons. Decay h10 (-1.196%) and Compact h10 (-0.589%) retain unfavorable point estimates and intervals crossing zero. These are modest development gains, not new confirmatory or state-of-the-art claims.

All results appear in Tables 23–26 and the reviewed Figure 21 (page 51 of the 54-page snapshot). Job `200187` published all 36 models in the [development release](https://github.com/aj-das-research/WM-ICLR/releases/tag/generalization-v1); the 443,302,514-byte archive and its five companion assets passed public checksum verification. Together with the original release, these provide 48 real-video predictors; a separate simulator prerelease now adds 42 and the ten-step prerelease adds 12, for 102 public predictors overall. Sources and measured outputs are preserved in [the full report](real_droid_generalization_results.md).

## Next registered revisions

See [current workstream summary](research_progress_2026-09-19.md) for the h10 control, causal reliability experiment, spatial-token architecture revision and additional recorded IWS benchmark preparation. These are development studies; no new result or novelty is presumed.

## Completed causal reliability development study

The full original-train/validation study finished on CPU (job 200201, 95.57 seconds). All eleven arms and three seeds use the same thirteen-frame observed prefix and ten-frame query: 132 eligible validation episodes / 57 sessions. The shrunk reliability gate gains only 0.0075% at h5 over its global prior, with a paired interval crossing zero and mixed seed effects. It fails the registered promotion rule; ordinary averaging performs better. Three reusable gate packages passed exact offline reload checks, with existing donors and no new neural weights. Appendix R / Table 27 were compiled and visually reviewed in the 54-page snapshot; source hashes bind that review.


## Completed ten-step training control

All twelve models completed 30 epochs and sixty evaluations; all selected epoch 1, preserving the overfitting diagnosis. Twenty paired contrasts retain 19 favorable and one unfavorable point estimate; 14 intervals favor the first method and six include zero. Against equally h10-trained Framewise, ShiftWM gains 0.405% at the h10 endpoint and 0.252% on the all-ten mean, with nominal paired intervals excluding zero. Against its own h5-trained version on identical h10 windows, gains are 2.772% and 1.058%, respectively. The matched h5-prefix mean slightly worsens (−0.116%; interval crosses zero). These exploratory development comparisons are unadjusted for multiple comparisons and do not replace held-out results. See [all results](real_droid_horizon 10_results.md). Paper builder 200206 generated the complete appendix. Independent review recomputed all 20 intervals and inspected all 32 table rows and actual Tables 28–33; numeric and visual checks passed (see `reports/evidence/horizon 10_actual_paper_review.json`). A float barrier was added before the appendix to keep the preceding reliability table in its own section.

All twelve h10 predictors are [publicly released](https://github.com/aj-das-research/WM-ICLR/releases/tag/horizon 10-development-v1); all sixty source ledgers, twenty contrasts, twelve relocated parity checks and six anonymous public asset downloads passed.

The reviewed horizon-matching summary is now **Figure 22, page 56 of the 58-page snapshot**. It distinguishes within-method horizon matching from equally trained method comparisons, displays paired intervals and green positive point gains, and keeps the unfavorable matched five-step mean explicit. Root inspected the actual integrated page and neighboring artifact page; the full build has no overflow or reference warnings. Evidence: `reports/evidence/horizon 10_integrated_figure_review.json`.

## Completed spatial study — 19 September, 06:42 UTC status

All fifteen spatial models completed 30 epochs. Tables 34–38 were independently inspected on the actual pages; all 16 paired intervals were independently reproduced from the raw window and episode ledgers. Native ten-step gains are 5.30% vs autoregression, 3.55% vs anchoring and 4.50% vs the action-free control. Context ablations are inconclusive. The original-coordinate gain vs anchoring is only 0.47% with an interval crossing zero. All outcomes remain reported. A new interpretation subsection makes these limits explicit.

All fifteen models are publicly released, bringing the total to **117 predictors across five releases**. Their 217 archive payload hashes, selected-state bindings and exact offline parity were independently checked. See `reports/evidence/spatial_completed_independent_audit.md`. Earlier counts and page references above are historical snapshots.

No GPU jobs were running at 06:42 UTC. Six full component runs (bounded additive and unbounded transport, three seeds each) are being prepared under a separate protocol and independent prelaunch review. Spatial qualitative figures are undergoing actual-pixel review; their representative median episode has a slightly negative displayed first window, which must remain visible. A clearer DROID input → prediction → withheld reference explainer is being prepared. Focused IWS PushT/box/rope study preparation is separate from any official RLA-WM reproduction.

### Later update: component jobs and integrated figures

Jobs 200443 and 200446 are now training on two RTX5000 Ada GPUs, with four sequential runs queued and CPU finalizer 200449 after both lanes. The registered follow-up passed 83 tests plus independent review. Figures 23–25 are integrated on pages 60–62 of the 64-page draft, with root actual-pixel review and no overflow/reference warnings. The task and gallery also appear on the project page. The exact reviewed figure-reproduction pack is included under `paper/figure_sources/spatial_qualitative`; its single NPZ is allowed only at the pinned path and SHA256.
