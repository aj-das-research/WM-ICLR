# Compact world-model manuscript

## Current single-method manuscript

The main paper proposes **ShiftWM (ours): observation-anchored spatial mixing**.
The selected implementation is the frozen `transport` arm (historical Ours-5).
Its component removals are ablations, not additional proposed algorithms.
The earlier two-context model is a distinct historical study, not an ablation
of the spatial predictor or evidence for its planning performance.

The main and appendix remain **one continuous PDF**, using the official
5.5-inch ICLR column without changing body fonts, margins or line spacing.
Four main figures present the idea, detailed computation, measured comparisons
and visual examples. Complementary panels sit side by side where legible.

| Main display | Reproducible source | Purpose |
|---|---|---|
| Observation anchoring teaser | `scripts/render_teaser_camera_story.py` | Rounded camera and three recorded frames introduce a retained visual reference and early/later feature forecasts; eight further task inputs and complete-population DROID evidence stay separate |
| Detailed spatial architecture | `scripts/render_architecture_visual_design.py` | Fixed observed memory and causal conditioning feed a numbered patch-mixing example, complementary gate and bounded correction; withheld targets have a separate scoring path |
| Spatial evidence | `scripts/render_results_refined.py` | Forecast curves beside horizontal paired confidence intervals against named controls |
| Qualitative comparison | `scripts/render_qualitative_compact.py` | Three aligned case rows with recorded frames, common-scale error maps and signed error-gap curves |
| Main comparison and ablations | `scripts/render_editorial_tables.py` | All eight spatial endpoint rows, with one named proposed model |
| Appendix decoder diagnostics | `scripts/render_mixing_story.py` | Aligned measured mixing comparisons with gate traces for the same three cases |

The former family overview and duplicate spatial architecture are archived as
source assets rather than repeated in the PDF. The earlier complete context
architecture (formerly Figure 13) is also archived; its historical equations,
protocol and full numerical comparisons remain in the appendix. Its operations
are not imported into the current spatial decoder. The main teaser shows nine attributed input examples: simulated PushT, Reacher,
drone and tissue manipulation; recorded DROID and IWS PushT/Box/Rope; and an
Open-H physical-phantom input illustration. Group labels distinguish the
historical simulation models, ongoing IWS training and Open-H ingestion scope.
Only the DROID example connects to the current spatial-decoder comparison,
with source-bound evidence across all 141 development episodes. The viewfinder,
feature glyphs, source-patch pieces and mixing weights are illustrative. All three
prespecified visual cases now appear in the main
paper; the appendix adds unique measured source weights and gate traces.

Main sources are `sections/main_*.tex`; grouped appendix sources are in
`sections/appendix/`. Detailed methods, every paired contrast, historical
protocols and the complete qualitative replay are retained. Native bounding
increments are inconclusive, and the unbounded ablation has the slightly better
five-step mean; neither fact is concealed by selecting the bounded ten-step arm.
Fresh held-out confirmation of this spatial design remains pending.

Regenerate the displays with `.venv/bin/python paper/scripts/<renderer>.py`,
then run `bash paper/build.sh`. Numerical figures check completed source ledgers.
The revised introduction and architecture keep portable inputs and provenance
in `figure_sources/`; proofs and design decisions remain in `design/`.
The teaser and architecture are respectively 5.5 by 3.0 and 5.5 by 3.67 inches,
with at least 8-point labels. The personal `paper-visual-design` skill guided
three composition alternatives per figure and the source-based reviews. A
generated camera/stream and architecture references informed their compositions; scientific wiring,
equations and results use editable vector objects. The compact camera is an original vector illustration; three unchanged
DROID frames form the observed stream. The inspected web camera reference has
its original license retained in `figure_sources/teaser_camera_story/`. Native `.drawio`
files accompany both figures: `generated/editorial/teaser_camera_story.drawio`
and `figure_sources/architecture_visual_design/architecture-visual-design.drawio`.
The Python scene is the canonical source for regeneration; native application
exports are reviewed separately from the Matplotlib paper exports.
The measured decoder diagnostic aligns case rows beside a shared gate plot.
The former standalone task diagram (previously Figure 6) is omitted from the
manuscript because Figures 2 and 4 cover its input/evaluation path and error
example. Its editable sources and project-page tutorial remain available;
the precise frame/action timing is retained in the appendix text.
`design/refined_style.json` retains the quantitative display palette. Both use a
5.5-inch width and regular 8 pt labels. The earlier
evidence style remains archived.
`generated/editorial/` contains editable vector outputs, captions and evidence.
`evidence/manuscript_sources.json` pins the manuscript sources for each build.

Build with `bash paper/build.sh` from the project root. Outputs:

- `world_model_draft.pdf`: current world-model manuscript.
- `proposal.pdf`: same current PDF, preserving the earlier link.
- `generated/world_teaser.pdf`, `.svg`, `.png`: illustrative task scene alongside source-linked forecast comparisons; editable source is `scripts/render_teaser.py`.
- `figures/method.pdf`, `.svg`, `.png`: architecture exports; canonical editable source is `figures/world_method.tex`.
- `figures/split.pdf`, `.svg`, `.png`: real task scenes, matched appearance changes, schematic physical responses, and the factor split; editable source is `figures/factor_split.tex`.
- `generated/forecast_comparison.pdf`, `.svg`, `.png`: completed forecast comparisons, regenerated by `scripts/render_forecast.py`.
- `generated/training_curves.*` and `generated/goal_calibration.*`: logged validation curves and development calibration diagnostics.

The previous TTA manuscript is preserved in `archive_tta/`. The official ICLR2027 style package remains unmodified in `template/` with download provenance. Document-level overrides label the PDF as a research draft and remove the template's under-review status; no submission is implied.

## Historical experiments and artifact status

All 30 original configured training runs and the three-seed planning campaign are complete. Forecasts lower held-out-composition error for the historical context model versus Framewise calibration in both environments, but Reacher extrapolation worsens. Planning gains are mixed, and every paired primary interval includes zero. The completed eight-arm rollout/context follow-up fails its registered promotion criterion. All 36 additional simulator models and six observation-gain ablations have also completed their registered development evaluations; their positive and negative results remain separate from the original test campaign. Only completed, source-validated evidence populates the tables. No acceptance, universal transfer, or planning superiority is claimed.

The additional real-video study uses 1,126 audited DROID recordings, with session-disjoint splits and a second-camera evaluation. Its [protocol](../reports/real_droid_protocol.md), [progress](../reports/real_video_progress.md), and generated results distinguish offline real-video forecasting from physical robot control. Its portable release and tables were verified against all 12 complete runs, 48 evaluations, and exact offline package reloads. Later spatial development and component studies have separate protocols and releases.

The model implementation is in `../src/shiftwm/`. Dataset manifests and completed evaluation records are authoritative; reconcile the manuscript whenever a configuration or benchmark changes. An environment-family checkpoint is not a universal robot or medical model.

## Method names and historical controls

In the current main paper, **ShiftWM (ours)** is the bounded spatial-mixing
predictor. Names such as “No mixing,” “No bounding,” “No support context,”
and “No actions” denote component removals. The additive-anchor arm removes
both mixing and bounding. Internal checkpoint IDs remain unchanged.

Historical Ours-2/3/4 map to additive, bounded-additive and unbounded-mixing
controls; Ours-5 maps to the proposed spatial model. Ours-1 is the earlier
paired-context model. These IDs are retained in the appendix for reproducibility,
not as a list of separately proposed algorithms or a performance ranking.

The original context study retains its own method names and primary planning
comparison (`generated/primary_results.tex`, `tab:measured-primary` / `tab:primary`):

- **Original paired-context model:** separate observation and dynamics contexts with paired consistency.
- **Unpaired contexts:** the same two-context architecture without consistency losses.
- **Shared context:** one inferred context, with matched context-network size.
- **Framewise calibration:** learned per-image correction without temporal context.
- **Frozen LeWM:** the released pretrained reference without additional training.
- **Unaligned predictor (diagnostic):** continued prediction training without learned visual calibration.

Visible gains are always tied to a named reference. The primary planning table
includes paired percentage-point differences versus Shared context with 95%
task-cluster intervals, not only favorable point estimates. The compact forecast
summary reports relative MSE reduction against Framewise calibration and the
closest aligned held-out control, Unpaired contexts. For this summary, positive
means lower error; plot annotations instead use signed error change, where
negative means lower error. Both retain the Reacher extrapolation regression.
The strongest completed Framewise-reference result is 26.78% lower held-out
Reacher forecast MSE, while the closest aligned Unpaired comparison is 9.21%.
Neither establishes an advantage in closed-loop planning.

In comparison tables, bold dark-green numbers mark positive mean gains against
the explicitly named reference. Negative differences, rounded zeros, pending
cells, and uncertainty intervals stay neutral; green does not mean statistical
significance. The style is generated by the reporting scripts and persists when
new results arrive.

The three learned comparison methods are matched in-house controls on LeWM,
not reproductions of independent published methods. The separately reported
official AdaJEPA reproduction uses different inputs, checkpoints and planning
budgets and cannot rank against ShiftWM directly. ContextWM, ICWM, MoVie and
SG-JEPA remain cited, unreproduced related work.

Appendices retain exact data and planner settings, dimensions and parameter
accounting, temporal indexing and full losses, optimization and checkpoints,
information-access checks, statistical protocol, development-budget history,
seed-zero development results, pending mechanism controls, the goal diagnostic
and elementary cost bound, released-evaluator checks, full condition and
unaligned diagnostic results, paired intervals, the separate dynamics revision,
and artifact provenance. The live training ledger belongs in the appendix.
See `editorial_revision.md` for this revision's organization and review scope.

## Historical simulation reporting commands

Use the project Python environment and explicitly name completed evaluation JSON files:

```bash
.venv/bin/python paper/scripts/render_results.py results/world/pusht_factorized_s0/planning_test.json
bash paper/build.sh
```

Replace the example filename with an existing completed run. The renderer refuses incomplete records and missing provenance. It writes `generated/result_ledger.json`, a CSV, supplementary LaTeX rows, and per-run PDF/SVG/PNG planning plots. It preserves source SHA256 and evaluator intervals, never merges training seeds, and never converts missing measurements to zero. The optional generated appendix is included automatically. The original simulation comparison table and method-capability table are generated by `scripts/aggregate_results.py`, together with `generated/primary_results_appendix.tex` for canonical, eligible, and diagnostic results. Trained-model cells require all three prescribed seeds, while frozen controls require their one checkpoint.

Trajectory-cluster bootstrap intervals within a run do not measure training-seed variability. Raw all-condition success and success conditional on tasks not already solved during support acquisition are different metrics; retain both. Latency uses actual planning calls rather than assigning zero latency to a task that never replanned.

## Historical context-study figures and skill provenance

The figure-creation skill was applied. See `figures/figure_brief.md` for input/operation/target contracts, alternate layouts considered, scope and provenance; `figures/review.md` records pixel inspection and limitations. The SVG is a vector export with outlined glyphs; the editable text and geometry master is TikZ. The method computation is schematic; observed simulator frames are input illustrations with an asset ledger, not predicted images or evidence of planning success. Numerical plots preserve uncertainty type and comparison scope. See the current per-figure briefs and `figures/visual_refresh_review.md` for the new review.

The latest verified user-requested figure-skill improvements were pushed to [`Mishrakshitij/paper-figure-creation-skill`](https://github.com/Mishrakshitij/paper-figure-creation-skill) on `main` at commit `436ea49b7c677210ed67cf1c44624db6ff6a3068`. They cover evidence-grounded qualitative comparisons, visual assets, connectors, and paper-width/detail/crop inspection. [The publication receipt](../reports/evidence/figure_skill_publication_2026-09-19.json) records the revision, installed hashes and checks; earlier receipts describe earlier skill versions. The research code and manuscript are also published on GitHub, the paper is synchronized with Overleaf, and five model releases and the project page are public. Six newer component predictors remain local. The user handles conference submission.

The earlier visual redesign replaced paragraph boxes with observed frames, feature transformations, candidate branches, and a symbol-based factor grid. Detailed previous sources are preserved as `world_method_detailed.tex` and `factor_split_detailed.tex`; their geometry and asset provenance are described in `method_objectflow_brief.md`, `split_visual_brief.md`, and `teaser_brief.md`. The generated introduction scene is explicitly illustrative. Its prompt and checksum are saved, and paper rebuilds reuse the asset. Quantitative plots are never generated as artwork. The skill now explicitly rejects text-heavy redesigns and requires scientific objects to remain meaningful before labels are read.

The skill also supports web/image search, original asset downloads, official logos/model illustrations, and background removal with before/after edge checks. The method uses a transparent [Material lock icon](https://github.com/google/material-design-icons), with its source and Apache-2.0 license retained in `figures/assets/`. The source was already transparent. `figures/geometry_repair_review.md` records the stricter connector review: continuous unmasked paths, explicit junctions and target ports, corrected first-block execution, and readable labels clear of wires. The reusable skill script `scripts/inspect_figure.py` generates paper-width/detail PNG proofs and recorded zoom crops for visual inspection; it does not certify aesthetic or scientific correctness automatically.

The earlier connector repair, checked against both screenshots in `../figure-issues/`, is retained in `figures/arrow_regeneration_review.md` as a historical review. It repaired arrow tips and shafts, input ports, setup spacing, and the teaser's annotation collision.

The archived historical context architecture foregrounds shared calibration (A), dynamics inference from corrected transitions and executed actions (B), and training-only paired context supervision (C), with reused prediction/planning machinery kept secondary. Its contract is `figures/method_editorial_redesign_brief.md`, and the accompanying review is `figures/method_editorial_redesign_review.md`. The editable master remains `figures/world_method.tex`; paper-width and enlarged proofs are in `build/method_pairing_redesign/proof/`. Real PushT deployment inputs and separate training examples are retained in the archived artwork; the latter illustrate representative support clips and matching rules. Feature amplitudes, candidate action glyphs, and latent branches are schematic. The dashed context links denote regularization penalties, not measured equality or identified physical factors. The diagram is no longer repeated in the current manuscript; its equations and study remain in the attached appendix.

The subsequent planning-flow clarification replaces the ambiguous fan between
LeWM's predictor and CEM with candidate latent rollouts and an explicit terminal
goal-distance calculation. CEM refits and resamples low-cost elite action
sequences; execution uses the first block of the final mean plan. The goal enters
the cost calculation, and the rollout/cost example remains schematic. See
`figures/planning_flow_redesign_brief.md` and
`figures/planning_flow_redesign_review.md` for the current focused repair and
visual checks. The model, scoring objective, and experiments are unchanged.

## Refresh historical simulation figures

```bash
.venv/bin/python paper/scripts/render_training.py
.venv/bin/python scripts/aggregate_results.py
.venv/bin/python scripts/summarize_paired_planning.py
.venv/bin/python paper/scripts/render_planning_comparison.py
.venv/bin/python paper/scripts/render_planning_controls.py
.venv/bin/python paper/scripts/render_forecast.py
.venv/bin/python paper/scripts/render_gain_summary.py
.venv/bin/python paper/scripts/render_teaser.py
.venv/bin/python paper/scripts/render_goal_calibration.py
bash paper/build.sh
```

Matplotlib SVG exports preserve editable text. Diagram SVGs retain editable geometry with outlined glyphs; TikZ remains the canonical editable text source. The method/setup include observed raster images, and the teaser includes a generated raster illustration, so those PDF/SVGs are hybrid exports.

## Historical simulation training refresh

```bash
.venv/bin/python paper/scripts/refresh_training.py
.venv/bin/python paper/scripts/refresh_training.py --watch --interval 120
```

The watcher consumes `configs/world/full_campaign.json` (30 trained runs) and rebuilds after configuration, metrics, summaries, or evaluation files change; it can run for up to five days by default. Completion requires the final summary, all configured epochs, and a final checkpoint. Curves are included only after at least one complete epoch has been logged, with no smoothing or extrapolation. Each refresh invokes `scripts/aggregate_results.py`; the generated main table is included automatically. Unfinished evaluation files do not populate result cells. Inspect the resulting numerical plots before using empirical conclusions.

## Historical random-action and replay controls

`paper/scripts/render_planning_controls.py` validates the eight completed
control files and writes `generated/planning_controls.json` and `.tex` for the
full-results appendix. It reports raw, support, and eligible counts on the
held-out and extrapolation populations. Random actions use one fixed RNG rule
(base seed 1701 plus initial-state seed), so their rates are not estimates over
three training seeds. Replay uses privileged recorded future actions; it checks
goal reachability and is not a practical inference baseline. Neither policy
uses world-model predictions or CEM to choose actions.

The original-study appendix retains the adverse PushT comparison: the historical context model's
three-seed success means are 2.60% held-out and 6.77% extrapolation, below the
random control's 7.81% and 9.38% point estimates. Reacher is also reported, so
the discussion does not select only favorable or unfavorable environments.
These are descriptive comparisons without a matched policy-seed significance
claim. `../reports/results_diagnosis_2026-09-18.md` separates demonstrated
failures from untested explanations; its completion count is a dated snapshot,
not a live scheduler status.

## Planning-budget audit

The original context study's learned-policy planning uses the official 300-candidate, 30-iteration, 30-elite CEM sampling budget, fixed across methods before full-test planning. Historical 128/5/16 development checks are isolated in `evidence/development_budget.json` and described as diagnostics in the manuscript. Recreate that source-linked note with `.venv/bin/python paper/scripts/record_development_budget.py`; the script rejects records with a different budget or split.

Per-run latency rendering requires explicit `execution_context.main_efficiency_claim_eligible=true`, `shared_gpu_with_training=false`, and test/extrapolation planning. Absent or shared execution metadata leaves the efficiency cell missing. Development timings are never main efficiency results.

## Paired planning and the separate dynamics revision

`scripts/summarize_paired_planning.py` validates paired comparisons across all
three training seeds and writes the appendix table. Its trajectory-cluster
intervals are conditional on those trained models; seed variation is reported
separately. Missing paired runs leave numerical contrasts blank. The protocol
is documented in `../reports/paired_planning_protocol.md`.

`paper/scripts/render_dynamics_revision.py` handles the separately trained,
development-only dynamics residual over a frozen framewise model. It verifies
the full 30-epoch record, checkpoint/control/source identities, common support,
and binary outcomes before rendering counts and paired wins/losses. Added
optimization makes this an exploratory revision, not a comparison with matched
training budgets. The original main study remains distinct. Both reporters run
inside `refresh_training.py`; interrupted evaluations remain pending.

## Matched development comparison

`.venv/bin/python paper/scripts/render_official_development.py` writes the separate `generated/official_development.json` evidence ledger and LaTeX table from `results/development_official_budget/*/planning_development.json`. It reads development identities from the two dataset manifests, without reading final-test results. Each completed 32-task row appears immediately; missing or incomplete model/environment records remain dashes. The four prespecified models are Frozen LeWM, Shared context, the original paired-context model (historically ShiftWM (ours)), and Framewise calibration, using the official 300/30/30 budget and trained seed 0.

The renderer reconstructs raw, support, and policy-eligible counts from task records and checks their denominators against the supplied summaries. It verifies the same 32 manifest seed/trajectory identities and, among available completed models, the same goals, support-success masks, eligibility masks, and post-support distances. Mixed budgets, inconsistent signatures, and incompatible evaluator identities are rejected. Development timing is excluded from efficiency claims. This single-seed development table does not change the three-seed completion gate for the main test table or establish a factorization benefit.

The existing refresh watcher now fingerprints completed official-budget development files and invokes this renderer before compilation. Per-episode `.progress.json` journals do not trigger rebuilds; canonical evaluation outputs and training metrics still do. A watcher already running an older version must be stopped and restarted once; do not run duplicate watchers. Validate reporting guards with `.venv/bin/python -m pytest -q paper/tests/test_official_development.py`.
# Matched qualitative examples

`sections/positive_qualitative.tex` and
`scripts/render_positive_qualitative.py` explain all four recorded
original-context-success/Framewise-failure development cases. Registered goal outlines,
actual shared-time frames, common Reacher detail crops and native-action rulers
show the outcome without claiming a causal component effect. Their source
packet is `generated/qualitative/positive_evidence.json`; the gallery leads with
these cases while retaining all tasks and the balanced panels. A standalone
two-page PDF is in `artifacts/qualitative/positive_cases.pdf` at the project root.

`scripts/render_technical_contract.py` explains the original inference paths
without implying shared trained predictor weights or a proven causal benefit.
`scripts/render_technical_qualitative.py` uses exact saved-action replay and
paired same-transition inference diagnostics to show physical goal criteria
and next-observation prediction errors at several decision points. The source
diagnostics are `results/qualitative_diagnostics/{replay,prediction}_diagnostics.json`
at the project root. Reproduce them using the standalone
`scripts/diagnose_qualitative_replay.py` and
`scripts/diagnose_qualitative_prediction.py`; these never change training or
select new control actions. The measured figures are exported to
`artifacts/qualitative/technical_comparisons.pdf` and retain negative comparisons.

`sections/qualitative_rollouts.tex` adds eight observed development examples in
four paired figures, including successes, failures and success during shared
support. Rebuild their editable PDF/SVG/PNG exports with
`python paper/scripts/render_qualitative.py` from the project root. The local
`artifacts/qualitative/index.html` gallery includes all 64 tasks and all three
recorded methods, with every saved frame at its actual native-action time.
The evidence ledger preserves source hashes, complete records and the explicit
outcome-stratified selection rule; these examples do not estimate win rates.

The new rollout-objective/context study is separate from the original campaign.
`paper/scripts/render_rollout_revision.py` fills its appendix table only from
validated completed artifacts; missing outcomes remain pending. The paper
watcher refreshes both reports. Its eight conditions and source hashes are
frozen before execution in `runs/rollout_revision/campaign_state`.

## Domain extension diagnostics

`sections/domain_extensions.tex` records the separate drone/tissue protocol and
a dated preparation/completion snapshot. `scripts/record_extension_diagnostics.py`
checks all three training seeds and eight matched tasks for both drone predictor
families before writing `generated/extension_drone_development.tex` and the
source-linked `evidence/extension_diagnostics.json`. The table reports sample SD
across training seeds, not a confidence interval or 24 independent tasks.
The completion snapshot remains explicitly dated; it is not a live scheduler view.

`figures/goal_geometry.pdf` uses observed drone goal images and all sixteen
development goals. Its hatched bar is a fixed-context analytic gain bound,
not a new result. The accompanying four-state, 128-branch action-ranking probe
retains unsafe branches in its evidence and uses safe equal-horizon branches
for rank/regret calculations. These diagnostics motivate the separately
registered six-run observation-capacity ablation, whose outcomes are not
included in this draft snapshot. Original negative outcomes remain visible.

The local packet `artifacts/domain_extension_diagnostics.zip` (at the project
root) contains the appendix excerpt, editable figure, source-linked numerical
reports, and a review record. It contains no new model-release or clinical claim.
