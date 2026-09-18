# Manuscript organization revision

This editorial pass answers the request for a clearer main narrative, improved
comparison presentation, and explicit attribution of the proposed method.
It changes manuscript organization and display names, not trained models,
evaluation protocols, checkpoint selection, or measured outcomes.

## Main paper

1. Introduce ShiftWM and the observation/dynamics composition question.
2. Explain the relation to existing work without calling it reproduced evidence.
3. State information access and matched-supervision caveats before the method.
4. Present context inference, prediction, objective, and planning with the architecture diagram.
5. Define tasks, splits, methods, and primary outcomes; include a compact capability table.
6. Lead results with the existing main planning table, then the forecast plot and interpretation.
7. Close with the actual evidential limits and local artifact status.

The main table already existed. Its revised generator keeps the five aligned
methods in the primary comparison and moves extended metrics plus the unaligned
diagnostic to the appendix. ShiftWM (ours) is the canonical display name for
the paired two-context method. The learned controls are explicitly identified
as matched in-house implementations on the same LeWM backbone. Published
related work and planned adaptation baselines are not represented as completed
experiments.

## Appendix organization

Exact experimental protocol; implementation/training and parameter accounting;
information access; statistics; development comparison and budget history;
mechanism controls; goal calibration and elementary cost bound; released-evaluator
check; full results and the unaligned diagnostic; paired planning intervals;
development-only dynamics revision; artifact/visual provenance and complete
training/evaluation ledgers.

The edit moves operational training status, pending-control placeholders, exact
loss-index and optimizer details, and post-hoc diagnostics out of the primary
argument while retaining their disclosures. Main text shrank from approximately
4,200 to 2,400 whitespace-delimited words (including LaTeX/captions); final page
count depends on the rebuilt tables and figures. No font-size reduction or
negative-spacing compression was introduced to achieve this change.

## Checks and integration

The main primary table retains `tab:measured-primary` with `tab:primary` as an
alias. The capability table uses `tab:methods`; full metrics use
`tab:measured-full`. Source includes the new paired-planning appendix figure
conditionally. Main LaTeX and this guide were reviewed for naming, table scope,
missing evidence, index consistency, and separation of test and development.
Compilation and rendered-page inspection were performed after the parallel table
and plot changes were integrated; the final review is recorded below.

## Integrated layout review and repairs

The first integrated build produced 19 pages without overflow or unresolved
references. Independent pixel review covered pages 1, 3, 5, 6, 7, and 8. The
main comparison table and capability table were readable and correctly scoped.
The review found a title word split, a forecast float interrupting the
conclusion, and appendix floats crossing section boundaries. The source now
breaks the title at a phrase boundary, places a float barrier before the main
limitations/conclusion and each appendix section, and starts the appendix on
a fresh page after the bibliography. These changes retain the original fonts
and margins. The second build and rendered-page review confirmed these repairs.

## Previous organization-pass review (historical)

The rebuilt manuscript has **21 pages: main text 1–7, references 8, and
appendices 9–21**. No overflow or unresolved-reference warnings were reported
by the final LaTeX build. Independent pixel review of the final proofs covered
pages 1, 7, 8, 9, 13, 14, 18, and 19, following the earlier review of main tables
on pages 5–6.

- Page 1: the title now occupies two balanced lines without splitting Dynamics.
- Page 7: the forecast figure and caption precede the complete limitations and conclusion section.
- Pages 8–9: references occupy their own page; appendix A begins on a clean page.
- Pages 13–14: the development comparison is introduced by its own appendix section; its table no longer precedes that heading.
- Pages 18–19: the full metric ledger remains with appendix I; paired Table 9 and Figure 7 appear after appendix J is introduced and before appendix K starts.

The paired-planning plot was also reviewed independently at standalone size:
pending comparisons have text but no zero markers; all four available confidence
intervals cross zero; labels, zero lines, points, and interval ends are clear.
The plot supports no established planning advantage. The forecast caption now
explicitly states logarithmic axes and distinguishes training-seed standard
deviations from confidence intervals. No source or figure repair remained
necessary after the second review. This is a layout/evidence-scope review, not a
claim that missing scientific controls are complete.

Reviewed PDF SHA256: `8bda07f401d6556387b5a3a6b8085ea0b60cd37504d4b3f3fc20debfbf8c1d77`. Final page proofs are in
`build/editorial_review/final_pages/`.

## Figure 2 and visible-gain revision

A subsequent revision redesigns the method figure while preserving the earlier
main/appendix organization. The results narrative now highlights the fixed
Framewise-reference held-out forecast gains (26.78% Reacher, 3.68% PushT) and
the closest aligned Unpaired-context comparison (9.21%, 2.30%). Values were
independently recomputed from full-precision three-seed means in the forecast
ledger. The compact generated gain block defines relative error reduction
explicitly, including negative reductions for regressions; the plot caption
separately defines signed error change, whose favorable direction is negative.
The main planning table gains a source-linked paired-difference row with its
confidence intervals; the main text continues to state that the completed
intervals include zero. No new performance experiment, selected checkpoint,
or evaluation protocol is introduced by this presentation change.

The Figure 2 caption now matches the redesigned shared-calibration and
right-to-left planning flow, explicitly labeling feature amplitudes, candidate
futures, and cost bars as schematic. Independent standalone inspection covered
both paper-width and enlarged proofs: goal/history sharing, history-only
observation-context inference, corrected-transition dynamics inference, and
the first-block feedback loop are coherent, with no visible broken connectors
or collisions. The new rendering and gain layout were then checked in an integrated build,
with the final review below. The earlier PDF hash and page counts above refer
to the preceding revision.

The gain reporter was independently reviewed against the full-precision ledgers.
All eight forecast reductions agree with the ratio-of-means definition; both
negative Reacher extrapolation entries are retained. All four planning changes
agree with raw success differences in percentage points, and every reported
paired interval includes zero. The closest-comparator description is limited
to held-out forecasts; extrapolation tables use named fixed references without
a best-baseline claim. The reporter reuses the established forecast and paired
source validators and rejects incomplete three-seed inputs.

## Historical Figure 2 and gain-presentation review

The reviewed final snapshot has **21 pages: main text on pages 1–8, references
on page 8, and appendices on pages 9–21**. The concluding paragraph and AI
assistance disclosure share page 8 with the references. The build reports no
overflow or unresolved-reference warnings.

Independent integrated pixel review covered Figure 2 on page 3, the primary
comparison and compact forecast-gain block on page 6, the gain-annotated
forecast on page 7, the main/reference/appendix transitions, and the paired
planning/dynamics-revision transition. An intermediate 22-page build left
avoidable whitespace after Figure 4: the strict float barrier was forcing a
page break while the figure remained deferred. Loading `float` and setting
only Figure 4 to `[H]` keeps it in reading order without changing its size,
fonts, margins, or the other section barriers. Final pages 7, 8, and 9 were
inspected again after that repair; the text now flows naturally and the
bibliography fits on one page.

Figure 2 has continuous visible connectors, a shared history/goal calibration
path, separate dynamics inference, and a traceable candidate-selection loop.
Its caption distinguishes observed inputs from the illustrative feature
amplitudes, candidate action glyphs, latent futures, and costs. Main table
planning differences retain their conditional intervals; the forecast gain
block identifies both references and displays every negative reduction. The
forecast plot's signed changes retain the visible Reacher extrapolation
regression. No new scientific or visual defect remained after the final
review; incomplete experiments and lack of an established planning advantage
remain stated.

Reviewed PDF SHA256: `9fa36dbded42bbbea7661afb02e4fa7bb11bcf04006a31814733d3d75c50f094`. Proofs:
`build/figure_gain_review/final/`. This hash identifies the reviewed snapshot
before the automatic watcher resumes; later automatic rebuilds may change PDF
timestamps and hashes.

## Mechanism-focused Figure 2 revision

A further Figure 2 pass makes the proposed mechanism visible as three linked
components: shared observation calibration (A), dynamics inference from
corrected transitions and executed actions (B), and training-only paired
context supervision (C). The diagram contract was independently checked
against `model.py`, `data.py`, and `train.py`. In particular, the dynamics
penalty uses the same physical trajectory/actions under different allowed
appearances; the observation penalty uses distinct batch indices with the
same appearance, without requiring different dynamics or trajectories.
Canonical query targets and factor IDs are not deployment inputs.

The caption distinguishes frozen visual encoding from offline training of
contexts, adapters, and reused prediction modules. It describes representative
training frames as support-clip illustrations and dashed links as penalties,
not achieved context equality or physical identification. The previous
worked-example caption's cost-bar description no longer applies and was
removed. Standalone paper-width and zoomed pairing/goal-action proofs were
inspected: the shared executed-action input, encoder labels, independent goal
route, and training/deployment boundary are clear, with no broken connectors
or unintended overlaps. The integrated pagination review is recorded below;
the hashes above identify historical reviewed snapshots.

## Control-policy transparency addition

The full-results appendix now includes the validated random-action and
privileged-replay control table, and the main limitations state the lower
PushT planning point estimates versus random actions. The note distinguishes
three-training-seed learned-model means from one fixed random-policy RNG
rule and records both environments, both populations, and support exclusion.
Replay is explicitly a privileged recorded-action reachability check, not
a fair practical baseline or proof that every learned planning operation is
correct. This addition exposes previously completed control evidence; it does
not introduce a new experiment or a significance claim.

## Integrated paired-supervision and control review

The latest integrated edition has **22 pages: main text on pages 1–8,
references on page 8, and appendices on pages 9–22**. Independent pixel review
covered pages 4, 6–9, and 18–20 in `build/pairing_control_review/`. Figure 2
now occupies page 4: its A/B/C mechanism, training-only penalty panel, and
full caption fit without clipping, broken connectors, or text collisions.
The frozen encoder, offline learned modules, shared calibration, and paired
loss roles remain distinguished; schematic amplitudes and latent branches
are not presented as measured recovery.

The main comparison and forecast-gain block on page 6 remain readable. Both
negative Reacher extrapolation reductions and all zero-crossing planning
intervals are retained. Figure 4 and limitations flow across pages 7–8
without the former avoidable blank region. The adverse PushT comparison
against random actions is visible in the main limitations and explained in
Appendix I.1 on page 18; its reference resolves to control Table 9 on page 19.
The note distinguishes fixed-RNG random actions from trained-seed means and
privileged replay from fair learned-policy inference. Table 9 remains before
Appendix J, and paired Table 10/Figure 7 precede Appendix K on page 20. The
bibliography ends on page 8 and the appendix begins cleanly on page 9.

No new layout or evidence-scope defect was found. The control-caption wording
was separately corrected from spending 50 steps to sharing a 50-step native
budget, since early-success episodes can stop sooner. Independent inspection
of corrected page 19 confirms that this wording fits cleanly and preserves
the control/paired-comparison section order. The final caption-only build
remains 22 pages and reports no LaTeX warnings.

Reviewed snapshot: `build/pairing_control_review/final/reviewed_draft.pdf`.
SHA256: `7a78c252db8d55c43179077cd31dd38bd7a9570db91e80dfdea1833acdcb3c11`.
The corrected page proof is `build/pairing_control_review/final/page-19.png`;
other reviewed page proofs are in the parent review directory. This record
identifies the snapshot before the watcher restarts; later automatic rebuilds
may change PDF timestamps, hashes, and completion statuses.
