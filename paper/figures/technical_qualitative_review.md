# Technical qualitative delivery review

Reviewed 2026-09-19 (Dubai). The final manuscript contains 35 pages;
references start on page 8, so the main text remains within the nine-page
budget. The integrated technical figures are Figure 8 on page 18 (inference
contract), Figure 9 on page 19 (PushT), and Figure 10 on page 20 (Reacher).
The inspected PDF SHA-256 is
`160c2807556f28baf29f3e3e47af7de58097907b8de75b5e23f9b91f5b62f7c9`.

## Pixel review

Opened actual 130-ppi integrated proofs at
`paper/build/technical_qualitative_review/final-page-18.png`,
`final-page-19.png`, and `final-page-20.png`, as well as the two standalone
case-study proofs. The prior inference-contract review also examined paper-size
and enlarged connector crops. Figure labels, units, tolerances, actual stop
times, goal outlines, percent changes, and captions are visible without
clipping or overlap. Connectors remain continuous and enter the intended ports.
Both success and failure endpoints remain visible. Reacher uses one common
crop across both methods, every shown time, and the goal; there is no
per-method scale change. The PushT filmstrip and caption now explicitly state
that the green T is an unscored renderer marker.

The first integrated build left two summary lines on an otherwise empty page.
Compacting the repetitive inference-caption wording removed this orphan page
without reducing figure size or omitting a scientific qualification. The final
three integrated pages were then reopened and checked.

## Independent evidence checks

`paper/build/technical_qualitative_review/numeric_audit.json` records the
read-only recheck of 39 exact replay traces, 1,670 native states, and all
70 paired transitions / 140 saved prediction vectors. Replay NPZ hashes match
the report. Strict success flags agree with both physical criteria at every
native time; each prediction MSE was recomputed from its saved feature vector
and canonical target. No generated frames or interpolated terminal observations
are used. Partial terminal blocks remain explicitly unmeasured in the forecast
panels.

- PushT episode 2031024: at time 20, combined position errors are 51.7567 px
  (ShiftWM) and 106.4768 px (Framewise), while Framewise has the smaller angle
  error. ShiftWM stops successfully at 21 with 16.2583 px and 0.159493 rad;
  Framewise fails at 50 with 457.0217 px and 1.383574 rad. Both strict limits
  are required: 20 px and pi/9 rad. ShiftWM has higher forecast MSE on all five
  paired transitions in this case.
- Reacher episode 2031004: ShiftWM stops successfully at 23 with joint errors
  0.010717 and 0.046594 rad; Framewise ends at 50 with 0.568675 and 0.056548 rad.
  Both individual unwrapped joint errors must be below 0.05 rad. ShiftWM has
  lower prediction MSE on two of two transitions along its own execution and
  one of three along Framewise execution.
- Across all four outcome-selected positive cases, lower ShiftWM forecast MSE
  occurs on 6/20 paired transitions and lower support-calibration MSE on 0/20.
  The manuscript retains these unfavorable counts and explicitly avoids causal,
  independent-win, significance, or benchmark-superiority interpretations.

The existing qualitative-reporting, positive-qualitative, exact-replay, and
conditional-prediction suites passed: **110 passed**, with seven environment
warnings (optional ALE package unavailable and Gymnasium array casts). No
scientific model, evaluation, training, or frozen campaign source was edited.

## Build and scope

SWoMo was added to related work using the parent agent's verified citation,
and the text explicitly disclaims novelty for appearance/dynamics separation
itself. No surgical experiment is claimed. The recursive-revision paragraph now
matches the six-arm validated snapshot: recursive/inferred ties the PushT donor
and scores 13/30 versus 11/30 on Reacher; it does not claim to beat the matched
constant-input control.

`integrated_build_audit.json` records the final build. Compilation succeeds
without overfull boxes, undefined citations, or undefined references. Existing
document-level warnings remain: eight underfull-box notices and 28 duplicate
PDF destinations for figure/table anchors. These were reported to the parent;
they do not clip these figures, but this review does not certify all PDF links.
The paper watcher (PID 1433930) was resumed after the brief protected build and
verified alive in sleeping/running-capable state.

## Local release bundle

`artifacts/qualitative_gallery.zip` contains 204 files (10,486,905 bytes),
including the complete browser gallery, `technical_comparisons.pdf`, the
inference-contract PDF, positive and technical evidence ledgers, and the full
replay/prediction JSON reports. The README describes selection, negative
diagnostics, original source paths, and the large replay arrays/checkpoints
that remain in the project. All ZIP members passed CRC validation; copied
evidence files were checked byte-for-byte against their authoritative sources.
`bundle_audit.json` records the archive hash and copied-evidence hashes.

## PDF navigation repair follow-up

The duplicate-destination warnings recorded above were subsequently repaired
by loading the existing `float` package before `hyperref` in `paper/main.tex`.
Loading it afterward had replaced hyperref's caption handler, causing two
anchors per ordinary caption. A minimal two-caption reproduction confirmed
the cause. The full build now has **zero duplicate-destination warnings**;
the eight unrelated underfull notices remain.

`paper/evidence/pdf_navigation_repair.json` records the checked build
(`4ba1e656103f11e112121f7ff94b879b9757ca0644db3e5f318aff9f8c3f24a1`).
All 29 figure/table destination names and pages are preserved, including the
uncaptioned, unreferenced longtable destination. All 28 visible caption targets
resolve to their actual pages; 27 label mappings and 52 resolved internal-link
spans were checked. Printed numbering, figure sources, data, and scientific
code are unchanged. Correct caption handling changes some spacing, so exact
pixel identity is not claimed. Actual proofs of pages 6 and 18--20 in
`paper/build/anchor_repair/` were reopened: the primary table and technical
figures remain readable and unclipped. The document remains 35 pages, and the
watcher was resumed after the protected build.
