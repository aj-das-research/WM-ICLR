# Completed spatial study: independent audit

**Passed numerical, rendered-table and public-release verification.** This audit covers the registered 15-model original-validation study, not a new test evaluation.

- All 15 models completed 30 epochs. Checked 280 bound files, all 30 selected/final checkpoint packages, every journal and selected first validation minimum.
- Recomputed all 24,465 window records into 84,600 episode-level metric cells and 600 model-summary cells. The population is 141 episodes, 59 recording sessions and 1,631 windows per model.
- Independently reproduced all 16 confidence intervals using seed/session multiplicity-count contractions, without calling either existing bootstrap implementation. Maximum interval difference: 1.22e-17.
- Of the 16 registered comparisons, 14 have positive point gains and 2 negative point gains. Nine intervals favor transport; seven include zero; none favor the comparator. These are exploratory, unadjusted validation comparisons.

| Comparison at h10 | Relative reduction | Interpretation |
|---|---:|---|
| Native4×4: transport vs autoregression | +5.300% | Paired interval favors transport |
| Native4×4: transport vs anchored additive | +3.547% | Paired interval favors transport |
| Native4×4: transport vs action-free | +4.503% | Paired interval favors transport |
| Native4×4: transport vs context-off | +0.097% | Inconclusive |
| Original2×2: transport vs autoregression | +2.538% | Paired interval favors transport |
| Original2×2: transport vs anchored additive | +0.472% | Inconclusive |
| Original2×2: transport vs context-off | −0.108% | Inconclusive |
| Original2×2: transport vs action-free | +4.085% | Paired interval favors transport |

The package improves native spatial-feature prediction, with a smaller gain over autoregression in the original pooled coordinates. The context module has no established benefit here. The comparatively stronger native-grid effect is consistent with gains in spatial detail, but this is an interpretation, not an isolated causal result. Mixing, bounded innovation, initialization and active capacity remain partly confounded; the proposed component follow-up is justified. The study establishes neither a physical warp nor SOTA performance.

All models completed the full budget. Selected epochs by seed0/1/2: autoregressive16/25/22; anchored additive15/25/16; transport14/19/11; context-off10/17/14; action-free6/5/4.

## Actual manuscript inspection

Viewed the saved manuscript pages57–59, including all five spatial tables34–38 at page size and enlarged. All table values, selected epochs and active parameter counts match the ledgers; 111 numeric tokens were checked against actual PDF text. No overlaps, clipped cells or missing minus signs were found. Both negative context comparisons, all seven inconclusive markers and the action-free pooled-h10 regression remain visible. Captions correctly explain green bold as a positive point estimate rather than significance.

The exact table counts qualify capacity differences. A small wording improvement is to replace “active parameter counts differ slightly” with “active parameter counts differ by arm”: the transport/additive gap is1.84%, while context-off has14.40% fewer active parameters. Page59 has generous float spacing; this is cosmetic.

## Public artifacts

Anonymous full downloads of all six spatial-release assets pass their checksums. All217 archive payloads match the public manifest; the fifteen model files exactly match the selected local checkpoints. The attached physically relocated CPU proof reports exact forecast agreement for all15 and zero network attempts. This audit verifies that proof's binding; it does not claim a second independent inference execution.

- Release: https://github.com/aj-das-research/WM-ICLR/releases/tag/spatial-world-models-v1
- Archive:151,557,890bytes; SHA256 `d84a817a74b03168417e0174052132fbb21364d6768301b67868a5c4b07790a1`.
- Published predictor total: **117** =12 original+36 generalization+42 simulator+12 horizon-ten+15 spatial. Twelve calibration wrappers are separate and are not extra neural models.

The accompanying JSON binds source hashes, checkpoint rows, all paired intervals, downloaded assets, exact PDF and reviewed image hashes. No frozen scientific, manuscript, site or main-branch files were changed.
