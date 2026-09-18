# Matched horizon-ten training control

Status: in_progress; 0/12 full 30-epoch runs.

Original h5 models selected window-weighted all-five validation; new models preserve window weighting and select all-ten validation. The training and matching selection horizon change together. Equal-episode evaluation is unchanged; the auxiliary equal-episode journal metric never selects checkpoints.

| Comparison | Horizon | Metric | First MSE | Second MSE | First relative reduction | Paired difference 95% interval |
|---|---:|---|---:|---:|---:|---|

Report every mode/seed. H10 benefit requires negative upper paired 95% CI for new-minus-original all-ten mean error; compare ours with equally h10-trained Framewise separately. Validation evidence only; no new test claim.

Intervals resample training seeds and recording-session clusters (10,000 draws). Exploratory, unadjusted for multiple comparisons. Signed changes and negative results are retained.
