# Prespecified spatial qualitative replay and candidate figures

Apply the paper-figure-creation skill to an observed-evidence, matched-diagnostic figure. This protocol and the deterministic `selection.py` rule are registered before reading any spatial campaign scores. No generated illustration, synthetic trajectory or decoded RGB prediction enters the evidence. The final candidate is outside the paper until numerical and actual rendered-pixel review both pass.

## Population and selection

Use only the completely verified fifteen-model original-validation spatial campaign. Compare the registered spatial transport model with the same 4×4 autoregressive baseline, at all three seeds 0/1/2, on identical original-validation episode/session/window identities. Rank each episode by percentage reduction of native h10 standardized feature MSE: first average all complete windows within the episode, then equally average the three seed-specific errors; gain is 100 × (autoregressive − transport) / autoregressive. Require a positive baseline denominator.

Sort by descending gain with ascending episode-ID ties. Select ranks 0, floor((N−1)/2), and N−1. These are the largest-gain, median-gain and worst-gain cases; call the final case “largest regression” only if its measured gain is negative. If no episode regresses, explicitly say so and retain the smallest-gain case. Preserve all three examples even when the observed first-window effect differs from its episode-average rank.

For every selected episode replay the smallest eligible window_start, never a visually selected window. Use all three seeds and both methods. The pictured endpoint is query step ten. Show all three observed support images and the actual tenth-query target using recorded native frame indices. The full population gain inventory, positive/negative/tied episode counts, medians, and the pooled ratio-of-means gain accompany the conditioned examples. A selected example is not a population estimate.

## Numeric replay and semantics

Load only immutable validation feature payloads, original RGB payloads and selected best checkpoint packages after validating the complete campaign's registration, finalizer, cache/resource gate, evaluation files and package/source hashes. Verify validation membership before opening any payload. Recompute per-window ten-step native errors from exact released model weights and cached features/actions; compare every replayed step against the source evaluator's window ledger using preregistered CPU-vs-GPU tolerance rtol2e−5/atol2e−6. Save measured replay discrepancies. Do not call tolerance agreement bit-exact cross-device prediction equality.

For feature maps, reshape channel-major native features to 384×4×4. Square errors in the frozen shared-per-channel standardized coordinates; average over channels and then training seeds. Both methods and all three cases share one linear color scale from zero to the maximum across the six displayed endpoint maps, without clipping, percentile normalization or case-wise scaling. Full raw per-seed predictions, targets, maps and all-ten-step errors remain in the candidate array artifact.

Obtain the actual model transport matrix, gate and bounded innovation through `return_details=True`; verify reconstructed normalized predictions against the ordinary predictor. Display the mean h10 transport row associated with fixed zero-based target patch (1,1), reshaped to a 4×4 source grid, using scale [0,1]. Show its mean gate separately. Save both the raw transport matrix and effective mixture `(1−gate)I + gate T`, plus innovation. These are learned mixtures of semantic features, not physical displacement, optical flow, attention-grounded correspondence or predicted RGB. The transport panel explains a measured intermediate computation; it does not establish a causal source of improved error.

## Visual and provenance contract

Use a compact 5.5-inch-wide figure with three matched case rows. Actual support/target RGB images remain intact and equally sized; annotations and separate feature maps are vector layers. Do not alter pixels or remove backgrounds. One full shared legend explains map units and the measured transfer weights. Include the first-window endpoint gain separately from the episode gain that selected the case, and retain an all-ten-step error curve with means and optional three-seed variability.

Export candidate PDF, editable SVG, PNG, manuscript-size and enlarged/grayscale proofs, per-case data, and provenance ledger only to `artifacts/qualitative/spatial_v1_candidate/`. Source ledger covers selection registration, full population scores, raw image/feature payloads, model manifests/weights, scripts and finalizer/evaluation files. Export untouched PNG frames with decoded-pixel hashes verified against the source arrays.

Reject incomplete campaigns, missing seeds/windows, changed source hashes, nonfinite arrays, misaligned frame/action indices, replay discrepancy, invalid transport row sums, altered RGB pixels, ambiguous scales, and generated tables/captions that imply RGB prediction. Automated bbox/font/compile checks supplement actual pixel review; mark visual review pending until the candidate PDF/PNG has been inspected at paper width and enlarged. No manuscript include or publication is created by this candidate pipeline.
