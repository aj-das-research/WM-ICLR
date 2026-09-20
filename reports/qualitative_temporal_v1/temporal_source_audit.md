# Temporal source audit

Status: passed. Source: `paper/figure_sources/qualitative_closest_v1/derived.json` (`da07f473966c180061d1ee6c67e27c94ce6ad4aef33b21af39492889ea45c7df`). Exact values, source hashes, per-offset curves and checks are in `temporal_source_audit.json`.

## Recommended compact temporal story

Use the four inherited middle-ranked cases, one row per study, with the same three methods at every shown horizon. DROID offers the actual observed strip0/5/10 followed by recorded targets35/60 (h5/h10); IWS offers one observed frame and four recorded targets at native offsets14/29/44/59 (H15/H30/H45/H60). Show full scalar error curves to retain behavior between those map snapshots. Keep PushT and Box adverse rows as explicit boundaries; retain the full existing gallery by reference. All images are recorded evidence, not decoded predictions.

### Middle-ranked cases: errors and same-window gains

| Study / fixed case | Shown frame indices | Method | Early → late standardized MSE | Early → late reduction vs AR |
|---|---|---|---:|---:|
| DROID `droid-2dce8777c34ed372dc9ff50c` | 0/5/10/35/60 | AR | 0.094072 → 0.172157 | +0.000% → +0.000% |
| DROID `droid-2dce8777c34ed372dc9ff50c` | 0/5/10/35/60 | Bounded ours | 0.084888 → 0.172682 | +9.763% → -0.305% |
| DROID `droid-2dce8777c34ed372dc9ff50c` | 0/5/10/35/60 | No-tanh ablation | 0.085541 → 0.169763 | +9.068% → +1.390% |
| PushT `000007` | 7/21/36/51/66 | AR | 0.178365 → 0.198750 | +0.000% → +0.000% |
| PushT `000007` | 7/21/36/51/66 | Bounded ours | 0.169938 → 0.204945 | +4.725% → -3.117% |
| PushT `000007` | 7/21/36/51/66 | No-tanh ablation | 0.158875 → 0.182175 | +10.927% → +8.340% |
| Box `000006` | 18/32/47/62/77 | AR | 0.249115 → 0.250762 | +0.000% → +0.000% |
| Box `000006` | 18/32/47/62/77 | Bounded ours | 0.234603 → 0.252496 | +5.825% → -0.692% |
| Box `000006` | 18/32/47/62/77 | No-tanh ablation | 0.226997 → 0.231768 | +8.879% → +7.574% |
| Rope `000000` | 2/16/31/46/61 | AR | 0.202638 → 0.227802 | +0.000% → +0.000% |
| Rope `000000` | 2/16/31/46/61 | Bounded ours | 0.207413 → 0.234136 | -2.356% → -2.780% |
| Rope `000000` | 2/16/31/46/61 | No-tanh ablation | 0.197435 → 0.210326 | +2.568% → +7.671% |

“Early” means first exported target map: h5 for DROID and H15 for IWS. Full curves begin at h1 and H2 respectively. These are different native axes, not aligned seconds.

DROID’s middle window changes sign: bounded ours improves on AR at h5 (+9.763%) but slightly regresses at h10 (−0.305%). Its trajectory ranking gain is +5.358%, so the displayed-window endpoint must not be labeled as that episode average. The no-tanh component ends at +1.390% against AR; no superiority of bounded over the closest component is established.

The three IWS middle windows all favor no-tanh at the four displayed offsets, but their curves are not monotonic gains. PushT gains are +10.927,+6.685,+12.304,+8.340%; Box +8.879,+7.322,+4.602,+7.574%; Rope +2.568,+2.630,+7.656,+7.671%. Bounded ours is worse than AR at H60 in all three displayed middle windows (−3.117%,−0.692%,−2.780%). This selected-case fact is not a benchmark-frequency estimate. At the first available IWS prediction (offset1 = H2), the middle Rope window instead favors AR: no-tanh reduction is−33.688%, changing to+2.568% byH15. Keep the complete scalar curve rather than suggesting that the first exported map is the first forecast.

### Executed mechanism, early → late at the same displayed offsets

| Study | Model | Mean raw self-weight Tjj | Mean effective self-weight Mjj | Mean gate | Global correction RMS |
|---|---|---:|---:|---:|---:|
| DROID | Bounded ours | 0.7407 → 0.7084 | 0.9054 → 0.8528 | 0.2846 → 0.4135 | 0.1286 → 0.1995 |
| DROID | No-tanh ablation | 0.7376 → 0.6983 | 0.9045 → 0.8486 | 0.2865 → 0.4152 | 0.1330 → 0.2150 |
| PushT | Bounded ours | 0.2906 → 0.2479 | 0.6059 → 0.5695 | 0.5538 → 0.5737 | 0.4196 → 0.4774 |
| PushT | No-tanh ablation | 0.2623 → 0.1857 | 0.5212 → 0.4228 | 0.6441 → 0.7073 | 0.5443 → 0.6865 |
| Box | Bounded ours | 0.1700 → 0.1861 | 0.5702 → 0.5354 | 0.5183 → 0.5719 | 0.4638 → 0.4892 |
| Box | No-tanh ablation | 0.1603 → 0.1373 | 0.4497 → 0.3806 | 0.6532 → 0.7162 | 0.6030 → 0.7135 |
| Rope | Bounded ours | 0.1976 → 0.2114 | 0.5365 → 0.5340 | 0.5862 → 0.5978 | 0.4886 → 0.5066 |
| Rope | No-tanh ablation | 0.1515 → 0.1542 | 0.3884 → 0.3682 | 0.7228 → 0.7486 | 0.6888 → 0.7351 |

These are means over all16 patch positions and all3seeds. They must not be confused with the existing decoder figure’s single fixed query q=5. The corresponding full T/M/g maps already exist at every displayed offset. The scalar correction curve averages each seed’s RMS over16×384 coordinates; correction-map cells average each seed’s RMS over384 channels at one patch. This distinction was independently checked from the per-seed saved map values.

In the IWS middle windows, no-tanh ends with more off-location mixing and a larger correction RMS than bounded ours. That coincides with lower endpoint error in these windows, but is not a causal explanation: both models were separately trained and their gates/mixing differ. Neither larger corrections nor lower same-location weight guarantees success; the adverse cases below exhibit the same broad changes.

### Boundary cases that should remain visible

| Case / method | H15 | H30 | H45 | H60 | Trajectory ranking gain |
|---|---:|---:|---:|---:|---:|
| PushT `000004@2`, Bounded ours | +2.772% | -3.722% | -10.291% | -20.050% | +0.674% (no-tanh vs AR) |
| PushT `000004@2`, No-tanh ablation | +7.377% | +4.582% | -0.549% | -11.784% | +0.674% (no-tanh vs AR) |
| Box `000008@9`, Bounded ours | +6.812% | +7.534% | +1.484% | -25.193% | -7.558% (no-tanh vs AR) |
| Box `000008@9`, No-tanh ablation | +8.664% | +9.047% | +10.124% | -17.838% | -7.558% (no-tanh vs AR) |

PushT’s weakest selected trajectory remains positive on its full trajectory average (+0.674%), while this exact first registered window regresses at H60 (−11.784% no-tanh). Box’s weakest trajectory and displayed endpoint both regress (−7.558% and−17.838%). The error curves are essential: Box absolute errors at H45→H60 decrease for all three methods, but AR decreases more, causing the relative-gain reversal. “Errors keep accumulating” would therefore be false.

DROID’s already selected weakest case offers an optional additional boundary: bounded relative gain +59.885% at h5 becomes−14.524% at h10. It should supplement, not replace or reselect, the inherited middle case.

## Interpretation and source limits

- The horizontal axis is supplied causal command-prefix length/forecast offset, not online adaptation or refreshed visual evidence. The mixing source remains the initial/last observed grid at every horizon.
- DROID uses three actual observed grids and support context; IWS uses one actual observed image repeated into architectural slots, without invented history. These are separately trained architectures and scopes.
- The four middle-ranked cases are inherited fixed cases, not cases newly selected for visually attractive temporal behavior. The complete existing largest/middle/smallest gallery remains authoritative.
- DROID case ranking used bounded-vs-AR development episode averages. IWS case ranking used no-tanh-vs-AR reserved trajectory averages. A displayed minimum-start window can have a different sign.
- Each curve is a mean over three model seeds on one fixed window. Adjacent horizons and patches are dependent; no intervals or population-frequency claims are inferred from these selected examples.
- DROID public numeric maps/target photos exist only at h5/h10. Full scalar curves h1–h10 are available. Do not invent h1 maps or intermediate target photos from this public pack.
- IWS saved target maps/photos are offset14/29/44/59, labels H15/H30/H45/H60. Offset1 is H2, not H1. There is no offset0 prediction or gate measurement.
- Native frame indices are source frame numbers, not seconds; no common physical-time alignment across DROID/IWS or across separate trajectories is established.
- Error maps are standardized feature-coordinate MSE averaged across384 channels at each4x4 patch and then acrossseeds. They are not RGB predictions, pixel losses, object masks or semantic saliency.
- Mean T and M are per-seed matrix means. Mean M cannot be reconstructed from mean g and mean T. Effective diagonal≥raw diagonal is an anchoring identity, not evidence of causal benefit.
- The correction scalar curve is mean_seed sqrt(mean_over_16x384(delta^2)); each map cell is mean_seed sqrt(mean_over384(delta_patch^2)). Averaging map cells does not equal the scalar curve.
- Correction RMS does not reveal signs or directions, prove saturation, or isolate why no-tanh changes accuracy. Removing tanh changes trained weights and their learned gates/mixing as well.
- Gate/self-weight/correction co-movement with error remains descriptive. No causal relation, object tracking, physical correspondence or learned online strategy is established.
- Use shared within-task zero-based feature-error and correction-map scales across all compared methods/cases/horizons. Cross-task magnitudes use different fitted normalization and should not be presented as directly comparable difficulty.
- Recorded target RGB is withheld evidence, never model output or new support. IWS originals staylocal; only parent-approved low-resolution attributed excerpts may be embedded in annotated figures with unspecified dataset-license status disclosed.

## Verification performed

Recomputed all1,683 mean curve entries from all5,049 seed entries; all4,488 mean mechanism entries; all126 error maps and84 T/M/g/correction map sets. Verified all60 existingPNG file and pixel hashes, all role/native-index mappings, within-study scales and displayed endpoint gains. This pass reads the derived pack and already reviewed localPNG assets only; it does not rerun inference or reopen original videos/caches. Frozen author sources were not modified.

## External raw-v2 completion status

Checked2026-09-20T18:03:14.883982+00:00: five of six100-epoch completion markers exist. `202064_3` (recursive seed1) remains RUNNING; finalizer`202074` is PENDING Dependency. No finalization/completion file exists in the reporter namespace. No partial accuracy payload was opened or aggregated.
