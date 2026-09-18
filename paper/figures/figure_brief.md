# Figure brief index and original design record

Current revisions: `method_redesign_brief.md`, `split_brief.md`, `forecast_brief.md`, `training_brief.md`, and `goal_calibration_brief.md`. The final review is `visual_refresh_review.md`. The original notes below record the earlier placeholder-stage design; they do not describe the current availability of empirical results.

# Figure brief and representation contract

Version: 2026-09-18. Authoring source is TikZ. Intended venue: official ICLR 2027 template; actual text width 5.5 inches. Figure sources remain editable; rendered PDF geometry is vector. This is an implementation-stage draft, not evidence of experimental improvement.

## Selected modes

1. Method: `world_method.tex`, width 13.6 cm, target height 8 cm. The key relationship is **observation context corrects observation latents, while dynamics context conditions action-dependent prediction**. Conventional LeWM modules stay gray; observation operations are blue; dynamics operations are orange. Symbols and labels also distinguish paths in grayscale.
2. Experimental design: `factor_split.tex`, width 13.6 cm, target height 4 cm. Eligible because the manuscript explicitly defines a held-out Cartesian factor protocol. Existing environments are reused; factor split construction is adapted for this study. The matrix matches the prespecified split in `src/shiftwm/data.py`: seven training pairs, development `(1,1)`, held-out composition `(2,2)`. It is not a measured result.
3. Quantitative result slots: `world_plot_slots.tex`. Axes only; no plotted values, trends, or error bars. Dataset-specific plots will be generated from recorded run evidence.

No empirical teaser is drawn: no result is available to occupy its evidence panel. No images are represented as actual simulator observations.

## Three method compositions considered before authoring

- Parallel baseline/proposal lanes: clarifies changed modules but repeats the full predictor and hides support/query separation.
- Two-axis factor matrix feeding a model: foregrounds compositionality but cannot expose inference information access clearly enough.
- **Selected: support-conditioned inference above a separate training-only target lane.** The upper portion traces actual observation/action history to both contexts; the middle shows calibrated prediction and planning; the lower lane shows privileged canonical targets and loss. Context sources remain distinct from future targets.

## Nodes and operations

| Node | Inputs and outputs | Weight state in default configuration |
|---|---|---|
| Base visual encoder/projector E0 | Images → raw latent sequence u | Frozen |
| Observation context Co | Raw support features → co | Trained offline; fixed online |
| Observation adapter Ao | (u, co) → calibrated state z | Trained offline; fixed online |
| Dynamics context Cd | Corrected chronological support plus executed actions → cd | Trained offline; fixed online |
| Action conditioning Ad | Candidate action embedding + cd | Trained offline; fixed online |
| LeWM predictor P | Corrected states + conditioned actions → future canonical-space latent | Fine-tuned offline; fixed online |
| Reference E0 | Same-state canonical training render → target z* | Frozen, training target path only |
| MPC/CEM | Rollout + corrected goal embedding → executable action chunk | Existing solver |

Solid arrow: forward computation or conditioning, edge label disambiguates object. Dashed arrow: training-only loss connection. No gradient arrows are drawn; weight state is stated in node labels/caption. The target encoder is the same fixed coordinate system as the frozen base encoder, not an EMA update.

## Sources

- `reports/small_world_models_proposal.md`: motivation and factor protocol.
- `external/le-wm/jepa.py`: upstream encoding, action-conditioned prediction, rollout, and terminal latent-goal cost.
- `src/shiftwm/model.py`, inspected 2026-09-18: raw support→observation context; corrected support and executed actions→dynamics context; query prediction and canonical alignment; paired consistency in factorized mode; no context swap in the initial implementation.
- `paper/main.tex`: mathematical definitions; the paper must be reconciled against the final implementation before result claims are inserted.

## Known boundaries

The diagram is a computation schematic, with no learned latent values or empirical output. No inference input contains simulator physics IDs, future frames, rewards, or canonical current-state renders. Goal images are corrected using the inferred observation context; this assumes they share the observation setting, which must be tested or separately specified. Context-update cadence, exact support length, loss weights, and model dimensions belong to versioned run configurations, not assumed constants in the figure.
