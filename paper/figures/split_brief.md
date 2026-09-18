# Split figure: brief and representation contract

The setup is **present** in `paper/main.tex`, Experimental protocol / Environments and splits and Primary outcomes and uncertainty. Its status is an **adapted setup on existing PushT and Reacher environments**, not a new simulator. This is a configuration figure, not performance evidence. Intended width is the manuscript's **5.5 inches** (13.97cm); `factor_split.tex` is the canonical editable geometry. All task illustrations are explicitly schematic vector drawings.

## Visual thesis and contract

One held-out combination of familiar appearance and dynamics values tests composition; three separate combinations with new values test extrapolation. Both use the same model-visible planning interface and budget.

An evaluation case starts from a recorded simulator state and supplies an image goal. The model sees shifted RGB history, past executed controls and a shifted goal image. It never receives hidden simulator state, factor IDs or evaluation scores as predictor inputs. The shared CEM/MPC planner executes five native controls and observes again. Upstream environment-specific goal-success criteria stop execution at success or the 50-control budget. Initial three-frame support costs ten of those controls. Support-only successes are separately recorded; policy-eligible summaries exclude them.

The matrix shows seven training combinations, development `(v1,p1)`, held-out composition `(v2,p2)`, and extrapolation `(v3,p0)`, `(v0,p3)`, `(v3,p3)`. The four other new-level intersections are unevaluated. Independent test trajectories cover all nine in-range combinations, including pairs encountered during training/development. Initial-state seeds are separated before paired appearance rendering. Validation uses the seven training combinations with independent validation trajectories.

## Source ledger

| Encoded configuration | Authoritative source |
|---|---|
| Canonical/warm/cool/dim appearance factors; paired views preserve physical state | `src/shiftwm/data.py`: `APPEARANCES`, `appearance_transform`, `pixels_to_tensor` |
| Seven train cells; dev11; held-out22; all9 test cells; three extra cells | `src/shiftwm/data.py`: combination constants and `split_combinations` |
| Trajectory/seed split integrity; factor IDs absent from inputs | `src/shiftwm/data.py`: `validate_manifest`, `TrajectoryDataset`; manuscript split paragraph |
| Real PushT damping and Reacher arm/finger density changes | `src/shiftwm/generate.py`: `DYNAMICS`, `make_env` |
| Observed support, goal construction, receding horizon and scoring boundary | `src/shiftwm/evaluate.py`: `evaluate_planning`; `src/shiftwm/model.py`: `infer_context`, `goal_embedding` |
| Main300/30/30 CEM, horizon5groups, native50 budget | `scripts/run_evaluation_campaign.py`: `OFFICIAL_PLANNER_BUDGET`; manuscript primary outcomes paragraph |

Physical values are omitted from tiny axis labels: PushT damping `[0,.15,.65,.95]`; Reacher arm/finger density `[1000,650,1350,1500]`. Parameter details and units remain in the manuscript/code. The appearance change is photometric, never a camera/viewpoint change.

## Representation contract

- Focal relation: the blue held-out cell lies inside the familiar-factor grid; purple extrapolation cells cross a new factor boundary.
- Direct cell labels and dashed new-factor guides supplement color. The matrix is not a score heat map.
- The T block, circular pusher and two-link arm make the established tasks recognizable. Three support frames and a separate goal depict a schematic PushT episode; they are not observed output or a claimed successful rollout.
- Reading order: task identities → crossed matrix → composition/extrapolation distinction → visible history/goal → shared planner/action loop → separate evaluator note. The bottom strip explains seed splitting and paired images.
- Information boundary: no arrow connects private state/factor IDs or scoring to model inputs. No policy-gradient or reward-learning feedback is implied. The stop rule may terminate the outer evaluation loop.
- Omitted detail: learned architecture, losses, exact task-specific success thresholds, baseline supervision differences and all experimental results. Pairing supervision is not claimed to be identical across every baseline.

## Three structurally distinct composition sketches

**A: matrix beside an observed episode — selected.**
```text
 PushT + Reacher                 observed frames    goal
 familiar 3x3 | new column               |            |
 new row + corner                    shared planner
 composition versus extrapolation    act / observe loop
 -------- seed split before paired rendering --------
```
One matrix makes familiar factors versus new values directly comparable; the right panel clarifies model access without becoming a second method diagram.

**B: construction followed by separate evaluation panels.**
```text
 seed families -> simulate physics -> paired images
 [7 train cells] [held-out22] [three extra cases]
                 shared execution strip
```
Rejected: separating conditions obscures that each held-out factor was seen independently, and duplicates axis labels at5.5in.

**C: centered environment episode with surrounding shift insets.**
```text
 same-state RGB variants -> [PushT / Reacher episode]
 physical variations    -> [action / observation loop]
                          [small matrix inset]
```
Rejected: the defining task population becomes a secondary inset; visually rich but less effective for explaining composition versus extrapolation.

## Caption suggestion

**Adapted evaluation on existing PushT and Reacher environments.** Seven appearance–dynamics combinations are used for training, `(v1,p1)` is reserved for development, and `(v2,p2)` tests composition of factors seen separately. Three extra combinations introduce a new visual value, physical value, or both; dashes mark unevaluated combinations. Independent test trajectories cover all nine in-range pairs. The schematic episode shows the RGB history, executed controls and goal image available to each model under a shared planning budget. Hidden state and factor labels remain evaluator metadata. Initial-state seeds are split before paired rendering; the figure specifies configuration, not measured performance.

## Reproduction

From `paper`, compile `figures/split_standalone.tex` into `build/split_figure_review`; export PDF→SVG with `pdftocairo -svg` and PNG with `pdftoppm`. Main manuscript/build/watch processes remain owned by the coordinating agent. See `split_review.md` for pixel review, export dimensions and remaining limitations.
