# Positive qualitative cases — figure brief

The user requests stronger visual explanations of positive outcomes. Include
every ShiftWM-only success versus Framewise in the existing 64-task seed-zero
development gallery: PushT2031024 and Reacher2031004/2031008/2031011. Shared
context also fails on each. This selection is explicitly conditional on
success; it is not evidence of a superior average success rate.

At 5.5-inch paper width, the focal relationship is whether the actual block,
pusher or arm configuration approaches the common image goal. The existing
balanced success/failure/control panels and complete gallery remain available.

Three composition sketches considered:

```
1. Goal-reference lanes (selected)
             goal    same time t20    actual endpoint
  ours        []          [] + outline      [] + outline
  Framewise   []          [] + outline      [] + outline
        shared support | native-action budget / termination

2. Matched spatial detail
   full scenes -> same crop of goal / ours / Framewise
   endpoint metric strip

3. Endpoint error dashboard
   four task rows × terminal error / actions / success
   small image key
```

Choice 1 preserves full scenes, exact matched time and explicit endpoint times.
Choice 2 would reveal subtle joint differences but sacrifices scene context
unless another row is added. Choice 3 repeats quantitative tables and does not
show what the robot physically did. The chosen goal contour makes pose
differences visible without estimating physical coordinates from screenshots.

Representation contract: RGB frames are unchanged source observations; dashed
purple contours come from the saved goal image and are display-only annotations.
PushT contours trace the gray block and blue pusher, excluding the unscored
green marker. Reacher contours trace the orange arm. Pixel coordinates are
explicitly registered to the original images; this is not a model prediction,
attention map or inferred trajectory. Numeric physical errors and success
labels come only from evaluation records. The middle frames are actually
recorded at native call20 for both methods. Each terminal image keeps its own
time,21/23 versus50. Timelines show paid support and stopping; they do not
claim a speedup against a baseline that never succeeded.

Labels explain the case-specific observation: attaining block/pusher target,
matching a folded arm, a near-looking joint mismatch, and forearm orientation.
The caption separates what happened and why the score passes from the unproven
internal explanation. Context necessity and causal component attribution need
the controlled experiments now running. The two new pages supplement the
four earlier balanced pages. Reproduce via
`paper/scripts/render_positive_qualitative.py`; preserve `positive_evidence.json`.

Print-size repair: the full-frame Reacher arms were too small to communicate
the orientation differences clearly. Add the spatial-detail idea to the paired
lanes: one common square crop per case contains the goal and every displayed
arm pose across both methods, with twelve pixels of padding. All cells share
its scale and bounds. A full-goal overview locates the crop, and the measured
display magnification is labeled. Crop coordinates are stored in the evidence
packet. PushT remains full-frame because off-screen departure is part of its
observed behavior. This increases detail without selecting different crops for
the methods or changing any physical measurement.
