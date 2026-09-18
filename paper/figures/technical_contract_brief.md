# Code-derived inference comparison

Reader/slot: compact ICLR explanatory inset, 5.5 × 2.4 inches. Method mode.

One sentence: Given the same observed history, executed actions and goal,
ShiftWM introduces separate observation and transition contexts into a compact
world model; this schematic identifies the implemented difference without
claiming that it causes any selected success.

Three compositions considered: (1) a large single-branch architecture, rejected
because it makes the Framewise control seem absent; (2) a side-by-side pair of
complete diagrams, rejected because repeated CEM internals crowd the contexts;
(3) aligned lanes with explicit changed operators and a quiet repeated
predictor/CEM region, selected to preserve comparison and avoid implying shared
trained predictor weights.

Representation contract: the real last-support and available goal images from
Reacher development episode 2031004 anchor shared input objects. Three feature
bars denote H=3 support, while two orange bars denote two executed action blocks;
these bars are schematic identities, not measured activations. Only the last
support frame is pictured. No generated image or quantitative output is used.
Blue marks observation features/calibration; orange marks dynamics conditioning
and action embeddings; purple marks the corrected goal path. Operators and
connections are editable vector geometry. The two input thumbnails are embedded
original raster observations.

Scientific contract: Framewise applies u+MLP(LN(u)) independently to history and
goal, retaining temporal, action-conditioned prediction. ShiftWM computes c_o
from support mean/variance, calibrates both history and goal with residual FiLM_o, infers c_d
from corrected support transitions and executed actions, and FiLM-modulates
action embeddings. Goal data never enters context inference. Candidate future
actions never enter context inference. Goal arrows bypass the predictor and
terminate at CEM's goal-dependent cost. The repeated gray region indicates common
architecture/procedure, with each method's separately trained weights. All model
weights are fixed online; contexts are fixed within CEM and recomputed after a
real action block. Offline paired losses are omitted, not present online.
The E_a(a) operator represents action embeddings for both executed past and
candidate future actions in each recursive predictor window; it is not limited
to candidate actions. Predictor parameters are independently trained for the
two methods, even though both are denoted P_theta.

Source authority: src/shiftwm/model.py and src/shiftwm/evaluate.py. The renderer
writes original-image/code hashes and named edge routes to
paper/generated/qualitative/technical_contract.json. No paper code, experiment
configuration, model, or scientific runtime is changed. No external reference
image was needed: the topology is original and source-derived. Rebuild with
`.venv/bin/python paper/scripts/render_technical_contract.py`.
Use `--if-needed` to skip only when all source and export hashes match.

Caption draft: **Implemented inference difference.** Framewise retains a
temporal action-conditioned predictor and applies residual per-image calibration
to history and goal. ShiftWM infers observation context from history statistics,
then dynamics context from corrected transitions and executed actions; the latter
modulates action embeddings. The gray region denotes the same architecture and
planning procedure, separately parameterized for each method. The goal bypasses
context inference. Weights remain fixed online, and contexts update only after
real execution. Thumbnails are actual available inputs; feature bars and
operators are schematic. This diagram describes computation, not a causal
explanation of the selected outcomes.

The GRU transition input is exactly [z_i, z_{i+1}−z_i, normalized a_i].
The compact [z, Δz] glyph denotes the first two pieces, with executed a_{0:1}
shown at a separate input port. FiLM_o and FiLM_d are residual affine adapters;
their scales use 1+0.1 tanh(scale) in the implementation.
