# IWS development diagnosis plan

This is a read-only scientific diagnosis and proposed follow-up, not a result table or an explanation already established by experiments. No provisional benchmark score, gain, or ranking is reported here. All comparisons remain subject to the complete, common-device evaluation and finalization gates. No model prediction, training, feature-cache extraction, development/reserved payload read, or configuration change was performed for this memo.

## What is proven by the implemented decoder

`SingleObservationWorldModel._predict_normalized` uses the same initial standardized grid at every forecast offset in both fixed-anchor arms. In the bounded mixing arm, let source vectors be `z0[j,c]`, row-softmax transport be `T`, and the patch gate be `g[i]`:

- `m[i,c] = (1-g[i])*z0[i,c] + g[i]*sum_j T[i,j]*z0[j,c]`.
- `prediction[i,c] = m[i,c] + b*tanh(R(H)[i,c])`, with the registered `b=1`.
- Thus `min_j z0[j,c] - b <= prediction[i,c] <= max_j z0[j,c] + b` for each target patch and channel. This envelope is **independent of horizon**; the method adds one bounded correction to the fixed source at each query, not an accumulated correction over time.

The normalization buffers are fixed training statistics shared across all 16 positions within each channel. The bound is in standardized coordinates. In original feature coordinates its additive extent is `b*std_train[c]`; it is not a bound on RGB displacement, action displacement, forecast error or physical motion.

For a target `y`, define `l[c]=min_j z0[j,c]-b` and `u[c]=max_j z0[j,c]+b`. The squared distance to this interval,

`LB(y,z0,b) = mean_(i,c) (max(l[c]-y[i,c],0) + max(y[i,c]-u[c],0))^2`,

is a lower bound on every bounded-mixing prediction's standardized MSE for that input and target. A positive value proves that some residual error is unavoidable **under this representational envelope**, even with perfect fitting. It does not prove that the observed method gap is caused by this restriction.

This is a deliberately loose relaxation: the same nonnegative row weights apply to all channels. The true mixture lies in the convex hull of the sixteen full source vectors, with a coordinatewise bounded correction; independent channel minima/maxima form a larger box. Therefore a zero interval floor does not prove the target is reachable by actual shared weights. Conversely, a large interval floor is decisive evidence of an unavailable target region, though optimization can add further error. Neither an interval-clipped target nor its floor is an attainable oracle predictor.

The existing AR arm instead updates `register[-1] + R(H)` and feeds the resulting feature state back, with no tanh bound and no detach across its 59-step training graph. It can accumulate representational change; it has no corresponding fixed-initial-source channel envelope. The additive-anchor arm is also unbounded (`z0 + R(H)`), so a weakness shared by both anchor arms relative to AR would not be explained by tanh alone.

## What the 27 training histories establish, and what they do not

Parsed only compact scalar fields from all 810 saved epoch records: epoch, training all-59-window MSE, validation equal-trajectory H60 MSE, learning rate, and selected epoch. Every run has epochs 1…30, finite inspected scalars, and its summary-selected epoch agrees with the earliest minimum of its logged selector. Full ledger/CPU evaluation validation remains the finalizer's responsibility.

- All runs end below their epoch-1 training and validation scalars.
- Every run's training scalar strictly decreases through the final five epochs.
- All eighteen fixed-source runs (additive and bounded mixing across three tasks/seeds) select epoch 30 and have strictly decreasing validation scalars over their final five epochs.
- Twenty-three of all twenty-seven runs select epoch 30. AR's selected epochs are PushT `[25,30,26]`, Box `[30,30,28]`, Rope `[30,30,29]` for seeds 0/1/2.

These logs do **not** show a late validation upturn in the fixed-source arms; claiming simple late overfitting as the cause is unsupported. Continued improvement at the registered stopping budget makes limited optimization a plausible alternative. It does not establish that more epochs would eliminate any gap. The learning rate has nearly reached its registered minimum, and loss descent by itself does not measure reachable capacity or extrapolation.

Do not interpret training/validation loss gaps directly: training is equal-window all-59-offset loss with the training numerical/dropout regime, while checkpoint selection is float32 equal-trajectory H60 validation. These are different aggregates and execution modes. The original logged selector histories also do not replace the common-CPU re-evaluation audit.

## Ranked hypotheses and discriminating observations

1. **Fixed unit envelope / bounded-head saturation.** The representational limit above is certain; its practical importance is unmeasured. Measure the interval floor and out-of-envelope coordinate fraction by horizon first. On a fixed training-only model sample, a high fraction of `abs(innovation)/b > .95` would support saturation as an additional optimization limitation, but would not by itself show that targets require more range. Inspect raw residual projection or tanh derivative if needed; do not infer saturation from gate weights or mean retention.
2. **Fixed visual state versus recurrent state evolution.** Code-confirmed distinction: fixed-source arms repeatedly encode the initial visual grid, whereas AR re-encodes its changing three-slot register. All methods receive the same 96D causal command-prefix GRU state; fixed-source arms must express long changes from that state and the original visual evidence without an evolving predicted visual register. Whether this creates a harmful compression/optimization bottleneck is a hypothesis. A weakness shared by both unbounded additive anchoring and bounded mixing would make this more plausible than tanh alone.
3. **Finite training budget / objective emphasis.** Late descent permits this hypothesis. The loss weights all 59 offsets equally while selection targets the endpoint, so early-offset optimization may compete with H60. This mismatch is shared by all arms and cannot explain a difference without an interaction. Do not extend only the proposed arm or change only its loss and call the result matched.
4. **Identity-favoring initialization and tanh gradients.** `+4I`, initial gate logit `-3`, and a zero residual head favor near-persistence initially. This is not exact identity: sigmoid(-3) is nonzero and softmax retains off-diagonal mass. The mixing gate can learn; its initialization is not a hard restriction. Saturation or slow departures require trained-head/gradient measurements, not inference from the initialization constants.
5. **4×4 pooling and single-image ambiguity.** Both are shared across current arms. They limit available scene detail/dynamics information, but cannot alone explain a method-specific ordering. A coarse grid may interact with convex source assembly, since one weight per source patch is shared across 384 channels. A single view can also omit velocity/contact state. Context/FiLM were correctly disabled because no observed transition exists; restoring a DROID transition context from future RGB would violate the task. Giving multiple observed frames is a different input protocol, not a fair remedy within this single-image comparison.

## Smallest diagnostic, now implemented by root

Root's new `scripts/real_video_iws_recovery/diagnose_training_envelope.py` is the appropriate first experiment. It uses no learned model or checkpoint and can run on CPU:

- Frozen `internal_train` inventories only, all eligible starts at stride five under the exact strict `s+60<N` rule; no development or reserved payloads.
- Shared training normalization; stored offsets 14/29/44/59 (H15/H30/H45/H60).
- Bounds 0/1/2/4, clearly labeled analytical counterfactual envelopes, not separately trained models.
- Per-window interval floor and fraction outside; average windows within trajectory, then trajectories equally per task. Retain every eligible trajectory and explicit exclusions. Persistence MSE on the same training targets is a scale reference, not a benchmark-performance claim.
- Float64 is intentional diagnostic arithmetic. It is not a replacement for the frozen model's float32 evaluation or proof of bitwise model parity.

**Independent source review: PASS, no access or mathematical blocker.** `train.open_cache` opens registered metadata only; `cache.episode(eid, "internal_train")` authorizes before reading, checks receipt/index/payload hashes and validates finite native arrays. Registration binds the cache/statistics identities and is checked again before completion. Shape/layout and normalization are channel-major 384×16; the arithmetic and equal-trajectory aggregation are correct. No checkpoint load/model construction/development/reserved payload route is present.

I independently extracted only `interval_error` into a NumPy-only namespace (no model/data imports), and checked exact agreement with interval projection, zero violation for random gated convex mixtures plus bounded correction, the MSE lower-bound inequality for arbitrary targets, and monotonic nonincrease of both diagnostic quantities for bounds 0/1/2/4. The reviewed script SHA is recorded below. No training data diagnostic was run by this reviewer. Optional defensive assertions for finite positive/shared mean/std could make the helper more self-contained, but the registered immutable cache statistics already establish those conditions for this job.

Interpretation after results: a small floor weakens the simple axis-aligned-envelope explanation but does not rule out shared-vector convex-hull or optimization limits. A large floor motivates a controlled bound-removal experiment, but neither its fraction nor its ratio to persistence equals the fraction of a learned model gap causally explained. Do not pool this training diagnostic with finalized development errors from different windows.

## Smallest matched follow-up after the diagnostic

The first targeted model revision should be **the existing fixed-source mixing decoder with only tanh bounding removed**, implemented under a new registered source/config/package namespace. Keep query/key/gate, fixed source/keys, all inputs, feature coordinates, seeds, initialization, optimizer, training budget, data split and selector matched. It adds no new residual parameters and directly tests whether bounded correction limits this IWS interface. Present it as “ShiftWM without the innovation bound,” an ablation of one method. The existing additive-anchor control alone does not isolate bounding because it also removes mixing and its heads.

Do not simply enlarge b on a trained checkpoint and report the altered predictions as a fair trained comparison; that is at most an explicitly labeled intervention diagnostic. If a finite enlarged bound is the chosen follow-up, select its rule using the training diagnostic before reserved outcomes, register it once, and retain the original bounded model and all unfavorable findings. Avoid an unreported sweep of bounds using the development endpoint.

If the envelope floor is small and fixed-source arms are still improving, a later equal-budget extension for **all compared arms** or an explicitly matched objective study can test optimization. An evolving latent-query state while retaining observed-source decoding is a larger architectural hypothesis, not an immediate fix justified by this memo. Do not combine bound, recurrence, resolution, context and training-duration changes into one unexplained improvement.

Keep completed v1 results immutable. All follow-ups are exploratory development revisions after observed development evidence, not fresh confirmations. Preserve the reserved-set lock until intended methods/checkpoints/evaluation are fixed, then use the agreed untouched scope rather than repeatedly retuning against it.

## Source and history provenance

Read-only audit written 2026-09-19T21:47:34.397815+00:00.

Scientific sources:

```json
{
  "src/shiftwm/real_video_iws/model.py": "f10c5f3855d510d982e80001da19d040cb280eaa747fdb01f8fcffb5af4a1f1a",
  "src/shiftwm/real_video_iws/windows.py": "350df1c8f81ff3b3155f1c92de56c71af8d6f803d82906d2891b0f3feef0c578",
  "scripts/real_video_iws/train.py": "2e84d01d1ff085325a93925fe3385f30f5ee0a965823c14b23800572fbd24d74",
  "configs/real_video_iws/training_v1.json": "6c8b34b9422891a23ccf64df853aeee82eff63cd80842f6244f4409ed51332ac",
  "configs/real_video_iws/training_registration_v1.json": "edbff0ee007201be302adc70a607270759a89f598f76361464a0088df4e25995",
  "scripts/real_video_iws_recovery/diagnose_training_envelope.py": "ffc3fc441920d11e750ecc80ee3d14f452cbabe1b8f92f19c3002b37f1ca68f0"
}
```

Compact history checks (no loss values or provisional method rankings):

```json
[
  {
    "run": "bimanual_box_anchored_additive_s0",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "13e5c6b827e11b14a8fb0f1588b1dbecb501363b8bb36e6ec199d05f426bcae5",
    "history_sha256": "3c31225de368e3a722075562f6f162f409e0486fd0eeccbc3e088a0190404432"
  },
  {
    "run": "bimanual_box_anchored_additive_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "78d4a4d7b6afec0b29ee7bc5ff145a41a605bb1d3dcf86b45143670a8b44d8f2",
    "history_sha256": "21f35a9a51dafed52c455d2714660e096b0b0a13e224ce8d38c9d2c53493b34d"
  },
  {
    "run": "bimanual_box_anchored_additive_s2",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "a8387efa00e31d91a6c04976be1b2eab85f365b08effd6d82b92f6f48782495f",
    "history_sha256": "7e924df717c5ea2e6a22a60c7d6f1a0e94762a64c62bbf8a0d52120b39dc80b7"
  },
  {
    "run": "bimanual_box_autoregressive_s0",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "fd1c03452107af65230fe3f6c235253811fbfe49fc2a51a745f023c87eef2bbf",
    "history_sha256": "b0c37f8fac5d5f4c4769c49d77145b217e7178d8e4071dc0b164af07cf315d13"
  },
  {
    "run": "bimanual_box_autoregressive_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": false,
    "summary_sha256": "04fa9abae6d9c46130084b0f70251642de74740fa9c0ea51a92bf01a29dc52d6",
    "history_sha256": "c6430da06427016e158f5c1333f866ceebcae11955e3649f2d60332775265e0d"
  },
  {
    "run": "bimanual_box_autoregressive_s2",
    "selected_epoch": 28,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": false,
    "summary_sha256": "d06304faa6db6e84aa081a776d72cadd1efb7ee8d48c2500a04796bcf8ad2aa2",
    "history_sha256": "901a7db8502caffaac195cd4a2a2180e9117153c6ad64fe5e60ab5ff1c47cf92"
  },
  {
    "run": "bimanual_box_bounded_spatial_mix_s0",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "7924941113ea27b050af82e5777ff10f584d4de38e56712976a3c27dc57b0533",
    "history_sha256": "51b36aaf1e7f76d86ba408042a91ae55a6e5cd4c5c124e67a6a64f65c4f42da5"
  },
  {
    "run": "bimanual_box_bounded_spatial_mix_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "adf19c177407d1ec8c849e0f7dd4cba9d901d7cd1df307aa9735578758a40af3",
    "history_sha256": "cfe2d9f8a07c9f08d3217454d02cbd00f62a0cef960a601e0f2cb60cf174e5f0"
  },
  {
    "run": "bimanual_box_bounded_spatial_mix_s2",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "62ea85b4a3deddcdcf91b8f05ada336fdf78781677846e4a56a1e64c4c1af347",
    "history_sha256": "12136fb8866fe46b7d7a914bd7f4ad8bf8bfa2c84fbee5897b95caea2ebdadcd"
  },
  {
    "run": "bimanual_rope_anchored_additive_s0",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "c960963c54d3447b29f8d8d9437d406c11dd896ccc342ea8ec15c9962738c4aa",
    "history_sha256": "caf3f5db6e8d101c8f445fa47bcd1a23e38428ddb2ed8dda2c2b26b2b8ab303e"
  },
  {
    "run": "bimanual_rope_anchored_additive_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "0ddc66f68e9e52c341e14490a2251da0ae7efdf0090e74260c7de0b36d793a6e",
    "history_sha256": "de5984429384e62d213903bebf3859bd6295774a67f8f388f9c94bc3ebc289db"
  },
  {
    "run": "bimanual_rope_anchored_additive_s2",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "39dea66e302732a9a4f9fa419d573d48e917cff517f46fc91586db04042b24f7",
    "history_sha256": "b46ca19b64c27912c471ebe32debaab4d9d3bcd63cda8a9db388659bb6dcf38f"
  },
  {
    "run": "bimanual_rope_autoregressive_s0",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "4c302f171827635d395698206286e8dd2f4b33c5a5fa6b039b2497df43ebac03",
    "history_sha256": "9db63c54cbc13d0505299522d49e42c2ac748957e7cd8c62b570958bef55d709"
  },
  {
    "run": "bimanual_rope_autoregressive_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "9db6ce2da901ceb4405361d7cbb7bf5c07210a9a84d7f380b3c591d2ace3ff67",
    "history_sha256": "933e7cbdd5fedd065be442dc06699bfc806ec5ffbe9a50a29ad86db2c28c9664"
  },
  {
    "run": "bimanual_rope_autoregressive_s2",
    "selected_epoch": 29,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": false,
    "summary_sha256": "ae7db99af67f1e678c9be797642aea5424156ab6332f39ccecef6f7555f9a47f",
    "history_sha256": "3b13ed27925b75f8d6fcbfd81a87354ef0709940a33dd65b145c20408ea32ec0"
  },
  {
    "run": "bimanual_rope_bounded_spatial_mix_s0",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "7843663e886fb87bf4b7e0dae1181a87f4238eff3f8848d10b780e6270c9d5d5",
    "history_sha256": "b2a03e9a1ecad215d28623c046cc6ebcdefaa301a68f4d62c2699f345a8f8384"
  },
  {
    "run": "bimanual_rope_bounded_spatial_mix_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "33947bedd21045f99ca07defa662552da8ee3f66329e24314dfff434f4c59f62",
    "history_sha256": "247a8d58d7f075cf07585ec2d2d71015821ed1ac501063f93959ded4e00ee140"
  },
  {
    "run": "bimanual_rope_bounded_spatial_mix_s2",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "49418700496f5e5ddf20ae9a7201f371f86e75b3b241737ddccb1c8f8c5d3f6b",
    "history_sha256": "9f5862cf75859c008f33581a9aee191b66ab83bdc2c36a0ab5632d929a22088d"
  },
  {
    "run": "pusht_anchored_additive_s0",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "8a7dbc713a3f9cf5c53f05a54d663c68b639a1f69b2ad15dba03f43a60e02fa3",
    "history_sha256": "b2353c0ccf1b2aa2efbba710700feca60d7491d74ac0dc9785c2eb21d0ea9acd"
  },
  {
    "run": "pusht_anchored_additive_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "ff6f011fb79e72d927a53980b9b4d9b88f0bf6efc44439e2fb2d7fa4604291f9",
    "history_sha256": "e52ef0e9e1d68d4ee09b5078ea76a9c46ad8c469e0fe99d3762df6b27c5bd8c5"
  },
  {
    "run": "pusht_anchored_additive_s2",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "5ac797ec6c9226b4e971aaecb15d87f20e3f5d95987c80bb85636c5379876ed8",
    "history_sha256": "6b975121a4e1eb3c1f458b13175a1699d7cfdc1075b5aed043af74d7a5ebbce3"
  },
  {
    "run": "pusht_autoregressive_s0",
    "selected_epoch": 25,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": false,
    "summary_sha256": "970b25361889ab4b3c8695c0b9bbee6f299e6201cd9f0755d2c19bd76010ce00",
    "history_sha256": "7fa42f2f7d141b4ba74d1963195e9c5bc2af358c1983dabd9d3591e6081f029a"
  },
  {
    "run": "pusht_autoregressive_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": false,
    "summary_sha256": "f9ae2386e1db7d1f47379216c436f5cc2409b8bc7790152ad23adffcb531d856",
    "history_sha256": "895fd36f17c1613790f5df51c5a70c1b5852fb32321874ccb4c9be267901dd3d"
  },
  {
    "run": "pusht_autoregressive_s2",
    "selected_epoch": 26,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": false,
    "summary_sha256": "c36ea053fc9ce5c277791af1f7d86e5284b6d6e97e90925b81fd2d28934e1d2b",
    "history_sha256": "3730daa38b66155e47c0172e8fc6d367fb1d70d280785a13a7fb952cb95be814"
  },
  {
    "run": "pusht_bounded_spatial_mix_s0",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "f3216998a8a47df62317521d5bd4c6b6002f26022e6136e6ac1fdc4a3a773a3d",
    "history_sha256": "1eef3fd1c8b35d31f5ef02ea2d8904c371c8ccf42ad540b11e8a5e616ac34d15"
  },
  {
    "run": "pusht_bounded_spatial_mix_s1",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "8579f88e711d41b1316c69762698cc7cab99dce3cb8ff0257ac1d83ae385edf2",
    "history_sha256": "e3c60e4c269a858c46c8309a2d8f9665420e6405ba7ed5b3ac5f6c1ed7ca8b54"
  },
  {
    "run": "pusht_bounded_spatial_mix_s2",
    "selected_epoch": 30,
    "train_down_from_first": true,
    "val_down_from_first": true,
    "last5_train_strictly_decreasing": true,
    "last5_val_strictly_decreasing": true,
    "summary_sha256": "a863e4249693cdcc87b6d2a8f7c8b08cc78f29af3d5d66b48c99dd397625c34a",
    "history_sha256": "aa35177dda0e65c903aa32e2999b2661b94a181f7a26e8e5aa108922ad74b693"
  }
]
```
