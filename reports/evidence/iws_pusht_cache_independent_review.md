# Independent review: IWS PushT native-row cache

**Verdict: passed; no remaining blocker for registration and cache extraction.**
This is a data-preparation review, not approval of a prediction-window protocol,
training campaign, official evaluation or benchmark-performance claim.

I independently ran the 26 cache tests plus two frozen-split tests: **28 passed in
1.94 seconds**. Separate unequal-episode-length numerical probes matched direct
float64 channel moments (maximum mean error 1.46e-11 and standard-deviation error
8.60e-13), verified RGB scaling and exact channel-major spatial pooling, and
rejected development, missing, duplicate and reordered-row inputs.

The review found one blocking identity issue before completion: internal
self-consistency alone could accept a cache from another encoder or registration.
The final reader requires the complete current static contract, and its production
entry point verifies the immutable registration before opening the cache. Added
adversarial cases reject self-consistent wrong registration, encoder and config.

The final implementation preserves all N native frames and all N recorded
four-coordinate command rows. Reserved validation identities are rejected before
payload resolution. Normalization uses only internal training records. Per-episode
arrays and receipts become visible together through atomic rename; incomplete or
corrupt committed outputs cannot be treated as valid. The complete manifest is
written only after package/statistics checks and a final source identity guard.

The author's actual-data compatibility receipt is source-bound and consistent:
internal-training episode000011 decoded all200 rows, commands had shape200x4, and
first/last CPU DINO features matched the literal frozen spatial recipe exactly.
I verified the receipt and its source hashes; I did not repeat that real-data
inference or inspect additional recordings. The registration wording correctly
says *before full-cache extraction*, since two training frames were already encoded
for this compatibility check.

The earlier preview of000010 remains disclosed as internal development. Native
row retention does not resolve the upstream horizon60 convention; that still
needs a separate reviewed training/evaluation protocol. No held-out payload was
opened by this independent review, no GPU job was submitted, and no new model or
performance result is claimed. The JSON companion binds all reviewed source
hashes, test evidence, numerical probes and limitations.
