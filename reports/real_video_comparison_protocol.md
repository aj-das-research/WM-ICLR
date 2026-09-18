# Recorded DROID comparison figure: fixed selection protocol

Specified before the complete twelve-run / forty-eight-evaluation report exists.
This is a post-hoc descriptive figure protocol, not a new benchmark, checkpoint
selection rule or confirmatory hypothesis test. It uses the registered primary
population only: exterior camera 1, five future action blocks.

For each common held-out episode, average its saved h5 train-standardized
feature MSE across training seeds 0, 1 and 2, separately for ShiftWM (ours),
Framewise and Constant dynamics. Each saved episode metric already averages
all complete evaluated windows within that episode. Define difference as
ours minus Framewise. Select the smallest difference, largest difference,
and the remaining episode nearest the median difference; break ties by
episode ID. Use descriptive improvement/loss labels only if the measured
sign supports them. Never substitute a prettier recording or stronger result.

For each selected episode, show the last support image and final recorded
query target of its first evaluated window, using the actual stored native
frame indices. These two images illustrate the recording; they are not
model-generated predictions and are not alleged to explain the aggregate
error. Preserve the complete field of view and colors. No image synthesis,
object annotations, inferred action projections or reconstructed RGB frames.
The accompanying h1/h3/h5 curves are episode-average errors over all saved
windows and all three seeds, explicitly labeled as such. Connect only the
three available measured horizons with straight lines; do not invent h2/h4.

Give the whole primary-test population's lower/tied/higher h5-error episode
counts and equal-episode/equal-seed mean relative MSE reduction as context.
An exact tie is equality of the computed three-seed means. Keep the strongest
regression and median case visible alongside the strongest improvement.
Per-case ratios are descriptive; they are neither uncertainty intervals nor
real-robot success. The figure cannot identify a causal mechanism.

Rendering must fail closed until the finalizer has validated all 12 complete
30-epoch runs and 48 evaluations, committed its verified inference release,
and written the paper-source ledger. Check the report/release hashes against
that ledger, every raw-result hash, common episode/window identities, and
source RGB hashes. Record all episode scores, selected IDs, images, source
hashes, and exported figure hashes in a companion ledger.

Design: 5.5-inch width, three equal case rows; paired recorded images on the
left and measured horizon curves on the right. Short labels outside images;
minimum 8-point type. Use the installed paper-figure-creation skill. Inspect
paper-width, enlarged crops, grayscale and an isolated compiled figure proof.
