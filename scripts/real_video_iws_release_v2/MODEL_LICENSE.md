# Distribution terms for this inference bundle

The project-authored trained predictor tensors, wrapper code and configurations
are distributed under the project MIT license in `LICENSE`. This follows the
project's existing learned-predictor distribution terms. Attribution must retain
the copyright and permission notice. No warranty of accuracy, fitness or safety
is provided.

The unchanged vendored LeWM source retains its own MIT license and provenance
in `src/shiftwm/vendor/lewm/{LICENSE,NOTICE.json}`. The source hashes and pinned
revision are preserved. DINOv2-small, used for feature extraction, is Apache-2.0
at the revision recorded in `preprocessing.json`; no DINOv2 encoder weights are
included. No RLA-WM model weights are included.

These terms do not relicense third-party training data. The saved RLA-WM dataset
metadata lacked an explicit dataset-license field; a separate RLA-WM model-card
license does not establish dataset-example redistribution rights. This bundle
contains no genuine image, dataset feature/command fixture or future target.
Its public synthetic fixtures are generated independently by the included code.
