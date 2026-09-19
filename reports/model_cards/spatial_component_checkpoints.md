# ShiftWM spatial component checkpoints

Six completed predictors: `bounded_additive_s0`–`s2` (Ours-3) and
`unbounded_transport_s0`–`s2` (Ours-4). Each trained for all 30 registered epochs;
selection uses the same original DROID development criterion as the frozen controls.
This local bundle has not yet been uploaded as a public checkpoint release.

Each `models/<name>/` contains inference weights, architecture and normalization
configuration, and a SHA256 manifest. Use the matching published repository source;
the checkpoint directories do not independently bundle Python dependencies.

From the repository's configured Python environment, with `PYTHONPATH=src`:

```python
from pathlib import Path
import hashlib
import json
import torch
from shiftwm.real_video_spatial_components.model import from_config

package = Path('artifacts/releases/spatial_components_v1/models/bounded_additive_s0')
manifest = json.loads((package / 'package_manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
config = json.loads((package / 'config.json').read_text())
checkpoint = torch.load(package / 'model.pt', map_location='cpu', weights_only=True)
assert checkpoint['config'] == config
model = from_config(config)
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()
# Actual inputs from the registered DROID spatial feature/action pipeline:
# support_features: [B, 3, 6144], channel-major DINOv2-small 384 x 4 x 4
# past_actions:     [B, 2, 35], chronological recorded command blocks
# future_actions:   [B, 10, 35], supplied recorded query command blocks
# predict() applies the configuration's train-fitted normalization internally.
with torch.inference_mode():
    future_features = model.predict(support_features, past_actions, future_actions)
```

Outputs are visual features, not decoded RGB frames or selected physical actions.
Use the exact recorded frame sampling, channel order and five-command block recipe
from `reports/real_video_development/spatial_components_protocol.md` and the frozen
spatial data loader. Different feature normalization or action units are not compatible.

All six predictors passed exact CPU prediction equality after a physical relocation
with the recorded source files, offline library flags and a Python socket guard
(not operating-system network isolation). An independent review repeated those checks. Evidence:
`reports/evidence/spatial_components_completed_independent_audit.json` and
`reports/real_video_spatial_components/finalization.json`.

The complete component comparison uses original development validation data after
previous controls were revealed. Mixing helps native-coordinate forecasting;
incremental native gains from bounding and all mixing-by-bounding interactions
remain inconclusive. All outcomes are in `results.json` and the manuscript appendix.
No independent-test, RGB-generation, physical-control or SOTA claim follows from this bundle.
