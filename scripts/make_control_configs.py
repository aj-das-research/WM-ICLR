#!/usr/bin/env python3
"""Add controls required by the independent implementation review.

The original plain model is retained as an unaligned diagnostic. Framewise
calibration provides a consistent non-temporal planning baseline. The unpaired
factorized model separates architectural effects from pairing supervision.
"""
import json
from pathlib import Path

root = Path("configs/world")
core = json.loads((root / "training_campaign.json").read_text())["tasks"]
controls = []
for seed in (0, 1, 2):
    for environment in ("reacher", "pusht"):
        for mode in ("framewise", "factorized_unpaired"):
            name = f"{environment}_{mode}_s{seed}"
            config = json.loads((root / f"{environment}_factorized_s{seed}.json").read_text())
            config["model"]["mode"] = mode
            config["output_dir"] = f"runs/world/{name}"
            path = root / f"{name}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            controls.append({"id": name, "config": str(path)})
(root / "control_campaign.json").write_text(json.dumps({"tasks": controls}, indent=2) + "\n")
(root / "full_campaign.json").write_text(json.dumps({"tasks": core + controls}, indent=2) + "\n")
print(f"Wrote {len(controls)} controls; combined campaign has {len(core) + len(controls)} complete runs")
