#!/usr/bin/env python3
"""Export small reusable calibration JSONs; neural base weights stay separate."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_residual import load_calibrated_package
from diagnose import development_dataset
from shiftwm.real_video.data import atomic_json, sha256


def export():
    campaign_root = ROOT / "runs/real_video_development/residual_calibration_v1"
    campaign = json.loads((campaign_root / "campaign.json").read_text())
    if campaign["status"] != "completed" or len(campaign["runs"]) != 12:
        raise ValueError("Require all matched calibrations")
    destination = ROOT / "configs/real_video_development/calibrations"
    if destination.exists():
        raise FileExistsError("Export destination already exists")
    destination.mkdir(parents=True)
    torch.set_num_threads(2)
    dataset = development_dataset(ROOT / "data/features/droid_selected_v1", "val", horizon=10, stride=5)
    batch = dataset[0]
    support = batch["features"][:3][None]
    past, future = batch["actions"][:2][None], batch["actions"][2:][None]
    records = []
    for run in campaign["runs"]:
        name = f"droid_{run['mode']}_s{run['seed']}"
        original = campaign_root / name / "calibration.json"
        package = json.loads(original.read_text())
        package["base_checkpoint"] = f"../../../artifacts/releases/real_droid_v1/models/{name}"
        target = destination / (name + ".json")
        atomic_json(package, target)
        reference = load_calibrated_package(original, device="cpu")
        relocated = load_calibrated_package(target, device="cpu")
        with torch.inference_mode():
            expected = reference.predict(support, past, future)
            actual = relocated.predict(support, past, future)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        records.append({"name": name, "calibration_json": str(target.relative_to(ROOT)), "sha256": sha256(target),
                        "base_model_sha256": package["base_checkpoint_sha256"], "scale": package["fit"]["scale"],
                        "offline_relocated_reload_max_abs_error": float((actual - expected).abs().max())})
    atomic_json({"status": "completed", "source_sha256": sha256(__file__), "campaign_sha256": sha256(campaign_root / "campaign.json"),
                 "test_evaluated": False, "records": records}, ROOT / "reports/evidence/real_droid_calibration_exports.json")
    print(json.dumps({"status": "completed", "exported": len(records), "destination": str(destination)}))


if __name__ == "__main__":
    export()
