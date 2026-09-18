#!/usr/bin/env python3
"""Run actual CPU checkpoint inference and save reproducible demo validation."""
import json
from pathlib import Path
import tempfile
import os

from backend import HERE, checkpoint_catalog, forecast, samples, render_plot


def main():
    catalog = checkpoint_catalog()
    records = []
    selected = []
    for environment in ("pusht", "reacher"):
        name = next(name for name, value in catalog.items() if value["environment"] == environment and value["mode"] == "frozen")
        sample = next(name for name, value in samples().items() if value["environment"] == environment)
        selected.append((name, sample))
    # No reliance on cwd or the training repository's module search path.
    with tempfile.TemporaryDirectory(prefix="shiftwm-demo-") as temporary:
        previous = Path.cwd()
        os.chdir(temporary)
        try:
            for checkpoint, sample in selected:
                for appearance in (0, 2):
                    record, frames, metadata = forecast(checkpoint, sample, appearance)
                    assert frames.shape == (8, 224, 224, 3)
                    assert len(record["rows"]) == 5
                    assert all(row["recorded_action_mse"] >= 0 for row in record["rows"])
                    assert any(row["action_effect_mse"] > 1e-12 for row in record["rows"])
                    records.append(record)
        finally:
            os.chdir(previous)
    output = HERE / "validation"
    output.mkdir(exist_ok=True)
    evidence = {"status": "passed", "scope": "actual frozen-checkpoint CPU inference on two environments and two appearances, outside project cwd",
                "checks": ["weights-only portable load", "sample hashes", "eight actual RGB frames", "five finite forecast rows", "nonzero action sensitivity"],
                "records": records, "note": "These are demo integration measurements, not benchmark results or newly trained model validation."}
    (output / "inference.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    fig = render_plot(records[0])
    fig.savefig(output / "actual_forecast.png", dpi=180)
    print(json.dumps({"status": "passed", "real_inference_cases": len(records), "evidence": str(output / "inference.json")}))


if __name__ == "__main__":
    main()
