#!/usr/bin/env python3
"""Run a registered seed's complete real-video training and evaluations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from shiftwm.real_video.data import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(0,1,2))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    campaign = json.loads((root / "configs/real_video/campaign.json").read_text())
    relevant = [row for row in campaign["runs"] if row["seed"] == args.seed]
    if len(relevant) != 4:
        raise ValueError("The registered campaign requires four modes per seed")
    sources = ["src/shiftwm/real_video/model.py", "src/shiftwm/real_video/data.py",
               "src/shiftwm/model.py", "src/shiftwm/upstream.py",
               "src/shiftwm/vendor/lewm/module.py", "src/shiftwm/vendor/lewm/NOTICE.json",
               "scripts/real_video/train.py", "scripts/real_video/evaluate.py",
               "scripts/real_video/run_campaign.py", "reports/real_droid_protocol.md",
               "configs/real_video/campaign.json"] + [row["config"] for row in relevant]
    identity = {"sources": {path:sha256(root/path) for path in sources}, "seed":args.seed}
    identity_path = root / "runs/real_video" / f"seed{args.seed}_campaign_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Registered code/config changed since campaign start")
    atomic_json(identity, identity_path)
    for row in relevant:
        for path, expected in identity["sources"].items():
            if sha256(root/path) != expected:
                raise ValueError("Code/config changed while campaign was running")
        config = json.loads((root/row["config"]).read_text())
        print(json.dumps({"event":"train", "run":row["name"]}),flush=True)
        subprocess.run([sys.executable, str(root/"scripts/real_video/train.py"),
                        "--config", str(root/row["config"]), "--resume-if-present"],cwd=root,check=True)
        for camera in ("exterior_image_1_left", "exterior_image_2_left"):
            for horizon in (5,10):
                destination = root/"results/real_video/droid_selected_v1"/row["name"]/camera/f"h{horizon}"/"results.json"
                print(json.dumps({"event":"evaluate", "run":row["name"], "camera":camera,"horizon":horizon}),flush=True)
                subprocess.run([sys.executable,str(root/"scripts/real_video/evaluate.py"),
                    "--checkpoint",str(root/config["output_dir"]/"best"),"--output",str(destination),
                    "--horizon",str(horizon),"--camera",camera],cwd=root,check=True)
    atomic_json({"status":"complete", "identity":identity,"runs":[row["name"] for row in relevant]},
                root/"runs/real_video"/f"seed{args.seed}_campaign_complete.json")


if __name__ == "__main__":
    main()
