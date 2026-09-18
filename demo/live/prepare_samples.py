"""Bundle only the three previously published DROID illustrative episodes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ledger_path = ROOT / "paper/generated/real_video/comparison_recorded_droid_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    cache = ROOT / "data/features/droid_selected_v1"
    manifest = json.loads((cache / "manifest.json").read_text())
    episodes = {r["episode_id"]: r for r in manifest["episodes"]}
    output = HERE / "samples"
    output.mkdir(exist_ok=True)
    rows = []
    for selected in ledger["selected"]:
        episode_id = selected["score"]["episode_id"]
        episode = episodes[episode_id]
        assert episode["split"] == "test"
        cameras = {}
        for camera in (1, 2):
            name = f"exterior_image_{camera}_left"
            feature_source = cache / episode["cameras"][name]["file"]
            assert sha(feature_source) == episode["cameras"][name]["sha256"]
            rgb_source = ROOT / f"data/real_video/droid_selected/processed/cameras/{name}/{episode_id}.npz"
            assert sha(rgb_source) == episode["cameras"][name]["source_sha256"]
            with np.load(feature_source, allow_pickle=False) as source:
                features, actions, indices = source["features"][:13], source["actions"][:12], source["frame_indices"][:13]
                assert features.shape == (13, 1536) and actions.shape == (12, 35)
            bundle = output / f"{episode_id}_c{camera}.npz"
            np.savez_compressed(bundle, features=features, actions=actions, frame_indices=indices)
            frames = []
            with np.load(rgb_source, allow_pickle=False) as source:
                assert np.array_equal(source["frame_indices"][:13], indices)
                for i, rgb in enumerate(source["images"][:13]):
                    path = output / f"{episode_id}_c{camera}_f{i:02}.jpg"
                    Image.fromarray(rgb).save(path, quality=92)
                    frames.append({"file": path.name, "sha256": sha(path), "native_frame": int(indices[i])})
            cameras[str(camera)] = {"file": bundle.name, "sha256": sha(bundle), "frames": frames,
                "source_feature_sha256": sha(feature_source), "source_recording_sha256": sha(rgb_source)}
        rows.append({"id": episode_id, "selection_label": selected["title"],
                     "session": episode["session_id"], "window_start": 0, "cameras": cameras})
    record = {"format_version": 1, "selection": "Three previously published best, median, and regression episodes; selected by episode-mean three-seed h5 difference. Demo runs their first window only, so its scores differ from episode averages.",
        "source_ledger_sha256": sha(ledger_path), "feature_manifest_sha256": sha(cache / "manifest.json"),
        "license": "DROID dataset: CC BY 4.0", "attribution": "Khazatsky et al., DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (2024). https://droid-dataset.github.io/",
        "samples": rows}
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"samples": len(rows), "cameras": 2, "bytes": sum(p.stat().st_size for p in output.iterdir())}))


if __name__ == "__main__":
    main()
