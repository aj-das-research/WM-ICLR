"""Frozen-identity and test-only population adapters for fresh DROID evaluation."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRESH = ROOT / "data/real_video/droid_fresh_v1"
DECODED = FRESH / "decoded_confirmatory_v1"
CACHE = ROOT / "data/features/droid_fresh_confirmatory_v1"
OUTPUT = ROOT / "results/real_video_development/fresh_confirmatory_v1"
FREEZE = ROOT / "configs/real_video_development/fresh_evaluation_v1.json"
PROTOCOL = ROOT / "reports/real_droid_fresh_evaluation_protocol.md"
STATS = ROOT / "data/features/droid_selected_v1/training_statistics.json"
MODES = ("framewise", "constant_dynamics", "factorized", "action_free")
CAMERAS = ("exterior_image_1_left", "exterior_image_2_left")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_freeze():
    registration = json.loads(FREEZE.read_text())
    if registration["status"] != "frozen_before_fresh_image_decoding":
        raise ValueError("Fresh evaluation freeze is missing")
    if registration["primary"] != {"camera": CAMERAS[0], "horizon": 5,
                                    "metric": "h5_standardized_mse", "method": "calibrated/factorized",
                                    "reference": "calibrated/framewise"}:
        raise ValueError("Primary comparison changed")
    for path, expected in registration["dependencies"].items():
        if sha256(path) != expected:
            raise ValueError("Frozen dependency changed: " + path)
    return registration


def validate_population(manifest, expected):
    if manifest.get("status") != "complete" or not manifest.get("episodes"):
        raise ValueError("Incomplete fresh manifest")
    actual = [(row["episode_id"], row["session_id"]) for row in manifest["episodes"]]
    if len(set(actual)) != len(actual) or len({row[0] for row in actual}) != len(actual):
        raise ValueError("Duplicate fresh episode identity")
    if set(actual) != set(expected) or any(row["split"] != "test" for row in manifest["episodes"]):
        raise ValueError("Fresh population changed or includes a training/validation split")


def validate_fresh_manifest(manifest):
    metadata = json.loads((FRESH / "metadata_manifest.json").read_text())
    expected = [(row["episode_id"], row["session_id"]) for row in metadata["retained"]]
    validate_population(manifest, expected)


@contextmanager
def scoped_adapter(module, name, replacement):
    """Adapt one caller-owned module binding; restore even on a failed stage.

    No source is modified and no fake training/validation episodes are added.
    Only the all-three-splits guard and train-statistics callback are adapted.
    """
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)

