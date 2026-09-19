"""Inference-only export and a new physically relocated, offline CPU proof."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shiftwm.real_video.data import sha256
from shiftwm.real_video_spatial.data import SpatialDataset
from shiftwm.real_video_spatial_components.model import PACKAGE_KIND

SOURCES = ("shiftwm/__init__.py", "shiftwm/model.py", "shiftwm/upstream.py",
           "shiftwm/real_video_spatial/__init__.py", "shiftwm/real_video_spatial/model.py",
           "shiftwm/real_video_spatial_components/__init__.py", "shiftwm/real_video_spatial_components/model.py",
           "shiftwm/vendor/lewm/module.py", "shiftwm/vendor/lewm/NOTICE.json")


def train_module():
    spec = importlib.util.spec_from_file_location("component_parity_private_train", Path(__file__).with_name("train.py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def export_and_check(config, destination):
    train = train_module()
    source = ROOT / config["output_dir"] / "best"
    path, state = train.read_package(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("model.pt", "config.json"):
        shutil.copy2(path / name, destination / name)
    train.atomic_json({"format_version": 1, "package_kind": PACKAGE_KIND,
        "files": {n: sha256(destination / n) for n in ("model.pt", "config.json")}}, destination / "package_manifest.json")
    item = SpatialDataset(ROOT / config["cache_root"], "val", horizon=10, stride=5)[0]
    inputs = {"support": item["features"][:3][None], "past": item["actions"][:2][None], "future": item["actions"][2:][None]}
    torch.set_num_threads(2)
    model, _ = train.load_package(source, "cpu")
    with torch.inference_mode():
        expected = model.predict(inputs["support"], inputs["past"], inputs["future"]).numpy()
    with tempfile.TemporaryDirectory(prefix="spatial-components-relocation-") as directory:
        directory = Path(directory)
        shutil.copytree(destination, directory / "weights")
        for name in SOURCES:
            target = directory / "src" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "src" / name, target)
        np.savez(directory / "inputs.npz", **{k: v.numpy() for k, v in inputs.items()})
        (directory / "check.py").write_text('''import json,sys,socket,hashlib
from pathlib import Path
import numpy as np,torch
def deny(*args,**kwargs): raise RuntimeError("Network forbidden by offline parity guard")
socket.socket.connect=deny; socket.socket.connect_ex=deny; socket.create_connection=deny
sys.path.insert(0,str(Path("src").resolve()))
from shiftwm.real_video_spatial_components.model import from_config,PACKAGE_KIND
torch.set_num_threads(2)
p=Path("weights"); manifest=json.loads((p/"package_manifest.json").read_text())
assert manifest["package_kind"]==PACKAGE_KIND
assert set(manifest["files"])=={"model.pt","config.json"}
for name,digest in manifest["files"].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
c=json.loads((p/"config.json").read_text()); m=from_config(c)
s=torch.load(p/"model.pt",map_location="cpu",weights_only=True)
assert s["config"]==c
m.load_state_dict(s["state_dict"],strict=True); m.eval()
x=np.load("inputs.npz",allow_pickle=False)
with torch.inference_mode(): y=m.predict(*(torch.from_numpy(x[k]) for k in ("support","past","future")))
np.save("output.npy",y.numpy())
''')
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        subprocess.run([sys.executable, "-I", "check.py"], cwd=directory, env=env, check=True, capture_output=True)
        actual = np.load(directory / "output.npy", allow_pickle=False)
        np.testing.assert_array_equal(actual, expected)
    return {"status": "passed", "max_abs_error": float(np.max(np.abs(actual-expected))),
            "device": "cpu", "input_split": "original_validation", "input": "first eligible window; predictions only",
            "selected_checkpoint_sha256": sha256(source / "model.pt"),
            "package_manifest_sha256": sha256(destination / "package_manifest.json"),
            "source_sha256": {name: sha256(ROOT / "src" / name) for name in SOURCES},
            "mechanism": "physical copy; Python -I; local vendored source; HF/Transformers offline; Python socket guard (not OS network isolation)"}
