"""Portable loader boundary tests; no training/dataset/checkpoint reads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("iws_release_runtime", Path(__file__).with_name("runtime.py"))
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class LoaderBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.document = {"schema": "iws_unbounded_local_inference_bundle_v1", "package_kind": runtime.KIND,
                         "benchmark_finalization_claimed": False,
                         "models": [{"name": f"{t}_{m}_s{s}", "directory": f"models/{t}_{m}_s{s}",
                                     "task": t, "mode": m, "seed": s,
                                     "fixture": f"fixtures/{t}.npz",
                                     "expected_prediction": f"fixtures/{t}_{m}_s{s}_prediction.npz"}
                                    for t in ("pusht", "bimanual_box", "bimanual_rope")
                                    for m in ("unbounded_spatial_mix",)
                                    for s in range(3)],
                         "files": {}}
        files = set(runtime.RUNTIME_FILES)
        for row in self.document["models"]:
            files.update(row["directory"] + "/" + n for n in ("model.pt", "config.json", "package_manifest.json"))
            files.update((row["fixture"], row["expected_prediction"]))
        for name in files:
            target = self.root / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture")
            self.document["files"][name] = runtime.sha(target)
        self.commit()

    def commit(self):
        (self.root / "manifest.json").write_text(json.dumps(self.document))

    def test_complete_identity_manifest(self):
        self.assertEqual(len(runtime.verify_bundle(self.root)["models"]), 9)

    def test_incomplete_grid_rejected(self):
        self.document["models"].pop(); self.commit()
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)

    def test_duplicate_model_rejected(self):
        self.document["models"][-1] = self.document["models"][0]; self.commit()
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)

    def test_foreign_package_kind_rejected(self):
        self.document["package_kind"] = "shiftwm_real_video_spatial_v1"; self.commit()
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)

    def test_mislabelled_arm_rejected(self):
        self.document["models"][0]["mode"] = "bounded_spatial_mix"; self.commit()
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)

    def test_changed_input_digest_rejected(self):
        (self.root / "fixtures/pusht.npz").write_text("changed")
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)

    def test_external_paths_rejected(self):
        for name in ("../escape", "/tmp/escape", "a\\b", "a/../fixture.txt", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.regular(self.root, name)

    def test_symlink_and_optimizer_state_rejected(self):
        (self.root / "alias").symlink_to(self.root / "README.md")
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)
        (self.root / "alias").unlink()
        (self.root / "training_state.pt").write_bytes(b"not for inference")
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)

    def test_unregistered_package_directory_rejected(self):
        with self.assertRaises(ValueError):
            runtime.load_model(self.root / "foreign", self.root)

    def test_omitted_source_hash_rejected(self):
        del self.document["files"]["src/shiftwm/real_video_iws_unbounded/model.py"]; self.commit()
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)

    def test_extra_unlisted_code_rejected(self):
        (self.root / "src/shiftwm/extra.py").write_text("raise RuntimeError('unlisted')")
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)

    def test_traversing_row_directory_rejected(self):
        self.document["models"][0]["directory"] = "../elsewhere"; self.commit()
        with self.assertRaises(ValueError):
            runtime.verify_bundle(self.root)


if __name__ == "__main__":
    unittest.main()
