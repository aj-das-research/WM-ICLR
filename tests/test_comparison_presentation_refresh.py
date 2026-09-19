"""Operational pre-publication guards; fixtures contain no research results."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/publishing/refresh_comparison_presentations.py"
spec = importlib.util.spec_from_file_location("comparison_refresh", SOURCE)
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


class PresentationRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / ".venv/bin").mkdir(parents=True)
        (self.root / ".venv/bin/python").symlink_to(sys.executable)
        (self.root / "frozen.txt").write_text("immutable scientific source\n")
        (self.root / "registration.json").write_text(json.dumps({"dependencies": {
            "frozen.txt": refresh.sha(self.root / "frozen.txt")}}))
        self.registrations = ("registration.json",)
        self.renderer = self.root / "renderer.py"
        self.renderer.write_text("from pathlib import Path\np=Path('paper/generated/benchmark_scorecards/test.tex')\np.parent.mkdir(parents=True,exist_ok=True)\np.write_text('test fixture only')\nprint('{\"status\": \"complete\"}')\n")
        self.renderers = (("renderer.py",),)

    def run_refresh(self, **kwargs):
        return refresh.refresh(self.root, self.renderers, self.registrations, **kwargs)

    def test_missing_program_fails_before_running_any(self):
        self.renderers += (("missing.py",),)
        with self.assertRaisesRegex(FileNotFoundError, "missing.py"):
            self.run_refresh()
        self.assertFalse((self.root / "paper/generated").exists())

    def test_changed_scientific_source_fails_before_render(self):
        (self.root / "frozen.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Frozen dependency changed"):
            self.run_refresh()
        self.assertFalse((self.root / "paper/generated").exists())

    def test_renderer_failure_never_commits_success_receipt(self):
        self.renderer.write_text("raise RuntimeError('invalid finalizer fixture')\n")
        with self.assertRaisesRegex(RuntimeError, "invalid finalizer fixture"):
            self.run_refresh()
        self.assertFalse((self.root / refresh.RECEIPT).exists())

    def test_mutation_during_render_rejected(self):
        with self.renderer.open("a") as stream:
            stream.write("Path('frozen.txt').write_text('mutated')\n")
        with self.assertRaisesRegex(ValueError, "Frozen dependency changed"):
            self.run_refresh()
        self.assertFalse((self.root / refresh.RECEIPT).exists())

    def test_preflight_does_not_render(self):
        self.assertEqual(self.run_refresh(check_only=True)["status"], "preflight_passed")
        self.assertFalse((self.root / "paper/generated").exists())

    def test_pending_plot_has_no_artifact_and_is_recorded(self):
        plot = self.root / "plot.py"
        plot.write_text("print('{\"status\": \"pending\", \"outputs_written\": false}')\n")
        self.renderers += (("plot.py", "--if-ready"),)
        self.run_refresh()
        data = json.loads((self.root / refresh.RECEIPT).read_text())
        self.assertEqual(data["renderer_gate_states"]["plot.py"], "pending")
        self.assertFalse((self.root / "paper/generated/iws_variant_forecasts").exists())

    def test_zero_exit_without_structured_status_cannot_commit(self):
        self.renderer.write_text("print('not a completion receipt')\n")
        with self.assertRaisesRegex(ValueError, "structured completion status"):
            self.run_refresh()
        self.assertFalse((self.root / refresh.RECEIPT).exists())

    def test_complete_plot_requires_validated_caption_reference(self):
        plot = self.root / refresh.VARIANT_PLOT
        plot.parent.mkdir(parents=True)
        plot.write_text("print('{\"status\": \"complete_9_plus_27\"}')\n")
        self.renderers += ((refresh.VARIANT_PLOT, "--if-ready"),)
        with self.assertRaises(FileNotFoundError):
            self.run_refresh()
        self.assertFalse((self.root / refresh.RECEIPT).exists())
        table = self.root / refresh.COMPONENT_TEX
        table.parent.mkdir(parents=True)
        table.write_text(r"\label{tab:iws-unbounded-component}")
        (self.root / refresh.COMPONENT_JSON).write_text(json.dumps({
            "status": "complete_validated_exploratory_development",
            "official_validation_payloads_read": 0, "tex_sha256": refresh.sha(table)}))
        self.assertEqual(self.run_refresh()["status"], "refreshed")
        table.write_text("tampered table")
        with self.assertRaisesRegex(ValueError, "validated paired-comparison table"):
            self.run_refresh()

    def test_production_order_refreshes_table_between_pack_and_plot(self):
        paths = [row[0] for row in refresh.RENDERERS]
        self.assertLess(paths.index("paper/scripts/render_current_real_scorecards.py"),
                        paths.index("paper/scripts/render_iws_unbounded_results.py"))
        self.assertLess(paths.index("paper/scripts/render_iws_unbounded_results.py"),
                        paths.index(refresh.VARIANT_PLOT))
        self.assertLess(paths.index(refresh.VARIANT_PLOT),
                        paths.index("paper/scripts/render_comparison_inventory.py"))

    def test_success_receipt_is_idempotent_and_binds_outputs(self):
        self.assertEqual(self.run_refresh()["status"], "refreshed")
        receipt = self.root / refresh.RECEIPT
        before = receipt.stat().st_mtime_ns
        self.assertEqual(self.run_refresh()["status"], "unchanged_verified")
        self.assertEqual(receipt.stat().st_mtime_ns, before)
        data = json.loads(receipt.read_text())
        for name, digest in data["outputs"].items():
            self.assertEqual(refresh.sha(self.root / name), digest)
        self.assertFalse(data["training_or_evaluation_invoked"])


if __name__ == "__main__":
    unittest.main()
