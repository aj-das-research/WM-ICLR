"""Synthetic completion-gate fixtures only; no datasets/checkpoint reads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('unbounded_release_build_tests', Path(__file__).with_name('build.py'))
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

class CompletionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.final = self.root / 'final.json'
        reg = self.root / 'configs/real_video_iws_unbounded/registration_v1.json'
        reg.parent.mkdir(parents=True); reg.write_text('{}')
        source = self.root / 'source.py'; source.write_text('# synthetic test only\n')
        rows = [{'name':f'{t}_{m}_s{s}', 'mode':m, 'completed_epochs':30}
                for t in ('pusht','bimanual_box','bimanual_rope') for s in range(3)
                for m in ('autoregressive','anchored_additive','bounded_spatial_mix','unbounded_spatial_mix')]
        self.registration = {'runs':[r for r in rows if r['mode']=='unbounded_spatial_mix']}
        self.value = {'status':'passed','completed_new_runs':9,'completed_v1_comparator_runs':27,
                      'official_validation_payloads_read':0,'scope':'exploratory_internal_development_after_v1',
                      'registration_sha256':build.sha(reg),'per_run':rows,
                      'source_dependencies':{'source.py':build.sha(source)}}
        self.commit()
        self.patcher = patch.multiple(build, ROOT=self.root, FINALIZATION=self.final)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def commit(self): self.final.write_text(json.dumps(self.value))
    def reject(self):
        self.commit()
        with self.assertRaises(ValueError): build.completed_study(self.registration)
    def test_complete_grid(self): self.assertEqual(len(build.completed_study(self.registration)),9)
    def test_missing_finalization(self):
        self.final.unlink()
        with self.assertRaises(FileNotFoundError): build.completed_study(self.registration)
    def test_partial_completion(self): self.value['completed_new_runs']=8; self.reject()
    def test_duplicate_comparator(self): self.value['per_run'][0]=self.value['per_run'][1]; self.reject()
    def test_short_training(self): self.value['per_run'][0]['completed_epochs']=29; self.reject()
    def test_wrong_registration(self): self.value['registration_sha256']='0'*64; self.reject()
    def test_reserved_access(self): self.value['official_validation_payloads_read']=1; self.reject()
    def test_stale_source(self): (self.root/'source.py').write_text('modified'); self.reject()
    def test_no_bound_sources(self): self.value['source_dependencies']={}; self.reject()
    def test_no_replace_receipt(self):
        with self.assertRaises(ValueError): build.write(self.final,{})
    def test_destination_normalized_inside_release_root(self):
        target = self.root/'artifacts/releases/new-bundle'
        self.assertEqual(build.release_destination(target), target.resolve())
    def test_destination_parent_escape_rejected(self):
        with self.assertRaises(ValueError):
            build.release_destination(self.root/'artifacts/releases/../outside')
    def test_destination_symlink_escape_rejected(self):
        directory = self.root/'artifacts/releases'; directory.mkdir(parents=True)
        (directory/'linked').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): build.release_destination(directory/'linked/outside')
    def test_destination_dangling_symlink_rejected(self):
        directory = self.root/'artifacts/releases'; directory.mkdir(parents=True)
        (directory/'broken').symlink_to(self.root/'missing')
        with self.assertRaises(ValueError): build.release_destination(directory/'broken')

if __name__ == '__main__': unittest.main()
