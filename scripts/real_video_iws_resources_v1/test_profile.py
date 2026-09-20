"""Resource-protocol boundaries; synthetic tensors only, no learned inference."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('iws_resource_tests',Path(__file__).with_name('profile.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class ProtocolTests(unittest.TestCase):
    def setUp(self): self.c=json.loads(p.CONFIG.read_text())
    def test_registered_protocol_shape(self): p.check_protocol(self.c)
    def test_no_short_horizon_batch_or_precision_change(self):
        for key,value in [('command_rows',15),('batch_size',2),('dtype','bfloat16'),('autocast',True),('tf32',True),('timed_calls_per_device',1),('warmup_calls_per_device',0),('torch_intraop_threads',8)]:
            with self.subTest(key=key):
                c=dict(self.c);c[key]=value
                with self.assertRaises(ValueError):p.check_protocol(c)
    def test_no_missing_device_or_mode(self):
        for key,value in [('devices',['cpu']),('learned_modes',['bounded_spatial_mix']),('seeds',[0,1])]:
            with self.subTest(key=key):
                c=dict(self.c);c[key]=value
                with self.assertRaises(ValueError):p.check_protocol(c)
    def grid(self):return [{'task':t,'mode':m,'seed':s} for t in p.TASK_WIDTHS for m in p.MODES for s in range(3)]
    def test_all36_grid(self):p.check_case_grid(self.grid())
    def test_missing_or_duplicate_model(self):
        rows=self.grid()
        for bad in [rows[:-1],rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):p.check_case_grid(bad)
    def test_fixture_all_native_widths(self):
        for task,width in p.TASK_WIDTHS.items():p.validate_fixture(np.zeros((1,6144),np.float32),np.zeros((1,60,width),np.float32),task)
    def test_fixture_rejects_wrong_shape_dtype_nan(self):
        goodz=np.zeros((1,6144),np.float32);gooda=np.zeros((1,60,4),np.float32)
        for z,a in [(goodz.astype('float64'),gooda),(goodz,gooda[:,:15]),(goodz[:,:3000],gooda),(goodz,np.full_like(gooda,np.nan))]:
            with self.assertRaises(ValueError):p.validate_fixture(z,a,'pusht')
    def test_latency_statistics_retain_all_samples(self):
        got=p.summarize(list(range(1,31)))
        self.assertEqual(got['median_ms'],15.5);self.assertEqual(got['samples_ms'],list(range(1,31)))
        self.assertAlmostEqual(got['p95_ms'],28.55)
    def test_latency_rejects_missing_or_nonfinite_samples(self):
        for values in [[1]*29,[1]*29+[0],[1]*29+[float('nan')]]:
            with self.assertRaises(ValueError):p.summarize(values)
    def test_both_sync_calls_are_inside_wall_measurement_boundary(self):
        events=[];ticks=iter([1000000,4500000])
        def sync():events.append('sync')
        def clock():events.append('clock');return next(ticks)
        def call():events.append('predict');return 'output'
        out,ms=p.timed_call(call,sync,clock)
        self.assertEqual(events,['sync','clock','predict','sync','clock'])
        self.assertEqual(out,'output');self.assertEqual(ms,3.5)
    def test_persistence_allocates_full_output(self):
        import torch
        z=torch.arange(6144,dtype=torch.float32).reshape(1,-1)
        got=p.persistence_forecast(z)
        self.assertEqual(list(got.shape),[1,59,6144]);self.assertTrue(got.is_contiguous())
        self.assertEqual(got.untyped_storage().nbytes(),1*59*6144*4)
        got[0,0,0]=-1
        self.assertEqual(float(z[0,0]),0);self.assertEqual(float(got[0,1,0]),0)
    def test_loaded_identity_matches_registered_label_and_checkpoint(self):
        case={'bundle':'bundle','directory':'models/model','name':'model'}
        identity={'name':'model','checkpoint_sha256':'selected'}
        bindings={'bundle/models/model/model.pt':'selected'}
        p.validate_loaded_identity(case,identity,bindings)
        for bad in [identity|{'name':'other'},identity|{'checkpoint_sha256':'other'}]:
            with self.assertRaises(ValueError):p.validate_loaded_identity(case,bad,bindings)
    def test_unsupported_device_not_reported_as_measured(self):
        row={'case':{'task':'pusht','mode':'persistence'},'parameters':{'total':0,'trainable':0},'devices':{'cuda':{'status':'unsupported'}}}
        self.assertEqual(p.aggregate([row]),[{'task':'pusht','mode':'persistence','device':'cuda','status':'unsupported'}])

if __name__=='__main__':unittest.main()
