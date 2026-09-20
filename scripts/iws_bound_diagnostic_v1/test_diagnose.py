"""Synthetic analytical/gating checks; no actual feature/target payload reads."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

P=Path(__file__).with_name('diagnose.py')
spec=importlib.util.spec_from_file_location('bound_diagnostic_tests',P)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def inputs():
 return np.zeros(6144,np.float32),np.zeros((59,6144),np.float32),np.zeros(6144,np.float32),np.ones(6144,np.float32)


def test_zero_and_boundary_are_not_violations():
 x,t,m,s=inputs();t[0]=1.;t[1]=-1.
 out=d.envelope_diagnostics(x,t,m,s)
 assert all(np.array_equal(v,np.zeros(59)) for v in out.values())


def test_exact_fraction_and_mse_floor_single_coordinate():
 x,t,m,s=inputs();t[0,17]=3.;t[1,32]=-2.
 out=d.envelope_diagnostics(x,t,m,s)
 assert out['coordinate_violation_fraction'][0]==1/6144
 assert out['relaxed_mse_lower_bound'][0]==4/6144
 assert out['relaxed_mse_lower_bound'][1]==1/6144
 assert out['lower_coordinate_violation_fraction'][1]==1/6144
 assert out['upper_coordinate_violation_fraction'][0]==1/6144
 assert out['any_coordinate_violation'][:2].tolist()==[1,1]
 assert np.all(out['roundoff_screened_relaxed_mse_lower_bound']<=out['relaxed_mse_lower_bound'])


def test_channel_major_patch_axis_and_positive_shared_scale():
 x,t,m,s=inputs();m[:16]=8.;s[:16]=2.;x[:16]=np.arange(16,dtype=np.float32)*2+8
 t[:,:16]=np.tile(x[:16],(59,1));t[0,0]=40. # normalized16, upper observed15+1: boundary
 t[1,0]=42. # normalized17: distance1
 out=d.envelope_diagnostics(x,t,m,s)
 assert out['relaxed_mse_lower_bound'][0]==0
 assert out['relaxed_mse_lower_bound'][1]==1/6144
 assert out['coordinate_violation_fraction'][1]==1/6144


def test_real_arithmetic_box_distance_is_attained_by_coordinate_projection():
 rng=np.random.default_rng(13)
 x,t,m,s=inputs();x[:]=rng.normal(size=6144);t[:]=rng.normal(scale=4,size=t.shape)
 z=x.reshape(384,16).astype(np.float64);lo=z.min(1)[:,None]-1;hi=z.max(1)[:,None]+1
 projection=np.clip(t.reshape(59,384,16).astype(np.float64),lo,hi)
 expected=((projection-t.reshape(59,384,16))**2).mean((1,2))
 out=d.envelope_diagnostics(x,t,m,s)
 np.testing.assert_allclose(out['relaxed_mse_lower_bound'],expected,rtol=2e-15,atol=2e-15)
 # Projection is feasible for the BOX; no claim it is attainable by shared T/g.


def test_fp32_normalization_matches_frozen_model_method():
 from shiftwm.real_video_iws.model import SingleObservationWorldModel
 rng=np.random.default_rng(5);x,t,m,s=inputs()
 m[:]=np.repeat(rng.normal(size=384).astype(np.float32),16)
 s[:]=np.repeat(rng.uniform(.05,4,size=384).astype(np.float32),16);x[:]=rng.normal(size=6144)
 fake=SimpleNamespace(feature_mean=torch.from_numpy(m),feature_std=torch.from_numpy(s))
 actual=SingleObservationWorldModel.normalize_features(fake,torch.from_numpy(x)).numpy()
 np.testing.assert_array_equal(actual,((torch.from_numpy(x)-fake.feature_mean)/fake.feature_std).numpy())
 d.envelope_diagnostics(x,t,m,s)  # also verifies the equivalent raw envelope.


def test_raw_identity_at_std_floor_with_large_mean_does_not_change_primary_floor():
 rng=np.random.default_rng(19)
 mean=np.repeat(rng.uniform(-10,10,384).astype(np.float32),16)
 std=np.full(6144,1e-5,np.float32)
 initial=(mean+rng.normal(size=6144)*std).astype(np.float32)
 targets=(mean+4*rng.normal(size=(59,6144))*std).astype(np.float32)
 out=d.envelope_diagnostics(initial,targets,mean,std)
 z=((torch.from_numpy(initial)-torch.from_numpy(mean))/torch.from_numpy(std)).numpy().reshape(384,16).astype(np.float64)
 target=(targets.astype(np.float64)-mean.astype(np.float64))/std.astype(np.float64)
 lo=np.repeat(z.min(1)-1,16);hi=np.repeat(z.max(1)+1,16)
 distance=np.maximum(np.maximum(lo-target,target-hi),0)
 np.testing.assert_array_equal(out['relaxed_mse_lower_bound'],np.square(distance).mean(1))


def test_frozen_mixer_output_is_inside_envelope_with_nonzero_correction():
 from shiftwm.real_video_iws.model import SingleObservationWorldModel,IWSConfig
 torch.set_num_threads(2);torch.manual_seed(11)
 model=SingleObservationWorldModel(IWSConfig(hidden_dim=6,depth=1),np.zeros(6144),np.ones(6144),np.zeros(4),np.ones(4)).eval()
 with torch.no_grad():
  model.output_projection[-1].weight.normal_(0,3);model.output_projection[-1].bias.normal_(0,2)
  initial=torch.randn(2,6144);pred,detail=model._predict_normalized(initial,torch.randn(2,3,4),True)
  anchor=model.tokens(initial);lo=anchor.amin(1)[:,None]-1;hi=anchor.amax(1)[:,None]+1
  for row in detail:
   assert torch.all(row['prediction']>=lo-2e-6) and torch.all(row['prediction']<=hi+2e-6)
   assert torch.allclose(row['transport'].sum(-1),torch.ones(2,16),atol=2e-6)
   assert (row['gate']>=0).all() and (row['gate']<=1).all()
   assert row['innovation'].abs().max()<=1


@pytest.mark.parametrize('kind',['nonshared','zero_std','nonfinite','wrong_dtype','wrong_horizon'])
def test_bad_layout_or_normalization_fails(kind):
 x,t,m,s=inputs()
 if kind=='nonshared':s[0]=2
 if kind=='zero_std':s[:16]=0
 if kind=='nonfinite':t[1,1]=np.nan
 if kind=='wrong_dtype':t=t.astype(np.float64)
 if kind=='wrong_horizon':t=t[:-1]
 with pytest.raises(ValueError):d.envelope_diagnostics(x,t,m,s)


def test_no_review_denies_access_before_any_dependency_hash(tmp_path,monkeypatch):
 monkeypatch.setattr(d,'sha',lambda _:pytest.fail('Must reject before any hashing/payload access'))
 with pytest.raises(ValueError,match='Independent source review'):d.checked_registration(tmp_path)


def test_review_registration_and_source_tamper_fail_closed(tmp_path):
 report=tmp_path/d.REPORT;report.mkdir(parents=True)
 source=tmp_path/'frozen.py';source.write_text('original')
 reg={'schema':'iws_bound_diagnostic_registration_v1','protocol':d.PROTOCOL,'dependencies':{'frozen.py':d.sha(source)}}
 path=tmp_path/d.REG;path.write_text(json.dumps(reg));review=report/'source_review.json'
 review.write_text(json.dumps({'status':'passed','registration_sha256':'0'*64}))
 with pytest.raises(ValueError,match='approval'):d.checked_registration(tmp_path)
 review.write_text(json.dumps({'status':'passed','registration_sha256':d.sha(path)}))
 assert d.checked_registration(tmp_path)==reg
 source.write_text('changed')
 with pytest.raises(ValueError,match='Changed registered'):d.checked_registration(tmp_path)


def test_unsafe_path_and_reserved_identity_denied():
 for p in ('../private','/tmp/absolute'):
  with pytest.raises(ValueError):d.local(p)
 from shiftwm.real_video_iws.windows import eligible_starts
 assert eligible_starts(60)==[]
 assert eligible_starts(61)==[0]
 assert eligible_starts(66)==[0,5]


def test_payload_allowlist_rejects_train_reserved_and_model_reads(tmp_path):
 allowed={'data/features/dev/episodes/000010/arrays.npz'}
 d.check_payload_path(tmp_path/'data/features/dev/episodes/000010/arrays.npz',allowed,tmp_path)
 for relative in ('data/features/dev/episodes/000011/arrays.npz','data/features/iws_reserved_v1/pusht/arrays.npz',
                  'data/real_video/episode/metadata.h5','data/real_video/episode/camera.mp4','runs/best/model.pt'):
  with pytest.raises(ValueError):d.check_payload_path(tmp_path/relative,allowed,tmp_path)
