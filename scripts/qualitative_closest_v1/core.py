"""Pure measured-array reductions. No model/data loading or illustration values."""
from __future__ import annotations
import numpy as np


def require(ok,message):
    if not ok:raise ValueError(message)


def selected_cases(group,focal):
    rows=group['all_ranked_episodes'];require(len(rows)>=3,'Need at least three episodes')
    require(len({r['episode_id'] for r in rows})==len(rows),'Duplicate episode')
    for row in rows:
        base=row['episode_endpoint_means']['autoregressive'];ours=row['episode_endpoint_means'][focal]
        require(np.isfinite([base,ours]).all() and base>0 and ours>=0,'Invalid gain denominator/metric')
        require(np.isclose(row['gain_percent'],100*(base-ours)/base,rtol=1e-12,atol=1e-12),'Episode gain differs')
    ranked=sorted(rows,key=lambda r:(-r['gain_percent'],r['episode_id']))
    result=[ranked[i] for i in (0,(len(rows)-1)//2,len(rows)-1)]
    saved=group['selected']
    require(len(saved)==3 and [(r['episode_id'],r['window_start']) for r in result]==[(r['episode_id'],r['window_start']) for r in saved],'Audited selected identities differ')
    return saved


def patch_error(prediction,target,std):
    for a in (prediction,target,std):require(np.isfinite(a).all(),'Nonfinite feature array')
    require(prediction.shape==target.shape and prediction.ndim==2 and prediction.shape[-1]==6144,'Expected Hx6144 features')
    require(std.shape==(6144,) and (std>0).all(),'Invalid fixed normalization')
    # FP32 arithmetic matches frozen evaluator; per-seed before averaging.
    return np.square((prediction.astype(np.float32)-target.astype(np.float32))/std.astype(np.float32)).reshape(-1,384,4,4).mean(1)


def measured_details(T,g,delta,prediction,anchor,bounded):
    h=len(prediction)
    for name,a,shape in [('T',T,(h,16,16)),('g',g,(h,16,1)),('delta',delta,(h,16,384)),('prediction',prediction,(h,16,384)),('anchor',anchor,(16,384))]:
        require(a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all(),'Invalid actual tensor '+name)
    require((T>=0).all() and (T<=1).all() and (g>=0).all() and (g<=1).all(),'Mixing probabilities outside[0,1]')
    np.testing.assert_allclose(T.sum(-1),1,rtol=2e-6,atol=2e-7)
    M=(1-g)*np.eye(16,dtype=np.float32)+g*T
    if bounded:require(np.abs(delta).max()<=1+1e-7,'Bounded correction exceeds unit bound')
    reconstructed=M.astype(np.float64)@anchor.astype(np.float64)+delta.astype(np.float64)
    np.testing.assert_allclose(prediction,reconstructed,rtol=5e-6,atol=2e-6)
    return M,{'reconstruction_max_abs':float(abs(prediction-reconstructed).max()),'T_row_sum_max_abs':float(abs(T.sum(-1)-1).max()),'bounded':bool(bounded),'max_abs_correction':float(abs(delta).max())}


def numeric_record(patches,offsets,T=None,M=None,g=None,delta=None):
    require(patches.ndim==3 and patches.shape[1:]==(4,4) and np.isfinite(patches).all() and (patches>=0).all(),'Invalid patch errors')
    values={}
    for offset in offsets:
        require(type(offset)is int and 1<=offset<=len(patches),'Invalid displayed offset')
        i=offset-1;row={'patch_mse4x4':patches[i].tolist()}
        if T is not None:
            row.update(T16x16=T[i].tolist(),M16x16=M[i].tolist(),gate4x4=g[i,:,0].reshape(4,4).tolist(),
                       correction_rms4x4=np.sqrt(np.square(delta[i].astype(np.float64)).mean(-1)).reshape(4,4).tolist())
        values[str(offset)]=row
    out={'mse_curve':patches.astype(np.float64).mean((-1,-2)).tolist(),'by_offset':values}
    if T is not None:
        out['mechanism_curves']={'raw_self_weight':np.diagonal(T,axis1=-2,axis2=-1).astype(np.float64).mean(-1).tolist(),
          'effective_self_weight':np.diagonal(M,axis1=-2,axis2=-1).astype(np.float64).mean(-1).tolist(),
          'mean_gate':g.astype(np.float64).mean((-1,-2)).tolist(),
          'correction_rms':np.sqrt(np.square(delta.astype(np.float64)).mean((-1,-2))).tolist()}
    return out


def average_records(rows):
    require(len(rows)==3 and sorted(r['seed'] for r in rows)==[0,1,2],'All three seeds required')
    def average(values):
        if isinstance(values[0],dict):
            require(all(set(v)==set(values[0]) for v in values),'Mismatched numeric fields')
            return {k:average([v[k] for v in values]) for k in values[0]}
        arrays=[np.asarray(v,dtype=np.float64) for v in values]
        require(all(a.shape==arrays[0].shape and np.isfinite(a).all() for a in arrays),'Invalid mean arrays')
        return np.mean(arrays,axis=0).tolist()
    fields=['mse_curve','by_offset']+(['mechanism_curves'] if 'mechanism_curves'in rows[0] else [])
    return average([{k:r[k] for k in fields} for r in rows])
