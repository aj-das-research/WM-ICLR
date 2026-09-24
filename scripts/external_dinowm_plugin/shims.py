"""Wrapper-level shims for running the unmodified official DINO-WM code (external/dino_wm) in our cluster env.

None of them changes model / loss / planner arithmetic.  `sys.path` must already contain external/dino_wm.
"""
import os
import sys
import types

# upstream was written for torch 2.3 (torch.load pickles whole modules); torch>=2.6 defaults to weights_only=True.
# Only our own checkpoints and the official OSF checkpoints are loaded.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch


class _NoCuda:
    def __init__(self, t):
        self.t = t

    def to(self, *a, **k):
        return self.t


def install_cpu_mask_shim(force=False):
    """models.vit.Attention does `generate_mask_matrix(...).to('cuda')`; keep the mask on CPU when there is no GPU."""
    if torch.cuda.is_available() and not force:
        return
    import models.vit as vit
    if getattr(vit.generate_mask_matrix, "_shiftwm_shim", False):
        return
    orig = vit.generate_mask_matrix

    def gen(*a, **k):
        return _NoCuda(orig(*a, **k))

    gen._shiftwm_shim = True
    vit.generate_mask_matrix = gen


def install_pointmaze_stub():
    """env/__init__.py imports env.pointmaze (mujoco_py + d4rl) unconditionally.  For PushT / Wall planning we stub
    it so the environments we use can be registered without a MuJoCo-210 build.  Never use for env=point_maze."""
    if "env.pointmaze" in sys.modules:
        return
    stub = types.ModuleType("env.pointmaze")
    stub.U_MAZE = "U_MAZE_STUB"

    def _unavailable(*a, **k):
        raise RuntimeError("point_maze needs mujoco_py; stubbed by shims.install_pointmaze_stub")

    stub.PointMazeWrapper = _unavailable
    sys.modules["env.pointmaze"] = stub


def fast_slice_getitem(self, idx, _orig=None):
    """Output-identical replacement for TrajSlicerDataset.__getitem__ on PushT: decode only the returned frames."""
    from einops import rearrange
    from datasets.pusht_dset import PushTDataset
    from datasets.traj_dset import TrajSubset

    i, start, end = self.slices[idx]
    ds = self.dataset
    base, j = (ds.dataset, ds.indices[i]) if isinstance(ds, TrajSubset) else (ds, i)
    if not isinstance(base, PushTDataset):
        return _ORIG_GETITEM(self, idx)
    obs, _, state, _ = base.get_frames(j, list(range(start, end, self.frameskip)))
    act = rearrange(base.actions[j, start:end], "(n f) d -> n (f d)", n=self.num_frames)
    return tuple([obs, act, state])


_ORIG_GETITEM = None


def install_fast_slices():
    global _ORIG_GETITEM
    from datasets.traj_dset import TrajSlicerDataset
    if _ORIG_GETITEM is None:
        _ORIG_GETITEM = TrajSlicerDataset.__getitem__
        TrajSlicerDataset.__getitem__ = fast_slice_getitem


def install_preprocessor_5d_shim():
    """preprocessor.Preprocessor applies torchvision Resize to a (b,t,c,h,w) tensor.  With img_size=224 and 224px
    renders torchvision returns early (no-op resize); for other sizes torchvision>=0.19 rejects 5-D input.  Apply the
    same per-image transform on (b*t,c,h,w) -- output-identical, only needed for reduced-size smoke tests."""
    import preprocessor as pp
    if getattr(pp.Preprocessor.transform_obs_visual, "_shiftwm_shim", False):
        return

    def transform_obs_visual(self, obs_visual):
        x = self.preprocess_obs_visual(torch.tensor(obs_visual))
        b, t = x.shape[:2]
        y = self.transform(x.reshape(b * t, *x.shape[2:]))
        return y.reshape(b, t, *y.shape[1:])

    transform_obs_visual._shiftwm_shim = True
    pp.Preprocessor.transform_obs_visual = transform_obs_visual


def install_chunked_decode_shim(chunk=8):
    """Plot-only: VWorldModel.decode_obs is used by the evaluator solely to render rollout videos after the success
    metrics are computed. Decoding all n_evals trajectories at once can exceed 140 GB on Wall; decode in chunks under
    no_grad instead. Outputs are identical; planning and metrics are untouched."""
    import torch
    from einops import rearrange
    from models import visual_world_model as vwm

    def decode_obs(self, z_obs):
        b, t = z_obs["visual"].shape[:2]
        vis, diffs = [], []
        with torch.no_grad():
            for i in range(0, b, chunk):
                v, d = self.decoder(z_obs["visual"][i:i + chunk])
                vis.append(rearrange(v, "(b t) c h w -> b t c h w", t=t)); diffs.append(d)
        return {"visual": torch.cat(vis), "proprio": z_obs["proprio"]}, torch.stack([torch.as_tensor(d) for d in diffs]).mean()
    vwm.VWorldModel.decode_obs = decode_obs
