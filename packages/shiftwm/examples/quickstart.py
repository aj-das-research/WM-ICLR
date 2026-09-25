"""ShiftWM quickstart: random features, CPU, a few seconds.

    python examples/quickstart.py
"""
import time

import torch

import shiftwm
from shiftwm import ShiftHead, ShiftWM, forecast, skill, transport_field

torch.manual_seed(0)
t0 = time.time()

# 1) A ShiftWM forecaster on a 16x16 grid of 384-d patch features (DINOv2-S sized), 35-d actions.
model = ShiftWM.from_config(channels=384, grid=16, action_dim=35, history=3, horizon=10)
print(f"shiftwm {shiftwm.__version__}: {model.num_params() / 1e6:.2f}M parameters")

B, H, K, N, C, A = 2, 3, 10, 256, 384, 35
hist = torch.randn(B, H, N, C)            # observed (standardised) patch features
past = torch.randn(B, H - 1, A)            # actions between observed frames
future = torch.randn(B, K, A)              # planned actions

# 2) Forecast, with the transport details.
pred, det = forecast(model, hist, past, future, return_details=True)
print("pred", tuple(pred.shape), "weights", tuple(det["weights"].shape), "gate", tuple(det["gate"].shape))
g = det["gate"]
print(f"gate: mean {g.mean():.3f}  min {g.min():.3f}  max {g.max():.3f}")

# 3) Where does each patch copy from?  (dx, dy) in patches, weights summed over source frames.
flow = transport_field(det)
print("transport field", tuple(flow.shape), f"mean |offset| {flow.norm(dim=-1).mean():.3f} patches")

# 4) Skill over persistence against a (here random) target.
target = hist[:, -1:] + 0.1 * torch.randn(B, K, N, C)
print(f"skill vs persistence: {skill(pred, target, hist):+.3f}")

# 5) Plug the head into any backbone that forecasts patch tokens.
def backbone(obs, k=K):                    # toy predictor: repeat the last frame
    return obs[:, -1:].expand(-1, k, -1, -1)

wrapped = ShiftHead(token_dim=C, grid=16, sources=2).wrap(backbone)
z, d = wrapped(hist, return_details=True)
diff = (z - backbone(hist)).abs().max()
print(f"ShiftHead at init: gate {d['gate'].mean():.4f}, max |Z_hat - P| {diff:.3f}")
print(f"done in {time.time() - t0:.1f}s")
