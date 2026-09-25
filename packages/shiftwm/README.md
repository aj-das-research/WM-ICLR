# shiftwm

Patch-feature world models that **transport what they observe**.

ShiftWM forecasts future frozen-encoder patch features (e.g. DINOv2) from a few observed frames and
an action sequence. Each predicted patch is a gated mixture of the last observed feature and a local
soft transport of observed features, plus a small learned correction:

    Z_hat_k = (1 - g) * Z_0 + g * T_k(Z_{-S+1..0}) + s * W h_k

The same transport can be bolted onto an existing predictor as a plug-in head,
`Z_hat = (1 - g) * P + g * T`, whose gate starts closed so the wrapped model initially reproduces
the backbone.

## Install

```bash
pip install shiftwm            # or: pip install dist/shiftwm-0.1.0-py3-none-any.whl
```

Dependencies: `torch>=2.0`, `numpy`. Runs on CPU or GPU.

## Quickstart

```python
import torch
from shiftwm import ShiftWM, ShiftHead, forecast, transport_field, skill

model = ShiftWM.from_config(channels=384, grid=16, action_dim=35, history=3, horizon=10)
# or: model = ShiftWM.from_checkpoint("best.pt")   # {"model": state_dict, "config": dict}

hist   = torch.randn(2, 3, 256, 384)   # [B, H, N, C] standardised patch features
past   = torch.randn(2, 2, 35)         # [B, H-1, A]
future = torch.randn(2, 10, 35)        # [B, K, A]

pred, det = forecast(model, hist, past, future, return_details=True)
pred.shape                 # [2, 10, 256, 384]
det["gate"].mean()         # how much of each patch is transported
flow = transport_field(det)  # [2, 10, 256, 2] expected (dx, dy) source offset, in patches
skill(pred, target, hist)  # 1 - MSE / MSE(persistence)

# Plug-in head for any patch-token predictor P = backbone(obs):
head = ShiftHead(token_dim=384, grid=16, sources=2)   # gate_bias=-4 -> g = 0.018 at init
z_hat = head(pred_from_backbone, hist)                 # or head.wrap(backbone)(hist)
```

`examples/quickstart.py` runs the above on random features in a few seconds;
`examples/forecast_checkpoint.py` loads a trained checkpoint and forecasts one test window.

## API

| name | what |
|---|---|
| `ShiftWM` / `WorldModel` | the forecaster; `from_config(**kw)`, `from_checkpoint(path)`; arms `shiftwm` (default), `direct`, `ar`, `ar_tf`, `persistence`, `linear` |
| `WorldModelConfig` | dataclass of hyper-parameters |
| `ShiftHead(token_dim, grid, hidden_dim=None, sources=1, window=7, ...)` | plug-in head; `head(P, observed, hidden=None, return_details=False)`, `head.wrap(backbone)` |
| `local_transport(q, k, v, grid, window, identity_bias)` | the identity-biased local soft transport |
| `forecast(model, hist, past, future, return_details=False)` | no-grad forecast; details: `weights`, `gate`, `correction` |
| `transport_field(details)` | expected source offset `(dx, dy)` per predicted patch |
| `skill(pred, target, history)` | `1 - MSE / MSE_persistence` |

Tensor layout everywhere: `[batch, frames, N = grid*grid, channels]`.

## Tests

```bash
pip install "shiftwm[test]" && pytest tests
```

The tests check that the package reproduces the research implementation bit-for-bit up to float
tolerance (max abs diff < 1e-5) when that implementation is available, and that `ShiftHead` at
initialisation reproduces its backbone.

## License

MIT, see `LICENSE`.
