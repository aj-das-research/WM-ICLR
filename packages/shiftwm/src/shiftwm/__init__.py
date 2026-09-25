"""shiftwm: patch-feature world models that transport what they observe.

Public API::

    from shiftwm import ShiftWM, ShiftHead, forecast, transport_field, skill

* :class:`ShiftWM` (= :class:`WorldModel`): the ShiftWM forecaster, ``from_config`` / ``from_checkpoint``.
* :class:`ShiftHead`: plug-in head ``Z_hat = (1-g) P + g T`` for any patch-token predictor.
* :func:`forecast`: no-grad forecasting, optionally returning transport weights, gate and correction.
* :func:`transport_field`: expected source offset per patch; :func:`skill`: 1 - MSE / MSE_persistence.
"""
from .api import forecast, skill, transport_field
from .head import ShiftHead, ShiftWrapped, local_transport
from .model import ARMS, ShiftWM, WorldModel, WorldModelConfig

__version__ = "0.1.0"

__all__ = ["ShiftWM", "WorldModel", "WorldModelConfig", "ARMS", "ShiftHead", "ShiftWrapped", "local_transport",
           "forecast", "transport_field", "skill", "__version__"]
