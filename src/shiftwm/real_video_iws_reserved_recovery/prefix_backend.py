"""CPU inference-only GRU execution order; identical modules, weights and keys.

Evaluate one command row per GRU call so requested sequence length cannot alter
the input-projection GEMM shape. This is a separately reviewed numerical backend,
not a change to trained parameters, architecture, commands, or tolerances.
"""
from contextlib import contextmanager
from types import MethodType
import torch

VERSION = 'one_command_row_gru_cpu_fp32_v1'


def _rowwise_forward(gru, inputs, hx=None):
    if (gru.training or inputs.device.type != 'cpu' or inputs.dtype != torch.float32
            or inputs.ndim != 3 or inputs.shape[1] < 1):
        raise ValueError('Rowwise reserved backend requires eval-mode CPU FP32 command sequences')
    outputs = []
    state = hx
    for index in range(inputs.shape[1]):
        value, state = torch.nn.GRU.forward(gru, inputs[:, index:index+1].contiguous(), state)
        outputs.append(value)
    return torch.cat(outputs, dim=1), state


def install_rowwise_gru(model):
    """Patch execution in place without replacing a module or altering state keys."""
    gru = model.action_prefix
    if type(gru) is not torch.nn.GRU or not gru.batch_first or gru.bidirectional or gru.num_layers != 1 or gru.dropout != 0:
        raise ValueError('Expected the frozen single-layer unidirectional action GRU')
    if getattr(gru, '_reserved_prefix_backend', None) == VERSION:
        return model
    if 'forward' in gru.__dict__:
        raise ValueError('Refusing to replace an existing custom GRU execution method')
    gru.forward = MethodType(_rowwise_forward, gru)
    gru._reserved_prefix_backend = VERSION
    return model


@contextmanager
def rowwise_gru_context(model):
    already = getattr(model.action_prefix, '_reserved_prefix_backend', None) == VERSION
    install_rowwise_gru(model)
    try:
        yield model
    finally:
        if not already:
            del model.action_prefix.forward
            del model.action_prefix._reserved_prefix_backend
