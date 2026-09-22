"""Recovery checks: unchanged science, exact serialization, bounded CUDA diagnostics."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import droid_common as old
import droid_recovery_v2_common as new
from droid_recovery_v2_evaluate import exact_state_dict, repeat_difference, check_reload


def test_only_roundtrip_contract_changes():
    policy = copy.deepcopy(new.POLICY)
    policy.pop("roundtrip_contract")
    assert policy == old.POLICY
    assert new.REG != old.REG and new.REPORT.is_relative_to(old.REPORT)
    assert not new.POLICY["roundtrip_contract"]["scoring_backend_changed"]
    assert not new.POLICY["roundtrip_contract"]["prior_mse_tolerance_changed"]
    assert not new.POLICY["roundtrip_contract"]["v1_result_reuse"]


def test_exact_state_dict_detects_one_changed_parameter():
    first = {"weight": torch.ones(3, 4), "buffer": torch.arange(4)}
    second = {k: v.clone() for k, v in first.items()}
    exact_state_dict(first, second)
    second["weight"][1, 2] += 1e-6
    with pytest.raises(AssertionError):
        exact_state_dict(first, second)


def test_repeat_diagnostic_records_difference_without_threshold():
    first = torch.zeros(2, 10, 4)
    second = first.clone(); second[1, 3, 2] = .25
    target = torch.ones_like(first)
    result = repeat_difference(first, second, target, torch.ones(4))
    assert result["differing_elements"] == 1 and not result["exact"]
    assert result["max_absolute_prediction_difference"] == .25
    assert result["max_absolute_window_mse_difference"] == .109375
    second[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        repeat_difference(first, second, target, torch.ones(4))


def test_cpu_roundtrip_calls_official_loader_and_restores_thread_count():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.eye(4))

        def predict(self, support, past, future):
            return (support[:, -1:] @ self.weight).expand(-1, future.shape[1], -1)

    class Loader:
        calls = []

        @classmethod
        def load_package(cls, path, device):
            cls.calls.append(str(device))
            return Toy().to(device).eval(), {"test_fixture": True}

    model = Toy().eval()
    x = torch.arange(4 * 13 * 4, dtype=torch.float32).reshape(4, 13, 4) / 100
    actions = torch.zeros(4, 12, 2)
    threads = torch.get_num_threads()
    with torch.inference_mode():
        prediction = model.predict(x[:, :3], actions[:, :2], actions[:, 2:])
        checks = check_reload(Loader, {"checkpoint": "reports/metrics_completion_v1/droid/test_fixture_unused_path"},
                              model, x, actions, prediction, x[:, 3:], torch.ones(4))
    assert checks["checkpoint_prediction"] == checks["checkpoint_state_dict"] == "exact"
    assert checks["checkpoint_prediction_device"] == "cpu" and checks["checkpoint_prediction_windows"] == 2
    assert checks["gpu_reproducibility_diagnostic"]["same_model"]["exact"]
    assert not checks["gpu_reproducibility_diagnostic"]["equality_is_acceptance_gate"]
    assert len(Loader.calls) == 3 and torch.get_num_threads() == threads
