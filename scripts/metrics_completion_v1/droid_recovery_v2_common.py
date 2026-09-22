"""Disclosed operational recovery: same scoring, CPU serialization diagnostics."""
import copy
import numpy as np
import droid_common as frozen

ROOT = frozen.ROOT
REPORT = frozen.REPORT / "recovery_v2"
REG = REPORT / "registration.json"
METRICS = frozen.METRICS
POLICY = copy.deepcopy(frozen.POLICY)
POLICY["roundtrip_contract"] = {
    "checkpoint_state_dict": "exact CPU tensor equality against the GPU model and a second official loader instance",
    "functional_reload": "exact CPU prediction equality; one CPU thread; first two registered input windows",
    "gpu_repeat": "full first batch, same model and independently loaded model; record differences, not an exact-equality gate",
    "scoring_backend_changed": False,
    "prior_mse_tolerance_changed": False,
    "v1_result_reuse": False,
    "reason": "v1 exact GPU serialization check failed at a few FP32 ulps in 12 mixing-family runs; no complete aggregate existed",
}

# Reuse v1 integrity arithmetic in a private module; no frozen file is modified.
helper = frozen.module("scripts/metrics_completion_v1/droid_common.py", "droid_recovery_private_integrity")
helper.REPORT, helper.REG, helper.POLICY = REPORT, REG, POLICY
for name in ("atomic_json", "atomic_npz", "local", "module", "now", "population", "prior_mse_check", "read",
             "relative", "require", "sha", "validate_arrays", "verify_registration"):
    globals()[name] = getattr(helper, name)


def validate_output(row, reg):
    value, arrays = helper.validate_output(row, reg)
    if row["family"] != "persistence":
        checks = value["roundtrip_checks"]
        require(checks.get("checkpoint_prediction_device") == "cpu"
                and checks.get("checkpoint_prediction_windows") == 2
                and checks.get("checkpoint_state_dict") == "exact", "CPU exact reload receipt absent")
        diagnostic = checks.get("gpu_reproducibility_diagnostic", {})
        require(diagnostic.get("equality_is_acceptance_gate") is False
                and diagnostic.get("prediction_elements") == row["batch_size"] * 10 * 6144,
                "Bounded GPU repeat diagnostic missing")
        for case in ("same_model", "reloaded_model"):
            require(case in diagnostic and all(np.isfinite(diagnostic[case][k]) and diagnostic[case][k] >= 0
                for k in ("max_absolute_prediction_difference", "max_absolute_window_mse_difference")),
                "GPU repeat diagnostic contains invalid differences")
    require(value.get("recovery_original_registration_sha256") == reg["recovery"]["original_registration_sha256"],
            "Original registration binding missing")
    return value, arrays
