"""Numerical recovery after access, preserving the original scientific roster.

The original cache authorization remains explicit. This new registration is
not presented as having preceded the first reserved-data access.
"""
from __future__ import annotations

from pathlib import Path

from shiftwm.real_video_iws_reserved import protocol as original

ROOT = Path(__file__).resolve().parents[3]
REGISTRATION_PATH = Path("configs/real_video_iws_reserved_recovery_v2/registration.json")
REVIEW_PATH = Path("reports/real_video_iws_reserved_recovery_v2/source_review.json")
SCHEMA = "shiftwm_iws_reserved_numeric_recovery_registration_v2"
STATE = "numeric_recovery_registered_after_initial_access"


def validate_recovery(registry, baseline):
    if registry.get("schema") != SCHEMA or registry.get("status") != STATE:
        raise ValueError("Explicit post-access numerical recovery registration required")
    for field in ("runs", "tasks", "evaluation", "extraction", "qualitative_selection", "expected_runs"):
        if registry.get(field) != baseline[field]:
            raise ValueError("Recovery cannot change the frozen scientific contract: " + field)
    if registry.get("cache_registration_path") != str(original.REGISTRATION_PATH):
        raise ValueError("Recovery must reuse the unchanged, already authorized feature cache")
    recovery = registry.get("numerical_recovery", {})
    expected = {"command_gru_dispatch": "one_native_row_per_call",
                "same_weights": True, "same_normalization": True,
                "same_tolerance": True, "same_metrics": True,
                "same_examples": True, "same_comparators": True,
                "all36_rerun": True, "original_results_retained": True,
                "future_targets_used_to_choose_backend": False,
                "accuracy_used_to_choose_backend": False,
                "initial_reserved_access_precedes_this_revision": True}
    if recovery != expected:
        raise ValueError("Recovery scope or numerical implementation differs")
    deps = registry.get("dependencies", {})
    if not isinstance(deps, dict) or not deps:
        raise ValueError("Recovery source dependencies are required")
    if any(deps.get(name) != digest for name, digest in baseline["dependencies"].items()):
        raise ValueError("Recovery must retain every original frozen source binding")
    return registry


def checked_registration(root=ROOT, registration_path=None, require_review=True):
    root = Path(root).resolve()
    baseline = original.checked_registration(root)
    path = Path(registration_path) if registration_path is not None else root / REGISTRATION_PATH
    if not path.is_absolute():
        path = root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError("Recovery registration leaves the workspace")
    registry = validate_recovery(original.read_json(path), baseline)
    if registry.get("original_registration_sha256") != original.sha(root / original.REGISTRATION_PATH):
        raise ValueError("Original registration identity differs")
    if require_review:
        review = original.read_json(root / REVIEW_PATH)
        if (review.get("status") != "passed" or review.get("registration_sha256") != original.sha(path)
                or review.get("source_sha256") != registry["dependencies"]
                or review.get("future_targets_used_to_choose_backend") is not False
                or review.get("accuracy_used_to_choose_backend") is not False):
            raise ValueError("Independent numerical recovery review is required")
    for name, expected in registry["dependencies"].items():
        source = original.local(root, name)
        resolved = str(source.resolve().relative_to(root)) + "/"
        if any(resolved.startswith(prefix) for prefix in original.reserved_payload_prefixes(baseline)):
            raise ValueError("Recovery registration must not read raw/cache payloads")
        if original.sha(source) != expected:
            raise ValueError("Frozen recovery dependency changed: " + name)
    return registry
