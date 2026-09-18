"""Numerical and endpoint checks for the publicly exposed model demo."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backend
import app as application


@pytest.fixture
def client():
    application.CALL_TIMES.clear()
    return TestClient(application.app)


def test_independent_inference_parity_and_patch_layout():
    sample = next(iter(backend.SAMPLE_MAP))
    result = backend.forecast(sample, camera=2, seed=1, horizon=10)
    features, actions, _ = backend.load_sample(sample, 2, 10)
    model, _ = backend.training.load_package(backend.RELEASE / "models/droid_factorized_s1", device="cpu")
    with torch.inference_mode():
        expected = model.predict(features[:, :3], actions[:, :2], actions[:, 2:])
    assert result["methods"]["factorized"]["prediction_sha256"] == hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    error = ((expected - features[:, 3:]) / model.feature_std).square()[0]
    np.testing.assert_array_equal(result["methods"]["factorized"]["mse"], error.mean(-1).numpy())
    # Cached features flatten [channel, y, x], so spatial cells stride by four.
    expected_patch = torch.stack([error[:, i::4].mean(-1) for i in range(4)], dim=1)
    np.testing.assert_allclose(result["methods"]["factorized"]["patch_mse"], expected_patch.numpy(), rtol=1e-6)


def test_query_features_do_not_change_forecast(monkeypatch):
    sample = next(iter(backend.SAMPLE_MAP))
    original = backend.forecast(sample)
    loader = backend.load_sample
    def altered(*args):
        features, actions, metadata = loader(*args)
        features[:, 3:] += 100
        return features, actions, metadata
    monkeypatch.setattr(backend, "load_sample", altered)
    changed = backend.forecast(sample)
    for method in ("factorized", "framewise"):
        assert original["methods"][method]["prediction_sha256"] == changed["methods"][method]["prediction_sha256"]
        assert original["methods"][method]["mse"] != changed["methods"][method]["mse"]


def test_all_bundled_bytes_match_manifest():
    for sample in backend.MANIFEST["samples"]:
        for camera in sample["cameras"].values():
            for asset in [camera, *camera["frames"]]:
                assert backend.sha(backend.SAMPLES / asset["file"]) == asset["sha256"]


def test_assets_are_allowlisted_and_parameters_bounded(client):
    sample = next(iter(backend.SAMPLE_MAP))
    for path in ("/assets/manifest.json", "/assets/model.pt", "/.git/config", "/docs", "/api/openapi.json"):
        assert client.get(path).status_code == 404
    for payload in ({"sample_id": "../../secret"}, {"sample_id": sample, "horizon": 1000},
                    {"sample_id": sample, "seed": 999}, {"sample_id": sample, "path": "/etc/passwd"}):
        assert client.post("/api/forecast", json=payload).status_code in (400, 422)
    assert client.post("/api/forecast", content=b"x" * 3000).status_code == 413


def test_fresh_request_identity_and_global_rate_limit(client):
    payload = {"sample_id": next(iter(backend.SAMPLE_MAP))}
    first = client.post("/api/forecast", json=payload)
    second = client.post("/api/forecast", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json()["request_id"] != second.json()["request_id"]
    assert first.json()["methods"] == second.json()["methods"]
    application.CALL_TIMES.extend([application.time.monotonic()] * 30)
    assert client.post("/api/forecast", json=payload).status_code == 429


def test_content_security_and_cross_origin_contract(client):
    response = client.get("/health", headers={"Origin": "https://aj-das-research.github.io"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://aj-das-research.github.io"
    assert "frame-ancestors" in response.headers["content-security-policy"]
    response = client.get("/api/catalog", headers={"Origin": "https://unrelated.example"})
    assert "access-control-allow-origin" not in response.headers
    assert client.get("/health", headers={"Origin": "http://127.0.0.1:8047"}).headers["access-control-allow-origin"] == "*"
