"""Model and API tests."""
import json
import os
import pickle

import numpy as np
import pytest

MODEL_DIR = os.getenv("MODEL_DIR", "models")


@pytest.fixture(scope="module")
def model():
    with open(f"{MODEL_DIR}/model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def config():
    with open(f"{MODEL_DIR}/model_config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def client():
    # import here so other tests can run without FastAPI installed
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_model_loads(model):
    assert model is not None


def test_model_predicts_valid_class(model):
    x = np.array([[0.5, 0.5, 1, 0, 1, 0.8, 0, 2, 3, 0, 0, 0, 0, 1, 2]])
    assert model.predict(x)[0] in (0, 1, 2)


def test_probabilities_sum_to_one(model):
    x = np.array([[0.5, 0.5, 1, 0, 1, 0.8, 0, 2, 3, 0, 0, 0, 0, 1, 2]])
    proba = model.predict_proba(x)[0]
    assert abs(proba.sum() - 1.0) < 1e-6


def test_config_shape(config):
    assert len(config["feature_columns"]) == 15
    assert config["label_names"] == ["incompatible", "partial", "compatible"]


# Golden set: pairs where the answer is unambiguous to a human installer.
# Feature order matches scripts/train.FEATURES.
GOLDEN = [
    ("aqara_hub + aqara_sensor (same brand, same protocol)",
     [1.0, 1.0, 1, 0, 1, 1.0, 1, 1, 3, 0, 1, 0, 0, 4, 4], 2),
    ("ring (alexa only) + eve (apple only), disjoint ecosystems",
     [0.0, 0.0, 0, 0, 0, 1.0, 0, 2, 2, 0, 0, 1, 0, 4, 4], 0),
]


@pytest.mark.parametrize("name,features,expected", GOLDEN)
def test_golden(model, name, features, expected):
    pred = model.predict(np.array([features]))[0]
    assert pred == expected, f"{name}: got {pred}, expected {expected}"


def test_hue_nanoleaf_at_least_partial(model):
    # philips hue + nanoleaf: shared WiFi, all ecosystems, same category
    x = np.array([[0.5, 1.0, 0, 1, 1, 0.9, 0, 2, 3, 1, 0, 0, 0, 2, 2]])
    assert model.predict(x)[0] >= 1


def test_metrics_above_threshold():
    path = f"{MODEL_DIR}/metrics.json"
    if not os.path.exists(path):
        pytest.skip("no metrics file, run training first")
    with open(path) as f:
        m = json.load(f)
    assert m["f1_macro"] >= 0.85
    assert m["accuracy"] >= 0.85


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "model_version" in body


def test_predict_endpoint(client):
    payload = {
        "device_a": {
            "brand": "Aqara", "category": "security",
            "connectivity": ["zigbee", "wifi"],
            "ecosystems": ["apple", "google", "alexa"],
            "hub_required": False, "cloud_dependency": "optional",
        },
        "device_b": {
            "brand": "Aqara", "category": "security",
            "connectivity": ["zigbee"],
            "ecosystems": ["apple", "google", "alexa"],
            "hub_required": True, "cloud_dependency": "optional",
        },
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["compatibility"] in ("compatible", "partial", "incompatible")
    assert 0 <= body["confidence"] <= 1
    assert "reason" in body and len(body["reason"]) > 10


def test_predict_disjoint_ecosystems_is_incompatible(client):
    payload = {
        "device_a": {
            "brand": "Ring", "category": "security",
            "connectivity": ["wifi"], "ecosystems": ["alexa"],
            "hub_required": False, "cloud_dependency": "required",
        },
        "device_b": {
            "brand": "Eve", "category": "security",
            "connectivity": ["bluetooth"], "ecosystems": ["apple"],
            "hub_required": False, "cloud_dependency": "optional",
        },
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["compatibility"] == "incompatible"
