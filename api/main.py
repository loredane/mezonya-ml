"""Compatibility prediction API.

Endpoints:
    POST /predict            score one device pair
    POST /predict-catalog    score one device against the full catalog
    GET  /health             liveness probe

Predictions are appended as JSON lines to $PREDICTION_LOG_PATH (default
data/production_predictions.jsonl). scripts/monitor.py reads this file to
compute drift. On Cloud Run the file lives on the container FS and is
shipped to Cloud Logging via stdout; in production we replicate it to
BigQuery via a log sink (see docs/MODEL_CARD.md > Monitoring section).
"""
import json
import logging
import os
import pickle
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = os.getenv("MODEL_DIR", "models")
CATALOG_PATH = os.getenv("CATALOG_PATH", "data/device_catalog.json")
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "data/production_predictions.jsonl")


# JSON-formatted log records so Cloud Logging picks up structured fields
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # merge extras set via log.info("msg", extra={"foo": "bar"})
        for k, v in record.__dict__.items():
            if k not in ("msg", "args", "levelname", "levelno", "pathname", "filename",
                         "module", "exc_info", "exc_text", "stack_info", "lineno",
                         "funcName", "created", "msecs", "relativeCreated", "thread",
                         "threadName", "processName", "process", "name", "message"):
                payload[k] = v
        return json.dumps(payload)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
log = logging.getLogger("api")


state: dict = {"model": None, "encoder": None, "config": None, "catalog": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(f"{MODEL_DIR}/model.pkl", "rb") as f:
        state["model"] = pickle.load(f)
    with open(f"{MODEL_DIR}/category_encoder.pkl", "rb") as f:
        state["encoder"] = pickle.load(f)
    with open(f"{MODEL_DIR}/model_config.json") as f:
        state["config"] = json.load(f)
    log.info("model loaded", extra={"version": state["config"]["model_version"]})

    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            state["catalog"] = json.load(f)
        log.info("catalog loaded", extra={"count": len(state["catalog"])})

    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH) or ".", exist_ok=True)

    yield


app = FastAPI(
    title="Mezonya Compatibility API",
    description=(
        "Predicts compatibility between smart-home devices. Responses are "
        "structured so clients can render visual indicators AND accessible "
        "text for screen readers. Every prediction is logged for drift "
        "monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mezonya.com",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


Connectivity = Literal["zigbee", "wifi", "bluetooth", "zwave", "matter", "thread", "ethernet"]
Ecosystem = Literal["apple", "google", "alexa", "smartthings", "ha"]
Category = Literal["security", "lighting", "climate", "energy", "comfort", "audio", "access"]
CloudDependency = Literal["required", "optional", "local_only", "none"]
Compatibility = Literal["compatible", "partial", "incompatible"]


class DeviceSpec(BaseModel):
    model_config = {"populate_by_name": True}

    device_id: Optional[int] = Field(default=None, alias="id")
    name: Optional[str] = None
    brand: str = Field(..., examples=["Aqara"])
    category: Category = Field(..., examples=["security"])
    connectivity: list[Connectivity] = Field(..., examples=[["zigbee", "wifi"]])
    ecosystems: list[Ecosystem] = Field(..., examples=[["apple", "google", "alexa"]])
    hub_required: bool = False
    cloud_dependency: CloudDependency = "optional"


class PairRequest(BaseModel):
    device_a: DeviceSpec
    device_b: DeviceSpec


class PredictionResult(BaseModel):
    device_a_name: Optional[str] = None
    device_b_name: Optional[str] = None
    compatibility: Compatibility
    confidence: float
    probabilities: dict[str, float]
    reason: str  # plain-text explanation, usable as aria-label


class CatalogRequest(BaseModel):
    device: DeviceSpec


class CatalogResult(BaseModel):
    device_name: Optional[str]
    results: list[PredictionResult]
    model_version: str


class Health(BaseModel):
    status: str
    model_version: str
    timestamp: str


CATEGORY_SYNERGY = {
    frozenset({"security"}): 1.0,
    frozenset({"lighting"}): 0.9,
    frozenset({"security", "lighting"}): 0.8,
    frozenset({"security", "energy"}): 0.7,
    frozenset({"climate", "energy"}): 0.9,
    frozenset({"comfort", "lighting"}): 0.8,
    frozenset({"comfort", "climate"}): 0.7,
    frozenset({"audio", "comfort"}): 0.6,
    frozenset({"audio", "lighting"}): 0.7,
    frozenset({"access", "security"}): 0.9,
}


def _cloud_compatible(a_dep: str, b_dep: str) -> int:
    """Option B friction rule - mirrors data/generate_dataset.py:cloud_compatibility.

    Two devices can share a cloud-level control plane unless:
        * both require their own cloud (two apps, no shared orchestration), or
        * one is local_only and the other is required (no common plane at all).
    """
    pair = frozenset({a_dep, b_dep})
    if pair == frozenset({"required"}):
        return 0
    if pair == frozenset({"local_only", "required"}):
        return 0
    return 1


def build_features(a: DeviceSpec, b: DeviceSpec) -> list:
    """Return the 15-feature vector for one pair. Same logic as scripts/train.py."""
    conn_a, conn_b = set(a.connectivity), set(b.connectivity)
    eco_a, eco_b = set(a.ecosystems), set(b.ecosystems)

    protocol_overlap = len(conn_a & conn_b) / max(len(conn_a | conn_b), 1)
    ecosystem_overlap = len(eco_a & eco_b) / max(len(eco_a | eco_b), 1)
    same_brand = int(a.brand == b.brand)

    # Zigbee hub-required paired with non-Zigbee = conflict
    hub_conflict = int(
        (a.hub_required and not b.hub_required and "zigbee" in conn_a and "zigbee" not in conn_b)
        or (b.hub_required and not a.hub_required and "zigbee" in conn_b and "zigbee" not in conn_a)
    )
    cloud_compatible = _cloud_compatible(a.cloud_dependency, b.cloud_dependency)
    category_synergy = CATEGORY_SYNERGY.get(frozenset({a.category, b.category}), 0.4)

    encoder = state["encoder"]
    known = set(encoder.classes_)
    cat_a = int(encoder.transform([a.category])[0]) if a.category in known else -1
    cat_b = int(encoder.transform([b.category])[0]) if b.category in known else -1

    return [
        protocol_overlap, ecosystem_overlap, same_brand,
        hub_conflict, cloud_compatible, category_synergy,
        int(a.hub_required and b.hub_required),
        len(conn_a | conn_b), len(eco_a | eco_b),
        int(a.hub_required), int(b.hub_required),
        int(a.cloud_dependency == "required"),
        int(b.cloud_dependency == "required"),
        cat_a, cat_b,
    ]


def explain(a: DeviceSpec, b: DeviceSpec, label: str) -> str:
    """Human-readable text, used as the aria-label on the frontend."""
    shared_eco = set(a.ecosystems) & set(b.ecosystems)
    shared_conn = set(a.connectivity) & set(b.connectivity)

    if label == "compatible":
        eco = ", ".join(sorted(shared_eco)) or "none"
        return f"{a.brand} and {b.brand} work together. Shared ecosystems: {eco}."
    if label == "partial":
        proto = ", ".join(sorted(shared_conn)) or "none"
        return (
            f"{a.brand} and {b.brand} work together with limitations. "
            f"Shared protocols: {proto}."
        )
    return (
        f"{a.brand} and {b.brand} are not compatible. "
        f"No shared ecosystem between {list(a.ecosystems)} and {list(b.ecosystems)}."
    )


def predict_batch(source: DeviceSpec, others: list[DeviceSpec]) -> list[PredictionResult]:
    """Run one batched predict_proba call for N pairs instead of N sequential calls."""
    if not others:
        return []
    X = np.array([build_features(source, other) for other in others])
    probas = state["model"].predict_proba(X)  # shape (N, 3)
    labels = state["config"]["label_names"]

    results = []
    for i, other in enumerate(others):
        idx = int(np.argmax(probas[i]))
        label = labels[idx]
        results.append(PredictionResult(
            device_a_name=source.name,
            device_b_name=other.name,
            compatibility=label,  # type: ignore
            confidence=round(float(probas[i][idx]), 4),
            probabilities={labels[j]: round(float(probas[i][j]), 4) for j in range(len(labels))},
            reason=explain(source, other, label),
        ))
    return results


def log_prediction(result: PredictionResult, features: list):
    """Append one JSON line with features + prediction for monitor.py to consume."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "model_version": state["config"]["model_version"] if state["config"] else None,
        "device_a_name": result.device_a_name,
        "device_b_name": result.device_b_name,
        "protocol_overlap": features[0],
        "ecosystem_overlap": features[1],
        "same_brand": features[2],
        "hub_conflict": features[3],
        "cloud_compatible": features[4],
        "category_synergy": features[5],
        "both_hub_required": features[6],
        "total_protocols": features[7],
        "total_ecosystems": features[8],
        "predicted_label": result.compatibility,
        "confidence": result.confidence,
    }
    try:
        with open(PREDICTION_LOG_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        # Don't fail the request if the log sink is unavailable
        log.warning("prediction log write failed: %s", e)


@app.get("/health", response_model=Health)
async def health():
    loaded = state["model"] is not None and state["encoder"] is not None and state["config"] is not None
    return Health(
        status="healthy" if loaded else "degraded",
        model_version=state["config"]["model_version"] if state["config"] else "unknown",
        timestamp=datetime.now(timezone.utc).isoformat() + "Z",
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(req: PairRequest):
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")

    features = build_features(req.device_a, req.device_b)
    proba = state["model"].predict_proba(np.array([features]))[0]
    labels = state["config"]["label_names"]
    idx = int(np.argmax(proba))
    label = labels[idx]

    result = PredictionResult(
        device_a_name=req.device_a.name,
        device_b_name=req.device_b.name,
        compatibility=label,  # type: ignore
        confidence=round(float(proba[idx]), 4),
        probabilities={labels[j]: round(float(proba[j]), 4) for j in range(len(labels))},
        reason=explain(req.device_a, req.device_b, label),
    )

    log_prediction(result, features)
    log.info("predict", extra={
        "brand_a": req.device_a.brand, "brand_b": req.device_b.brand,
        "label": label, "confidence": result.confidence,
    })
    return result


@app.post("/predict-catalog", response_model=CatalogResult)
async def predict_catalog(req: CatalogRequest):
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")
    if not state["catalog"]:
        raise HTTPException(503, "catalog not loaded")

    others = []
    for entry in state["catalog"]:
        if req.device.device_id and entry.get("id") == req.device.device_id:
            continue
        others.append(DeviceSpec(**entry))

    results = predict_batch(req.device, others)

    # log each prediction
    for result, other in zip(results, others):
        features = build_features(req.device, other)
        log_prediction(result, features)

    priority = {"compatible": 3, "partial": 2, "incompatible": 1}
    results.sort(key=lambda r: (priority[r.compatibility], r.confidence), reverse=True)

    return CatalogResult(
        device_name=req.device.name,
        results=results,
        model_version=state["config"]["model_version"],
    )
