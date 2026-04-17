# API Reference

Base URL (production): `https://mezonya-compatibility-api-<hash>-ew.a.run.app`

OpenAPI spec is served live at `/docs` (Swagger UI) and `/redoc`, generated from the pydantic models in `api/main.py`.

## Authentication

The API is currently unauthenticated, locked down at the network layer via CORS (`https://mezonya.com` and localhost dev origins only). Cloud Run can be fronted with an API Gateway when we need per-client quotas or token-based auth; no code change is needed in the app.

## `GET /health`

Liveness probe used by Cloud Run and the uptime check.

```json
{
  "status": "healthy",
  "model_version": "20260411_171640",
  "timestamp": "2026-04-17T14:22:00.000Z"
}
```

Status values: `healthy`, `degraded`.

## `POST /predict`

Score a single device pair.

**Request**
```json
{
  "device_a": {
    "brand": "Aqara",
    "category": "security",
    "connectivity": ["zigbee", "wifi"],
    "ecosystems": ["apple", "google", "alexa"],
    "hub_required": false,
    "cloud_dependency": "optional"
  },
  "device_b": { "...": "same shape" }
}
```

Required: `brand`, `category`, `connectivity`, `ecosystems`. Optional: `device_id`, `name`, `hub_required` (default `false`), `cloud_dependency` (default `"optional"`).

**Response**
```json
{
  "device_a_name": null,
  "device_b_name": null,
  "compatibility": "partial",
  "confidence": 0.87,
  "probabilities": {
    "incompatible": 0.04,
    "partial": 0.87,
    "compatible": 0.09
  },
  "reason": "Aqara and Philips work together with limitations. Shared protocols: zigbee, wifi."
}
```

The `reason` field is plain text, safe to drop into an `aria-label` or `<p>`. It never contains HTML.

**Errors**
- `422`: validation failed. Pydantic returns the offending fields. Most common: a value outside the allowed enum (see field reference below).
- `503`: model not loaded (container just started or `models/` missing).

## `POST /predict-catalog`

Score one device against every device in the Mezonya catalog. Used by the "find compatible products" button on a product page. Internally uses one batched `predict_proba` call, so latency is roughly constant in the catalog size up to a few thousand entries.

**Request**
```json
{ "device": { "brand": "Aqara", "...": "" } }
```

**Response**
```json
{
  "device_name": null,
  "model_version": "20260411_171640",
  "results": [
    { "compatibility": "compatible",   "confidence": 0.95, "...": "" },
    { "compatibility": "compatible",   "confidence": 0.91, "...": "" },
    { "compatibility": "partial",      "confidence": 0.82, "...": "" },
    { "compatibility": "incompatible", "confidence": 0.88, "...": "" }
  ]
}
```

Results are pre-sorted: `compatible` first, then `partial`, then `incompatible`. Within each group, higher confidence first.

## Field reference

All enum-like fields are validated against a fixed set. Anything outside the set returns `422`.

| field | allowed values |
|---|---|
| `category` | `security`, `lighting`, `climate`, `energy`, `comfort`, `audio`, `access` |
| `connectivity` (list) | `zigbee`, `wifi`, `bluetooth`, `zwave`, `matter`, `thread`, `ethernet` |
| `ecosystems` (list) | `apple`, `google`, `alexa`, `smartthings`, `ha` |
| `cloud_dependency` | `required`, `optional`, `none` |
| `compatibility` (response) | `compatible`, `partial`, `incompatible` |

When the catalog gains a new category or protocol, add it to the `Literal` types in `api/main.py`, retrain so the `LabelEncoder` picks it up, and ship together.

## Prediction logging

Every successful `/predict` and `/predict-catalog` call appends a JSON line to `$PREDICTION_LOG_PATH` (default `data/production_predictions.jsonl`):

```json
{
  "timestamp": "2026-04-17T14:22:00.000Z",
  "model_version": "20260411_171640",
  "device_a_name": "Aqara Hub",
  "device_b_name": "Philips Hue Bridge",
  "protocol_overlap": 1.0,
  "ecosystem_overlap": 1.0,
  "same_brand": 0,
  "hub_conflict": 0,
  "cloud_compatible": 1,
  "category_synergy": 0.8,
  "both_hub_required": 1,
  "total_protocols": 2,
  "total_ecosystems": 3,
  "predicted_label": "partial",
  "confidence": 0.87
}
```

`scripts/monitor.py` consumes this file to compute drift and prediction quality. Log sink write failures are caught and logged but do not fail the request.

## Rate limits

Cloud Run is configured with `max-instances=3`, ~80 concurrent requests each. No per-user limit at the app layer. Add an API Gateway quota if abuse becomes an issue.
