"""Model & data monitoring. Run on a schedule against recent predictions."""
import argparse
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from scipy import stats

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

log = logging.getLogger("monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DRIFT_SHARE_THRESHOLD = 0.3
KS_P_VALUE = 0.05
MIN_ACCURACY = 0.80

FEATURE_COLS = [
    "protocol_overlap",
    "ecosystem_overlap",
    "same_brand",
    "hub_conflict",
    "cloud_compatible",
    "category_synergy",
    "both_hub_required",
    "total_protocols",
    "total_ecosystems",
]


def drift_evidently(reference, current):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[FEATURE_COLS], current_data=current[FEATURE_COLS])
    result = report.as_dict()
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
    drifted = [
        col for col, info in result["metrics"][1]["result"]["drift_by_columns"].items()
        if info["drift_detected"]
    ]
    return {
        "drift_detected": bool(drift_share > DRIFT_SHARE_THRESHOLD),
        "drift_share": float(drift_share),
        "drifted_columns": drifted,
        "method": "evidently",
    }


def drift_ks(reference, current):
    """Kolmogorov-Smirnov fallback when Evidently is not available."""
    drifted = []
    for col in FEATURE_COLS:
        if col not in current.columns:
            continue
        # KS test is robust to non-normal distributions, unlike z-score
        _, p = stats.ks_2samp(reference[col], current[col])
        if p < KS_P_VALUE:
            drifted.append(col)
    return {
        "drift_detected": len(drifted) / len(FEATURE_COLS) > DRIFT_SHARE_THRESHOLD,
        "drift_share": round(len(drifted) / len(FEATURE_COLS), 4),
        "drifted_columns": drifted,
        "method": "ks_test",
    }


def drift(reference, current):
    if HAS_EVIDENTLY:
        return drift_evidently(reference, current)
    log.warning("evidently not installed, using KS fallback")
    return drift_ks(reference, current)


def accuracy_against_installers(predictions):
    """Compare model output to installer-validated labels if available."""
    if "installer_label" not in predictions.columns:
        return {"status": "no_feedback"}

    validated = predictions.dropna(subset=["installer_label"])
    if validated.empty:
        return {"status": "no_feedback"}

    agree = (validated["predicted_label"] == validated["installer_label"]).mean()
    return {
        "accuracy_vs_installers": round(float(agree), 4),
        "sample_size": int(len(validated)),
        "disagreements": int((validated["predicted_label"] != validated["installer_label"]).sum()),
        "below_threshold": bool(agree < MIN_ACCURACY),
    }


def run(reference_path, production_path, out_dir="reports"):
    reference = pd.read_csv(reference_path)

    if not os.path.exists(production_path):
        log.info("no production data yet at %s", production_path)
        return {"status": "no_production_data", "timestamp": datetime.now(timezone.utc).isoformat()}

    # API writes predictions as JSONL (one JSON object per line). Fall back
    # to CSV for backfilled/exported data so this function is usable from a
    # notebook too.
    if production_path.endswith(".jsonl"):
        production = pd.read_json(production_path, lines=True)
    else:
        production = pd.read_csv(production_path)
    if production.empty:
        return {"status": "no_production_data", "timestamp": datetime.now(timezone.utc).isoformat()}

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_drift": drift(reference, production),
        "prediction_quality": accuracy_against_installers(production),
    }

    if "confidence" in production.columns:
        report["confidence"] = {
            "mean": round(float(production["confidence"].mean()), 4),
            "median": round(float(production["confidence"].median()), 4),
            "low_confidence_count": int((production["confidence"] < 0.5).sum()),
            "total": int(len(production)),
        }

    needs_retrain = (
        report["data_drift"].get("drift_detected", False)
        or report["prediction_quality"].get("below_threshold", False)
    )
    report["action"] = "retrain" if needs_retrain else "none"

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"monitoring_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("report saved to %s", out_path)

    if needs_retrain:
        log.warning("retrain recommended")

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/compatibility_dataset.csv")
    ap.add_argument("--production", default="data/production_predictions.jsonl")
    ap.add_argument("--output", default="reports")
    args = ap.parse_args()

    report = run(args.reference, args.production, args.output)
    print(json.dumps(report, indent=2))
