"""Scheduled retraining: pull latest feedback, retrain, promote if it beats prod.

Exits non-zero on:
    - corrupted or unreadable dataset CSV
    - training failure (e.g. schema mismatch)
    - promotion-time artifact write failure

A rejected promotion (new model doesn't beat prod) is NOT an error and exits 0.
"""
import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# make scripts.train importable when running as `python scripts/retrain.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import FEATURES, LABEL, PARAMS

log = logging.getLogger("retrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MIN_F1 = 0.85
MIN_DELTA = 0.0


def load_prod_metrics(model_dir: str) -> dict:
    path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(path):
        return {"f1_macro": 0.0}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("could not read %s (%s), assuming no prod model", path, e)
        return {"f1_macro": 0.0}


def load_feedback(feedback_dir: str) -> pd.DataFrame:
    if not os.path.isdir(feedback_dir):
        return pd.DataFrame()
    files = [f for f in os.listdir(feedback_dir)
             if f.startswith("feedback_") and f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    frames = []
    for fname in files:
        try:
            frames.append(pd.read_csv(os.path.join(feedback_dir, fname)))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
            log.warning("skipping corrupted feedback file %s: %s", fname, e)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    log.info("loaded %d feedback rows from %d files", len(df), len(frames))
    return df


def merge(base_path: str, feedback: pd.DataFrame) -> pd.DataFrame:
    try:
        base = pd.read_csv(base_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, FileNotFoundError) as e:
        log.error("cannot read base dataset %s: %s", base_path, e)
        raise

    if feedback.empty:
        return base

    required = {"device_a_id", "device_b_id"}
    if not required.issubset(feedback.columns):
        log.warning("feedback missing required columns %s, ignoring", required - set(feedback.columns))
        return base

    merged = pd.concat([base, feedback], ignore_index=True)
    merged = merged.drop_duplicates(subset=["device_a_id", "device_b_id"], keep="last")
    log.info("merged dataset: %d rows (was %d)", len(merged), len(base))
    return merged


def train(df: pd.DataFrame) -> tuple:
    """Raises ValueError if the dataset is too small or missing required columns."""
    required_cols = ["device_a_category", "device_b_category", LABEL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")
    if len(df) < 50:
        raise ValueError(f"dataset too small to train: {len(df)} rows")

    df = df.copy()
    enc = LabelEncoder()
    enc.fit(pd.concat([df["device_a_category"], df["device_b_category"]]))
    df["device_a_category_encoded"] = enc.transform(df["device_a_category"])
    df["device_b_category_encoded"] = enc.transform(df["device_b_category"])

    X = df[FEATURES].values
    y = df[LABEL].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred = model.predict(X_te)

    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(df),
    }
    log.info("new model: acc=%.4f f1=%.4f", metrics["accuracy"], metrics["f1_macro"])
    return model, enc, metrics


def should_promote(new_metrics: dict, prod_metrics: dict) -> bool:
    new_f1 = new_metrics["f1_macro"]
    prod_f1 = prod_metrics.get("f1_macro", 0.0)

    if new_f1 < MIN_F1:
        log.warning("reject: f1 %.4f below floor %.2f", new_f1, MIN_F1)
        return False
    if new_f1 < prod_f1 - MIN_DELTA:
        log.warning("reject: f1 %.4f < prod %.4f", new_f1, prod_f1)
        return False
    return True


def promote(model, encoder, metrics: dict, model_dir: str):
    """Atomic-ish promotion: write new files, then rename the old model aside."""
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metrics["model_version"] = version

    os.makedirs(model_dir, exist_ok=True)
    current = os.path.join(model_dir, "model.pkl")
    if os.path.exists(current):
        backup = os.path.join(model_dir, f"model_backup_{version}.pkl")
        os.rename(current, backup)
        log.info("backed up previous model to %s", backup)

    try:
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(model_dir, f"model_{version}.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(model_dir, "category_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    except (OSError, pickle.PicklingError):
        # roll back the backup
        log.error("promotion write failed, rolling back to backup")
        if os.path.exists(current):
            os.remove(current)
        backup_path = os.path.join(model_dir, f"model_backup_{version}.pkl")
        if os.path.exists(backup_path):
            os.rename(backup_path, current)
        raise

    log.info("promoted model %s", version)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/compatibility_dataset.csv")
    ap.add_argument("--feedback-dir", default="data/feedback")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--force", action="store_true", help="skip promotion checks")
    args = ap.parse_args()

    try:
        prod = load_prod_metrics(args.model_dir)
        log.info("prod f1 macro: %s", prod.get("f1_macro", "n/a"))

        feedback = load_feedback(args.feedback_dir)
        df = merge(args.data, feedback)
        model, encoder, metrics = train(df)

        if args.force or should_promote(metrics, prod):
            promote(model, encoder, metrics, args.model_dir)
            df.to_csv(args.data, index=False)
            log.info("done: new model in production")
        else:
            log.info("done: kept existing model")
    except Exception as e:
        log.error("retraining failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
