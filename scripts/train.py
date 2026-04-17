"""Train the compatibility classifier."""
import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

try:
    import mlflow
    import mlflow.xgboost
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


FEATURES = [
    "protocol_overlap",
    "ecosystem_overlap",
    "same_brand",
    "hub_conflict",
    "cloud_compatible",
    "category_synergy",
    "both_hub_required",
    "total_protocols",
    "total_ecosystems",
    "device_a_hub_required",
    "device_b_hub_required",
    "device_a_cloud_required",
    "device_b_cloud_required",
    "device_a_category_encoded",
    "device_b_category_encoded",
]

LABEL = "compatibility_label"
CLASSES = ["incompatible", "partial", "compatible"]

PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}


def prepare(df):
    df = df.copy()
    enc = LabelEncoder()
    enc.fit(pd.concat([df["device_a_category"], df["device_b_category"]]))
    df["device_a_category_encoded"] = enc.transform(df["device_a_category"])
    df["device_b_category_encoded"] = enc.transform(df["device_b_category"])

    X = df[FEATURES].values
    y = df[LABEL].values
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), enc


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")

    print(f"accuracy  {acc:.4f}")
    print(f"f1 macro  {f1m:.4f}")
    print(f"f1 weight {f1w:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    print(confusion_matrix(y_test, y_pred))

    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\ntop features:")
    for name, val in imp[:5]:
        print(f"  {name:32s} {val:.4f}")

    conf = y_proba.max(axis=1)
    print(f"\nmean confidence   {conf.mean():.4f}")
    print(f"confidence > 0.8  {(conf > 0.8).mean():.1%}")

    return {
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "mean_confidence": float(conf.mean()),
        "feature_importance": dict(imp),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save(model, encoder, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metrics["model_version"] = version

    with open(f"{out_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{out_dir}/model_{version}.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{out_dir}/category_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(f"{out_dir}/model_config.json", "w") as f:
        json.dump({
            "feature_columns": FEATURES,
            "label_names": CLASSES,
            "model_version": version,
            "xgboost_params": PARAMS,
        }, f, indent=2)

    return version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/compatibility_dataset.csv")
    ap.add_argument("--output", default="models")
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"loaded {len(df)} pairs from {args.data}")

    (X_train, X_test, y_train, y_test), encoder = prepare(df)
    print(f"train {len(X_train)}  test {len(X_test)}")

    # 5-fold CV before fitting the final model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        xgb.XGBClassifier(**PARAMS), X_train, y_train, cv=cv, scoring="f1_macro"
    )
    print(f"cv f1 macro  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    metrics = evaluate(model, X_test, y_test)
    metrics["cv_f1_macro"] = float(cv_scores.mean())

    if args.mlflow and HAS_MLFLOW:
        mlflow.set_experiment("mezonya-compatibility")
        with mlflow.start_run():
            mlflow.log_params(PARAMS)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.xgboost.log_model(model, "model")

    version = save(model, encoder, metrics, args.output)
    print(f"\nsaved to {args.output}/ (version {version})")


if __name__ == "__main__":
    main()
