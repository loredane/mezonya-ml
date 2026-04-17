"""Tests for scripts/retrain.py."""
import json
import pickle

import pandas as pd
import pytest

from scripts import retrain


class DummyModel:
    """Module-level stand-in so pickle can resolve it by qualified name."""
    pass


class DummyEncoder:
    """Module-level stand-in so pickle can resolve it by qualified name."""
    pass


def make_dataset(tmp_path, n=200):
    """Build a minimally valid compatibility dataset."""
    rows = []
    for i in range(n):
        label = i % 3
        rows.append({
            "device_a_id": i,
            "device_b_id": i + 1000,
            "device_a_category": ["security", "lighting", "climate"][i % 3],
            "device_b_category": ["security", "lighting", "audio"][i % 3],
            "protocol_overlap": (i % 10) / 10,
            "ecosystem_overlap": ((i + 3) % 10) / 10,
            "same_brand": i % 2,
            "hub_conflict": (i + 1) % 2,
            "cloud_compatible": i % 2,
            "category_synergy": 0.5 + (i % 5) * 0.1,
            "both_hub_required": 0,
            "total_protocols": 2,
            "total_ecosystems": 3,
            "device_a_hub_required": 0,
            "device_b_hub_required": 0,
            "device_a_cloud_required": 0,
            "device_b_cloud_required": 0,
            "compatibility_label": label,
        })
    path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_prod_metrics_missing(tmp_path):
    assert retrain.load_prod_metrics(str(tmp_path)) == {"f1_macro": 0.0}


def test_load_prod_metrics_corrupt(tmp_path):
    (tmp_path / "metrics.json").write_text("not json at all {{{")
    # should not crash, returns a safe default
    result = retrain.load_prod_metrics(str(tmp_path))
    assert result["f1_macro"] == 0.0


def test_load_feedback_empty(tmp_path):
    assert retrain.load_feedback(str(tmp_path)).empty


def test_load_feedback_skips_corrupt(tmp_path):
    # good file
    pd.DataFrame([{"device_a_id": 1, "device_b_id": 2, "compatibility_label": 0}]).to_csv(
        tmp_path / "feedback_good.csv", index=False
    )
    # corrupt file
    (tmp_path / "feedback_bad.csv").write_text('"""not a csv"""')
    df = retrain.load_feedback(str(tmp_path))
    # the good row should be there, corrupt file logged and skipped
    assert len(df) == 1


def test_merge_no_feedback(tmp_path):
    path = make_dataset(tmp_path, n=20)
    result = retrain.merge(str(path), pd.DataFrame())
    assert len(result) == 20


def test_merge_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, pd.errors.ParserError, OSError)):
        retrain.merge(str(tmp_path / "missing.csv"), pd.DataFrame())


def test_merge_deduplicates(tmp_path):
    base = make_dataset(tmp_path, n=10)
    feedback = pd.DataFrame([
        {"device_a_id": 0, "device_b_id": 1000, "compatibility_label": 2,
         "device_a_category": "security", "device_b_category": "security"},
    ])
    merged = retrain.merge(str(base), feedback)
    # row with device_a_id=0 should now have the updated label
    row = merged[(merged["device_a_id"] == 0) & (merged["device_b_id"] == 1000)].iloc[-1]
    assert row["compatibility_label"] == 2


def test_train_rejects_tiny_dataset(tmp_path):
    df = pd.DataFrame([{
        "device_a_category": "security", "device_b_category": "lighting",
        "compatibility_label": 0, **{c: 0 for c in retrain.FEATURES}
    }] * 5)
    with pytest.raises(ValueError, match="too small"):
        retrain.train(df)


def test_train_rejects_missing_columns():
    df = pd.DataFrame({"foo": range(100)})
    with pytest.raises(ValueError, match="missing"):
        retrain.train(df)


def test_should_promote_rejects_below_floor():
    assert retrain.should_promote({"f1_macro": 0.80}, {"f1_macro": 0.0}) is False


def test_should_promote_rejects_worse_than_prod():
    assert retrain.should_promote({"f1_macro": 0.86}, {"f1_macro": 0.90}) is False


def test_should_promote_accepts_improvement():
    assert retrain.should_promote({"f1_macro": 0.92}, {"f1_macro": 0.88}) is True


def test_promote_creates_backup(tmp_path):
    # existing model
    with open(tmp_path / "model.pkl", "wb") as f:
        pickle.dump(DummyModel(), f)

    retrain.promote(DummyModel(), DummyEncoder(), {"f1_macro": 0.9}, str(tmp_path))

    files = [f.name for f in tmp_path.iterdir()]
    assert "model.pkl" in files
    assert "category_encoder.pkl" in files
    assert "metrics.json" in files
    assert any(f.startswith("model_backup_") for f in files)
    assert any(f.startswith("model_") and not f.startswith("model_backup_") and f != "model.pkl" for f in files)
