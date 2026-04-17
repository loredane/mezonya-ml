"""Tests for scripts/monitor.py."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import monitor


@pytest.fixture
def reference_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame({col: rng.uniform(0, 1, 200) for col in monitor.FEATURE_COLS})


def test_drift_ks_no_drift(reference_df):
    # current is sampled from the same distribution, no drift expected
    rng = np.random.RandomState(99)
    current = pd.DataFrame({col: rng.uniform(0, 1, 200) for col in monitor.FEATURE_COLS})
    result = monitor.drift_ks(reference_df, current)
    assert result["method"] == "ks_test"
    assert result["drift_detected"] is False


def test_drift_ks_detects_shift(reference_df):
    # shift every feature by +2 standard deviations, should trigger drift
    rng = np.random.RandomState(99)
    current = pd.DataFrame({col: rng.uniform(2, 3, 200) for col in monitor.FEATURE_COLS})
    result = monitor.drift_ks(reference_df, current)
    assert result["drift_detected"] is True
    assert len(result["drifted_columns"]) == len(monitor.FEATURE_COLS)


def test_drift_ks_handles_missing_columns(reference_df):
    # current is missing half the feature columns; shouldn't crash
    partial = reference_df[monitor.FEATURE_COLS[:4]].copy()
    result = monitor.drift_ks(reference_df, partial)
    # method runs; some columns reported as drifted because they're absent
    assert "method" in result


def test_accuracy_no_feedback():
    predictions = pd.DataFrame({"predicted_label": ["compatible"] * 10})
    result = monitor.accuracy_against_installers(predictions)
    assert result["status"] == "no_feedback"


def test_accuracy_all_correct():
    predictions = pd.DataFrame({
        "predicted_label": ["compatible"] * 10,
        "installer_label": ["compatible"] * 10,
    })
    result = monitor.accuracy_against_installers(predictions)
    assert result["accuracy_vs_installers"] == 1.0
    assert result["disagreements"] == 0
    assert result["below_threshold"] is False


def test_accuracy_below_threshold():
    predictions = pd.DataFrame({
        "predicted_label":  ["compatible"] * 5 + ["partial"] * 5,
        "installer_label":  ["incompatible"] * 5 + ["incompatible"] * 5,
    })
    result = monitor.accuracy_against_installers(predictions)
    assert result["accuracy_vs_installers"] == 0.0
    assert result["below_threshold"] is True


def test_run_no_production_data(tmp_path, reference_df):
    ref_path = tmp_path / "ref.csv"
    reference_df.assign(compatibility_label=0).to_csv(ref_path, index=False)

    result = monitor.run(str(ref_path), str(tmp_path / "nonexistent.csv"), str(tmp_path))
    assert result["status"] == "no_production_data"


def test_run_produces_report(tmp_path, reference_df):
    ref_path = tmp_path / "ref.csv"
    reference_df.assign(compatibility_label=0).to_csv(ref_path, index=False)

    prod = reference_df.copy()
    prod["confidence"] = 0.92
    prod["predicted_label"] = "compatible"
    prod_path = tmp_path / "prod.csv"
    prod.to_csv(prod_path, index=False)

    result = monitor.run(str(ref_path), str(prod_path), str(tmp_path))
    assert "data_drift" in result
    assert "prediction_quality" in result
    assert "confidence" in result
    assert result["action"] in ("retrain", "none")

    # report file was written
    reports = list(tmp_path.glob("monitoring_*.json"))
    assert len(reports) == 1
    with open(reports[0]) as f:
        assert json.load(f)["action"] in ("retrain", "none")
