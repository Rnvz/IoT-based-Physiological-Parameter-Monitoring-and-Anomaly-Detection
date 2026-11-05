"""
Unit tests for ml_eval.baselines: model training, isolation, and validation threshold calibration.
"""

import pandas as pd
import numpy as np
import pytest

from ml.eval.features import extract_features
from ml.eval.baselines import train_baselines, NORMAL_SCENARIOS


def test_threshold_chosen_on_val_only(synthetic_fixture_df):
    """
    Verify that operating thresholds are calibrated exclusively on validation sessions.
    Varying validation data must alter the threshold; test data must have zero influence.
    """
    from ml.eval.sessions import load_and_segment_sessions
    from ml.eval.split import split_session_ids

    segmented = load_and_segment_sessions(synthetic_fixture_df)
    sessions = sorted(segmented["session_id"].unique())
    partitions = split_session_ids(sessions, seed=42)

    train_raw = segmented[segmented["session_id"].isin(partitions["train"])]
    val_raw = segmented[segmented["session_id"].isin(partitions["val"])].copy()

    train_feats = extract_features(train_raw, feature_set="core")
    val_feats_1 = extract_features(val_raw, feature_set="core")

    # Fit baseline with original val data
    models_1 = train_baselines(train_feats, val_feats_1, feature_set="core", target_fpr=0.05, seed=42)
    th_if_1 = models_1["Isolation Forest"].threshold

    # Modify validation data: shift HR distribution slightly
    val_raw_shifted = val_raw.copy()
    val_raw_shifted["hr"] = val_raw_shifted["hr"] + 15.0
    val_feats_2 = extract_features(val_raw_shifted, feature_set="core")

    models_2 = train_baselines(train_feats, val_feats_2, feature_set="core", target_fpr=0.05, seed=42)
    th_if_2 = models_2["Isolation Forest"].threshold

    # Threshold must have changed because it depends on the validation score percentile
    assert not np.isclose(th_if_1, th_if_2), f"Expected threshold shift, but got {th_if_1} vs {th_if_2}"


def test_training_fits_on_normal_scenarios_only():
    """Verify that models and scaler are fitted only on REST and TYPING rows in train set."""
    # Create train set with REST, TYPING, and MOTION
    rows = []
    from datetime import datetime, timezone, timedelta
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    # 40 rows REST
    for i in range(40):
        rows.append({
            "ts": (base + timedelta(seconds=i)).isoformat(),
            "hr": 70.0, "spo2": 98.0, "temp": 36.0, "finger": True,
            "session_id": "session_001", "scenario": "REST"
        })
    # 40 rows MOTION (extreme values that would skew scaler if included)
    curr = base + timedelta(seconds=50)
    for i in range(40):
        rows.append({
            "ts": (curr + timedelta(seconds=i)).isoformat(),
            "hr": 190.0, "spo2": 85.0, "temp": 39.0, "finger": True,
            "session_id": "session_001", "scenario": "MOTION"
        })

    train_df = extract_features(pd.DataFrame(rows), feature_set="core")

    # Validation set with REST
    val_rows = [
        {"ts": (base + timedelta(seconds=100 + i)).isoformat(), "hr": 72.0, "spo2": 98.0, "temp": 36.2, "finger": True, "session_id": "session_002", "scenario": "REST"}
        for i in range(25)
    ]
    val_df = extract_features(pd.DataFrame(val_rows), feature_set="core")

    models = train_baselines(train_df, val_df, feature_set="core", seed=42)
    scaler = models["Isolation Forest"].scaler

    # The mean of HR in the scaler should reflect REST (around 70), NOT the combined REST + MOTION (around 130)
    hr_scaler_mean = scaler.mean_[0]  # hr is index 0
    assert hr_scaler_mean < 80.0, f"Scaler mean {hr_scaler_mean} indicates MOTION data leaked into scaler!"


def test_rule_engine_baseline():
    """Verify Rule Engine wrapper produces standard 0.0, 0.5, 1.0 scores and predicts correctly."""
    from datetime import datetime, timezone, timedelta
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Normal row: HR 75, SpO2 98, Temp 36.5
    # Critical row: HR 180 (severe tachycardia), SpO2 85 (severe hypoxemia)
    rows = [
        {"ts": (base + timedelta(seconds=i)).isoformat(), "hr": 75.0, "spo2": 98.0, "temp": 36.5, "finger": True, "session_id": "s1", "scenario": "REST"}
        for i in range(16)
    ]
    rows.append(
        {"ts": (base + timedelta(seconds=16)).isoformat(), "hr": 180.0, "spo2": 85.0, "temp": 36.5, "finger": True, "session_id": "s1", "scenario": "REST"}
    )
    df = extract_features(pd.DataFrame(rows), feature_set="core")

    val_df = df.copy()
    models = train_baselines(df, val_df, feature_set="core", seed=42)
    rule_model = models["Rule Engine"]

    scores = rule_model.score(df)
    preds = rule_model.predict(df)

    # Initial normal rows should have score 0.0, prediction 0
    assert scores[0] == 0.0
    assert preds[0] == 0

    # Critical row should have score 1.0, prediction 1
    assert scores[-1] == 1.0
    assert preds[-1] == 1
