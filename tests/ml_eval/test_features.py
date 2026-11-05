"""
Unit tests for ml_eval.features: causal rolling window feature computation.
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import pytest

from ml.eval.features import (
    extract_session_features,
    extract_features,
    CORE_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
    DEFAULT_WINDOW_SIZE
)


def test_warmup_rows_dropped_never_zero_filled():
    """Verify that the first window_size - 1 rows are dropped and zero-filling is never used."""
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    N = 25
    rows = []
    for i in range(N):
        rows.append({
            "ts": (base + timedelta(seconds=i)).isoformat(),
            "hr": 70.0 + i,
            "spo2": 98.0,
            "temp": 36.0,
            "finger": True,
            "session_id": "session_001",
            "scenario": "REST"
        })

    df = pd.DataFrame(rows)
    feats = extract_session_features(df, window_size=15, feature_set="core")

    # With window_size=15, first 14 rows are warm-up, so 25 - 14 = 11 rows should remain
    assert len(feats) == N - (DEFAULT_WINDOW_SIZE - 1)
    # The first row in feats must correspond to the 15th row (index 14) of df
    assert feats.iloc[0]["ts"] == rows[14]["ts"]
    assert feats.iloc[0]["hr"] == 70.0 + 14

    # Verify that no rolling variance or mean is computed with zeros
    # For a ramp 70, 71, ..., 84: mean is (70+84)/2 = 77.0
    expected_mean = np.mean([70.0 + i for i in range(15)])
    assert np.isclose(feats.iloc[0]["hr_mean"], expected_mean)


def test_no_cross_session_windows():
    """Verify that features NEVER cross session boundaries."""
    base1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    base2 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    rows1 = [
        {"ts": (base1 + timedelta(seconds=i)).isoformat(), "hr": 60.0, "spo2": 99.0, "temp": 36.0, "finger": True, "session_id": "session_001", "scenario": "REST"}
        for i in range(20)
    ]
    # Session 2 has a very different HR: 120.0
    rows2 = [
        {"ts": (base2 + timedelta(seconds=i)).isoformat(), "hr": 120.0, "spo2": 95.0, "temp": 37.0, "finger": True, "session_id": "session_002", "scenario": "TYPING"}
        for i in range(20)
    ]

    df = pd.DataFrame(rows1 + rows2)
    feats = extract_features(df, window_size=15, feature_set="core")

    sess1_feats = feats[feats["session_id"] == "session_001"]
    sess2_feats = feats[feats["session_id"] == "session_002"]

    assert len(sess1_feats) == 20 - 14  # 6 rows
    assert len(sess2_feats) == 20 - 14  # 6 rows

    # First row of session 2 must NOT have any influence from session 1 (HR 60.0)
    # If session 1 leaked, the mean would be much lower than 120.0
    assert np.isclose(sess2_feats.iloc[0]["hr_mean"], 120.0)
    assert np.isclose(sess2_feats.iloc[0]["hr_var"], 0.0)


def test_intra_session_gap_resets_buffer():
    """Verify that an intra-session sample gap > 5 seconds resets the window buffer."""
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    rows = []

    # Segment A: 20 rows
    for i in range(20):
        rows.append({
            "ts": (base + timedelta(seconds=i)).isoformat(),
            "hr": 70.0, "spo2": 98.0, "temp": 36.0, "finger": True,
            "session_id": "session_001", "scenario": "REST"
        })

    # Gap of 10 seconds (> 5s) inside the same session
    curr = base + timedelta(seconds=20 + 10)
    # Segment B: 20 rows
    for i in range(20):
        rows.append({
            "ts": (curr + timedelta(seconds=i)).isoformat(),
            "hr": 80.0, "spo2": 98.0, "temp": 36.0, "finger": True,
            "session_id": "session_001", "scenario": "REST"
        })

    df = pd.DataFrame(rows)
    feats = extract_session_features(df, window_size=15, max_gap_seconds=5.0)

    # Segment A produced 20 - 14 = 6 rows.
    # Buffer was reset at the gap, so Segment B needs 14 warm-up rows before producing 6 rows.
    # Total rows = 6 + 6 = 12.
    assert len(feats) == 12


def test_feature_sets_columns():
    """Verify core (12) and extended (24) feature columns are present."""
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "ts": (base + timedelta(seconds=i)).isoformat(),
            "hr": 72.0, "spo2": 98.0, "temp": 36.5, "room_temp": 24.0, "hum": 50.0,
            "finger": True, "session_id": "session_001", "scenario": "REST"
        }
        for i in range(25)
    ]
    df = pd.DataFrame(rows)

    core_feats = extract_features(df, feature_set="core")
    ext_feats = extract_features(df, feature_set="extended")

    for col in CORE_FEATURE_NAMES:
        assert col in core_feats.columns

    for col in EXTENDED_FEATURE_NAMES:
        assert col in ext_feats.columns
