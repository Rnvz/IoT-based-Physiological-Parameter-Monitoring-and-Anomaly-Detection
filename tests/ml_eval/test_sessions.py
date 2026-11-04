"""
Unit tests for ml_eval.sessions: session segmentation and data cleaning.
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from ml_eval.sessions import (
    load_dataset,
    clean_for_modelling,
    segment_sessions,
    summarize_sessions,
    load_and_segment_sessions
)


def test_segment_sessions_boundaries():
    """Verify that gaps exceeding max_gap_seconds create new session_ids."""
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    rows = []

    # Session 1: 5 rows at 1s intervals
    for i in range(5):
        rows.append({
            "ts": (base + timedelta(seconds=i)).isoformat(),
            "hr": 70, "spo2": 98, "temp": 36.0, "finger": True, "scenario": "REST"
        })

    # Gap of 700 seconds (> 600s default) -> Session 2
    sess2_base = base + timedelta(seconds=5 + 700)
    for i in range(5):
        rows.append({
            "ts": (sess2_base + timedelta(seconds=i)).isoformat(),
            "hr": 75, "spo2": 97, "temp": 36.2, "finger": True, "scenario": "TYPING"
        })

    df = pd.DataFrame(rows)
    df = clean_for_modelling(load_dataset(df))
    segmented = segment_sessions(df, max_gap_seconds=600.0)

    assert "session_id" in segmented.columns
    unique_sessions = segmented["session_id"].unique()
    assert len(unique_sessions) == 2
    assert unique_sessions[0] == "session_001"
    assert unique_sessions[1] == "session_002"

    assert (segmented["session_id"] == "session_001").sum() == 5
    assert (segmented["session_id"] == "session_002").sum() == 5


def test_cleaning_drops_finger_false_and_invalid_vitals():
    """Verify that rows with finger=False or biologically impossible vitals are dropped."""
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    test_rows = [
        # 1. Valid row
        {"ts": (base + timedelta(seconds=1)).isoformat(), "hr": 72, "spo2": 98, "temp": 36.0, "finger": True, "scenario": "REST"},
        # 2. finger=False
        {"ts": (base + timedelta(seconds=2)).isoformat(), "hr": 72, "spo2": 98, "temp": 36.0, "finger": False, "scenario": "REST"},
        # 3. Invalid HR too low (<30)
        {"ts": (base + timedelta(seconds=3)).isoformat(), "hr": 10, "spo2": 98, "temp": 36.0, "finger": True, "scenario": "REST"},
        # 4. Invalid HR too high (>220)
        {"ts": (base + timedelta(seconds=4)).isoformat(), "hr": 250, "spo2": 98, "temp": 36.0, "finger": True, "scenario": "REST"},
        # 5. Invalid SpO2 too low (<70)
        {"ts": (base + timedelta(seconds=5)).isoformat(), "hr": 72, "spo2": 60, "temp": 36.0, "finger": True, "scenario": "REST"},
        # 6. Invalid SpO2 too high (>100)
        {"ts": (base + timedelta(seconds=6)).isoformat(), "hr": 72, "spo2": 105, "temp": 36.0, "finger": True, "scenario": "REST"},
        # 7. Invalid Temp too low (<25)
        {"ts": (base + timedelta(seconds=7)).isoformat(), "hr": 72, "spo2": 98, "temp": 20.0, "finger": True, "scenario": "REST"},
        # 8. Invalid Temp too high (>45)
        {"ts": (base + timedelta(seconds=8)).isoformat(), "hr": 72, "spo2": 98, "temp": 50.0, "finger": True, "scenario": "REST"},
        # 9. Valid row
        {"ts": (base + timedelta(seconds=9)).isoformat(), "hr": 74, "spo2": 97, "temp": 36.1, "finger": True, "scenario": "REST"},
    ]

    df = pd.DataFrame(test_rows)
    raw = load_dataset(df)
    clean = clean_for_modelling(raw)

    # Only rows 1 and 9 should survive
    assert len(clean) == 2
    assert list(clean["hr"]) == [72.0, 74.0]


def test_summarize_sessions(synthetic_fixture_df):
    """Verify session summary correctly computes durations and scenario breakdowns."""
    segmented = load_and_segment_sessions(synthetic_fixture_df, max_gap_seconds=600.0)
    summary = summarize_sessions(segmented)

    assert summary["total_sessions"] == 5
    assert summary["total_valid_rows"] == 5 * 80
    assert len(summary["sessions"]) == 5

    for sid, info in summary["sessions"].items():
        assert info["rows"] == 80
        # 80 rows at 1s intervals = 79.0s duration
        assert 78.0 <= info["duration_seconds"] <= 80.0
        assert sum(info["scenarios"].values()) == 80
