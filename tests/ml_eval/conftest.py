"""
Synthetic test fixtures for ml_eval unit and integration tests.
Guarantees 4+ sessions, realistic timestamps, physiological bounds, and known boundaries.
"""

import os
from datetime import datetime, timezone, timedelta
import pytest
import pandas as pd
import numpy as np


def make_synthetic_sessions(
    n_sessions: int = 5,
    rows_per_session: int = 80,
    gap_minutes: float = 15.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic session fixture data with:
    - >= 4 sessions
    - Realistic physiological values (HR 60-90, SpO2 96-99, Temp 35-37)
    - Ambient room_temp and hum
    - 1-second sampling rate within session
    - Contiguous time gap (> gap_minutes) between sessions
    - Alternating standard scenarios ('REST', 'TYPING')
    """
    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    scenarios = ["REST", "TYPING", "REST", "TYPING", "REST"]

    all_rows = []
    current_time = base_time

    for s_idx in range(n_sessions):
        scen = scenarios[s_idx % len(scenarios)]
        base_hr = rng.uniform(68.0, 78.0)
        base_spo2 = rng.uniform(97.0, 99.0)
        base_temp = rng.uniform(35.5, 36.5)

        for r_idx in range(rows_per_session):
            ts = current_time.isoformat()
            hr = float(np.clip(base_hr + rng.normal(0, 1.2), 45.0, 150.0))
            spo2 = float(np.clip(base_spo2 + rng.normal(0, 0.4), 85.0, 100.0))
            temp = float(np.clip(base_temp + rng.normal(0, 0.1), 30.0, 40.0))
            room_temp = float(np.clip(24.0 + rng.normal(0, 0.2), 15.0, 35.0))
            hum = float(np.clip(50.0 + rng.normal(0, 0.5), 20.0, 80.0))

            all_rows.append({
                "ts": ts,
                "hr": round(hr, 2),
                "spo2": round(spo2, 2),
                "temp": round(temp, 2),
                "room_temp": round(room_temp, 2),
                "hum": round(hum, 2),
                "finger": True,
                "scenario": scen
            })
            current_time += timedelta(seconds=1.0)

        # Gap between sessions: gap_minutes
        current_time += timedelta(minutes=gap_minutes)

    df = pd.DataFrame(all_rows)
    return df


@pytest.fixture
def synthetic_fixture_df():
    """Provides an in-memory synthetic fixture DataFrame with 5 sessions."""
    return make_synthetic_sessions(n_sessions=5, rows_per_session=80, gap_minutes=15.0, seed=42)


@pytest.fixture
def synthetic_fixture_csv(tmp_path, synthetic_fixture_df):
    """Provides path to a temporary synthetic fixture CSV file."""
    path = tmp_path / "synthetic_fixture.csv"
    synthetic_fixture_df.to_csv(path, index=False)
    return str(path)
