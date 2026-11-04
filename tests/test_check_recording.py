import os
import pytest
import pandas as pd
from scripts.check_recording import analyze_recording, resolve_csv_path

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_fixture.csv")


def test_resolve_csv_path_file():
    resolved = resolve_csv_path(FIXTURE_PATH)
    assert os.path.isfile(resolved)
    assert resolved.endswith("sample_fixture.csv")


def test_resolve_csv_path_dir(tmp_path):
    csv_file = tmp_path / "readings.csv"
    csv_file.write_text("ts,hr,spo2,temp,finger\n2026-01-01T00:00:00Z,75,98,36.5,True\n")
    resolved = resolve_csv_path(str(tmp_path))
    assert resolved == str(csv_file)


def test_analyze_recording_on_fixture():
    assert os.path.exists(FIXTURE_PATH), "Fixture sample_fixture.csv must exist."
    res = analyze_recording(FIXTURE_PATH)

    assert res["total_rows"] == 500
    assert res["valid_rows"] == 500
    assert res["valid_ratio"] == 100.0
    assert res["null_percentages"]["hum"] == 0.0
    assert res["null_percentages"]["room_temp"] == 0.0

    # Sensor stats
    stats = res["sensor_stats"]
    assert "HR" in stats and stats["HR"]["count"] == 500
    assert 70.0 <= stats["HR"]["mean"] <= 85.0
    assert stats["HR"]["std"] > 0.5

    assert "SpO2" in stats and stats["SpO2"]["count"] == 500
    assert 95.0 <= stats["SpO2"]["mean"] <= 100.0

    assert "temp" in stats and stats["temp"]["count"] == 500
    assert 35.0 <= stats["temp"]["mean"] <= 38.0

    # No stuck sensor
    assert res["stuck_rows_count"] == 0
    assert res["stuck_streaks_count"] == 0

    # ML eval qualification
    assert res["ml_eval_qualified"] is True
    assert len(res["ml_eval_reasons"]) == 0


def test_analyze_recording_detects_simulator(tmp_path):
    sim_csv = tmp_path / "readings.csv"
    # Create synthetic data with simulator device_id and zero temperature variance
    lines = ["ts,device_id,hr,spo2,temp,room_temp,hum,finger"]
    for i in range(40):
        lines.append(f"2026-01-01T08:00:{i:02d}Z,esp32-sim,75.0,98.0,36.5,24.0,50.0,True")
    sim_csv.write_text("\n".join(lines) + "\n")

    res = analyze_recording(str(sim_csv))
    assert res["total_rows"] == 40
    assert res["simulator_indicators"]["is_simulator"] is True
    assert any("simulator" in r.lower() or "device_id" in r.lower() for r in res["simulator_indicators"]["reasons"])


def test_analyze_recording_detects_stuck_sensor(tmp_path):
    stuck_csv = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,finger"]
    # 40 consecutive seconds with exact same vitals
    for i in range(40):
        lines.append(f"2026-01-01T08:00:{i:02d}Z,75.12,98.00,36.50,True")
    stuck_csv.write_text("\n".join(lines) + "\n")

    res = analyze_recording(str(stuck_csv))
    assert res["stuck_streaks_count"] >= 1
    assert res["stuck_rows_count"] >= 10


def test_analyze_recording_short_session_qualification(tmp_path):
    short_csv = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,finger"]
    for i in range(10):
        lines.append(f"2026-01-01T08:00:{i:02d}Z,75.0,98.0,36.5,True")
    short_csv.write_text("\n".join(lines) + "\n")

    res = analyze_recording(str(short_csv))
    assert res["valid_rows"] == 10
    assert res["ml_eval_qualified"] is False
    assert any("window_size" in r for r in res["ml_eval_reasons"])
