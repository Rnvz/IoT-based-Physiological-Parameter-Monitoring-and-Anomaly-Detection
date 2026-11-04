import os
from datetime import datetime, timezone, timedelta
import pytest
import pandas as pd
from scripts.validate_recording import validate_recording, resolve_csv_path

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_fixture.csv")


def test_resolve_csv_path_file():
    resolved = resolve_csv_path(FIXTURE_PATH)
    assert os.path.isfile(resolved)
    assert resolved.endswith("sample_fixture.csv")


def test_resolve_csv_path_dir(tmp_path):
    csv_file = tmp_path / "readings.csv"
    csv_file.write_text("ts,hr,spo2,temp,hum,room_temp,finger\n2026-01-01T00:00:00Z,75,98,36.5,50,24,True\n")
    resolved = resolve_csv_path(str(tmp_path))
    assert resolved == str(csv_file)


def test_validate_recording_on_fixture():
    assert os.path.exists(FIXTURE_PATH), "Fixture sample_fixture.csv must exist."
    res = validate_recording(FIXTURE_PATH, min_rows=100, is_fixture=True)

    # Fixture validation assertions
    assert res["status"] == "ACCEPT"
    assert len(res["reasons"]) == 0
    assert res["total_rows"] == 500
    assert res["total_sessions"] == 5

    # Check session_001
    s1 = res["sessions"]["session_001"]
    assert s1["status"] == "ACCEPT"
    assert s1["total_rows"] == 100
    assert s1["valid_rows"] == 100
    assert s1["gaps_gt_3s_count"] == 0
    assert s1["finger_true_pct"] == 100.0
    assert s1["zero_vitals_pct"] == 0.0
    assert s1["null_hum_pct"] == 0.0
    assert s1["null_room_temp_pct"] == 0.0
    assert s1["duration_seconds"] == 99.0

    # Feature ranges check
    ranges = s1["feature_ranges"]
    assert "HR" in ranges and ranges["HR"]["count"] == 100
    assert 60.0 <= ranges["HR"]["min"] <= ranges["HR"]["max"] <= 90.0
    assert "SpO2" in ranges and ranges["SpO2"]["count"] == 100
    assert 95.0 <= ranges["SpO2"]["min"] <= ranges["SpO2"]["max"] <= 100.0


def test_validate_recording_rejects_gap_gt_3s(tmp_path):
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    # 150 rows with one 5-second gap at index 50
    curr_t = base_t
    for i in range(150):
        if i == 50:
            curr_t += timedelta(seconds=5)
        else:
            curr_t += timedelta(seconds=1)
        ts_str = curr_t.isoformat()
        lines.append(f"{ts_str},75.0,98.0,36.5,50.0,24.0,True")
    csv_file.write_text("\n".join(lines) + "\n")

    res = validate_recording(str(csv_file), min_rows=100)
    assert res["status"] == "REJECT"
    s1 = res["sessions"]["session_001"]
    assert s1["gaps_gt_3s_count"] == 1
    assert any("jeda waktu > 3.0 s" in r for r in s1["reasons"])


def test_validate_recording_rejects_low_finger_pct(tmp_path):
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(150):
        t = (base_t + timedelta(seconds=i)).isoformat()
        # 10% finger=False -> finger_true_pct = 90% (< 95%)
        finger = "False" if i % 10 == 0 else "True"
        lines.append(f"{t},75.0,98.0,36.5,50.0,24.0,{finger}")
    csv_file.write_text("\n".join(lines) + "\n")

    res = validate_recording(str(csv_file), min_rows=100)
    assert res["status"] == "REJECT"
    s1 = res["sessions"]["session_001"]
    assert s1["finger_true_pct"] < 95.0
    assert any("finger=True" in r for r in s1["reasons"])


def test_validate_recording_rejects_zero_vitals(tmp_path):
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(150):
        t = (base_t + timedelta(seconds=i)).isoformat()
        # 5% zero vitals -> zero_vitals_pct = 5% (>= 2%)
        hr = 0.0 if i % 20 == 0 else 75.0
        lines.append(f"{t},{hr},98.0,36.5,50.0,24.0,True")
    csv_file.write_text("\n".join(lines) + "\n")

    res = validate_recording(str(csv_file), min_rows=100)
    assert res["status"] == "REJECT"
    s1 = res["sessions"]["session_001"]
    assert s1["zero_vitals_pct"] >= 2.0
    assert any("sensor vital" in r for r in s1["reasons"])


def test_validate_recording_null_dht_and_allow_flag(tmp_path):
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(150):
        t = (base_t + timedelta(seconds=i)).isoformat()
        lines.append(f"{t},75.0,98.0,36.5,,,True")
    csv_file.write_text("\n".join(lines) + "\n")

    # Without allow flag -> REJECT
    res_reject = validate_recording(str(csv_file), min_rows=100, allow_null_dht=False)
    assert res_reject["status"] == "REJECT"
    s1 = res_reject["sessions"]["session_001"]
    assert any("Kelembapan" in r for r in s1["reasons"])

    # With allow flag -> ACCEPT
    res_accept = validate_recording(str(csv_file), min_rows=100, allow_null_dht=True)
    assert res_accept["status"] == "ACCEPT"


def test_validate_recording_rejects_short_length(tmp_path):
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(50):
        t = (base_t + timedelta(seconds=i)).isoformat()
        lines.append(f"{t},75.0,98.0,36.5,50.0,24.0,True")
    csv_file.write_text("\n".join(lines) + "\n")

    res = validate_recording(str(csv_file), min_rows=100)
    assert res["status"] == "REJECT"
    s1 = res["sessions"]["session_001"]
    assert s1["valid_rows"] == 50
    assert any("kurang dari batas minimum" in r for r in s1["reasons"])
