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


def test_validate_recording_fixture_rejected_on_default_300_rows():
    """Verify that by default (min_rows=300), sample_fixture.csv is REJECTED because each session has only 100 rows."""
    assert os.path.exists(FIXTURE_PATH), "Fixture sample_fixture.csv must exist."
    res = validate_recording(FIXTURE_PATH, is_fixture=True)
    assert res["status"] == "REJECT"
    assert any("kurang dari batas minimum (300 baris)" in r for r in res["reasons"])


def test_validate_recording_fixture_accepted_with_explicit_min_rows_100():
    """Verify that sample_fixture.csv passes when passing min_rows=100."""
    assert os.path.exists(FIXTURE_PATH), "Fixture sample_fixture.csv must exist."
    res = validate_recording(FIXTURE_PATH, min_rows=100, is_fixture=True)

    assert res["status"] == "ACCEPT"
    assert len(res["reasons"]) == 0
    assert res["total_rows"] == 500
    assert res["total_sessions"] == 5

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


def test_validate_recording_rejects_gap_gt_3s(tmp_path):
    """CASE 1: Rejection when gap > 3.0s occurs."""
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    curr_t = base_t
    for i in range(350):
        if i == 50:
            curr_t += timedelta(seconds=5)  # 5s gap > 3.0s
        else:
            curr_t += timedelta(seconds=1)
        ts_str = curr_t.isoformat()
        lines.append(f"{ts_str},75.0,98.0,36.5,50.0,24.0,True")
    csv_file.write_text("\n".join(lines) + "\n")

    res = validate_recording(str(csv_file), min_rows=300)
    assert res["status"] == "REJECT"
    s1 = res["sessions"]["session_001"]
    assert s1["gaps_gt_3s_count"] == 1
    assert any("jeda waktu > 3.0 s" in r for r in s1["reasons"])


def test_validate_recording_rejects_low_finger_pct(tmp_path):
    """CASE 2: Rejection when finger=True percentage < 95.0%."""
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(350):
        t = (base_t + timedelta(seconds=i)).isoformat()
        finger = "False" if i % 10 == 0 else "True"  # 90% true < 95%
        lines.append(f"{t},75.0,98.0,36.5,50.0,24.0,{finger}")
    csv_file.write_text("\n".join(lines) + "\n")

    res = validate_recording(str(csv_file), min_rows=300)
    assert res["status"] == "REJECT"
    s1 = res["sessions"]["session_001"]
    assert s1["finger_true_pct"] < 95.0
    assert any("finger=True" in r for r in s1["reasons"])


def test_validate_recording_rejects_wrong_or_mixed_device_id(tmp_path):
    """CASE 3: Rejection on wrong or mixed device_id."""
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

    # Subcase A: Wrong device_id
    csv_wrong = tmp_path / "wrong.csv"
    lines_wrong = ["ts,device_id,hr,spo2,temp,hum,room_temp,finger"]
    for i in range(350):
        t = (base_t + timedelta(seconds=i)).isoformat()
        lines_wrong.append(f"{t},esp32-sim,75.0,98.0,36.5,50.0,24.0,True")
    csv_wrong.write_text("\n".join(lines_wrong) + "\n")

    res_wrong = validate_recording(str(csv_wrong), min_rows=300, expected_device_id="esp32-01")
    assert res_wrong["status"] == "REJECT"
    s1_w = res_wrong["sessions"]["session_001"]
    assert s1_w["unique_device_ids"] == ["esp32-sim"]
    assert any("device_id tidak sesuai" in r for r in s1_w["reasons"])

    # Subcase B: Mixed device_id in a single session
    csv_mixed = tmp_path / "mixed.csv"
    lines_mixed = ["ts,device_id,hr,spo2,temp,hum,room_temp,finger"]
    for i in range(350):
        t = (base_t + timedelta(seconds=i)).isoformat()
        dev = "esp32-01" if i < 200 else "esp32-02"
        lines_mixed.append(f"{t},{dev},75.0,98.0,36.5,50.0,24.0,True")
    csv_mixed.write_text("\n".join(lines_mixed) + "\n")

    res_mixed = validate_recording(str(csv_mixed), min_rows=300)
    assert res_mixed["status"] == "REJECT"
    s1_m = res_mixed["sessions"]["session_001"]
    assert len(s1_m["unique_device_ids"]) == 2
    assert any("campuran device_id" in r for r in s1_m["reasons"])

    # Subcase C: Correct device_id matches expected
    csv_correct = tmp_path / "correct.csv"
    lines_correct = ["ts,device_id,hr,spo2,temp,hum,room_temp,finger"]
    for i in range(350):
        t = (base_t + timedelta(seconds=i)).isoformat()
        lines_correct.append(f"{t},esp32-01,75.0,98.0,36.5,50.0,24.0,True")
    csv_correct.write_text("\n".join(lines_correct) + "\n")

    res_correct = validate_recording(str(csv_correct), min_rows=300, expected_device_id="esp32-01")
    assert res_correct["status"] == "ACCEPT"
    assert res_correct["sessions"]["session_001"]["unique_device_ids"] == ["esp32-01"]


def test_validate_recording_rejects_zero_vitals(tmp_path):
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(350):
        t = (base_t + timedelta(seconds=i)).isoformat()
        hr = 0.0 if i % 20 == 0 else 75.0  # 5% zero vitals
        lines.append(f"{t},{hr},98.0,36.5,50.0,24.0,True")
    csv_file.write_text("\n".join(lines) + "\n")

    res = validate_recording(str(csv_file), min_rows=300)
    assert res["status"] == "REJECT"
    s1 = res["sessions"]["session_001"]
    assert s1["zero_vitals_pct"] >= 2.0
    assert any("sensor vital" in r for r in s1["reasons"])


def test_validate_recording_null_dht_and_allow_flag(tmp_path):
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger"]
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i in range(350):
        t = (base_t + timedelta(seconds=i)).isoformat()
        lines.append(f"{t},75.0,98.0,36.5,,,True")
    csv_file.write_text("\n".join(lines) + "\n")

    # Without allow flag -> REJECT
    res_reject = validate_recording(str(csv_file), min_rows=300, allow_null_dht=False)
    assert res_reject["status"] == "REJECT"
    s1 = res_reject["sessions"]["session_001"]
    assert any("Kelembapan" in r for r in s1["reasons"])

    # With allow flag -> ACCEPT
    res_accept = validate_recording(str(csv_file), min_rows=300, allow_null_dht=True)
    assert res_accept["status"] == "ACCEPT"


def test_validate_recording_scenario_label_check(tmp_path):
    """Verify that expected scenario label matches or rejects if mismatched."""
    base_t = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    csv_file = tmp_path / "readings.csv"
    lines = ["ts,hr,spo2,temp,hum,room_temp,finger,scenario"]
    for i in range(350):
        t = (base_t + timedelta(seconds=i)).isoformat()
        lines.append(f"{t},75.0,98.0,36.5,50.0,24.0,True,REST")
    csv_file.write_text("\n".join(lines) + "\n")

    # Matching scenario -> ACCEPT
    res_ok = validate_recording(str(csv_file), min_rows=300, expected_scenario="REST")
    assert res_ok["status"] == "ACCEPT"
    assert res_ok["sessions"]["session_001"]["scenario_label"] == "REST"

    # Mismatched scenario -> REJECT
    res_bad = validate_recording(str(csv_file), min_rows=300, expected_scenario="TYPING")
    assert res_bad["status"] == "REJECT"
    assert any("Label skenario tidak sesuai" in r for r in res_bad["sessions"]["session_001"]["reasons"])
