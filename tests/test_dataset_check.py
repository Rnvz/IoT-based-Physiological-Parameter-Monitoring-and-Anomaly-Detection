import os
import csv
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from scripts.dataset_check import analyze_dataset

def create_temp_csv(rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = ["ts", "device_id", "hr", "spo2", "temp", "hum", "room_temp", "finger", "fw_status"]
    tf = tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False)
    writer = csv.DictWriter(tf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    tf.close()
    return tf.name

def test_empty_dataset():
    fpath = create_temp_csv([])
    try:
        res = analyze_dataset(fpath)
        assert res["total_rows"] == 0
        assert res["valid_rows"] == 0
        assert res["pass_check"] is False
        assert res["target_met"] is False
    finally:
        os.remove(fpath)

def test_short_dataset_fails_target():
    base_time = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(10):
        t = (base_time + timedelta(seconds=i)).isoformat()
        rows.append({
            "ts": t,
            "device_id": "esp32-01",
            "hr": "75.0",
            "spo2": "98.0",
            "temp": "33.0",
            "hum": "60.0",
            "room_temp": "28.0",
            "finger": "True",
            "fw_status": "NORMAL"
        })
    fpath = create_temp_csv(rows)
    try:
        res = analyze_dataset(fpath)
        assert res["total_rows"] == 10
        assert res["valid_rows"] == 10
        assert res["invalid_rows"] == 0
        assert res["target_met"] is False
        assert res["pass_check"] is False
        assert res["longest_valid_stretch"] == 10
        assert res["burst_detected"] is False
    finally:
        os.remove(fpath)

def test_dataset_meets_target_pass():
    base_time = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(1005):
        t = (base_time + timedelta(seconds=i)).isoformat()
        rows.append({
            "ts": t,
            "device_id": "esp32-01",
            "hr": "72.0",
            "spo2": "98.0",
            "temp": "33.5",
            "hum": "60.0",
            "room_temp": "28.0",
            "finger": "True",
            "fw_status": "NORMAL"
        })
    fpath = create_temp_csv(rows)
    try:
        res = analyze_dataset(fpath)
        assert res["total_rows"] == 1005
        assert res["valid_rows"] == 1005
        assert res["target_met"] is True
        assert res["pass_check"] is True
        assert res["longest_valid_stretch"] == 1005
        assert res["burst_detected"] is False
    finally:
        os.remove(fpath)

def test_invalid_rows_and_scenarios():
    base_time = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    fieldnames = [
        "ts", "hr", "spo2", "temp", "hum", "room_temp", "finger",
        "fw_status", "rule_severity", "ml_severity", "published_severity",
        "decision_mode", "scenario", "mode", "led_color", "brightness",
        "buzzer_test", "source"
    ]
    rows = [
        {"ts": (base_time + timedelta(seconds=0)).isoformat(), "hr": "70", "spo2": "98", "temp": "33.0", "finger": "True", "scenario": "REST"},
        {"ts": (base_time + timedelta(seconds=1)).isoformat(), "hr": "71", "spo2": "98", "temp": "33.1", "finger": "True", "scenario": "REST"},
        {"ts": (base_time + timedelta(seconds=2)).isoformat(), "hr": "0", "spo2": "0", "temp": "0", "finger": "False", "scenario": "SENSOR_OFF"},
        {"ts": (base_time + timedelta(seconds=3)).isoformat(), "hr": "null", "spo2": "98", "temp": "33.0", "finger": "True", "scenario": "REST"},
        {"ts": (base_time + timedelta(seconds=4)).isoformat(), "hr": "75", "spo2": "97", "temp": "33.2", "finger": "True", "scenario": "TYPING"},
    ]
    fpath = create_temp_csv(rows, fieldnames=fieldnames)
    try:
        res = analyze_dataset(fpath)
        assert res["total_rows"] == 5
        assert res["valid_rows"] == 3
        assert res["invalid_rows"] == 2
        assert res["has_scenario_col"] is True
        assert res["scenario_counts"]["REST"] == 2
        assert res["scenario_counts"]["TYPING"] == 1
        assert "SENSOR_OFF" not in res["scenario_counts"]
        assert res["longest_valid_stretch"] == 2
    finally:
        os.remove(fpath)

def test_burst_detection_and_duplicate_timestamps():
    base_time = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(5):
        t = (base_time + timedelta(milliseconds=100 * i)).isoformat()
        rows.append({
            "ts": t,
            "device_id": "esp32-01",
            "hr": "72.0",
            "spo2": "98.0",
            "temp": "33.0",
            "finger": "True"
        })
    rows.append({
        "ts": (base_time).isoformat(),
        "device_id": "esp32-01",
        "hr": "72.0",
        "spo2": "98.0",
        "temp": "33.0",
        "finger": "True"
    })
    fpath = create_temp_csv(rows)
    try:
        res = analyze_dataset(fpath)
        assert res["burst_detected"] is True
        assert res["max_rows_in_1s"] == 6
        assert res["duplicate_ts_count"] == 1
    finally:
        os.remove(fpath)

def test_gap_resets_contiguous_stretch():
    base_time = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    rows = [
        {"ts": (base_time + timedelta(seconds=0)).isoformat(), "hr": "70", "spo2": "98", "temp": "33.0", "finger": "True"},
        {"ts": (base_time + timedelta(seconds=1)).isoformat(), "hr": "70", "spo2": "98", "temp": "33.0", "finger": "True"},
        {"ts": (base_time + timedelta(seconds=2)).isoformat(), "hr": "70", "spo2": "98", "temp": "33.0", "finger": "True"},
        {"ts": (base_time + timedelta(seconds=12)).isoformat(), "hr": "70", "spo2": "98", "temp": "33.0", "finger": "True"},
        {"ts": (base_time + timedelta(seconds=13)).isoformat(), "hr": "70", "spo2": "98", "temp": "33.0", "finger": "True"},
    ]
    fpath = create_temp_csv(rows)
    try:
        res = analyze_dataset(fpath)
        assert res["longest_valid_stretch"] == 3
        assert res["max_gap_s"] == 10.0
    finally:
        os.remove(fpath)
