import os
import shutil
import pytest
from datetime import datetime, timezone, timedelta
from app.database.csv_repository import CSVRepository
from app.database.interfaces import ReadingRecord, PredictionRecord, ActuatorEventRecord

@pytest.fixture
def temp_csv_repo(tmp_path):
    csv_dir = str(tmp_path / "csv_data")
    repo = CSVRepository(csv_dir=csv_dir)
    return repo

@pytest.mark.anyio
async def test_csv_repo_save_and_get_reading(temp_csv_repo):
    repo = temp_csv_repo
    now = datetime.now(timezone.utc).isoformat()
    reading = ReadingRecord(
        ts=now,
        device_id="esp32-01",
        hr=75.0,
        spo2=98.0,
        temp=36.5,
        hum=60.0,
        room_temp=28.0,
        finger=True,
        fw_status="NORMAL"
    )
    saved = await repo.save_reading(reading)
    assert saved is True

    latest = await repo.get_latest_reading("esp32-01")
    assert latest is not None
    assert latest.device_id == "esp32-01"
    assert latest.hr == 75.0
    assert latest.spo2 == 98.0
    assert latest.finger is True

@pytest.mark.anyio
async def test_csv_repo_reading_deduplication(temp_csv_repo):
    repo = temp_csv_repo
    ts = "2026-10-29T10:00:00+00:00"
    reading = ReadingRecord(
        ts=ts,
        device_id="esp32-01",
        hr=72.0,
        spo2=99.0,
        temp=36.6,
        finger=True
    )
    # First save succeeds
    assert await repo.save_reading(reading) is True

    # Duplicate (same device_id and ts) is rejected
    assert await repo.save_reading(reading) is False

@pytest.mark.anyio
async def test_csv_repo_null_vitals_handling(temp_csv_repo):
    """0.0 or None vitals should be stored as empty strings in CSV and parsed back as None."""
    repo = temp_csv_repo
    ts = "2026-10-29T10:01:00+00:00"
    reading = ReadingRecord(
        ts=ts,
        device_id="esp32-01",
        hr=None,
        spo2=None,
        temp=None,
        hum=None,
        room_temp=None,
        finger=False,
        fw_status="NO_FINGER"
    )
    assert await repo.save_reading(reading) is True

    latest = await repo.get_latest_reading("esp32-01")
    assert latest is not None
    assert latest.hr is None
    assert latest.spo2 is None
    assert latest.finger is False

@pytest.mark.anyio
async def test_csv_repo_history_retrieval(temp_csv_repo):
    repo = temp_csv_repo
    now = datetime.now(timezone.utc)
    
    # Save reading from 10 minutes ago
    ts_10m = (now - timedelta(minutes=10)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_10m, device_id="esp32-01", hr=70.0, spo2=98.0, temp=36.5, finger=True))

    # Save reading from 45 minutes ago
    ts_45m = (now - timedelta(minutes=45)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_45m, device_id="esp32-01", hr=72.0, spo2=98.0, temp=36.5, finger=True))

    history_30m = await repo.get_history("esp32-01", minutes=30)
    assert len(history_30m) == 1
    assert history_30m[0].ts == ts_10m

@pytest.mark.anyio
async def test_csv_repo_predictions(temp_csv_repo):
    repo = temp_csv_repo
    ts = datetime.now(timezone.utc).isoformat()
    pred = PredictionRecord(
        ts=ts,
        device_id="esp32-01",
        rule_severity=1,
        ml_severity=1,
        ml_score=0.1500,
        published_severity=1,
        decision_mode="RULE",
        label="ANOMALY",
        model_version="if_v1"
    )
    saved = await repo.save_prediction(pred)
    assert saved is True

    # Duplicate prediction rejected
    assert await repo.save_prediction(pred) is False

    preds = await repo.get_predictions("esp32-01", limit=10)
    assert len(preds) == 1
    assert preds[0].rule_severity == 1
    assert preds[0].ml_score == 0.1500
    assert preds[0].decision_source == "RULE"

@pytest.mark.anyio
async def test_csv_repo_actuator_events(temp_csv_repo):
    repo = temp_csv_repo
    ts = datetime.now(timezone.utc).isoformat()
    event_cmd = ActuatorEventRecord(
        ts=ts,
        kind="cmd",
        seq=501,
        payload_json='{"mode":"AUTO","seq":501}'
    )
    assert await repo.save_actuator_event(event_cmd) is True

    latest = await repo.get_latest_actuator_event("cmd")
    assert latest is not None
    assert latest.seq == 501
    assert latest.kind == "cmd"

@pytest.mark.anyio
async def test_csv_repo_scenario_events(temp_csv_repo):
    from app.database.interfaces import ScenarioEventRecord
    repo = temp_csv_repo

    # Initial scenario defaults to UNLABELED
    init_sc = await repo.get_current_scenario()
    assert init_sc.scenario == "UNLABELED"
    assert init_sc.note is None

    # Save a new scenario
    t1 = datetime.now(timezone.utc).isoformat()
    rec1 = ScenarioEventRecord(ts=t1, scenario="REST", note="resting baseline")
    assert await repo.save_scenario_event(rec1) is True

    curr1 = await repo.get_current_scenario()
    assert curr1.scenario == "REST"
    assert curr1.note == "resting baseline"
    assert curr1.ts == t1

    # Save second scenario
    t2 = datetime.now(timezone.utc).isoformat()
    rec2 = ScenarioEventRecord(ts=t2, scenario="TYPING", note="laptop typing")
    assert await repo.save_scenario_event(rec2) is True

    curr2 = await repo.get_current_scenario()
    assert curr2.scenario == "TYPING"
    assert curr2.note == "laptop typing"
    assert curr2.ts == t2
