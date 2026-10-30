import os
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.database.interfaces import ReadingRecord, PredictionRecord, ActuatorEventRecord
from app.database.csv_repository import CSVRepository
from app.schemas.cmd import CommandPayload

@pytest.fixture
def client_and_repo(tmp_path, mocker):
    csv_dir = str(tmp_path / "api_test_data")
    repo = CSVRepository(csv_dir=csv_dir)
    mocker.patch("app.api.telemetry.get_repository", return_value=repo)
    mocker.patch("app.api.actuator.get_repository", return_value=repo)
    mocker.patch.object(settings, "csv_dir", csv_dir)
    mocker.patch.object(settings, "api_key", "test_secret_key")

    mock_pub = MagicMock()
    async def fake_publish_cmd(mode=None, color=None, brightness=None, buzzer_test=None):
        return CommandPayload(
            mode=mode or "AUTO",
            color=color,
            brightness=brightness,
            buzzer_test=buzzer_test,
            seq=101
        )
    mock_pub.publish_command = AsyncMock(side_effect=fake_publish_cmd)
    mocker.patch("app.api.actuator._get_publisher", return_value=mock_pub)

    client = TestClient(app, raise_server_exceptions=False)
    return client, repo, mock_pub

def test_api_health_endpoint(client_and_repo):
    client, _, _ = client_and_repo
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_api_health_details_endpoint(client_and_repo):
    client, _, _ = client_and_repo
    res = client.get("/health/details")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "mqtt_connected" in data
    assert "decision_mode" in data

def test_api_empty_database_responses(client_and_repo):
    client, _, _ = client_and_repo

    res_latest = client.get("/latest")
    assert res_latest.status_code == 200
    assert res_latest.json() == {"reading": None, "prediction": None}

    res_hist = client.get("/history")
    assert res_hist.status_code == 200
    assert res_hist.json() == {"items": []}

    res_preds = client.get("/predictions")
    assert res_preds.status_code == 200
    assert res_preds.json() == {"items": []}

    res_dev = client.get("/device/status")
    assert res_dev.status_code == 200
    assert res_dev.json() == {"online": False, "age_s": -1.0}

    res_act = client.get("/actuator/status")
    assert res_act.status_code == 200
    data = res_act.json()
    assert data["mode"] == "AUTO"
    assert data["ack_seq"] is None
    assert data["source"] == "NO_RESULT"

@pytest.mark.anyio
async def test_api_populated_responses_and_shapes(client_and_repo):
    client, repo, _ = client_and_repo
    now_str = datetime.now(timezone.utc).isoformat()

    reading = ReadingRecord(
        ts=now_str,
        device_id="esp32-01",
        hr=75.0,
        spo2=98.0,
        temp=36.5,
        hum=60.0,
        room_temp=28.0,
        finger=True
    )
    await repo.save_reading(reading)

    pred = PredictionRecord(
        ts=now_str,
        device_id="esp32-01",
        rule_severity=0,
        ml_severity=0,
        ml_score=0.2800,
        published_severity=0,
        decision_mode="RULE",
        label="NORMAL",
        model_version="if_v1"
    )
    await repo.save_prediction(pred)

    res_latest = client.get("/latest")
    assert res_latest.status_code == 200
    data = res_latest.json()
    assert data["reading"]["hr"] == 75.0
    assert data["reading"]["spo2"] == 98.0
    assert data["reading"]["finger"] is True
    assert data["prediction"]["label"] == "NORMAL"
    assert data["prediction"]["severity"] == 0
    assert data["prediction"]["score"] == 0.2800

    res_preds = client.get("/predictions")
    assert res_preds.status_code == 200
    p_data = res_preds.json()
    assert len(p_data["items"]) == 1
    assert p_data["items"][0]["rule_severity"] == 0
    assert p_data["items"][0]["ml_severity"] == 0
    assert p_data["items"][0]["decision_mode"] == "RULE"
    assert p_data["items"][0]["decision_source"] == "RULE"

@pytest.mark.anyio
async def test_api_history_filtering_and_validation(client_and_repo):
    client, repo, _ = client_and_repo
    now = datetime.now(timezone.utc)

    ts_10m = (now - timedelta(minutes=10)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_10m, device_id="esp32-01", hr=72.0, spo2=98.0, temp=36.5, finger=True))

    ts_50m = (now - timedelta(minutes=50)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_50m, device_id="esp32-01", hr=80.0, spo2=97.0, temp=36.5, finger=True))

    res = client.get("/history?minutes=30")
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1
    assert items[0]["hr"] == 72.0

    res60 = client.get("/history?minutes=60")
    assert res60.status_code == 200
    assert len(res60.json()["items"]) == 2

    assert client.get("/history?minutes=0").status_code == 422
    assert client.get("/history?minutes=1441").status_code == 422

@pytest.mark.anyio
async def test_api_device_status_online_boundary(client_and_repo):
    client, repo, _ = client_and_repo
    now = datetime.now(timezone.utc)

    ts_10s = (now - timedelta(seconds=10)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_10s, device_id="esp32-01", finger=True))

    res_online = client.get("/device/status")
    assert res_online.status_code == 200
    assert res_online.json()["online"] is True
    assert 9.0 <= res_online.json()["age_s"] <= 12.0

    ts_20s = (now - timedelta(seconds=20)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_20s, device_id="esp32-02", finger=True))

    res_offline = client.get("/device/status?device_id=esp32-02")
    assert res_offline.status_code == 200
    assert res_offline.json()["online"] is False
    assert res_offline.json()["age_s"] >= 19.0

def test_api_auth_missing_wrong_right(client_and_repo):
    client, _, _ = client_and_repo

    res_no_auth = client.post("/actuator/mode", json={"mode": "AUTO"})
    assert res_no_auth.status_code == 401
    assert res_no_auth.json()["detail"] == "Invalid or missing API Key"

    res_wrong_auth = client.post(
        "/actuator/mode",
        json={"mode": "AUTO"},
        headers={"X-API-Key": "wrong_key"}
    )
    assert res_wrong_auth.status_code == 401
    assert res_wrong_auth.json()["detail"] == "Invalid or missing API Key"

    res_ok = client.post(
        "/actuator/mode",
        json={"mode": "AUTO"},
        headers={"X-API-Key": "test_secret_key"}
    )
    assert res_ok.status_code == 200
    assert res_ok.json()["mode"] == "AUTO"

def test_api_invalid_colour_brightness_and_mode(client_and_repo):
    client, _, _ = client_and_repo
    headers = {"X-API-Key": "test_secret_key"}

    res_bad_col = client.post("/actuator/led", json={"color": "PURPLE", "brightness": 50}, headers=headers)
    assert res_bad_col.status_code == 422

    res_bad_b1 = client.post("/actuator/led", json={"color": "RED", "brightness": 120}, headers=headers)
    assert res_bad_b1.status_code == 422

    res_bad_b2 = client.post("/actuator/led", json={"color": "RED", "brightness": -10}, headers=headers)
    assert res_bad_b2.status_code == 422

    res_good_led = client.post("/actuator/led", json={"color": "GREEN", "brightness": 60}, headers=headers)
    assert res_good_led.status_code == 200

    res_bad_mode = client.post("/actuator/mode", json={"mode": "INVALID_MODE"}, headers=headers)
    assert res_bad_mode.status_code == 422

def test_api_buzzer_command(client_and_repo):
    client, _, _ = client_and_repo
    headers = {"X-API-Key": "test_secret_key"}

    res = client.post("/actuator/buzzer", json={"test": True}, headers=headers)
    assert res.status_code == 200
    assert res.json()["buzzer_test"] is True

def test_api_rate_limiter(client_and_repo):
    client, _, _ = client_and_repo
    headers = {"X-API-Key": "test_secret_key"}

    from app.api.deps import rate_limiter
    rate_limiter.requests.clear()

    for _ in range(20):
        r = client.post("/actuator/mode", json={"mode": "AUTO"}, headers=headers)
        assert r.status_code == 200

    r_over = client.post("/actuator/mode", json={"mode": "AUTO"}, headers=headers)
    assert r_over.status_code == 429
    assert "Rate limit exceeded" in r_over.json()["detail"]

    rate_limiter.requests.clear()

def test_api_export_readings_csv(client_and_repo):
    client, _, _ = client_and_repo

    res_unauth = client.get("/export/readings.csv")
    assert res_unauth.status_code == 401

    res_auth = client.get("/export/readings.csv", headers={"X-API-Key": "test_secret_key"})
    assert res_auth.status_code == 200
    assert "text/csv" in res_auth.headers["content-type"]
    assert "attachment; filename=readings.csv" in res_auth.headers["content-disposition"]
    assert "ts,device_id,hr,spo2" in res_auth.text

def test_rate_limiter_client_ip_resolution():
    from app.api.deps import get_client_ip
    from starlette.datastructures import Headers, Address

    class DummyClient:
        def __init__(self, host: str):
            self.host = host

    class DummyRequest:
        def __init__(self, host: str, headers: dict):
            self.client = DummyClient(host)
            self.headers = Headers(headers)

    req_lb4 = DummyRequest("127.0.0.1", {"cf-connecting-ip": "203.0.113.19"})
    assert get_client_ip(req_lb4) == "203.0.113.19"

    req_lb6 = DummyRequest("::1", {"CF-Connecting-IP": "198.51.100.42"})
    assert get_client_ip(req_lb6) == "198.51.100.42"

    req_ext = DummyRequest("192.168.1.50", {"CF-Connecting-IP": "8.8.8.8"})
    assert get_client_ip(req_ext) == "192.168.1.50"

    req_plain = DummyRequest("127.0.0.1", {})
    assert get_client_ip(req_plain) == "127.0.0.1"

def test_read_auth_required_toggle(client_and_repo):
    client, _, _ = client_and_repo
    from app.config import settings

    orig = settings.read_auth_required
    try:
        settings.read_auth_required = False
        res_public = client.get("/latest")
        assert res_public.status_code == 200

        settings.read_auth_required = True
        res_denied = client.get("/latest")
        assert res_denied.status_code == 401

        res_allowed = client.get("/latest", headers={"X-API-Key": "test_secret_key"})
        assert res_allowed.status_code == 200
    finally:
        settings.read_auth_required = orig

def test_as_of_join_actuator_unit():
    from app.database.csv_repository import as_of_join_actuator

    timeline = [
        ("2026-10-29T10:01:00Z", {"mode": "AUTO", "led_color": "GREEN", "brightness": 50, "buzzer_test": False, "source": "RULE"}),
        ("2026-10-29T10:05:00Z", {"mode": "MANUAL", "led_color": "RED", "brightness": 100, "buzzer_test": True, "source": "MANUAL"})
    ]
    ts_list = [x[0] for x in timeline]

    res_before = as_of_join_actuator("2026-10-29T10:00:30Z", timeline, ts_list)
    assert res_before["mode"] == "AUTO"
    assert res_before["led_color"] == "OFF"
    assert res_before["source"] == "BOOT"

    res_mid = as_of_join_actuator("2026-10-29T10:03:00Z", timeline, ts_list)
    assert res_mid["mode"] == "AUTO"
    assert res_mid["led_color"] == "GREEN"
    assert res_mid["source"] == "RULE"

    res_exact = as_of_join_actuator("2026-10-29T10:05:00Z", timeline, ts_list)
    assert res_exact["mode"] == "MANUAL"
    assert res_exact["led_color"] == "RED"
    assert res_exact["source"] == "MANUAL"

    res_after = as_of_join_actuator("2026-10-29T10:10:00Z", timeline, ts_list)
    assert res_after["mode"] == "MANUAL"
    assert res_after["led_color"] == "RED"
    assert res_after["source"] == "MANUAL"

@pytest.mark.anyio
async def test_api_export_dataset_csv_as_of_join(client_and_repo):
    import csv as py_csv
    import io
    client, repo, _ = client_and_repo

    res_unauth = client.get("/export/dataset.csv")
    assert res_unauth.status_code == 401

    t0 = "2026-10-29T10:00:00.000000+00:00"
    await repo.save_reading(ReadingRecord(
        ts=t0, device_id="esp32-01", hr=72.0, spo2=98.0, temp=36.5, finger=True, fw_status="NORMAL"
    ))
    await repo.save_prediction(PredictionRecord(
        ts=t0, device_id="esp32-01", rule_severity=0, ml_severity=0, ml_score=0.31,
        published_severity=0, decision_mode="RULE", decision_source="RULE", label="NORMAL"
    ))

    t1 = "2026-10-29T10:01:00.000000+00:00"
    await repo.save_actuator_event(ActuatorEventRecord(
        ts=t1, kind="cmd", seq=101, payload_json='{"mode":"MANUAL","led":{"color":"RED","brightness":100},"buzzer":{"test":false},"source":"MANUAL"}'
    ))

    t2 = "2026-10-29T10:02:00.000000+00:00"
    await repo.save_reading(ReadingRecord(
        ts=t2, device_id="esp32-01", hr=None, spo2=None, temp=None, finger=False, fw_status="NO_FINGER"
    ))

    res = client.get("/export/dataset.csv", headers={"X-API-Key": "test_secret_key"})
    assert res.status_code == 200
    assert "attachment; filename=dataset.csv" in res.headers["content-disposition"]

    reader = list(py_csv.DictReader(io.StringIO(res.text)))
    assert len(reader) == 2

    r1 = reader[0]
    assert r1["ts"] == t0
    assert float(r1["hr"]) == 72.0
    assert r1["finger"] == "True"
    assert r1["rule_severity"] == "0"
    assert r1["scenario"] == "UNLABELED"
    assert r1["mode"] == "AUTO"
    assert r1["led_color"] == "OFF"
    assert r1["source"] == "BOOT"

    r2 = reader[1]
    assert r2["ts"] == t2
    assert r2["finger"] == "False"
    assert r2["rule_severity"] == ""
    assert r2["scenario"] == "UNLABELED"
    assert r2["mode"] == "MANUAL"
    assert r2["led_color"] == "RED"
    assert r2["source"] == "MANUAL"

def test_session_scenario_validation_and_auth(client_and_repo):
    client, _, _ = client_and_repo
    from app.config import settings

    res_unauth = client.post("/session/scenario", json={"scenario": "REST"})
    assert res_unauth.status_code == 401

    headers = {"X-API-Key": "test_secret_key"}

    res_invalid_sc = client.post("/session/scenario", json={"scenario": "HYPOXIA_INDUCED"}, headers=headers)
    assert res_invalid_sc.status_code == 422

    long_note = "A" * 201
    res_long_note = client.post("/session/scenario", json={"scenario": "REST", "note": long_note}, headers=headers)
    assert res_long_note.status_code == 422

    res_extra = client.post("/session/scenario", json={"scenario": "REST", "extra": "forbidden"}, headers=headers)
    assert res_extra.status_code == 422

    valid_scenarios = ["REST", "TYPING", "LIGHT_ACTIVITY", "MOTION", "FINGER_SHIFT", "SENSOR_OFF", "OTHER"]
    for sc in valid_scenarios:
        res_ok = client.post("/session/scenario", json={"scenario": sc, "note": f"Testing {sc}"}, headers=headers)
        assert res_ok.status_code == 200
        assert res_ok.json()["status"] == "ok"
        assert res_ok.json()["scenario"] == sc
        assert res_ok.json()["note"] == f"Testing {sc}"

    res_get = client.get("/session/scenario")
    assert res_get.status_code == 200
    assert res_get.json()["scenario"] == "OTHER"
    assert res_get.json()["note"] == "Testing OTHER"

    orig = settings.read_auth_required
    try:
        settings.read_auth_required = True
        res_get_unauth = client.get("/session/scenario")
        assert res_get_unauth.status_code == 401

        res_get_auth = client.get("/session/scenario", headers=headers)
        assert res_get_auth.status_code == 200
        assert res_get_auth.json()["scenario"] == "OTHER"
    finally:
        settings.read_auth_required = orig

def test_as_of_join_scenario_unit():
    from app.database.csv_repository import as_of_join_scenario

    timeline = [
        ("2026-10-29T10:01:00Z", "REST"),
        ("2026-10-29T10:05:00Z", "TYPING"),
    ]
    ts_list = [x[0] for x in timeline]

    assert as_of_join_scenario("2026-10-29T10:00:30Z", timeline, ts_list) == "UNLABELED"

    assert as_of_join_scenario("2026-10-29T10:03:00Z", timeline, ts_list) == "REST"

    assert as_of_join_scenario("2026-10-29T10:05:00Z", timeline, ts_list) == "TYPING"

    assert as_of_join_scenario("2026-10-29T10:15:00Z", timeline, ts_list) == "TYPING"

@pytest.mark.anyio
async def test_dataset_export_includes_scenario_column_with_as_of_join(client_and_repo):
    import csv as py_csv
    import io
    client, repo, _ = client_and_repo
    from app.database.interfaces import ScenarioEventRecord

    headers = {"X-API-Key": "test_secret_key"}

    t0 = "2026-10-29T11:00:00.000000+00:00"
    await repo.save_reading(ReadingRecord(
        ts=t0, device_id="esp32-01", hr=70.0, spo2=99.0, temp=33.1, finger=True
    ))

    t1 = "2026-10-29T11:01:00.000000+00:00"
    await repo.save_scenario_event(ScenarioEventRecord(ts=t1, scenario="REST", note="baseline"))

    t2 = "2026-10-29T11:02:00.000000+00:00"
    await repo.save_reading(ReadingRecord(
        ts=t2, device_id="esp32-01", hr=71.0, spo2=98.5, temp=33.2, finger=True
    ))

    t3 = "2026-10-29T11:03:00.000000+00:00"
    await repo.save_scenario_event(ScenarioEventRecord(ts=t3, scenario="MOTION", note="arm movement"))

    t4 = "2026-10-29T11:04:00.000000+00:00"
    await repo.save_reading(ReadingRecord(
        ts=t4, device_id="esp32-01", hr=85.0, spo2=97.0, temp=33.3, finger=True
    ))

    res = client.get("/export/dataset.csv", headers=headers)
    assert res.status_code == 200

    reader = list(py_csv.DictReader(io.StringIO(res.text)))
    assert len(reader) == 3

    assert "scenario" in reader[0]
    assert reader[0]["ts"] == t0
    assert reader[0]["scenario"] == "UNLABELED"

    assert reader[1]["ts"] == t2
    assert reader[1]["scenario"] == "REST"

    assert reader[2]["ts"] == t4
    assert reader[2]["scenario"] == "MOTION"
