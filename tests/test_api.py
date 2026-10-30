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
    # Mock publish_command to return CommandPayload
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

    # /latest
    res_latest = client.get("/latest")
    assert res_latest.status_code == 200
    assert res_latest.json() == {"reading": None, "prediction": None}

    # /history
    res_hist = client.get("/history")
    assert res_hist.status_code == 200
    assert res_hist.json() == {"items": []}

    # /predictions
    res_preds = client.get("/predictions")
    assert res_preds.status_code == 200
    assert res_preds.json() == {"items": []}

    # /device/status
    res_dev = client.get("/device/status")
    assert res_dev.status_code == 200
    assert res_dev.json() == {"online": False, "age_s": -1.0}

    # /actuator/status
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

    # Save a reading and prediction
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

    # Check /latest
    res_latest = client.get("/latest")
    assert res_latest.status_code == 200
    data = res_latest.json()
    assert data["reading"]["hr"] == 75.0
    assert data["reading"]["spo2"] == 98.0
    assert data["reading"]["finger"] is True
    assert data["prediction"]["label"] == "NORMAL"
    assert data["prediction"]["severity"] == 0
    assert data["prediction"]["score"] == 0.2800

    # Check /predictions includes rule_severity, ml_severity, decision_mode
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

    # 10 minutes ago
    ts_10m = (now - timedelta(minutes=10)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_10m, device_id="esp32-01", hr=72.0, spo2=98.0, temp=36.5, finger=True))

    # 50 minutes ago
    ts_50m = (now - timedelta(minutes=50)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_50m, device_id="esp32-01", hr=80.0, spo2=97.0, temp=36.5, finger=True))

    # Query with default 30 minutes -> only 10m reading returned
    res = client.get("/history?minutes=30")
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1
    assert items[0]["hr"] == 72.0

    # Query with 60 minutes -> both readings returned
    res60 = client.get("/history?minutes=60")
    assert res60.status_code == 200
    assert len(res60.json()["items"]) == 2

    # Validation: minutes < 1 or > 1440 returns 422
    assert client.get("/history?minutes=0").status_code == 422
    assert client.get("/history?minutes=1441").status_code == 422

@pytest.mark.anyio
async def test_api_device_status_online_boundary(client_and_repo):
    client, repo, _ = client_and_repo
    now = datetime.now(timezone.utc)

    # Reading 10 seconds ago -> online=True
    ts_10s = (now - timedelta(seconds=10)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_10s, device_id="esp32-01", finger=True))

    res_online = client.get("/device/status")
    assert res_online.status_code == 200
    assert res_online.json()["online"] is True
    assert 9.0 <= res_online.json()["age_s"] <= 12.0

    # Reading 20 seconds ago -> online=False
    ts_20s = (now - timedelta(seconds=20)).isoformat()
    await repo.save_reading(ReadingRecord(ts=ts_20s, device_id="esp32-02", finger=True))

    res_offline = client.get("/device/status?device_id=esp32-02")
    assert res_offline.status_code == 200
    assert res_offline.json()["online"] is False
    assert res_offline.json()["age_s"] >= 19.0

def test_api_auth_missing_wrong_right(client_and_repo):
    client, _, _ = client_and_repo

    # Missing X-API-Key -> 401
    res_no_auth = client.post("/actuator/mode", json={"mode": "AUTO"})
    assert res_no_auth.status_code == 401
    assert res_no_auth.json()["detail"] == "Invalid or missing API Key"

    # Wrong X-API-Key -> 401
    res_wrong_auth = client.post(
        "/actuator/mode",
        json={"mode": "AUTO"},
        headers={"X-API-Key": "wrong_key"}
    )
    assert res_wrong_auth.status_code == 401
    assert res_wrong_auth.json()["detail"] == "Invalid or missing API Key"

    # Correct X-API-Key -> 200
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

    # Invalid color
    res_bad_col = client.post("/actuator/led", json={"color": "PURPLE", "brightness": 50}, headers=headers)
    assert res_bad_col.status_code == 422

    # Invalid brightness > 100
    res_bad_b1 = client.post("/actuator/led", json={"color": "RED", "brightness": 120}, headers=headers)
    assert res_bad_b1.status_code == 422

    # Invalid brightness < 0
    res_bad_b2 = client.post("/actuator/led", json={"color": "RED", "brightness": -10}, headers=headers)
    assert res_bad_b2.status_code == 422

    # Valid LED command
    res_good_led = client.post("/actuator/led", json={"color": "GREEN", "brightness": 60}, headers=headers)
    assert res_good_led.status_code == 200

    # Invalid Mode
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

    # Trigger requests up to limit (20)
    for _ in range(20):
        r = client.post("/actuator/mode", json={"mode": "AUTO"}, headers=headers)
        assert r.status_code == 200

    # 21st request hits 429 Too Many Requests
    r_over = client.post("/actuator/mode", json={"mode": "AUTO"}, headers=headers)
    assert r_over.status_code == 429
    assert "Rate limit exceeded" in r_over.json()["detail"]

    # Reset for other tests
    rate_limiter.requests.clear()

def test_api_export_readings_csv(client_and_repo):
    client, _, _ = client_and_repo

    # Missing API Key on export -> 401
    res_unauth = client.get("/export/readings.csv")
    assert res_unauth.status_code == 401

    # Authorized export -> 200 and streaming CSV
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

    # 1. Loopback 127.0.0.1 with CF-Connecting-IP -> trusts CF header
    req_lb4 = DummyRequest("127.0.0.1", {"cf-connecting-ip": "203.0.113.19"})
    assert get_client_ip(req_lb4) == "203.0.113.19"

    # 2. Loopback ::1 with CF-Connecting-IP -> trusts CF header
    req_lb6 = DummyRequest("::1", {"CF-Connecting-IP": "198.51.100.42"})
    assert get_client_ip(req_lb6) == "198.51.100.42"

    # 3. Direct external peer (e.g. 192.168.1.50) with spoofed CF-Connecting-IP -> IGNORES CF header!
    req_ext = DummyRequest("192.168.1.50", {"CF-Connecting-IP": "8.8.8.8"})
    assert get_client_ip(req_ext) == "192.168.1.50"

    # 4. Loopback without CF header -> returns loopback IP
    req_plain = DummyRequest("127.0.0.1", {})
    assert get_client_ip(req_plain) == "127.0.0.1"

def test_read_auth_required_toggle(client_and_repo):
    client, _, _ = client_and_repo
    from app.config import settings

    orig = settings.read_auth_required
    try:
        # Default / Disabled: public GET
        settings.read_auth_required = False
        res_public = client.get("/latest")
        assert res_public.status_code == 200

        # Enabled: 401 if missing X-API-Key, 200 if provided
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

    # Reading before first event -> default BOOT state
    res_before = as_of_join_actuator("2026-10-29T10:00:30Z", timeline, ts_list)
    assert res_before["mode"] == "AUTO"
    assert res_before["led_color"] == "OFF"
    assert res_before["source"] == "BOOT"

    # Reading between events -> first event state
    res_mid = as_of_join_actuator("2026-10-29T10:03:00Z", timeline, ts_list)
    assert res_mid["mode"] == "AUTO"
    assert res_mid["led_color"] == "GREEN"
    assert res_mid["source"] == "RULE"

    # Reading at exact timestamp of second event -> second event state
    res_exact = as_of_join_actuator("2026-10-29T10:05:00Z", timeline, ts_list)
    assert res_exact["mode"] == "MANUAL"
    assert res_exact["led_color"] == "RED"
    assert res_exact["source"] == "MANUAL"

    # Reading after second event -> second event state
    res_after = as_of_join_actuator("2026-10-29T10:10:00Z", timeline, ts_list)
    assert res_after["mode"] == "MANUAL"
    assert res_after["led_color"] == "RED"
    assert res_after["source"] == "MANUAL"

@pytest.mark.anyio
async def test_api_export_dataset_csv_as_of_join(client_and_repo):
    import csv as py_csv
    import io
    client, repo, _ = client_and_repo

    # 1. Unauthenticated request -> 401
    res_unauth = client.get("/export/dataset.csv")
    assert res_unauth.status_code == 401

    # 2. Populate test data with timestamps:
    # t0: Reading 1
    t0 = "2026-10-29T10:00:00.000000+00:00"
    await repo.save_reading(ReadingRecord(
        ts=t0, device_id="esp32-01", hr=72.0, spo2=98.0, temp=36.5, finger=True, fw_status="NORMAL"
    ))
    await repo.save_prediction(PredictionRecord(
        ts=t0, device_id="esp32-01", rule_severity=0, ml_severity=0, ml_score=0.31,
        published_severity=0, decision_mode="RULE", decision_source="RULE", label="NORMAL"
    ))

    # t1: Actuator event (mode=MANUAL, led=RED)
    t1 = "2026-10-29T10:01:00.000000+00:00"
    await repo.save_actuator_event(ActuatorEventRecord(
        ts=t1, kind="cmd", seq=101, payload_json='{"mode":"MANUAL","led":{"color":"RED","brightness":100},"buzzer":{"test":false},"source":"MANUAL"}'
    ))

    # t2: Reading 2 (finger=False, no prediction saved)
    t2 = "2026-10-29T10:02:00.000000+00:00"
    await repo.save_reading(ReadingRecord(
        ts=t2, device_id="esp32-01", hr=None, spo2=None, temp=None, finger=False, fw_status="NO_FINGER"
    ))

    # 3. Call GET /export/dataset.csv with X-API-Key
    res = client.get("/export/dataset.csv", headers={"X-API-Key": "test_secret_key"})
    assert res.status_code == 200
    assert "attachment; filename=dataset.csv" in res.headers["content-disposition"]

    reader = list(py_csv.DictReader(io.StringIO(res.text)))
    assert len(reader) == 2

    # Check reading 1 (before actuator event t1): actuator should be default BOOT state
    r1 = reader[0]
    assert r1["ts"] == t0
    assert float(r1["hr"]) == 72.0
    assert r1["finger"] == "True"
    assert r1["rule_severity"] == "0"
    assert r1["mode"] == "AUTO"
    assert r1["led_color"] == "OFF"
    assert r1["source"] == "BOOT"

    # Check reading 2 (after actuator event t1): actuator should be joined with t1 event state
    r2 = reader[1]
    assert r2["ts"] == t2
    assert r2["finger"] == "False"
    assert r2["rule_severity"] == ""  # No prediction saved
    assert r2["mode"] == "MANUAL"
    assert r2["led_color"] == "RED"
    assert r2["source"] == "MANUAL"
