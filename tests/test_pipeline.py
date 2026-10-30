import json
import pytest
from unittest.mock import MagicMock
from app.mqtt.consumer import MQTTService
from app.config import settings

@pytest.fixture
def mock_mqtt_service(tmp_path, mocker):
    csv_dir = str(tmp_path / "mqtt_test_csv")
    mocker.patch.object(settings, "csv_dir", csv_dir)
    mocker.patch.object(settings, "topic_prefix", "physiomonitor/test")

    from app.database.csv_repository import CSVRepository
    repo = CSVRepository(csv_dir=csv_dir)
    mocker.patch("app.mqtt.consumer.get_repository", return_value=repo)
    mocker.patch("app.mqtt.publisher.get_repository", return_value=repo)

    service = MQTTService()
    service.client = MagicMock()
    service.publisher.client = service.client
    return service, repo

@pytest.mark.anyio
async def test_pipeline_normal_message_publishes_result_and_persists(mock_mqtt_service):
    service, repo = mock_mqtt_service
    topic = f"{settings.topic_prefix}/data"
    payload = {
        "device_id": "esp32-01",
        "hr": 75.0,
        "spo2": 98.0,
        "temp": 36.5,
        "hum": 60.0,
        "room_temp": 28.0,
        "finger": True,
        "status": "NORMAL"
    }

    await service.handle_message(topic, json.dumps(payload).encode("utf-8"))

    latest_reading = await repo.get_latest_reading("esp32-01")
    assert latest_reading is not None
    assert latest_reading.hr == 75.0
    assert latest_reading.finger is True

    assert service.client.publish.called
    call_topic, call_payload = service.client.publish.call_args[0]
    assert call_topic == f"{settings.topic_prefix}/result"
    
    result_dict = json.loads(call_payload)
    assert result_dict["label"] == "NORMAL"
    assert result_dict["severity"] == 0
    assert result_dict["model_version"] == settings.model_version

    preds = await repo.get_predictions("esp32-01", limit=1)
    assert len(preds) == 1
    assert preds[0].rule_severity == 0

@pytest.mark.anyio
async def test_pipeline_finger_false_persists_raw_but_omits_result(mock_mqtt_service):
    """
    Requirement: finger:false means HR and SpO2 not valid.
    Backend persists raw reading, resets window, and does NOT publish /result.
    """
    service, repo = mock_mqtt_service
    topic = f"{settings.topic_prefix}/data"
    payload = {
        "device_id": "esp32-01",
        "hr": 0.0,
        "spo2": 0.0,
        "temp": 36.5,
        "finger": False,
        "status": "NO_FINGER"
    }

    await service.handle_message(topic, json.dumps(payload).encode("utf-8"))

    latest = await repo.get_latest_reading("esp32-01")
    assert latest is not None
    assert latest.finger is False
    assert latest.hr is None

    assert not service.client.publish.called

@pytest.mark.anyio
async def test_pipeline_critical_anomaly_publishes_severity_2(mock_mqtt_service):
    service, repo = mock_mqtt_service
    topic = f"{settings.topic_prefix}/data"
    payload = {
        "device_id": "esp32-01",
        "hr": 140.0,
        "spo2": 85.0,
        "temp": 39.5,
        "finger": True,
        "status": "NORMAL"
    }

    await service.handle_message(topic, json.dumps(payload).encode("utf-8"))

    assert service.client.publish.called
    call_topic, call_payload = service.client.publish.call_args[0]
    result_dict = json.loads(call_payload)
    assert result_dict["severity"] == 2
    assert result_dict["label"] == "ANOMALY"

@pytest.mark.anyio
async def test_pipeline_malformed_json_gracefully_ignored(mock_mqtt_service):
    service, repo = mock_mqtt_service
    topic = f"{settings.topic_prefix}/data"
    bad_payload = b"NOT_A_JSON_STRING{xyz"

    await service.handle_message(topic, bad_payload)
    assert not service.client.publish.called

@pytest.mark.anyio
async def test_pipeline_actuator_message_saved(mock_mqtt_service):
    service, repo = mock_mqtt_service
    topic = f"{settings.topic_prefix}/actuator"
    payload = {
        "mode": "AUTO",
        "led": {"color": "GREEN", "brightness": 40},
        "buzzer": {"test": False},
        "source": "ML_RESULT",
        "ack_seq": 999,
        "manual_left_s": 0
    }

    await service.handle_message(topic, json.dumps(payload).encode("utf-8"))

    event = await repo.get_latest_actuator_event(kind="status")
    assert event is not None
    assert event.seq == 999
    assert "ML_RESULT" in event.payload_json

def test_mqtt_service_refuses_empty_topic_prefix(monkeypatch):
    """
    MQTTService must refuse to start (raise ValueError) when TOPIC_PREFIX is empty.
    """
    monkeypatch.setattr(settings, "topic_prefix", "")
    with pytest.raises(ValueError, match="Invalid TOPIC_PREFIX"):
        MQTTService()

def test_mqtt_service_refuses_placeholder_topic_prefix(monkeypatch):
    """
    MQTTService must refuse to start (raise ValueError) when TOPIC_PREFIX contains GANTI-ID-UNIK.
    """
    monkeypatch.setattr(settings, "topic_prefix", "physiomonitor/GANTI-ID-UNIK")
    with pytest.raises(ValueError, match="GANTI-ID-UNIK"):
        MQTTService()

