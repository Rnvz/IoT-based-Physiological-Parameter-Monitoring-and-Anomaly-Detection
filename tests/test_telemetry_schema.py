import pytest
from pydantic import ValidationError
from app.schemas.telemetry import TelemetryPayload
from app.schemas.result import ResultPayload
from app.schemas.cmd import CommandPayload
from app.schemas.actuator import ActuatorStatusPayload

def test_telemetry_payload_valid_parsing():
    data = {
        "device_id": "esp32-01",
        "hr": 75.0,
        "spo2": 98.0,
        "temp": 36.5,
        "hum": 62.3,
        "room_temp": 29.1,
        "finger": True,
        "status": "NORMAL"
    }
    payload = TelemetryPayload.model_validate(data)
    assert payload.device_id == "esp32-01"
    assert payload.hr == 75.0
    assert payload.spo2 == 98.0
    assert payload.temp == 36.5
    assert payload.hum == 62.3
    assert payload.room_temp == 29.1
    assert payload.finger is True
    assert payload.status == "NORMAL"
    assert payload.has_valid_vitals is True

def test_telemetry_zeros_converted_to_none():
    """Requirement: hr, spo2, temp == 0.0 must be stored as None (never 0)."""
    data = {
        "hr": 0.0,
        "spo2": 0.0,
        "temp": 0.0,
        "hum": None,
        "room_temp": None,
        "finger": True,
        "status": "MENUNGGU SENSOR"
    }
    payload = TelemetryPayload.model_validate(data)
    assert payload.hr is None
    assert payload.spo2 is None
    assert payload.temp is None
    assert payload.hum is None
    assert payload.room_temp is None
    assert payload.has_valid_vitals is False

def test_telemetry_finger_false_marks_vitals_invalid():
    """Requirement: finger=False means HR and SpO2 are not valid."""
    data = {
        "hr": 75.0,
        "spo2": 98.0,
        "temp": 36.5,
        "finger": False,
        "status": "NORMAL"
    }
    payload = TelemetryPayload.model_validate(data)
    assert payload.finger is False
    assert payload.has_valid_vitals is False

def test_telemetry_range_validation_hr():
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"hr": 10.0, "finger": True})
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"hr": 300.0, "finger": True})

def test_telemetry_range_validation_spo2():
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"spo2": 35.0, "finger": True})
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"spo2": 105.0, "finger": True})

def test_telemetry_range_validation_temp():
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"temp": 15.0, "finger": True})
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"temp": 50.0, "finger": True})

def test_telemetry_range_validation_dht22():
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"hum": -5.0})
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"hum": 105.0})
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"room_temp": -25.0})
    with pytest.raises(ValidationError):
        TelemetryPayload.model_validate({"room_temp": 70.0})

def test_telemetry_extra_fields_ignored():
    data = {
        "hr": 72.0,
        "spo2": 99.0,
        "temp": 36.6,
        "finger": True,
        "unknown_extra_sensor": 1234.5,
        "nested_garbage": {"a": [1, 2, 3]}
    }
    payload = TelemetryPayload.model_validate(data)
    assert payload.hr == 72.0
    assert not hasattr(payload, "unknown_extra_sensor")

def test_result_payload_validation():
    res = ResultPayload(label="NORMAL", severity=0, score=0.1234, model_version="if_v1")
    assert res.severity == 0
    assert res.label == "NORMAL"
    assert res.score == 0.1234

    with pytest.raises(ValidationError):
        ResultPayload(label="ANOMALY", severity=3)
    with pytest.raises(ValidationError):
        ResultPayload(label="NORMAL", severity=-1)

def test_command_payload_validation():
    cmd1 = CommandPayload(mode="AUTO", seq=101)
    assert cmd1.mode == "AUTO"
    assert cmd1.seq == 101

    cmd2 = CommandPayload(mode="MANUAL", color="RED", brightness=60, seq=102)
    assert cmd2.color == "RED"
    assert cmd2.brightness == 60

    cmd3 = CommandPayload(buzzer_test=True, seq=103)
    assert cmd3.buzzer_test is True

    with pytest.raises(ValidationError):
        CommandPayload(mode="MANUAL", color="PURPLE", seq=104)

    with pytest.raises(ValidationError):
        CommandPayload(mode="MANUAL", color="BLUE", brightness=150, seq=105)

    with pytest.raises(ValidationError):
        CommandPayload(mode="AUTO", seq=0)
    with pytest.raises(ValidationError):
        CommandPayload(mode="AUTO", seq=2147483648)

def test_actuator_status_payload_validation():
    data = {
        "mode": "AUTO",
        "led": {"color": "GREEN", "brightness": 40},
        "buzzer": {"test": False},
        "source": "ML_RESULT",
        "ack_seq": 103,
        "manual_left_s": 0
    }
    payload = ActuatorStatusPayload.model_validate(data)
    assert payload.mode == "AUTO"
    assert payload.source == "ML_RESULT"
    assert payload.ack_seq == 103
    assert payload.led["color"] == "GREEN"
