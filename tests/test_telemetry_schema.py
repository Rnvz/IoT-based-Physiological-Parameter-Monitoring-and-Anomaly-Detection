import pytest
from app.schemas.telemetry import TelemetryPayload, FeedbackPayload, FeedbackDisplay

def test_firmware_json_format_parsing():
    firmware_data = {
        "device_id": "esp32_sensor_01",
        "sequence_id": 42,
        "timestamp_ms": 125000,
        "telemetry": {
            "heart_rate": 78.5,
            "spo2": 98.0,
            "temperature": 36.6,
            "ppg_amplitude": 2500.0,
            "finger_detected": True
        },
        "status": "normal"
    }
    payload = TelemetryPayload.model_validate(firmware_data)
    
    assert payload.device_id == "esp32_sensor_01"
    assert payload.timestamp == 125000
    assert payload.sequence_id == 42
    assert payload.raw_sensors.heart_rate == 78.5
    assert payload.raw_sensors.spo2 == 98.0
    assert payload.raw_sensors.temperature == 36.6
    assert payload.sensor_status.ppg_amplitude == 2500.0
    assert payload.sensor_status.max30102_ok is True
    assert payload.sensor_status.ds18b20_ok is True
    assert payload.network.sequence_id == 42
    assert payload.network.wifi_rssi == -50

def test_firmware_json_format_finger_off():
    firmware_data = {
        "device_id": "esp32_sensor_01",
        "sequence_id": 43,
        "timestamp_ms": 126000,
        "telemetry": {
            "heart_rate": 0.0,
            "spo2": 0.0,
            "temperature": 36.6,
            "ppg_amplitude": 0.0,
            "finger_detected": False
        },
        "status": "normal"
    }
    payload = TelemetryPayload.model_validate(firmware_data)
    assert payload.sensor_status.max30102_ok is False
    assert payload.sensor_status.ppg_amplitude == 0.0

def test_feedback_payload_defaults():
    fb = FeedbackPayload(
        device_id="esp32_sensor_01",
        reply_sequence_id=42,
        status="normal",
        buzzer_active=False,
        sqa_status="GOOD"
    )
    data = fb.model_dump()
    assert data["status"] == "normal"
    assert data["buzzer_active"] is False
    assert data["display"]["line_status"] == "NORMAL"
    assert data["display"]["oled_mode"] == "STANDARD"

def test_arduino_code_flat_json_normal():
    # Format persis seperti yang dikirim oleh Esp/Arduino_code.ino
    flat_data = {
        "hr": 75.5,
        "spo2": 98.0,
        "temp": 36.6,
        "status": "NORMAL"
    }
    payload = TelemetryPayload.model_validate(flat_data)
    assert payload.device_id == "esp32_hardware"
    assert payload.raw_sensors.heart_rate == 75.5
    assert payload.raw_sensors.spo2 == 98.0
    assert payload.raw_sensors.temperature == 36.6
    assert payload.sensor_status.max30102_ok is True
    assert payload.sensor_status.ppg_amplitude == 2000.0

def test_arduino_code_flat_json_menunggu_sensor():
    flat_data = {
        "hr": 0.0,
        "spo2": 98.0,
        "temp": 36.5,
        "status": "MENUNGGU_SENSOR"
    }
    payload = TelemetryPayload.model_validate(flat_data)
    assert payload.sensor_status.max30102_ok is False
    assert payload.sensor_status.ppg_amplitude == 0.0

