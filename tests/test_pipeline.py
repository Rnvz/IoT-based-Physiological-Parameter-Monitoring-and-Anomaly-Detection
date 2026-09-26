import pytest
import numpy as np
from app.main import process_telemetry, anomaly_detector, persistent_evaluator, feature_engine
from app.services.anomaly_model import AnomalyDetector

@pytest.fixture(autouse=True)
def setup_detector():
    import app.main as main_mod
    if main_mod.anomaly_detector is None:
        main_mod.anomaly_detector = AnomalyDetector()
    main_mod.persistent_evaluator.reset()
    main_mod.feature_engine.reset()

def test_pipeline_normal_stream():
    # Send 16 normal samples to fill the feature buffer (size 15)
    for i in range(16):
        sample = {
            "device_id": "test_esp32",
            "sequence_id": i + 1,
            "timestamp_ms": 1000 + i * 1000,
            "telemetry": {
                "heart_rate": 75.0,
                "spo2": 98.0,
                "temperature": 32.5,
                "ppg_amplitude": 2500.0,
                "finger_detected": True
            },
            "status": "normal"
        }
        feedback, dash = process_telemetry(sample, publish_mqtt=False)

    assert feedback.device_id == "test_esp32"
    assert feedback.status == "normal"
    assert feedback.buzzer_active is False
    assert feedback.sqa_status == "GOOD"

def test_pipeline_sqa_finger_off_triggers_cek_sensor():
    sample = {
        "device_id": "test_esp32",
        "sequence_id": 99,
        "timestamp_ms": 50000,
        "telemetry": {
            "heart_rate": 0.0,
            "spo2": 0.0,
            "temperature": 32.5,
            "ppg_amplitude": 0.0,
            "finger_detected": False
        },
        "status": "normal"
    }
    feedback, dash = process_telemetry(sample, publish_mqtt=False)
    assert feedback.status == "cek_sensor"
    assert feedback.buzzer_active is False
    assert feedback.sqa_status == "POOR_QUALITY"

def test_pipeline_persistent_anomaly_buzzes():
    # Pre-fill feature engine with extreme readings
    for i in range(20):
        sample = {
            "device_id": "test_esp32_anom",
            "sequence_id": i + 1,
            "timestamp_ms": 10000 + i * 1000,
            "telemetry": {
                "heart_rate": 150.0, # High tachycardia
                "spo2": 85.0,        # Hypoxia
                "temperature": 39.5, # Fever
                "ppg_amplitude": 1500.0,
                "finger_detected": True
            },
            "status": "normal"
        }
        feedback, dash = process_telemetry(sample, publish_mqtt=False)
        
    # By sample 20, persistent anomaly count must exceed 3 and buzzer must be active
    assert feedback.status == "anomaly"
    assert feedback.buzzer_active is True
