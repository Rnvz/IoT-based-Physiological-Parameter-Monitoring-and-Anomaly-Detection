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
    assert feedback.status == "NORMAL"
    assert feedback.display.line_status == "NORMAL"
    assert feedback.buzzer_active is False
    assert feedback.sqa_status == "GOOD"

def test_pipeline_sqa_finger_off_triggers_signal_quality_low():
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
    assert feedback.status == "SIGNAL QUALITY LOW"
    assert feedback.display.line_status == "SIG QUAL LOW"
    assert feedback.buzzer_active is False
    assert feedback.sqa_status == "POOR_QUALITY"

def test_pipeline_persistent_anomaly_buzzes():
    # Pre-fill feature buffer (15 samples) and advance count
    # Samples 1-14: buffer filling (features is None)
    # Sample 15: count = 1
    # Sample 18: count = 4 (>= 3 -> buzzer active, < 10 -> HIGH DEVIATION)
    # Sample 25: count = 11 (>= 10 -> HIGH DEVIATION (SUSTAINED))
    for i in range(25):
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
        if i == 17:  # Sample 18 (count = 4)
            assert feedback.status == "HIGH DEVIATION"
            assert feedback.display.line_status == "HIGH DEV"
            assert feedback.buzzer_active is True
        
    # By sample 25 (count = 11 >= 10), status is HIGH DEVIATION (SUSTAINED) and buzzer active
    assert feedback.status == "HIGH DEVIATION (SUSTAINED)"
    assert feedback.display.line_status == "HIGH DEV (S)"
    assert feedback.buzzer_active is True

def test_pipeline_low_deviation_and_sustained():
    # Pre-fill feature engine with 15 normal samples
    for i in range(15):
        sample = {
            "device_id": "test_esp32_low",
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
        process_telemetry(sample, publish_mqtt=False)

    # Send 4 mild anomaly samples (HR=105, SpO2=94) -> count=4 (>= 3 -> buzz=True, < 10 -> LOW DEVIATION)
    for i in range(4):
        sample = {
            "device_id": "test_esp32_low",
            "sequence_id": 20 + i,
            "timestamp_ms": 20000 + i * 1000,
            "telemetry": {
                "heart_rate": 105.0,
                "spo2": 94.0,
                "temperature": 32.5,
                "ppg_amplitude": 2500.0,
                "finger_detected": True
            },
            "status": "normal"
        }
        fb4, _ = process_telemetry(sample, publish_mqtt=False)
    assert fb4.status == "LOW DEVIATION"
    assert fb4.display.line_status == "LOW DEV"
    assert fb4.buzzer_active is True

    # Send 6 more mild anomaly samples (total count = 10 -> LOW DEVIATION (SUSTAINED))
    for i in range(6):
        sample = {
            "device_id": "test_esp32_low",
            "sequence_id": 30 + i,
            "timestamp_ms": 30000 + i * 1000,
            "telemetry": {
                "heart_rate": 105.0,
                "spo2": 94.0,
                "temperature": 32.5,
                "ppg_amplitude": 2500.0,
                "finger_detected": True
            },
            "status": "normal"
        }
        fb10, _ = process_telemetry(sample, publish_mqtt=False)
    assert fb10.status == "LOW DEVIATION (SUSTAINED)"
    assert fb10.display.line_status == "LOW DEV (S)"
    assert fb10.buzzer_active is True

    # Recovery stream (flush rolling buffer with normal readings) -> Strict Reset
    for i in range(15):
        recovery_sample = {
            "device_id": "test_esp32_low",
            "sequence_id": 40 + i,
            "timestamp_ms": 40000 + i * 1000,
            "telemetry": {
                "heart_rate": 75.0,
                "spo2": 98.0,
                "temperature": 32.5,
                "ppg_amplitude": 2500.0,
                "finger_detected": True
            },
            "status": "normal"
        }
        fb_rec, _ = process_telemetry(recovery_sample, publish_mqtt=False)
    assert fb_rec.status == "NORMAL"
    assert fb_rec.display.line_status == "NORMAL"
    assert fb_rec.buzzer_active is False
