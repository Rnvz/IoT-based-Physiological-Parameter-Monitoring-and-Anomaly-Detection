import pytest
import numpy as np

@pytest.fixture
def sample_normal_telemetry():
    from app.schemas.telemetry import TelemetryPayload, RawSensors, SensorStatus, NetworkInfo
    return TelemetryPayload(
        device_id="test_dev",
        timestamp=1000,
        raw_sensors=RawSensors(heart_rate=75.0, spo2=98.0, temperature=36.5),
        sensor_status=SensorStatus(max30102_ok=True, ds18b20_ok=True, ppg_amplitude=2000.0),
        network=NetworkInfo(wifi_rssi=-50, sequence_id=1)
    )

@pytest.fixture
def sample_anomaly_telemetry():
    from app.schemas.telemetry import TelemetryPayload, RawSensors, SensorStatus, NetworkInfo
    return TelemetryPayload(
        device_id="test_dev",
        timestamp=1000,
        raw_sensors=RawSensors(heart_rate=150.0, spo2=92.0, temperature=39.5),
        sensor_status=SensorStatus(max30102_ok=True, ds18b20_ok=True, ppg_amplitude=2000.0),
        network=NetworkInfo(wifi_rssi=-50, sequence_id=1)
    )

@pytest.fixture
def mock_model(mocker):
    # Mocking joblib.load for testing model service
    return mocker.patch('joblib.load')
