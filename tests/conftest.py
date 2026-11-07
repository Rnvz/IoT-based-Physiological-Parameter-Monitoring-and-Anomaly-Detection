import sys
import os
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
backend_dir = os.path.join(repo_root, 'backend')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

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
    return mocker.patch('joblib.load')

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture(autouse=True)
def configure_test_settings(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "api_key", "test_secret_key")
    monkeypatch.setattr(settings, "topic_prefix", "physiomonitor/test")
    from app.mqtt.consumer import MQTTService
    monkeypatch.setattr(MQTTService, "start", lambda self: None)
    monkeypatch.setattr(MQTTService, "stop", lambda self: None)

