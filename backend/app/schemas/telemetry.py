from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

class RawSensors(BaseModel):
    """Data sensor mentah."""
    heart_rate: float
    spo2: float
    temperature: float

class SensorStatus(BaseModel):
    """Status dari masing-masing sensor."""
    max30102_ok: bool
    ds18b20_ok: bool
    ppg_amplitude: float

class NetworkInfo(BaseModel):
    """Informasi jaringan ESP32."""
    wifi_rssi: int
    sequence_id: int

class TelemetryPayload(BaseModel):
    """
    Payload telemetri yang diterima dari perangkat ESP32.
    """
    device_id: str
    timestamp: int
    raw_sensors: RawSensors
    sensor_status: SensorStatus
    network: NetworkInfo

    @field_validator("raw_sensors")
    def check_ranges(cls, v):
        if not (0 <= v.spo2 <= 100):
            raise ValueError("SpO2 must be between 0 and 100")
        return v

class FeedbackDisplay(BaseModel):
    line_status: str
    oled_mode: str

class FeedbackPayload(BaseModel):
    """
    Payload umpan balik yang dikirim kembali ke perangkat ESP32.
    """
    device_id: str
    reply_sequence_id: int
    status: str
    buzzer_active: bool
    sqa_status: str
    display: FeedbackDisplay

class DashboardPayload(BaseModel):
    """
    Payload untuk dashboard Streamlit.
    """
    timestamp: int
    metrics: Dict[str, float]
    temporal_features: Optional[List[float]]
    diagnostics: Dict[str, Any]
    system: Dict[str, Any]
