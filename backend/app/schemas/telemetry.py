from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Optional, Any

class RawSensors(BaseModel):
    heart_rate: float = 75.0
    spo2: float = 98.0
    temperature: float = 36.5

class SensorStatus(BaseModel):
    max30102_ok: bool = True
    ds18b20_ok: bool = True
    ppg_amplitude: float = 2000.0

class NetworkInfo(BaseModel):
    wifi_rssi: int = -50
    sequence_id: int = 1

class FeedbackDisplay(BaseModel):
    line_status: str = "NORMAL"
    oled_mode: str = "STANDARD"

class FeedbackPayload(BaseModel):
    device_id: str
    reply_sequence_id: int = 0
    status: str = "NORMAL"
    buzzer_active: bool = False
    sqa_status: str = "GOOD"
    display: FeedbackDisplay = FeedbackDisplay()
    anomaly_score: Optional[float] = None

class DashboardPayload(BaseModel):
    timestamp: int = 0
    metrics: dict = {}
    temporal_features: Optional[list] = None
    diagnostics: dict = {}
    system: dict = {}

class TelemetryPayload(BaseModel):
    """
    Schema for incoming /data MQTT messages from ESP32.
    Format: {"hr":75.0,"spo2":98.0,"temp":36.5,"hum":62.3,"room_temp":29.1,"finger":true,"status":"NORMAL"}
    """
    model_config = ConfigDict(extra="ignore")

    device_id: Optional[str] = "esp32-01"
    hr: Optional[float] = None
    spo2: Optional[float] = None
    temp: Optional[float] = None
    hum: Optional[float] = None
    room_temp: Optional[float] = None
    finger: bool = False
    status: Optional[str] = "NORMAL"

    timestamp: Optional[int] = None
    sequence_id: Optional[int] = None
    raw_sensors: Optional[RawSensors] = None
    sensor_status: Optional[SensorStatus] = None
    network: Optional[NetworkInfo] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        
        if "telemetry" in data and isinstance(data["telemetry"], dict):
            t = data["telemetry"]
            data.setdefault("hr", t.get("heart_rate"))
            data.setdefault("spo2", t.get("spo2"))
            data.setdefault("temp", t.get("temperature"))
            data.setdefault("finger", t.get("finger_detected", True))
        elif "raw_sensors" in data and isinstance(data["raw_sensors"], dict):
            r = data["raw_sensors"]
            data.setdefault("hr", r.get("heart_rate"))
            data.setdefault("spo2", r.get("spo2"))
            data.setdefault("temp", r.get("temperature"))

        return data

    @field_validator("hr", mode="before")
    @classmethod
    def clean_hr(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid HR value: {v}")
        if val <= 0.0:
            return None
        if not (20.0 <= val <= 250.0):
            raise ValueError(f"HR out of biological bounds (20-250): {val}")
        return val

    @field_validator("spo2", mode="before")
    @classmethod
    def clean_spo2(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid SpO2 value: {v}")
        if val <= 0.0:
            return None
        if not (40.0 <= val <= 100.0):
            raise ValueError(f"SpO2 out of biological bounds (40-100): {val}")
        return val

    @field_validator("temp", mode="before")
    @classmethod
    def clean_temp(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid skin temp value: {v}")
        if val <= 0.0:
            return None
        if not (20.0 <= val <= 45.0):
            raise ValueError(f"Skin temp out of biological bounds (20-45): {val}")
        return val

    @field_validator("hum", mode="before")
    @classmethod
    def clean_hum(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid humidity value: {v}")
        if not (0.0 <= val <= 100.0):
            raise ValueError(f"Humidity out of bounds (0-100): {val}")
        return val

    @field_validator("room_temp", mode="before")
    @classmethod
    def clean_room_temp(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid room temp value: {v}")
        if not (-20.0 <= val <= 65.0):
            raise ValueError(f"Room temp out of bounds (-20-65): {val}")
        return val

    @property
    def has_valid_vitals(self) -> bool:
        """Returns True if finger is detected and all core vital signs are non-null."""
        return bool(self.finger and self.hr is not None and self.spo2 is not None and self.temp is not None)
