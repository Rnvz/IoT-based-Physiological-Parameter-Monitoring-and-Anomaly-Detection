from pydantic import BaseModel, Field, field_validator, model_validator
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
    Mendukung format canonical backend maupun format langsung dari firmware ESP32.
    """
    device_id: str
    timestamp: int
    raw_sensors: RawSensors
    sensor_status: SensorStatus
    network: NetworkInfo
    sequence_id: Optional[int] = None
    timestamp_ms: Optional[int] = None
    status: Optional[str] = None
    amp: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_firmware_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalisasi timestamp jika diberikan timestamp_ms
            if "timestamp_ms" in data and "timestamp" not in data:
                data["timestamp"] = int(data["timestamp_ms"])
            elif "timestamp" in data and "timestamp_ms" not in data:
                data["timestamp_ms"] = int(data["timestamp"])

            # Normalisasi sequence_id
            seq = 0
            if "sequence_id" in data and data["sequence_id"] is not None:
                seq = int(data["sequence_id"])
            elif "network" in data:
                if hasattr(data["network"], "sequence_id"):
                    seq = int(data["network"].sequence_id)
                elif isinstance(data["network"], dict):
                    seq = int(data["network"].get("sequence_id", 0))
            data.setdefault("sequence_id", seq)

            # Format Flat JSON firmware (contoh: {"hr": 75.0, "spo2": 98.0, "temp": 36.5, "amp": 45000, "status": "NORMAL"})
            if "hr" in data or ("temp" in data and "raw_sensors" not in data and "telemetry" not in data):
                import time
                data.setdefault("device_id", "esp32_hardware")
                
                # Timestamp fallback ke epoch milliseconds saat ini
                if "timestamp" not in data:
                    data["timestamp"] = int(time.time() * 1000)
                if "timestamp_ms" not in data:
                    data["timestamp_ms"] = data["timestamp"]

                hr = float(data.get("hr", 0.0))
                spo2 = float(data.get("spo2", 98.0))
                temp = float(data.get("temp", 0.0))
                status_str = str(data.get("status", "NORMAL"))
                
                # Heuristik deteksi jari: jika status "MENUNGGU_SENSOR", "MENUNGGU SENSOR", "MENDETEKSI", atau HR <= 0 atau SpO2 <= 0
                is_menunggu = (
                    status_str.upper() in ["MENUNGGU_SENSOR", "MENUNGGU SENSOR", "MENDETEKSI"]
                    or hr <= 0.0
                    or spo2 <= 0.0
                )

                # Ambil nilai amplitudo PPG nyata jika dikirim oleh firmware (lastIR)
                amp_in = data.get("amp")
                if amp_in is not None:
                    amp = float(amp_in)
                    max_ok = not is_menunggu and amp > 0.0
                else:
                    amp = 0.0 if is_menunggu else 2000.0
                    max_ok = not is_menunggu

                if "raw_sensors" not in data:
                    data["raw_sensors"] = {
                        "heart_rate": hr,
                        "spo2": spo2,
                        "temperature": temp,
                    }
                if "sensor_status" not in data:
                    data["sensor_status"] = {
                        "max30102_ok": max_ok,
                        "ds18b20_ok": temp > 0.0,
                        "ppg_amplitude": amp,
                    }
                if "network" not in data:
                    data["network"] = {
                        "wifi_rssi": int(data.get("wifi_rssi", -50)),
                        "sequence_id": seq,
                    }

            # Format hierarkis dengan blok 'telemetry'
            elif "telemetry" in data and isinstance(data["telemetry"], dict):
                t = data["telemetry"]
                if "raw_sensors" not in data:
                    data["raw_sensors"] = {
                        "heart_rate": float(t.get("heart_rate", 0.0)),
                        "spo2": float(t.get("spo2", 0.0)),
                        "temperature": float(t.get("temperature", 0.0)),
                    }
                if "sensor_status" not in data:
                    finger = bool(t.get("finger_detected", True))
                    amp = float(t.get("ppg_amplitude", 2000.0 if finger else 0.0))
                    temp = float(t.get("temperature", 0.0))
                    data["sensor_status"] = {
                        "max30102_ok": finger and amp > 0,
                        "ds18b20_ok": temp > 0,
                        "ppg_amplitude": amp,
                    }
                if "network" not in data:
                    data["network"] = {
                        "wifi_rssi": int(data.get("wifi_rssi", -50)),
                        "sequence_id": seq,
                    }
        return data

    @field_validator("raw_sensors")
    @classmethod
    def check_ranges(cls, v):
        if not (0 <= v.spo2 <= 100):
            raise ValueError("SpO2 must be between 0 and 100")
        return v

class FeedbackDisplay(BaseModel):
    line_status: str = "NORMAL"
    oled_mode: str = "STANDARD"

class FeedbackPayload(BaseModel):
    """
    Payload umpan balik yang dikirim kembali ke perangkat ESP32.
    """
    device_id: str
    reply_sequence_id: int = 0
    status: str
    buzzer_active: bool
    sqa_status: str
    display: FeedbackDisplay = Field(default_factory=FeedbackDisplay)
    anomaly_score: Optional[float] = None

class DashboardPayload(BaseModel):
    """
    Payload untuk dashboard Streamlit.
    """
    timestamp: int
    metrics: Dict[str, float]
    temporal_features: Optional[List[float]]
    diagnostics: Dict[str, Any]
    system: Dict[str, Any]
