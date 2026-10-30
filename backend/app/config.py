from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    mqtt_broker: str = "broker.hivemq.com"
    mqtt_port: int = 1883
    topic_prefix: str = "physiomonitor/test"
    device_id: str = "esp32-01"

    db_backend: str = "csv"
    csv_dir: str = "./data"
    supabase_url: Optional[str] = None
    supabase_service_key: Optional[str] = None

    api_key: str = "default_secret_key"
    read_auth_required: bool = False
    cors_origins: str = "http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080"
    log_level: str = "INFO"

    model_path: str = "ml_pipeline/models/isolation_forest_model.joblib"
    scaler_path: str = "ml_pipeline/models/scaler.joblib"
    thresholds_path: str = "ml_pipeline/models/thresholds.joblib"
    model_version: str = "if_v1"
    window_size: int = 15
    max_message_gap_seconds: float = 5.0
    persistent_anomaly_count: int = 3
    ml_use_debounce: bool = True

    decision_mode: str = "RULE"

    # NOTE: Thresholds are demo values to be set from the user's own baseline recordings and are not clinical.
    rule_hr_low_alert: float = 50.0
    rule_hr_high_alert: float = 100.0
    rule_spo2_alert: float = 90.0
    rule_temp_high_alert: Optional[float] = 38.0
    rule_temp_low_alert: Optional[float] = None

    rule_hr_low_warn: float = 55.0
    rule_hr_high_warn: float = 95.0
    rule_spo2_warn: float = 94.0
    rule_temp_high_warn: Optional[float] = 37.5
    rule_temp_low_warn: Optional[float] = None

    cmd_seq_file: str = "./data/cmd_seq.json"

    from pydantic import field_validator
    @field_validator("rule_temp_high_alert", "rule_temp_low_alert", "rule_temp_high_warn", "rule_temp_low_warn", mode="before")
    @classmethod
    def empty_temp_to_none(cls, v):
        if v is None or v == "" or (isinstance(v, str) and v.strip() == ""):
            return None
        return float(v)

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "../.env"),
        extra="ignore",
        env_file_encoding="utf-8"
    )

settings = Settings()
