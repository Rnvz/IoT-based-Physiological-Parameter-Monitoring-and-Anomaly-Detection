try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings

class Settings(BaseSettings):
    """
    Konfigurasi aplikasi backend dan parameter SQA & model.
    """
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    model_path: str = "ml_pipeline/models/isolation_forest_model.joblib"
    scaler_path: str = "ml_pipeline/models/scaler.joblib"
    db_path: str = "sqlite+aiosqlite:///data.db"
    device_token: str = "default_token"
    log_level: str = "INFO"

    # SQA thresholds
    hr_min: float = 30.0
    hr_max: float = 220.0
    spo2_min: float = 70.0
    spo2_max: float = 100.0
    temp_min: float = 25.0
    temp_max: float = 42.0
    sqa_good_threshold: float = 0.7
    max_hr_jump: float = 40.0

    # Anomaly detection params
    persistent_anomaly_count: int = 3
    sliding_window_size: int = 15

    class Config:
        env_file = ".env"

settings = Settings()
