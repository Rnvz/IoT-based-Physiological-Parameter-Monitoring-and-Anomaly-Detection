import os
import joblib
from pydantic import BaseModel
from typing import Optional, Dict
import numpy as np
from app.config import settings
from app.ml.device_window import DeviceWindowManager
from app.ml.severity import ThresholdEvaluator, DevicePersistentEvaluator

class MLInferenceResult(BaseModel):
    raw_score: Optional[float] = None
    raw_severity: Optional[int] = None
    debounced_severity: Optional[int] = None
    label: str = "NO_DATA"  # "NORMAL", "ANOMALY", "NO_DATA", "SENSOR_ERROR"
    model_version: str = "if_v1"
    is_warmup: bool = False

class MLService:
    """
    ML Adapter wrapping Isolation Forest model and RobustScaler.
    Enforces per-device rolling window and debouncing.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold_evaluator = ThresholdEvaluator(thresholds_path=settings.thresholds_path)
        self.window_manager = DeviceWindowManager(
            window_size=settings.window_size,
            max_gap_seconds=settings.max_message_gap_seconds
        )
        self.evaluators: Dict[str, DevicePersistentEvaluator] = {}
        self.load_artifacts()

    def load_artifacts(self):
        candidate_model_paths = [
            settings.model_path,
            os.path.join(os.path.dirname(__file__), "../../../ml_pipeline/models/isolation_forest_model.joblib"),
            "ml_pipeline/models/isolation_forest_model.joblib",
            "../ml_pipeline/models/isolation_forest_model.joblib",
        ]
        candidate_scaler_paths = [
            settings.scaler_path,
            os.path.join(os.path.dirname(__file__), "../../../ml_pipeline/models/scaler.joblib"),
            "ml_pipeline/models/scaler.joblib",
            "../ml_pipeline/models/scaler.joblib",
        ]

        resolved_model = next((p for p in candidate_model_paths if os.path.exists(p)), None)
        resolved_scaler = next((p for p in candidate_scaler_paths if os.path.exists(p)), None)

        if resolved_model and resolved_scaler:
            try:
                self.model = joblib.load(resolved_model)
                self.scaler = joblib.load(resolved_scaler)
                print(f"[MLService] Loaded model from {resolved_model} and scaler from {resolved_scaler}")
            except Exception as e:
                print(f"[MLService] Failed loading ML artifacts: {e}")
        else:
            print("[MLService] ML artifacts not found. Model inference will be unavailable.")

    def _get_evaluator(self, device_id: str) -> DevicePersistentEvaluator:
        if device_id not in self.evaluators:
            self.evaluators[device_id] = DevicePersistentEvaluator(
                debounce_count=settings.persistent_anomaly_count
            )
        return self.evaluators[device_id]

    def reset_device(self, device_id: str):
        self.window_manager.reset_device(device_id)
        if device_id in self.evaluators:
            self.evaluators[device_id].reset()

    def process_reading(
        self,
        device_id: str,
        hr: Optional[float],
        spo2: Optional[float],
        temp: Optional[float],
        finger: bool,
        timestamp: Optional[float] = None
    ) -> MLInferenceResult:
        """
        Processes a reading for ML anomaly scoring.
        Resets window if finger=False.
        Returns NO_DATA during warm-up (<15 samples), never NORMAL.
        """
        # If finger is off or vital values are missing/invalid, reset window and skip inference
        if not finger or hr is None or spo2 is None or temp is None:
            self.reset_device(device_id)
            return MLInferenceResult(label="NO_DATA", model_version=settings.model_version)

        if self.model is None or self.scaler is None:
            return MLInferenceResult(label="NO_DATA", model_version=settings.model_version)

        # Push to per-device rolling window
        window = self.window_manager.get_window(device_id)
        feature_vector = window.push(hr, spo2, temp, timestamp=timestamp)

        # During warm-up (< 15 samples), return NO_DATA, never NORMAL
        if feature_vector is None:
            return MLInferenceResult(
                label="NO_DATA",
                model_version=settings.model_version,
                is_warmup=True
            )

        # Scale and infer
        try:
            features_2d = feature_vector.reshape(1, -1)
            # Support sklearn feature names warning avoidance if available
            if hasattr(self.scaler, "feature_names_in_"):
                import pandas as pd
                cols = list(self.scaler.feature_names_in_)
                features_input = pd.DataFrame(features_2d, columns=cols)
                features_scaled = self.scaler.transform(features_input)
                features_scaled_input = pd.DataFrame(features_scaled, columns=cols)
                score = float(self.model.decision_function(features_scaled_input)[0])
            else:
                features_scaled = self.scaler.transform(features_2d)
                score = float(self.model.decision_function(features_scaled)[0])

            raw_sev, raw_label = self.threshold_evaluator.score_to_raw_severity(score)
            evaluator = self._get_evaluator(device_id)
            debounced_sev = evaluator.evaluate(raw_sev)

            return MLInferenceResult(
                raw_score=round(score, 4),
                raw_severity=raw_sev,
                debounced_severity=debounced_sev,
                label=raw_label,
                model_version=settings.model_version,
                is_warmup=False
            )
        except Exception as e:
            print(f"[MLService] Inference error for device {device_id}: {e}")
            return MLInferenceResult(label="SENSOR_ERROR", model_version=settings.model_version)

_ml_service_instance: Optional[MLService] = None

def get_ml_service() -> MLService:
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLService()
    return _ml_service_instance
