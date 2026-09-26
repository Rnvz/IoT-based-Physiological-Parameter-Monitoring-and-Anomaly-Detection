import joblib
import numpy as np
from pydantic import BaseModel
import os
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class AnomalyResult(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    baseline_anomaly: bool

class PersistentResult(BaseModel):
    persistent_count: int
    is_persistent_anomaly: bool
    should_buzz: bool

class AnomalyDetector:
    """
    Layanan deteksi anomali menggunakan model Isolation Forest yang telah dilatih.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        candidate_model_paths = [
            settings.model_path,
            os.path.join(os.path.dirname(__file__), "../../../ml_pipeline/models/isolation_forest_model.joblib"),
            "ml_pipeline/models/isolation_forest_model.joblib",
            "../ml_pipeline/models/isolation_forest_model.joblib",
            "models/isolation_forest.joblib",
        ]
        candidate_scaler_paths = [
            settings.scaler_path,
            os.path.join(os.path.dirname(__file__), "../../../ml_pipeline/models/scaler.joblib"),
            "ml_pipeline/models/scaler.joblib",
            "../ml_pipeline/models/scaler.joblib",
            "models/scaler.joblib",
        ]

        resolved_model = next((p for p in candidate_model_paths if os.path.exists(p)), None)
        resolved_scaler = next((p for p in candidate_scaler_paths if os.path.exists(p)), None)

        if resolved_model and resolved_scaler:
            try:
                self.model = joblib.load(resolved_model)
                self.scaler = joblib.load(resolved_scaler)
                logger.info(f"ML Model loaded from {resolved_model} and Scaler from {resolved_scaler}.")
                
                thr_candidate_paths = [
                    os.path.join(os.path.dirname(resolved_model), "thresholds.joblib"),
                    os.path.join(os.path.dirname(__file__), "../../../ml_pipeline/models/thresholds.joblib"),
                    "ml_pipeline/models/thresholds.joblib",
                ]
                resolved_thr = next((p for p in thr_candidate_paths if os.path.exists(p)), None)
                if resolved_thr:
                    self.thresholds = joblib.load(resolved_thr)
                    # Default operating point terkunci: Target FPR 13% (threshold ~0.2535)
                    self.anomaly_threshold = self.thresholds.get(0.13, 0.2535)
                    logger.info(f"Calibrated threshold loaded: {self.anomaly_threshold:.4f} (Target FPR 13%)")
                else:
                    self.anomaly_threshold = 0.0
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning("Model or Scaler file not found. Inference will fall back to baseline thresholds only.")

    def predict(self, feature_vector: np.ndarray) -> AnomalyResult:
        # Check baseline thresholds on current values (features 0, 1, 2 = HR, SpO2, Temp)
        hr, spo2, temp = feature_vector[0], feature_vector[1], feature_vector[2]
        
        baseline_anomaly = False
        if not (60 <= hr <= 100) or spo2 < 95 or not (31 <= temp <= 35):
            baseline_anomaly = True
            
        anomaly_score = 0.0
        is_anomaly = baseline_anomaly
        
        if self.model and self.scaler:
            try:
                # Reshape for sklearn with proper column names if fitted with feature names
                features_2d = feature_vector.reshape(1, -1)
                if hasattr(self.scaler, "feature_names_in_"):
                    import pandas as pd
                    features_input = pd.DataFrame(features_2d, columns=self.scaler.feature_names_in_)
                    features_scaled = self.scaler.transform(features_input)
                    features_scaled_input = pd.DataFrame(features_scaled, columns=self.scaler.feature_names_in_)
                    anomaly_score = float(self.model.decision_function(features_scaled_input)[0])
                else:
                    features_scaled = self.scaler.transform(features_2d)
                    anomaly_score = float(self.model.decision_function(features_scaled)[0])

                # Prediksi menggunakan threshold terkalibrasi
                threshold = getattr(self, 'anomaly_threshold', 0.0)
                is_ml_anomaly = anomaly_score < threshold
                is_anomaly = is_ml_anomaly or baseline_anomaly
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                
        return AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            baseline_anomaly=baseline_anomaly
        )

class PersistentAnomalyEvaluator:
    """
    Mengevaluasi anomali berulang untuk mengurangi peringatan palsu (false positives).
    """
    def __init__(self, threshold: int = settings.persistent_anomaly_count):
        self.threshold = threshold
        self.count = 0

    def evaluate(self, is_anomaly: bool) -> PersistentResult:
        if is_anomaly:
            self.count += 1
        else:
            self.count = 0  # Strict Reset: counter langsung reset ke 0 saat sampel normal terdeteksi
            
        is_persistent = self.count >= self.threshold
        should_buzz = is_persistent
        
        return PersistentResult(
            persistent_count=self.count,
            is_persistent_anomaly=is_persistent,
            should_buzz=should_buzz
        )

    def reset(self):
        self.count = 0

