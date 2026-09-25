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
        if os.path.exists(settings.model_path) and os.path.exists(settings.scaler_path):
            try:
                self.model = joblib.load(settings.model_path)
                self.scaler = joblib.load(settings.scaler_path)
                logger.info("ML Model and Scaler loaded successfully.")
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
                # Reshape for sklearn
                features_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
                # Isolation Forest returns -1 for anomaly, 1 for normal
                pred = self.model.predict(features_scaled)[0]
                anomaly_score = float(self.model.decision_function(features_scaled)[0])
                is_anomaly = pred == -1 or baseline_anomaly
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
            self.count = max(0, self.count - 1)  # gradual decay or reset to 0
            
        is_persistent = self.count >= self.threshold
        should_buzz = is_persistent
        
        return PersistentResult(
            persistent_count=self.count,
            is_persistent_anomaly=is_persistent,
            should_buzz=should_buzz
        )
