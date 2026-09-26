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
    severity: str = "NORMAL"  # "NORMAL", "LOW DEVIATION", or "HIGH DEVIATION"

class PersistentResult(BaseModel):
    persistent_count: int
    is_persistent_anomaly: bool
    should_buzz: bool
    severity: str = "NORMAL"   # "NORMAL", "LOW DEVIATION", "LOW DEVIATION (SUSTAINED)", "HIGH DEVIATION", "HIGH DEVIATION (SUSTAINED)"
    oled_status: str = "NORMAL" # "NORMAL", "LOW DEV", "LOW DEV (S)", "HIGH DEV", "HIGH DEV (S)"

class AnomalyDetector:
    """
    Layanan deteksi anomali menggunakan model Isolation Forest yang telah dilatih.
    Mendukung penentuan tingkat deviasi (LOW DEVIATION vs HIGH DEVIATION)
    yang terinspirasi dari National Early Warning Score (NEWS).
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.thresholds = {}
        self.anomaly_threshold = 0.0
        self.high_threshold = -0.08
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
                    # Operating point terkunci: Target FPR 13% dari thresholds.joblib
                    self.anomaly_threshold = float(self.thresholds.get(0.13, 0.0629))
                    # High deviation threshold: Target FPR 5% dari thresholds.joblib
                    self.high_threshold = float(self.thresholds.get(0.05, -0.0280))
                    logger.info(f"Calibrated thresholds: LOW/ANOMALY={self.anomaly_threshold:.4f} (FPR 13%), HIGH={self.high_threshold:.4f} (FPR 5%)")
                else:
                    self.anomaly_threshold = 0.0629
                    self.high_threshold = -0.0280
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning("Model or Scaler file not found. Inference will default to NORMAL.")

    def predict(self, feature_vector: np.ndarray) -> AnomalyResult:
        """
        Klasifikasi anomali dan penentuan tingkat deviasi (severity) MURNI
        berdasarkan skor decision_function model Machine Learning dibandingkan
        terhadap threshold terkalibrasi (FPR 13% & FPR 5% dari thresholds.joblib):
        - anomaly_score >= anomaly_threshold (FPR 13%) -> NORMAL
        - high_threshold <= anomaly_score < anomaly_threshold -> LOW DEVIATION
        - anomaly_score < high_threshold (FPR 5%) -> HIGH DEVIATION
        
        Murni statistik tanpa aturan manual/hardcoded pada tanda vital.
        """
        anomaly_score = 0.0
        is_anomaly = False
        severity = "NORMAL"

        if self.model and self.scaler:
            try:
                features_2d = feature_vector.reshape(1, -1)
                if hasattr(self.scaler, "feature_names_in_"):
                    import pandas as pd
                    scaler_cols = list(self.scaler.feature_names_in_)
                    if len(scaler_cols) == 14 and len(feature_vector) == 13:
                        features_2d = np.append(feature_vector, 1.0).reshape(1, -1)
                    features_input = pd.DataFrame(features_2d, columns=scaler_cols)
                    features_scaled = self.scaler.transform(features_input)
                    features_scaled_input = pd.DataFrame(features_scaled, columns=scaler_cols)
                    anomaly_score = float(self.model.decision_function(features_scaled_input)[0])
                else:
                    if getattr(self.scaler, "n_features_in_", len(feature_vector)) == 14 and len(feature_vector) == 13:
                        features_2d = np.append(feature_vector, 1.0).reshape(1, -1)
                    features_scaled = self.scaler.transform(features_2d)
                    anomaly_score = float(self.model.decision_function(features_scaled)[0])

                threshold = getattr(self, 'anomaly_threshold', 0.0629)
                high_threshold = getattr(self, 'high_threshold', -0.0280)
                
                # Klasifikasi murni berdasarkan skor model ML
                if anomaly_score >= threshold:
                    is_anomaly = False
                    severity = "NORMAL"
                elif anomaly_score < high_threshold:
                    is_anomaly = True
                    severity = "HIGH DEVIATION"
                else:
                    is_anomaly = True
                    severity = "LOW DEVIATION"

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                severity = "NORMAL"
        else:
            logger.warning("ML Model/Scaler not loaded. Prediction defaulted to NORMAL.")
                
        return AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            baseline_anomaly=False,
            severity=severity
        )

class PersistentAnomalyEvaluator:
    """
    Mengevaluasi anomali berulang dan menentukan status severity bergradasi.
    Terinspirasi dari National Early Warning Score (NEWS) untuk early-warning non-diagnostik:
    - NORMAL: data berada dalam profil normal
    - LOW DEVIATION: deviasi ringan (< 10 sampel)
    - LOW DEVIATION (SUSTAINED): deviasi ringan bertahan >= 10 sampel
    - HIGH DEVIATION: deviasi tinggi/kritis (< 10 sampel)
    - HIGH DEVIATION (SUSTAINED): deviasi tinggi bertahan >= 10 sampel
    """
    def __init__(self, threshold: int = settings.persistent_anomaly_count, sustained_threshold: int = 10):
        self.threshold = threshold          # Default 3 sampel untuk aktivasi buzzer
        self.sustained_threshold = sustained_threshold # Default 10 sampel untuk status SUSTAINED
        self.count = 0
        self.current_base_severity = "NORMAL"

    def evaluate(self, is_anomaly: bool, base_severity: str = "LOW DEVIATION") -> PersistentResult:
        if not is_anomaly:
            self.count = 0  # Strict Reset: counter langsung reset ke 0 saat sampel normal terdeteksi
            self.current_base_severity = "NORMAL"
            return PersistentResult(
                persistent_count=0,
                is_persistent_anomaly=False,
                should_buzz=False,
                severity="NORMAL",
                oled_status="NORMAL"
            )
            
        self.count += 1
        self.current_base_severity = base_severity
        
        is_high = (base_severity == "HIGH DEVIATION")
        is_sustained = (self.count >= self.sustained_threshold)
        
        if is_high:
            severity = "HIGH DEVIATION (SUSTAINED)" if is_sustained else "HIGH DEVIATION"
            oled_status = "HIGH DEV (S)" if is_sustained else "HIGH DEV"
        else:
            severity = "LOW DEVIATION (SUSTAINED)" if is_sustained else "LOW DEVIATION"
            oled_status = "LOW DEV (S)" if is_sustained else "LOW DEV"

        is_persistent = self.count >= self.threshold
        should_buzz = is_persistent
        
        return PersistentResult(
            persistent_count=self.count,
            is_persistent_anomaly=is_persistent,
            should_buzz=should_buzz,
            severity=severity,
            oled_status=oled_status
        )

    def reset(self):
        self.count = 0
        self.current_base_severity = "NORMAL"
