import os
import joblib
import logging
from typing import Dict, Tuple
from app.config import settings

logger = logging.getLogger(__name__)

class ThresholdEvaluator:
    """
    Evaluates ML anomaly score against thresholds loaded dynamically from thresholds.joblib.
    Sign convention (Isolation Forest decision_function):
      - Higher score: more normal
      - Lower score: more anomalous
    Mapping:
      - score >= thr_warn -> severity 0 (NORMAL)
      - thr_crit <= score < thr_warn -> severity 1 (ANOMALY / warning)
      - score < thr_crit -> severity 2 (ANOMALY / critical)
    """

    def __init__(self, thresholds_path: str = settings.thresholds_path):
        self.thresholds_path = thresholds_path
        self.thr_013: float = 0.048891
        self.thr_005: float = -0.016385
        self.thresholds: Dict = {}
        self.load_thresholds()

    def load_thresholds(self):
        paths_to_try = [
            self.thresholds_path,
            os.path.join(os.path.dirname(__file__), "../../../ml/models/thresholds.joblib"),
            "ml/models/thresholds.joblib",
            "../ml/models/thresholds.joblib",
            os.path.join(os.path.dirname(__file__), "../../../ml/models/physio_synthetic/thresholds.joblib"),
            "ml/models/physio_synthetic/thresholds.joblib",
            "../ml/models/physio_synthetic/thresholds.joblib",
            os.path.join(os.path.dirname(__file__), "../../../ml_pipeline/models/physio_synthetic/thresholds.joblib"),
            os.path.join(os.path.dirname(__file__), "../../../models/physio_synthetic/thresholds.joblib"),
            "ml_pipeline/models/physio_synthetic/thresholds.joblib",
            "../ml_pipeline/models/physio_synthetic/thresholds.joblib",
            "models/physio_synthetic/thresholds.joblib",
            "../models/physio_synthetic/thresholds.joblib",
            os.path.join(os.path.dirname(__file__), "../../../ml_pipeline/models/thresholds.joblib"),
            "ml_pipeline/models/thresholds.joblib",
            "../ml_pipeline/models/thresholds.joblib",
        ]
        resolved = next((p for p in paths_to_try if os.path.exists(p)), None)
        if resolved:
            try:
                self.thresholds = joblib.load(resolved)
                self.thr_013 = float(
                    self.thresholds.get("fpr_10pct")
                    or self.thresholds.get("fpr_13pct")
                    or self.thresholds.get(0.13)
                    or 0.048891
                )
                self.thr_005 = float(
                    self.thresholds.get("fpr_05pct")
                    or self.thresholds.get(0.05)
                    or -0.016385
                )
                logger.info(f"[ThresholdEvaluator] Loaded thresholds from {resolved}: thr_warn={self.thr_013}, thr_crit={self.thr_005}")
            except Exception as e:
                logger.error(f"[ThresholdEvaluator] Error loading thresholds from {resolved}: {e}")
        else:
            logger.warning("[ThresholdEvaluator] Threshold artifacts not found. Fallback thresholds active.")

    def score_to_raw_severity(self, score: float) -> Tuple[int, str]:
        if score >= self.thr_013:
            return 0, "NORMAL"
        elif score >= self.thr_005:
            return 1, "ANOMALY"
        else:
            return 2, "ANOMALY"

class DevicePersistentEvaluator:
    """Maintains consecutive anomaly debouncing per device."""

    def __init__(self, debounce_count: int = settings.persistent_anomaly_count):
        self.debounce_count = debounce_count
        self.consecutive_count: int = 0
        self.consecutive_crit: int = 0
        self.last_severity: int = 0

    def evaluate(self, raw_severity: int) -> int:
        if raw_severity == 0:
            self.consecutive_count = 0
            self.consecutive_crit = 0
            self.last_severity = 0
            return 0

        self.consecutive_count += 1
        if raw_severity >= 2:
            self.consecutive_crit += 1
        else:
            self.consecutive_crit = 0

        self.last_severity = raw_severity

        if self.consecutive_crit >= self.debounce_count:
            return 2
        elif self.consecutive_count >= self.debounce_count:
            return 1
        else:
            return 0

    def reset(self):
        self.consecutive_count = 0
        self.consecutive_crit = 0
        self.last_severity = 0
