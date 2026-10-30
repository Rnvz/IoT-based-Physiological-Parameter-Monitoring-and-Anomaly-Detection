import os
import joblib
from typing import Dict, Tuple, Optional
from app.config import settings

class ThresholdEvaluator:
    """
    Evaluates ML anomaly score against thresholds loaded dynamically from thresholds.joblib.
    Sign convention (Isolation Forest decision_function):
      - Higher score: more normal
      - Lower score: more anomalous
    Mapping:
      - score >= thr[0.13] -> severity 0 (NORMAL)
      - thr[0.05] <= score < thr[0.13] -> severity 1 (ANOMALY / warning)
      - score < thr[0.05] -> severity 2 (ANOMALY / critical)
    """

    def __init__(self, thresholds_path: str = settings.thresholds_path):
        self.thresholds_path = thresholds_path
        self.thr_013: float = 0.252450
        self.thr_005: float = 0.097345
        self.thresholds: Dict[float, float] = {}
        self.load_thresholds()

    def load_thresholds(self):
        paths_to_try = [
            self.thresholds_path,
            os.path.join(os.path.dirname(__file__), "../../../ml_pipeline/models/thresholds.joblib"),
            "ml_pipeline/models/thresholds.joblib",
            "../ml_pipeline/models/thresholds.joblib",
        ]
        resolved = next((p for p in paths_to_try if os.path.exists(p)), None)
        if resolved:
            try:
                self.thresholds = joblib.load(resolved)
                self.thr_013 = float(self.thresholds.get(0.13, 0.252450))
                self.thr_005 = float(self.thresholds.get(0.05, 0.097345))
            except Exception as e:
                print(f"[ThresholdEvaluator] Error loading thresholds from {resolved}: {e}")

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
        self.last_severity: int = 0

    def evaluate(self, raw_severity: int) -> int:
        if raw_severity == 0:
            # Strict reset: immediately clear counter on normal reading
            self.consecutive_count = 0
            self.last_severity = 0
            return 0

        self.consecutive_count += 1
        self.last_severity = raw_severity

        # If debounced limit reached, emit raw severity (1 or 2); otherwise hold at 0 or 1
        if self.consecutive_count >= self.debounce_count:
            return raw_severity
        else:
            # During debounce window (< 3 consecutive anomalies), suppress critical severity 2 to warning 1
            return 1 if raw_severity >= 1 else 0

    def reset(self):
        self.consecutive_count = 0
        self.last_severity = 0
