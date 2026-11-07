"""
Self-contained NumPy forward pass for dense_ae_kaggle_v1 (13 physiological features).
Requires only NumPy and SciPy (for exact GELU erf).
"""

import numpy as np
from scipy.special import erf

FEATURE_ORDER = [
    "Heart_Rate", "SpO2", "Temperature",
    "Heart_Rate_Delta", "SpO2_Delta", "Temperature_Delta",
    "MA_Heart_Rate", "MA_SpO2", "MA_Temperature",
    "Var_Heart_Rate", "Var_SpO2", "Var_Temperature",
    "Rate_of_Change"
]

SCALER_MIN = np.array([40.0, 70.0, 25.0, -40.0, -15.0, -15.0, 40.0, 70.0, 25.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
SCALER_MAX = np.array([160.0, 100.0, 43.0, 40.0, 15.0, 15.0, 160.0, 100.0, 43.0, 500.0, 50.0, 70.0, 1.0], dtype=np.float32)
SCALER_RANGE = SCALER_MAX - SCALER_MIN

# Default operational and clinical thresholds
WARN_THRESHOLD = 0.000200
CRIT_THRESHOLD = 0.001000
IOT_STREAM_WARN = 0.006000
IOT_STREAM_CRIT = 0.010400
PERSISTENCE_N = 3

def gelu_np(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(np.float32)

def layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias

class EdgeAutoencoderInference:
    def __init__(self, weights_path: str = "dl/artifacts/dl_v1/model_weights.npz"):
        data = np.load(weights_path)
        self.w = {k: data[k] for k in data.files}
        self.consecutive_warn = 0
        self.consecutive_crit = 0
        self.active_severity = 0

    def forward(self, norm: np.ndarray) -> np.ndarray:
        w = self.w
        h = norm @ w["encoder.0.weight"].T + w["encoder.0.bias"]
        h = layer_norm_np(h, w["encoder.1.weight"], w["encoder.1.bias"])
        h = gelu_np(h)
        h = h @ w["encoder.4.weight"].T + w["encoder.4.bias"]
        h = layer_norm_np(h, w["encoder.5.weight"], w["encoder.5.bias"])
        h = gelu_np(h)
        z = h @ w["encoder.7.weight"].T + w["encoder.7.bias"]

        d = z @ w["decoder.0.weight"].T + w["decoder.0.bias"]
        d = layer_norm_np(d, w["decoder.1.weight"], w["decoder.1.bias"])
        d = gelu_np(d)
        d = d @ w["decoder.4.weight"].T + w["decoder.4.bias"]
        d = layer_norm_np(d, w["decoder.5.weight"], w["decoder.5.bias"])
        d = gelu_np(d)
        rec = d @ w["decoder.7.weight"].T + w["decoder.7.bias"]
        return rec

    def predict_sample(self, values: dict) -> dict:
        """
        values: dict containing keys ['Heart_Rate', 'SpO2'] and optionally ['Temperature']
        """
        hr = float(values.get("Heart_Rate", values.get("hr", 75.0)))
        spo2 = float(values.get("SpO2", values.get("spo2", 98.0)))
        temp = float(values.get("Temperature", values.get("temp", 36.5)))

        vec = np.array([[
            hr, spo2, temp,
            0.0, 0.0, 0.0,
            hr, spo2, temp,
            0.0, 0.0, 0.0,
            0.0
        ]], dtype=np.float32)

        norm = (vec - SCALER_MIN) / SCALER_RANGE
        rec = self.forward(norm)
        score = float(np.mean((norm - rec) ** 2))

        if score >= CRIT_THRESHOLD:
            self.consecutive_crit += 1
            self.consecutive_warn += 1
        elif score >= WARN_THRESHOLD:
            self.consecutive_warn += 1
            self.consecutive_crit = 0
        else:
            self.consecutive_warn = 0
            self.consecutive_crit = 0

        if self.consecutive_crit >= PERSISTENCE_N:
            self.active_severity = 2
        elif self.consecutive_warn >= PERSISTENCE_N:
            self.active_severity = 1
        elif self.consecutive_warn == 0:
            self.active_severity = 0

        return {
            "score": round(score, 6),
            "severity": self.active_severity,
            "is_anomaly": self.active_severity > 0
        }

    def predict_features(self, features_13: np.ndarray) -> dict:
        raw = np.atleast_2d(features_13).astype(np.float32)
        norm = (raw - SCALER_MIN) / SCALER_RANGE

        rec = self.forward(norm)
        score = float(np.mean((norm - rec) ** 2))

        if score >= CRIT_THRESHOLD:
            self.consecutive_crit += 1
            self.consecutive_warn += 1
        elif score >= WARN_THRESHOLD:
            self.consecutive_warn += 1
            self.consecutive_crit = 0
        else:
            self.consecutive_warn = 0
            self.consecutive_crit = 0

        if self.consecutive_crit >= PERSISTENCE_N:
            self.active_severity = 2
        elif self.consecutive_warn >= PERSISTENCE_N:
            self.active_severity = 1
        elif self.consecutive_warn == 0:
            self.active_severity = 0

        return {
            "score": round(score, 6),
            "severity": self.active_severity,
            "is_anomaly": self.active_severity > 0
        }
