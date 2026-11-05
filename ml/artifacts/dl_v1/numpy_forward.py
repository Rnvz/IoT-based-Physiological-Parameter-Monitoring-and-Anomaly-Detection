"""
Self-contained NumPy forward pass for ae_v1 (v2).
Requires only NumPy and SciPy (for exact GELU erf).
"""

import numpy as np
from scipy.special import erf

FEATURE_ORDER = ["Heart_Rate", "SpO2"]
SCALER_MEAN = np.array([76.17215172123677, 97.45148557669606], dtype=np.float32)
SCALER_SCALE = np.array([9.768103813500051, 1.4986419025808964], dtype=np.float32)
WARN_THRESHOLD = 0.013571
CRIT_THRESHOLD = 0.024805
PERSISTENCE_N = 3

def gelu_np(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(np.float32)

def layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias

class EdgeAutoencoderInference:
    def __init__(self, weights_path: str = "ml/artifacts/dl_v1/model_weights.npz"):
        data = np.load(weights_path)
        self.w = {k: data[k] for k in data.files}
        self.consecutive_warn = 0
        self.consecutive_crit = 0
        self.active_severity = 0

    def predict_sample(self, values: dict) -> dict:
        """
        values: dict containing keys ['Heart_Rate', 'SpO2']
        Returns: {"score": float, "severity": int, "is_anomaly": bool}
        """
        raw = np.array([[values[c] for c in FEATURE_ORDER]], dtype=np.float32)
        norm = (raw - SCALER_MEAN) / SCALER_SCALE

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

        score = float(np.mean((norm - rec) ** 2))

        if score >= CRIT_THRESHOLD:
            self.consecutive_crit += 1
        else:
            self.consecutive_crit = 0

        if score >= WARN_THRESHOLD:
            self.consecutive_warn += 1
        else:
            self.consecutive_warn = 0

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
