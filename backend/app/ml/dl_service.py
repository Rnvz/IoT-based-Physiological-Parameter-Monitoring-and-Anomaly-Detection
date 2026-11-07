import os
import json
import time
import logging
from typing import Optional, Dict, Tuple
import numpy as np

from app.config import settings
from app.ml.inference import MLInferenceResult
from app.ml.device_window import DeviceWindowManager

logger = logging.getLogger(__name__)

FEATURE_ORDER_13 = [
    "Heart_Rate", "SpO2", "Temperature",
    "Heart_Rate_Delta", "SpO2_Delta", "Temperature_Delta",
    "MA_Heart_Rate", "MA_SpO2", "MA_Temperature",
    "Var_Heart_Rate", "Var_SpO2", "Var_Temperature",
    "Rate_of_Change"
]

DEFAULT_MIN_BOUNDS = np.array([40.0, 70.0, 25.0, -40.0, -15.0, -15.0, 40.0, 70.0, 25.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_MAX_BOUNDS = np.array([160.0, 100.0, 43.0, 40.0, 15.0, 15.0, 160.0, 100.0, 43.0, 500.0, 50.0, 70.0, 1.0], dtype=np.float32)

def gelu_np(x: np.ndarray) -> np.ndarray:
    """Exact GELU using erf."""
    from scipy.special import erf
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(np.float32)

def layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """NumPy LayerNorm implementation matching PyTorch LayerNorm."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias

class NumPyAutoencoderForward:
    """Self-contained NumPy forward pass for FlexibleDenseAutoencoder from model_weights.npz."""
    def __init__(self, weights_path: str):
        data = np.load(weights_path)
        self.w = {k: data[k] for k in data.files}

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

class DeviceDLState:
    """
    Maintains temporal persistence filter per device for Deep Learning anomaly detection.
    - Escalates to severity 1 or 2 only after N consecutive samples exceed the corresponding threshold.
    - Resets immediately when finger=false or when a telemetry data gap exceeds max_gap_seconds (default >5.0s).
    """
    def __init__(self, persistence_n: int = 3, max_gap_seconds: float = 5.0):
        self.persistence_n = persistence_n
        self.max_gap_seconds = max_gap_seconds
        self.last_timestamp: Optional[float] = None
        self.consecutive_crit: int = 0
        self.consecutive_warn: int = 0
        self.active_severity: int = 0

    def reset(self):
        self.last_timestamp = None
        self.consecutive_crit = 0
        self.consecutive_warn = 0
        self.active_severity = 0

    def update(
        self,
        score: float,
        thr_warn: float,
        thr_crit: float,
        timestamp: Optional[float] = None
    ) -> Tuple[int, int]:
        """
        Updates consecutive counters and returns (raw_severity, debounced_severity).
        """
        now = timestamp if timestamp is not None else time.time()
        if self.last_timestamp is not None and (now - self.last_timestamp) > self.max_gap_seconds:
            self.reset()
        self.last_timestamp = now

        if score >= thr_crit:
            raw_sev = 2
            self.consecutive_crit += 1
            self.consecutive_warn += 1
        elif score >= thr_warn:
            raw_sev = 1
            self.consecutive_crit = 0
            self.consecutive_warn += 1
        else:
            raw_sev = 0
            self.consecutive_crit = 0
            self.consecutive_warn = 0

        if self.consecutive_crit >= self.persistence_n:
            self.active_severity = 2
        elif self.consecutive_warn >= self.persistence_n:
            self.active_severity = 1
        elif self.consecutive_warn == 0:
            self.active_severity = 0

        return raw_sev, self.active_severity

class DLInferenceResult(MLInferenceResult):
    dl_score: Optional[float] = None
    dl_severity: Optional[int] = None
    margin: Optional[float] = None
    inference_ms: Optional[float] = None

class DLService:
    """
    Deep Learning Anomaly Detection Service wrapping FlexibleDenseAutoencoder.
    Supports 13-feature physiological anomaly scoring with Bounded Physiological Scaling.
    Loads ONNX model with automatic fallback to high-precision NumPy forward pass.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or settings.model_dir
        self.ort_session = None
        self.numpy_model = None

        self.model_version: str = "dense_ae_kaggle_v1"
        self.feature_order = FEATURE_ORDER_13
        self.min_bounds = DEFAULT_MIN_BOUNDS
        self.max_bounds = DEFAULT_MAX_BOUNDS
        self.scaler_range = self.max_bounds - self.min_bounds

        # Default thresholds
        self.thr_warn: float = 0.000200
        self.thr_crit: float = 0.001000
        self.iot_stream_warn: float = 0.006000
        self.iot_stream_crit: float = 0.010400
        self.persistence_n: int = 3
        self.max_gap_seconds: float = getattr(settings, "dl_max_gap_seconds", 5.0)

        self.device_states: Dict[str, DeviceDLState] = {}
        self.window_manager = DeviceWindowManager(
            window_size=getattr(settings, "window_size", 15),
            max_gap_seconds=self.max_gap_seconds
        )
        self._last_error_log_time: float = 0.0

        self.load_artifacts()

    @property
    def model(self):
        return self.ort_session or self.numpy_model

    @property
    def is_loaded(self) -> bool:
        return self.ort_session is not None or self.numpy_model is not None

    def _resolve_path(self, filename: str) -> Optional[str]:
        candidates = [
            os.path.join(self.model_dir, filename),
            os.path.join(os.path.dirname(__file__), "../../../dl/artifacts/dl_v1", filename),
            os.path.join("dl/artifacts/dl_v1", filename),
            os.path.join("../dl/artifacts/dl_v1", filename),
            os.path.join(os.path.dirname(__file__), "../../../ml/artifacts/dl_v1", filename),
            os.path.join("ml/artifacts/dl_v1", filename),
            os.path.join("../ml/artifacts/dl_v1", filename),
        ]
        return next((p for p in candidates if os.path.exists(p)), None)

    def load_artifacts(self):
        metadata_path = self._resolve_path("metadata.json")
        if metadata_path:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.model_version = str(meta.get("model_version", "dense_ae_kaggle_v1"))
                self.feature_order = meta.get("feature_order", FEATURE_ORDER_13)

                if "scaler" in meta and "min_bounds" in meta["scaler"]:
                    self.min_bounds = np.array(meta["scaler"]["min_bounds"], dtype=np.float32)
                    self.max_bounds = np.array(meta["scaler"]["max_bounds"], dtype=np.float32)
                    self.scaler_range = np.maximum(1e-6, self.max_bounds - self.min_bounds)

                thresholds = meta.get("thresholds", {})
                self.thr_warn = float(thresholds.get("warning_p95", 0.000200))
                self.thr_crit = float(thresholds.get("critical_p99", 0.001000))
                self.iot_stream_warn = float(thresholds.get("iot_stream_warn", 0.006000))
                self.iot_stream_crit = float(thresholds.get("iot_stream_crit", 0.010400))
                self.persistence_n = int(meta.get("persistence_N", 3))

                logger.info(f"[DLService] Loaded metadata from {metadata_path} (version: {self.model_version}, thr_warn: {self.thr_warn})")
            except Exception as e:
                logger.error(f"[DLService] Error parsing metadata.json: {e}")

        onnx_path = self._resolve_path("model.onnx")
        if onnx_path:
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                logger.info(f"[DLService] Loaded ONNX model from {onnx_path}")
            except Exception as e:
                logger.info(f"[DLService] ONNX runtime skipped ({e}). Using NumPy forward pass.")
                self.ort_session = None

        weights_path = self._resolve_path("model_weights.npz")
        if weights_path:
            try:
                self.numpy_model = NumPyAutoencoderForward(weights_path)
                logger.info(f"[DLService] Loaded NumPy weights from {weights_path}")
            except Exception as e:
                logger.warning(f"[DLService] Failed loading NumPy weights from {weights_path}: {e}")

        if not self.is_loaded:
            logger.warning("[DLService] Neither ONNX nor NumPy fallback models could be loaded. DL inference unavailable.")

    def _get_state(self, device_id: str) -> DeviceDLState:
        if device_id not in self.device_states:
            self.device_states[device_id] = DeviceDLState(
                persistence_n=self.persistence_n,
                max_gap_seconds=self.max_gap_seconds
            )
        return self.device_states[device_id]

    def reset_device(self, device_id: str):
        if device_id in self.device_states:
            self.device_states[device_id].reset()
        self.window_manager.reset_device(device_id)

    def _log_error_rate_limited(self, msg: str):
        now = time.time()
        if now - self._last_error_log_time >= 60.0:
            logger.error(msg)
            self._last_error_log_time = now

    def process_reading(
        self,
        device_id: str,
        hr: Optional[float],
        spo2: Optional[float],
        temp: Optional[float] = None,
        finger: bool = False,
        timestamp: Optional[float] = None,
        features_13: Optional[np.ndarray] = None
    ) -> DLInferenceResult:
        """
        Processes vital signs for Deep Learning anomaly detection.
        Supports both direct 13-feature inference and real-time streaming (HR, SpO2, Temp).
        """
        if not finger or hr is None or spo2 is None:
            self.reset_device(device_id)
            return DLInferenceResult(
                label="NO_DATA",
                model_version=self.model_version,
                is_warmup=False
            )

        if self.ort_session is None and self.numpy_model is None:
            self._log_error_rate_limited("[DLService] Neither ONNX session nor NumPy fallback model is available.")
            return DLInferenceResult(
                label="SENSOR_ERROR",
                model_version=self.model_version,
                is_warmup=False,
                raw_severity=None,
                debounced_severity=None
            )

        t_start = time.perf_counter()
        try:
            temp_val = float(temp) if temp is not None else 36.5

            if features_13 is not None:
                raw_vec = np.atleast_2d(features_13).astype(np.float32)
            else:
                # Update device rolling window buffer
                stream_feats = self.window_manager.get_window(device_id).push(
                    float(hr), float(spo2), temp_val, timestamp=timestamp
                )
                if stream_feats is not None:
                    raw_vec = stream_feats.reshape(1, -1).astype(np.float32)
                else:
                    # Initial / single-row baseline representation
                    raw_vec = np.array([[
                        float(hr), float(spo2), temp_val,
                        0.0, 0.0, 0.0,
                        float(hr), float(spo2), temp_val,
                        0.0, 0.0, 0.0,
                        0.0
                    ]], dtype=np.float32)

            # Bounded Physiological Scaling: scale to [0, 1]
            norm = (raw_vec - self.min_bounds) / self.scaler_range

            if self.ort_session is not None:
                rec = self.ort_session.run(None, {"input": norm})[0]
            else:
                rec = self.numpy_model.forward(norm)

            score = float(np.mean((norm - rec) ** 2))
            inference_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

            state = self._get_state(device_id)
            raw_sev, debounced_sev = state.update(
                score=score,
                thr_warn=self.thr_warn,
                thr_crit=self.thr_crit,
                timestamp=timestamp
            )

            denom = max(1e-9, self.thr_crit - self.thr_warn)
            raw_margin = (score - self.thr_warn) / denom
            margin = round(float(np.clip(raw_margin, -1.0, 2.0)), 4)

            rounded_score = round(score, 6)
            label = "NORMAL" if debounced_sev == 0 else "ANOMALY"

            return DLInferenceResult(
                raw_score=rounded_score,
                raw_severity=raw_sev,
                debounced_severity=debounced_sev,
                label=label,
                model_version=self.model_version,
                is_warmup=False,
                dl_score=rounded_score,
                dl_severity=debounced_sev,
                margin=margin,
                inference_ms=inference_ms
            )

        except Exception as e:
            self._log_error_rate_limited(f"[DLService] Inference exception for device {device_id}: {e}")
            return DLInferenceResult(
                label="SENSOR_ERROR",
                model_version=self.model_version,
                is_warmup=False,
                raw_severity=None,
                debounced_severity=None
            )

_dl_service_instance: Optional[DLService] = None

def get_dl_service() -> DLService:
    global _dl_service_instance
    if _dl_service_instance is None:
        _dl_service_instance = DLService()
    return _dl_service_instance
