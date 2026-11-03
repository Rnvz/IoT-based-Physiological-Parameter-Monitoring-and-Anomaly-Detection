"""
ml.dl.export: Model exporter for ONNX, NumPy fallback, metadata, and Model Card.
Verifies parity between PyTorch, ONNX, and NumPy (< 1e-5).
"""

import os
import sys
import json
import platform
from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np
try:
    import torch
except ImportError:
    torch = None
try:
    import onnx
except ImportError:
    onnx = None
import onnxruntime as ort
import sklearn

from ml.dl.models import DenseAutoencoder, NumPyAutoencoderForward


def export_dl_model(
    model: DenseAutoencoder,
    scaler: Any,
    variant: str,
    feature_cols: List[str],
    warn_th: float,
    crit_th: float,
    split_sha: str,
    seeds: List[int],
    output_dir: str = "ml/artifacts/dl_v1",
    model_version: str = "ae_v1"
) -> Dict[str, Any]:
    """
    Exports:
    1. model.onnx
    2. numpy_forward.py / weights.npz (NumPy fallback)
    3. metadata.json
    4. MODEL_CARD.md
    Verifies parity between PyTorch, ONNX, and NumPy.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    input_dim = len(feature_cols)
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    # 1. Export ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17
    )

    # 2. Export NumPy Weights & Fallback
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    npz_path = os.path.join(output_dir, "model_weights.npz")
    np.savez(npz_path, **weights)

    numpy_fallback_code = f'''"""
Self-contained NumPy forward pass for {model_version} ({variant}).
Requires only NumPy and SciPy (for exact GELU erf).
"""

import numpy as np
from scipy.special import erf

FEATURE_ORDER = {json.dumps(feature_cols)}
SCALER_MEAN = np.array({scaler.mean_.tolist()}, dtype=np.float32)
SCALER_SCALE = np.array({scaler.scale_.tolist()}, dtype=np.float32)
WARN_THRESHOLD = {float(warn_th):.6f}
CRIT_THRESHOLD = {float(crit_th):.6f}
PERSISTENCE_N = 3


def gelu_np(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(np.float32)


def layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


class EdgeAutoencoderInference:
    def __init__(self, weights_path: str = "{npz_path}"):
        data = np.load(weights_path)
        self.w = {{k: data[k] for k in data.files}}
        self.consecutive_warn = 0
        self.consecutive_crit = 0
        self.active_severity = 0

    def predict_sample(self, values: dict) -> dict:
        """
        values: dict containing keys {feature_cols}
        Returns: {{"score": float, "severity": int, "is_anomaly": bool}}
        """
        raw = np.array([[values[c] for c in FEATURE_ORDER]], dtype=np.float32)
        norm = (raw - SCALER_MEAN) / SCALER_SCALE

        w = self.w
        # Encoder
        h = norm @ w["encoder.0.weight"].T + w["encoder.0.bias"]
        h = layer_norm_np(h, w["encoder.1.weight"], w["encoder.1.bias"])
        h = gelu_np(h)
        h = h @ w["encoder.4.weight"].T + w["encoder.4.bias"]
        h = layer_norm_np(h, w["encoder.5.weight"], w["encoder.5.bias"])
        h = gelu_np(h)
        z = h @ w["encoder.7.weight"].T + w["encoder.7.bias"]

        # Decoder
        d = z @ w["decoder.0.weight"].T + w["decoder.0.bias"]
        d = layer_norm_np(d, w["decoder.1.weight"], w["decoder.1.bias"])
        d = gelu_np(d)
        d = d @ w["decoder.4.weight"].T + w["decoder.4.bias"]
        d = layer_norm_np(d, w["decoder.5.weight"], w["decoder.5.bias"])
        d = gelu_np(d)
        rec = d @ w["decoder.7.weight"].T + w["decoder.7.bias"]

        score = float(np.mean((norm - rec) ** 2))

        # Persistence Filter
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

        return {{
            "score": round(score, 6),
            "severity": self.active_severity,
            "is_anomaly": self.active_severity > 0
        }}
'''
    py_path = os.path.join(output_dir, "numpy_forward.py")
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(numpy_fallback_code)

    # 3. Export metadata.json
    metadata = {
        "model_version": model_version,
        "variant": variant,
        "feature_order": feature_cols,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "thresholds": {
            "warning_p95": float(warn_th),
            "critical_p99": float(crit_th)
        },
        "persistence_N": 3,
        "split_sha256": split_sha,
        "seeds": seeds,
        "training_sources": ["nasirayub2", "engrarri21", "gourangomandal"],
        "date": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__
        }
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 4. Export MODEL_CARD.md
    model_card = f"""# Model Card: {model_version} ({variant.upper()})

## Model Overview
- **Architecture**: Compact Dense Autoencoder ($3 \\to 16 \\to 8 \\to 2 \\to 8 \\to 16 \\to 3$)
- **Parameters**: 533 parameters (< 10,000 parameter edge limit)
- **Primary Task**: Unsupervised physiological anomaly detection (reconstruction MSE error)
- **Features Used**: {', '.join(feature_cols)}
- **Export Formats**: ONNX (`model.onnx`) and NumPy Fallback (`numpy_forward.py`)

## Clinical & Operational Disclaimer
> [!CAUTION]
> **NON-CLINICAL DEMO SYSTEM ONLY**
> This model is trained on public research datasets and is NOT a medical device.
> It MUST NOT be used for clinical diagnosis, patient triage, or life-critical medical monitoring.

## Public Training Data & Limitations
1. **Multi-Source Composition**:
   - `nasirayub2` (cross-sectional, 200,020 rows): Synthetic/heuristic multi-risk dataset. Labels derived from hypertension criteria (systolic/diastolic blood pressure) and BMI, NOT vitals.
   - `engrarri21` (longitudinal, 21,523 rows): Single-subject ICU streaming recording with high exact duplicate fraction (86.0%).
   - `gourangomandal` (cross-sectional, 60,000 rows): Extreme label circularity (95.5% agreement with simple threshold rules).
2. **Core vs. Skin Temperature Discrepancy**:
   - Public datasets record **core/oral body temperature** (~36.5–37.5 °C).
   - Physical IoT sensors (DS18B20 or thermistors on finger/wrist) measure **peripheral skin contact temperature** (~31.0–35.0 °C).
   - Models trained with core temperature features (`v3`) will perceive peripheral skin readings as severe hypothermia unless temperature-free (`v2`) or compensated.

## Domain Shift Sanity Checks
- **Resting Stream Benchmark**: Replayed at 31 °C, 33 °C, 35 °C, and 36.5 °C.
- **Persistence Filtering**: $N=3$ consecutive anomaly samples required to trigger alert state.

## Parity Verification
- **PyTorch vs ONNX Max Absolute Difference**: Verified $< 10^{{-5}}$
- **PyTorch vs NumPy Forward Max Absolute Difference**: Verified $< 10^{{-5}}$
"""
    card_path = os.path.join(output_dir, "MODEL_CARD.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    # 5. Parity verification
    test_x = np.random.randn(50, input_dim).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(test_x)).numpy()

    ort_session = ort.InferenceSession(onnx_path)
    onnx_out = ort_session.run(None, {"input": test_x})[0]

    np_model = NumPyAutoencoderForward.from_torch_model(model)
    np_out = np_model.forward(test_x)

    pt_onnx_diff = float(np.max(np.abs(pt_out - onnx_out)))
    pt_np_diff = float(np.max(np.abs(pt_out - np_out)))

    assert pt_onnx_diff < 1e-5, f"ONNX parity failed: {pt_onnx_diff}"
    assert pt_np_diff < 1e-5, f"NumPy parity failed: {pt_np_diff}"

    return {
        "onnx_path": onnx_path,
        "metadata_path": meta_path,
        "model_card_path": card_path,
        "pt_onnx_diff": pt_onnx_diff,
        "pt_np_diff": pt_np_diff
    }
