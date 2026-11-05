"""
ml_eval.metrics: Standardized evaluation metrics computation.

Computes:
1. PR-AUC (Precision-Recall Area Under Curve).
2. ROC-AUC (Receiver Operating Characteristic Area Under Curve).
3. F1-Score, Precision, and Recall at the calibrated operating threshold.
4. False Alarms per Hour on benign operational test sessions (REST / TYPING).
5. False Alarms per Hour at a fixed target recall (e.g. Recall >= 90%).
6. Detection Latency (samples and seconds from anomaly onset to first alarm).
7. Inference latency per row (mean, p95, p99 in microseconds).
8. Serialized model artifact size on disk / memory.

This script is the single authoritative scoring implementation. Future deep learning
models (e.g. LSTM-AE, 1D-CNN) must be evaluated with this exact module.
"""

import io
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from ml.eval.sessions import parse_timestamp

def compute_pr_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Computes PR-AUC and ROC-AUC safely, handling edge cases where only one class is present."""
    if len(np.unique(y_true)) < 2:
        return None, None

    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc_val = float(auc(rec, prec))
    except Exception:
        pr_auc_val = None

    try:
        roc_auc_val = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc_val = None

    return pr_auc_val, roc_auc_val

def compute_false_alarms(
    benign_df: pd.DataFrame,
    benign_scores: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Computes false alarm rate on clean benign sessions (REST / TYPING).
    """
    if benign_df.empty or len(benign_scores) == 0:
        return {
            "duration_hours": 0.0,
            "false_alarms_count": 0,
            "false_alarms_per_hour": 0.0
        }

    dts = benign_df["dt"].tolist() if "dt" in benign_df.columns else [parse_timestamp(x) for x in benign_df["ts"]]
    duration_s = max(1.0, (dts[-1] - dts[0]).total_seconds()) if len(dts) > 1 else len(benign_df) * 1.0
    duration_h = duration_s / 3600.0

    preds = (benign_scores >= threshold).astype(int)
    fa_count = int(np.sum(preds == 1))
    fa_per_hour = fa_count / max(duration_h, 1e-4)

    return {
        "duration_hours": round(duration_h, 3),
        "false_alarms_count": fa_count,
        "false_alarms_per_hour": round(fa_per_hour, 2)
    }

def compute_detection_latency(
    df: pd.DataFrame,
    preds: np.ndarray,
    target_anomalies: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes latency (samples & seconds) from the onset of a synthetic anomaly to detection.
    """
    if "anomaly_type" not in df.columns or "is_synthetic" not in df.columns:
        return {"mean_latency_samples": None, "mean_latency_seconds": None, "detected_ratio": None}

    if target_anomalies is None:
        target_anomalies = ["HR_SPIKE", "SPO2_DROP", "FLAT_SIGNAL"]

    latencies_samples = []
    latencies_seconds = []
    total_episodes = 0
    detected_episodes = 0

    dts = df["dt"].tolist() if "dt" in df.columns else [parse_timestamp(x) for x in df["ts"]]

    is_anom = (df["is_synthetic"] == True).values
    types = df["anomaly_type"].values

    i = 0
    n = len(df)
    while i < n:
        if is_anom[i] and types[i] in target_anomalies:
            a_type = types[i]
            start_idx = i
            while i < n and is_anom[i] and types[i] == a_type:
                i += 1
            end_idx = i

            total_episodes += 1
            episode_preds = preds[start_idx:end_idx]

            hits = np.where(episode_preds == 1)[0]
            if len(hits) > 0:
                first_hit_idx = hits[0]
                detected_episodes += 1
                lat_samples = int(first_hit_idx)
                lat_sec = (dts[start_idx + first_hit_idx] - dts[start_idx]).total_seconds()
                latencies_samples.append(lat_samples)
                latencies_seconds.append(max(0.0, lat_sec))
        else:
            i += 1

    mean_samples = float(np.mean(latencies_samples)) if latencies_samples else None
    mean_seconds = float(np.mean(latencies_seconds)) if latencies_seconds else None
    ratio = detected_episodes / max(1, total_episodes) if total_episodes > 0 else None

    return {
        "total_episodes": total_episodes,
        "detected_episodes": detected_episodes,
        "detected_ratio": round(ratio, 4) if ratio is not None else None,
        "mean_latency_samples": round(mean_samples, 2) if mean_samples is not None else None,
        "mean_latency_seconds": round(mean_seconds, 2) if mean_seconds is not None else None
    }

def benchmark_inference_speed_and_size(model_wrapper: Any, sample_df: pd.DataFrame) -> Tuple[Dict[str, float], Optional[int]]:
    """
    Benchmarks single-row inference speed and serialized model size.
    """
    size_bytes = None
    if model_wrapper.model is not None:
        buf = io.BytesIO()
        joblib.dump(model_wrapper.model, buf)
        size_bytes = buf.tell()

    if sample_df.empty:
        return {"mean_us": 0.0, "p95_us": 0.0}, size_bytes

    single_row = sample_df.iloc[[0]]
    times_us = []
    for _ in range(20):
        _ = model_wrapper.score(single_row)

    for _ in range(300):
        t0 = time.perf_counter()
        _ = model_wrapper.score(single_row)
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    return {
        "mean_us": round(float(np.mean(times_us)), 2),
        "p95_us": round(float(np.percentile(times_us, 95)), 2),
    }, size_bytes

def evaluate_model(
    model_wrapper: Any,
    real_test_df: pd.DataFrame,
    injected_test_df: pd.DataFrame,
    target_recall: float = 0.90
) -> Dict[str, Any]:
    """
    Executes full standardized evaluation protocol for a single model:
    - Discriminative power on synthetic injected test set (PR-AUC, ROC-AUC, F1).
    - Operational reliability on clean physical test set (False alarms / hour).
    - Detection latency on injected failure onset.
    - Computational footprint (inference time, artifact size).
    """
    injected_scores = model_wrapper.score(injected_test_df)
    injected_preds = model_wrapper.predict(injected_test_df)
    y_true = injected_test_df["y_true"].values.astype(int)

    pr_auc_val, roc_auc_val = compute_pr_roc_auc(y_true, injected_scores)
    f1 = float(f1_score(y_true, injected_preds, zero_division=0))
    prec = float(precision_score(y_true, injected_preds, zero_division=0))
    rec = float(recall_score(y_true, injected_preds, zero_division=0))
    cm = confusion_matrix(y_true, injected_preds).tolist()

    benign_mask = real_test_df["scenario"].isin(["REST", "TYPING"])
    benign_df = real_test_df[benign_mask] if benign_mask.sum() > 0 else real_test_df
    benign_scores = model_wrapper.score(benign_df)

    fa_stats = compute_false_alarms(
        benign_df=benign_df,
        benign_scores=benign_scores,
        threshold=model_wrapper.threshold
    )

    if len(np.unique(y_true)) >= 2:
        p_curve, r_curve, th_curve = precision_recall_curve(y_true, injected_scores)
        eligible = np.where(r_curve[:-1] >= target_recall)[0]
        if len(eligible) > 0:
            chosen_th = float(th_curve[eligible[-1]])
            fa_at_recall = compute_false_alarms(benign_df, benign_scores, threshold=chosen_th)
            fa_at_target_rec = fa_at_recall["false_alarms_per_hour"]
        else:
            fa_at_target_rec = None
    else:
        fa_at_target_rec = None

    latency_stats = compute_detection_latency(injected_test_df, injected_preds)

    speed_stats, size_bytes = benchmark_inference_speed_and_size(model_wrapper, real_test_df)

    return {
        "model_name": model_wrapper.name,
        "operating_threshold": round(float(model_wrapper.threshold), 5),
        "target_val_fpr": model_wrapper.target_fpr,
        "metrics_synthetic": {
            "pr_auc": round(pr_auc_val, 4) if pr_auc_val is not None else None,
            "roc_auc": round(roc_auc_val, 4) if roc_auc_val is not None else None,
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "confusion_matrix": cm
        },
        "metrics_benign_real": {
            "duration_hours": fa_stats["duration_hours"],
            "false_alarms_count": fa_stats["false_alarms_count"],
            "false_alarms_per_hour": fa_stats["false_alarms_per_hour"],
            "false_alarms_per_hour_at_90_recall": fa_at_target_rec
        },
        "detection_latency": latency_stats,
        "computational_footprint": {
            "inference_time_us": speed_stats,
            "serialized_size_bytes": size_bytes,
            "serialized_size_kb": round(size_bytes / 1024.0, 2) if size_bytes else None
        }
    }
