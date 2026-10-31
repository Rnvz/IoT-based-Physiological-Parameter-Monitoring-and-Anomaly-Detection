"""
ml.dl.evaluate: Standardized evaluation metrics, patient-grouped bootstrap CIs,
Leave-One-Source-Out (LOSO) benchmarking, and domain-shift deployment checks.
"""

import os
import sys
import io
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
import torch

from ml.dl.models import DenseAutoencoder, DenseVAE, NumPyAutoencoderForward


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    warn_threshold: float,
    crit_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes PR-AUC, ROC-AUC, F1, Precision, Recall at warn_threshold,
    and Recall at fixed 5% FPR on normal rows.
    """
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        return {
            "pr_auc": None,
            "roc_auc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "recall_at_5pct_fpr": None
        }

    # 1. PR-AUC & ROC-AUC
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
    pr_auc_val = float(auc(rec_curve, prec_curve))

    try:
        roc_auc_val = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc_val = None

    # 2. Metrics at fixed warning threshold
    preds = (scores >= warn_threshold).astype(int)
    f1_val = float(f1_score(y_true, preds, zero_division=0))
    prec_val = float(precision_score(y_true, preds, zero_division=0))
    rec_val = float(recall_score(y_true, preds, zero_division=0))

    # 3. Recall at fixed 5% FPR on normal rows (y_true == 0)
    normal_scores = scores[y_true == 0]
    if len(normal_scores) > 0:
        p95_norm_th = float(np.percentile(normal_scores, 95.0))
        abnormal_scores = scores[y_true == 1]
        rec_at_5pct = float(np.mean(abnormal_scores >= p95_norm_th)) if len(abnormal_scores) > 0 else None
    else:
        rec_at_5pct = None

    return {
        "pr_auc": pr_auc_val,
        "roc_auc": roc_auc_val,
        "f1": f1_val,
        "precision": prec_val,
        "recall": rec_val,
        "recall_at_5pct_fpr": rec_at_5pct
    }


def patient_group_bootstrap(
    df: pd.DataFrame,
    scores: np.ndarray,
    warn_threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Calculates 95% Confidence Intervals for PR-AUC, ROC-AUC, F1, and Recall
    by resampling with replacement at the PATIENT level.
    For single-patient datasets (engrarri21), resamples contiguous 50-row blocks.
    Returns Dict mapping metric name to (mean, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    patients = df["Patient_ID"].unique()
    n_pts = len(patients)

    y_true = df["OUTPUT"].values
    n_samples = len(df)

    boot_pr_aucs = []
    boot_roc_aucs = []
    boot_f1s = []
    boot_rec_5pcts = []

    if n_pts == n_samples:
        # Cross-sectional where every patient is exactly 1 row
        for _ in range(n_bootstrap):
            sampled_idx = rng.integers(0, n_samples, size=n_samples)
            sub_y = y_true[sampled_idx]
            sub_sc = scores[sampled_idx]

            if len(np.unique(sub_y)) < 2:
                continue

            m = compute_metrics(sub_y, sub_sc, warn_threshold)
            if m["pr_auc"] is not None:
                boot_pr_aucs.append(m["pr_auc"])
            if m["roc_auc"] is not None:
                boot_roc_aucs.append(m["roc_auc"])
            if m["f1"] is not None:
                boot_f1s.append(m["f1"])
            if m["recall_at_5pct_fpr"] is not None:
                boot_rec_5pcts.append(m["recall_at_5pct_fpr"])

    elif n_pts > 1:
        # Mixed pooled dataset with some multi-row patients
        pt_idx_list = [np.array(idxs, dtype=np.int64) for idxs in df.groupby("Patient_ID").indices.values()]
        for _ in range(n_bootstrap):
            chosen = rng.integers(0, len(pt_idx_list), size=len(pt_idx_list))
            sampled_idx = np.concatenate([pt_idx_list[c] for c in chosen])

            sub_y = y_true[sampled_idx]
            sub_sc = scores[sampled_idx]

            if len(np.unique(sub_y)) < 2:
                continue

            m = compute_metrics(sub_y, sub_sc, warn_threshold)
            if m["pr_auc"] is not None:
                boot_pr_aucs.append(m["pr_auc"])
            if m["roc_auc"] is not None:
                boot_roc_aucs.append(m["roc_auc"])
            if m["f1"] is not None:
                boot_f1s.append(m["f1"])
            if m["recall_at_5pct_fpr"] is not None:
                boot_rec_5pcts.append(m["recall_at_5pct_fpr"])

    else:
        # Block bootstrap for single patient stream (50-row blocks)
        block_size = 50
        n_blocks = max(1, n_samples // block_size)

        for _ in range(n_bootstrap):
            block_starts = rng.integers(0, max(1, n_samples - block_size), size=n_blocks)
            sampled_idx = np.concatenate([np.arange(b, b + block_size) for b in block_starts])
            sampled_idx = sampled_idx[sampled_idx < n_samples]

            sub_y = y_true[sampled_idx]
            sub_sc = scores[sampled_idx]

            if len(np.unique(sub_y)) < 2:
                continue

            m = compute_metrics(sub_y, sub_sc, warn_threshold)
            if m["pr_auc"] is not None:
                boot_pr_aucs.append(m["pr_auc"])
            if m["roc_auc"] is not None:
                boot_roc_aucs.append(m["roc_auc"])
            if m["f1"] is not None:
                boot_f1s.append(m["f1"])
            if m["recall_at_5pct_fpr"] is not None:
                boot_rec_5pcts.append(m["recall_at_5pct_fpr"])

    def get_ci(vals: List[float]) -> Tuple[float, float, float]:
        if len(vals) < 10:
            return (0.0, 0.0, 0.0)
        return (
            float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5))
        )

    return {
        "pr_auc": get_ci(boot_pr_aucs),
        "roc_auc": get_ci(boot_roc_aucs),
        "f1": get_ci(boot_f1s),
        "recall_at_5pct_fpr": get_ci(boot_rec_5pcts)
    }


def compute_persistence_events(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    N: int = 3
) -> Dict[str, Any]:
    """
    Computes false alarm persistence events per 1000 normal rows on time-ordered data.
    An alarm event triggers when score >= threshold for N consecutive samples.
    """
    # Evaluate only on normal segments (y_true == 0)
    normal_mask = (y_true == 0)
    normal_scores = scores[normal_mask]
    n_normal = len(normal_scores)

    if n_normal == 0:
        return {"events_count": 0, "events_per_1000": 0.0}

    consecutive = 0
    in_alarm_state = False
    alarm_events = 0

    for s in normal_scores:
        if s >= threshold:
            consecutive += 1
            if consecutive >= N and not in_alarm_state:
                alarm_events += 1
                in_alarm_state = True
        else:
            consecutive = 0
            in_alarm_state = False

    events_per_1000 = (alarm_events / max(1, n_normal)) * 1000.0

    return {
        "total_normal_rows": n_normal,
        "alarm_events_count": alarm_events,
        "events_per_1000_normal_rows": round(events_per_1000, 2)
    }


def benchmark_model_speed_and_size(model_obj: Any, sample_x: np.ndarray) -> Tuple[float, int]:
    """Measures inference time per sample (microseconds) and in-memory serialized size (bytes)."""
    # Timing
    times = []
    # Warmup
    for _ in range(20):
        if hasattr(model_obj, "score"):
            _ = model_obj.score(sample_x[:1])
        elif isinstance(model_obj, torch.nn.Module):
            with torch.no_grad():
                _ = model_obj.compute_anomaly_score(torch.tensor(sample_x[:1], dtype=torch.float32))

    for _ in range(200):
        t0 = time.perf_counter()
        if hasattr(model_obj, "score"):
            _ = model_obj.score(sample_x[:1])
        elif isinstance(model_obj, torch.nn.Module):
            with torch.no_grad():
                _ = model_obj.compute_anomaly_score(torch.tensor(sample_x[:1], dtype=torch.float32))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    mean_us = float(np.mean(times))

    # Size
    size_bytes = 0
    if isinstance(model_obj, torch.nn.Module):
        buf = io.BytesIO()
        torch.save(model_obj.state_dict(), buf)
        size_bytes = buf.tell()
    elif hasattr(model_obj, "model") and model_obj.model is not None:
        import joblib
        buf = io.BytesIO()
        joblib.dump(model_obj.model, buf)
        size_bytes = buf.tell()

    return round(mean_us, 1), size_bytes


# ==============================================================================
# DOMAIN SHIFT / DEPLOYMENT SANITY (PART 6)
# ==============================================================================

def generate_synthetic_resting_stream(
    temp_target: float,
    n_samples: int = 400,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic resting stream:
    HR 72 +- 3 BPM, SpO2 98 +- 0.5%, Temp fixed at temp_target +- 0.3 °C (400 samples at 1 Hz).
    """
    rng = np.random.default_rng(seed)
    hr = rng.normal(72.0, 3.0, size=n_samples)
    spo2 = np.clip(rng.normal(98.0, 0.5, size=n_samples), 85.0, 100.0)
    temp = rng.normal(temp_target, 0.3, size=n_samples)

    return pd.DataFrame({
        "Heart_Rate": np.round(hr, 2),
        "SpO2": np.round(spo2, 2),
        "Temperature": np.round(temp, 2),
        "OUTPUT": 0,
        "Patient_ID": f"synthetic_resting_{temp_target}C"
    })


def evaluate_domain_shift_stream(
    score_func: Any,
    scaler: Any,
    feature_cols: List[str],
    warn_th: float,
    crit_th: float,
    persistence_n: int = 3,
    temp_targets: List[float] = [31.0, 33.0, 35.0, 36.5]
) -> Dict[str, Any]:
    """
    Runs synthetic resting streams at various temperatures through persistence filter.
    Returns share of severity >= 1 and severity == 2.
    """
    results = {}

    for t_target in temp_targets:
        stream_df = generate_synthetic_resting_stream(temp_target=t_target, n_samples=400, seed=42)
        X = scaler.transform(stream_df[feature_cols].values)

        scores = score_func(X)

        # Apply persistence filter N
        consec_warn = 0
        consec_crit = 0
        sev_active = 0

        warn_count = 0
        crit_count = 0

        for s in scores:
            if s >= crit_th:
                consec_crit += 1
            else:
                consec_crit = 0

            if s >= warn_th:
                consec_warn += 1
            else:
                consec_warn = 0

            # State resolution
            if consec_crit >= persistence_n:
                sev_active = 2
            elif consec_warn >= persistence_n:
                sev_active = 1
            elif consec_warn == 0:
                sev_active = 0

            if sev_active >= 1:
                warn_count += 1
            if sev_active == 2:
                crit_count += 1

        total = len(stream_df)
        results[f"{t_target}C"] = {
            "temperature_target": t_target,
            "total_samples": total,
            "warning_share_pct": round((warn_count / total) * 100.0, 2),
            "critical_share_pct": round((crit_count / total) * 100.0, 2)
        }

    return results
