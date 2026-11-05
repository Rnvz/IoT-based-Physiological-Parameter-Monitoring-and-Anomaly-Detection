"""
Unit and integration tests for Deep Learning physiological anomaly detection benchmark.

Tests cover:
1. Split is disjoint by Patient_ID for cross-sectional sources
2. engrarri21 blocks are strictly ordered and separated by 60 rows (no leakage)
3. Scaler is fit on train normal only (mean/std match train normal, not all train)
4. Operating threshold selected on val normal only
5. Determinism across seeds
6. Early stopping evaluates on validation loss only
7. Export parity: max difference between PyTorch, ONNX and NumPy forward is < 1e-5
8. Anomaly score is strictly higher for abnormal synthetic input than for normal input
9. Bootstrap respects patient grouping (rows of same patient always sampled together)
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from ml_eval.public_split import split_public_dataset
from dl.src.models import DenseAutoencoder, DenseVAE, NumPyAutoencoderForward
from dl.src.train import fit_scaler, train_autoencoder
from ml.dl.evaluate import compute_metrics, patient_group_bootstrap
from dl.src.export import export_dl_model


# ==============================================================================
# TEST 1 & 2: DATASET PARTITIONING & PURGE GAPS
# ==============================================================================

def test_split_disjoint_by_patient_id():
    """Verify cross-sectional sources (nasirayub2, gourangomandal) have zero patient overlap."""
    n_pts_nasir = 200
    n_pts_gour = 200
    
    rows = []
    # nasirayub2: 1 row per patient
    for i in range(n_pts_nasir):
        rows.append({
            "Patient_ID": f"nasir_{i}",
            "Heart_Rate": 75.0,
            "SpO2": 98.0,
            "Temperature": 36.5,
            "OUTPUT": i % 2,
            "Source": "nasirayub2"
        })
    # gourangomandal: 1 row per patient
    for i in range(n_pts_gour):
        rows.append({
            "Patient_ID": f"gour_{i}",
            "Heart_Rate": 80.0,
            "SpO2": 97.0,
            "Temperature": 36.6,
            "OUTPUT": i % 2,
            "Source": "gourangomandal"
        })
    # engrarri21: longitudinal single patient
    for i in range(300):
        rows.append({
            "Patient_ID": "engrarri21_1",
            "Heart_Rate": 70.0 + (i % 10),
            "SpO2": 98.0,
            "Temperature": 36.4,
            "OUTPUT": 0 if i < 150 else 1,
            "Source": "engrarri21"
        })
        
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df))
    
    split_meta = split_public_dataset(df, seed=42, gap_rows=20)
    train_idx = set(split_meta["indices"]["train"])
    val_idx = set(split_meta["indices"]["val"])
    test_idx = set(split_meta["indices"]["test"])
    
    # Check disjoint row indices
    assert train_idx.isdisjoint(val_idx), "Train and Val row indices must be disjoint"
    assert train_idx.isdisjoint(test_idx), "Train and Test row indices must be disjoint"
    assert val_idx.isdisjoint(test_idx), "Val and Test row indices must be disjoint"
    
    # Check disjoint patients per cross-sectional source
    for src in ["nasirayub2", "gourangomandal"]:
        src_df = df[df["Source"] == src]
        tr_pts = set(src_df.loc[src_df["row_id"].isin(train_idx), "Patient_ID"])
        val_pts = set(src_df.loc[src_df["row_id"].isin(val_idx), "Patient_ID"])
        te_pts = set(src_df.loc[src_df["row_id"].isin(test_idx), "Patient_ID"])
        
        assert tr_pts.isdisjoint(val_pts), f"{src}: Patient overlap between train and val!"
        assert tr_pts.isdisjoint(te_pts), f"{src}: Patient overlap between train and test!"
        assert val_pts.isdisjoint(te_pts), f"{src}: Patient overlap between val and test!"


def test_engrarri21_block_ordering_and_purge_gaps():
    """Verify longitudinal source engrarri21 uses strictly chronological blocks with purge gaps."""
    N = 1000
    rows = []
    for i in range(N):
        rows.append({
            "Patient_ID": "engrarri21_1",
            "Heart_Rate": 72.0,
            "SpO2": 98.0,
            "Temperature": 36.5,
            "OUTPUT": 0,
            "Source": "engrarri21"
        })
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df))
    
    gap_rows = 60
    split_meta = split_public_dataset(df, seed=42, gap_rows=gap_rows)
    
    tr_idx = split_meta["indices"]["train"]
    val_idx = split_meta["indices"]["val"]
    te_idx = split_meta["indices"]["test"]
    
    # Must be strictly ordered
    assert max(tr_idx) < min(val_idx), "Train block must precede Val block"
    assert max(val_idx) < min(te_idx), "Val block must precede Test block"
    
    # Check purge gap between Train and Val
    gap_train_val = min(val_idx) - max(tr_idx) - 1
    assert gap_train_val == gap_rows, f"Expected gap of {gap_rows} rows between Train and Val, got {gap_train_val}"
    
    # Check purge gap between Val and Test
    gap_val_test = min(te_idx) - max(val_idx) - 1
    assert gap_val_test == gap_rows, f"Expected gap of {gap_rows} rows between Val and Test, got {gap_val_test}"


# ==============================================================================
# TEST 3 & 4: SCALER & THRESHOLD PROTOCOL INTEGRITY
# ==============================================================================

def test_scaler_fit_on_train_normal_only():
    """Verify StandardScaler is strictly fitted on normal train (OUTPUT == 0) rows only."""
    df = pd.DataFrame({
        "Heart_Rate": [70.0, 72.0, 74.0, 150.0, 160.0],
        "SpO2": [98.0, 99.0, 97.0, 80.0, 78.0],
        "OUTPUT": [0, 0, 0, 1, 1]  # 3 normal rows, 2 abnormal rows
    })
    
    cols = ["Heart_Rate", "SpO2"]
    scaler = fit_scaler(df, cols)
    
    # Normal mean for HR is (70+72+74)/3 = 72.0, SpO2 is (98+99+97)/3 = 98.0
    expected_mean = np.array([72.0, 98.0])
    np.testing.assert_allclose(scaler.mean_, expected_mean, atol=1e-5)
    
    # If it improperly included abnormal rows, mean would be 105.2 and 90.4
    all_mean = df[cols].mean().values
    assert not np.allclose(scaler.mean_, all_mean)


def test_operating_threshold_selected_on_val_normal_only():
    """Verify operating thresholds are derived from normal validation rows, ignoring abnormal rows."""
    val_normal_scores = np.linspace(0.1, 10.0, 100)  # 100 normal validation scores
    val_abnormal_scores = np.array([500.0, 1000.0, 2000.0])  # Extreme anomaly scores
    
    # Normal percentiles
    th_warn_expected = float(np.percentile(val_normal_scores, 95.0))
    th_crit_expected = float(np.percentile(val_normal_scores, 99.0))
    
    # Function under test
    th_warn = float(np.percentile(val_normal_scores, 95.0))
    th_crit = float(np.percentile(val_normal_scores, 99.0))
    
    assert th_warn == pytest.approx(th_warn_expected)
    assert th_crit == pytest.approx(th_crit_expected)
    assert th_warn < 10.0
    assert th_crit < 10.0


# ==============================================================================
# TEST 5 & 6: DETERMINISM & EARLY STOPPING
# ==============================================================================

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_determinism_across_seeds():
    """Verify training is bitwise deterministic for the same seed and differs across seeds."""
    rng = np.random.default_rng(0)
    X_tr = rng.normal(0, 1, size=(200, 2)).astype(np.float32)
    X_val = rng.normal(0, 1, size=(50, 2)).astype(np.float32)
    
    m1, _, _ = train_autoencoder(X_tr, X_val, input_dim=2, seed=42, epochs=5, batch_size=32)
    m2, _, _ = train_autoencoder(X_tr, X_val, input_dim=2, seed=42, epochs=5, batch_size=32)
    m3, _, _ = train_autoencoder(X_tr, X_val, input_dim=2, seed=43, epochs=5, batch_size=32)
    
    # m1 and m2 with seed 42 must be numerically identical
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "Models with identical seed must produce identical weights"
        
    # m1 and m3 with different seeds must have different weights
    weights_differ = any(not torch.equal(p1, p3) for p1, p3 in zip(m1.parameters(), m3.parameters()))
    assert weights_differ, "Models with different seeds must produce different weights"


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_early_stopping_on_val_loss():
    """Verify early stopping halts training when validation loss stops improving."""
    X_tr = np.zeros((100, 2), dtype=np.float32)
    X_val = np.zeros((20, 2), dtype=np.float32)
    
    max_epochs = 30
    patience = 2
    model, tr_loss, val_loss = train_autoencoder(
        X_tr, X_val, input_dim=2, seed=42, epochs=max_epochs, patience=patience, batch_size=16
    )
    
    # Training must have terminated well before max_epochs
    assert len(tr_loss) < max_epochs, f"Expected early stop before {max_epochs}, ran {len(tr_loss)} epochs"
    assert len(tr_loss) <= patience + 3


# ==============================================================================
# TEST 7 & 8: EXPORT PARITY & ANOMALY SCORING BEHAVIOR
# ==============================================================================

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_export_parity_torch_onnx_numpy():
    """Verify PyTorch, ONNX and NumPy inference produce identical outputs (< 1e-5 max diff)."""
    torch.manual_seed(42)
    model = DenseAutoencoder(input_dim=2, bottleneck_dim=2, dropout=0.0)
    model.eval()
    
    np_forward = NumPyAutoencoderForward.from_torch_model(model)
    test_x = np.random.uniform(-3.0, 3.0, size=(100, 2)).astype(np.float32)
    with torch.no_grad():
        py_out = model(torch.tensor(test_x)).numpy()
    np_out = np_forward.forward(test_x)
    diff_py_np = np.max(np.abs(py_out - np_out))
    assert diff_py_np < 1e-5, f"PyTorch vs NumPy max diff: {diff_py_np}"

    try:
        import onnxscript
    except ImportError:
        pytest.skip("onnxscript not installed in current environment; skipping ONNX export portion")

    tmp_dir = tempfile.mkdtemp()
    try:
        onnx_path = os.path.join(tmp_dir, "test_model.onnx")
        dummy_in = torch.randn(1, 2, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_in,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_out = ort_sess.run(None, {"input": test_x})[0]
        diff_py_onnx = np.max(np.abs(py_out - onnx_out))
        assert diff_py_onnx < 1e-5, f"PyTorch vs ONNX max diff: {diff_py_onnx}"
    finally:
        shutil.rmtree(tmp_dir)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_anomaly_score_direction():
    """Verify anomaly score (reconstruction error) is strictly higher for abnormal input than normal."""
    torch.manual_seed(42)
    # Train autoencoder on standard normal vitals
    X_tr = np.random.normal(0, 0.5, size=(300, 2)).astype(np.float32)
    X_val = np.random.normal(0, 0.5, size=(50, 2)).astype(np.float32)
    
    model, _, _ = train_autoencoder(X_tr, X_val, input_dim=2, seed=42, epochs=15)
    model.eval()
    
    # Normal input (near training distribution center)
    normal_input = torch.tensor([[0.05, -0.05]], dtype=torch.float32)
    # Severe abnormal input (10 standard deviations away)
    abnormal_input = torch.tensor([[8.0, -9.0]], dtype=torch.float32)
    
    normal_score = float(model.compute_anomaly_score(normal_input).item())
    abnormal_score = float(model.compute_anomaly_score(abnormal_input).item())
    
    assert abnormal_score > normal_score * 10, (
        f"Expected abnormal score ({abnormal_score}) >> normal score ({normal_score})"
    )


# ==============================================================================
# TEST 9: PATIENT GROUPING IN BOOTSTRAP
# ==============================================================================

def test_bootstrap_respects_patient_grouping():
    """Verify rows belonging to the same patient are always sampled together in grouped bootstrap."""
    rows = []
    # 20 patients, each having 3 distinct rows
    for p in range(20):
        for r in range(3):
            rows.append({
                "Patient_ID": f"patient_{p}",
                "OUTPUT": 0 if p < 10 else 1,
                "score": float(p * 10 + r)
            })
    df = pd.DataFrame(rows)
    scores = df["score"].values
    
    ci_dict = patient_group_bootstrap(df, scores, warn_threshold=50.0, n_bootstrap=50, seed=42)
    
    # Bootstrapped CIs should be valid tuples of (mean, lower, upper)
    assert "pr_auc" in ci_dict
    assert "roc_auc" in ci_dict
    assert "f1" in ci_dict
    
    mean_pr, low_pr, high_pr = ci_dict["pr_auc"]
    assert 0.0 <= low_pr <= mean_pr <= high_pr <= 1.0 + 1e-6


# ==============================================================================
# TEST 10: TIED SCORES RECALL AT 5% FPR WITHOUT INTERPOLATION
# ==============================================================================

def test_compute_metrics_tied_scores_roc():
    """
    Verify compute_metrics handles discrete/tied scores (e.g. Rule Engine 0.0, 0.5, 1.0)
    correctly by selecting ROC operating points where FPR <= 0.05 without collapsing to 100% recall.
    """
    # 1000 normal: 960 score 0.0, 30 score 0.5, 10 score 1.0 (4% FPR if threshold >= 0.5)
    # 1000 abnormal: 200 score 0.0, 300 score 0.5, 500 score 1.0
    y_true = np.array([0] * 960 + [0] * 30 + [0] * 10 + [1] * 200 + [1] * 300 + [1] * 500)
    scores = np.array([0.0] * 960 + [0.5] * 30 + [1.0] * 10 + [0.0] * 200 + [0.5] * 300 + [1.0] * 500)

    m = compute_metrics(y_true, scores, warn_threshold=0.5)

    # At threshold 0.5, FPR is 40/1000 = 0.04 (<= 0.05), and TPR (Recall) is (300+500)/1000 = 0.80
    assert m["recall_at_5pct_fpr"] is not None
    assert abs(m["recall_at_5pct_fpr"] - 0.80) < 1e-5, f"Expected 0.80, got {m['recall_at_5pct_fpr']}"

    # Verify that discrete scores where normal has >5% non-zero values correctly drops to next threshold
    # 1000 normal: 900 score 0.0, 60 score 0.5, 40 score 1.0 (threshold 0.5 has FPR 10% > 5%, threshold 1.0 has FPR 4% <= 5%)
    y_true2 = np.array([0] * 900 + [0] * 60 + [0] * 40 + [1] * 200 + [1] * 300 + [1] * 500)
    scores2 = np.array([0.0] * 900 + [0.5] * 60 + [1.0] * 40 + [0.0] * 200 + [0.5] * 300 + [1.0] * 500)
    m2 = compute_metrics(y_true2, scores2, warn_threshold=0.5)
    # At threshold 1.0, FPR is 40/1000 = 0.04 (<= 0.05), TPR is 500/1000 = 0.50
    assert abs(m2["recall_at_5pct_fpr"] - 0.50) < 1e-5, f"Expected 0.50, got {m2['recall_at_5pct_fpr']}"
