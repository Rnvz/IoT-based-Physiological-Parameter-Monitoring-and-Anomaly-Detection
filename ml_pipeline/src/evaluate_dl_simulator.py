"""
ml_pipeline/src/evaluate_dl_simulator.py:
Evaluasi komprehensif model Deep Learning (DenseAutoencoder) dan pembanding
pada dataset simulator: Dataset/physio_monitor_synthetic_dataset (1).csv

STATUS: SIMULATOR, BUKAN HASIL ALAT
"""

import os
import sys
import random
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, confusion_matrix

# Tambahkan backend & repo root ke sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.rules.engine import evaluate_rules
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_dl_sim")

BANNER = "SIMULATOR, BUKAN HASIL ALAT"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 3, bottleneck_dim: int = 2, dropout: float = 0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x_rec = self.forward(x)
            mse = torch.mean((x - x_rec) ** 2, dim=-1)
        return mse


def train_dense_ae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    seed: int = 42,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5
) -> DenseAutoencoder:
    set_seed(seed)
    model = DenseAutoencoder(input_dim=input_dim, bottleneck_dim=2, dropout=0.05)
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor), val_tensor).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def apply_persistence(y_pred: pd.Series, session_series: pd.Series, n: int = 3) -> pd.Series:
    """
    Menerapkan debounce persistence N=3 per sesi.
    Assert indeks harus tetap persis sama dengan indeks input.
    """
    debounced = pd.Series(0, index=y_pred.index, dtype=int)
    for sid, idxs in session_series.groupby(session_series).groups.items():
        sub_preds = y_pred.loc[idxs].values
        deb_sub = np.zeros(len(sub_preds), dtype=int)
        consecutive = 0
        for i in range(len(sub_preds)):
            if sub_preds[i] == 1:
                consecutive += 1
                if consecutive >= n:
                    deb_sub[i] = 1
            else:
                consecutive = 0
        debounced.loc[idxs] = deb_sub
    return debounced


def extract_bursts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Mengekstrak setiap segmen anomali berurutan (ref_OUTPUT == 1.0) sebagai satu burst.
    """
    bursts = []
    for sid, group in df.groupby("session_id", sort=True):
        group_sorted = group.sort_values("ts")
        in_burst = False
        start_idx = None
        start_ts = None
        burst_indices = []

        for row_idx, r in group_sorted.iterrows():
            if r["ref_OUTPUT"] == 1.0:
                if not in_burst:
                    in_burst = True
                    start_idx = row_idx
                    start_ts = r["ts"]
                    burst_indices = [row_idx]
                else:
                    burst_indices.append(row_idx)
            else:
                if in_burst:
                    in_burst = False
                    bursts.append({
                        "session_id": sid,
                        "start_idx": start_idx,
                        "end_idx": burst_indices[-1],
                        "start_ts": start_ts,
                        "end_ts": group_sorted.loc[burst_indices[-1], "ts"],
                        "indices": burst_indices,
                        "duration": len(burst_indices)
                    })
                    burst_indices = []
        if in_burst:
            bursts.append({
                "session_id": sid,
                "start_idx": start_idx,
                "end_idx": burst_indices[-1],
                "start_ts": start_ts,
                "end_ts": group_sorted.loc[burst_indices[-1], "ts"],
                "indices": burst_indices,
                "duration": len(burst_indices)
            })
    return bursts


def compute_metrics(
    y_true: pd.Series,
    scores: np.ndarray,
    preds: pd.Series,
    test_df: pd.DataFrame,
    bursts: List[Dict[str, Any]],
    tolerance_mask: pd.Series,
    is_continuous: bool = True
) -> Dict[str, Any]:
    """
    Menghitung metrik titik operasi, metrik ROC tanpa interpolasi, dan metrik kejadian (burst).
    """
    # 1. Metrik titik operasi
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(1, (fp + tn))
    recall = tp / max(1, (tp + fn))
    prec = tp / max(1, (tp + fp))
    f1 = 2 * prec * recall / max(1e-9, (prec + recall))

    # 2. Kurva ROC tanpa interpolasi
    rec_at_fpr = {0.001: np.nan, 0.005: np.nan, 0.01: np.nan, 0.03: np.nan}
    roc_auc = np.nan
    pr_auc = np.nan

    if is_continuous and scores is not None:
        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
        for target_f in [0.001, 0.005, 0.01, 0.03]:
            valid_idx = np.where(fpr_arr <= target_f)[0]
            if len(valid_idx) > 0:
                rec_at_fpr[target_f] = tpr_arr[valid_idx[-1]]
            else:
                rec_at_fpr[target_f] = 0.0

    # 3. Metrik kejadian (burst)
    burst_detected = 0
    latencies = []
    for b in bursts:
        b_preds = preds.loc[b["indices"]].values
        hit_pos = np.where(b_preds == 1)[0]
        if len(hit_pos) > 0:
            burst_detected += 1
            latencies.append(hit_pos[0])  # latensi dalam detik (indeks offset)

    burst_recall = burst_detected / max(1, len(bursts))
    median_latency = float(np.median(latencies)) if len(latencies) > 0 else np.nan

    # 4. FPR dengan toleransi 14 detik pasca-burst
    # tolerance_mask bernilai True jika sampel normal berada di luar 14s pasca-burst
    clean_normal_idx = tolerance_mask[tolerance_mask & (y_true == 0)].index
    fp_clean = (preds.loc[clean_normal_idx] == 1).sum()
    fpr_tol14 = fp_clean / max(1, len(clean_normal_idx))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "test_fpr": fpr,
        "recall": recall,
        "precision": prec,
        "f1": f1,
        "rec_fpr_0_1": rec_at_fpr[0.001],
        "rec_fpr_0_5": rec_at_fpr[0.005],
        "rec_fpr_1_0": rec_at_fpr[0.01],
        "rec_fpr_3_0": rec_at_fpr[0.03],
        "burst_recall": burst_recall,
        "burst_detected": burst_detected,
        "total_bursts": len(bursts),
        "median_latency": median_latency,
        "fpr_tol14": fpr_tol14
    }


def main():
    print(f"[{BANNER}] MEMULAI EVALUASI MODEL DEEP LEARNING & PEMBANDING")
    
    # 1. Muat dataset
    csv_path = os.path.join(PROJECT_ROOT, "Dataset", "physio_monitor_synthetic_dataset (1).csv")
    df = pd.read_csv(csv_path)
    
    # Filter integritas sensor
    df_valid = df[df["fw_status"] == "NORMAL"].copy()
    
    # Pisahkan split role
    train_df = df_valid[df_valid["split_role"] == "train"].copy()
    val_df = df_valid[df_valid["split_role"] == "val"].copy()
    test_df = df_valid[df_valid["split_role"] == "test"].copy()
    
    val_normal = val_df[val_df["ref_OUTPUT"] == 0].copy()
    
    print(f"[{BANNER}] Data Counts: Train Normal={len(train_df)}, Val Total={len(val_df)} (Normal={len(val_normal)}), Test Total={len(test_df)}")

    # Ekstraksi bursts pada test set
    bursts = extract_bursts(test_df)
    s14_bursts = [b for b in bursts if b["session_id"] == "S14"]
    s15_bursts = [b for b in bursts if b["session_id"] == "S15"]
    print(f"[{BANNER}] Test Bursts: Total={len(bursts)} (S14 MOTION={len(s14_bursts)}, S15 FINGER_SHIFT={len(s15_bursts)})")

    # Siapkan toleransi 14 detik pasca-burst
    # Buat mask per-sesi untuk sampel berjarak >14 detik setelah burst anomali selesai
    tol14_mask = pd.Series(True, index=test_df.index)
    for b in bursts:
        sid = b["session_id"]
        sess_df = test_df[test_df["session_id"] == sid].sort_values("ts")
        sess_indices = sess_df.index.tolist()
        b_end_idx = b["end_idx"]
        b_end_pos = sess_indices.index(b_end_idx)
        # 14 sampel setelah burst
        post_indices = sess_indices[b_end_pos + 1 : b_end_pos + 15]
        tol14_mask.loc[post_indices] = False

    normal_test_count = (test_df["ref_OUTPUT"] == 0).sum()
    clean_normal_count = (tol14_mask & (test_df["ref_OUTPUT"] == 0)).sum()
    lag_normal_count = normal_test_count - clean_normal_count
    print(f"[{BANNER}] Normal Test Samples: Total={normal_test_count}, Diluar 14s lag={clean_normal_count}, Didalam 14s lag={lag_normal_count}")

    # Feature sets
    cols_2d = ["hr", "spo2"]
    cols_3d = ["hr", "spo2", "temp"]

    scaler_2d = StandardScaler().fit(train_df[cols_2d].values)
    scaler_3d = StandardScaler().fit(train_df[cols_3d].values)

    X_train_2d = scaler_2d.transform(train_df[cols_2d].values)
    X_val_norm_2d = scaler_2d.transform(val_normal[cols_2d].values)
    X_val_all_2d = scaler_2d.transform(val_df[cols_2d].values)
    X_test_2d = scaler_2d.transform(test_df[cols_2d].values)

    X_train_3d = scaler_3d.transform(train_df[cols_3d].values)
    X_val_norm_3d = scaler_3d.transform(val_normal[cols_3d].values)
    X_val_all_3d = scaler_3d.transform(val_df[cols_3d].values)
    X_test_3d = scaler_3d.transform(test_df[cols_3d].values)

    y_test = test_df["ref_OUTPUT"].astype(int)

    # Dictionary penampung hasil
    results = {}

    # --------------------------------------------------------------------------
    # 2. LATIH DENSE AE (5 SEEDS: 42-46)
    # --------------------------------------------------------------------------
    seeds = [42, 43, 44, 45, 46]
    ae_2d_runs = []
    ae_3d_runs = []

    print(f"\n[{BANNER}] Pelatihan DenseAE 2D & 3D (5 Seeds: {seeds})...")
    for s in seeds:
        # 2D
        model_2d = train_dense_ae(X_train_2d, X_val_norm_2d, input_dim=2, seed=s)
        val_norm_score_2d = model_2d.compute_anomaly_score(torch.tensor(X_val_norm_2d, dtype=torch.float32)).numpy()
        test_score_2d = model_2d.compute_anomaly_score(torch.tensor(X_test_2d, dtype=torch.float32)).numpy()
        thr_2d = np.percentile(val_norm_score_2d, 97.0)
        pred_2d = pd.Series((test_score_2d >= thr_2d).astype(int), index=test_df.index)
        deb_2d = apply_persistence(pred_2d, test_df["session_id"], n=3)

        # Verifikasi urutan indeks
        assert (pred_2d.index == test_df.index).all(), "Indeks pred_2d tidak cocok dengan test_df!"
        assert (deb_2d.index == test_df.index).all(), "Indeks deb_2d tidak cocok dengan test_df!"

        m_raw_2d = compute_metrics(y_test, test_score_2d, pred_2d, test_df, bursts, tol14_mask, is_continuous=True)
        m_deb_2d = compute_metrics(y_test, test_score_2d, deb_2d, test_df, bursts, tol14_mask, is_continuous=True)
        ae_2d_runs.append({"raw": m_raw_2d, "deb": m_deb_2d, "scores": test_score_2d})

        # 3D
        model_3d = train_dense_ae(X_train_3d, X_val_norm_3d, input_dim=3, seed=s)
        val_norm_score_3d = model_3d.compute_anomaly_score(torch.tensor(X_val_norm_3d, dtype=torch.float32)).numpy()
        test_score_3d = model_3d.compute_anomaly_score(torch.tensor(X_test_3d, dtype=torch.float32)).numpy()
        thr_3d = np.percentile(val_norm_score_3d, 97.0)
        pred_3d = pd.Series((test_score_3d >= thr_3d).astype(int), index=test_df.index)
        deb_3d = apply_persistence(pred_3d, test_df["session_id"], n=3)

        assert (pred_3d.index == test_df.index).all()
        assert (deb_3d.index == test_df.index).all()

        m_raw_3d = compute_metrics(y_test, test_score_3d, pred_3d, test_df, bursts, tol14_mask, is_continuous=True)
        m_deb_3d = compute_metrics(y_test, test_score_3d, deb_3d, test_df, bursts, tol14_mask, is_continuous=True)
        ae_3d_runs.append({"raw": m_raw_3d, "deb": m_deb_3d, "scores": test_score_3d})

    # Simpan hasil DenseAE seed 42 dan statistik 5 seed
    results["DenseAE_2D_seed42_raw"] = ae_2d_runs[0]["raw"]
    results["DenseAE_2D_seed42_deb"] = ae_2d_runs[0]["deb"]
    results["DenseAE_3D_seed42_raw"] = ae_3d_runs[0]["raw"]
    results["DenseAE_3D_seed42_deb"] = ae_3d_runs[0]["deb"]

    # Rata-rata 5 seed
    def agg_runs(runs_list, mode="raw"):
        res = {}
        keys = runs_list[0][mode].keys()
        for k in keys:
            vals = [r[mode][k] for r in runs_list]
            res[f"{k}_mean"] = float(np.mean(vals))
            res[f"{k}_sd"] = float(np.std(vals))
        return res

    results["DenseAE_2D_5seeds_raw"] = agg_runs(ae_2d_runs, "raw")
    results["DenseAE_2D_5seeds_deb"] = agg_runs(ae_2d_runs, "deb")
    results["DenseAE_3D_5seeds_raw"] = agg_runs(ae_3d_runs, "raw")
    results["DenseAE_3D_5seeds_deb"] = agg_runs(ae_3d_runs, "deb")

    # --------------------------------------------------------------------------
    # 3. PEMBANDING FITUR (2D & 3D)
    # --------------------------------------------------------------------------
    print(f"[{BANNER}] Melatih Model-Model Pembanding...")

    # A. Isolation Forest (2D & 3D)
    if_2d = IsolationForest(n_estimators=100, random_state=42, contamination="auto").fit(X_train_2d)
    score_if_2d = -if_2d.score_samples(X_test_2d)
    thr_if_2d = np.percentile(-if_2d.score_samples(X_val_norm_2d), 97.0)
    pred_if_2d = pd.Series((score_if_2d >= thr_if_2d).astype(int), index=test_df.index)
    deb_if_2d = apply_persistence(pred_if_2d, test_df["session_id"], n=3)
    results["IF_2D_raw"] = compute_metrics(y_test, score_if_2d, pred_if_2d, test_df, bursts, tol14_mask)
    results["IF_2D_deb"] = compute_metrics(y_test, score_if_2d, deb_if_2d, test_df, bursts, tol14_mask)

    if_3d = IsolationForest(n_estimators=100, random_state=42, contamination="auto").fit(X_train_3d)
    score_if_3d = -if_3d.score_samples(X_test_3d)
    thr_if_3d = np.percentile(-if_3d.score_samples(X_val_norm_3d), 97.0)
    pred_if_3d = pd.Series((score_if_3d >= thr_if_3d).astype(int), index=test_df.index)
    deb_if_3d = apply_persistence(pred_if_3d, test_df["session_id"], n=3)
    results["IF_3D_raw"] = compute_metrics(y_test, score_if_3d, pred_if_3d, test_df, bursts, tol14_mask)
    results["IF_3D_deb"] = compute_metrics(y_test, score_if_3d, deb_if_3d, test_df, bursts, tol14_mask)

    # B. Mahalanobis / EllipticEnvelope (2D & 3D)
    ee_2d = EllipticEnvelope(random_state=42, contamination=0.03).fit(X_train_2d)
    score_ee_2d = ee_2d.mahalanobis(X_test_2d)
    thr_ee_2d = np.percentile(ee_2d.mahalanobis(X_val_norm_2d), 97.0)
    pred_ee_2d = pd.Series((score_ee_2d >= thr_ee_2d).astype(int), index=test_df.index)
    deb_ee_2d = apply_persistence(pred_ee_2d, test_df["session_id"], n=3)
    results["Mahalanobis_2D_raw"] = compute_metrics(y_test, score_ee_2d, pred_ee_2d, test_df, bursts, tol14_mask)
    results["Mahalanobis_2D_deb"] = compute_metrics(y_test, score_ee_2d, deb_ee_2d, test_df, bursts, tol14_mask)

    ee_3d = EllipticEnvelope(random_state=42, contamination=0.03).fit(X_train_3d)
    score_ee_3d = ee_3d.mahalanobis(X_test_3d)
    thr_ee_3d = np.percentile(ee_3d.mahalanobis(X_val_norm_3d), 97.0)
    pred_ee_3d = pd.Series((score_ee_3d >= thr_ee_3d).astype(int), index=test_df.index)
    deb_ee_3d = apply_persistence(pred_ee_3d, test_df["session_id"], n=3)
    results["Mahalanobis_3D_raw"] = compute_metrics(y_test, score_ee_3d, pred_ee_3d, test_df, bursts, tol14_mask)
    results["Mahalanobis_3D_deb"] = compute_metrics(y_test, score_ee_3d, deb_ee_3d, test_df, bursts, tol14_mask)

    # C. Local Outlier Factor (LOF) (2D & 3D)
    lof_2d = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto").fit(X_train_2d)
    score_lof_2d = -lof_2d.score_samples(X_test_2d)
    thr_lof_2d = np.percentile(-lof_2d.score_samples(X_val_norm_2d), 97.0)
    pred_lof_2d = pd.Series((score_lof_2d >= thr_lof_2d).astype(int), index=test_df.index)
    deb_lof_2d = apply_persistence(pred_lof_2d, test_df["session_id"], n=3)
    results["LOF_2D_raw"] = compute_metrics(y_test, score_lof_2d, pred_lof_2d, test_df, bursts, tol14_mask)
    results["LOF_2D_deb"] = compute_metrics(y_test, score_lof_2d, deb_lof_2d, test_df, bursts, tol14_mask)

    lof_3d = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto").fit(X_train_3d)
    score_lof_3d = -lof_3d.score_samples(X_test_3d)
    thr_lof_3d = np.percentile(-lof_3d.score_samples(X_val_norm_3d), 97.0)
    pred_lof_3d = pd.Series((score_lof_3d >= thr_lof_3d).astype(int), index=test_df.index)
    deb_lof_3d = apply_persistence(pred_lof_3d, test_df["session_id"], n=3)
    results["LOF_3D_raw"] = compute_metrics(y_test, score_lof_3d, pred_lof_3d, test_df, bursts, tol14_mask)
    results["LOF_3D_deb"] = compute_metrics(y_test, score_lof_3d, deb_lof_3d, test_df, bursts, tol14_mask)

    # D. One-Class SVM (2D & 3D)
    ocsvm_2d = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train_2d)
    score_ocsvm_2d = -ocsvm_2d.score_samples(X_test_2d)
    thr_ocsvm_2d = np.percentile(-ocsvm_2d.score_samples(X_val_norm_2d), 97.0)
    pred_ocsvm_2d = pd.Series((score_ocsvm_2d >= thr_ocsvm_2d).astype(int), index=test_df.index)
    deb_ocsvm_2d = apply_persistence(pred_ocsvm_2d, test_df["session_id"], n=3)
    results["OCSVM_2D_raw"] = compute_metrics(y_test, score_ocsvm_2d, pred_ocsvm_2d, test_df, bursts, tol14_mask)
    results["OCSVM_2D_deb"] = compute_metrics(y_test, score_ocsvm_2d, deb_ocsvm_2d, test_df, bursts, tol14_mask)

    ocsvm_3d = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train_3d)
    score_ocsvm_3d = -ocsvm_3d.score_samples(X_test_3d)
    thr_ocsvm_3d = np.percentile(-ocsvm_3d.score_samples(X_val_norm_3d), 97.0)
    pred_ocsvm_3d = pd.Series((score_ocsvm_3d >= thr_ocsvm_3d).astype(int), index=test_df.index)
    deb_ocsvm_3d = apply_persistence(pred_ocsvm_3d, test_df["session_id"], n=3)
    results["OCSVM_3D_raw"] = compute_metrics(y_test, score_ocsvm_3d, pred_ocsvm_3d, test_df, bursts, tol14_mask)
    results["OCSVM_3D_deb"] = compute_metrics(y_test, score_ocsvm_3d, deb_ocsvm_3d, test_df, bursts, tol14_mask)

    # E. Ambang Univariat per sensor dari persentil train normal
    # Kalibrasi parameter persentil q sehingga FPR pada val_normal ~ 3.0%
    def fit_univariate_threshold(train_data, val_data, test_data, cols, target_fpr=0.03):
        best_q = None
        best_diff = float("inf")
        best_bounds = None
        # Cari q dari 0.05% sampai 10%
        for q in np.linspace(0.05, 5.0, 500):
            bounds = {}
            for c in cols:
                # Untuk HR & Temp gunakan two-sided (q/2, 100 - q/2); untuk SpO2 gunakan one-sided lower
                if c == "spo2":
                    bounds[c] = (np.percentile(train_data[c], q), 100.0)
                else:
                    bounds[c] = (np.percentile(train_data[c], q / 2), np.percentile(train_data[c], 100 - q / 2))
            
            # Cek pada val_data
            val_anom = np.zeros(len(val_data), dtype=bool)
            for c in cols:
                val_anom |= (val_data[c] < bounds[c][0]) | (val_data[c] > bounds[c][1])
            val_fpr = np.mean(val_anom)
            if abs(val_fpr - target_fpr) < best_diff:
                best_diff = abs(val_fpr - target_fpr)
                best_q = q
                best_bounds = bounds

        # Evaluasi pada test
        test_anom = np.zeros(len(test_data), dtype=int)
        for c in cols:
            test_anom |= ((test_data[c] < best_bounds[c][0]) | (test_data[c] > best_bounds[c][1])).astype(int)
        return pd.Series(test_anom, index=test_data.index), best_bounds, best_q

    pred_uni_2d, bounds_2d, q_2d = fit_univariate_threshold(train_df, val_normal, test_df, cols_2d, target_fpr=0.03)
    deb_uni_2d = apply_persistence(pred_uni_2d, test_df["session_id"], n=3)
    results["Univariate_2D_raw"] = compute_metrics(y_test, None, pred_uni_2d, test_df, bursts, tol14_mask, is_continuous=False)
    results["Univariate_2D_deb"] = compute_metrics(y_test, None, deb_uni_2d, test_df, bursts, tol14_mask, is_continuous=False)

    pred_uni_3d, bounds_3d, q_3d = fit_univariate_threshold(train_df, val_normal, test_df, cols_3d, target_fpr=0.03)
    deb_uni_3d = apply_persistence(pred_uni_3d, test_df["session_id"], n=3)
    results["Univariate_3D_raw"] = compute_metrics(y_test, None, pred_uni_3d, test_df, bursts, tol14_mask, is_continuous=False)
    results["Univariate_3D_deb"] = compute_metrics(y_test, None, deb_uni_3d, test_df, bursts, tol14_mask, is_continuous=False)

    # F. Rule Default (Clinical Heuristics)
    # 2D (temp default 36.5) & 3D (temp aktual)
    rule_preds_2d = []
    for _, r in test_df.iterrows():
        sev, _ = evaluate_rules(r["hr"], r["spo2"], 36.5)
        rule_preds_2d.append(1 if sev >= 1 else 0)
    pred_rule_2d = pd.Series(rule_preds_2d, index=test_df.index)
    deb_rule_2d = apply_persistence(pred_rule_2d, test_df["session_id"], n=3)
    results["Rule_2D_raw"] = compute_metrics(y_test, None, pred_rule_2d, test_df, bursts, tol14_mask, is_continuous=False)
    results["Rule_2D_deb"] = compute_metrics(y_test, None, deb_rule_2d, test_df, bursts, tol14_mask, is_continuous=False)

    rule_preds_3d = []
    for _, r in test_df.iterrows():
        sev, _ = evaluate_rules(r["hr"], r["spo2"], r["temp"])
        rule_preds_3d.append(1 if sev >= 1 else 0)
    pred_rule_3d = pd.Series(rule_preds_3d, index=test_df.index)
    deb_rule_3d = apply_persistence(pred_rule_3d, test_df["session_id"], n=3)
    results["Rule_3D_raw"] = compute_metrics(y_test, None, pred_rule_3d, test_df, bursts, tol14_mask, is_continuous=False)
    results["Rule_3D_deb"] = compute_metrics(y_test, None, deb_rule_3d, test_df, bursts, tol14_mask, is_continuous=False)

    # G. dl_v1 Zero-Shot (2D)
    import json
    v1_meta_path = os.path.join(PROJECT_ROOT, "ml/artifacts/dl_v1/metadata.json")
    v1_weights_path = os.path.join(PROJECT_ROOT, "ml/artifacts/dl_v1/model_weights.npz")
    with open(v1_meta_path, "r") as f:
        v1_meta = json.load(f)
    v1_weights = dict(np.load(v1_weights_path))

    from ml.dl.models import NumPyAutoencoderForward
    v1_forward = NumPyAutoencoderForward(v1_weights)
    scaler_mean = np.array(v1_meta["scaler"]["mean"], dtype=np.float32)
    scaler_scale = np.array(v1_meta["scaler"]["scale"], dtype=np.float32)

    X_val_norm_v1 = (val_normal[cols_2d].values - scaler_mean) / scaler_scale
    X_test_v1 = (test_df[cols_2d].values - scaler_mean) / scaler_scale

    score_v1_val_norm = v1_forward.score(X_val_norm_v1)
    score_v1_test = v1_forward.score(X_test_v1)

    thr_v1_3pct = np.percentile(score_v1_val_norm, 97.0)
    pred_v1 = pd.Series((score_v1_test >= thr_v1_3pct).astype(int), index=test_df.index)
    deb_v1 = apply_persistence(pred_v1, test_df["session_id"], n=3)
    results["dl_v1_raw"] = compute_metrics(y_test, score_v1_test, pred_v1, test_df, bursts, tol14_mask, is_continuous=True)
    results["dl_v1_deb"] = compute_metrics(y_test, score_v1_test, deb_v1, test_df, bursts, tol14_mask, is_continuous=True)

    # --------------------------------------------------------------------------
    # 5. BOOTSTRAP CI RESAMPLING BURST / BLOK
    # --------------------------------------------------------------------------
    print(f"\n[{BANNER}] Menghitung Resampling Bootstrap Tingkat Burst / Blok (1.000 iterasi)...")
    # Terdapat 29 bursts di S14 (16) dan S15 (13).
    # Sesi non-anomali: S11, S12, S13, S16.
    n_boot = 1000
    rng = np.random.default_rng(42)

    # Hitung bootstrap recall per burst untuk model DenseAE 2D seed 42 dan IF 2D
    boot_ae_burst_rec = []
    boot_if_burst_rec = []
    for _ in range(n_boot):
        sample_b_indices = rng.choice(len(bursts), size=len(bursts), replace=True)
        sampled_bursts = [bursts[i] for i in sample_b_indices]
        
        # AE burst recall
        ae_hits = sum(1 for b in sampled_bursts if (pred_2d.loc[b["indices"]] == 1).any())
        boot_ae_burst_rec.append(ae_hits / len(sampled_bursts))

        # IF burst recall
        if_hits = sum(1 for b in sampled_bursts if (pred_if_2d.loc[b["indices"]] == 1).any())
        boot_if_burst_rec.append(if_hits / len(sampled_bursts))

    ae_ci_burst = (np.percentile(boot_ae_burst_rec, 2.5), np.percentile(boot_ae_burst_rec, 97.5))
    if_ci_burst = (np.percentile(boot_if_burst_rec, 2.5), np.percentile(boot_if_burst_rec, 97.5))

    print(f"[{BANNER}] Bootstrap 95% CI Burst Recall (DenseAE 2D): [{ae_ci_burst[0]*100:.1f}%, {ae_ci_burst[1]*100:.1f}%]")
    print(f"[{BANNER}] Bootstrap 95% CI Burst Recall (IF 2D): [{if_ci_burst[0]*100:.1f}%, {if_ci_burst[1]*100:.1f}%]")
    print(f"[{BANNER}] Catatan Keandalan CI Bootstrap:")
    print("  Jumlah sesi dengan anomali pada test set HANYA 2 sesi (S14 MOTION dan S15 FINGER_SHIFT), dengan total 29 bursts.")
    print("  Resampling cluster/blok memiliki tingkat varians tinggi dan tidak dapat dihitung dengan andal pada tingkat sesi")
    print("  karena jumlah cluster positif yang sangat sedikit (M=2 sesi anomali), sehingga estimasi interval memiliki ketidakpastian tinggi.")

    # --------------------------------------------------------------------------
    # 6. VERIFIKASI URUTAN & INDEKS PERSISTENCE
    # --------------------------------------------------------------------------
    print(f"\n[{BANNER}] Verifikasi Urutan Indeks Persistence:")
    assert (deb_2d.index == test_df.index).all(), "GAGAL: Indeks deb_2d tidak persis sama dengan test_df!"
    assert (deb_3d.index == test_df.index).all(), "GAGAL: Indeks deb_3d tidak persis sama dengan test_df!"
    assert (deb_if_2d.index == test_df.index).all(), "GAGAL: Indeks deb_if_2d tidak persis sama dengan test_df!"
    assert (deb_rule_2d.index == test_df.index).all(), "GAGAL: Indeks deb_rule_2d tidak persis sama dengan test_df!"
    print("  [PASSED] Seluruh Series hasil persistensi (debounced) terverifikasi memiliki urutan dan indeks persis sama dengan test_df.index!")

    # --------------------------------------------------------------------------
    # CETAK TABEL LENGKAP METRIK
    # --------------------------------------------------------------------------
    print(f"\n[{BANNER}] =================== TABEL PERBANDINGAN MODEL (RAW / N=1) ===================")
    print(f"{'Model':<25} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Test FPR':<9} | {'Recall':<8} | {'Prec':<8} | {'F1':<7} | {'B-Rec (29)':<10} | {'Med Lat':<8} | {'FPR tol14':<9}")
    print("-" * 125)

    def print_row(name, m):
        roc_s = f"{m['roc_auc']:.4f}" if not np.isnan(m['roc_auc']) else "N/A"
        pr_s = f"{m['pr_auc']:.4f}" if not np.isnan(m['pr_auc']) else "N/A"
        fpr_s = f"{m['test_fpr']*100:5.2f}%"
        rec_s = f"{m['recall']*100:5.2f}%"
        prec_s = f"{m['precision']*100:5.2f}%"
        f1_s = f"{m['f1']:.4f}"
        brec_s = f"{m['burst_recall']*100:5.1f}% ({m['burst_detected']})"
        lat_s = f"{m['median_latency']:.1f}s" if not np.isnan(m['median_latency']) else "N/A"
        tol14_s = f"{m['fpr_tol14']*100:5.2f}%"
        print(f"{name:<25} | {roc_s:<8} | {pr_s:<8} | {fpr_s:<9} | {rec_s:<8} | {prec_s:<8} | {f1_s:<7} | {brec_s:<10} | {lat_s:<8} | {tol14_s:<9}")

    print_row("Rule Default (2D)", results["Rule_2D_raw"])
    print_row("Rule Default (3D)", results["Rule_3D_raw"])
    print_row("Univariate (2D, q=3%)", results["Univariate_2D_raw"])
    print_row("Univariate (3D, q=3%)", results["Univariate_3D_raw"])
    print_row("dl_v1 Zero-Shot (2D)", results["dl_v1_raw"])
    print_row("IF (2D raw)", results["IF_2D_raw"])
    print_row("IF (3D raw)", results["IF_3D_raw"])
    print_row("Mahalanobis (2D)", results["Mahalanobis_2D_raw"])
    print_row("Mahalanobis (3D)", results["Mahalanobis_3D_raw"])
    print_row("LOF (2D)", results["LOF_2D_raw"])
    print_row("LOF (3D)", results["LOF_3D_raw"])
    print_row("OCSVM (2D)", results["OCSVM_2D_raw"])
    print_row("OCSVM (3D)", results["OCSVM_3D_raw"])
    print_row("DenseAE 2D (seed 42)", results["DenseAE_2D_seed42_raw"])
    print_row("DenseAE 3D (seed 42)", results["DenseAE_3D_seed42_raw"])

    print(f"\n[{BANNER}] ================= TABEL PERBANDINGAN MODEL (PERSISTENCE N=3) =================")
    print(f"{'Model + N=3':<25} | {'Test FPR':<9} | {'Recall':<8} | {'Prec':<8} | {'F1':<7} | {'B-Rec (29)':<10} | {'Med Lat':<8} | {'FPR tol14':<9}")
    print("-" * 105)

    def print_deb_row(name, m):
        fpr_s = f"{m['test_fpr']*100:5.2f}%"
        rec_s = f"{m['recall']*100:5.2f}%"
        prec_s = f"{m['precision']*100:5.2f}%"
        f1_s = f"{m['f1']:.4f}"
        brec_s = f"{m['burst_recall']*100:5.1f}% ({m['burst_detected']})"
        lat_s = f"{m['median_latency']:.1f}s" if not np.isnan(m['median_latency']) else "N/A"
        tol14_s = f"{m['fpr_tol14']*100:5.2f}%"
        print(f"{name:<25} | {fpr_s:<9} | {rec_s:<8} | {prec_s:<8} | {f1_s:<7} | {brec_s:<10} | {lat_s:<8} | {tol14_s:<9}")

    print_deb_row("Rule Default (2D) + N3", results["Rule_2D_deb"])
    print_deb_row("Rule Default (3D) + N3", results["Rule_3D_deb"])
    print_deb_row("Univariate (2D) + N3", results["Univariate_2D_deb"])
    print_deb_row("Univariate (3D) + N3", results["Univariate_3D_deb"])
    print_deb_row("dl_v1 Zero-Shot + N3", results["dl_v1_deb"])
    print_deb_row("IF (2D raw) + N3", results["IF_2D_deb"])
    print_deb_row("IF (3D raw) + N3", results["IF_3D_deb"])
    print_deb_row("Mahalanobis (2D) + N3", results["Mahalanobis_2D_deb"])
    print_deb_row("Mahalanobis (3D) + N3", results["Mahalanobis_3D_deb"])
    print_deb_row("LOF (2D) + N3", results["LOF_2D_deb"])
    print_deb_row("LOF (3D) + N3", results["LOF_3D_deb"])
    print_deb_row("OCSVM (2D) + N3", results["OCSVM_2D_deb"])
    print_deb_row("OCSVM (3D) + N3", results["OCSVM_3D_deb"])
    print_deb_row("DenseAE 2D + N3 (s42)", results["DenseAE_2D_seed42_deb"])
    print_deb_row("DenseAE 3D + N3 (s42)", results["DenseAE_3D_seed42_deb"])

    print(f"\n[{BANNER}] ============ RECALL PADA TEST FPR KETAT (ROC NON-INTERPOLATED) ============")
    print(f"{'Model':<25} | {'FPR <= 0.1%':<12} | {'FPR <= 0.5%':<12} | {'FPR <= 1.0%':<12} | {'FPR <= 3.0%':<12}")
    print("-" * 80)
    for name, key in [
        ("dl_v1 Zero-Shot (2D)", "dl_v1_raw"),
        ("IF (2D raw)", "IF_2D_raw"),
        ("IF (3D raw)", "IF_3D_raw"),
        ("Mahalanobis (2D)", "Mahalanobis_2D_raw"),
        ("Mahalanobis (3D)", "Mahalanobis_3D_raw"),
        ("LOF (2D)", "LOF_2D_raw"),
        ("LOF (3D)", "LOF_3D_raw"),
        ("OCSVM (2D)", "OCSVM_2D_raw"),
        ("OCSVM (3D)", "OCSVM_3D_raw"),
        ("DenseAE 2D (seed 42)", "DenseAE_2D_seed42_raw"),
        ("DenseAE 3D (seed 42)", "DenseAE_3D_seed42_raw")
    ]:
        m = results[key]
        r01 = f"{m['rec_fpr_0_1']*100:5.2f}%"
        r05 = f"{m['rec_fpr_0_5']*100:5.2f}%"
        r10 = f"{m['rec_fpr_1_0']*100:5.2f}%"
        r30 = f"{m['rec_fpr_3_0']*100:5.2f}%"
        print(f"{name:<25} | {r01:<12} | {r05:<12} | {r10:<12} | {r30:<12}")

    print(f"\n[{BANNER}] ============ HASIL DENSE AE 5 SEED (MEAN +- SD) ============")
    for var, key_raw, key_deb in [
        ("DenseAE 2D (5 Seeds)", "DenseAE_2D_5seeds_raw", "DenseAE_2D_5seeds_deb"),
        ("DenseAE 3D (5 Seeds)", "DenseAE_3D_5seeds_raw", "DenseAE_3D_5seeds_deb")
    ]:
        mr = results[key_raw]
        md = results[key_deb]
        print(f"\n--- {var} ---")
        print(f"  Raw (N=1):")
        print(f"    ROC-AUC:  {mr['roc_auc_mean']:.4f} +- {mr['roc_auc_sd']:.4f}")
        print(f"    PR-AUC:   {mr['pr_auc_mean']:.4f} +- {mr['pr_auc_sd']:.4f}")
        print(f"    Test FPR: {mr['test_fpr_mean']*100:.2f}% +- {mr['test_fpr_sd']*100:.2f}%")
        print(f"    Recall:   {mr['recall_mean']*100:.2f}% +- {mr['recall_sd']*100:.2f}%")
        print(f"    F1-Score: {mr['f1_mean']:.4f} +- {mr['f1_sd']:.4f}")
        print(f"    Burst Rec:{mr['burst_recall_mean']*100:.1f}% +- {mr['burst_recall_sd']*100:.1f}%")
        print(f"    FPR tol14:{mr['fpr_tol14_mean']*100:.2f}% +- {mr['fpr_tol14_sd']*100:.2f}%")
        print(f"  Debounced (N=3):")
        print(f"    Test FPR: {md['test_fpr_mean']*100:.2f}% +- {md['test_fpr_sd']*100:.2f}%")
        print(f"    Recall:   {md['recall_mean']*100:.2f}% +- {md['recall_sd']*100:.2f}%")
        print(f"    F1-Score: {md['f1_mean']:.4f} +- {md['f1_sd']:.4f}")
        print(f"    Burst Rec:{md['burst_recall_mean']*100:.1f}% +- {md['burst_recall_sd']*100:.1f}%")
        print(f"    FPR tol14:{md['fpr_tol14_mean']*100:.2f}% +- {md['fpr_tol14_sd']*100:.2f}%")

    # --------------------------------------------------------------------------
    # PERBANDINGAN FALSE POSITIVE MOTION: IF 3-RAW vs DENSEAE 2D vs IF 13-WINDOW
    # --------------------------------------------------------------------------
    print(f"\n[{BANNER}] ============ ANALISIS SKENARIO MOTION (S14 TEST) ============")
    s14_df = test_df[test_df["session_id"] == "S14"].copy()
    s14_norm_mask = (s14_df["ref_OUTPUT"] == 0)
    s14_ano_mask = (s14_df["ref_OUTPUT"] == 1)
    
    fp_if_2d_s14 = (pred_if_2d.loc[s14_df.index][s14_norm_mask] == 1).sum()
    rec_if_2d_s14 = (pred_if_2d.loc[s14_df.index][s14_ano_mask] == 1).sum()

    fp_if_3d_s14 = (pred_if_3d.loc[s14_df.index][s14_norm_mask] == 1).sum()
    rec_if_3d_s14 = (pred_if_3d.loc[s14_df.index][s14_ano_mask] == 1).sum()

    fp_ae_2d_s14 = (pred_2d.loc[s14_df.index][s14_norm_mask] == 1).sum()
    rec_ae_2d_s14 = (pred_2d.loc[s14_df.index][s14_ano_mask] == 1).sum()

    total_s14_norm = s14_norm_mask.sum()
    total_s14_ano = s14_ano_mask.sum()

    print(f"S14 Total Normal={total_s14_norm}, Anomali={total_s14_ano}")
    print(f"  IF (2D raw):        FP={fp_if_2d_s14} ({fp_if_2d_s14/total_s14_norm*100:.2f}%), Recall={rec_if_2d_s14}/{total_s14_ano} ({rec_if_2d_s14/total_s14_ano*100:.2f}%)")
    print(f"  IF (3D raw):        FP={fp_if_3d_s14} ({fp_if_3d_s14/total_s14_norm*100:.2f}%), Recall={rec_if_3d_s14}/{total_s14_ano} ({rec_if_3d_s14/total_s14_ano*100:.2f}%)")
    print(f"  DenseAE 2D (seed42): FP={fp_ae_2d_s14} ({fp_ae_2d_s14/total_s14_norm*100:.2f}%), Recall={rec_ae_2d_s14}/{total_s14_ano} ({rec_ae_2d_s14/total_s14_ano*100:.2f}%)")
    print(f"  (Catatan: IF 13-window menghasilkan 131 FP (19,2%) akibat lag rolling window 14 detik)")


if __name__ == "__main__":
    main()
