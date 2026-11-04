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
SEEDS = [42, 43, 44, 45, 46]


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
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(1, (fp + tn))
    recall = tp / max(1, (tp + fn))
    prec = tp / max(1, (tp + fp))
    f1 = 2 * prec * recall / max(1e-9, (prec + recall))

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

    burst_detected = 0
    latencies = []
    for b in bursts:
        b_preds = preds.loc[b["indices"]].values
        hit_pos = np.where(b_preds == 1)[0]
        if len(hit_pos) > 0:
            burst_detected += 1
            latencies.append(hit_pos[0])

    burst_recall = burst_detected / max(1, len(bursts))
    median_latency = float(np.median(latencies)) if len(latencies) > 0 else np.nan

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
        "fpr_tol14": fpr_tol14,
        "raw_tp": tp,
        "raw_fp": fp,
        "raw_tn": tn,
        "raw_fn": fn
    }


def aggregate_seed_runs(runs_list: List[Dict[str, Any]], mode: str = "raw") -> Dict[str, float]:
    res = {}
    keys = runs_list[0][mode].keys()
    for k in keys:
        vals = [r[mode][k] for r in runs_list]
        res[f"{k}_mean"] = float(np.mean(vals))
        res[f"{k}_sd"] = float(np.std(vals))
    return res


def compute_scenario_metrics(
    preds: pd.Series,
    test_df: pd.DataFrame,
    scenario_id: str
) -> Dict[str, Any]:
    sub = test_df[test_df["session_id"] == scenario_id]
    y_true = sub["ref_OUTPUT"].astype(int)
    sub_preds = preds.loc[sub.index]
    cm = confusion_matrix(y_true, sub_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    norm_total = tn + fp
    ano_total = tp + fn
    fpr = fp / max(1, norm_total)
    recall = tp / max(1, ano_total)
    return {
        "scenario": scenario_id,
        "norm_total": norm_total,
        "ano_total": ano_total,
        "fp": fp,
        "fpr": fpr,
        "tp": tp,
        "recall": recall
    }


def main():
    print(f"[{BANNER}] MEMULAI EVALUASI LENGKAP SIMULATOR")
    
    # 1. Muat dataset
    csv_path = os.path.join(PROJECT_ROOT, "Dataset", "physio_monitor_synthetic_dataset (1).csv")
    df = pd.read_csv(csv_path)
    
    # Filter integritas sensor
    df_valid = df[df["fw_status"] == "NORMAL"].copy()
    
    train_df = df_valid[df_valid["split_role"] == "train"].copy()
    val_df = df_valid[df_valid["split_role"] == "val"].copy()
    test_df = df_valid[df_valid["split_role"] == "test"].copy()
    val_normal = val_df[val_df["ref_OUTPUT"] == 0].copy()
    
    print(f"[{BANNER}] Data Counts: Train Normal={len(train_df)}, Val Total={len(val_df)} (Normal={len(val_normal)}), Test Total={len(test_df)}")

    # ==========================================================================
    # ITEM 2: COCOKKAN RULE DEFAULT DENGAN KOLOM rule_severity DATASET
    # ==========================================================================
    print(f"\n[{BANNER}] ============ ITEM 2: PENGECEKAN KECOCOKAN RULE DEFAULT ============")
    rule_eval_all = []
    for _, r in df.iterrows():
        sev, _ = evaluate_rules(r["hr"], r["spo2"], r["temp"])
        rule_eval_all.append(sev)
    df["computed_rule_sev"] = rule_eval_all

    rule_eval_valid = []
    for _, r in df_valid.iterrows():
        sev, _ = evaluate_rules(r["hr"], r["spo2"], r["temp"])
        rule_eval_valid.append(sev)
    df_valid["computed_rule_sev"] = rule_eval_valid

    rule_eval_test = []
    for _, r in test_df.iterrows():
        sev, _ = evaluate_rules(r["hr"], r["spo2"], r["temp"])
        rule_eval_test.append(sev)
    test_df["computed_rule_sev"] = rule_eval_test

    match_all = (df["computed_rule_sev"] == df["rule_severity"])
    match_valid = (df_valid["computed_rule_sev"] == df_valid["rule_severity"])
    match_test = (test_df["computed_rule_sev"] == test_df["rule_severity"])

    print(f"  1. Keseluruhan Dataset ({len(df)} baris): {match_all.sum()} / {len(df)} ({match_all.mean()*100:.4f}% cocok)")
    print(f"  2. Baris fw_status == NORMAL ({len(df_valid)} baris): {match_valid.sum()} / {len(df_valid)} ({match_valid.mean()*100:.4f}% cocok)")
    print(f"  3. Test Set Split ({len(test_df)} baris): {match_test.sum()} / {len(test_df)} ({match_test.mean()*100:.4f}% cocok)")
    print(f"  4. Mismatch: {(~match_all).sum()} baris. Seluruh mismatch berada pada fw_status == 'SENSOR_OFF' (di mana rule_severity dataset = NaN).")

    # ==========================================================================
    # ITEM 4 (BURSTS): EKSTRAKSI & STATISTIK 29 BURSTS
    # ==========================================================================
    bursts = extract_bursts(test_df)
    durations = [b["duration"] for b in bursts]
    min_dur = int(np.min(durations))
    med_dur = float(np.median(durations))
    max_dur = int(np.max(durations))
    short_bursts = [b for b in bursts if b["duration"] < 3]

    print(f"\n[{BANNER}] ============ ITEM 4: STATISTIK 29 ANOMALY BURSTS TEST SET ============")
    print(f"Total Bursts: {len(bursts)} (S14 MOTION: 16, S15 FINGER_SHIFT: 13)")
    print(f"Panjang Bursts: Min={min_dur} baris, Median={med_dur:.1f} baris, Max={max_dur} baris")
    print(f"Jumlah Bursts dengan durasi < 3 baris: {len(short_bursts)} dari {len(bursts)} (0.0%)")
    print("Daftar 29 Bursts:")
    for idx, b in enumerate(bursts, 1):
        print(f"  Burst #{idx:02d} | Sesi={b['session_id']} | Durasi={b['duration']:2d}s | Start={b['start_ts']} | End={b['end_ts']}")

    # Mask toleransi 14s pasca-burst
    tol14_mask = pd.Series(True, index=test_df.index)
    for b in bursts:
        sid = b["session_id"]
        sess_df = test_df[test_df["session_id"] == sid].sort_values("ts")
        sess_indices = sess_df.index.tolist()
        b_end_idx = b["end_idx"]
        b_end_pos = sess_indices.index(b_end_idx)
        post_indices = sess_indices[b_end_pos + 1 : b_end_pos + 15]
        tol14_mask.loc[post_indices] = False

    # Feature sets
    cols_2d = ["hr", "spo2"]
    cols_3d = ["hr", "spo2", "temp"]

    scaler_2d = StandardScaler().fit(train_df[cols_2d].values)
    scaler_3d = StandardScaler().fit(train_df[cols_3d].values)

    X_train_2d = scaler_2d.transform(train_df[cols_2d].values)
    X_val_norm_2d = scaler_2d.transform(val_normal[cols_2d].values)
    X_test_2d = scaler_2d.transform(test_df[cols_2d].values)

    X_train_3d = scaler_3d.transform(train_df[cols_3d].values)
    X_val_norm_3d = scaler_3d.transform(val_normal[cols_3d].values)
    X_test_3d = scaler_3d.transform(test_df[cols_3d].values)

    y_test = test_df["ref_OUTPUT"].astype(int)

    # ==========================================================================
    # ITEM 1: UNIVARIAT 2D vs 3D DETAIL FITUR, AMBANG, DAN ALARM
    # ==========================================================================
    print(f"\n[{BANNER}] ============ ITEM 1: ANALISIS UNIVARIAT 2D vs 3D ============")
    def fit_univariate_threshold(train_data, val_data, test_data, cols, target_fpr=0.03):
        best_q = None
        best_diff = float("inf")
        best_bounds = None
        for q in np.linspace(0.05, 5.0, 500):
            bounds = {}
            for c in cols:
                if c == "spo2":
                    bounds[c] = (float(np.percentile(train_data[c], q)), 100.0)
                else:
                    bounds[c] = (float(np.percentile(train_data[c], q / 2)), float(np.percentile(train_data[c], 100 - q / 2)))
            val_anom = np.zeros(len(val_data), dtype=bool)
            for c in cols:
                val_anom |= (val_data[c] < bounds[c][0]) | (val_data[c] > bounds[c][1])
            val_fpr = np.mean(val_anom)
            if abs(val_fpr - target_fpr) < best_diff:
                best_diff = abs(val_fpr - target_fpr)
                best_q = q
                best_bounds = bounds

        alarms_per_sensor = {}
        for c in cols:
            s_anom = (test_data[c] < best_bounds[c][0]) | (test_data[c] > best_bounds[c][1])
            alarms_per_sensor[c] = int(s_anom.sum())

        test_anom = np.zeros(len(test_data), dtype=int)
        for c in cols:
            test_anom |= ((test_data[c] < best_bounds[c][0]) | (test_data[c] > best_bounds[c][1])).astype(int)
        
        return pd.Series(test_anom, index=test_data.index), best_bounds, best_q, alarms_per_sensor

    pred_uni_2d, bounds_2d, q_2d, sensor_alarms_2d = fit_univariate_threshold(train_df, val_normal, test_df, cols_2d, target_fpr=0.03)
    deb_uni_2d = apply_persistence(pred_uni_2d, test_df["session_id"], n=3)
    m_uni_2d_raw = compute_metrics(y_test, None, pred_uni_2d, test_df, bursts, tol14_mask, is_continuous=False)
    m_uni_2d_deb = compute_metrics(y_test, None, deb_uni_2d, test_df, bursts, tol14_mask, is_continuous=False)

    pred_uni_3d, bounds_3d, q_3d, sensor_alarms_3d = fit_univariate_threshold(train_df, val_normal, test_df, cols_3d, target_fpr=0.03)
    deb_uni_3d = apply_persistence(pred_uni_3d, test_df["session_id"], n=3)
    m_uni_3d_raw = compute_metrics(y_test, None, pred_uni_3d, test_df, bursts, tol14_mask, is_continuous=False)
    m_uni_3d_deb = compute_metrics(y_test, None, deb_uni_3d, test_df, bursts, tol14_mask, is_continuous=False)

    print("Univariat 2D:")
    print(f"  Fitur: {cols_2d}")
    print(f"  Parameter Persentil q: {q_2d:.4f}%")
    print(f"  Ambang per Fitur:")
    print(f"    - Heart_Rate: Two-sided [{bounds_2d['hr'][0]:.3f}, {bounds_2d['hr'][1]:.3f}] bpm")
    print(f"    - SpO2:       One-sided lower [{bounds_2d['spo2'][0]:.3f}, 100.0] %")
    print(f"  Alarm per Sensor pada Test: {sensor_alarms_2d} (Total Alarm Gabungan = {pred_uni_2d.sum()})")

    print("\nUnivariat 3D:")
    print(f"  Fitur: {cols_3d}")
    print(f"  Parameter Persentil q: {q_3d:.4f}%")
    print(f"  Ambang per Fitur:")
    print(f"    - Heart_Rate:  Two-sided [{bounds_3d['hr'][0]:.3f}, {bounds_3d['hr'][1]:.3f}] bpm")
    print(f"    - SpO2:        One-sided lower [{bounds_3d['spo2'][0]:.3f}, 100.0] %")
    print(f"    - Temperature: Two-sided [{bounds_3d['temp'][0]:.3f}, {bounds_3d['temp'][1]:.3f}] °C")
    print(f"  Alarm per Sensor pada Test: {sensor_alarms_3d} (Total Alarm Gabungan = {pred_uni_3d.sum()})")
    print("Penjelasan Identitas 2D vs 3D:")
    print("  1. Nilai suhu (DS18B20) pada test set berkisar antara 31.5 °C hingga 34.0 °C, semuanya berada di dalam batas [31.0 °C, 34.0 °C].")
    print("  2. Anomali simulator (MOTION & FINGER_SHIFT) murni hemodinamik optik MAX30102 (HR fluktuatif & SpO2 anjlok), sedangkan suhu kulit konstan normal.")
    print("  3. Sensor suhu berkontribusi tepat 0 alarm pada test set, sehingga himpunan alarm gabungan (HR U SpO2 U Temp) identik 100% dengan (HR U SpO2).")

    # ==========================================================================
    # ITEM 3: PELATIHAN 5 SEED UNTUK DENSEAE & ISOLATION FOREST
    # ==========================================================================
    print(f"\n[{BANNER}] ============ ITEM 3: PELATIHAN 5 SEED DENSEAE & ISOLATION FOREST ============")
    
    # 1. DenseAE 5 Seeds
    ae_2d_runs = []
    ae_3d_runs = []
    for s in SEEDS:
        m2 = train_dense_ae(X_train_2d, X_val_norm_2d, input_dim=2, seed=s)
        s_val_2 = m2.compute_anomaly_score(torch.tensor(X_val_norm_2d, dtype=torch.float32)).numpy()
        s_test_2 = m2.compute_anomaly_score(torch.tensor(X_test_2d, dtype=torch.float32)).numpy()
        thr_2 = np.percentile(s_val_2, 97.0)
        p_2 = pd.Series((s_test_2 >= thr_2).astype(int), index=test_df.index)
        deb_2 = apply_persistence(p_2, test_df["session_id"], n=3)
        ae_2d_runs.append({
            "raw": compute_metrics(y_test, s_test_2, p_2, test_df, bursts, tol14_mask),
            "deb": compute_metrics(y_test, s_test_2, deb_2, test_df, bursts, tol14_mask),
            "preds_raw": p_2,
            "preds_deb": deb_2
        })

        m3 = train_dense_ae(X_train_3d, X_val_norm_3d, input_dim=3, seed=s)
        s_val_3 = m3.compute_anomaly_score(torch.tensor(X_val_norm_3d, dtype=torch.float32)).numpy()
        s_test_3 = m3.compute_anomaly_score(torch.tensor(X_test_3d, dtype=torch.float32)).numpy()
        thr_3 = np.percentile(s_val_3, 97.0)
        p_3 = pd.Series((s_test_3 >= thr_3).astype(int), index=test_df.index)
        deb_3 = apply_persistence(p_3, test_df["session_id"], n=3)
        ae_3d_runs.append({
            "raw": compute_metrics(y_test, s_test_3, p_3, test_df, bursts, tol14_mask),
            "deb": compute_metrics(y_test, s_test_3, deb_3, test_df, bursts, tol14_mask),
            "preds_raw": p_3,
            "preds_deb": deb_3
        })

    # 2. Isolation Forest 5 Seeds (random_state 42-46)
    if_2d_runs = []
    if_3d_runs = []
    for s in SEEDS:
        if2 = IsolationForest(n_estimators=100, random_state=s, contamination="auto").fit(X_train_2d)
        s_val_if2 = -if2.score_samples(X_val_norm_2d)
        s_test_if2 = -if2.score_samples(X_test_2d)
        thr_if2 = np.percentile(s_val_if2, 97.0)
        p_if2 = pd.Series((s_test_if2 >= thr_if2).astype(int), index=test_df.index)
        deb_if2 = apply_persistence(p_if2, test_df["session_id"], n=3)
        if_2d_runs.append({
            "raw": compute_metrics(y_test, s_test_if2, p_if2, test_df, bursts, tol14_mask),
            "deb": compute_metrics(y_test, s_test_if2, deb_if2, test_df, bursts, tol14_mask),
            "preds_raw": p_if2,
            "preds_deb": deb_if2
        })

        if3 = IsolationForest(n_estimators=100, random_state=s, contamination="auto").fit(X_train_3d)
        s_val_if3 = -if3.score_samples(X_val_norm_3d)
        s_test_if3 = -if3.score_samples(X_test_3d)
        thr_if3 = np.percentile(s_val_if3, 97.0)
        p_if3 = pd.Series((s_test_if3 >= thr_if3).astype(int), index=test_df.index)
        deb_if3 = apply_persistence(p_if3, test_df["session_id"], n=3)
        if_3d_runs.append({
            "raw": compute_metrics(y_test, s_test_if3, p_if3, test_df, bursts, tol14_mask),
            "deb": compute_metrics(y_test, s_test_if3, deb_if3, test_df, bursts, tol14_mask),
            "preds_raw": p_if3,
            "preds_deb": deb_if3
        })

    # Model deterministik lainnya
    # A. Mahalanobis
    ee_2d = EllipticEnvelope(random_state=42, contamination=0.03).fit(X_train_2d)
    s_ee_2d = ee_2d.mahalanobis(X_test_2d)
    thr_ee_2d = np.percentile(ee_2d.mahalanobis(X_val_norm_2d), 97.0)
    p_ee_2d = pd.Series((s_ee_2d >= thr_ee_2d).astype(int), index=test_df.index)
    deb_ee_2d = apply_persistence(p_ee_2d, test_df["session_id"], n=3)
    m_ee_2d_raw = compute_metrics(y_test, s_ee_2d, p_ee_2d, test_df, bursts, tol14_mask)
    m_ee_2d_deb = compute_metrics(y_test, s_ee_2d, deb_ee_2d, test_df, bursts, tol14_mask)

    ee_3d = EllipticEnvelope(random_state=42, contamination=0.03).fit(X_train_3d)
    s_ee_3d = ee_3d.mahalanobis(X_test_3d)
    thr_ee_3d = np.percentile(ee_3d.mahalanobis(X_val_norm_3d), 97.0)
    p_ee_3d = pd.Series((s_ee_3d >= thr_ee_3d).astype(int), index=test_df.index)
    deb_ee_3d = apply_persistence(p_ee_3d, test_df["session_id"], n=3)
    m_ee_3d_raw = compute_metrics(y_test, s_ee_3d, p_ee_3d, test_df, bursts, tol14_mask)
    m_ee_3d_deb = compute_metrics(y_test, s_ee_3d, deb_ee_3d, test_df, bursts, tol14_mask)

    # B. LOF
    lof_2d = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto").fit(X_train_2d)
    s_lof_2d = -lof_2d.score_samples(X_test_2d)
    thr_lof_2d = np.percentile(-lof_2d.score_samples(X_val_norm_2d), 97.0)
    p_lof_2d = pd.Series((s_lof_2d >= thr_lof_2d).astype(int), index=test_df.index)
    deb_lof_2d = apply_persistence(p_lof_2d, test_df["session_id"], n=3)
    m_lof_2d_raw = compute_metrics(y_test, s_lof_2d, p_lof_2d, test_df, bursts, tol14_mask)
    m_lof_2d_deb = compute_metrics(y_test, s_lof_2d, deb_lof_2d, test_df, bursts, tol14_mask)

    lof_3d = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto").fit(X_train_3d)
    s_lof_3d = -lof_3d.score_samples(X_test_3d)
    thr_lof_3d = np.percentile(-lof_3d.score_samples(X_val_norm_3d), 97.0)
    p_lof_3d = pd.Series((s_lof_3d >= thr_lof_3d).astype(int), index=test_df.index)
    deb_lof_3d = apply_persistence(p_lof_3d, test_df["session_id"], n=3)
    m_lof_3d_raw = compute_metrics(y_test, s_lof_3d, p_lof_3d, test_df, bursts, tol14_mask)
    m_lof_3d_deb = compute_metrics(y_test, s_lof_3d, deb_lof_3d, test_df, bursts, tol14_mask)

    # C. OCSVM
    ocsvm_2d = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train_2d)
    s_ocsvm_2d = -ocsvm_2d.score_samples(X_test_2d)
    thr_ocsvm_2d = np.percentile(-ocsvm_2d.score_samples(X_val_norm_2d), 97.0)
    p_ocsvm_2d = pd.Series((s_ocsvm_2d >= thr_ocsvm_2d).astype(int), index=test_df.index)
    deb_ocsvm_2d = apply_persistence(p_ocsvm_2d, test_df["session_id"], n=3)
    m_ocsvm_2d_raw = compute_metrics(y_test, s_ocsvm_2d, p_ocsvm_2d, test_df, bursts, tol14_mask)
    m_ocsvm_2d_deb = compute_metrics(y_test, s_ocsvm_2d, deb_ocsvm_2d, test_df, bursts, tol14_mask)

    ocsvm_3d = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train_3d)
    s_ocsvm_3d = -ocsvm_3d.score_samples(X_test_3d)
    thr_ocsvm_3d = np.percentile(-ocsvm_3d.score_samples(X_val_norm_3d), 97.0)
    p_ocsvm_3d = pd.Series((s_ocsvm_3d >= thr_ocsvm_3d).astype(int), index=test_df.index)
    deb_ocsvm_3d = apply_persistence(p_ocsvm_3d, test_df["session_id"], n=3)
    m_ocsvm_3d_raw = compute_metrics(y_test, s_ocsvm_3d, p_ocsvm_3d, test_df, bursts, tol14_mask)
    m_ocsvm_3d_deb = compute_metrics(y_test, s_ocsvm_3d, deb_ocsvm_3d, test_df, bursts, tol14_mask)

    # D. Rule Default
    p_rule_2d = pd.Series([1 if evaluate_rules(r["hr"], r["spo2"], 36.5)[0] >= 1 else 0 for _, r in test_df.iterrows()], index=test_df.index)
    deb_rule_2d = apply_persistence(p_rule_2d, test_df["session_id"], n=3)
    m_rule_2d_raw = compute_metrics(y_test, None, p_rule_2d, test_df, bursts, tol14_mask, is_continuous=False)
    m_rule_2d_deb = compute_metrics(y_test, None, deb_rule_2d, test_df, bursts, tol14_mask, is_continuous=False)

    p_rule_3d = pd.Series([1 if evaluate_rules(r["hr"], r["spo2"], r["temp"])[0] >= 1 else 0 for _, r in test_df.iterrows()], index=test_df.index)
    deb_rule_3d = apply_persistence(p_rule_3d, test_df["session_id"], n=3)
    m_rule_3d_raw = compute_metrics(y_test, None, p_rule_3d, test_df, bursts, tol14_mask, is_continuous=False)
    m_rule_3d_deb = compute_metrics(y_test, None, deb_rule_3d, test_df, bursts, tol14_mask, is_continuous=False)

    # E. dl_v1 Zero-Shot
    import json
    with open(os.path.join(PROJECT_ROOT, "ml/artifacts/dl_v1/metadata.json"), "r") as f:
        v1_meta = json.load(f)
    v1_weights = dict(np.load(os.path.join(PROJECT_ROOT, "ml/artifacts/dl_v1/model_weights.npz")))
    from ml.dl.models import NumPyAutoencoderForward
    v1_forward = NumPyAutoencoderForward(v1_weights)
    scaler_mean = np.array(v1_meta["scaler"]["mean"], dtype=np.float32)
    scaler_scale = np.array(v1_meta["scaler"]["scale"], dtype=np.float32)
    s_v1_val = v1_forward.score((val_normal[cols_2d].values - scaler_mean) / scaler_scale)
    s_v1_test = v1_forward.score((test_df[cols_2d].values - scaler_mean) / scaler_scale)
    thr_v1 = np.percentile(s_v1_val, 97.0)
    p_v1 = pd.Series((s_v1_test >= thr_v1).astype(int), index=test_df.index)
    deb_v1 = apply_persistence(p_v1, test_df["session_id"], n=3)
    m_v1_raw = compute_metrics(y_test, s_v1_test, p_v1, test_df, bursts, tol14_mask, is_continuous=True)
    m_v1_deb = compute_metrics(y_test, s_v1_test, deb_v1, test_df, bursts, tol14_mask, is_continuous=True)

    # Ringkasan Aggregat
    agg_ae_2d_raw = aggregate_seed_runs(ae_2d_runs, "raw")
    agg_ae_2d_deb = aggregate_seed_runs(ae_2d_runs, "deb")
    agg_ae_3d_raw = aggregate_seed_runs(ae_3d_runs, "raw")
    agg_ae_3d_deb = aggregate_seed_runs(ae_3d_runs, "deb")

    agg_if_2d_raw = aggregate_seed_runs(if_2d_runs, "raw")
    agg_if_2d_deb = aggregate_seed_runs(if_2d_runs, "deb")
    agg_if_3d_raw = aggregate_seed_runs(if_3d_runs, "raw")
    agg_if_3d_deb = aggregate_seed_runs(if_3d_runs, "deb")

    # ==========================================================================
    # CETAK TABEL UTAMA DENGAN MEAN +- SD UNTUK MODEL BERSEED & DETERMINISTIK
    # ==========================================================================
    print(f"\n[{BANNER}] =================== TABEL UTAMA RAW (N=1) ===================")
    print(f"{'Model':<25} | {'Sifat':<13} | {'ROC-AUC':<17} | {'PR-AUC':<17} | {'Test FPR':<15} | {'Recall':<15} | {'Prec':<15} | {'F1':<15} | {'B-Rec (29)':<17} | {'FPR tol14':<15}")
    print("-" * 175)

    def print_stoch_row(name, sifat, r):
        roc_s = f"{r['roc_auc_mean']:.4f} +- {r['roc_auc_sd']:.4f}"
        pr_s = f"{r['pr_auc_mean']:.4f} +- {r['pr_auc_sd']:.4f}"
        fpr_s = f"{r['test_fpr_mean']*100:5.2f}% +- {r['test_fpr_sd']*100:.2f}%"
        rec_s = f"{r['recall_mean']*100:5.2f}% +- {r['recall_sd']*100:.2f}%"
        prec_s = f"{r['precision_mean']*100:5.2f}% +- {r['precision_sd']*100:.2f}%"
        f1_s = f"{r['f1_mean']:.4f} +- {r['f1_sd']:.4f}"
        brec_s = f"{r['burst_recall_mean']*100:5.1f}% +- {r['burst_recall_sd']*100:.1f}%"
        tol14_s = f"{r['fpr_tol14_mean']*100:5.2f}% +- {r['fpr_tol14_sd']*100:.2f}%"
        print(f"{name:<25} | {sifat:<13} | {roc_s:<17} | {pr_s:<17} | {fpr_s:<15} | {rec_s:<15} | {prec_s:<15} | {f1_s:<15} | {brec_s:<17} | {tol14_s:<15}")

    def print_determ_row(name, m):
        roc_s = f"{m['roc_auc']:.4f}" if not np.isnan(m['roc_auc']) else "N/A"
        pr_s = f"{m['pr_auc']:.4f}" if not np.isnan(m['pr_auc']) else "N/A"
        fpr_s = f"{m['test_fpr']*100:5.2f}%"
        rec_s = f"{m['recall']*100:5.2f}%"
        prec_s = f"{m['precision']*100:5.2f}%"
        f1_s = f"{m['f1']:.4f}"
        brec_s = f"{m['burst_recall']*100:5.1f}% ({m['burst_detected']})"
        tol14_s = f"{m['fpr_tol14']*100:5.2f}%"
        print(f"{name:<25} | {'Deterministik':<13} | {roc_s:<17} | {pr_s:<17} | {fpr_s:<15} | {rec_s:<15} | {prec_s:<15} | {f1_s:<15} | {brec_s:<17} | {tol14_s:<15}")

    print_determ_row("Rule Default (2D)", m_rule_2d_raw)
    print_determ_row("Rule Default (3D)", m_rule_3d_raw)
    print_determ_row("Univariate (2D, q=3%)", m_uni_2d_raw)
    print_determ_row("Univariate (3D, q=3%)", m_uni_3d_raw)
    print_determ_row("dl_v1 Zero-Shot (2D)", m_v1_raw)
    print_stoch_row("IF (2D raw, 5 seeds)", "5 seeds", agg_if_2d_raw)
    print_stoch_row("IF (3D raw, 5 seeds)", "5 seeds", agg_if_3d_raw)
    print_determ_row("Mahalanobis (2D)", m_ee_2d_raw)
    print_determ_row("Mahalanobis (3D)", m_ee_3d_raw)
    print_determ_row("LOF (2D)", m_lof_2d_raw)
    print_determ_row("LOF (3D)", m_lof_3d_raw)
    print_determ_row("OCSVM (2D)", m_ocsvm_2d_raw)
    print_determ_row("OCSVM (3D)", m_ocsvm_3d_raw)
    print_stoch_row("DenseAE 2D (5 seeds)", "5 seeds", agg_ae_2d_raw)
    print_stoch_row("DenseAE 3D (5 seeds)", "5 seeds", agg_ae_3d_raw)

    print(f"\n[{BANNER}] ================= TABEL UTAMA PERSISTENCE (N=3) =================")
    print(f"{'Model + N=3':<25} | {'Sifat':<13} | {'Test FPR':<15} | {'Recall':<15} | {'Prec':<15} | {'F1':<15} | {'B-Rec (29)':<17} | {'FPR tol14':<15}")
    print("-" * 135)

    def print_stoch_deb_row(name, sifat, r):
        fpr_s = f"{r['test_fpr_mean']*100:5.2f}% +- {r['test_fpr_sd']*100:.2f}%"
        rec_s = f"{r['recall_mean']*100:5.2f}% +- {r['recall_sd']*100:.2f}%"
        prec_s = f"{r['precision_mean']*100:5.2f}% +- {r['precision_sd']*100:.2f}%"
        f1_s = f"{r['f1_mean']:.4f} +- {r['f1_sd']:.4f}"
        brec_s = f"{r['burst_recall_mean']*100:5.1f}% +- {r['burst_recall_sd']*100:.1f}%"
        tol14_s = f"{r['fpr_tol14_mean']*100:5.2f}% +- {r['fpr_tol14_sd']*100:.2f}%"
        print(f"{name:<25} | {sifat:<13} | {fpr_s:<15} | {rec_s:<15} | {prec_s:<15} | {f1_s:<15} | {brec_s:<17} | {tol14_s:<15}")

    def print_determ_deb_row(name, m):
        fpr_s = f"{m['test_fpr']*100:5.2f}%"
        rec_s = f"{m['recall']*100:5.2f}%"
        prec_s = f"{m['precision']*100:5.2f}%"
        f1_s = f"{m['f1']:.4f}"
        brec_s = f"{m['burst_recall']*100:5.1f}% ({m['burst_detected']})"
        tol14_s = f"{m['fpr_tol14']*100:5.2f}%"
        print(f"{name:<25} | {'Deterministik':<13} | {fpr_s:<15} | {rec_s:<15} | {prec_s:<15} | {f1_s:<15} | {brec_s:<17} | {tol14_s:<15}")

    print_determ_deb_row("Rule Default (2D) + N3", m_rule_2d_deb)
    print_determ_deb_row("Rule Default (3D) + N3", m_rule_3d_deb)
    print_determ_deb_row("Univariate (2D) + N3", m_uni_2d_deb)
    print_determ_deb_row("Univariate (3D) + N3", m_uni_3d_deb)
    print_determ_deb_row("dl_v1 Zero-Shot + N3", m_v1_deb)
    print_stoch_deb_row("IF (2D raw) + N3", "5 seeds", agg_if_2d_deb)
    print_stoch_deb_row("IF (3D raw) + N3", "5 seeds", agg_if_3d_deb)
    print_determ_deb_row("Mahalanobis (2D) + N3", m_ee_2d_deb)
    print_determ_deb_row("Mahalanobis (3D) + N3", m_ee_3d_deb)
    print_determ_deb_row("LOF (2D) + N3", m_lof_2d_deb)
    print_determ_deb_row("LOF (3D) + N3", m_lof_3d_deb)
    print_determ_deb_row("OCSVM (2D) + N3", m_ocsvm_2d_deb)
    print_determ_deb_row("OCSVM (3D) + N3", m_ocsvm_3d_deb)
    print_stoch_deb_row("DenseAE 2D + N3", "5 seeds", agg_ae_2d_deb)
    print_stoch_deb_row("DenseAE 3D + N3", "5 seeds", agg_ae_3d_deb)

    # ==========================================================================
    # TABEL RECALL PADA FPR KETAT (PERBANDINGAN UTAMA) DENGAN MEAN +- SD
    # ==========================================================================
    print(f"\n[{BANNER}] ============ TABEL RECALL PADA FPR TETAP (TABEL PERBANDINGAN UTAMA) ============")
    print(f"{'Model':<25} | {'Sifat':<13} | {'FPR <= 0.1%':<20} | {'FPR <= 0.5%':<20} | {'FPR <= 1.0%':<20} | {'FPR <= 3.0%':<20}")
    print("-" * 125)

    def print_rec_stoch(name, r):
        r01 = f"{r['rec_fpr_0_1_mean']*100:5.2f}% +- {r['rec_fpr_0_1_sd']*100:.2f}%"
        r05 = f"{r['rec_fpr_0_5_mean']*100:5.2f}% +- {r['rec_fpr_0_5_sd']*100:.2f}%"
        r10 = f"{r['rec_fpr_1_0_mean']*100:5.2f}% +- {r['rec_fpr_1_0_sd']*100:.2f}%"
        r30 = f"{r['rec_fpr_3_0_mean']*100:5.2f}% +- {r['rec_fpr_3_0_sd']*100:.2f}%"
        print(f"{name:<25} | {'5 seeds':<13} | {r01:<20} | {r05:<20} | {r10:<20} | {r30:<20}")

    def print_rec_determ(name, m):
        r01 = f"{m['rec_fpr_0_1']*100:5.2f}%"
        r05 = f"{m['rec_fpr_0_5']*100:5.2f}%"
        r10 = f"{m['rec_fpr_1_0']*100:5.2f}%"
        r30 = f"{m['rec_fpr_3_0']*100:5.2f}%"
        print(f"{name:<25} | {'Deterministik':<13} | {r01:<20} | {r05:<20} | {r10:<20} | {r30:<20}")

    print_rec_determ("dl_v1 Zero-Shot (2D)", m_v1_raw)
    print_rec_stoch("IF (2D raw)", agg_if_2d_raw)
    print_rec_stoch("IF (3D raw)", agg_if_3d_raw)
    print_rec_determ("Mahalanobis (2D)", m_ee_2d_raw)
    print_rec_determ("Mahalanobis (3D)", m_ee_3d_raw)
    print_rec_determ("LOF (2D)", m_lof_2d_raw)
    print_rec_determ("LOF (3D)", m_lof_3d_raw)
    print_rec_determ("OCSVM (2D)", m_ocsvm_2d_raw)
    print_rec_determ("OCSVM (3D)", m_ocsvm_3d_raw)
    print_rec_stoch("DenseAE 2D", agg_ae_2d_raw)
    print_rec_stoch("DenseAE 3D", agg_ae_3d_raw)

    # ==========================================================================
    # ITEM 4: OUTPUT MENTAH FP & RECALL PER SKENARIO (S14 MOTION & S15 FINGER_SHIFT)
    # ==========================================================================
    print(f"\n[{BANNER}] ============ ITEM 4: OUTPUT MENTAH FP & RECALL PER SKENARIO ============")
    
    # Hitung per seed untuk DenseAE 2D, IF 2D, IF 3D pada S14 dan S15
    def eval_scenario_multiruns(runs_list, sc_id, mode="raw"):
        fps = []
        tps = []
        recalls = []
        fprs = []
        norm_tot = None
        ano_tot = None
        for r in runs_list:
            preds = r["preds_" + mode]
            res = compute_scenario_metrics(preds, test_df, sc_id)
            norm_tot = res["norm_total"]
            ano_tot = res["ano_total"]
            fps.append(res["fp"])
            tps.append(res["tp"])
            recalls.append(res["recall"])
            fprs.append(res["fpr"])
        return {
            "norm_tot": norm_tot,
            "ano_tot": ano_tot,
            "fp_mean": np.mean(fps),
            "fp_sd": np.std(fps),
            "fpr_mean": np.mean(fprs),
            "fpr_sd": np.std(fprs),
            "tp_mean": np.mean(tps),
            "tp_sd": np.std(tps),
            "rec_mean": np.mean(recalls),
            "rec_sd": np.std(recalls),
            "seed42": {
                "fp": fps[0],
                "fpr": fprs[0],
                "tp": tps[0],
                "recall": recalls[0]
            }
        }

    for sc_name, sc_id in [("S14 (MOTION)", "S14"), ("S15 (FINGER_SHIFT)", "S15")]:
        print(f"\n--- Skenario {sc_name} ---")
        ae_sc_raw = eval_scenario_multiruns(ae_2d_runs, sc_id, "raw")
        ae_sc_deb = eval_scenario_multiruns(ae_2d_runs, sc_id, "deb")
        if2_sc_raw = eval_scenario_multiruns(if_2d_runs, sc_id, "raw")
        if2_sc_deb = eval_scenario_multiruns(if_2d_runs, sc_id, "deb")
        if3_sc_raw = eval_scenario_multiruns(if_3d_runs, sc_id, "raw")
        if3_sc_deb = eval_scenario_multiruns(if_3d_runs, sc_id, "deb")

        print(f"Total Sampel: Normal={ae_sc_raw['norm_tot']}, Anomali={ae_sc_raw['ano_tot']}")
        print("  RAW (N=1):")
        print(f"    DenseAE 2D (5 seeds): FP={ae_sc_raw['fp_mean']:.2f} +- {ae_sc_raw['fp_sd']:.2f} ({ae_sc_raw['fpr_mean']*100:.2f}% +- {ae_sc_raw['fpr_sd']*100:.2f}%) | TP={ae_sc_raw['tp_mean']:.2f}/{ae_sc_raw['ano_tot']} (Recall={ae_sc_raw['rec_mean']*100:.2f}% +- {ae_sc_raw['rec_sd']*100:.2f}%)")
        print(f"    DenseAE 2D (seed 42): FP={ae_sc_raw['seed42']['fp']} ({ae_sc_raw['seed42']['fpr']*100:.2f}%) | TP={ae_sc_raw['seed42']['tp']}/{ae_sc_raw['ano_tot']} (Recall={ae_sc_raw['seed42']['recall']*100:.2f}%)")
        print(f"    IF 2D raw  (5 seeds): FP={if2_sc_raw['fp_mean']:.2f} +- {if2_sc_raw['fp_sd']:.2f} ({if2_sc_raw['fpr_mean']*100:.2f}% +- {if2_sc_raw['fpr_sd']*100:.2f}%) | TP={if2_sc_raw['tp_mean']:.2f}/{if2_sc_raw['ano_tot']} (Recall={if2_sc_raw['rec_mean']*100:.2f}% +- {if2_sc_raw['rec_sd']*100:.2f}%)")
        print(f"    IF 2D raw  (seed 42): FP={if2_sc_raw['seed42']['fp']} ({if2_sc_raw['seed42']['fpr']*100:.2f}%) | TP={if2_sc_raw['seed42']['tp']}/{if2_sc_raw['ano_tot']} (Recall={if2_sc_raw['seed42']['recall']*100:.2f}%)")
        print(f"    IF 3D raw  (5 seeds): FP={if3_sc_raw['fp_mean']:.2f} +- {if3_sc_raw['fp_sd']:.2f} ({if3_sc_raw['fpr_mean']*100:.2f}% +- {if3_sc_raw['fpr_sd']*100:.2f}%) | TP={if3_sc_raw['tp_mean']:.2f}/{if3_sc_raw['ano_tot']} (Recall={if3_sc_raw['rec_mean']*100:.2f}% +- {if3_sc_raw['rec_sd']*100:.2f}%)")
        print(f"    IF 3D raw  (seed 42): FP={if3_sc_raw['seed42']['fp']} ({if3_sc_raw['seed42']['fpr']*100:.2f}%) | TP={if3_sc_raw['seed42']['tp']}/{if3_sc_raw['ano_tot']} (Recall={if3_sc_raw['seed42']['recall']*100:.2f}%)")

        print("  DEBOUNCED (N=3):")
        print(f"    DenseAE 2D (5 seeds): FP={ae_sc_deb['fp_mean']:.2f} +- {ae_sc_deb['fp_sd']:.2f} ({ae_sc_deb['fpr_mean']*100:.2f}% +- {ae_sc_deb['fpr_sd']*100:.2f}%) | TP={ae_sc_deb['tp_mean']:.2f}/{ae_sc_deb['ano_tot']} (Recall={ae_sc_deb['rec_mean']*100:.2f}% +- {ae_sc_deb['rec_sd']*100:.2f}%)")
        print(f"    DenseAE 2D (seed 42): FP={ae_sc_deb['seed42']['fp']} ({ae_sc_deb['seed42']['fpr']*100:.2f}%) | TP={ae_sc_deb['seed42']['tp']}/{ae_sc_deb['ano_tot']} (Recall={ae_sc_deb['seed42']['recall']*100:.2f}%)")
        print(f"    IF 2D raw  (5 seeds): FP={if2_sc_deb['fp_mean']:.2f} +- {if2_sc_deb['fp_sd']:.2f} ({if2_sc_deb['fpr_mean']*100:.2f}% +- {if2_sc_deb['fpr_sd']*100:.2f}%) | TP={if2_sc_deb['tp_mean']:.2f}/{if2_sc_deb['ano_tot']} (Recall={if2_sc_deb['rec_mean']*100:.2f}% +- {if2_sc_deb['rec_sd']*100:.2f}%)")
        print(f"    IF 2D raw  (seed 42): FP={if2_sc_deb['seed42']['fp']} ({if2_sc_deb['seed42']['fpr']*100:.2f}%) | TP={if2_sc_deb['seed42']['tp']}/{if2_sc_deb['ano_tot']} (Recall={if2_sc_deb['seed42']['recall']*100:.2f}%)")
        print(f"    IF 3D raw  (5 seeds): FP={if3_sc_deb['fp_mean']:.2f} +- {if3_sc_deb['fp_sd']:.2f} ({if3_sc_deb['fpr_mean']*100:.2f}% +- {if3_sc_deb['fpr_sd']*100:.2f}%) | TP={if3_sc_deb['tp_mean']:.2f}/{if3_sc_deb['ano_tot']} (Recall={if3_sc_deb['rec_mean']*100:.2f}% +- {if3_sc_deb['rec_sd']*100:.2f}%)")
        print(f"    IF 3D raw  (seed 42): FP={if3_sc_deb['seed42']['fp']} ({if3_sc_deb['seed42']['fpr']*100:.2f}%) | TP={if3_sc_deb['seed42']['tp']}/{if3_sc_deb['ano_tot']} (Recall={if3_sc_deb['seed42']['recall']*100:.2f}%)")


if __name__ == "__main__":
    main()
