import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURE_COLS = [
    'Heart_Rate', 'SpO2', 'Temperature',
    'Heart_Rate_Delta', 'SpO2_Delta', 'Temperature_Delta',
    'MA_Heart_Rate', 'MA_SpO2', 'MA_Temperature',
    'Var_Heart_Rate', 'Var_SpO2', 'Var_Temperature',
    'Rate_of_Change'
]

def calc_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0))
    }

def calibrate_threshold(scores_normal: np.ndarray, target_fpr: float) -> float:
    """
    Cari threshold skor anomali sehingga persentase sampel NORMAL
    yang salah ditandai anomali == target_fpr.
    
    Isolation Forest decision_function: skor rendah = lebih anomali.
    FPR = fraksi sampel normal yang skor-nya < threshold.
    """
    # Percentile dari bawah: target_fpr persen dari normal harus di bawah threshold
    threshold = np.percentile(scores_normal, target_fpr * 100)
    return threshold

def predict_with_threshold(model, X: pd.DataFrame, threshold: float) -> np.ndarray:
    """Prediksi anomali: skor < threshold -> anomali (1), else normal (0)."""
    scores = model.decision_function(X)
    return np.where(scores < threshold, 1, 0)

def main(train_path: str, val_path: str, test_path: str, model_path: str,
         report_path: str, n_estimators: int):

    logging.info(f"Memuat data...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # --- Training: hanya pola NORMAL dari dataset tepercaya (tanpa rishanmascarenhas) ---
    train_normal_mask = (
        ((df_train['Source'] == 'nasirayub2') & (df_train['OUTPUT'] == 0)) |
        ((df_train['Source'] == 'engrarri21') & (df_train['OUTPUT'] == 0)) |
        ((df_train['Source'] == 'gourangomandal') & (df_train['OUTPUT'] == 0))
    )
    X_train = df_train.loc[train_normal_mask, FEATURE_COLS]
    logging.info(f"Training samples (normal only): {X_train.shape[0]:,}")

    # --- Train Isolation Forest ---
    # contamination rendah (0.01) agar model belajar profil "normal" secara ketat.
    # Keputusan anomali ditentukan oleh threshold yang dikalibrasi, bukan contamination.
    clf = IsolationForest(
        n_estimators=n_estimators, contamination=0.01,
        max_samples='auto', random_state=42, n_jobs=-1
    )
    clf.fit(X_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    logging.info(f"Model disimpan di {model_path}")

    # ==================================================
    # TAHAP KALIBRASI: pada Validation Set
    # ==================================================
    X_val = df_val[FEATURE_COLS]
    y_val = df_val['OUTPUT'].values
    val_normal_mask = y_val == 0
    val_scores = clf.decision_function(X_val)
    scores_val_normal = val_scores[val_normal_mask]

    logging.info(f"Validation set: {len(y_val):,} total, {val_normal_mask.sum():,} normal, {(~val_normal_mask).sum():,} abnormal")

    target_fprs = [0.05, 0.10, 0.13, 0.15, 0.20]
    thresholds = {}
    val_results = []

    for fpr_target in target_fprs:
        thr = calibrate_threshold(scores_val_normal, fpr_target)
        thresholds[fpr_target] = thr
        pred_val = np.where(val_scores < thr, 1, 0)

        actual_fpr = pred_val[val_normal_mask].mean()
        recall_val = recall_score(y_val, pred_val, zero_division=0)
        prec_val = precision_score(y_val, pred_val, zero_division=0)
        f1_val = f1_score(y_val, pred_val, zero_division=0)

        val_results.append({
            'Target_FPR': f"{fpr_target*100:.0f}%",
            'Actual_FPR_Val': f"{actual_fpr*100:.1f}%",
            'Threshold': f"{thr:.6f}",
            'Recall_Val': f"{recall_val:.4f}",
            'Precision_Val': f"{prec_val:.4f}",
            'F1_Val': f"{f1_val:.4f}"
        })
        logging.info(f"  FPR target={fpr_target*100:.0f}%: threshold={thr:.4f}, actual_FPR={actual_fpr:.4f}, Recall={recall_val:.4f}")

    # ==================================================
    # TAHAP EVALUASI FINAL: pada Test Set (belum pernah disentuh)
    # ==================================================
    X_test = df_test[FEATURE_COLS]
    y_test = df_test['OUTPUT'].values
    test_normal_mask = y_test == 0
    test_scores = clf.decision_function(X_test)

    logging.info(f"Test set: {len(y_test):,} total, {test_normal_mask.sum():,} normal, {(~test_normal_mask).sum():,} abnormal")

    test_results = []
    for fpr_target in target_fprs:
        thr = thresholds[fpr_target]
        pred_test = np.where(test_scores < thr, 1, 0)

        actual_fpr_test = pred_test[test_normal_mask].mean()
        recall_test = recall_score(y_test, pred_test, zero_division=0)
        prec_test = precision_score(y_test, pred_test, zero_division=0)
        f1_test = f1_score(y_test, pred_test, zero_division=0)
        cm = confusion_matrix(y_test, pred_test)

        test_results.append({
            'Target_FPR': f"{fpr_target*100:.0f}%",
            'Actual_FPR_Test': f"{actual_fpr_test*100:.1f}%",
            'Threshold': thr,
            'Recall_Test': recall_test,
            'Precision_Test': prec_test,
            'F1_Test': f1_test,
            'CM': cm
        })

    # --- Per-source breakdown pada test set (menggunakan FPR 10% sebagai contoh) ---
    mid_thr = thresholds[0.10]
    mid_pred = np.where(test_scores < mid_thr, 1, 0)

    # ==================================================
    # TULIS LAPORAN
    # ==================================================
    report = "=" * 65 + "\n"
    report += "LAPORAN EVALUASI v3 — THRESHOLD CALIBRATION (3-PARTISI)\n"
    report += "=" * 65 + "\n\n"

    report += "METODOLOGI:\n"
    report += "  - Data dibagi 3 partisi per pasien/sesi: Train(60%) / Val(20%) / Test(20%)\n"
    report += "  - Model dilatih pada sampel NORMAL dari Train set (contamination='auto')\n"
    report += "  - Threshold dikalibrasi pada sampel NORMAL di Validation set\n"
    report += "    berdasarkan target False Positive Rate (FPR)\n"
    report += "  - Evaluasi final dilakukan pada Test set yang BELUM PERNAH disentuh\n"
    report += "    untuk training maupun tuning\n\n"

    report += f"  Train samples (normal): {X_train.shape[0]:,}\n"
    report += f"  Val samples: {len(y_val):,} (Norm: {val_normal_mask.sum():,}, Abn: {(~val_normal_mask).sum():,})\n"
    report += f"  Test samples: {len(y_test):,} (Norm: {test_normal_mask.sum():,}, Abn: {(~test_normal_mask).sum():,})\n\n"

    report += "-" * 65 + "\n"
    report += "KALIBRASI THRESHOLD (pada Validation Set — sampel NORMAL)\n"
    report += "-" * 65 + "\n"
    report += pd.DataFrame(val_results).to_string(index=False) + "\n\n"

    report += "-" * 65 + "\n"
    report += "EVALUASI FINAL PADA TEST SET (belum pernah disentuh)\n"
    report += "-" * 65 + "\n\n"

    for r in test_results:
        report += f"Target FPR = {r['Target_FPR']}:\n"
        report += f"  Actual FPR (test normal) : {r['Actual_FPR_Test']}\n"
        report += f"  Precision                : {r['Precision_Test']:.4f}\n"
        report += f"  Recall                   : {r['Recall_Test']:.4f}\n"
        report += f"  F1-Score                 : {r['F1_Test']:.4f}\n"
        report += f"  Confusion Matrix:\n"
        report += f"    {r['CM']}\n\n"

    report += "-" * 65 + "\n"
    report += "TABEL RINGKASAN — PILIHAN UNTUK REVIEWER\n"
    report += "-" * 65 + "\n"
    report += f"{'Target FPR':>12} | {'FPR Aktual (Test)':>18} | {'Recall (Test)':>14} | {'Precision':>10} | {'F1':>8}\n"
    report += "-" * 65 + "\n"
    for r in test_results:
        report += f"{r['Target_FPR']:>12} | {r['Actual_FPR_Test']:>18} | {r['Recall_Test']:>14.4f} | {r['Precision_Test']:>10.4f} | {r['F1_Test']:>8.4f}\n"
    report += "\n"

    report += "-" * 65 + "\n"
    report += "BREAKDOWN PER SUMBER (Test Set, threshold FPR=10%)\n"
    report += "-" * 65 + "\n"
    for source in df_test['Source'].unique():
        mask = df_test['Source'] == source
        if mask.sum() == 0:
            continue
        y_s = y_test[mask]
        p_s = mid_pred[mask]
        m_s = calc_metrics(y_s, p_s)
        n_s = (y_s == 0).sum()
        a_s = (y_s == 1).sum()
        fpr_s = p_s[y_s == 0].mean() if n_s > 0 else 0
        report += f"  {source}: N={mask.sum():,} (Norm={n_s:,}, Abn={a_s:,})"
        report += f" -> FPR={fpr_s:.2%} P={m_s['Precision']:.4f} R={m_s['Recall']:.4f} F1={m_s['F1']:.4f}\n"

    report += "\n"
    report += "-" * 65 + "\n"
    report += "PERBANDINGAN HISTORIS\n"
    report += "-" * 65 + "\n"
    report += "  v1 (ada overlap, c=0.08):  P=0.4806 R=0.1723 F1=0.2536\n"
    report += "  v2 (holdout, c=0.40):      P=0.7233 R=0.7234 F1=0.7234 (FPR~37%)\n"
    report += "  v3 (3-partisi, FPR=10%):   lihat tabel di atas\n"

    # Simpan threshold yang dipilih bersama model
    threshold_path = os.path.join(os.path.dirname(model_path), 'thresholds.joblib')
    joblib.dump(thresholds, threshold_path)
    logging.info(f"Threshold disimpan di {threshold_path}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logging.info(f"Laporan evaluasi disimpan di {report_path}")
    print("\n" + report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IF & Calibrate Threshold on Validation Set")
    basedir = os.path.dirname(__file__)
    default_train = os.path.abspath(os.path.join(basedir, '..', 'data', 'features_train.csv'))
    default_val = os.path.abspath(os.path.join(basedir, '..', 'data', 'features_val.csv'))
    default_test = os.path.abspath(os.path.join(basedir, '..', 'data', 'features_test.csv'))
    default_model = os.path.abspath(os.path.join(basedir, '..', 'models', 'isolation_forest_model.joblib'))
    default_report = os.path.abspath(os.path.join(basedir, '..', 'models', 'evaluation_report.txt'))

    parser.add_argument('--train', type=str, default=default_train)
    parser.add_argument('--val', type=str, default=default_val)
    parser.add_argument('--test', type=str, default=default_test)
    parser.add_argument('--model', type=str, default=default_model)
    parser.add_argument('--report', type=str, default=default_report)
    parser.add_argument('--n_estimators', type=int, default=150)

    args = parser.parse_args()
    main(args.train, args.val, args.test, args.model, args.report, args.n_estimators)
