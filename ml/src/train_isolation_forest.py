import pandas as pd
import numpy as np
import argparse
import os
import json
import logging
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 13 Fitur fisiologis murni (Heart_Rate, SpO2, Temperature dan turunan temporalnya).
# Parameter lingkungan (humidity, room_temp) bersifat kontekstual/pendukung di backend.
FEATURE_COLS = [
    'Heart_Rate', 'SpO2', 'Temperature',
    'Heart_Rate_Delta', 'SpO2_Delta', 'Temperature_Delta',
    'MA_Heart_Rate', 'MA_SpO2', 'MA_Temperature',
    'Var_Heart_Rate', 'Var_SpO2', 'Var_Temperature',
    'Rate_of_Change'
]

def calc_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Hitung metrik evaluasi lengkap untuk klasifikasi biner anomali."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    b_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        'FPR': fpr,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': acc,
        'Balanced_Accuracy': b_acc,
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'CM': cm.tolist()
    }

def calibrate_fpr_threshold(scores_normal: np.ndarray, target_fpr: float) -> float:
    """
    Cari threshold decision_function sedemikian hingga persentase sampel normal
    yang salah ditandai anomali (score < threshold) == target_fpr.
    """
    return float(np.percentile(scores_normal, target_fpr * 100))

def find_optimal_thresholds(y_val: np.ndarray, val_scores: np.ndarray) -> Dict[str, float]:
    """
    Mencari threshold optimal pada Validation Set:
    1. Youden's J Index: Maximize (Sensitivity + Specificity - 1) = (TPR - FPR)
    2. Best F1-Score: Maximize Harmonic Mean of Precision and Recall
    """
    # Anomaly score (semakin tinggi = semakin anomali)
    anomaly_scores = -val_scores

    # 1. Youden's J
    fprs, tprs, roc_thrs = roc_curve(y_val, anomaly_scores)
    j_scores = tprs - fprs
    best_j_idx = int(np.argmax(j_scores))
    best_j_tau_anomaly = float(roc_thrs[best_j_idx])
    tau_youden_j = -best_j_tau_anomaly

    # 2. Best F1-Score
    precisions, recalls, pr_thrs = precision_recall_curve(y_val, anomaly_scores)
    # Exclude boundary terakhir karena pr_thrs berukuran n, sedangkan precisions & recalls n+1
    p_sub = precisions[:-1]
    r_sub = recalls[:-1]
    denom = p_sub + r_sub
    f1_scores = np.where(denom > 0, 2 * p_sub * r_sub / denom, 0.0)
    best_f1_idx = int(np.argmax(f1_scores))
    best_f1_tau_anomaly = float(pr_thrs[best_f1_idx])
    tau_best_f1 = -best_f1_tau_anomaly

    return {
        'tau_youden_j': tau_youden_j,
        'youden_j_val_score': float(j_scores[best_j_idx]),
        'tau_best_f1': tau_best_f1,
        'best_f1_val_score': float(f1_scores[best_f1_idx])
    }

def main(train_path: str, val_path: str, test_path: str, model_path: str,
         report_path: str, n_estimators: int):

    logging.info(f"Memuat data fitur fisiologis...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Latih Isolation Forest HANYA pada data Training Normal (OUTPUT == 0)
    train_normal_mask = df_train['OUTPUT'] == 0
    X_train = df_train.loc[train_normal_mask, FEATURE_COLS]
    logging.info(f"Training samples (normal only): {X_train.shape[0]:,} dari total {len(df_train):,}")

    logging.info(f"Melatih Isolation Forest (n_estimators={n_estimators}, max_samples='auto', random_state=42)...")
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples='auto',
        random_state=42,
        contamination='auto',
        n_jobs=-1
    )
    clf.fit(X_train)

    # Simpan model artefak
    models_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf, model_path)
    
    # Simpan juga dengan nama isolation_forest.joblib
    alt_model_path = os.path.join(models_dir, 'isolation_forest.joblib')
    joblib.dump(clf, alt_model_path)
    logging.info(f"Model disimpan di {model_path} dan {alt_model_path}")

    # ==========================================
    # VALIDATION SET EVALUATION & CALIBRATION
    # ==========================================
    X_val = df_val[FEATURE_COLS]
    y_val = df_val['OUTPUT'].values
    val_normal_mask = y_val == 0
    val_scores = clf.decision_function(X_val)
    scores_val_normal = val_scores[val_normal_mask]

    roc_val = float(roc_auc_score(y_val, -val_scores))
    pr_val = float(average_precision_score(y_val, -val_scores))

    logging.info(f"Validation set: {len(y_val):,} total (Normal: {val_normal_mask.sum():,}, Anomaly: {(~val_normal_mask).sum():,})")
    logging.info(f"Validation Area Under Curve: ROC-AUC={roc_val:.4f}, PR-AUC={pr_val:.4f}")

    # 1. Target FPR Quantile Thresholds
    target_fprs = [0.01, 0.03, 0.05, 0.10, 0.13, 0.15, 0.20]
    fpr_thresholds = {fpr: calibrate_fpr_threshold(scores_val_normal, fpr) for fpr in target_fprs}

    # 2. Optimal Thresholds (Youden's J & Best F1)
    opt_dict = find_optimal_thresholds(y_val, val_scores)
    tau_youden_j = opt_dict['tau_youden_j']
    tau_best_f1 = opt_dict['tau_best_f1']

    logging.info(f"Threshold Youden's J: {tau_youden_j:.6f} (Val J-Score: {opt_dict['youden_j_val_score']:.4f})")
    logging.info(f"Threshold Best F1   : {tau_best_f1:.6f} (Val F1-Score: {opt_dict['best_f1_val_score']:.4f})")

    # Kumpulan seluruh threshold kandidat
    eval_thresholds: Dict[str, float] = {
        'Youden_J (Optimal)': tau_youden_j,
        'Best_F1 (Optimal)': tau_best_f1,
        'Target_FPR_5%': fpr_thresholds[0.05],
        'Target_FPR_10%': fpr_thresholds[0.10],
        'Default_0.0': 0.0
    }
    # Tambahkan FPR targets lainnya
    for fpr in target_fprs:
        key = f"Target_FPR_{int(fpr*100)}%"
        if key not in eval_thresholds:
            eval_thresholds[key] = fpr_thresholds[fpr]

    # Evaluasi Validation Set untuk setiap threshold
    val_eval_summary = []
    for name, tau in eval_thresholds.items():
        pred_val = np.where(val_scores < tau, 1, 0)
        m = calc_detailed_metrics(y_val, pred_val)
        val_eval_summary.append({
            'Strategy': name,
            'Threshold': tau,
            'FPR': m['FPR'],
            'Recall': m['Recall'],
            'Precision': m['Precision'],
            'F1': m['F1'],
            'Accuracy': m['Accuracy'],
            'Balanced_Acc': m['Balanced_Accuracy']
        })

    # ==========================================
    # TEST SET EVALUATION (Unseen Data)
    # ==========================================
    X_test = df_test[FEATURE_COLS]
    y_test = df_test['OUTPUT'].values
    test_normal_mask = y_test == 0
    test_scores = clf.decision_function(X_test)

    roc_test = float(roc_auc_score(y_test, -test_scores))
    pr_test = float(average_precision_score(y_test, -test_scores))

    logging.info(f"Test set: {len(y_test):,} total (Normal: {test_normal_mask.sum():,}, Anomaly: {(~test_normal_mask).sum():,})")
    logging.info(f"Test Area Under Curve: ROC-AUC={roc_test:.4f}, PR-AUC={pr_test:.4f}")

    test_eval_summary = []
    test_metrics_by_tau = {}
    for name, tau in eval_thresholds.items():
        pred_test = np.where(test_scores < tau, 1, 0)
        m = calc_detailed_metrics(y_test, pred_test)
        test_metrics_by_tau[name] = m
        test_eval_summary.append({
            'Strategy': name,
            'Threshold': tau,
            'Actual_FPR': m['FPR'],
            'Recall': m['Recall'],
            'Precision': m['Precision'],
            'F1': m['F1'],
            'Accuracy': m['Accuracy'],
            'Balanced_Acc': m['Balanced_Accuracy'],
            'TN': m['TN'],
            'FP': m['FP'],
            'FN': m['FN'],
            'TP': m['TP']
        })

    # Breakdown per Sumber Data pada Test Set
    source_breakdowns = {}
    key_strategies = ['Youden_J (Optimal)', 'Best_F1 (Optimal)', 'Target_FPR_10%', 'Default_0.0']
    
    for strat in key_strategies:
        tau = eval_thresholds[strat]
        pred_test = np.where(test_scores < tau, 1, 0)
        source_breakdowns[strat] = {}
        for source in df_test['Source'].unique():
            src_mask = df_test['Source'] == source
            y_s = y_test[src_mask]
            p_s = pred_test[src_mask]
            m_s = calc_detailed_metrics(y_s, p_s)
            source_breakdowns[strat][source] = {
                'Total': int(src_mask.sum()),
                'Normal': int((y_s == 0).sum()),
                'Anomaly': int((y_s == 1).sum()),
                'Metrics': m_s
            }

    # ==========================================
    # SAVE ARTIFACTS: thresholds.joblib & config.json
    # ==========================================
    # Dictionary thresholds untuk backend (kompatibel dengan float & string keys)
    thresholds_dict: Dict[Any, Any] = {
        # Float target FPR keys (backward compatibility)
        0.01: fpr_thresholds[0.01],
        0.03: fpr_thresholds[0.03],
        0.05: fpr_thresholds[0.05],
        0.10: fpr_thresholds[0.10],
        0.13: fpr_thresholds[0.13],
        0.15: fpr_thresholds[0.15],
        0.20: fpr_thresholds[0.20],
        # String keys
        'fpr_01pct': fpr_thresholds[0.01],
        'fpr_05pct': fpr_thresholds[0.05],
        'fpr_10pct': fpr_thresholds[0.10],
        'fpr_13pct': fpr_thresholds[0.13],
        'youden_j': tau_youden_j,
        'best_f1': tau_best_f1,
        'default': 0.0,
        # Operational thresholds for runtime backend
        'optimal_decision_threshold': tau_youden_j,
        'warning_threshold': tau_youden_j,
        'critical_threshold': fpr_thresholds[0.05],
    }
    threshold_path = os.path.join(models_dir, 'thresholds.joblib')
    joblib.dump(thresholds_dict, threshold_path)
    logging.info(f"Thresholds joblib disimpan di {threshold_path}")

    # Simpan juga metadata config JSON
    config_metadata = {
        'model_name': 'IsolationForest',
        'n_estimators': n_estimators,
        'max_samples': 'auto',
        'random_state': 42,
        'contamination': 'auto',
        'training_scope': 'Normal data only (OUTPUT == 0)',
        'num_train_samples': int(X_train.shape[0]),
        'num_features': len(FEATURE_COLS),
        'features': FEATURE_COLS,
        'scaling': {
            'type': 'BoundedPhysiologicalMinMaxScaler',
            'scaler_file': 'ml/models/scaler.joblib',
            'fixed_bounds': {
                'Heart_Rate': [40.0, 160.0],
                'SpO2': [70.0, 100.0],
                'Temperature': [30.0, 43.0],
                'Heart_Rate_Delta': [-40.0, 40.0],
                'SpO2_Delta': [-15.0, 15.0],
                'Temperature_Delta': [-4.0, 4.0],
                'Rate_of_Change': [0.0, 50.0]
            }
        },
        'calibrated_thresholds': {
            'youden_j': tau_youden_j,
            'best_f1': tau_best_f1,
            'target_fpr_05pct': fpr_thresholds[0.05],
            'target_fpr_10pct': fpr_thresholds[0.10],
            'default_0_0': 0.0
        },
        'test_metrics_summary': {
            'roc_auc': roc_test,
            'pr_auc': pr_test,
            'youden_j_test': test_metrics_by_tau['Youden_J (Optimal)'],
            'best_f1_test': test_metrics_by_tau['Best_F1 (Optimal)'],
            'default_test': test_metrics_by_tau['Default_0.0']
        }
    }
    config_json_path = os.path.join(models_dir, 'config.json')
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_metadata, f, indent=2)
    logging.info(f"Metadata config disimpan di {config_json_path}")

    # ==========================================
    # BUILD TEXT & JSON EVALUATION REPORT
    # ==========================================
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LAPORAN EVALUASI MODEL ISOLATION FOREST — THRESHOLD CALIBRATION")
    report_lines.append("=" * 80)
    report_lines.append("\n1. METODOLOGI & ARSITEKTUR PELATIHAN:")
    report_lines.append(f"  - Dataset Bersih: features_train.csv (tanpa nasirayub2, bounded physiological scaling)")
    report_lines.append(f"  - Scope Pelatihan: HANYA Data Normal (OUTPUT == 0) -> One-Class Baseline")
    report_lines.append(f"  - Jumlah Sampel Training: {X_train.shape[0]:,} normal")
    report_lines.append(f"  - Parameter Model: n_estimators={n_estimators}, max_samples='auto', random_state=42")
    report_lines.append(f"  - Validation Set: {len(y_val):,} sampel (Normal: {val_normal_mask.sum():,}, Anomaly: {(~val_normal_mask).sum():,})")
    report_lines.append(f"  - Test Set (Unseen): {len(y_test):,} sampel (Normal: {test_normal_mask.sum():,}, Anomaly: {(~test_normal_mask).sum():,})")
    report_lines.append(f"  - Test Discriminative Power: ROC-AUC = {roc_test:.4f} | PR-AUC = {pr_test:.4f}")

    report_lines.append("\n" + "-" * 80)
    report_lines.append("2. KALIBRASI THRESHOLD PADA VALIDATION SET")
    report_lines.append("-" * 80)
    df_val_res = pd.DataFrame(val_eval_summary)
    report_lines.append(df_val_res.to_string(index=False, justify='left'))

    report_lines.append("\n" + "-" * 80)
    report_lines.append("3. EVALUASI FINAL PADA TEST SET (16,305 SAMPEL UNSEEN)")
    report_lines.append("-" * 80)
    df_test_res = pd.DataFrame(test_eval_summary)
    report_lines.append(df_test_res.to_string(index=False, justify='left'))

    report_lines.append("\n" + "-" * 80)
    report_lines.append("4. CONFUSION MATRIX & DETAIL PADA STRATEGI THRESHOLD UTAMA (TEST SET)")
    report_lines.append("-" * 80)
    for strat in ['Youden_J (Optimal)', 'Best_F1 (Optimal)', 'Default_0.0']:
        m = test_metrics_by_tau[strat]
        tau = eval_thresholds[strat]
        report_lines.append(f"\n[Strategi: {strat} | Threshold tau = {tau:+.6f}]")
        report_lines.append(f"  Confusion Matrix : [TN={m['TN']:,}, FP={m['FP']:,}, FN={m['FN']:,}, TP={m['TP']:,}]")
        report_lines.append(f"  Actual FPR       : {m['FPR']*100:.2f}% (FP={m['FP']:,} dari {test_normal_mask.sum():,} normal)")
        report_lines.append(f"  Recall (Sens)    : {m['Recall']*100:.2f}% (TP={m['TP']:,} dari {(~test_normal_mask).sum():,} anomali)")
        report_lines.append(f"  Precision        : {m['Precision']*100:.2f}%")
        report_lines.append(f"  F1-Score         : {m['F1']*100:.2f}%")
        report_lines.append(f"  Accuracy         : {m['Accuracy']*100:.2f}%")
        report_lines.append(f"  Balanced Acc     : {m['Balanced_Accuracy']*100:.2f}%")

    report_lines.append("\n" + "-" * 80)
    report_lines.append("5. BREAKDOWN PER SUMBER DATA (engrarri21 vs gourangomandal) PADA TEST SET")
    report_lines.append("-" * 80)
    for strat in ['Youden_J (Optimal)', 'Best_F1 (Optimal)', 'Default_0.0']:
        tau = eval_thresholds[strat]
        report_lines.append(f"\nStrategi: {strat} (tau = {tau:+.6f})")
        for src, data in source_breakdowns[strat].items():
            m_s = data['Metrics']
            report_lines.append(
                f"  {src:<16}: N={data['Total']:,} (Norm={data['Normal']:,}, Abn={data['Anomaly']:,}) "
                f"| FPR={m_s['FPR']*100:5.2f}% | Recall={m_s['Recall']*100:5.2f}% | "
                f"Precision={m_s['Precision']*100:5.2f}% | F1={m_s['F1']*100:5.2f}% | "
                f"CM=[TN={m_s['TN']}, FP={m_s['FP']}, FN={m_s['FN']}, TP={m_s['TP']}]"
            )

    report_lines.append("\n" + "=" * 80)
    report_text = "\n".join(report_lines)

    # Simpan file laporan
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logging.info(f"Laporan evaluasi teks disimpan di {report_path}")

    json_report_path = os.path.splitext(report_path)[0] + '.json'
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'val_eval_summary': val_eval_summary,
            'test_eval_summary': test_eval_summary,
            'source_breakdowns': source_breakdowns,
            'test_roc_auc': roc_test,
            'test_pr_auc': pr_test
        }, f, indent=2)
    logging.info(f"Laporan evaluasi JSON disimpan di {json_report_path}")

    print("\n" + report_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest & Calibrate Threshold")
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
