"""
ml_pipeline.src.train_physio_synthetic:
Pipeline pemrosesan data dan pelatihan model Isolation Forest untuk dataset:
Dataset/physio_monitor_synthetic_dataset (1).csv

Tahapan:
1. Pembersihan data: filter fw_status == 'NORMAL' & finger == True
2. Feature engineering berbasis sesi (window=15):
   13 fitur fisiologis temporal: nilai instan, delta, moving average, varians, dan rate of change.
3. Split train/val/test/report_only sesuai label split_role dataset.
4. Penskalaan fitur dengan RobustScaler (fit pada train set normal).
5. Pelatihan model Isolation Forest (unsupervised pada sampel train normal).
6. Kalibrasi ambang batas (thresholds) pada validation set untuk berbagai target FPR (1%, 3%, 5%, 10%).
7. Evaluasi menyeluruh pada test set & breakdown per skenario (REST, TYPING, MOTION, FINGER_SHIFT, dll).
8. Penyimpanan artefak: model, scaler, thresholds, dan laporan evaluasi lengkap.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_physio_synthetic")

FEATURE_COLS = [
    "Heart_Rate", "SpO2", "Temperature",
    "Heart_Rate_Delta", "SpO2_Delta", "Temperature_Delta",
    "MA_Heart_Rate", "MA_SpO2", "MA_Temperature",
    "Var_Heart_Rate", "Var_SpO2", "Var_Temperature",
    "Rate_of_Change"
]

def extract_session_features(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """
    Mengekstrak 13 fitur temporal rolling window per sesi untuk menghindari data leakage antar sesi.
    """
    df_valid = df[df["fw_status"] == "NORMAL"].copy()
    rename_map = {
        "hr": "Heart_Rate",
        "spo2": "SpO2",
        "temp": "Temperature",
        "ref_OUTPUT": "OUTPUT"
    }
    df_valid = df_valid.rename(columns=rename_map)

    session_features = []
    for sid, group in df_valid.groupby("session_id", sort=True):
        group = group.sort_values("ts").reset_index(drop=True)
        for col in ["Heart_Rate", "SpO2", "Temperature"]:
            group[f"{col}_Delta"] = group[col].diff().fillna(0.0)
            group[f"MA_{col}"] = group[col].rolling(window, min_periods=1).mean()
            group[f"Var_{col}"] = group[col].rolling(window, min_periods=2).var().fillna(0.0)

        group["Rate_of_Change"] = (
            group["Heart_Rate_Delta"].abs() / group["MA_Heart_Rate"] +
            group["SpO2_Delta"].abs() / group["MA_SpO2"] +
            group["Temperature_Delta"].abs() / group["MA_Temperature"]
        )

        if len(group) >= window:
            group = group.iloc[window - 1:].reset_index(drop=True)

        session_features.append(group)

    return pd.concat(session_features, ignore_index=True)

def train_and_evaluate(
    input_csv: str,
    output_data_dir: str,
    output_model_dir: str,
    n_estimators: int = 100,
    contamination: float = 0.01,
    random_state: int = 42,
    window: int = 15
):
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)

    logger.info(f"Membaca dataset mentah dari: {input_csv}")
    raw_df = pd.read_csv(input_csv)
    logger.info(f"Total baris mentah: {len(raw_df):,}")

    logger.info(f"Mengekstrak fitur temporal 13-dimensi (window={window})...")
    feat_df = extract_session_features(raw_df, window=window)
    logger.info(f"Total baris setelah warm-up window: {len(feat_df):,}")

    train_df = feat_df[feat_df["split_role"] == "train"].copy().reset_index(drop=True)
    val_df = feat_df[feat_df["split_role"] == "val"].copy().reset_index(drop=True)
    test_df = feat_df[feat_df["split_role"] == "test"].copy().reset_index(drop=True)
    rep_df = feat_df[feat_df["split_role"] == "report_only"].copy().reset_index(drop=True)

    logger.info(f"Distribusi partisi:")
    logger.info(f"  Train:       {len(train_df):,} baris (Normal: {(train_df['OUTPUT']==0).sum():,}, Anomali: {(train_df['OUTPUT']==1).sum():,})")
    logger.info(f"  Val:         {len(val_df):,} baris (Normal: {(val_df['OUTPUT']==0).sum():,}, Anomali: {(val_df['OUTPUT']==1).sum():,})")
    logger.info(f"  Test:        {len(test_df):,} baris (Normal: {(test_df['OUTPUT']==0).sum():,}, Anomali: {(test_df['OUTPUT']==1).sum():,})")
    logger.info(f"  Report-only: {len(rep_df):,} baris")

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(train_df[FEATURE_COLS]), columns=FEATURE_COLS)
    X_val_scaled = pd.DataFrame(scaler.transform(val_df[FEATURE_COLS]), columns=FEATURE_COLS)
    X_test_scaled = pd.DataFrame(scaler.transform(test_df[FEATURE_COLS]), columns=FEATURE_COLS)
    X_rep_scaled = pd.DataFrame(scaler.transform(rep_df[FEATURE_COLS]), columns=FEATURE_COLS)

    meta_cols = ["ts", "device_id", "session_id", "scenario", "split_role", "OUTPUT"]
    for c in meta_cols:
        if c in train_df.columns:
            X_train_scaled[c] = train_df[c].values
            X_val_scaled[c] = val_df[c].values
            X_test_scaled[c] = test_df[c].values
            X_rep_scaled[c] = rep_df[c].values

    train_feat_path = os.path.join(output_data_dir, "features_train.csv")
    val_feat_path = os.path.join(output_data_dir, "features_val.csv")
    test_feat_path = os.path.join(output_data_dir, "features_test.csv")
    rep_feat_path = os.path.join(output_data_dir, "features_report_only.csv")

    X_train_scaled.to_csv(train_feat_path, index=False)
    X_val_scaled.to_csv(val_feat_path, index=False)
    X_test_scaled.to_csv(test_feat_path, index=False)
    X_rep_scaled.to_csv(rep_feat_path, index=False)
    logger.info(f"Fitur terproses disimpan di {output_data_dir}")

    scaler_path = os.path.join(output_model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"RobustScaler disimpan di {scaler_path}")

    logger.info(f"Melatih IsolationForest (n_estimators={n_estimators}, contamination={contamination})...")
    X_train_fit = X_train_scaled[FEATURE_COLS].values
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train_fit)

    model_path = os.path.join(output_model_dir, "isolation_forest_model.joblib")
    joblib.dump(clf, model_path)
    logger.info(f"Model IsolationForest disimpan di {model_path}")

    scores_val = clf.decision_function(X_val_scaled[FEATURE_COLS].values)
    scores_test = clf.decision_function(X_test_scaled[FEATURE_COLS].values)
    scores_rep = clf.decision_function(X_rep_scaled[FEATURE_COLS].values)

    y_val = val_df["OUTPUT"].astype(int).values
    y_test = test_df["OUTPUT"].astype(int).values

    roc_val = float(roc_auc_score(y_val, -scores_val))
    pr_val = float(average_precision_score(y_val, -scores_val))
    roc_test = float(roc_auc_score(y_test, -scores_test))
    pr_test = float(average_precision_score(y_test, -scores_test))

    logger.info(f"Area Under Curve:")
    logger.info(f"  Val  -> ROC-AUC: {roc_val:.4f}, PR-AUC: {pr_val:.4f}")
    logger.info(f"  Test -> ROC-AUC: {roc_test:.4f}, PR-AUC: {pr_test:.4f}")

    val_normal_scores = scores_val[y_val == 0]
    target_fprs = [0.01, 0.02, 0.03, 0.05, 0.10]
    threshold_dict = {}
    threshold_metrics = []

    for fpr in target_fprs:
        thr = float(np.percentile(val_normal_scores, fpr * 100))
        threshold_dict[f"fpr_{int(fpr*100):02d}pct"] = thr

        pred_val = (scores_val < thr).astype(int)
        pred_test = (scores_test < thr).astype(int)

        prec_v = float(precision_score(y_val, pred_val, zero_division=0))
        rec_v = float(recall_score(y_val, pred_val, zero_division=0))
        f1_v = float(f1_score(y_val, pred_val, zero_division=0))

        prec_t = float(precision_score(y_test, pred_test, zero_division=0))
        rec_t = float(recall_score(y_test, pred_test, zero_division=0))
        f1_t = float(f1_score(y_test, pred_test, zero_division=0))
        fpr_t = float((pred_test[y_test == 0]).mean())

        threshold_metrics.append({
            "target_val_fpr": fpr,
            "threshold": thr,
            "val_precision": prec_v,
            "val_recall": rec_v,
            "val_f1": f1_v,
            "test_fpr": fpr_t,
            "test_precision": prec_t,
            "test_recall": rec_t,
            "test_f1": f1_t
        })

    thr_joblib_path = os.path.join(output_model_dir, "thresholds.joblib")
    joblib.dump(threshold_dict, thr_joblib_path)
    logger.info(f"Thresholds tersimpan di {thr_joblib_path}")

    thr_3 = threshold_dict["fpr_03pct"]
    thr_5 = threshold_dict["fpr_05pct"]

    test_df_eval = test_df.copy()
    test_df_eval["pred_3"] = (scores_test < thr_3).astype(int)
    test_df_eval["pred_5"] = (scores_test < thr_5).astype(int)

    scenario_eval = {}
    for sc, grp in test_df_eval.groupby("scenario"):
        y_g = grp["OUTPUT"].astype(int).values
        n_norm = int((y_g == 0).sum())
        n_ano = int((y_g == 1).sum())

        p3_g = grp["pred_3"].values
        p5_g = grp["pred_5"].values

        fp3 = int((p3_g[y_g == 0] == 1).sum()) if n_norm > 0 else 0
        rec3 = float(recall_score(y_g, p3_g, zero_division=0)) if n_ano > 0 else 0.0

        fp5 = int((p5_g[y_g == 0] == 1).sum()) if n_norm > 0 else 0
        rec5 = float(recall_score(y_g, p5_g, zero_division=0)) if n_ano > 0 else 0.0

        scenario_eval[sc] = {
            "total_samples": len(grp),
            "normal_count": n_norm,
            "anomaly_count": n_ano,
            "thr_3pct": {
                "false_positives": fp3,
                "fpr": float(fp3 / n_norm) if n_norm > 0 else 0.0,
                "recall": rec3
            },
            "thr_5pct": {
                "false_positives": fp5,
                "fpr": float(fp5 / n_norm) if n_norm > 0 else 0.0,
                "recall": rec5
            }
        }

    rep_df_eval = rep_df.copy()
    rep_df_eval["pred_3"] = (scores_rep < thr_3).astype(int)
    rep_df_eval["pred_5"] = (scores_rep < thr_5).astype(int)

    rep_eval = {}
    for sc, grp in rep_df_eval.groupby("scenario"):
        rep_eval[sc] = {
            "total_samples": len(grp),
            "flagged_thr_3pct": int(grp["pred_3"].sum()),
            "flagged_rate_3pct": float(grp["pred_3"].mean()),
            "flagged_thr_5pct": int(grp["pred_5"].sum()),
            "flagged_rate_5pct": float(grp["pred_5"].mean())
        }

    report_payload = {
        "dataset_name": "physio_monitor_synthetic_dataset (1)",
        "model_architecture": "IsolationForest",
        "n_estimators": n_estimators,
        "contamination": contamination,
        "window_size": window,
        "feature_columns": FEATURE_COLS,
        "sample_counts": {
            "train_normal": len(train_df),
            "val_total": len(val_df),
            "val_normal": int((y_val == 0).sum()),
            "val_anomaly": int((y_val == 1).sum()),
            "test_total": len(test_df),
            "test_normal": int((y_test == 0).sum()),
            "test_anomaly": int((y_test == 1).sum()),
            "report_only_total": len(rep_df)
        },
        "auc_metrics": {
            "val_roc_auc": roc_val,
            "val_pr_auc": pr_val,
            "test_roc_auc": roc_test,
            "test_pr_auc": pr_test
        },
        "threshold_calibration": threshold_metrics,
        "test_scenario_evaluation": scenario_eval,
        "report_only_evaluation": rep_eval
    }

    report_json_path = os.path.join(output_model_dir, "evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_payload, f, indent=2)
    logger.info(f"Laporan evaluasi JSON disimpan di {report_json_path}")

    txt_lines = [
        "=" * 70,
        "LAPORAN PELATIHAN & EVALUASI ISOLATION FOREST",
        f"Dataset: physio_monitor_synthetic_dataset (1)",
        "=" * 70,
        f"Jumlah Fitur: 13 ({', '.join(FEATURE_COLS[:3])}, ...)",
        f"ROC-AUC (Val / Test):  {roc_val:.4f} / {roc_test:.4f}",
        f"PR-AUC  (Val / Test):  {pr_val:.4f} / {pr_test:.4f}",
        "-" * 70,
        "KALIBRASI AMBANG BATAS & PERFORMA TEST SET:",
        f"{'Target FPR':<12} | {'Threshold':<10} | {'Test FPR':<9} | {'Test Prec':<10} | {'Test Rec':<10} | {'Test F1':<8}",
        "-" * 70
    ]
    for m in threshold_metrics:
        txt_lines.append(
            f"{m['target_val_fpr']*100:4.1f}%{'':<7} | {m['threshold']:+9.5f} | {m['test_fpr']*100:5.2f}%{'':<3} | {m['test_precision']:9.4f}  | {m['test_recall']:9.4f}  | {m['test_f1']:6.4f}"
        )
    txt_lines.extend([
        "-" * 70,
        "PERFORMA PER SKENARIO PADA TEST SET (Threshold 3% FPR):",
        f"{'Skenario':<16} | {'Normal (FP / FPR)':<22} | {'Anomali (Recall)':<20}",
        "-" * 70
    ])
    for sc, sinfo in scenario_eval.items():
        n_norm = sinfo["normal_count"]
        fp = sinfo["thr_3pct"]["false_positives"]
        fpr = sinfo["thr_3pct"]["fpr"] * 100
        n_ano = sinfo["anomaly_count"]
        rec = sinfo["thr_3pct"]["recall"] * 100
        norm_str = f"{n_norm} (FP: {fp}, {fpr:.1f}%)" if n_norm > 0 else "N/A"
        ano_str = f"{n_ano} ({rec:.1f}%)" if n_ano > 0 else "N/A"
        txt_lines.append(f"{sc:<16} | {norm_str:<22} | {ano_str:<20}")

    txt_lines.append("=" * 70)
    report_txt_content = "\n".join(txt_lines)

    report_txt_path = os.path.join(output_model_dir, "evaluation_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(report_txt_content)
    logger.info(f"Laporan evaluasi TXT disimpan di {report_txt_path}")
    print("\n" + report_txt_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proses & Latih Model ML untuk physio_monitor_synthetic_dataset (1)")
    default_input = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Dataset", "physio_monitor_synthetic_dataset (1).csv")
    )
    default_data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "physio_synthetic")
    )
    default_model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "physio_synthetic")
    )

    parser.add_argument("--input", type=str, default=default_input, help="Path ke dataset mentah CSV")
    parser.add_argument("--output_data", type=str, default=default_data_dir, help="Direktori output data fitur")
    parser.add_argument("--output_model", type=str, default=default_model_dir, help="Direktori output model artefak")
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah estimator Isolation Forest")
    parser.add_argument("--contamination", type=float, default=0.01, help="Contamination parameter")
    parser.add_argument("--window", type=int, default=15, help="Ukuran rolling window")

    args = parser.parse_args()
    train_and_evaluate(
        input_csv=args.input,
        output_data_dir=args.output_data,
        output_model_dir=args.output_model,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        window=args.window
    )
