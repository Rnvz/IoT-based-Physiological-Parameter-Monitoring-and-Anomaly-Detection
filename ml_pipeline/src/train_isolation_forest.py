import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from typing import Dict

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def baseline_predict(X_original: pd.DataFrame) -> np.ndarray:
    """
    Prediksi menggunakan simple threshold satu parameter (baseline komparator):
    - HR di luar 60-100 BPM
    - SpO2 < 95%
    - Temp di luar 36.0-37.5°C (rentang normal suhu klinis dataset)
    Anomali = 1, Normal = 0
    """
    is_anomaly = (
        (X_original['Heart_Rate'] < 60) | (X_original['Heart_Rate'] > 100) |
        (X_original['SpO2'] < 95) |
        (X_original['Temperature'] < 36.0) | (X_original['Temperature'] > 37.5)
    )
    return is_anomaly.astype(int).values

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Menghitung metrik performa klasifikasi."""
    return {
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0))
    }

def main(input_path: str, model_path: str, report_path: str, scaler_path: str, contamination: float, n_estimators: int):
    """Fungsi utama pelatihan dan evaluasi model."""
    logging.info(f"Memuat data fitur dari {input_path}")
    
    if not os.path.exists(input_path):
        logging.error(f"File {input_path} tidak ditemukan.")
        return
        
    df = pd.read_csv(input_path)
    
    feature_cols = [
        'Heart_Rate', 'SpO2', 'Temperature', 
        'Heart_Rate_Delta', 'SpO2_Delta', 'Temperature_Delta', 
        'MA_Heart_Rate', 'MA_SpO2', 'MA_Temperature', 
        'Var_Heart_Rate', 'Var_SpO2', 'Var_Temperature', 
        'Rate_of_Change', 'Signal_Quality_Score'
    ]
    
    # 1. Dataset Pembagian Pelatihan (Unsupervised Training Set)
    # Gunakan data pola normal murni dari seluruh sumber untuk mempelajari distribusi fisiologis normal
    train_mask = (
        ((df['Source'] == 'nasirayub2') & (df['OUTPUT'] == 0)) |
        ((df['Source'] == 'engrarri21') & (df['OUTPUT'] == 0)) |
        ((df['Source'] == 'gourangomandal') & (df['OUTPUT'] == 0)) |
        (df['Source'] == 'rishanmascarenhas')
    )
    
    X_train = df.loc[train_mask, feature_cols]
    logging.info(f"Ukuran Data Latih (Pola Normal Gabungan): {X_train.shape[0]:,} sampel")
    
    # 2. Pelatihan Isolation Forest
    logging.info(f"Melatih Isolation Forest (n_estimators={n_estimators}, contamination={contamination})...")
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train)
    
    # Simpan model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    logging.info(f"Model berhasil disimpan di {model_path}")
    
    # Load Scaler untuk evaluasi baseline pada nilai fisik asli
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    report_content = "=======================================================\n"
    report_content += "LAPORAN EVALUASI DETEKSI ANOMALI FISIOLOGIS MULTIVARIAT\n"
    report_content += "=======================================================\n\n"
    
    # 3. Evaluasi Terhadap Dataset engrarri21 (Benchmark Data Riil Klinis)
    val_eng = df[df['Source'] == 'engrarri21'].copy()
    if not val_eng.empty:
        X_val_eng = val_eng[feature_cols]
        y_val_eng = val_eng['OUTPUT'].values
        
        # Prediksi IF (-1: anomali, 1: normal) -> (1: anomali, 0: normal)
        pred_if_eng = np.where(clf.predict(X_val_eng) == -1, 1, 0)
        
        # Prediksi Baseline
        if scaler:
            X_orig_eng = pd.DataFrame(scaler.inverse_transform(X_val_eng), columns=feature_cols)
            pred_base_eng = baseline_predict(X_orig_eng)
        else:
            pred_base_eng = baseline_predict(X_val_eng)
            
        m_if_eng = calculate_metrics(y_val_eng, pred_if_eng)
        m_base_eng = calculate_metrics(y_val_eng, pred_base_eng)
        
        report_content += "1. EVALUASI DATASET ENGRARRI21 (Benchmark Riil)\n"
        report_content += "-"*45 + "\n"
        report_content += f"Total Sampel: {len(y_val_eng):,} (Normal: {(y_val_eng==0).sum():,}, Abnormal: {(y_val_eng==1).sum():,})\n\n"
        report_content += "A. Model Machine Learning (Isolation Forest):\n"
        report_content += f"   - Precision : {m_if_eng['Precision']:.4f}\n"
        report_content += f"   - Recall    : {m_if_eng['Recall']:.4f}\n"
        report_content += f"   - F1-Score  : {m_if_eng['F1']:.4f}\n"
        report_content += f"   - Confusion Matrix:\n{confusion_matrix(y_val_eng, pred_if_eng)}\n\n"
        report_content += "B. Model Baseline (Single-Parameter Threshold):\n"
        report_content += f"   - Precision : {m_base_eng['Precision']:.4f}\n"
        report_content += f"   - Recall    : {m_base_eng['Recall']:.4f}\n"
        report_content += f"   - F1-Score  : {m_base_eng['F1']:.4f}\n"
        report_content += f"   - Confusion Matrix:\n{confusion_matrix(y_val_eng, pred_base_eng)}\n\n"
    
    # 4. Evaluasi Terhadap Dataset gourangomandal (IoMT Alerts & Disease Breakdown)
    val_gourango = df[df['Source'] == 'gourangomandal'].copy()
    if not val_gourango.empty:
        X_val_g = val_gourango[feature_cols]
        y_val_g = val_gourango['OUTPUT'].values
        
        pred_if_g = np.where(clf.predict(X_val_g) == -1, 1, 0)
        
        if scaler:
            X_orig_g = pd.DataFrame(scaler.inverse_transform(X_val_g), columns=feature_cols)
            pred_base_g = baseline_predict(X_orig_g)
        else:
            pred_base_g = baseline_predict(X_val_g)
            
        m_if_g = calculate_metrics(y_val_g, pred_if_g)
        m_base_g = calculate_metrics(y_val_g, pred_base_g)
        
        report_content += "2. EVALUASI DATASET GOURANGOMANDAL (IoMT Multi-Parameter Alerts)\n"
        report_content += "-"*45 + "\n"
        report_content += f"Total Sampel: {len(y_val_g):,} (Normal: {(y_val_g==0).sum():,}, Abnormal: {(y_val_g==1).sum():,})\n\n"
        report_content += "A. Model Machine Learning (Isolation Forest):\n"
        report_content += f"   - Precision : {m_if_g['Precision']:.4f}\n"
        report_content += f"   - Recall    : {m_if_g['Recall']:.4f}\n"
        report_content += f"   - F1-Score  : {m_if_g['F1']:.4f}\n\n"
        report_content += "B. Model Baseline (Single-Parameter Threshold):\n"
        report_content += f"   - Precision : {m_base_g['Precision']:.4f}\n"
        report_content += f"   - Recall    : {m_base_g['Recall']:.4f}\n"
        report_content += f"   - F1-Score  : {m_base_g['F1']:.4f}\n\n"
        
        # Breakdown Deteksi per Kategori Predicted Disease
        if 'Metadata' in val_gourango.columns:
            val_gourango['Pred_IF'] = pred_if_g
            disease_pivot = val_gourango.groupby('Metadata')['Pred_IF'].agg(
                Total='count',
                Anomali_Detected=lambda x: (x == 1).sum(),
                Normal_Detected=lambda x: (x == 0).sum(),
                Anomaly_Rate_Pct=lambda x: round((x == 1).mean() * 100, 2)
            )
            report_content += "C. Deteksi Isolation Forest per Kategori Penyakit (Predicted Disease):\n"
            report_content += str(disease_pivot) + "\n\n"

    # Simpan laporan ke file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logging.info(f"Laporan evaluasi berhasil disimpan di {report_path}")
    print("\n" + report_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Model Isolation Forest & Benchmarking")
    default_input = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'features_engineered.csv'))
    default_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'isolation_forest_model.joblib'))
    default_report = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'evaluation_report.txt'))
    default_scaler = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib'))

    parser.add_argument('--input', type=str, default=default_input, help='Path input features_engineered.csv')
    parser.add_argument('--model', type=str, default=default_model, help='Path output model joblib')
    parser.add_argument('--report', type=str, default=default_report, help='Path output laporan evaluasi')
    parser.add_argument('--scaler', type=str, default=default_scaler, help='Path model scaler')
    parser.add_argument('--contamination', type=float, default=0.08, help='Rasio kontaminasi anomali')
    parser.add_argument('--n_estimators', type=int, default=150, help='Jumlah pohon Isolation Forest')
    
    args = parser.parse_args()
    main(args.input, args.model, args.report, args.scaler, args.contamination, args.n_estimators)
