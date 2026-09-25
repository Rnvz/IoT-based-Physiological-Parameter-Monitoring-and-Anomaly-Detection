import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib
from sklearn.preprocessing import RobustScaler
from typing import Tuple

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_temporal_features(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """
    Menghitung fitur temporal (Delta, Moving Average, Variance, Rate of Change)
    menggunakan sliding window.
    """
    df = df.copy()
    
    # Pastikan data diurutkan jika ada penanda waktu, namun di sini
    # diasumsikan data merupakan continuous stream sesuai urutan di file per Source
    
    # Hitung Delta (perbedaan dengan sample sebelumnya)
    for col in ['Heart_Rate', 'SpO2', 'Temperature']:
        df[f'{col}_Delta'] = df.groupby('Source')[col].diff().fillna(0)
        
    # Hitung Moving Average
    for col in ['Heart_Rate', 'SpO2', 'Temperature']:
        df[f'MA_{col}'] = df.groupby('Source')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
    # Hitung Variance
    for col in ['Heart_Rate', 'SpO2', 'Temperature']:
        # min_periods=2 agar tidak NaN
        df[f'Var_{col}'] = df.groupby('Source')[col].transform(lambda x: x.rolling(window, min_periods=2).var()).fillna(0)
        
    # Rate of Change (kombinasi nilai absolut delta, di-normalisasi secara lokal)
    df['Rate_of_Change'] = (
        df['Heart_Rate_Delta'].abs() / df['Heart_Rate'].mean() + 
        df['SpO2_Delta'].abs() / df['SpO2'].mean() + 
        df['Temperature_Delta'].abs() / df['Temperature'].mean()
    )
    
    # Signal Quality Score (Placeholder sintetik)
    # Dalam sistem real-time ini akan bergantung pada packet loss atau noise
    df['Signal_Quality_Score'] = 1.0 
    
    # Menghapus baris awal (window - 1) untuk tiap sumber karena window belum penuh secara ideal
    # Namun karena kita pakai min_periods=1/2, NaN sudah teratasi, 
    # tapi agar fitur var akurat, kita drop beberapa baris pertama per Source
    
    def drop_first_n(group):
        return group.iloc[window-1:] if len(group) > window else group
        
    df_filtered = df.groupby('Source', group_keys=False).apply(drop_first_n).reset_index(drop=True)
    
    return df_filtered

def main(input_path: str, output_path: str, scaler_path: str, window: int):
    """Fungsi utama untuk rekayasa fitur."""
    logging.info(f"Memuat data dari {input_path}")
    
    if not os.path.exists(input_path):
        logging.error(f"File {input_path} tidak ditemukan.")
        return
        
    df = pd.read_csv(input_path)
    
    logging.info(f"Membuat fitur temporal dengan window size = {window}...")
    df_features = compute_temporal_features(df, window=window)
    
    feature_cols = [
        'Heart_Rate', 'SpO2', 'Temperature', 
        'Heart_Rate_Delta', 'SpO2_Delta', 'Temperature_Delta', 
        'MA_Heart_Rate', 'MA_SpO2', 'MA_Temperature', 
        'Var_Heart_Rate', 'Var_SpO2', 'Var_Temperature', 
        'Rate_of_Change', 'Signal_Quality_Score'
    ]
    
    # Pisahkan label dan metadata
    # Asumsikan 'OUTPUT', 'Source', dan 'Metadata' tetap dipertahankan
    meta_cols = ['OUTPUT', 'Source', 'Metadata']
    available_meta = [col for col in meta_cols if col in df_features.columns]
    
    X = df_features[feature_cols]
    
    logging.info("Melakukan scaling dengan RobustScaler...")
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    
    # Gabungkan kembali dengan metadata
    for col in available_meta:
        X_scaled[col] = df_features[col].values
        
    # Simpan scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler disimpan di {scaler_path}")
    
    # Simpan data fitur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_scaled.to_csv(output_path, index=False)
    logging.info(f"Dataset fitur disimpan di {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rekayasa Fitur untuk Anomaly Detection")
    default_input = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'harmonized_vitals.csv'))
    default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'features_engineered.csv'))
    default_scaler = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib'))

    parser.add_argument('--input', type=str, default=default_input, help='Path input dataset harmonisasi')
    parser.add_argument('--output', type=str, default=default_output, help='Path output dataset dengan fitur')
    parser.add_argument('--scaler', type=str, default=default_scaler, help='Path output model scaler')
    parser.add_argument('--window', type=int, default=15, help='Ukuran sliding window')
    
    args = parser.parse_args()
    main(args.input, args.output, args.scaler, args.window)

