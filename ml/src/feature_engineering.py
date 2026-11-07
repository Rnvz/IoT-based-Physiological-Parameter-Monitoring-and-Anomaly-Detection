import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURE_COLS = [
    'Heart_Rate', 'SpO2', 'Temperature',
    'Heart_Rate_Delta', 'SpO2_Delta', 'Temperature_Delta',
    'MA_Heart_Rate', 'MA_SpO2', 'MA_Temperature',
    'Var_Heart_Rate', 'Var_SpO2', 'Var_Temperature',
    'Rate_of_Change'
]

def compute_temporal_features_timeseries(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """
    Menghitung fitur temporal HANYA pada stream/sesi time-series kontinu (engrarri21).
    Tidak mencampur aduk baris dari dataset berbeda.
    """
    df = df.copy()
    for col in ['Heart_Rate', 'SpO2', 'Temperature']:
        df[f'{col}_Delta'] = df[col].diff().fillna(0)
        df[f'MA_{col}'] = df[col].rolling(window, min_periods=1).mean()
        df[f'Var_{col}'] = df[col].rolling(window, min_periods=2).var().fillna(0)

    df['Rate_of_Change'] = (
        df['Heart_Rate_Delta'].abs() / df['Heart_Rate'].mean() +
        df['SpO2_Delta'].abs() / df['SpO2'].mean() +
        df['Temperature_Delta'].abs() / df['Temperature'].mean()
    )
    return df

def compute_temporal_features_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Untuk data cross-sectional (bukan time-series), delta dan var diset 0.0
    dan MA sama dengan nilai pengukuran aktual untuk menghindari artefak acak.
    """
    df = df.copy()
    for col in ['Heart_Rate', 'SpO2', 'Temperature']:
        df[f'{col}_Delta'] = 0.0
        df[f'MA_{col}'] = df[col]
        df[f'Var_{col}'] = 0.0

    df['Rate_of_Change'] = 0.0
    return df

def get_bounded_physiological_scaler() -> MinMaxScaler:
    """
    Opsi A (Rekomendasi Utama - Bounded Physiological Scaling):
    Menggunakan MinMaxScaler dengan batas domain fisik sensor medis tetap:
    - Heart Rate: [40.0, 160.0] BPM
    - SpO2: [70.0, 100.0] %
    - Temperature: [25.0, 43.0] °C
    - Turunan temporal diskalakan sesuai rentang deviasi fisiologis wajar.
    """
    bounds_min = [40.0, 70.0, 25.0, -40.0, -15.0, -15.0, 40.0, 70.0, 25.0, 0.0, 0.0, 0.0, 0.0]
    bounds_max = [160.0, 100.0, 43.0, 40.0, 15.0, 15.0, 160.0, 100.0, 43.0, 500.0, 50.0, 70.0, 1.0]
    dummy = pd.DataFrame([bounds_min, bounds_max], columns=FEATURE_COLS)
    scaler = MinMaxScaler()
    scaler.fit(dummy)
    return scaler

def main(input_path: str, output_train: str, output_val: str, output_test: str, scaler_path: str, window: int, scaler_type: str = 'bounded_minmax'):
    logging.info(f"Memuat data harmonisasi dari {input_path}")
    df = pd.read_csv(input_path)

    # 1. Feature Engineering per source secara terisolasi
    source_dfs = []
    for source in df['Source'].unique():
        s_df = df[df['Source'] == source].reset_index(drop=True)
        if source == 'engrarri21':
            logging.info(f"Menghitung fitur temporal time-series kontinu untuk {source} ({len(s_df):,} baris)...")
            feat_df = compute_temporal_features_timeseries(s_df, window=window)
        else:
            logging.info(f"Menghitung fitur cross-sectional untuk {source} ({len(s_df):,} baris)...")
            feat_df = compute_temporal_features_cross_sectional(s_df)
        source_dfs.append(feat_df)

    df_featured = pd.concat(source_dfs, ignore_index=True)

    # 2. Train / Val / Test Split
    # - engrarri21: continuous temporal split (60% Train, 20% Val, 20% Test)
    # - dataset cross-sectional: stratified random split (60% Train, 20% Val, 20% Test)
    train_dfs, val_dfs, test_dfs = [], [], []
    for source in df_featured['Source'].unique():
        s_df = df_featured[df_featured['Source'] == source].reset_index(drop=True)
        if source == 'engrarri21':
            n = len(s_df)
            t1, t2 = int(n * 0.6), int(n * 0.8)
            s_tr = s_df.iloc[:t1].copy()
            s_va = s_df.iloc[t1:t2].copy()
            s_te = s_df.iloc[t2:].copy()
            s_tr['Patient_ID'] = 'engrarri21_stream_train'
            s_va['Patient_ID'] = 'engrarri21_stream_val'
            s_te['Patient_ID'] = 'engrarri21_stream_test'
            train_dfs.append(s_tr)
            val_dfs.append(s_va)
            test_dfs.append(s_te)
        else:
            tr, rest = train_test_split(s_df, test_size=0.4, random_state=42, stratify=s_df['OUTPUT'])
            va, te = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest['OUTPUT'])
            train_dfs.append(tr)
            val_dfs.append(va)
            test_dfs.append(te)

    df_train_raw = pd.concat(train_dfs, ignore_index=True)
    df_val_raw = pd.concat(val_dfs, ignore_index=True)
    df_test_raw = pd.concat(test_dfs, ignore_index=True)

    logging.info(f"Dataset Split: Train={len(df_train_raw):,} (Normal: {(df_train_raw['OUTPUT']==0).sum():,}), "
                 f"Val={len(df_val_raw):,} (Normal: {(df_val_raw['OUTPUT']==0).sum():,}), "
                 f"Test={len(df_test_raw):,} (Normal: {(df_test_raw['OUTPUT']==0).sum():,})")

    # 3. Fitting Scaler
    if scaler_type == 'bounded_minmax':
        logging.info("Menggunakan Scaler Opsi A: Bounded Physiological Scaling (MinMaxScaler domain medis tetap)")
        scaler = get_bounded_physiological_scaler()
    elif scaler_type == 'train_normal_standard':
        logging.info("Menggunakan Scaler Opsi B: StandardScaler di-fit HANYA pada Train Normal (OUTPUT == 0)")
        scaler = StandardScaler()
        scaler.fit(df_train_raw[df_train_raw['OUTPUT'] == 0][FEATURE_COLS])
    elif scaler_type == 'train_normal_robust':
        logging.info("Menggunakan Scaler Opsi B: RobustScaler di-fit HANYA pada Train Normal (OUTPUT == 0)")
        scaler = RobustScaler()
        scaler.fit(df_train_raw[df_train_raw['OUTPUT'] == 0][FEATURE_COLS])
    else:
        raise ValueError(f"Tipe scaler tidak dikenali: {scaler_type}")

    # 4. Transformasi fitur
    X_train_scaled = pd.DataFrame(scaler.transform(df_train_raw[FEATURE_COLS]), columns=FEATURE_COLS)
    X_val_scaled = pd.DataFrame(scaler.transform(df_val_raw[FEATURE_COLS]), columns=FEATURE_COLS)
    X_test_scaled = pd.DataFrame(scaler.transform(df_test_raw[FEATURE_COLS]), columns=FEATURE_COLS)

    meta_cols = ['OUTPUT', 'Source', 'Metadata', 'Patient_ID']
    for col in meta_cols:
        if col in df_train_raw:
            X_train_scaled[col] = df_train_raw[col].values
        if col in df_val_raw:
            X_val_scaled[col] = df_val_raw[col].values
        if col in df_test_raw:
            X_test_scaled[col] = df_test_raw[col].values

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler tersimpan di: {scaler_path}")

    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    X_train_scaled.to_csv(output_train, index=False)
    X_val_scaled.to_csv(output_val, index=False)
    X_test_scaled.to_csv(output_test, index=False)
    logging.info(f"Features tersimpan: {output_train}, {output_val}, {output_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering & Scaling IoT Vitals")
    parser.add_argument('--input', type=str, default='../data/harmonized_vitals.csv')
    parser.add_argument('--output_train', type=str, default='../data/features_train.csv')
    parser.add_argument('--output_val', type=str, default='../data/features_val.csv')
    parser.add_argument('--output_test', type=str, default='../data/features_test.csv')
    parser.add_argument('--scaler', type=str, default='../models/scaler.joblib')
    parser.add_argument('--window', type=int, default=15)
    parser.add_argument('--scaler_type', type=str, default='bounded_minmax',
                        choices=['bounded_minmax', 'train_normal_standard', 'train_normal_robust'],
                        help='Tipe scaler: bounded_minmax (Opsi A) atau train_normal_standard / train_normal_robust (Opsi B)')
    args = parser.parse_args()

    basedir = os.path.dirname(__file__)
    main(
        os.path.abspath(os.path.join(basedir, args.input)),
        os.path.abspath(os.path.join(basedir, args.output_train)),
        os.path.abspath(os.path.join(basedir, args.output_val)),
        os.path.abspath(os.path.join(basedir, args.output_test)),
        os.path.abspath(os.path.join(basedir, args.scaler)),
        args.window,
        args.scaler_type
    )
