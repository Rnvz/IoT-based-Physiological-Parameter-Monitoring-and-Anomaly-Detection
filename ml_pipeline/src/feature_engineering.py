import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_temporal_features_timeseries(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
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
    
    # Drop first (window-1) rows
    if len(df) >= window:
        df = df.iloc[window-1:].reset_index(drop=True)
    return df

def compute_temporal_features_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Heart_Rate', 'SpO2', 'Temperature']:
        df[f'{col}_Delta'] = 0.0
        df[f'MA_{col}'] = df[col]
        df[f'Var_{col}'] = 0.0
        
    df['Rate_of_Change'] = 0.0
    return df.reset_index(drop=True)

def process_features(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    results = []
    for source in df['Source'].unique():
        source_df = df[df['Source'] == source].reset_index(drop=True)
        if source == 'engrarri21':
            feat_df = compute_temporal_features_timeseries(source_df, window)
        else:
            feat_df = compute_temporal_features_cross_sectional(source_df)
        results.append(feat_df)
    return pd.concat(results, ignore_index=True)

def main(input_path: str, output_train: str, output_val: str, output_test: str, scaler_path: str, window: int):
    logging.info(f"Memuat data dari {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. SPLIT DATA: Train (60%) / Validation (20%) / Test (20%)
    train_dfs, val_dfs, test_dfs = [], [], []
    for source in df['Source'].unique():
        source_df = df[df['Source'] == source].reset_index(drop=True)
        if source == 'engrarri21':
            # Temporal split kronologis 60/20/20 per sesi rekaman
            n = len(source_df)
            t1, t2 = int(n * 0.6), int(n * 0.8)
            s_tr = source_df.iloc[:t1].copy()
            s_va = source_df.iloc[t1:t2].copy()
            s_te = source_df.iloc[t2:].copy()
            s_tr['Patient_ID'] = 'engrarri21_session_train'
            s_va['Patient_ID'] = 'engrarri21_session_val'
            s_te['Patient_ID'] = 'engrarri21_session_test'
            train_dfs.append(s_tr)
            val_dfs.append(s_va)
            test_dfs.append(s_te)
        else:
            # Random split 60/20/20 (cross-sectional)
            tr, rest = train_test_split(source_df, test_size=0.4, random_state=42, stratify=source_df['OUTPUT'])
            va, te = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest['OUTPUT'])
            train_dfs.append(tr)
            val_dfs.append(va)
            test_dfs.append(te)
            
    df_train_raw = pd.concat(train_dfs, ignore_index=True)
    df_val_raw = pd.concat(val_dfs, ignore_index=True)
    df_test_raw = pd.concat(test_dfs, ignore_index=True)
    
    logging.info(f"Train: {len(df_train_raw)}, Val: {len(df_val_raw)}, Test: {len(df_test_raw)}")
    
    # 2. FEATURE ENGINEERING (terpisah per partisi)
    df_train_feat = process_features(df_train_raw, window)
    df_val_feat = process_features(df_val_raw, window)
    df_test_feat = process_features(df_test_raw, window)
    
    feature_cols = [
        'Heart_Rate', 'SpO2', 'Temperature', 
        'Heart_Rate_Delta', 'SpO2_Delta', 'Temperature_Delta', 
        'MA_Heart_Rate', 'MA_SpO2', 'MA_Temperature', 
        'Var_Heart_Rate', 'Var_SpO2', 'Var_Temperature', 
        'Rate_of_Change'
    ]
    
    # 3. SCALING (fit hanya pada train)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(df_train_feat[feature_cols]), columns=feature_cols)
    X_val_scaled = pd.DataFrame(scaler.transform(df_val_feat[feature_cols]), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(df_test_feat[feature_cols]), columns=feature_cols)
    
    meta_cols = ['OUTPUT', 'Source', 'Metadata', 'Patient_ID']
    for col in meta_cols:
        if col in df_train_feat:
            X_train_scaled[col] = df_train_feat[col].values
        if col in df_val_feat:
            X_val_scaled[col] = df_val_feat[col].values
        if col in df_test_feat:
            X_test_scaled[col] = df_test_feat[col].values
            
    # Simpan output
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    X_train_scaled.to_csv(output_train, index=False)
    X_val_scaled.to_csv(output_val, index=False)
    X_test_scaled.to_csv(output_test, index=False)
    logging.info(f"Train/Val/Test features saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/harmonized_vitals.csv')
    parser.add_argument('--output_train', type=str, default='../data/features_train.csv')
    parser.add_argument('--output_val', type=str, default='../data/features_val.csv')
    parser.add_argument('--output_test', type=str, default='../data/features_test.csv')
    parser.add_argument('--scaler', type=str, default='../models/scaler.joblib')
    parser.add_argument('--window', type=int, default=15)
    args = parser.parse_args()
    
    basedir = os.path.dirname(__file__)
    main(
        os.path.abspath(os.path.join(basedir, args.input)),
        os.path.abspath(os.path.join(basedir, args.output_train)),
        os.path.abspath(os.path.join(basedir, args.output_val)),
        os.path.abspath(os.path.join(basedir, args.output_test)),
        os.path.abspath(os.path.join(basedir, args.scaler)),
        args.window
    )

