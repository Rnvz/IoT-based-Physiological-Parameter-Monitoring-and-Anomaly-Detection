import pandas as pd
import numpy as np
import argparse
import os
import glob
import logging
from typing import Optional, List

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_biological_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menerapkan filter batas fisiologis yang masuk akal:
    - Heart Rate: 30 - 220 BPM
    - SpO2: 70 - 100 %
    - Temperature: 25 - 42 °C
    """
    initial_len = len(df)
    
    valid_hr = (df['Heart_Rate'] >= 30) & (df['Heart_Rate'] <= 220)
    valid_spo2 = (df['SpO2'] >= 70) & (df['SpO2'] <= 100)
    valid_temp = (df['Temperature'] >= 25) & (df['Temperature'] <= 42)
    
    df_filtered = df[valid_hr & valid_spo2 & valid_temp].copy()
    
    dropped = initial_len - len(df_filtered)
    if dropped > 0:
        logging.info(f"Dihapus {dropped:,} baris karena nilai berada di luar rentang biologis.")
        
    return df_filtered

def process_engrarri21(file_path: str) -> Optional[pd.DataFrame]:
    """Memproses dataset engrarri21 (Human_vital_signs_R.csv)."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        # Bersihkan spasi pada nama kolom
        df.columns = df.columns.str.strip()
        
        # Mapping kolom: 'HR (BPM)' -> Heart_Rate, 'SpO2 (%)' -> SpO2, 'TEMP (*C)' -> Temperature
        rename_map = {}
        for col in df.columns:
            if 'HR' in col:
                rename_map[col] = 'Heart_Rate'
            elif 'SpO2' in col or 'spo2' in col.lower():
                rename_map[col] = 'SpO2'
            elif 'TEMP' in col or 'temp' in col.lower():
                rename_map[col] = 'Temperature'
        df.rename(columns=rename_map, inplace=True)
        
        # Mapping OUTPUT: Normal -> 0, Abnormal -> 1
        output_series = df['OUTPUT'].astype(str).str.strip().str.lower()
        output_binary = np.where(output_series == 'normal', 0, 1)
        
        result = pd.DataFrame({
            'Heart_Rate': pd.to_numeric(df['Heart_Rate'], errors='coerce'),
            'SpO2': pd.to_numeric(df['SpO2'], errors='coerce'),
            'Temperature': pd.to_numeric(df['Temperature'], errors='coerce'),
            'OUTPUT': output_binary,
            'Source': 'engrarri21',
            'Metadata': df['OUTPUT'].astype(str)
        })
        logging.info(f"engrarri21 selesai diproses: {len(result):,} baris")
        return result
    except Exception as e:
        logging.error(f"Gagal memproses engrarri21 ({file_path}): {e}")
        return None

def process_nasirayub2(file_path: str) -> Optional[pd.DataFrame]:
    """Memproses dataset nasirayub2 (human_vital_signs_dataset_2024.csv)."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        df.columns = df.columns.str.strip()
        
        # Kolom utama: Heart Rate, Oxygen Saturation, Body Temperature, Risk Category
        hr = pd.to_numeric(df['Heart Rate'], errors='coerce')
        spo2 = pd.to_numeric(df['Oxygen Saturation'], errors='coerce')
        temp = pd.to_numeric(df['Body Temperature'], errors='coerce')
        
        # Risk Category: 'Low Risk' -> 0 (Normal), 'High Risk' -> 1 (Abnormal)
        risk = df['Risk Category'].astype(str).str.strip().str.lower()
        output_binary = np.where(risk == 'low risk', 0, 1)
        
        result = pd.DataFrame({
            'Heart_Rate': hr,
            'SpO2': spo2,
            'Temperature': temp,
            'OUTPUT': output_binary,
            'Source': 'nasirayub2',
            'Metadata': df['Risk Category'].astype(str)
        })
        logging.info(f"nasirayub2 selesai diproses: {len(result):,} baris")
        return result
    except Exception as e:
        logging.error(f"Gagal memproses nasirayub2 ({file_path}): {e}")
        return None

def process_rishanmascarenhas(file_path: str) -> Optional[pd.DataFrame]:
    """Memproses dataset rishanmascarenhas (qt_dataset.csv) dengan konversi Fahrenheit ke Celsius."""
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        df.columns = df.columns.str.strip()
        
        # Kolom: Oxygen, PulseRate, Temperature, Result
        hr = pd.to_numeric(df['PulseRate'], errors='coerce')
        spo2 = pd.to_numeric(df['Oxygen'], errors='coerce')
        temp_f = pd.to_numeric(df['Temperature'], errors='coerce')
        
        # Konversi Fahrenheit ke Celsius: (°F - 32) * 5/9
        # Nilai mentah di dataset sekitar 95 - 105 °F
        temp_c = (temp_f - 32.0) * (5.0 / 9.0)
        
        result = pd.DataFrame({
            'Heart_Rate': hr,
            'SpO2': spo2,
            'Temperature': temp_c,
            'OUTPUT': 0,  # Digunakan sebagai suplemen unsupervised training pola normal
            'Source': 'rishanmascarenhas',
            'Metadata': df['Result'].astype(str)
        })
        logging.info(f"rishanmascarenhas selesai diproses (suhu dikonversi ke °C): {len(result):,} baris")
        return result
    except Exception as e:
        logging.error(f"Gagal memproses rishanmascarenhas ({file_path}): {e}")
        return None

def process_gourangomandal(file_path: str) -> Optional[pd.DataFrame]:
    """
    Memproses dataset sintetis IoMT (Synthetic_patient-HealthCare-Monitoring_dataset.csv).
    Menghitung anomali berdasarkan kombinasi alert:
    - Jika seluruh alert (HR, SpO2, Temp) == 'NORMAL' -> OUTPUT = 0 (Normal)
    - Jika ada alert == 'ABNORMAL' -> OUTPUT = 1 (Anomali)
    - 'Predicted Disease' disimpan di kolom Metadata untuk analisis komparatif per penyakit.
    """
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        df.columns = df.columns.str.strip()
        
        # Identifikasi nama kolom dinamis
        temp_col = [c for c in df.columns if 'temp' in c.lower() and 'alert' not in c.lower()][0]
        hr_col = [c for c in df.columns if 'heart' in c.lower() and 'alert' not in c.lower()][0]
        spo2_col = [c for c in df.columns if 'spo2' in c.lower() and 'alert' not in c.lower()][0]
        
        hr = pd.to_numeric(df[hr_col], errors='coerce')
        spo2 = pd.to_numeric(df[spo2_col], errors='coerce')
        temp = pd.to_numeric(df[temp_col], errors='coerce')
        
        # Multi-parameter alert logic
        hr_alert = df['Heart Rate Alert'].astype(str).str.strip().str.upper()
        spo2_alert = df['SpO2 Level Alert'].astype(str).str.strip().str.upper()
        temp_alert = df['Temperature Alert'].astype(str).str.strip().str.upper()
        
        is_normal = (hr_alert == 'NORMAL') & (spo2_alert == 'NORMAL') & (temp_alert == 'NORMAL')
        output_binary = np.where(is_normal, 0, 1)
        
        result = pd.DataFrame({
            'Heart_Rate': hr,
            'SpO2': spo2,
            'Temperature': temp,
            'OUTPUT': output_binary,
            'Source': 'gourangomandal',
            'Metadata': df['Predicted Disease'].astype(str)
        })
        logging.info(f"gourangomandal selesai diproses: {len(result):,} baris")
        return result
    except Exception as e:
        logging.error(f"Gagal memproses gourangomandal ({file_path}): {e}")
        return None

def main(input_dir: str, output_path: str):
    """Pipeline pembersihan & harmonisasi semua dataset."""
    logging.info(f"Mencari dataset di direktori: {input_dir}")
    
    datasets: List[pd.DataFrame] = []
    
    files = glob.glob(os.path.join(input_dir, '**', '*.*'), recursive=True)
    
    for file in files:
        if not file.endswith(('.csv', '.xlsx')):
            continue
            
        fname_lower = os.path.basename(file).lower()
        
        if 'human_vital_signs_r' in fname_lower or 'engrarri' in fname_lower:
            logging.info(f"Mendeteksi engrarri21: {file}")
            df = process_engrarri21(file)
        elif 'human_vital_signs_dataset_2024' in fname_lower or 'nasirayub' in fname_lower:
            logging.info(f"Mendeteksi nasirayub2: {file}")
            df = process_nasirayub2(file)
        elif 'qt_dataset' in fname_lower or 'rishan' in fname_lower:
            logging.info(f"Mendeteksi rishanmascarenhas: {file}")
            df = process_rishanmascarenhas(file)
        elif 'synthetic' in fname_lower or 'healthcare-monitoring' in fname_lower or 'gourango' in fname_lower:
            logging.info(f"Mendeteksi gourangomandal / synthetic: {file}")
            df = process_gourangomandal(file)
        else:
            continue
            
        if df is not None:
            datasets.append(df)
            
    if not datasets:
        logging.error(f"Tidak ada dataset valid yang berhasil dimuat dari {input_dir}.")
        return
        
    combined_df = pd.concat(datasets, ignore_index=True)
    logging.info(f"Total baris gabungan sebelum pembersihan: {len(combined_df):,}")
    
    # Hapus missing values pada parameter inti
    combined_df.dropna(subset=['Heart_Rate', 'SpO2', 'Temperature'], inplace=True)
    
    # Filter validitas biologis
    cleaned_df = apply_biological_filters(combined_df)
    logging.info(f"Total baris bersih yang valid: {len(cleaned_df):,}")
    
    # Cetak statistik ringkasan per dataset sumber
    print("\n" + "="*50)
    print("RINGKASAN DATASET HARMONISASI")
    print("="*50)
    summary = cleaned_df.groupby(['Source', 'OUTPUT']).size().unstack(fill_value=0)
    summary.columns = ['Normal (0)', 'Abnormal (1)']
    print(summary)
    print("\nStatistik Deskriptif Parameter Fisiologis:")
    print(cleaned_df[['Heart_Rate', 'SpO2', 'Temperature']].describe().round(2))
    print("="*50 + "\n")
    
    # Simpan hasil harmonisasi
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    logging.info(f"Dataset harmonisasi tersimpan di: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pembersihan & Harmonisasi Dataset IoT Anomaly Detection")
    # Default mengarah ke folder Dataset/ di root proyek
    default_input = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset'))
    default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'harmonized_vitals.csv'))
    
    parser.add_argument('--input', type=str, default=default_input, help='Direktori input dataset')
    parser.add_argument('--output', type=str, default=default_output, help='Path output dataset harmonisasi')
    
    args = parser.parse_args()
    main(args.input, args.output)
