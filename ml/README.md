# Machine Learning (ML) Pipeline

Direktori ini berisi seluruh komponen algoritma **Machine Learning (Classical ML)** berbasis **Isolation Forest** untuk deteksi anomali sinyal fisiologis pasien.

## Struktur Direktori
- `src/`: Skrip pemrosesan data, ekstraksi 13 fitur temporal, pembersihan dataset, dan pelatihan Isolation Forest.
  - `clean_datasets.py`
  - `feature_engineering.py`
  - `train_isolation_forest.py`
  - `train_physio_synthetic.py`
- `models/`: Artefak model biner (`.joblib`), normalizer scaler, kalibrasi threshold, dan laporan evaluasi resmi.
  - `isolation_forest_model.joblib`
  - `scaler.joblib`
  - `thresholds.joblib`
  - `evaluation_report.txt`
  - `physio_synthetic/`
  - `physio_records_trained/`
- `eval/`: Modul metrik, evaluasi baseline, injeksi anomali, dan splitting dataset.
- `requirements.txt`: Dependensi spesifik modul machine learning.
