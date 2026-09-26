# IoT Physiological Parameter Monitoring & Anomaly Detection

---
⚠️ **CATATAN PENTING — CAKUPAN VALIDASI FASE 1**

Seluruh metrik performa (Precision, Recall, F1-Score, FPR) yang dilaporkan di dokumen ini berasal dari **EVALUASI OFFLINE pada dataset publik Kaggle** (`gourangomandal`, `engrarri21`, `nasirayub2`). Ini **BUKAN** hasil pengujian dengan perangkat keras fisik (ESP32 + MAX30102 + DS18B20).

Keterbatasan yang perlu diketahui:
1. Data training/evaluasi berasal dari alat ukur klinis standar, **BUKAN** dari sinyal PPG optik wearable seperti MAX30102. Ada domain gap yang belum terukur antara kedua sumber sinyal ini.
2. Model belum diuji terhadap noise sensor nyata: pergeseran strap, gerakan tangan, variasi tekanan kontak kulit -- faktor-faktor yang menjadi fokus 7 skenario pengujian fisik di proposal.
3. Threshold operasional (FPR 13% / 0.2535) adalah **TITIK AWAL (starting point)**, BUKAN nilai final. Threshold ini **WAJIB dikalibrasi ulang** menggunakan data riil dari device setelah Fase 2/3.
4. Angka Recall/Precision/F1 pada dokumen ini **TIDAK BOLEH dikutip sebagai "performa sistem IoT"** di laporan akhir tanpa embel-embel *"berdasarkan evaluasi dataset publik (offline)"*.

Validasi performa sesungguhnya dari sistem end-to-end akan dilaporkan terpisah setelah Fase 2 (integrasi firmware-backend) dan Fase 3 (pengujian 7 skenario fisik), sesuai `ROADMAP.md`.
---

## Ringkasan Proyek
Sistem pemantauan parameter fisiologis (Heart Rate, SpO2, Suhu Tubuh) berbasis IoT (ESP32 + MAX30102 + DS18B20) dengan deteksi anomali real-time menggunakan model Machine Learning (Isolation Forest) dan pemrosesan edge-cloud.

---

## Status & Dokumentasi Dataset

| Dataset | File Sumber | Sifat Data | Status di Pipeline | Catatan Metodologis |
|---|---|---|---|---|
| **gourangomandal** | `Synthetic_patient-HealthCare-Monitoring_dataset.csv` (60.000 baris) | Cross-sectional (1 baris = 1 pasien) | **AKTIF** (Train, Val, Test) | Dataset paling bersih untuk validasi batas parametrik. Nilai anomali dibangkitkan dari kombinasi multi-alert. |
| **nasirayub2** | `human_vital_signs_dataset_2024.csv` (200.020 baris) | Cross-sectional (1 baris = 1 pasien) | **AKTIF HANYA NORMAL** (Train & Val) | Hanya sampel `Low Risk` (Normal) yang dipakai untuk memperkaya variasi fisiologis normal. Sampel `High Risk` tidak digunakan untuk evaluasi sensitivitas sensor karena anomali ditentukan oleh BP/BMI/Usia (bukan HR/SpO2/Temp). |
| **engrarri21** | `Human_vital_signs_R.csv` (25.493 baris) | Time-series kontinu (~7 jam) | **AKTIF** (Train, Val, Test dinamis) | Satu-satunya dataset dengan urutan temporal valid untuk menguji fitur windowing ($W=15$) dan Strict Reset Debounce. Catatan: potongan 20% terakhir adalah skenario hipertermia berat (95% abnormal) dan sampel 'Normal'-nya mengalami mislabeling hipotermia (29–34°C). |
| **rishanmascarenhas** | `qt_dataset.csv` (10.000 baris) | Cross-sectional | **RESMI DIEKSKLUSI** (Dilarang Digunakan) | **Alasan Eksklusi**: Kolom `Result` adalah diagnosis RT-PCR COVID-19 (`Positive`/`Negative`). Sebanyak 96.1% sampel memiliki vital abnormal klinis (SpO2 < 95%, demam hingga 40.5°C) yang sebelumnya keliru dipaksa berlabel `OUTPUT = 0` (Normal) oleh repositori lama. Mengakibatkan pencemaran profil normal saat training dan false alarm artifisial masif saat evaluasi. |

---

## Standar ML Pipeline (Fase 1 Selesai)

1. **Vektor Fitur (13 Dimensi)**:
   - $x_1 - x_3$: `Heart_Rate`, `SpO2`, `Temperature` (Nilai instan)
   - $x_4 - x_6$: `Heart_Rate_Delta`, `SpO2_Delta`, `Temperature_Delta` ($Delta = 0$ pada cross-sectional)
   - $x_7 - x_9$: `MA_Heart_Rate`, `MA_SpO2`, `MA_Temperature` (Nilai instan pada cross-sectional)
   - $x_{10} - x_{12}$: `Var_Heart_Rate`, `Var_SpO2`, `Var_Temperature` ($Var = 0$ pada cross-sectional)
   - $x_{13}$: `Rate_of_Change` ($0.0$ pada cross-sectional)
   - `Signal_Quality_Score` ($x_{14}$ lama) **dikeluarkan dari fitur ML** dan difungsikan sebagai gate rule independen di layer backend/aplikasi.

2. **Pembagian Data (3-Way Split Per Pasien/Sesi)**:
   - Train (60%) / Validation (20%) / Holdout Test (20%) dengan **zero patient overlap**.
   - `engrarri21` dipotong kronologis 60/20/20.
   - RobustScaler di-*fit* **hanya pada Train set**.

3. **Kalibrasi Threshold Berbasis FPR**:
   - Model dilatih murni pada sampel Normal Train set.
   - Threshold dikalibrasi pada sampel Normal Validation set berdasarkan target False Positive Rate.
   - **Operating Point Baseline Terkunci**:
     * **Target FPR 13%** (Threshold: `0.2535`): Recall 92.0%, Precision 96.2%, FPR aktual 8.2% pada `gourangomandal`.
     * **Target FPR 15%** (Threshold: `0.2636`): Recall 93.7%, Precision 95.3%, FPR aktual 10.3% pada `gourangomandal`.
     * **Catatan Penting**: Angka Recall/Precision/FPR baseline ini berasal dari dataset publik (`gourangomandal`, `engrarri21`, `nasirayub2`), BUKAN dari pengujian device fisik, dan **wajib dikalibrasi ulang setelah Fase 2/3** dengan data sensor MAX30102 asli dari 7 skenario pengujian fisik.

4. **Mekanisme Debounce di Backend**:
   - `PersistentAnomalyEvaluator`: **Strict Reset** (`count = 0` seketika saat terdeteksi normal, trigger alarm saat `count >= 3`).

---

## Perintah Eksekusi Pipeline

```bash
# 1. Harmonisasi & filter biologis (tanpa rishanmascarenhas)
python ml_pipeline/src/clean_datasets.py

# 2. Rekayasa fitur 13-dim & 3-way split
python ml_pipeline/src/feature_engineering.py

# 3. Training model & kalibrasi threshold
python ml_pipeline/src/train_isolation_forest.py

# 4. Verifikasi unit tests
pytest tests/ -v
```
