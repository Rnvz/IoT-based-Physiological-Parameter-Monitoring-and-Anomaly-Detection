# Physiological IoT Anomaly Detection: Evaluation Protocol

Dokumen ini mendefinisikan protokol evaluasi baku untuk benchmarking model deteksi anomali pada sistem pemantauan fisiologis IoT (ESP32 + MAX30102 + DS18B20). Protokol ini mengikat seluruh model baseline maupun model deep learning masa depan.

---

## 1. Sumber Data (Data Sources)

1. **Dataset Publik**:
   - Dataset publik (seperti `nasirayub2`, `engrarri21`, `gourangomandal`) digunakan semata-mata untuk eksperimen awal pra-latih (pre-training) atau sanity check pipeline.
   - **Batasan Validitas**: Label anomali pada dataset publik sebagian besar diturunkan dari aturan ambang batas statis (misal: alert kolom pada `gourangomandal`) atau bergantung pada variabel klinis yang tidak diukur oleh perangkat IoT ini (seperti tekanan darah sistolik/diastolik dan BMI pada `nasirayub2`). Oleh karena itu, performa pada dataset publik **tidak dapat dijadikan klaim ilmiah** mengenai efektivitas deteksi anomali pada perangkat fisik.
2. **Data Perekaman Perangkat Sendiri (Own-Device Recordings)**:
   - Menjadi sumber kebenaran (ground truth) utama untuk evaluasi performa operasional sistem IoT.

---

## 2. Pengambilan Data Mandiri & Pembagian Partisi (Session Splitting)

1. **Multi-Sesi Perekaman**:
   - Data harus direkam dalam beberapa sesi terpisah pada hari yang berbeda, waktu yang berbeda, dan kondisi ruangan/suhu ambien yang bervariasi.
2. **Partisi Berbasis Sesi (By-Session Split)**:
   - Pembagian data ke dalam partisi **Train**, **Validation**, dan **Test** WAJIB dilakukan berdasarkan **Sesi Rekaman Penuh (Session-wise)**, BUKAN secara acak per baris (*random row split*).
   - Random row split pada data deret waktu fisiologis menyebabkan kebocoran data (*data leakage*) akibat autokorelasi temporal tinggi antar-sampel berdekatan.
3. **Hold-out Test Set yang Terkunci**:
   - Sesi rekaman yang dialokasikan untuk Test Set final harus disimpan dan **tidak boleh disentuh sama sekali** selama proses pelatihan, pemilihan fitur, maupun penyesuaian hyperparameter.

---

## 3. Kondisi Pengujian & Skenario (Test Conditions)

Evaluasi model harus mencakup dua kelompok kondisi yang dipisahkan secara tegas dalam pelaporan:

1. **Skenario Artefak Benign (Benign Operational Scenarios)**:
   - Aktivitas normal non-klinis yang diberi tag metadata via endpoint `/session/scenario`:
     - `REST`: Duduk tenang / istirahat santai (baseline).
     - `TYPING`: Mengetik keyboard komputer / interaksi kerja.
     - `LIGHT_ACTIVITY`: Peregangan ringan / berbicara.
     - `MOTION`: Gerakan lengan atau getaran fisik wajar.
     - `FINGER_SHIFT`: Pergeseran posisi jari pada sensor MAX30102.
     - `SENSOR_OFF`: Pelepasan probe / jari terangkat.
     - `OTHER`: Aktivitas harian jinak lainnya.
   - *Tujuan*: Menguji ketahanan model terhadap alarm palsu (*false alarms*) pada aktivitas normal.
2. **Anomali Sintetis Terinjeksi (Injected Synthetic Anomalies)**:
   - Injeksi sinyal anomali terukur pada rekaman baseline bersih:
     - Lonjakan denyut jantung (*tachycardia/bradycardia step spike*).
     - Penurunan saturasi oksigen (*hypoxemia drop* $< 90\%$).
     - Sinyal beku / datar (*sensor flatline*).
     - Perubahan mendadak di luar batas biologis (*out-of-range artifacts*).
   - *Aturan Pelaporan*: Seluruh anomali sintetis WAJIB dilaporkan secara terpisah dari data rekaman riil; dilarang mencampuradukkan injeksi sintetis sebagai data klinis asli.

---

## 4. Metrik Evaluasi Terstandarisasi (Metrics)

Model dievaluasi menggunakan kumpulan metrik kuantitatif berikut:

1. **PR-AUC (Precision-Recall Area Under Curve)**:
   - Metrik utama untuk data deteksi anomali yang mengalami ketidakseimbangan kelas (*class imbalance*) ekstrem.
2. **ROC-AUC (Receiver Operating Characteristic Area Under Curve)**:
   - Mengukur kemampuan separasi global model di seluruh rentang ambang batas.
3. **F1-Score pada Titik Operasi Baku (Fixed Operating Threshold)**:
   - Harmonic mean antara Precision dan Recall pada ambang batas FPR yang ditentukan (misal FPR $\le 5\%$).
4. **Tingkat Alarm Palsu per Jam (False Alarms per Hour)**:
   - Dihitung pada tingkat Recall target tertentu (misal pada Recall $\ge 95\%$) pada sesi aktivitas benign (`REST`, `TYPING`).
5. **Latensi Deteksi (Detection Latency)**:
   - Jumlah sampel atau detik yang dibutuhkan oleh filter debouncing dan model sejak onset anomali hingga status alarm dipublikasikan.
6. **Latensi Inferensi & Ukuran Model (Resource Footprint)**:
   - Waktu inferensi per sampel (mean, p95, p99 dalam milidetik).
   - Ukuran model terkompresi di disk (kilobyte/megabyte) dan konsumsi memori RAM.

---

## 5. Model Pembanding Baku (Baselines)

Seluruh model baru (termasuk model Deep Learning seperti LSTM-Autoencoder atau 1D-CNN) harus dibandingkan secara langsung (*head-to-head*) dengan model baseline berikut:

1. **Rule-Based Engine**: Logika deterministik ambang batas medis/operasional (`rule_severity`).
2. **Isolation Forest**: Model pohon berbasis isolasi fitur partisi acak (baseline saat ini).
3. **Local Outlier Factor (LOF)**: Algoritma berbasis kepadatan lokal (*density-based*, `novelty=True`).
4. **One-Class Support Vector Machine (OC-SVM)**: Estimasi batas distribusi data normal berbasis kernel RBF.

> **Persyaratan Komparasi**: Setiap model baru dan model pembanding WAJIB dievaluasi pada partisi sesi yang identik, urutan fitur yang identik, dan script perhitungan metrik yang sama persis tanpa pengecualian.

---

## 6. Aturan Pembekuan Protokol (Freeze Rule)

1. **Pemisahan Partisi Terkunci**:
   - Pembagian sesi rekaman (Train / Val / Test) dikunci sebelum pengujian dimulai.
2. **Script Metrik Terkunci**:
   - Kode evaluasi dan formula perhitungan metrik dibekukan di dalam repositori.
3. **Tanpa Tuning pada Test Set**:
   - Dilarang keras melakukan tuning hyperparameter, pemilihan threshold ulang, atau eksplorasi data pada Test Set.
   - Evaluasi pada Test Set hanya dilakukan **satu kali** sebagai verifikasi akhir performa generalisasi model.
