# LAPORAN EVALUASI PIPELINE ML PADA DATASET SINTETIS
> **STATUS**: SIMULATOR, BUKAN HASIL ALAT  
> **Catatan**: Seluruh data yang dianalisis dalam dokumen ini berasal dari simulator telemetri (`esp32-sim` / `origin=SYNTHETIC`), BUKAN rekaman pasien atau alat fisik riil. Test split telah dilihat sebelumnya dalam proses eksplorasi (bukan *unseen evaluation* sekali jalan).

---

## 1. Lingkungan Git & Integritas Repositori

- **Branch Saat Ini**: `recording-v2`
- **Pengecekan Aturan Git Ignore**:
  ```text
  .gitignore:136:data/	ml_pipeline/data/physio_synthetic
  .gitignore:89:ml_pipeline/models/*	ml_pipeline/models/physio_synthetic
  .gitignore:89:ml_pipeline/models/*	ml_pipeline/models/physio_synthetic/isolation_forest_model.joblib
  .gitignore:136:data/	ml_pipeline/data/physio_synthetic/features_train.csv
  ```
- **File Yang Dikelola**:
  - File kode implementasi: `ml_pipeline/src/train_physio_synthetic.py` (di-stage & commit).
  - File laporan: `docs/SIM_PIPELINE_REPORT.md` (di-stage & commit).
  - Data fitur dan model `.joblib` diabaikan oleh `.gitignore` dan **tidak akan ter-commit**.
  - File backend (`backend/app/config.py`), firmware, model lama, dan direktori `data/` tidak disentuh sama sekali.

---

## 2. Asal Usul Label & Karakteristik Dataset

- **File**: `Dataset/physio_monitor_synthetic_dataset (1).csv`
- **Kolom `origin`**: `SYNTHETIC` (100%)
- **Kolom `device_id`**: `esp32-sim` (100%)
- **Definisi Label Ground Truth**:
  - `ref_status`: Status referensi fisiologis (`NORMAL`, `ANOMALY`, `NO_DATA`).
  - `ref_OUTPUT`: Binerisasi referensi untuk pelatihan:
    - `0.0`: Kondisi fisiologis normal (`NORMAL`).
    - `1.0`: Kondisi anomali terinduksi (`ANOMALY`).
    - `NaN`: Kondisi sensor lepas / *warm-up* (`SENSOR_OFF` / `NO_DATA`).
- **Crosstab `fw_status` x `ref_status`**:
  ```text
  ref_status   ANOMALY   NORMAL   NO_DATA
  fw_status                            
  NORMAL           541    16858         0
  SENSOR_OFF         0        0       301
  ```
- **Distribusi Skenario per `split_role`**:
  - `train` (5.700 baris): `REST` (3.600), `TYPING` (1.200), `LIGHT_ACTIVITY` (900). Seluruh sampel normal (5.666 valid + 34 SENSOR_OFF), tanpa anomali.
  - `val` (5.400 baris): `REST` (1.200), `TYPING` (1.200), `FAN_AC` (1.200), `MOTION` (900), `FINGER_SHIFT` (900).
  - `test` (6.000 baris): `REST` (2.400), `TYPING` (900), `MOTION` (900), `FINGER_SHIFT` (900), `DHT_UNPLUGGED` (900).
  - `report_only` (600 baris): `COLD_HANDS` (300), `WALK_STAIRS` (300).
- **Rincian Panjang Tiap Sesi**:
  - `S01` (train, REST): 1.200 baris (Normal: 1.193, Anomali: 0, SENSOR_OFF: 7)
  - `S02` (train, REST): 1.200 baris (Normal: 1.196, Anomali: 0, SENSOR_OFF: 4)
  - `S03` (train, REST): 1.200 baris (Normal: 1.192, Anomali: 0, SENSOR_OFF: 8)
  - `S04` (train, TYPING): 1.200 baris (Normal: 1.193, Anomali: 0, SENSOR_OFF: 7)
  - `S05` (train, LIGHT_ACTIVITY): 900 baris (Normal: 892, Anomali: 0, SENSOR_OFF: 8)
  - `S06` (val, REST): 1.200 baris (Normal: 1.195, Anomali: 0, SENSOR_OFF: 5)
  - `S07` (val, TYPING): 1.200 baris (Normal: 1.192, Anomali: 0, SENSOR_OFF: 8)
  - `S08` (val, FAN_AC): 1.200 baris (Normal: 1.193, Anomali: 0, SENSOR_OFF: 7)
  - `S09` (val, MOTION): 900 baris (Normal: 705, Anomali: 191, SENSOR_OFF: 4)
  - `S10` (val, FINGER_SHIFT): 900 baris (Normal: 736, Anomali: 69, SENSOR_OFF: 95)
  - `S11` (test, REST): 1.200 baris (Normal: 1.195, Anomali: 0, SENSOR_OFF: 5)
  - `S12` (test, REST): 1.200 baris (Normal: 1.194, Anomali: 0, SENSOR_OFF: 6)
  - `S13` (test, TYPING): 900 baris (Normal: 892, Anomali: 0, SENSOR_OFF: 8)
  - `S14` (test, MOTION): 900 baris (Normal: 695, Anomali: 198, SENSOR_OFF: 7)
  - `S15` (test, FINGER_SHIFT): 900 baris (Normal: 711, Anomali: 83, SENSOR_OFF: 106)
  - `S16` (test, DHT_UNPLUGGED): 900 baris (Normal: 895, Anomali: 0, SENSOR_OFF: 5)
  - `S17` (report_only, WALK_STAIRS): 300 baris (Normal: 295, Anomali: 0, SENSOR_OFF: 5)
  - `S18` (report_only, COLD_HANDS): 300 baris (Normal: 294, Anomali: 0, SENSOR_OFF: 6)
- **Diff vs Versi Sebelumnya (`physio_monitor_synthetic_dataset.csv`)**:
  - Baris bertambah dari 12.900 baris menjadi 17.700 baris (+300 baris per sesi untuk S01–S16).
  - Kolom bertambah dari 13 kolom menjadi 24 kolom.
  - Kolom tambahan meliputi ground truth anotasi (`ref_status`, `ref_OUTPUT`, `split_role`, `report_group`) serta status inferensi simulasi runtime (`rule_severity`, `published_severity`, `label`, `decision_mode`, `decision_source`).

---

## 3. Analisis Transisi Label & Fenomena Alarm Palsu Skenario MOTION

### A. Transisi Dinamis di Dalam Sesi
Label `ref_OUTPUT` **berubah dinamis berkali-kali di dalam sesi** yang sama (bukan sesi statis satu label):
- **Sesi S09 (Val)**: 26 blok pergantian antara fase diam (`0.0`) dan burst gerakan (`1.0`).
  - Proporsi baris berlabel anomali (1): 191 / 900 baris (**21,22%**).
- **Sesi S14 (Test)**: 34 blok pergantian antara fase diam (`0.0`) dan burst gerakan (`1.0`).
  - Proporsi baris berlabel anomali (1): 198 / 900 baris (**22,00%**).

### B. Mengapa 19,2% Baris Normal Memicu Alarm (*Window Lag Effect*)
Pada skenario `MOTION` di test set (S14), dari 681 baris berlabel normal (`OUTPUT=0`), terdapat 131 baris (**19,24%**) yang ditandai anomali oleh model IF 13-Window pada ambang batas FPR 3%.

**Temuan Kuantitatif Pembuktian**:
- Sebanyak **131 dari 131 baris (100,0%)** alarm palsu tersebut terjadi tepat dalam rentang $\mathbf{\le 14}$ **baris pertama setelah burst anomali gerakan berhenti**!
- Sebanyak **0 baris (0,0%)** alarm palsu terjadi di luar jendela 14 baris pasca-anomali.
- **Penyebab Fisis & Matematis**:
  Fitur model mengandalkan *rolling window* berukuran 15 sampel ($W=15$). Saat gerakan fisik ekstrem berhenti, label ground truth seketika kembali ke `0.0`, namun buffer 15 detik masih menampung fluktuasi denyut nadi ekstrem dari beberapa detik sebelumnya.
  - Rata-rata `Var_Heart_Rate` pada True Normal murni: **7,50**
  - Rata-rata `Var_Heart_Rate` pada False Positive pasca-gerakan: **146,01** (hampir 20x lipat lebih tinggi!).
  Ini merupakan fenomena **Histeresis Jendela Geser (*Rolling Window Lag / Recovery Phase*)**, di mana fitur temporal mendeteksi dinamika pemulihan (*recovery dynamic*) sebelum sinyal benar-benar tenang kembali.

---

## 4. Evaluasi & Perbandingan Model (Operating Point FPR 3% Val)

Seluruh model dievaluasi pada partisi split bawaan yang sama:
- *Training*: Hanya sampel normal pada set `train` (S01–S05).
- *Ambang Batas*: Dikalibrasi pada data normal set `val` (S06–S10) pada target FPR = 3,0%.
- *Prevalensi Anomali*: Validation = 4,99%, Test = 4,86% (281 anomali / 5.779 baris).

### Tabel Perbandingan Metrik Global:
| Model | ROC-AUC | PR-AUC | PR-AUC - Prev | Val FPR | Test FPR | Test Recall | Test Prec | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Default (Vitals)** | 0.8790 | 0.7698 | 0.7212 | 0.0% | **0.00%** | 75.80% | **100.00%** | 0.8624 |
| **IF (3 Fitur Mentah)** | **0.9954** | **0.9384** | **0.8898** | 3.0% | 2.60% | **93.95%** | 64.86% | **0.7674** |
| **IF (13 Fitur Window)** | 0.9852 | 0.7444 | 0.6958 | 3.0% | 3.31% | 89.32% | 57.97% | 0.7031 |
| **One-Class SVM** | 0.9631 | 0.4089 | 0.3602 | 5.3% | 6.62% | 89.32% | 40.81% | 0.5603 |
| **LOF (Novelty)** | 0.9619 | 0.3935 | 0.3449 | 3.0% | 4.73% | 55.87% | 37.65% | 0.4499 |
| **Mahalanobis (EllipticEnv)** | 0.9583 | 0.3820 | 0.3334 | 3.0% | 4.62% | 56.58% | 38.50% | 0.4582 |

### Rincian Performa per Skenario pada Test Set:
- **IF (13 Fitur Window)**:
  - `REST`: FPR = 0,1% (3 FP)
  - `TYPING`: FPR = 0,2% (2 FP)
  - `DHT_UNPLUGGED`: FPR = 0,2% (2 FP)
  - `FINGER_SHIFT`: Recall = **98,8%** (82/83), FPR = 6,3%
  - `MOTION`: Recall = **85,4%** (169/198), FPR = 19,2% (efek window lag)
- **IF (3 Fitur Mentah)**:
  - `REST`: FPR = 2,7% (64 FP)
  - `TYPING`: FPR = 4,8% (42 FP)
  - `DHT_UNPLUGGED`: FPR = 1,0% (9 FP)
  - `FINGER_SHIFT`: Recall = 81,9% (68/83), FPR = 3,3%
  - `MOTION`: Recall = **99,0%** (196/198), FPR = **0,7%** (karena tidak ada window lag 15 detik)

---

## 5. Eksperimen Ekstensi DHT22 & Studi Ablasi Sensor

### A. Penanganan Skenario `DHT_UNPLUGGED` (Null Sensor Suhu & Kelembapan Ruang)
Pada sesi `S16` (`scenario == "DHT_UNPLUGGED"`), seluruh 900 baris memiliki `hum = NaN` dan `room_temp = NaN` (100% missing).
1. **Strategi Imputasi Median**:
   Mengisi nilai null dengan median data latih (`room_temp = 26.0 °C`, `hum = 65.0%`), sehingga `temp - room_temp` tetap terdefinisi.
   Hasil: Alarm palsu pada `DHT_UNPLUGGED` turun menjadi **0,00%** (0 false alarm dari 881 sampel uji).
2. **Strategi Fallback Model Routing (`FALLBACK_S3`)**:
   Sesuai label `report_group: FALLBACK_S3`, jika sensor DHT22 terputus, sistem secara otomatis beralih (*fallback*) ke model 13-fitur vital murni tanpa ambient, memastikan terputusnya DHT22 tidak memicu alarm palsu klinis pada pasien.

### B. Studi Ablasi Sensor (Uji Sensitivitas Fitur):
| Konfigurasi Fitur | ROC-AUC | PR-AUC | Test Recall | Test FPR | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Lengkap Vital + DHT22 (22 Fitur)** | 0.9791 | 0.7345 | 84.70% | 3.73% | Imputasi median pada DHT22. |
| **Vital Lengkap Tanpa DHT22 (13 Fitur)** | **0.9852** | **0.7444** | **89.32%** | **3.31%** | Baseline optimal. |
| **Ablasi: Tanpa DS18B20 (MAX30102 + DHT22)** | 0.9943 | 0.9075 | 94.66% | 2.33% | Performa tetap sangat tinggi. |
| **Ablasi: Tanpa MAX30102 (DS18B20 + DHT22)** | **0.5976** | **0.0614** | **0.71%** | 2.93% | **MODEL KOLAPS TOTAL!** |

**Kesimpulan Kritis Ablasi**:
Sensor fotopletismografi **MAX30102 (Heart Rate & SpO2) adalah sensor paling dominan dan esensial**. Tanpa MAX30102, anomali gerakan (`MOTION`) dan pergeseran jari (`FINGER_SHIFT`) sama sekali tidak dapat dideteksi oleh sensor suhu tubuh dan sensor lingkungan (Recall anjlok menjadi 0,71%).

---

## 6. Evaluasi Model Deep Learning (Autoencoder DenseAE)

### A. Pretrained Production Model (`dl_v1` - Transfer Zero-Shot dari Data Publik)
Model Autoencoder DenseAE v2 (`ml/artifacts/dl_v1/model.onnx` / `model_weights.npz`) yang telah dilatih pada dataset publik (2 fitur: `Heart_Rate`, `SpO2`) diuji langsung pada dataset sintetis:
- **Metrik Global**: $\text{ROC-AUC} = \mathbf{0.9060}$, $\text{PR-AUC} = \mathbf{0.7324}$ ($\text{PR-AUC} - \text{Prev} = 0.6845$).
- **Ambang Batas Default Metadata (`p95 = 0.013571`)**:
  - Tanpa Debounce: $\text{FPR} = 1,27\%$, $\text{Recall} = 65,84\%$, $\text{Precision} = 72,27\%$, $\text{F1} = 0.6890$.
  - Dengan Debounce Persistensi ($N=3$): $\text{FPR} = \mathbf{0.52\%}$, $\text{Recall} = 41,64\%$, $\text{Precision} = \mathbf{80.14\%}$, $\text{F1} = 0.5480$.
- **Ambang Batas Kalibrasi Val FPR 3%**:
  - $\text{FPR} = 5,93\%$, $\text{Recall} = 71,53\%$, $\text{Precision} = 37,78\%$, $\text{F1} = 0.4945$.

### B. Pelatihan Arsitektur DenseAE pada Dataset Sintetis
Arsitektur simetris DenseAE ($(16, 8, 2, 8, 16)$ encoder-decoder) dilatih langsung pada set data `train` normal sintetis:
1. **DenseAE 2D (Input: `Heart_Rate`, `SpO2`)**:
   - $\text{ROC-AUC} = \mathbf{0.9983}$, $\text{PR-AUC} = \mathbf{0.9939}$ ($\text{PR} - \text{Prev} = \mathbf{0.9460}$).
   - **Performa Test Set (Val FPR 3% Threshold)**:
     - Tanpa Debounce: $\text{FPR} = \mathbf{1.77\%}$, $\text{Recall} = \mathbf{99.29\%}$ (279 dari 281 anomali tertangkap), $\text{Precision} = 73,81\%$, $\text{F1} = \mathbf{0.8467}$.
     - Dengan Debounce Persistensi ($N=3$): $\text{FPR} = \mathbf{0.47\%}$ (hanya 26 false alarm), $\text{Recall} = 77,94\%$, $\text{Precision} = \mathbf{89.39\%}$, $\text{F1} = \mathbf{0.8327}$.
   - **Rincian Skenario Test (Tanpa Debounce)**:
     - `MOTION`: Recall = **100,0%** (198/198), FPR = 4,3% (30 FP)
     - `FINGER_SHIFT`: Recall = **97,6%** (81/83), FPR = 0,8% (6 FP)
     - `REST`: FPR = 1,3% (31 FP / 2.389)
     - `TYPING`: FPR = 2,8% (25 FP / 892)
     - `DHT_UNPLUGGED`: FPR = 0,8% (7 FP / 895)
2. **DenseAE 3D (Input: `Heart_Rate`, `SpO2`, `Temperature`)**:
   - $\text{ROC-AUC} = 0.9950$, $\text{PR-AUC} = 0.9820$, $\text{Test FPR} = 1,70\%$, $\text{Recall} = 97,86\%$, $\text{F1} = 0.8449$.
3. **DenseAE 13D (Input: 13 Fitur Window Temporal)**:
   - $\text{ROC-AUC} = 0.9699$, $\text{PR-AUC} = 0.4708$, $\text{Test FPR} = 4,35\%$, $\text{Recall} = 70,11\%$, $\text{F1} = 0.5495$.

### C. Ringkasan Komparatif ML vs DL pada Data Sintetis
- Model Deep Learning Autoencoder (**DenseAE 2D**) yang dilatih langsung pada data sintetis menghasilkan metrik akurasi tertinggi di antara seluruh pendekatan ($\text{ROC-AUC} = \mathbf{0.9983}$, $\text{PR-AUC} = \mathbf{0.9939}$, $\text{F1} = \mathbf{0.8467}$).
- Berbeda dari model 13-window temporal yang terdampak oleh *window lag* sebesar 19,2% pasca-gerakan, model DL per-baris DenseAE 2D langsung mengenali pemulihan sinyal secara instan (FPR pada MOTION hanya 4,3% dengan Recall 100,0%).
- Penambahan filter persistensi ($N=3$) pada DL secara drastis menekan tingkat alarm palsu ke level minimal (**0,47%** atau kurang dari 1 alarm palsu per 200 sampel) dengan presisi mendekati **90%**.
