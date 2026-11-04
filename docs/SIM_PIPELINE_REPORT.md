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
