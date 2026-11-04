# LAPORAN EVALUASI PIPELINE ML & DL PADA DATASET SINTETIS
> **STATUS**: SIMULATOR, BUKAN HASIL ALAT  
> **Pemberitahuan Metodologi Kritis**: Seluruh data yang dianalisis dalam dokumen ini berasal dari simulator telemetri (`esp32-sim` / `origin=SYNTHETIC`), BUKAN rekaman pasien atau alat fisik riil. Test split telah dilihat berulang kali (*repeatedly inspected*) oleh agent dalam proses iterasi dan eksplorasi, serta **TIDAK ADA protokol pra-registrasi (*no pre-registration*)** untuk dataset simulator ini. Semua hasil harus diperlakukan sebagai evaluasi eksploratori retrospektif, bukan pembuktian prospektif terdaftar (*registered trial*).

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
  - File implementasi evaluasi: `ml_pipeline/src/evaluate_dl_simulator.py` & `ml_pipeline/src/train_physio_synthetic.py` (di-stage & commit).
  - File laporan: `docs/SIM_PIPELINE_REPORT.md` (di-stage & commit).
  - Seluruh data `.joblib`, bobot `.pt`, dan data turunan diabaikan oleh `.gitignore` dan **tidak akan ter-commit**.
  - File backend (`backend/app/config.py`), firmware, artefak produksi publik (`ml/artifacts/dl_v1`), dan direktori `data/` tidak disentuh sama sekali.

---

## 2. Asal Usul Label & Karakteristik Dataset

- **File**: `Dataset/physio_monitor_synthetic_dataset (1).csv`
- **Kolom `origin`**: `SYNTHETIC` (100%)
- **Kolom `device_id`**: `esp32-sim` (100%)
- **Definisi Label Ground Truth**:
  - `ref_status`: Status referensi fisiologis (`NORMAL`, `ANOMALY`, `NO_DATA`).
  - `ref_OUTPUT`: Binerisasi referensi untuk evaluasi/pelatihan:
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
  - `val` (5.400 baris): `REST` (1.200), `TYPING` (1.200), `FAN_AC` (1.200), `MOTION` (900), `FINGER_SHIFT` (900). Total valid normal: 5.021, anomali: 260.
  - `test` (6.000 baris): `REST` (2.400), `TYPING` (900), `MOTION` (900), `FINGER_SHIFT` (900), `DHT_UNPLUGGED` (900). Total valid normal: 5.582, anomali: 281.
  - `report_only` (600 baris): `COLD_HANDS` (300), `WALK_STAIRS` (300).
- **Rincian Sesi Uji (Test Set)**:
  - `S11` (test, REST): 1.195 normal, 0 anomali, 5 SENSOR_OFF
  - `S12` (test, REST): 1.194 normal, 0 anomali, 6 SENSOR_OFF
  - `S13` (test, TYPING): 892 normal, 0 anomali, 8 SENSOR_OFF
  - `S14` (test, MOTION): 695 normal, 198 anomali, 7 SENSOR_OFF (16 anomaly bursts)
  - `S15` (test, FINGER_SHIFT): 711 normal, 83 anomali, 106 SENSOR_OFF (13 anomaly bursts)
  - `S16` (test, DHT_UNPLUGGED): 895 normal, 0 anomali, 5 SENSOR_OFF (100% DHT null)

---

## 3. Analisis Skenario MOTION: Perbandingan Window 13 vs Fitur Mentah (IF Raw & DenseAE)

### A. Fenomena Transisi Dinamis
Label `ref_OUTPUT` berubah dinamis berkali-kali di dalam sesi S14 (34 segmen transisi). Dari 900 baris data di S14, terdapat 198 baris anomali gerakan yang terbagi ke dalam **16 distinct bursts**, diselingi fase diam/normal (695 baris normal valid).

### B. Komparasi False Alarm S14: Window 13 vs Fitur Mentah
Pada analisis awal menggunakan model **IF (13 Fitur Window)**, skenario `MOTION` memicu alarm palsu pada 131 baris normal (**19,24% FPR pada MOTION**).
Ketika dibandingkan langsung dengan model berbasis **fitur mentah per-baris (raw per-row)** pada target kalibrasi yang sama:

| Model | Fitur Input | FP pada S14 Normal (695) | FPR pada MOTION | Recall MOTION (198) | Keterangan Dinamika |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **IF (13 Fitur Window)** | 13 Fitur Rolling Window ($W=15$) | 131 | **19,24%** | 85,35% (169/198) | Terkena *window lag* 14 detik pasca-gerak |
| **IF (2D raw)** | 2 Fitur Mentah (`hr`, `spo2`) | **11** | **1,58%** | **100,00%** (198/198) | Bebas buffer temporal; respons instan |
| **IF (3D raw)** | 3 Fitur Mentah (`hr`, `spo2`, `temp`) | **10** | **1,44%** | **100,00%** (198/198) | Bebas buffer temporal; respons instan |
| **DenseAE 2D (seed 42)**| 2 Fitur Mentah (`hr`, `spo2`) | **4** | **0,58%** | **100,00%** (198/198) | Rekonstruksi instan tanpa histeresis |

**Temuan Pembuktian**:
1. Sebanyak **131 dari 131 FP (100,0%)** pada model IF 13-window terjadi tepat pada rentang $\le 14$ baris pertama setelah anomali gerakan berhenti. Pada rentang tersebut, buffer 15 detik masih menampung varians sinyal ekstrem (`Var_Heart_Rate` rata-rata 146,01 vs 7,50 pada true normal).
2. Model dengan **fitur mentah per-baris (IF raw dan DenseAE)** sama sekali tidak memiliki buffer masa lalu, sehingga false alarm pada transisi gerak langsung turun drastis dari **19,24% menjadi 0,58% – 1,58%** dengan Recall anomali gerakan tetap sempurna (**100,0%**).
3. Tingginya false positive pada skenario MOTION pada model window bukanlah kegagalan deteksi anomali fisis, melainkan **artefak histeresis rolling window**.

---

## 4. Evaluasi Menyeluruh Model ML & DL pada Test Set

Ambang batas seluruh model kontinu dikalibrasi pada data normal set validasi (`val_normal`, 5.021 baris) dengan target FPR 3,0% (persentil 97 dari distribusi skor normal val). Ambang univariat dikalibrasi pada parameter persentil normal train untuk menghasilkan FPR 3,0% pada validasi normal. Rule Default menggunakan batas klinis standar.

### A. Tabel Titik Operasi Tanpa Debounce (Raw / $N=1$)
*Catatan: Kolom Test FPR menyajikan FPR aktual pada test set (5.582 baris normal). ROC-AUC dan PR-AUC hanya dilaporkan untuk model skor kontinu; model diskrit (Rule & Ambang Univariat) dilaporkan sebagai titik operasi.*

| Model | ROC-AUC | PR-AUC | Test FPR Aktual | Recall | Presisi | F1 | Burst Recall (29) | Med Lat | FPR tol14 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Default (2D)** | N/A | N/A | **0,00%** | 75,80% | **100,00%** | 0,8623 | 96,6% (28) | **0,0s** | **0,00%** |
| **Rule Default (3D)** | N/A | N/A | **0,00%** | 75,80% | **100,00%** | 0,8623 | 96,6% (28) | **0,0s** | **0,00%** |
| **Univariate (2D, q=3%)** | N/A | N/A | 1,24% | 98,58% | 80,06% | **0,8836** | **100,0%** (29) | **0,0s** | 1,23% |
| **Univariate (3D, q=3%)** | N/A | N/A | 1,24% | 98,58% | 80,06% | **0,8836** | **100,0%** (29) | **0,0s** | 1,23% |
| **dl_v1 Zero-Shot (2D)** | 0,9060 | 0,7324 | 5,93% | 71,53% | 37,78% | 0,4945 | 96,6% (28) | **0,0s** | 6,14% |
| **IF (2D raw)** | 0,9965 | 0,9776 | 1,40% | 98,93% | 78,09% | 0,8728 | **100,0%** (29) | **0,0s** | 1,33% |
| **IF (3D raw)** | 0,9952 | 0,9392 | 2,74% | 95,02% | 63,57% | 0,7618 | **100,0%** (29) | **0,0s** | 2,71% |
| **Mahalanobis (2D)** | 0,9978 | 0,9918 | 3,22% | **99,64%** | 60,87% | 0,7557 | **100,0%** (29) | **0,0s** | 2,90% |
| **Mahalanobis (3D)** | 0,9989 | 0,9909 | 3,89% | 98,93% | 56,16% | 0,7165 | **100,0%** (29) | **0,0s** | 3,62% |
| **LOF (2D)** | 0,9985 | 0,9921 | 1,65% | 98,58% | 75,07% | 0,8523 | **100,0%** (29) | **0,0s** | 1,62% |
| **LOF (3D)** | **0,9991** | 0,9935 | 2,13% | 99,29% | 70,10% | 0,8218 | **100,0%** (29) | **0,0s** | 2,06% |
| **OCSVM (2D)** | 0,9952 | 0,9929 | 1,42% | 99,29% | 77,93% | 0,8732 | **100,0%** (29) | **0,0s** | 1,38% |
| **OCSVM (3D)** | 0,9974 | **0,9938** | 1,76% | 99,29% | 74,01% | 0,8480 | **100,0%** (29) | **0,0s** | 1,81% |
| **DenseAE 2D (seed 42)** | 0,9971 | 0,9939 | 1,52% | 99,29% | 76,65% | 0,8651 | **100,0%** (29) | **0,0s** | 1,44% |
| **DenseAE 3D (seed 42)** | 0,9985 | 0,9864 | 3,67% | 98,22% | 57,38% | 0,7244 | **100,0%** (29) | **0,0s** | 3,69% |

---

### B. Tabel Performa dengan Persistence Debounce ($N=3$)

*Debounce $N=3$ mengharuskan alarm terpicu selama 3 detik berturut-turut di dalam sesi yang sama sebelum status dinyatakan anomali.*

| Model + $N=3$ | Test FPR Aktual | Recall | Presisi | F1 | Burst Recall (29) | Med Lat | FPR tol14 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Default (2D) + N3** | **0,00%** | 51,60% | **100,00%** | 0,6808 | 58,6% (17) | 2,0s | **0,00%** |
| **Rule Default (3D) + N3** | **0,00%** | 51,60% | **100,00%** | 0,6808 | 58,6% (17) | 2,0s | **0,00%** |
| **Univariate (2D) + N3** | 0,30% | 76,51% | 92,67% | 0,8382 | **100,0%** (29) | 2,0s | 0,27% |
| **Univariate (3D) + N3** | 0,30% | 76,51% | 92,67% | 0,8382 | **100,0%** (29) | 2,0s | 0,27% |
| **dl_v1 Zero-Shot + N3** | 3,87% | 47,33% | 38,11% | 0,4222 | 55,2% (16) | 2,0s | 4,04% |
| **IF (2D raw) + N3** | **0,25%** | **77,94%** | **93,99%** | **0,8521** | **100,0%** (29) | 2,0s | **0,19%** |
| **IF (3D raw) + N3** | 1,36% | 70,11% | 72,16% | 0,7112 | 89,7% (26) | 2,0s | 1,31% |
| **Mahalanobis (2D) + N3** | 1,67% | 78,29% | 70,29% | 0,7407 | **100,0%** (29) | 2,0s | 1,38% |
| **Mahalanobis (3D) + N3** | 2,10% | 77,22% | 64,97% | 0,7057 | **100,0%** (29) | 2,0s | 1,90% |
| **LOF (2D) + N3** | 0,48% | 76,87% | 88,89% | 0,8244 | **100,0%** (29) | 2,0s | 0,44% |
| **LOF (3D) + N3** | 0,84% | 77,58% | 82,26% | 0,7985 | **100,0%** (29) | 2,0s | 0,71% |
| **OCSVM (2D) + N3** | 0,39% | 77,94% | 90,87% | 0,8391 | **100,0%** (29) | 2,0s | 0,35% |
| **OCSVM (3D) + N3** | 0,66% | 77,94% | 85,55% | 0,8156 | **100,0%** (29) | 2,0s | 0,67% |
| **DenseAE 2D + N3 (seed 42)**| 0,56% | 77,94% | 87,60% | 0,8249 | **100,0%** (29) | 2,0s | 0,52% |
| **DenseAE 3D + N3 (seed 42)**| 1,65% | 77,58% | 70,32% | 0,7377 | **100,0%** (29) | 2,0s | 1,67% |

---

### C. Recall pada Ketat Test FPR (Kurva ROC Non-Interpolated)

*Dihitung langsung dari kurva ROC empiris tanpa interpolasi linear:*

| Model | Recall @ FPR $\le$ 0,1% | Recall @ FPR $\le$ 0,5% | Recall @ FPR $\le$ 1,0% | Recall @ FPR $\le$ 3,0% |
| :--- | :---: | :---: | :---: | :---: |
| **dl_v1 Zero-Shot (2D)** | 63,35% | 64,41% | 64,77% | 67,62% |
| **IF (2D raw)** | 85,05% | 92,17% | 96,80% | **99,29%** |
| **IF (3D raw)** | 69,04% | 83,99% | 86,83% | 95,37% |
| **Mahalanobis (2D)** | 97,15% | 98,22% | 98,58% | 98,93% |
| **Mahalanobis (3D)** | 97,15% | 97,51% | 98,22% | 98,93% |
| **LOF (2D)** | 97,51% | 98,58% | 98,58% | **99,29%** |
| **LOF (3D)** | 97,15% | 97,86% | 98,93% | **99,29%** |
| **OCSVM (2D)** | **98,22%** | **99,29%** | **99,29%** | **99,29%** |
| **OCSVM (3D)** | 97,86% | 98,58% | 98,93% | **99,29%** |
| **DenseAE 2D (seed 42)** | **98,22%** | 98,93% | **99,29%** | **99,29%** |
| **DenseAE 3D (seed 42)** | 95,73% | 97,15% | 97,15% | 98,22% |

---

### D. Hasil Pelatihan DenseAE (5 Seeds: 42–46, Mean $\pm$ SD)

Model PyTorch DenseAutoencoder (`ml/dl/models.py`) dilatih secara deterministik pada 5 seed acak berturut-turut:

1. **DenseAE 2D (`hr`, `spo2`)**:
   - **Tanpa Debounce (Raw, $N=1$)**:
     - $\text{ROC-AUC} = 0,9967 \pm 0,0014$
     - $\text{PR-AUC} = 0,9932 \pm 0,0007$
     - $\text{Test FPR} = 2,24\% \pm 0,55\%$
     - $\text{Recall} = 99,29\% \pm 0,00\%$
     - $\text{Presisi} = 70,55\% \pm 5,34\%$
     - $\text{F1-Score} = 0,8165 \pm 0,0367$
     - $\text{Burst Recall} = 100,0\% \pm 0,0\%$ (29/29)
     - $\text{FPR tol14} = 2,28\% \pm 0,63\%$
   - **Dengan Persistence ($N=3$)**:
     - $\text{Test FPR} = 0,79\% \pm 0,27\%$
     - $\text{Recall} = 77,94\% \pm 0,00\%$
     - $\text{Presisi} = 84,04\% \pm 4,77\%$
     - $\text{F1-Score} = 0,8054 \pm 0,0217$
     - $\text{Burst Recall} = 100,0\% \pm 0,0\%$ (29/29)
     - $\text{FPR tol14} = 0,80\% \pm 0,31\%$

2. **DenseAE 3D (`hr`, `spo2`, `temp`)**:
   - **Tanpa Debounce (Raw, $N=1$)**:
     - $\text{ROC-AUC} = 0,9984 \pm 0,0003$
     - $\text{PR-AUC} = 0,9874 \pm 0,0026$
     - $\text{Test FPR} = 3,65\% \pm 0,37\%$
     - $\text{Recall} = 98,43\% \pm 0,48\%$
     - $\text{F1-Score} = 0,7269 \pm 0,0206$
     - $\text{Burst Recall} = 100,0\% \pm 0,0\%$ (29/29)
     - $\text{FPR tol14} = 3,64\% \pm 0,47\%$
   - **Dengan Persistence ($N=3$)**:
     - $\text{Test FPR} = 1,57\% \pm 0,16\%$
     - $\text{Recall} = 76,87\% \pm 0,81\%$
     - $\text{F1-Score} = 0,7391 \pm 0,0089$
     - $\text{Burst Recall} = 100,0\% \pm 0,0\%$ (29/29)
     - $\text{FPR tol14} = 1,55\% \pm 0,21\%$

---

## 5. Analisis Kritis: Kegagalan Zero-Shot dan Realitas Kinerja DL vs ML

### A. Evaluasi Objektif: DL Tidak Unggul Mutlak
Klaim sebelumnya bahwa model Deep Learning Autoencoder (*DenseAE*) lebih unggul daripada seluruh pendekatan baseline **tidak terbukti secara empiris**:
1. Pada mode persistensi $N=3$, **Isolation Forest 2D raw** menghasilkan F1-Score yang lebih tinggi (**0,8521** vs **0,8054** mean pada DenseAE 2D) dengan Test FPR yang jauh lebih rendah (**0,25%** vs **0,79%** mean pada DenseAE 2D).
2. **Univariate Thresholding** sederhana per-sensor menghasilkan F1-Score sebesar **0,8382** dengan Test FPR hanya **0,30%** (debounced) dan menangkap 100% burst anomali (29/29), tanpa memerlukan komputasi matriks densitas tinggi ataupun pelatihan neural network.
3. Penambahan fitur suhu (`temp`) pada DenseAE 3D justru mendegradasi F1-Score dari 0,8165 ke 0,7269 (raw) dan menaikkan Test FPR dari 2,24% menjadi 3,65%, karena anomali simulator terfokus pada dinamika PPG fotopletismografi.

### B. Evaluasi Transfer Zero-Shot `dl_v1`: Terjadi Degradasi Domain
Model produksi pra-latih `dl_v1` (yang dilatih pada dataset publik 3 sumber: Nasir Ayub, Engr Arri, Gourango Mandal) **gagal melakukan transfer zero-shot secara andal** ke dataset simulator telemetri ini:
- **Test FPR Aktual Melonjak**: 5,93% pada raw ($N=1$) dan 3,87% pada $N=3$ (jauh melampaui toleransi target 3%).
- **Presisi Rendah**: Presisi hanya mencapai 37,78% (raw) dan 38,11% ($N=3$), dengan lebih dari 60% alarm adalah false alarm.
- **Deteksi Kejadian Terbatas**: Recall anomali hanya mencapai 71,53% (raw) dan anjlok ke 47,33% ($N=3$). Burst Recall hanya 55,2% (hanya 16 dari 29 burst yang terdeteksi dengan $N=3$).
- **Penyebab**: Pergeseran distribusi (*domain covariate shift*) antara profil noise sensor MAX30102 pada data publik klinis vs karakteristik sintesis generator telemetri ESP32. Model produksi membutuhkan kalibrasi domain spesifik (*domain adaptation*) sebelum dapat digunakan pada simulator ini.

---

## 6. Keterbatasan Sampel Uji & Analisis Ketidakandalan CI Bootstrap

### A. Karakteristik Kluster Test Set
- **Jumlah Sesi Uji**: 6 sesi total (S11–S16).
- **Jumlah Sesi Beranomali**: HANYA **2 sesi** (`S14 MOTION` dan `S15 FINGER_SHIFT`).
- **Sesi Bebas Anomali**: 4 sesi (`S11 REST`, `S12 REST`, `S13 TYPING`, `S16 DHT_UNPLUGGED`).
- **Total Anomaly Bursts**: **29 bursts** (16 di S14, 13 di S15).

### B. Mengapa Interval Kepercayaan (CI) Bootstrap Kluster Tidak Dapat Dihitung Secara Andal
1. **Jumlah Kluster Positif Terlalu Sedikit ($M=2$)**:
   Metode block / cluster bootstrap pada tingkat sesi mengharuskan jumlah kluster independen yang memadai ($M \ge 30-50$). Dengan hanya 2 sesi anomali, setiap penarikan sampel dengan pengembalian (*resampling with replacement*) memiliki probabilitas 25% hanya menarik S14, 25% hanya menarik S15, dan 50% menarik keduanya. Varians antar-penarikan menjadi sangat ekstrem dan estimasi interval parameter mengalami bias kluster parah.
2. **Ketergantungan Skenario**:
   Burst resampling (29 bursts) menghasilkan estimasi empiris 95% CI Burst Recall $[100,0\%, 100,0\%]$ untuk DenseAE 2D dan IF 2D raw. Namun interval ini **terlalu optimistik (degenerate)** karena seluruh 29 burst hanya mewakili dua variasi fisik buatan simulator telemetri (gerakan tangan dan pergeseran sensor), bukan variasi populasi pasien yang independen.

---

## 7. Verifikasi Integritas dan Urutan Data

- **Verifikasi Urutan Indeks**:
  Seluruh Series prediksi debounced ($N=3$) diverifikasi secara programatis:
  ```python
  assert (pred_deb.index == test_df.index).all()
  ```
  Hasil pengujian: **PASSED (100% konsisten)**.
- **Integritas Artefak**:
  Tidak ada modifikasi pada file runtime produksi `ml/artifacts/dl_v1/`, firmware, konfigurasi backend, atau database live. Evaluasi ini murni dijalankan pada script terisolasi `ml_pipeline/src/evaluate_dl_simulator.py`.
