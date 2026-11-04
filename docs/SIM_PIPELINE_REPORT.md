# LAPORAN EVALUASI PIPELINE ML & DL PADA DATASET SINTETIS
> **STATUS**: SIMULATOR, BUKAN HASIL ALAT  
> **Pemberitahuan Metodologi Kritis**: Seluruh data yang dianalisis dalam dokumen ini berasal dari simulator telemetri (`esp32-sim` / `origin=SYNTHETIC`), BUKAN rekaman pasien atau alat fisik riil. Test split telah dilihat berulang kali (*repeatedly inspected*) oleh agent dalam proses eksplorasi dan iterasi eksperimen, serta **TIDAK ADA protokol pra-registrasi (*no pre-registration*)** untuk dataset simulator ini. Semua hasil harus diperlakukan sebagai evaluasi eksploratori retrospektif, bukan pembuktian prospektif terdaftar (*registered trial*).

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
  - File implementasi evaluasi: `ml_pipeline/src/evaluate_dl_simulator.py` (di-stage & commit).
  - File panduan environment: `docs/ENV_DL_TRAIN.md` (di-stage & commit).
  - File laporan: `docs/SIM_PIPELINE_REPORT.md` (di-stage & commit).
  - Data fitur dan model `.joblib` diabaikan oleh `.gitignore` dan **tidak akan ter-commit**.
  - File backend (`backend/app/config.py`), firmware, artefak produksi publik (`ml/artifacts/dl_v1`), dan direktori `data/` tidak disentuh sama sekali.

---

## 2. Asal Usul Label & Verifikasi Kolom Dataset

- **File**: `Dataset/physio_monitor_synthetic_dataset (1).csv`
- **Kolom `origin`**: `SYNTHETIC` (100%)
- **Kolom `device_id`**: `esp32-sim` (100%)
- **Definisi Label Ground Truth**:
  - `ref_status`: Status referensi fisiologis (`NORMAL`, `ANOMALY`, `NO_DATA`).
  - `ref_OUTPUT`: Binerisasi referensi untuk evaluasi/pelatihan:
    - `0.0`: Kondisi fisiologis normal (`NORMAL`).
    - `1.0`: Kondisi anomali terinduksi (`ANOMALY`).
    - `NaN`: Kondisi sensor lepas / *warm-up* (`SENSOR_OFF` / `NO_DATA`).
- **Verifikasi Rule Default terhadap Kolom `rule_severity` Dataset**:
  Fungsi evaluasi `evaluate_rules(hr, spo2, temp)` dari `backend/app/rules/engine.py` dicocokkan baris-per-baris dengan kolom `rule_severity` mentah dataset:
  1. **Pada Baris Valid (`fw_status == 'NORMAL'`, 17.399 baris)**: **100,0000% COCOK** (17.399 / 17.399).
  2. **Pada Test Set Split (5.863 baris)**: **100,0000% COCOK** (5.863 / 5.863).
  3. **Pada Keseluruhan Dataset (17.700 baris)**: **98,2994% COCOK** (17.399 / 17.700).  
     Sebanyak 301 mismatch seluruhnya terjadi pada baris `fw_status == 'SENSOR_OFF'` di mana dataset mencatat `rule_severity = NaN` (karena sensor lepas), sedangkan fungsi aturan mentah mengevaluasi input 0.0 sebagai anomali kritis severity 2.
- **Distribusi Skenario per `split_role`**:
  - `train` (5.700 baris): `REST` (3.600), `TYPING` (1.200), `LIGHT_ACTIVITY` (900). Seluruh sampel normal (5.666 valid + 34 SENSOR_OFF), tanpa anomali.
  - `val` (5.400 baris): `REST` (1.200), `TYPING` (1.200), `FAN_AC` (1.200), `MOTION` (900), `FINGER_SHIFT` (900). Total valid normal: 5.021, anomali: 260.
  - `test` (6.000 baris): `REST` (2.400), `TYPING` (900), `MOTION` (900), `FINGER_SHIFT` (900), `DHT_UNPLUGGED` (900). Total valid normal: 5.582, anomali: 281.
  - `report_only` (600 baris): `COLD_HANDS` (300), `WALK_STAIRS` (300).

---

## 3. Analisis Skenario Anomali Test Set (S14 MOTION & S15 FINGER_SHIFT)

### A. Rincian Statistik 29 Anomaly Bursts
Pada test set, seluruh 281 baris anomali terkelompok ke dalam **29 distinct bursts**:
- Sesi `S14` (`MOTION`): **16 bursts** (total 198 baris anomali, 695 baris normal).
- Sesi `S15` (`FINGER_SHIFT`): **13 bursts** (total 83 baris anomali, 711 baris normal).
- Sesi `S11`, `S12`, `S13`, `S16`: Bebas anomali (0 burst).

**Distribusi Durasi / Panjang Burst**:
- **Min**: 5 baris (5 detik)
- **Median**: 6,0 baris (6 detik)
- **Maks**: 19 baris (19 detik)
- **Burst dengan durasi < 3 baris**: **0 dari 29 burst (0,0%)**!
- *Implikasi Fisis*: Karena seluruh burst berlangsung minimal 5 sampel berturut-turut, tidak ada burst yang terlewat semata-mata karena durasinya lebih pendek dari ukuran jendela filter persistensi ($N=3$).

---

### B. Output Mentah FP dan Recall per Skenario (S14 & S15)

Evaluasi per skenario untuk model DenseAE 2D, IF 2D, dan IF 3D (5 seeds acak 42–46, dilaporkan Mean $\pm$ SD dan Seed 42):

#### 1. Skenario S14 (MOTION: 695 Normal, 198 Anomali)
- **Mode RAW ($N=1$)**:
  - `DenseAE 2D` (5 seeds): $\text{FP} = 8,00 \pm 6,29$ ($\text{FPR} = 1,15\% \pm 0,91\%$) | $\text{TP} = 198,00 / 198$ ($\text{Recall} = \mathbf{100,00\%} \pm 0,00\%$)
    - *(Seed 42)*: $\text{FP} = 19$ ($2,73\%$) | $\text{TP} = 198 / 198$ ($\text{Recall} = 100,00\%$)
  - `IF (2D raw)` (5 seeds): $\text{FP} = 13,40 \pm 3,72$ ($\text{FPR} = 1,93\% \pm 0,54\%$) | $\text{TP} = 198,00 / 198$ ($\text{Recall} = \mathbf{100,00\%} \pm 0,00\%$)
    - *(Seed 42)*: $\text{FP} = 11$ ($1,58\%$) | $\text{TP} = 198 / 198$ ($\text{Recall} = 100,00\%$)
  - `IF (3D raw)` (5 seeds): $\text{FP} = 8,00 \pm 2,10$ ($\text{FPR} = 1,15\% \pm 0,30\%$) | $\text{TP} = 196,00 / 198$ ($\text{Recall} = 98,99\% \pm 1,28\%$)
    - *(Seed 42)*: $\text{FP} = 10$ ($1,44\%$) | $\text{TP} = 198 / 198$ ($\text{Recall} = 100,00\%$)
- **Mode DEBOUNCED ($N=3$)**:
  - `DenseAE 2D` (5 seeds): $\text{FP} = 4,00 \pm 3,63$ ($\text{FPR} = \mathbf{0,58\%} \pm 0,52\%$) | $\text{TP} = 166,00 / 198$ ($\text{Recall} = 83,84\% \pm 0,00\%$)
    - *(Seed 42)*: $\text{FP} = 11$ ($1,58\%$) | $\text{TP} = 166 / 198$ ($\text{Recall} = 83,84\%$)
  - `IF (2D raw)` (5 seeds): $\text{FP} = 5,80 \pm 2,14$ ($\text{FPR} = 0,83\% \pm 0,31\%$) | $\text{TP} = 166,00 / 198$ ($\text{Recall} = 83,84\% \pm 0,00\%$)
    - *(Seed 42)*: $\text{FP} = 4$ ($0,58\%$) | $\text{TP} = 166 / 198$ ($\text{Recall} = 83,84\%$)
  - `IF (3D raw)` (5 seeds): $\text{FP} = 2,20 \pm 0,75$ ($\text{FPR} = 0,32\% \pm 0,11\%$) | $\text{TP} = 162,60 / 198$ ($\text{Recall} = 82,12\% \pm 2,16\%$)
    - *(Seed 42)*: $\text{FP} = 3$ ($0,43\%$) | $\text{TP} = 166 / 198$ ($\text{Recall} = 83,84\%$)

*(Sebagai pembanding, model IF 13-window menghasilkan 131 FP / 19,24% FPR pada S14 akibat histeresis rolling buffer 14 detik).*

---

#### 2. Skenario S15 (FINGER_SHIFT: 711 Normal, 83 Anomali)
- **Mode RAW ($N=1$)**:
  - `DenseAE 2D` (5 seeds): $\text{FP} = 8,20 \pm 5,81$ ($\text{FPR} = 1,15\% \pm 0,82\%$) | $\text{TP} = 81,00 / 83$ ($\text{Recall} = \mathbf{97,59\%} \pm 0,00\%$)
    - *(Seed 42)*: $\text{FP} = 8$ ($1,13\%$) | $\text{TP} = 81 / 83$ ($\text{Recall} = 97,59\%$)
  - `IF (2D raw)` (5 seeds): $\text{FP} = 21,60 \pm 6,09$ ($\text{FPR} = 3,04\% \pm 0,86\%$) | $\text{TP} = 79,80 / 83$ ($\text{Recall} = 96,14\% \pm 0,48\%$)
    - *(Seed 42)*: $\text{FP} = 16$ ($2,25\%$) | $\text{TP} = 80 / 83$ ($\text{Recall} = 96,39\%$)
  - `IF (3D raw)` (5 seeds): $\text{FP} = 24,60 \pm 10,93$ ($\text{FPR} = 3,46\% \pm 1,54\%$) | $\text{TP} = 69,80 / 83$ ($\text{Recall} = 84,10\% \pm 2,79\%$)
    - *(Seed 42)*: $\text{FP} = 19$ ($2,67\%$) | $\text{TP} = 69 / 83$ ($\text{Recall} = 83,13\%$)
- **Mode DEBOUNCED ($N=3$)**:
  - `DenseAE 2D` (5 seeds): $\text{FP} = 0,60 \pm 0,49$ ($\text{FPR} = \mathbf{0,08\%} \pm 0,07\%$) | $\text{TP} = 53,00 / 83$ ($\text{Recall} = 63,86\% \pm 0,00\%$)
    - *(Seed 42)*: $\text{FP} = 1$ ($0,14\%$) | $\text{TP} = 53 / 83$ ($\text{Recall} = 63,86\%$)
  - `IF (2D raw)` (5 seeds): $\text{FP} = 3,20 \pm 2,86$ ($\text{FPR} = 0,45\% \pm 0,40\%$) | $\text{TP} = 52,40 / 83$ ($\text{Recall} = 63,13\% \pm 1,45\%$)
    - *(Seed 42)*: $\text{FP} = 1$ ($0,14\%$) | $\text{TP} = 53 / 83$ ($\text{Recall} = 63,86\%$)
  - `IF (3D raw)` (5 seeds): $\text{FP} = 13,80 \pm 6,49$ ($\text{FPR} = 1,94\% \pm 0,91\%$) | $\text{TP} = 35,00 / 83$ ($\text{Recall} = 42,17\% \pm 5,49\%$)
    - *(Seed 42)*: $\text{FP} = 13$ ($1,83\%$) | $\text{TP} = 31 / 83$ ($\text{Recall} = 37,35\%$)

---

## 4. Analisis Univariat 2D vs 3D: Fitur, Ambang, dan Kontribusi Alarm

Model Univariat dikalibrasi pada parameter persentil tail $q$ terhadap data train normal untuk mencapai target FPR 3,0% pada set validasi normal.

| Varian | Fitur Terpakai | Parameter $q$ | Ambang per Fitur & Arah | Alarm per Fitur pada Test | Total Alarm Gabungan |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Univariat 2D** | `hr`, `spo2` | 0,6551% | - `hr`: Two-sided $[63,056, 96,700]$ bpm<br>- `spo2`: One-sided lower $[96,711, 100,0]\%$ | `hr`: 165<br>`spo2`: 336 | **346** |
| **Univariat 3D** | `hr`, `spo2`, `temp` | 0,6551% | - `hr`: Two-sided $[63,056, 96,700]$ bpm<br>- `spo2`: One-sided lower $[96,711, 100,0]\%$<br>- `temp`: Two-sided $[31,000, 34,000]^\circ\text{C}$ | `hr`: 165<br>`spo2`: 336<br>`temp`: **0** | **346** |

**Mengapa Hasil 2D vs 3D Identik di Semua Digit**:
1. **Fitur Suhu Benar-Benar Terpakai**: Logika evaluasi secara aktif menguji kondisi `(temp < 31.0) | (temp > 34.0)` pada setiap baris uji.
2. **Karakteristik Fisik Simulator**: Pada test set, nilai sensor suhu kulit DS18B20 berkisar antara $31,5^\circ\text{C}$ hingga $34,0^\circ\text{C}$, sehingga **seluruh 5.863 baris uji berada di dalam rentang normal**.
3. **Anomali Simulator Murni Hemodinamik**: Skenario gerakan fisik (`MOTION`) dan pergeseran sensor optik (`FINGER_SHIFT`) memicu fluktuasi denyut nadi dan penurunan saturasi oksigen pada MAX30102, sementara suhu tubuh dimodelkan konstan oleh generator simulator telemetri.
4. **Himpunan Alarm Gabungan**: Karena `temp` menyumbangkan tepat 0 alarm, gabungan logika $\text{HR} \lor \text{SpO2} \lor \text{Temp}$ ekuivalen persis $100\%$ dengan $\text{HR} \lor \text{SpO2}$, sehingga semua metrik (FPR, Recall, F1) identik di semua digit.

---

## 5. Tabel Perbandingan Model

### A. Tabel Recall pada FPR Tetap (Tabel Perbandingan Utama)
> **Catatan Metodologi Kritis**: Membandingkan metrik F1-score pada titik operasi yang memiliki Test FPR berbeda (misalnya Rule Default pada Test FPR 0,00%, IF pada 1,79%, dan DenseAE pada 2,24%) **tidak adil dan menyesatkan (*unfair comparison*)** karena presisi dan F1 sangat dipengaruhi oleh tingkat false positive yang tidak setara.  
> **Tabel perbandingan utama yang valid secara metodologis adalah Recall pada tingkat FPR Test yang disamakan secara ketat** (dihitung langsung dari kurva ROC tanpa interpolasi linear):

| Model | Sifat Model | Recall @ FPR $\le$ 0,1% | Recall @ FPR $\le$ 0,5% | Recall @ FPR $\le$ 1,0% | Recall @ FPR $\le$ 3,0% |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **dl_v1 Zero-Shot (2D)** | Deterministik | 63,35% | 64,41% | 64,77% | 67,62% |
| **IF (2D raw)** | 5 seeds (42–46) | 84,77% $\pm$ 1,26% | 91,25% $\pm$ 2,08% | 95,73% $\pm$ 2,86% | 99,00% $\pm$ 0,14% |
| **IF (3D raw)** | 5 seeds (42–46) | 69,04% $\pm$ 6,97% | 81,07% $\pm$ 2,88% | 85,41% $\pm$ 2,00% | 95,59% $\pm$ 2,41% |
| **Mahalanobis (2D)** | Deterministik | 97,15% | 98,22% | 98,58% | 98,93% |
| **Mahalanobis (3D)** | Deterministik | 97,15% | 97,51% | 98,22% | 98,93% |
| **LOF (2D)** | Deterministik | 97,51% | 98,58% | 98,58% | **99,29%** |
| **LOF (3D)** | Deterministik | 97,15% | 97,86% | 98,93% | **99,29%** |
| **OCSVM (2D)** | Deterministik | **98,22%** | **99,29%** | **99,29%** | **99,29%** |
| **OCSVM (3D)** | Deterministik | 97,86% | 98,58% | 98,93% | **99,29%** |
| **DenseAE 2D** | 5 seeds (42–46) | **98,22%** $\pm$ 0,23% | 98,86% $\pm$ 0,27% | 99,22% $\pm$ 0,14% | **99,29%** $\pm$ 0,00% |
| **DenseAE 3D** | 5 seeds (42–46) | 96,09% $\pm$ 0,71% | 97,15% $\pm$ 0,78% | 97,58% $\pm$ 0,42% | 98,43% $\pm$ 0,48% |

---

### B. Tabel Titik Operasi Tanpa Debounce (RAW / $N=1$)
*Ambang batas dikalibrasi pada normal validasi dengan target FPR 3,0%. Kolom Test FPR menyajikan FPR aktual empiris pada test set normal (5.582 baris).*

| Model | Sifat | ROC-AUC | PR-AUC | Test FPR Aktual | Recall | Presisi | F1 | Burst Recall (29) | FPR tol14 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Default (2D)** | Deterministik | N/A | N/A | **0,00%** | 75,80% | **100,00%** | 0,8623 | 96,6% (28) | **0,00%** |
| **Rule Default (3D)** | Deterministik | N/A | N/A | **0,00%** | 75,80% | **100,00%** | 0,8623 | 96,6% (28) | **0,00%** |
| **Univariate (2D, q=3%)** | Deterministik | N/A | N/A | 1,24% | 98,58% | 80,06% | **0,8836** | **100,0%** (29) | 1,23% |
| **Univariate (3D, q=3%)** | Deterministik | N/A | N/A | 1,24% | 98,58% | 80,06% | **0,8836** | **100,0%** (29) | 1,23% |
| **dl_v1 Zero-Shot (2D)** | Deterministik | 0,9060 | 0,7324 | 5,93% | 71,53% | 37,78% | 0,4945 | 96,6% (28) | 6,14% |
| **IF (2D raw, 5 seeds)** | 5 seeds | 0,9964 $\pm$ 0,0004 | 0,9746 $\pm$ 0,0050 | 1,79% $\pm$ 0,24% | 98,86% $\pm$ 0,14% | 73,63% $\pm$ 2,69% | 0,8437 $\pm$ 0,0179 | **100,0%** $\pm$ 0,0% | 1,75% $\pm$ 0,25% |
| **IF (3D raw, 5 seeds)** | 5 seeds | 0,9951 $\pm$ 0,0011 | 0,9334 $\pm$ 0,0157 | 2,71% $\pm$ 0,36% | 94,59% $\pm$ 1,61% | 63,88% $\pm$ 3,30% | 0,7622 $\pm$ 0,0274 | **100,0%** $\pm$ 0,0% | 2,70% $\pm$ 0,36% |
| **Mahalanobis (2D)** | Deterministik | 0,9978 | 0,9918 | 3,22% | **99,64%** | 60,87% | 0,7557 | **100,0%** (29) | 2,90% |
| **Mahalanobis (3D)** | Deterministik | 0,9989 | 0,9909 | 3,89% | 98,93% | 56,16% | 0,7165 | **100,0%** (29) | 3,62% |
| **LOF (2D)** | Deterministik | 0,9985 | 0,9921 | 1,65% | 98,58% | 75,07% | 0,8523 | **100,0%** (29) | 1,62% |
| **LOF (3D)** | Deterministik | **0,9991** | 0,9935 | 2,13% | 99,29% | 70,10% | 0,8218 | **100,0%** (29) | 2,06% |
| **OCSVM (2D)** | Deterministik | 0,9952 | 0,9929 | 1,42% | 99,29% | 77,93% | 0,8732 | **100,0%** (29) | 1,38% |
| **OCSVM (3D)** | Deterministik | 0,9974 | **0,9938** | 1,76% | 99,29% | 74,01% | 0,8480 | **100,0%** (29) | 1,81% |
| **DenseAE 2D (5 seeds)** | 5 seeds | 0,9967 $\pm$ 0,0014 | 0,9932 $\pm$ 0,0007 | 2,24% $\pm$ 0,55% | 99,29% $\pm$ 0,00% | 69,50% $\pm$ 5,29% | 0,8165 $\pm$ 0,0367 | **100,0%** $\pm$ 0,0% | 2,28% $\pm$ 0,63% |
| **DenseAE 3D (5 seeds)** | 5 seeds | 0,9984 $\pm$ 0,0003 | 0,9874 $\pm$ 0,0026 | 3,65% $\pm$ 0,37% | 98,43% $\pm$ 0,48% | 57,67% $\pm$ 2,64% | 0,7269 $\pm$ 0,0206 | **100,0%** $\pm$ 0,0% | 3,64% $\pm$ 0,47% |

---

### C. Tabel Performa dengan Persistence Debounce ($N=3$)

| Model + $N=3$ | Sifat | Test FPR Aktual | Recall | Presisi | F1 | Burst Recall (29) | FPR tol14 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Default (2D) + N3** | Deterministik | **0,00%** | 51,60% | **100,00%** | 0,6808 | 58,6% (17) | **0,00%** |
| **Rule Default (3D) + N3** | Deterministik | **0,00%** | 51,60% | **100,00%** | 0,6808 | 58,6% (17) | **0,00%** |
| **Univariate (2D) + N3** | Deterministik | 0,30% | 76,51% | 92,67% | **0,8382** | **100,0%** (29) | 0,27% |
| **Univariate (3D) + N3** | Deterministik | 0,30% | 76,51% | 92,67% | **0,8382** | **100,0%** (29) | 0,27% |
| **dl_v1 Zero-Shot + N3** | Deterministik | 3,87% | 47,33% | 38,11% | 0,4222 | 55,2% (16) | 4,04% |
| **IF (2D raw) + N3** | 5 seeds | 0,44% $\pm$ 0,12% | 77,72% $\pm$ 0,43% | 89,94% $\pm$ 2,51% | 0,8337 $\pm$ 0,0121 | **100,0%** $\pm$ 0,0% | 0,40% $\pm$ 0,13% |
| **IF (3D raw) + N3** | 5 seeds | 1,31% $\pm$ 0,19% | 70,32% $\pm$ 2,83% | 73,08% $\pm$ 3,41% | 0,7166 $\pm$ 0,0294 | 91,7% $\pm$ 1,7% | 1,31% $\pm$ 0,18% |
| **Mahalanobis (2D) + N3** | Deterministik | 1,67% | 78,29% | 70,29% | 0,7407 | **100,0%** (29) | 1,38% |
| **Mahalanobis (3D) + N3** | Deterministik | 2,10% | 77,22% | 64,97% | 0,7057 | **100,0%** (29) | 1,90% |
| **LOF (2D) + N3** | Deterministik | 0,48% | 76,87% | 88,89% | 0,8244 | **100,0%** (29) | 0,44% |
| **LOF (3D) + N3** | Deterministik | 0,84% | 77,58% | 82,26% | 0,7985 | **100,0%** (29) | 0,71% |
| **OCSVM (2D) + N3** | Deterministik | 0,39% | **77,94%** | 90,87% | 0,8391 | **100,0%** (29) | 0,35% |
| **OCSVM (3D) + N3** | Deterministik | 0,66% | **77,94%** | 85,55% | 0,8156 | **100,0%** (29) | 0,67% |
| **DenseAE 2D + N3** | 5 seeds | 0,79% $\pm$ 0,27% | **77,94%** $\pm$ 0,00% | 83,47% $\pm$ 4,58% | 0,8054 $\pm$ 0,0217 | **100,0%** $\pm$ 0,0% | 0,80% $\pm$ 0,31% |
| **DenseAE 3D + N3** | 5 seeds | 1,57% $\pm$ 0,16% | 76,87% $\pm$ 0,81% | 71,21% $\pm$ 1,97% | 0,7391 $\pm$ 0,0089 | **100,0%** $\pm$ 0,0% | 1,55% $\pm$ 0,21% |

---

## 6. Temuan Pengujian Test Suite Repositori

Dijalankan menggunakan `/tmp/dl_train_venv/bin/python -m pytest tests/` dengan menyertakan dependensi PyTorch CPU:
- **Total Pengujian**: 116 tests terkumpul
- **Lulus**: **115 passed**
- **Gagal**: **1 failed**
- **Dilewati**: **0 skipped** (4 pengujian Torch yang sebelumnya di-skip sekarang aktif dijalankan)
- **Error Persis Pengujian Gagal**:
  ```text
  FAILED tests/test_dl_public.py::test_export_parity_torch_onnx_numpy - ModuleNotFoundError: No module named 'onnxscript'
  ```
  *Penyebab*: Pada PyTorch 2.15, pemanggilan fungsi `torch.onnx.export` secara default memicu dependensi internal `onnxscript` yang belum terpasang di venv minimal. Pengujian runtime ONNX murni (`test_onnx_runtime_parity`) tetap lulus 100%.

---

## 7. Verifikasi Urutan Indeks & Integritas Repositori

- **Verifikasi Urutan Indeks Persistence**:
  Seluruh Series debounced terverifikasi secara ketat:
  ```python
  assert (deb_2d.index == test_df.index).all()
  assert (deb_if2.index == test_df.index).all()
  ```
  Hasil verifikasi: **PASSED (100% konsisten)**.
- **Integritas Lingkungan**:
  Tidak ada perubahan pada file firmware, direktori data, model artefak produksi (`ml/artifacts/dl_v1`), atau backend app config. Seluruh eksekusi dilakukan pada script terisolasi `ml_pipeline/src/evaluate_dl_simulator.py`.
