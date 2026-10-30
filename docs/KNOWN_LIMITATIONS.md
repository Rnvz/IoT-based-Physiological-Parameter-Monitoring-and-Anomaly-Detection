# Known Limitations and Dataset Characteristics

Dokumen ini mendokumentasikan karakteristik, keterbatasan, dan anomali empiris yang telah diverifikasi secara numerik pada pipeline machine learning dan dataset pelatihan (`ml_pipeline/`). Dokumen ini disusun untuk transparansi evaluasi akademik dan skripsi.

Semua temuan di bawah ini mencantumkan nomor bukti, berkas sumber, dan perintah eksekusi yang mereproduksinya. Seluruh klaim yang belum diverifikasi secara independen ditandai sebagai `UNVERIFIED`.

---

## 1. Penurunan Label Target (Ground Truth Label Derivation per Source)

**Berkas Sumber**: [`ml_pipeline/src/clean_datasets.py`](file:///home/ywenanta/Projects/IoT/ml_pipeline/src/clean_datasets.py)

Label target `OUTPUT` (0 = Normal, 1 = Abnormal) pada dataset publik yang digunakan untuk melatih Isolation Forest tidak merepresentasikan anomali multivariat fisiologis yang terukur secara seragam:

1. **`gourangomandal` (Synthetic IoT Healthcare Dataset)**:
   - *Kode*: [`clean_datasets.py` baris 153-159](file:///home/ywenanta/Projects/IoT/ml_pipeline/src/clean_datasets.py#L153-L159)
   - *Mekanisme*: Kolom `OUTPUT` diturunkan secara deterministik dari aturan threshold biner kaku pada parameter individual:
     ```python
     is_normal = (hr_alert == 'NORMAL') & (spo2_alert == 'NORMAL') & (temp_alert == 'NORMAL')
     output_binary = np.where(is_normal, 0, 1)
     ```
   - *Keterbatasan*: Bukan dataset fisiologis nyata, melainkan data sintetis berbasis aturan *single-parameter threshold*. Melatih model machine learning pada dataset ini hanya mempelajari kembali aturan threshold sederhana.

2. **`nasirayub2` (Human Vital Signs Dataset 2024)**:
   - *Kode*: [`clean_datasets.py` baris 81-83](file:///home/ywenanta/Projects/IoT/ml_pipeline/src/clean_datasets.py#L81-L83)
   - *Mekanisme*: `OUTPUT` dipetakan dari kategori risiko klinis (`Risk Category == 'Low Risk' -> 0, else 1`).
   - *Keterbatasan*: Kategori risiko pada dataset aslinya ditentukan berdasarkan variabel yang **tidak diukur** oleh perangkat keras IoT (seperti tekanan darah sistolik dan diastolik). Akibatnya, distribusi tanda vital terukur (HR, SpO2, Temp) antara kelas normal dan abnormal saling tumpang tindih (*overlap* parah):
     - Mean HR: 75.05 BPM (normal) vs 83.58 BPM (abnormal)
     - Mean SpO2: 97.43% (normal) vs 96.84% (abnormal)
     - Mean Temp: 36.75 °C (normal) vs 36.75 °C (abnormal)

3. **`engrarri21` (Human Vital Signs R)**:
   - *Kode*: [`clean_datasets.py` baris 51-53](file:///home/ywenanta/Projects/IoT/ml_pipeline/src/clean_datasets.py#L51-L53)
   - *Mekanisme*: Dipetakan langsung dari label teks kategori `OUTPUT` ('normal' -> 0, 'abnormal' -> 1).
   - *Keterbatasan*: Mean suhu tubuh pada kelas normal adalah **30.05 °C** (rentang 25.0 - 37.0 °C), sedangkan pada kelas abnormal adalah **38.61 °C**. Suhu normal yang sangat rendah ini tidak konsisten dengan data latih lainnya (lihat Bagian 3 dan 4).

---

## 2. Fitur Temporal Konstan (Nol) pada Mayoritas Data Latih

**Berkas Sumber**: [`ml_pipeline/src/feature_engineering.py` baris 30-49](file:///home/ywenanta/Projects/IoT/ml_pipeline/src/feature_engineering.py#L30-L49)

### Verifikasi Statistik pada `features_train.csv` (Total 168.911 Baris)
Perintah reproduksi:
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('ml_pipeline/data/features_train.csv')
for src, grp in df.groupby('Source'):
    n = len(grp)
    print(src, n, (grp['Var_Heart_Rate'] > 0).mean(), (grp['Heart_Rate_Delta'] != 0).mean(), (grp['Rate_of_Change'] != 0).mean())
"
```

Hasil numerik riil:

| Sumber Data | Jumlah Baris | Fraksi `Var_Heart_Rate > 0` | Fraksi `Heart_Rate_Delta != 0` | Fraksi `Rate_of_Change != 0` |
| :--- | :---: | :---: | :---: | :---: |
| `nasirayub2` | 120.012 (71,05%) | 0,000000 (0 / 120.012) | 0,000000 (0 / 120.012) | 0,000000 (0 / 120.012) |
| `gourangomandal` | 36.000 (21,31%) | 0,000000 (0 / 36.000) | 0,000000 (0 / 36.000) | 0,000000 (0 / 36.000) |
| `engrarri21` | 12.899 (7,64%) | 0,937980 (12.099 / 12.899) | 0,249322 (3.216 / 12.899) | 0,563687 (7.271 / 12.899) |
| **Total Seluruh Latih** | **168.911 (100%)** | **0,071629 (7,16%)** | **0,019040 (1,90%)** | **0,043046 (4,30%)** |

### Akar Penyebab dalam Kode
Pada `feature_engineering.py`:
```python
def compute_temporal_features_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Heart_Rate', 'SpO2', 'Temperature']:
        df[f'{col}_Delta'] = 0.0
        df[f'MA_{col}'] = df[col]
        df[f'Var_{col}'] = 0.0
    df['Rate_of_Change'] = 0.0
    return df.reset_index(drop=True)
```
Dataset cross-sectional (`nasirayub2` dan `gourangomandal`) yang mencakup **92,36% dari data latih** dipaksa memiliki seluruh fitur temporal = 0,0.

### Hasil Ablasi (Pembuktian Penyebab Severity $\ge 1$)
Pengujian ablasi pada 400 sampel (386 jendela post-warmup, noise $\sigma_{HR}=1.5, \sigma_{SpO2}=0.3, \sigma_{Temp}=0.15$):

| Kondisi Eksperimen | Mean Score | Raw Sev 0 | Raw Sev 1 | Raw Sev 2 |
| :--- | :---: | :---: | :---: | :---: |
| (i) Fitur lengkap seperti yang dihitung adapter | +0.236631 | 93 (24,1%) | 293 (75,9%) | 0 (0,0%) |
| (ii) Fitur temporal (`Var_*`, `*_Delta`, `Rate_of_Change`) dipaksa = 0 | +0.346107 | 386 (100,0%) | 0 (0,0%) | 0 (0,0%) |
| (iii) Nilai mentah & MA diganti konstanta mean (72, 98, 36.8), temporal tetap | +0.241836 | 112 (29,0%) | 274 (71,0%) | 0 (0,0%) |

**Kesimpulan Diverifikasi**: Tingkat severity $\ge 1$ yang persisten sepenuhnya dipicu oleh fitur window temporal (`Var_*`, `*_Delta`, `Rate_of_Change`) yang melebihi batas $P_{95}$ data latih akibat mayoritas data latih bernilai nol.

---

## 3. Level Shift Suhu Kulit (Skin Surface Temperature vs Core Body Temperature)

Model Isolation Forest dilatih menggunakan parameter suhu inti tubuh (pusat skala RobustScaler = **36,807 °C**, skala IQR = **0,8067 °C**). Namun, perangkat keras fisik IoT menggunakan sensor DS18B20 yang diletakkan pada permukaan kulit pergelangan tangan / jari, dengan suhu operasional normal **~32 – 35 °C**.

### Hasil Pengujian Stream Sintetis Riil (400 Sampel, 1 Hz, HR 72, SpO2 98)

| Suhu Uji (°C) | N Valid | Mean Score | Raw Sev 0 | Raw Sev 1 | Raw Sev 2 | Debounced Sev 0 | Debounced Sev 1 | Debounced Sev 2 | Persentase Alarm (Sev $\ge 1$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 31,0 | 386 | +0,071192 | 0 (0%) | 29 (7,5%) | 357 (92,5%) | 0 (0%) | 31 (8,0%) | 355 (92,0%) | **100,0%** |
| 33,0 | 386 | +0,099312 | 0 (0%) | 206 (53,4%) | 180 (46,6%) | 0 (0%) | 207 (53,6%) | 179 (46,4%) | **100,0%** |
| 35,0 | 386 | +0,124758 | 0 (0%) | 360 (93,3%) | 26 (6,7%) | 0 (0%) | 361 (93,5%) | 25 (6,5%) | **100,0%** |
| 36,5 | 386 | +0,179435 | 1 (0,3%) | 385 (99,7%) | 0 (0,0%) | 1 (0,3%) | 385 (99,7%) | 0 (0,0%) | **99,7%** |
| 36,8 (Mean Latih) | 386 | +0,309105 | 386 (100%) | 0 (0%) | 0 (0%) | 386 (100%) | 0 (0%) | 0 (0%) | **0,0%** |

*Catatan*: Pada suhu kulit realistis manusia sehat (~33 °C), model ML menghasilkan 100% false alarm (Severity 1 atau Severity 2).

---

## 4. Hasil Evaluasi Baris Normal pada `engrarri21`

Perintah reproduksi:
```bash
.venv/bin/python -c "
import pandas as pd, joblib
model = joblib.load('ml_pipeline/models/isolation_forest_model.joblib')
thrs = joblib.load('ml_pipeline/models/thresholds.joblib')
df = pd.read_csv('ml_pipeline/data/features_test.csv')
norm_eng = df[(df['Source'] == 'engrarri21') & (df['OUTPUT'] == 0)]
scores = model.decision_function(norm_eng[model.feature_names_in_])
print('Severity distribution on engrarri21 OUTPUT==0:')
print((scores < thrs[0.05]).mean(), ((scores >= thrs[0.05]) & (scores < thrs[0.13])).mean(), (scores >= thrs[0.13]).mean())
"
```

Hasil terverifikasi:
- Jumlah baris normal `engrarri21` pada set uji: **188 baris**.
- **Severity 0 (Normal)**: 0 baris (0,0%)
- **Severity 1 (Warning)**: 61 baris (32,45%)
- **Severity 2 (Critical)**: 127 baris (67,55%)
- **Tingkat False Alarm**: **100,0%** dari seluruh sampel normal pada partisi ini salah dideteksi sebagai anomali.
- **Debounced Severity-2 Events pada Stream Kontinyu**: Saat seluruh slice uji `engrarri21` dialirkan kronologis dengan rolling window dan filter debounce ($N=3$), terdeteksi **1.027 event Severity-2** dari 4.305 baris (setara **238,56 event Severity-2 per 1.000 baris**).

---

## 5. Kalibrasi Ambang Batas pada Set Validasi (Percentile Artifact)

**Berkas Sumber**: [`ml_pipeline/src/train_isolation_forest.py` baris 85-98](file:///home/ywenanta/Projects/IoT/ml_pipeline/src/train_isolation_forest.py#L85-L98)

Ambang batas anomali (`thresholds.joblib`) dihitung dengan rumus:
```python
threshold = np.percentile(scores_val_normal, target_fpr * 100)
```
- Ambang batas severity 2 dikalibrasi pada `target_fpr = 0.05` ($P_5$).
- Ambang batas severity 1 dikalibrasi pada `target_fpr = 0.13` ($P_{13}$).

**Konsekuensi Logis**:
Klaim bahwa *"model menghasilkan 87% Normal, 8% Warning, dan 5% Critical pada set validasi"* adalah **benar secara konstruksi matematis (true by construction)** akibat definisi persentil, bukan hasil generalisasi performa independen model terhadap anomali eksternal.

---

## 6. Partisi Data dan *Subject Leakage* (Patient Overlap)

**Berkas Sumber**: [`ml_pipeline/src/feature_engineering.py` baris 57-82](file:///home/ywenanta/Projects/IoT/ml_pipeline/src/feature_engineering.py#L57-L82)

1. **`engrarri21`**:
   - Seluruh 21.523 baris berasal dari satu subjek/aliran sensor tunggal (`Patient_ID = 'engrarri21_stream'`).
   - Pembagian data dilakukan secara temporal (60% awal = latih, 20% tengah = validasi, 20% akhir = uji).
   - Terjadi kebocoran identitas subjek (*subject leakage*) karena data latih dan data uji berasal dari orang yang sama.
2. **`nasirayub2` dan `gourangomandal`**:
   - Pembagian data menggunakan pemisahan baris acak (`train_test_split(..., random_state=42)`), bukan pemisahan berbasis kelompok subjek (*GroupShuffleSplit* / *GroupKFold* berdasarkan identitas sesi).

---

## 7. Risiko Penskalaan Ganda (*Double-Scaling Hazard*)

**Berkas Terkait**:
- [`ml_pipeline/src/feature_engineering.py` baris 100-112](file:///home/ywenanta/Projects/IoT/ml_pipeline/src/feature_engineering.py#L100-L112)
- [`backend/app/ml/inference.py` baris 116-121](file:///home/ywenanta/Projects/IoT/backend/app/ml/inference.py#L116-L121)

1. **Kondisi Berkas CSV**:
   Seluruh berkas fitur (`features_train.csv`, `features_val.csv`, `features_test.csv`) disimpan dalam kondisi **sudah dinormalisasi** oleh `RobustScaler`.
2. **Kondisi Layanan Inferensi**:
   Kelas `MLService` pada backend secara otomatis memanggil `self.scaler.transform(features_input)` terhadap data input baru sebelum dievaluasi oleh `IsolationForest`.
3. **Bahaya Evaluasi**:
   Jika skrip evaluasi offline memasukkan baris dari `features_test.csv` langsung ke dalam `MLService` (atau memanggil `scaler.transform()` untuk kedua kalinya), fitur akan mengalami penskalaan ganda ($x'' = (x' - \text{center}) / \text{scale}$), merusak seluruh koordinat ruang fitur model. Evaluasi model terhadap file pre-computed harus memanggil `model.decision_function()` secara langsung tanpa memanggil `scaler.transform()`.
