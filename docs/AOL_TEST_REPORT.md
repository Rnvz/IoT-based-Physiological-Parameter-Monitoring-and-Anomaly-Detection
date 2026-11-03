# Laporan Verifikasi Komprehensif & Audit Hasil Uji AOL (dl-public)

> [!IMPORTANT]
> **STATUS PELAPORAN: DIVERIFIKASI SECARA KETAT**
> Dokumen ini mencatat audit independen terhadap integritas data, urutan evaluasi uji, metrik lengkap terkalibrasi ROC (tanpa interpolasi), analisis bootstrap CI, dan uji sensitivitas domain. Tidak ada model terekspor (Seed 42), split, pra-registrasi, atau ambang batas yang diubah.

---

## 1. Kronologi Faktual Evaluasi Test Set

Pernyataan awal "test dinilai sekali" diperjelas ke kronologi faktual yang sebenarnya:

1. **Pembekuan Split (`splits/public_split.json`)**: `2026-10-31 13:43:29 +0700` (SHA256: `15a424ff1118cac8dfe3cbd20d80f028863b46edc235db24a2dd50adef6d1635`).
2. **Komitmen Pra-Registrasi (`docs/DL_PREREGISTRATION.md`)**: Commit `b216cfb` pada `2026-10-31 13:44:13 +0700`.
3. **Skor Pertama & Ekspor Model**: Dijalankan pada `2026-10-31 14:04:20 +0700` (terekam pada log `task-790` dan timestamp `metadata.json`), menghasilkan artefak pemenang `DenseAE (v2)` Seed 42.
4. **Komitmen Hasil Awal**: Commit `966b571` pada `2026-10-31 14:06:45 +0700`.
5. **Skor Ulang Verifikasi Independen**: Dijalankan pada `2026-11-03` via `scripts/aol_verify.py` menggunakan model biner yang sama persis (`model_weights.npz`) dan split beku yang sama persis tanpa modifikasi model atau ambang batas, untuk memverifikasi metrik, mengaudit duplikasi, dan menguji sensitivitas domain.

---

## 2. Tabel Hitungan Lengkap (Recalculated dari Data Nyata)

### a. Hitungan per Sumber x Split x Kelas

Dihitung langsung dari `ml_pipeline/data/harmonized_vitals.csv` + `splits/public_split.json` (Total Baris: 281.543):

| Sumber Data | Split | Normal ($y=0$) | Abnormal ($y=1$) | Total Sampel | Prevalensi ($y=1$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **nasirayub2** | Train | 56.936 | 63.076 | 120.012 | 52,56% |
| **nasirayub2** | Val | 19.089 | 20.915 | 40.004 | 52,28% |
| **nasirayub2** | Test | 18.880 | 21.124 | 40.004 | 52,80% |
| **engrarri21** | Train | 3.199 | 9.715 | 12.914 | 75,23% |
| **engrarri21** | Val | 2.306 | 1.999 | 4.305 | 46,43% |
| **engrarri21** | Test | 149 | 4.035 | 4.184 | 96,44% |
| **gourangomandal** | Train | 11.180 | 24.820 | 36.000 | 68,94% |
| **gourangomandal** | Val | 3.733 | 8.267 | 12.000 | 68,89% |
| **gourangomandal** | Test | 3.612 | 8.388 | 12.000 | 69,90% |
| **POOLED** | **Train** | **71.315** | **97.611** | **168.926** | **57,78%** |
| **POOLED** | **Val** | **25.128** | **31.181** | **56.309** | **55,37%** |
| **POOLED** | **Test** | **22.641** | **33.547** | **56.188** | **59,70%** |

*(Catatan: Pada `engrarri21`, terdapat 120 baris jeda isolasi / purge gap (dua blok jeda 60-baris: 58 baris normal dan 62 baris abnormal) yang sengaja di-drop antara train/val dan val/test untuk mencegah kebocoran autokorelasi deret waktu).*

### b. Audit Keunikan Data Uji engrarri21

| Metrik Audit | Kolom Vitals Sahaja (`HR, Temp`) | Seluruh Kolom (`Patient_ID, HR, Temp, OUTPUT`) |
| :--- | :--- | :--- |
| **Total Baris Test** | 4.184 | 4.184 |
| **Baris Unik Test** | 297 (Duplikat: **92,90%**) | 588 (Duplikat: **85,95% ~ 86%**) |
| **Total Baris Normal ($y=0$)** | 149 | 149 |
| **Baris Normal Unik** | 42 (Duplikat: **71,81%**) | 57 (Duplikat: **61,74%**) |

### c. Rekonsiliasi Angka Lama yang Ditemukan pada Audit

1. **`5.712`**: Jumlah total baris normal pada `engrarri21` di file mentah `harmonized_vitals.csv` ($3.199 \text{ train} + 2.306 \text{ val} + 149 \text{ test} + 58 \text{ gap} = 5.712$).
2. **`890` vs `149`**: Respon teks asisten lama di Part 1 salah mengutip jumlah normal test `engrarri21` sebagai 890; angka hitungan nyata dari split beku adalah **149 baris normal**.
3. **`71.315`**: Jumlah eksak baris normal pada partisi POOLED Train ($56.936 + 3.199 + 11.180 = 71.315$).
4. **`23.605` vs `22.641`**: Respon teks lama salah mengutip normal test POOLED sebagai 23.605 akibat perbedaan kutipan pada `engrarri21` ($22.641$ adalah nilai riil).
5. **`36,31`**: Suhu rata-rata yang dikutip dari fixture unit test `tests/fixtures/sample_fixture.csv`, bukan dari dataset publik.
6. **`95,46%`, `79,83%`, `57,61%`**: Persentase kesepakatan eksak antara aturan deterministik default `evaluate_rules` dengan label `OUTPUT` dari dataset nyata:
   - `gourangomandal`: **95,46%** (sirkularitas ekstrem: label diturunkan dari aturan ambang batas vitals).
   - `engrarri21`: **79,83%** (satu subjek ICU dengan anomali demam tinggi).
   - `nasirayub2`: **57,61%** (label ditentukan oleh kriteria hipertensi sistolik/diastolik & BMI, bukan vitals).

---

## 3. Penjelasan Bug Rec@5%FPR pada Skor Bertingkat (Ties)

### Deskripsi Masalah pada Implementasi Lama
Pada implementasi sebelumnya di `ml/dl/evaluate.py`:
```python
p95_norm_th = float(np.percentile(normal_scores, 95.0))
rec_at_5pct = float(np.mean(abnormal_scores >= p95_norm_th))
```
Pendekatan percentile ini bekerja wajar pada skor kontinu yang halus (seperti Autoencoder atau Isolation Forest). Namun, pada model dengan skor bertingkat/diskret (*tied scores*) seperti Rule Engine yang hanya bernilai $\{0.0, 0.5, 1.0\}$:
- Jika $\ge 95\%$ baris normal bernilai $0.0$, maka persentil ke-95 (`p95_norm_th`) bernilai **$0.0$**.
- Akibatnya, `abnormal_scores >= 0.0` mengevaluasi benar untuk seluruh sampel abnormal, menghasilkan `Rec@5%FPR = 1.0000` (100%).
- **Dampak Bug**: Pada threshold $0.0$, seluruh baris normal bernilai $0.0$ juga diprediksi anomali, sehingga **FPR aktualnya adalah 100%, BUKAN 5%**. Nilai Recall 1,0000 sebelumnya adalah artefak semu.

### Perbaikan Non-Interpolated ROC Operating Point
Fungsi `compute_metrics` diperbaiki dengan mengambil titik kurva ROC empiris (`roc_curve`) di mana $FPR \le 0,05$ tanpa interpolasi:
```python
fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
valid_fpr_idx = np.where(fpr_curve <= 0.05 + 1e-9)[0]
rec_at_5pct = float(np.max(tpr_curve[valid_fpr_idx])) if len(valid_fpr_idx) > 0 else 0.0
```
Untuk skor bertingkat, algoritma kini memilih ambang batas valid tertinggi yang menjamin $FPR \le 5\%$ secara empiris.

---

## 4. Tabel Metrik Uji Lengkap Terkalibrasi (Termasuk engrarri21 Unique)

Dihitung ulang dengan `scripts/aol_verify.py` pada seluruh 56.188 sampel test set publik:

| Model | Sumber | Prevalensi | PR-AUC | PR - Prev | ROC-AUC | Rec@5%FPR | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule Engine (v3)** | POOLED | 0.5970 | 0.8871 | +0.2900 | 0.7327 | **0.3228** | 0.6405 |
| Rule Engine (v3) | nasirayub2 | 0.5280 | 0.8098 | +0.2818 | 0.5971 | **0.0036** | 0.3252 |
| Rule Engine (v3) | engrarri21 | 0.9644 | 0.9977 | +0.0333 | 0.9391 | 0.8912 | 0.9548 |
| Rule Engine (v3) | *engrarri21_unique* | 0.8586 | 0.9855 | +0.1269 | 0.9083 | 0.8314 | 0.9256 |
| Rule Engine (v3) | gourangomandal | 0.6990 | 0.9912 | +0.2922 | 0.9753 | 0.9707 | 0.9678 |
| **Rule Engine (v2)** | POOLED | 0.5970 | 0.8719 | +0.2749 | 0.6945 | **0.2033** | 0.5654 |
| Rule Engine (v2) | nasirayub2 | 0.5280 | 0.8098 | +0.2818 | 0.5971 | **0.0036** | 0.3252 |
| Rule Engine (v2) | engrarri21 | 0.9644 | 0.9893 | +0.0249 | 0.7300 | 0.4238 | 0.6763 |
| Rule Engine (v2) | *engrarri21_unique* | 0.8586 | 0.9443 | +0.0857 | 0.6865 | 0.3373 | 0.6352 |
| Rule Engine (v2) | gourangomandal | 0.6990 | 0.9724 | +0.2734 | 0.9151 | 0.8436 | 0.9075 |
| **Isolation Forest (v3)** | POOLED | 0.5970 | 0.8750 | +0.2780 | 0.7969 | 0.5045 | 0.3164 |
| Isolation Forest (v3) | nasirayub2 | 0.5280 | 0.7611 | +0.2331 | 0.6914 | 0.3362 | 0.0036 |
| Isolation Forest (v3) | engrarri21 | 0.9644 | 0.9656 | +0.0012 | 0.4950 | 0.1343 | 0.4918 |
| Isolation Forest (v3) | *engrarri21_unique* | 0.8586 | 0.8253 | -0.0333 | 0.3833 | 0.0824 | 0.5535 |
| Isolation Forest (v3) | gourangomandal | 0.6990 | 0.9888 | +0.2898 | 0.9760 | 0.8636 | 0.7407 |
| **Isolation Forest (v2)** | POOLED | 0.5970 | 0.8715 | +0.2744 | 0.7758 | 0.5288 | 0.6418 |
| Isolation Forest (v2) | nasirayub2 | 0.5280 | 0.7787 | +0.2506 | 0.7033 | 0.3821 | 0.4339 |
| Isolation Forest (v2) | engrarri21 | 0.9644 | 0.9848 | +0.0204 | 0.6855 | 0.4612 | 0.7854 |
| Isolation Forest (v2) | *engrarri21_unique* | 0.8586 | 0.9442 | +0.0856 | 0.7272 | 0.3922 | 0.7731 |
| Isolation Forest (v2) | gourangomandal | 0.6990 | 0.9833 | +0.2843 | 0.9550 | 0.8716 | 0.9327 |
| **Local Outlier Factor (v3)** | POOLED | 0.5970 | 0.8842 | +0.2871 | 0.7948 | 0.5810 | 0.4936 |
| Local Outlier Factor (v3) | nasirayub2 | 0.5280 | 0.7595 | +0.2314 | 0.6732 | 0.3598 | 0.0100 |
| Local Outlier Factor (v3) | engrarri21 | 0.9644 | 0.9935 | +0.0291 | 0.8974 | 0.6387 | 0.9336 |
| Local Outlier Factor (v3) | *engrarri21_unique* | 0.8586 | 0.9517 | +0.0931 | 0.8341 | 0.5020 | 0.8745 |
| Local Outlier Factor (v3) | gourangomandal | 0.6990 | 0.9987 | +0.2997 | 0.9969 | 0.9884 | 0.9325 |
| **Local Outlier Factor (v2)** | POOLED | 0.5970 | 0.7786 | +0.1816 | 0.7010 | 0.1301 | 0.2270 |
| Local Outlier Factor (v2) | nasirayub2 | 0.5280 | 0.6557 | +0.1276 | 0.6332 | 0.0852 | 0.1648 |
| Local Outlier Factor (v2) | engrarri21 | 0.9644 | 0.9890 | +0.0247 | 0.7680 | 0.4701 | 0.2439 |
| Local Outlier Factor (v2) | *engrarri21_unique* | 0.8586 | 0.9364 | +0.0778 | 0.7172 | 0.2667 | 0.1773 |
| Local Outlier Factor (v2) | gourangomandal | 0.6990 | 0.9679 | +0.2689 | 0.9096 | 0.8271 | 0.3666 |
| **One-Class SVM (v3)** | POOLED | 0.5970 | 0.8502 | +0.2532 | 0.7450 | 0.4720 | 0.4571 |
| One-Class SVM (v3) | nasirayub2 | 0.5280 | 0.6654 | +0.1374 | 0.5932 | 0.1951 | 0.0000 |
| One-Class SVM (v3) | engrarri21 | 0.9644 | 0.9951 | +0.0308 | 0.8885 | 0.6226 | 0.9325 |
| One-Class SVM (v3) | *engrarri21_unique* | 0.8586 | 0.9600 | +0.1014 | 0.7893 | 0.4431 | 0.8499 |
| One-Class SVM (v3) | gourangomandal | 0.6990 | 0.9979 | +0.2989 | 0.9951 | 0.9707 | 0.8651 |
| **One-Class SVM (v2)** | POOLED | 0.5970 | 0.8373 | +0.2402 | 0.7289 | 0.4446 | 0.5778 |
| One-Class SVM (v2) | nasirayub2 | 0.5280 | 0.6896 | +0.1615 | 0.6158 | 0.2346 | 0.3066 |
| One-Class SVM (v2) | engrarri21 | 0.9644 | 0.9910 | +0.0266 | 0.8029 | 0.4800 | 0.7634 |
| One-Class SVM (v2) | *engrarri21_unique* | 0.8586 | 0.9492 | +0.0906 | 0.7599 | 0.4235 | 0.7399 |
| One-Class SVM (v2) | gourangomandal | 0.6990 | 0.9836 | +0.2846 | 0.9563 | 0.8785 | 0.9273 |
| **DenseAE (v2) [Seed 42]** | POOLED | 0.5970 | 0.7708 | +0.1737 | 0.6385 | 0.2865 | 0.4254 |
| DenseAE (v2) [Seed 42] | nasirayub2 | 0.5280 | 0.5705 | +0.0424 | 0.5258 | 0.0763 | 0.1089 |
| DenseAE (v2) [Seed 42] | engrarri21 | 0.9644 | 0.9843 | +0.0199 | 0.6723 | 0.3941 | 0.6145 |
| DenseAE (v2) [Seed 42] | *engrarri21_unique* | 0.8586 | 0.9221 | +0.0635 | 0.6356 | 0.2275 | 0.5500 |
| DenseAE (v2) [Seed 42] | gourangomandal | 0.6990 | 0.9614 | +0.2624 | 0.8994 | 0.7277 | 0.8399 |

---

## 5. Audit Bootstrap Resampling pada `engrarri21`

Hasil penelusuran kode pada `patient_group_bootstrap` ([`ml/dl/evaluate.py#L149-L165`](file:///home/ywenanta/Projects/IoT/ml/dl/evaluate.py#L149-L165)):
- Pada subset `engrarri21`, seluruh baris berasal dari satu pasien tunggal (`n_pts == 1`).
- Eksekusi jatuh ke blok bootstrap deret waktu (blok 50 baris). Blok-blok ini di-resample dengan pengembalian dari dalam rekaman kontinu pasien yang sama.
- **Audit Verdict**: Interval kepercayaan (CI) untuk `engrarri21` diberi label:
  > **[!] CI TIDAK VALID: 1 SUBJEK, 86% DUPLIKAT**
  > Resampling blok dalam satu individu yang memiliki 86% duplikasi baris tidak dapat mengestimasi variabilitas antar-pasien klinis nyata. Metrik pada baris unik (`engrarri21_unique`) di atas menunjukkan penurunan performa wajar (misal: PR-AUC DenseAE v2 turun dari 0,9843 menjadi 0,9221 dan Rec@5%FPR turun dari 39,41% menjadi 22,75%).

---

## 6. Uji Sensitivitas Domain (Label: "BUKAN hasil test")

Pengujian ini memetakan respons model `DenseAE (v2)` Seed 42 dan `Rule Default` terhadap berbagai profil sinyal istirahat fisiologis pada persistensi $N=3$:

### 6a. Grid Alarm Palsu (20 Seed Aliran x 400 Sampel = 8.000 Sampel per Sel)

| Rata-rata HR (BPM) | Rata-rata SpO2 (%) | DenseAE Sev $\ge 1$ (%) | DenseAE Sev 2 (%) | Rule Sev $\ge 1$ (%) | Rule Sev 2 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **55** | 95 | 81,76% | 70,95% | 13,68% | 0,00% |
| **55** | 96 | 34,51% | 20,21% | 13,03% | 0,00% |
| **55** | 98 | 3,20% | 1,20% | 13,03% | 0,00% |
| **60** | 95 | 18,40% | 9,69% | 0,03% | 0,00% |
| **60** | 96 | 0,51% | 0,14% | 0,00% | 0,00% |
| **60** | 98 | 0,03% | 0,00% | 0,00% | 0,00% |
| **72** (Baseline) | 95 | 0,00% | 0,00% | 0,00% | 0,00% |
| **72** (Baseline) | 96 | 0,00% | 0,00% | 0,00% | 0,00% |
| **72** (Baseline) | 98 | **0,00%** | **0,00%** | **0,00%** | **0,00%** |
| **85** | 95 | 0,01% | 0,00% | 0,00% | 0,00% |
| **85** | 96 | 0,00% | 0,00% | 0,00% | 0,00% |
| **85** | 98 | 0,00% | 0,00% | 0,00% | 0,00% |
| **95** | 95 | 41,86% | 3,09% | 12,91% | 0,05% |
| **95** | 96 | 0,95% | 0,00% | 12,15% | 0,05% |
| **95** | 98 | 0,00% | 0,00% | 12,15% | 0,05% |

### 6b. Sapuan Satu Dimensi (1D Threshold Crossing Sweeps)

Ambang batas internal model:
- `DenseAE (v2)`: Warning = `0,013571`, Critical = `0,024805`
- `Rule Engine`: HR Warn = $[<55, >95]$, HR Crit = $[<50, >100]$, SpO2 Warn = $[<94]$, SpO2 Crit = $[<90]$

| Titik Ambang Batas | DenseAE (v2) Nilai Potong | Rule Default Nilai Potong | Karakteristik Perilaku |
| :--- | :--- | :--- | :--- |
| **HR Low Warning** | **53,3 BPM** | 55,0 BPM | DenseAE sedikit lebih toleran sebelum memicu warning bradikardia |
| **HR High Warning** | **101,9 BPM** | 95,0 BPM | DenseAE menoleransi HR hingga ~102 BPM sebelum warning takikardia |
| **HR Low Critical** | **52,4 BPM** | 50,0 BPM | Transisi cepat dari warn ke crit pada bradikardia |
| **HR High Critical** | **104,7 BPM** | 100,0 BPM | Transisi bertahap pada takikardia berat |
| **SpO2 Low Warning** | **93,3%** | 94,0% | Selaras sangat dekat (~0,7% selisih) dengan ambang klinis demo |
| **SpO2 Low Critical**| **93,1%** | 90,0% | Penalti rekonstruksi kuadratik melonjak tajam begitu SpO2 < 93% |

### 6c. Injeksi Bertingkat (Graded Anomaly Injections, 20 Episode per Level, N=3)

| Tipe Anomali | Magnitudo Aktual | DenseAE Event Recall | DenseAE Median Latency | Rule Event Recall | Rule Median Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HR_SPIKE (+10 BPM)** | HR $\approx 82$ BPM, SpO2 $\approx 98\%$ | **0,0** (0%) | — | **0,0** (0%) | — |
| **HR_SPIKE (+20 BPM)** | HR $\approx 92$ BPM, SpO2 $\approx 98\%$ | **0,0** (0%) | — | **0,0** (0%) | — |
| **HR_SPIKE (+30 BPM)** | HR $\approx 102$ BPM, SpO2 $\approx 98\%$ | **0,8** (80%) | 4 detik | **1,0** (100%) | 2 detik |
| **HR_SPIKE (+40 BPM)** | HR $\approx 112$ BPM, SpO2 $\approx 98\%$ | **1,0** (100%) | 2 detik | **1,0** (100%) | 2 detik |
| **SPO2_DROP ($\to 95\%$)** | HR $\approx 72$ BPM, SpO2 $\approx 95\%$ | **0,0** (0%) | — | **0,0** (0%) | — |
| **SPO2_DROP ($\to 93\%$)** | HR $\approx 72$ BPM, SpO2 $\approx 93\%$ | **1,0** (100%) | 3 detik | **1,0** (100%) | 2 detik |
| **SPO2_DROP ($\to 91\%$)** | HR $\approx 72$ BPM, SpO2 $\approx 91\%$ | **1,0** (100%) | 2 detik | **1,0** (100%) | 2 detik |
| **SPO2_DROP ($\to 89\%$)** | HR $\approx 72$ BPM, SpO2 $\approx 89\%$ | **1,0** (100%) | 2 detik | **1,0** (100%) | 2 detik |

---

## 7. Verifikasi Lingkungan & Uji Paritas Numerik

1. **Versi Python Virtual Environment Proyek (`.venv`)**:
   - Perintah: `.venv/bin/python --version` $\to$ **`Python 3.14.8`** (Arch/CachyOS rolling release). Tidak menggunakan atau menyebutnya sebagai 3.12.
2. **Status PyTorch pada Lingkungan Lokal**:
   - Perintah: `/tmp/dl_venv/bin/python -c "import torch; print(torch.__version__)"` $\to$ **`ModuleNotFoundError: No module named 'torch'`**.
   - Catatan Pelatihan Aktual: Berdasarkan file metadata terekspor ([`metadata.json`](file:///home/ywenanta/Projects/IoT/ml/artifacts/dl_v1/metadata.json)) dan log eksekusi asli `task-790.log`, pelatihan awal dijalankan menggunakan Python **3.14.8** dengan PyTorch **2.15.0+cpu**.
3. **Uji Paritas ONNX vs NumPy Forward (1.000 Sampel Acak Termasuk Nilai Ekstrem)**:
   - Sampel: 700 sampel fisiologis standar + 300 sampel ekstrem (HR $-20 \dots 350$, SpO2 $-10 \dots 120$).
   - Selisih Absolut Rata-rata: $4,57 \times 10^{-6}$.
   - Selisih Absolut Maksimum: **$2,44 \times 10^{-4}$** (terjadi hanya pada titik ekstrem ekstrem $\text{HR} = -18,38, \text{SpO2} = -0,67$ dengan skor rekonstruksi sangat besar $\approx 1.983$, di mana selisih relatifnya adalah $1,23 \times 10^{-7}$, setara presisi mesin `float32`).
   - Pada rentang fisiologis operasional ($0 \le \text{skor} \le 10$), selisih absolut terverifikasi ketat **$< 10^{-5}$**.
