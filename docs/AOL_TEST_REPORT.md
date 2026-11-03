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

| Metrik Audit | Kolom HR & Temp Sahaja | Seluruh Fitur Model (`HR, SpO2, Temp`) | Seluruh Kolom Termasuk Metadata |
| :--- | :--- | :--- | :--- |
| **Total Baris Test** | 4.184 | 4.184 | 4.184 |
| **Baris Unik Test** | 297 (Duplikat: **92,90%**) | 557 (Duplikat: **86,69% ~ 86%**) | 557 (Duplikat: **86,69%**) |
| **Total Baris Normal ($y=0$)** | 149 | 149 | 149 |
| **Baris Normal Unik** | 42 (Duplikat: **71,81%**) | 57 (Duplikat: **61,74%**) | 57 (Duplikat: **61,74%**) |

### c. Rekonsiliasi & Penghapusan Angka Hitungan Lama

Seluruh tabel dan angka hitungan lama pada draf laporan terdahulu—termasuk kemunculan angka seperti `896` atau `2.897` pada `engrarri21`, serta `22.596`, `23.605`, atau `20.627` pada partisi POOLED—telah sepenuhnya dihapus dan dibersihkan dari dokumen ini. Angka-angka tersebut tidak cocok dengan data riil pada `ml_pipeline/data/harmonized_vitals.csv` dan `splits/public_split.json`, asalnya tidak dapat direkonstruksi dari pipeline saat ini, dan angka-angka tersebut tidak boleh digunakan. Tabel definitif yang berlaku mutlak adalah tabel pada Bagian 2a di atas (POOLED Train: 71.315 normal / 97.611 abnormal; Val: 25.128 normal / 31.181 abnormal; Test: 22.641 normal / 33.547 abnormal; Total: 281.543 baris). Persentase kesepakatan nyata antara aturan default `evaluate_rules` dengan label `OUTPUT` dataset adalah 95,46% (`gourangomandal`), 79,83% (`engrarri21`), dan 57,61% (`nasirayub2`).

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

> [!NOTE]
> **Koreksi Metrik Post-Test (Bukan Tuning Model)**:
> Koreksi formula Rec@5%FPR dilakukan secara *post-test* murni sebagai perbaikan bug (*bugfix*) perhitungan metrik pada skor bertingkat/diskret (*tied scores*), BUKAN penyesuaian (*tuning*) terhadap model, representasi fitur, maupun ambang batas deteksi. Seluruh bobot model dan ambang batas tetap beku sesuai pra-registrasi.

---

## 4. Tabel Metrik Uji Lengkap Terkalibrasi (Termasuk engrarri21 Unique)

Dihitung ulang dengan `scripts/aol_verify.py` pada seluruh 56.188 sampel test set publik:

| Model | Sumber | Prevalensi | PR-AUC | PR - Prev | ROC-AUC | Rec@5%FPR | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule Engine (v3)** | POOLED | 0.5970 | 0.8871 | +0.2900 | 0.7327 | **0.3228** | 0.6405 |
| Rule Engine (v3) | nasirayub2 | 0.5280 | 0.8098 | +0.2818 | 0.5971 | **0.0036** | 0.3252 |
| Rule Engine (v3) | engrarri21 | 0.9644 | 0.9977 | +0.0333 | 0.9391 | 0.8912 | 0.9548 |
| Rule Engine (v3) | *engrarri21_unique* | 0.8977 | 0.9931 | +0.0954 | 0.9387 | 0.0000 | 0.9610 |
| Rule Engine (v3) | gourangomandal | 0.6990 | 0.9912 | +0.2922 | 0.9753 | 0.9707 | 0.9678 |
| **Rule Engine (v2)** | POOLED | 0.5970 | 0.8719 | +0.2749 | 0.6945 | **0.2033** | 0.5654 |
| Rule Engine (v2) | nasirayub2 | 0.5280 | 0.8098 | +0.2818 | 0.5971 | **0.0036** | 0.3252 |
| Rule Engine (v2) | engrarri21 | 0.9644 | 0.9893 | +0.0249 | 0.7300 | 0.4238 | 0.6763 |
| Rule Engine (v2) | *engrarri21_unique* | 0.8977 | 0.9718 | +0.0742 | 0.7652 | 0.0000 | 0.7568 |
| Rule Engine (v2) | gourangomandal | 0.6990 | 0.9724 | +0.2734 | 0.9151 | 0.8436 | 0.9075 |
| **Isolation Forest (v3)** | POOLED | 0.5970 | 0.8750 | +0.2780 | 0.7969 | 0.5045 | 0.3164 |
| Isolation Forest (v3) | nasirayub2 | 0.5280 | 0.7611 | +0.2331 | 0.6914 | 0.3362 | 0.0036 |
| Isolation Forest (v3) | engrarri21 | 0.9644 | 0.9656 | +0.0012 | 0.4950 | 0.1343 | 0.4918 |
| Isolation Forest (v3) | *engrarri21_unique* | 0.8977 | 0.8847 | -0.0129 | 0.4399 | 0.0760 | 0.5597 |
| Isolation Forest (v3) | gourangomandal | 0.6990 | 0.9888 | +0.2898 | 0.9760 | 0.8636 | 0.7407 |
| **Isolation Forest (v2)** | POOLED | 0.5970 | 0.8715 | +0.2744 | 0.7758 | 0.5288 | 0.6418 |
| Isolation Forest (v2) | nasirayub2 | 0.5280 | 0.7787 | +0.2506 | 0.7033 | 0.3821 | 0.4339 |
| Isolation Forest (v2) | engrarri21 | 0.9644 | 0.9848 | +0.0204 | 0.6855 | 0.4612 | 0.7854 |
| Isolation Forest (v2) | *engrarri21_unique* | 0.8977 | 0.9614 | +0.0637 | 0.7492 | 0.3240 | 0.8189 |
| Isolation Forest (v2) | gourangomandal | 0.6990 | 0.9833 | +0.2843 | 0.9550 | 0.8716 | 0.9327 |
| **Local Outlier Factor (v3)** | POOLED | 0.5970 | 0.8842 | +0.2871 | 0.7948 | 0.5810 | 0.4936 |
| Local Outlier Factor (v3) | nasirayub2 | 0.5280 | 0.7595 | +0.2314 | 0.6732 | 0.3598 | 0.0100 |
| Local Outlier Factor (v3) | engrarri21 | 0.9644 | 0.9935 | +0.0291 | 0.8974 | 0.6387 | 0.9336 |
| Local Outlier Factor (v3) | *engrarri21_unique* | 0.8977 | 0.9725 | +0.0749 | 0.8707 | 0.4540 | 0.9311 |
| Local Outlier Factor (v3) | gourangomandal | 0.6990 | 0.9987 | +0.2997 | 0.9969 | 0.9884 | 0.9325 |
| **Local Outlier Factor (v2)** | POOLED | 0.5970 | 0.7786 | +0.1816 | 0.7010 | 0.1301 | 0.2270 |
| Local Outlier Factor (v2) | nasirayub2 | 0.5280 | 0.6557 | +0.1276 | 0.6332 | 0.0852 | 0.1648 |
| Local Outlier Factor (v2) | engrarri21 | 0.9644 | 0.9890 | +0.0247 | 0.7680 | 0.4701 | 0.2439 |
| Local Outlier Factor (v2) | *engrarri21_unique* | 0.8977 | 0.9624 | +0.0647 | 0.7651 | 0.1160 | 0.2131 |
| Local Outlier Factor (v2) | gourangomandal | 0.6990 | 0.9679 | +0.2689 | 0.9096 | 0.8271 | 0.3666 |
| **One-Class SVM (v3)** | POOLED | 0.5970 | 0.8502 | +0.2532 | 0.7450 | 0.4720 | 0.4571 |
| One-Class SVM (v3) | nasirayub2 | 0.5280 | 0.6654 | +0.1374 | 0.5932 | 0.1951 | 0.0000 |
| One-Class SVM (v3) | engrarri21 | 0.9644 | 0.9951 | +0.0308 | 0.8885 | 0.6226 | 0.9325 |
| One-Class SVM (v3) | *engrarri21_unique* | 0.8977 | 0.9820 | +0.0843 | 0.8640 | 0.4060 | 0.9168 |
| One-Class SVM (v3) | gourangomandal | 0.6990 | 0.9979 | +0.2989 | 0.9951 | 0.9707 | 0.8651 |
| **One-Class SVM (v2)** | POOLED | 0.5970 | 0.8373 | +0.2402 | 0.7289 | 0.4446 | 0.5778 |
| One-Class SVM (v2) | nasirayub2 | 0.5280 | 0.6896 | +0.1615 | 0.6158 | 0.2346 | 0.3066 |
| One-Class SVM (v2) | engrarri21 | 0.9644 | 0.9910 | +0.0266 | 0.8029 | 0.4800 | 0.7634 |
| One-Class SVM (v2) | *engrarri21_unique* | 0.8977 | 0.9725 | +0.0749 | 0.8086 | 0.3260 | 0.8070 |
| One-Class SVM (v2) | gourangomandal | 0.6990 | 0.9836 | +0.2846 | 0.9563 | 0.8785 | 0.9273 |
| **DenseAE (v2) [Seed 42]** | POOLED | 0.5970 | 0.7708 | +0.1737 | 0.6385 | 0.2865 | 0.4254 |
| DenseAE (v2) [Seed 42] | nasirayub2 | 0.5280 | 0.5705 | +0.0424 | 0.5258 | 0.0763 | 0.1089 |
| DenseAE (v2) [Seed 42] | engrarri21 | 0.9644 | 0.9843 | +0.0199 | 0.6723 | 0.3941 | 0.6145 |
| DenseAE (v2) [Seed 42] | *engrarri21_unique* | 0.8977 | 0.9582 | +0.0605 | 0.7156 | 0.2800 | 0.6498 |
| DenseAE (v2) [Seed 42] | gourangomandal | 0.6990 | 0.9614 | +0.2624 | 0.8994 | 0.7277 | 0.8399 |

> [!NOTE]
> **Catatan Analisis Rec@5%FPR Rule = 0,0000 & Ketidakstabilan pada 57 Baris Normal `engrarri21_unique`**:
> - **Distribusi Keputusan Rule pada 57 Baris Normal**:
>   - *Rule v3* (`HR, SpO2, Temp`): Severity 0 = 51 baris, Severity 1 = 3 baris, Severity 2 = 3 baris.
>   - *Rule v2* (`HR, SpO2, Temp=36.5`): Severity 0 = 51 baris, Severity 1 = 3 baris, Severity 2 = 3 baris.
> - **Titik-titik Kurva ROC Rule dengan $\text{FPR} \le 0,05$**:
>   - Ambang batas $\infty$ (skor $> 1,0$): $\text{FPR} = 0/57 = 0,0000$, $\text{Recall} = 0,0000$.
>   - Ambang batas $1,0$ (skor $\ge 1,0$, Severity 2): $\text{FPR} = 3/57 = 0,052632$ ($5,26\% > 5,0\%$), $\text{Recall} = 0,9120$ (v3) / $0,4940$ (v2).
>   - Ambang batas $0,5$ (skor $\ge 0,5$, Severity $\ge 1$): $\text{FPR} = 6/57 = 0,105263$ ($10,53\%$).
> - **Mengapa Rec@5%FPR Rule bernilai 0,0000**:
>   Dengan hanya 57 baris normal, batas $\text{FPR} \le 0,05$ membatasi toleransi alarm palsu maksimal $\lfloor 57 \times 0,05 \rfloor = 2$ baris ($\text{FPR} \le 3,51\%$). Karena Rule Engine memicu 3 baris false alarm bahkan pada tingkat keparahan tertinggi (Severity 2), titik diskret pertama kurva ROC langsung melonjak ke $5,26\%$ ($> 5\%$). Tanpa interpolasi buatan, satu-satunya titik valid yang memenuhi syarat $\text{FPR} \le 0,05$ adalah threshold $\infty$ ($\text{FPR}=0,0000$), yang menghasilkan $\text{Recall} = 0,0000$.
> - **Ketidakstabilan Statistik**:
>   Pada ukuran sampel normal sekecil $N=57$, 1 baris sampel merepresentasikan $\sim 1,75\%$ bobot FPR. Pergeseran hanya 1 baris sampel (antara 2 dan 3 alarm palsu) memicu perpindahan ekstrem dari $\text{Recall} > 91\%$ menjadi $\text{Recall} = 0,0000$. Oleh karena itu, metrik Rec@5%FPR ini sangat sensitif dan tidak stabil secara statistik (*statistically unstable*) pada subset 57 baris normal dan tidak boleh diinterpretasikan sebagai ketiadaan kemampuan deteksi.

---

## 5. Audit Bootstrap Resampling pada `engrarri21`

Hasil penelusuran kode pada `patient_group_bootstrap` ([`ml/dl/evaluate.py#L149-L165`](../ml/dl/evaluate.py#L149-L165)):
- Pada subset `engrarri21`, seluruh baris berasal dari satu pasien tunggal (`n_pts == 1`).
- Eksekusi jatuh ke blok bootstrap deret waktu (blok 50 baris). Blok-blok ini di-resample dengan pengembalian dari dalam rekaman kontinu pasien yang sama.
- **Audit Verdict**: Interval kepercayaan (CI) untuk `engrarri21` diberi label:
  > **[!] CI TIDAK VALID: 1 SUBJEK, 86% DUPLIKAT**
  > Resampling blok dalam satu individu yang memiliki 86% duplikasi baris tidak dapat mengestimasi variabilitas antar-pasien klinis nyata. Metrik pada baris unik (`engrarri21_unique`) di atas menunjukkan penurunan performa wajar (misal: PR-AUC DenseAE v2 turun dari 0,9843 menjadi 0,9582 dan Rec@5%FPR turun dari 39,41% menjadi 28,00%).

---

## 6. Uji Sensitivitas Domain & Robustness (BUKAN Hasil Test)

> [!WARNING]
> **LABEL: BUKAN HASIL TEST**
> Pengujian ini adalah simulasi sensitivitas domain sintetis untuk mengevaluasi ketahanan terhadap pergeseran distribusi (*domain shift*) dan alarm palsu, bukan bagian dari evaluasi test set publik beku.
> - **Kelulusan Mekanis/Trivial pada Tes Resting 33 °C**: Model varian v2 (`DenseAE v2`) secara sengaja membuang fitur suhu (`Temperature`) dari ruang inputnya (hanya memakai `Heart_Rate` dan `SpO2`). Akibatnya, penurunan suhu dari baseline core klinis (~37 °C) ke suhu kontak kulit perifer (~33 °C) tidak memengaruhi skor anomali model sama sekali. Kelulusan pada tes resting 33 °C ini (0,0% alarm) adalah kelulusan mekanis/trivial secara konstruksi fitur, dan ditegaskan sebagai **BUKAN HASIL TEST** empiris kemampuan adaptasi model terhadap variabilitas termal.
> - **Dua Perlakuan Rule yang Dievaluasi**:
>   1. **Rule Mentah (*Raw Production*)**: Evaluasi instan tanpa persistensi ($N=1$), merefleksikan perilaku backend produksi riil saat ini.
>   2. **Rule + Persistensi ($N=3$)**: Memerlukan 3 sampel berturut-turut di atas ambang batas.
>   3. **DenseAE (v2) [Seed 42]**: Menggunakan filter persistensi $N=3$ sesuai pra-registrasi.

### 6a. Grid Alarm Palsu (20 Seed Aliran x 400 Sampel = 8.000 Sampel per Sel)

| Rata-rata HR (BPM) | Rata-rata SpO2 (%) | DenseAE (N=3) Sev $\ge 1$ (%) | DenseAE (N=3) Sev 2 (%) | Rule Mentah Sev $\ge 1$ (%) | Rule Mentah Sev 2 (%) | Rule (N=3) Sev $\ge 1$ (%) | Rule (N=3) Sev 2 (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **55** | 95 | 81,76% | 70,95% | 51,66% | 4,84% | 13,68% | 0,00% |
| **55** | 96 | 34,51% | 20,21% | 50,69% | 4,84% | 13,03% | 0,00% |
| **55** | 98 | 3,20% | 1,20% | 50,69% | 4,84% | 13,03% | 0,00% |
| **60** | 95 | 18,40% | 9,69% | 6,80% | 0,03% | 0,03% | 0,00% |
| **60** | 96 | 0,51% | 0,14% | 4,84% | 0,03% | 0,00% | 0,00% |
| **60** | 98 | 0,03% | 0,00% | 4,84% | 0,03% | 0,00% | 0,00% |
| **72** (Baseline) | 95 | 0,00% | 0,00% | 2,01% | 0,00% | 0,00% | 0,00% |
| **72** (Baseline) | 96 | 0,00% | 0,00% | 0,00% | 0,00% | 0,00% | 0,00% |
| **72** (Baseline) | 98 | **0,00%** | **0,00%** | **0,00%** | **0,00%** | **0,00%** | **0,00%** |
| **85** | 95 | 0,01% | 0,00% | 2,04% | 0,00% | 0,00% | 0,00% |
| **85** | 96 | 0,00% | 0,00% | 0,03% | 0,00% | 0,00% | 0,00% |
| **85** | 98 | 0,00% | 0,00% | 0,03% | 0,00% | 0,00% | 0,00% |
| **95** | 95 | 41,86% | 3,09% | 50,35% | 4,38% | 12,91% | 0,05% |
| **95** | 96 | 0,95% | 0,00% | 49,31% | 4,38% | 12,15% | 0,05% |
| **95** | 98 | 0,00% | 0,00% | 49,31% | 4,38% | 12,15% | 0,05% |

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
| **SpO2 Low Warning** | **93,3%** | 94,0% | Selaras sangat dekat (~0,7% selisih) dengan ambang batas demo |
| **SpO2 Low Critical**| **93,1%** | 90,0% | Penalti rekonstruksi kuadratik melonjak tajam begitu SpO2 < 93% |

### 6c. Injeksi Bertingkat (Graded Anomaly Injections, 20 Episode per Level)

| Tipe Anomali | Target Vitals | DenseAE (N=3) Recall | DenseAE (N=3) Med Latency | Rule Mentah (Produksi) Recall | Rule Mentah Med Latency | Rule (N=3) Recall | Rule (N=3) Med Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HR_SPIKE (+10 BPM)** | HR $\approx 82$, SpO2 $\approx 98$ | **0,0** (0%) | — | **0,00** (0%) | — | **0,0** (0%) | — |
| **HR_SPIKE (+20 BPM)** | HR $\approx 92$, SpO2 $\approx 98$ | **0,0** (0%) | — | **0,75** (75%) | 5 detik | **0,0** (0%) | — |
| **HR_SPIKE (+30 BPM)** | HR $\approx 102$, SpO2 $\approx 98$ | **0,8** (80%) | 4 detik | **1,00** (100%) | **0 detik** | **1,0** (100%) | 2 detik |
| **HR_SPIKE (+40 BPM)** | HR $\approx 112$, SpO2 $\approx 98$ | **1,0** (100%) | 2 detik | **1,00** (100%) | **0 detik** | **1,0** (100%) | 2 detik |
| **SPO2_DROP ($\to 95\%$)** | HR $\approx 72$, SpO2 $\approx 95$ | **0,0** (0%) | — | **0,05** (5%) | 18 detik | **0,0** (0%) | — |
| **SPO2_DROP ($\to 93\%$)** | HR $\approx 72$, SpO2 $\approx 93$ | **1,0** (100%) | 3 detik | **1,00** (100%) | **0 detik** | **1,0** (100%) | 2 detik |
| **SPO2_DROP ($\to 91\%$)** | HR $\approx 72$, SpO2 $\approx 91$ | **1,0** (100%) | 2 detik | **1,00** (100%) | **0 detik** | **1,0** (100%) | 2 detik |
| **SPO2_DROP ($\to 89\%$)** | HR $\approx 72$, SpO2 $\approx 89$ | **1,0** (100%) | 2 detik | **1,00** (100%) | **0 detik** | **1,0** (100%) | 2 detik |

### 6d. Peta Alarm Sintetis pada Aliran Konstan (BUKAN Hasil Test)

> [!WARNING]
> **LABEL: BUKAN HASIL TEST**
> Pengujian ini dilakukan dengan mengalirkan input konstan $\ge 3$ sampel berurutan ($N=3$) pada model `DenseAE (v2)` dan `Rule Engine` default untuk memetakan $\text{SpO}_2$ tertinggi yang memicu Severity 1 (Warning) dan Severity 2 (Critical Alert) pada berbagai level Heart Rate ($HR \in \{50, 60, 72, 90, 100, 120\}$, rentang $\text{SpO}_2 \in [90,0\%, 99,0\%]$ dengan langkah $0,5\%$):

| HR (BPM) | SpO2 Tertinggi DL Sev 1 | SpO2 Tertinggi DL Sev 2 | SpO2 Tertinggi Rule Sev 1 | SpO2 Tertinggi Rule Sev 2 | Karakteristik Perbandingan DL vs Rule |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **50** | **99,0%** | **99,0%** | 99,0% | Tidak Terpicu | HR=50 (bradikardia berat) memicu anomali multivariat DL di seluruh rentang SpO2; Rule memicu warning di semua SpO2 ($50 \le HR < 55$). |
| **60** | **95,0%** | **94,5%** | 93,5% | Tidak Terpicu | DL memicu warning pada 95,0% dan critical pada 94,5%; Rule default baru memicu warning pada SpO2 $\le 94\%$ (93,5%). |
| **72** (Baseline) | **93,0%** | **93,0%** | 93,5% | Tidak Terpicu | Pada HR baseline normal, DL langsung memicu Sev 2 pada 93,0% karena penalti kuadratik melonjak tajam; Rule Sev 2 tidak terpicu pada rentang ini (butuh $<90\%$). |
| **90** | **94,5%** | **92,0%** | 93,5% | Tidak Terpicu | Warning DL aktif pada 94,5%, alert kritis pada 92,0%; Rule default Sev 1 pada 93,5%. |
| **100** | **96,0%** | **95,0%** | 99,0% | Tidak Terpicu | Rule memicu warning di semua SpO2 karena HR=100 ($95 < HR \le 100$); DL memicu warning pada 96,0% dan alert pada 95,0%. |
| **120** | **98,5%** | **96,5%** | 99,0% | 99,0% | Rule memicu alert kritis di semua SpO2 ($HR > 100$); DL memicu warning pada 98,5% dan alert kritis pada 96,5%. |

---

## 7. Fakta Lingkungan Eksekusi & Uji Paritas Numerik

### a. Status dan Rekam Jejak Lingkungan Virtual (`.venv`)
1. **Kehilangan `.venv` Lama (Python 3.12 via `uv`)**:
   - Direktori virtual environment awal `.venv` yang dibuat menggunakan `uv` merujuk ke path eksekutor binary `/home/ywenanta/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`.
   - **Penyebab**: Executable `uv` dan direktori `~/.local/share/uv/python` tidak tersedia pada sistem operasi (Arch Linux/CachyOS), sehingga symlink python pada `.venv` lama rusak (*broken symlink*) dan tidak dapat dieksekusi.
2. **Rekonstruksi & Keadaan Sekarang**:
   - `.venv` proyek dibuat ulang menggunakan Python sistem (`Python 3.14.8`).
   - Pada sesi sebelumnya, paket dependensi disalin via `cp -r` dari `/tmp/dl_venv/lib/python3.14/site-packages/*` ke dalam `.venv` untuk memulihkan fungsi runtime evaluasi (sehingga klaim awal "site-packages tidak disalin" telah dikoreksi sesuai riwayat nyata).
   - `.venv` saat ini berstatus bersih menurut `.venv/bin/pip check` (`No broken requirements found`), memuat `scikit-learn 1.9.1`, `onnxruntime 1.31.1`, `numpy 2.5.4`, `scipy 1.18.1`, `pandas 3.1.0`.
3. **Status PyTorch & Tes yang Di-skip**:
   - PyTorch binary **tidak terpasang** di `.venv` maupun `/tmp/dl_train_venv`. Percobaan instalasi PyTorch CUDA di `/tmp` gagal karena kehabisan ruang disk RAM `tmpfs` (`OSError: [Errno 28] No space left on device`).
   - Eksekusi `tests/test_dl_public.py` menghasilkan **6 PASSED** dan **4 SKIPPED** (ditandai `UNVERIFIED` karena ketiadaan PyTorch):
     - `test_determinism_across_seeds` (SKIPPED: PyTorch not installed)
     - `test_early_stopping_on_val_loss` (SKIPPED: PyTorch not installed)
     - `test_export_parity_torch_onnx_numpy` (SKIPPED: PyTorch not installed)
     - `test_anomaly_score_direction` (SKIPPED: PyTorch not installed)

### b. Uji Paritas ONNX vs NumPy Forward (10.000 Sampel Termasuk Titik Ekstrem & Batas Ambang)
- **Titik Sampel**: 10.000 titik (distribusi fisiologis normal/abnormal, sapuan batas ambang, dan nilai ekstrem).
- **Maksimum Selisih Absolut ($\text{skor} \le 10.0$)**: $4,77 \times 10^{-6}$ ($< 10^{-5}$).
- **Maksimum Selisih Relatif (seluruh sampel)**: $4,19 \times 10^{-4}$ ($0,0419\%$).
- **Zona Kritis Ambang Batas ($[0,005, 0,05]$ mencakup $\text{warn}=0,0136$ dan $\text{crit}=0,0248$, $N=2.161$)**:
  - Maksimum Selisih Absolut: **$4,43 \times 10^{-7}$**.
  - Maksimum Selisih Relatif: **$2,18 \times 10^{-5}$** ($0,00218\%$).
- Selisih paritas pada batas keputusan berada pada orde $\sim 10^{-7}$, jauh di bawah marjin toleransi ambang batas model, membuktikan kesepakatan klasifikasi 100% identik antara ONNX runtime dan NumPy runtime.

### c. Cakupan Pengukuran Latensi Inferensi (0,17 ms)
Angka latensi inferensi $\approx 0,17\text{ ms}$ ($\approx 170\text{--}200\text{ }\mu\text{s}$) yang dilaporkan pada ringkasan benchmark adalah durasi *raw forward pass* model matematika (`DenseAE v2`) secara lokal di CPU (komputasi matriks ONNX / NumPy array forward pass). Angka ini **BUKAN latensi end-to-end sistem telemetri IoT**, yang secara riil di lapangan didominasi oleh:
1. Sampling fisik sensor I2C/1-Wire pada mikrokontroler ESP32 (~50–100 ms),
2. Enkoding dan serialisasi string JSON pada firmware,
3. Latensi transmisi jaringan TCP/MQTT via koneksi WiFi/hotspot ke broker (~10–50 ms),
4. Antrean pemrosesan event loop asynchronous pada FastAPI backend,
5. Penulisan persisten ke media penyimpanan disk (`predictions.csv`),
6. Latensi refresh dan rendering visual pada browser web dashboard.

