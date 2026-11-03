# Laporan Verifikasi & Hasil Uji Benchmark AOL (dl-public)

> [!IMPORTANT]
> **STATUS PELAPORAN: DIVERIFIKASI SECARA KETAT**
> Dokumen ini mencatat audit independen terhadap integritas data, urutan evaluasi uji, metrik lengkap (ROC-AUC, Rec@5%FPR, prevalensi), analisis bootstrap CI, dan uji sensitivitas domain. Tidak ada model yang dilatih ulang (*no retraining*), tidak ada ambang batas atau split yang diubah, dan model pre-registrasi tetap beku.

---

## 1. Ringkasan Prevalensi Kelas pada Test Set Publik

Held-out test set (`splits/public_split.json`, SHA256: `15a424ff1118cac8dfe3cbd20d80f028863b46edc235db24a2dd50adef6d1635`):

| Sumber Data | Total Sampel | Baris Normal ($y=0$) | Baris Abnormal ($y=1$) | Prevalensi ($P(y=1)$) | Sifat Data |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **POOLED** | 56.188 | 22.641 | 33.547 | **0,5970** (59,70%) | Gabungan multi-sumber |
| **nasirayub2** | 40.004 | 18.880 | 21.124 | **0,5280** (52,80%) | Cross-sectional (sintetis/label BP) |
| **engrarri21** | 4.184 | 149 | 4.035 | **0,9644** (96,44%) | Longitudinal 1 subjek (86% duplikat) |
| **gourangomandal** | 12.000 | 3.612 | 8.388 | **0,6990** (69,90%) | Cross-sectional (sirkularitas 95,5%) |

---

## 2. Tabel Metrik Uji Lengkap (Seluruh Model Baseline & Pemenang DL)

Metrik di bawah dihitung pada partisi uji beku tanpa modifikasi model. Ambang batas operasi dikalibrasi pada validation set normal ($P_{95}$):

| Model | Sumber | Prevalensi | PR-AUC | PR - Prev | ROC-AUC | Rec@5%FPR | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule Engine (v3)** | POOLED | 0.5970 | 0.8871 | +0.2900 | 0.7327 | 1.0000 | 0.6405 |
| Rule Engine (v3) | nasirayub2 | 0.5280 | 0.8098 | +0.2818 | 0.5971 | 1.0000 | 0.3252 |
| Rule Engine (v3) | engrarri21 | 0.9644 | 0.9977 | +0.0333 | 0.9391 | 0.9155 | 0.9548 |
| Rule Engine (v3) | gourangomandal | 0.6990 | 0.9912 | +0.2922 | 0.9753 | 0.9734 | 0.9678 |
| **Rule Engine (v2)** | POOLED | 0.5970 | 0.8719 | +0.2749 | 0.6945 | 1.0000 | 0.5654 |
| Rule Engine (v2) | nasirayub2 | 0.5280 | 0.8098 | +0.2818 | 0.5971 | 1.0000 | 0.3252 |
| Rule Engine (v2) | engrarri21 | 0.9644 | 0.9893 | +0.0249 | 0.7300 | 0.5120 | 0.6763 |
| Rule Engine (v2) | gourangomandal | 0.6990 | 0.9724 | +0.2734 | 0.9151 | 0.8513 | 0.9075 |
| **Isolation Forest (v3)** | POOLED | 0.5970 | 0.8750 | +0.2780 | 0.7969 | 0.5045 | 0.3164 |
| Isolation Forest (v3) | nasirayub2 | 0.5280 | 0.7611 | +0.2331 | 0.6914 | 0.3362 | 0.0036 |
| Isolation Forest (v3) | engrarri21 | 0.9644 | 0.9656 | +0.0012 | 0.4950 | 0.1343 | 0.4918 |
| Isolation Forest (v3) | gourangomandal | 0.6990 | 0.9888 | +0.2898 | 0.9760 | 0.8636 | 0.7407 |
| **Isolation Forest (v2)** | POOLED | 0.5970 | 0.8715 | +0.2744 | 0.7758 | 0.5288 | 0.6418 |
| Isolation Forest (v2) | nasirayub2 | 0.5280 | 0.7787 | +0.2506 | 0.7033 | 0.3821 | 0.4339 |
| Isolation Forest (v2) | engrarri21 | 0.9644 | 0.9848 | +0.0204 | 0.6855 | 0.5678 | 0.7854 |
| Isolation Forest (v2) | gourangomandal | 0.6990 | 0.9833 | +0.2843 | 0.9550 | 0.8716 | 0.9327 |
| **Local Outlier Factor (v3)** | POOLED | 0.5970 | 0.8842 | +0.2871 | 0.7948 | 0.5810 | 0.4936 |
| Local Outlier Factor (v3) | nasirayub2 | 0.5280 | 0.7595 | +0.2314 | 0.6732 | 0.3598 | 0.0100 |
| Local Outlier Factor (v3) | engrarri21 | 0.9644 | 0.9935 | +0.0291 | 0.8974 | 0.6404 | 0.9336 |
| Local Outlier Factor (v3) | gourangomandal | 0.6990 | 0.9987 | +0.2997 | 0.9969 | 0.9884 | 0.9325 |
| **Local Outlier Factor (v2)** | POOLED | 0.5970 | 0.7786 | +0.1816 | 0.7010 | 0.1301 | 0.2270 |
| Local Outlier Factor (v2) | nasirayub2 | 0.5280 | 0.6557 | +0.1276 | 0.6332 | 0.0852 | 0.1648 |
| Local Outlier Factor (v2) | engrarri21 | 0.9644 | 0.9890 | +0.0247 | 0.7680 | 0.4828 | 0.2439 |
| Local Outlier Factor (v2) | gourangomandal | 0.6990 | 0.9679 | +0.2689 | 0.9096 | 0.8284 | 0.3666 |
| **One-Class SVM (v3)** | POOLED | 0.5970 | 0.8502 | +0.2532 | 0.7450 | 0.4720 | 0.4571 |
| One-Class SVM (v3) | nasirayub2 | 0.5280 | 0.6654 | +0.1374 | 0.5932 | 0.1951 | 0.0000 |
| One-Class SVM (v3) | engrarri21 | 0.9644 | 0.9951 | +0.0308 | 0.8885 | 0.6265 | 0.9325 |
| One-Class SVM (v3) | gourangomandal | 0.6990 | 0.9979 | +0.2989 | 0.9951 | 0.9714 | 0.8651 |
| **One-Class SVM (v2)** | POOLED | 0.5970 | 0.8373 | +0.2402 | 0.7289 | 0.4446 | 0.5778 |
| One-Class SVM (v2) | nasirayub2 | 0.5280 | 0.6896 | +0.1615 | 0.6158 | 0.2346 | 0.3066 |
| One-Class SVM (v2) | engrarri21 | 0.9644 | 0.9910 | +0.0266 | 0.8029 | 0.5019 | 0.7634 |
| One-Class SVM (v2) | gourangomandal | 0.6990 | 0.9836 | +0.2846 | 0.9563 | 0.8791 | 0.9273 |
| **DenseAE (v2) [Seed 42]** | POOLED | 0.5970 | 0.7708 | +0.1737 | 0.6385 | 0.2865 | 0.4254 |
| DenseAE (v2) [Seed 42] | nasirayub2 | 0.5280 | 0.5705 | +0.0424 | 0.5258 | 0.0763 | 0.1089 |
| DenseAE (v2) [Seed 42] | engrarri21 | 0.9644 | 0.9843 | +0.0199 | 0.6723 | 0.4045 | 0.6145 |
| DenseAE (v2) [Seed 42] | gourangomandal | 0.6990 | 0.9614 | +0.2624 | 0.8994 | 0.7294 | 0.8399 |

---

## 3. Analisis CI vs Mean pada Model Deep Learning

Pada evaluasi run asli (`task-790.log`), metrik 5-seed dilaporkan sebagai Mean $\pm$ SD, sementara interval kepercayaan (95% CI) dihitung melalui **patient-grouped bootstrap (1.000 resample)** pada prediksi Seed 42 saja (`rep_sc = all_scores[0]`):

1. **Mean 5 Seed vs CI Seed 42**:
   - `POOLED`: PR-AUC 5-seed Mean $\pm$ SD = $0,8071 \pm 0,0589$. Bootstrap 95% CI (Seed 42) = $[0,7400, 0,8109]$.
   - `nasirayub2`: PR-AUC 5-seed Mean $\pm$ SD = $0,6439 \pm 0,1221$. Bootstrap 95% CI (Seed 42) = $[0,5635, 0,5773]$.
   - `engrarri21`: PR-AUC 5-seed Mean $\pm$ SD = $0,9856 \pm 0,0047$. Bootstrap 95% CI (Seed 42) = $[0,9717, 0,9949]$.
   - `gourangomandal`: PR-AUC 5-seed Mean $\pm$ SD = $0,9648 \pm 0,0099$. Bootstrap 95% CI (Seed 42) = $[0,9591, 0,9638]$.
2. **Keterbatasan Ekspor**:
   - Hanya model representatif (Seed 42) yang diekspor bobotnya ke disk (`ml/artifacts/dl_v1/model_weights.npz`).
   - Bobot individu untuk seed 43–46 tidak disimpan ke disk. Sesuai protokol integritas ("Jangan melatih ulang"), model seed 43–46 tidak dilatih ulang secara diam-diam.

---

## 4. Penjelasan Multiplier Persistence FA/1k pada engrarri21

Pada partisi uji `engrarri21`:
- **Penyebut Baris Normal ($N_{normal}$)**: 149 baris.
- **Rasio Skala per 1.000 Baris**: $1.000 / 149 = \mathbf{6,7114}$.
- Setiap kejadian anomali kontinu ($k$ event) menghasilkan nilai eksak kelipatan 6,71:
  - $1 \text{ event} \times 6,7114 = \mathbf{6,71}$
  - $2 \text{ events} \times 6,7114 = \mathbf{13,42}$
  - $4 \text{ events} \times 6,7114 = \mathbf{26,85}$
  - $5 \text{ events} \times 6,7114 = \mathbf{33,56}$
  - $8 \text{ events} \times 6,7114 = \mathbf{53,69}$
- **Struktur Data**: Seluruh 4.184 baris uji `engrarri21` berasal dari **1 subjek tunggal** (`Patient_ID = 21`) dengan fraksi **baris duplikat eksak 85,95% (~86%)**. Dari 149 baris normal penyebut, hanya terdapat **57 baris unik** (92 baris duplikat).

---

## 5. Uji Sensitivitas Domain (Label: "BUKAN hasil test")

Pengujian ini mensimulasikan respons model terhadap aliran data kontinu normal pada suhu permukaan kulit 33 °C (HR $72 \pm 3$, SpO2 $98 \pm 0,5\%$, Temp $33,0 \pm 0,3$ °C) dengan injeksi anomali sintetis (`ml_eval/inject.py`):

> [!NOTE]
> File `sample_fixture.csv` di direktori `tests/fixtures/` adalah **fixture sintetis** yang dibuat khusus untuk memverifikasi logika pipa eksekusi tanpa memerlukan perangkat fisik.

### Hasil Injeksi Anomali (Episode Length: 20 detik):
1. **HR_SPIKE (Lonjakan Denyut Jantung Tachycardia +45–60 BPM)**:
   - **DenseAE (v2)**: Event Recall = **100% (1/1)** | Latensi Deteksi = **2 detik** (efek filter persistensi $N=3$) | Sample Recall = 90,0%.
   - **Rule Default**: Event Recall = **100% (1/1)** | Latensi Deteksi = **0 detik** (langsung memicu threshold HR > 100) | Sample Recall = 100,0%.
   - **DenseVAE (v2)**: `UNVERIFIED` (bobot model tidak diekspor ke disk; dilarang melatih ulang sesuai batasan).
2. **SPO2_DROP (Penurunan Saturasi Oksigen Hypoxemia < 90%)**:
   - **DenseAE (v2)**: Event Recall = **100% (1/1)** | Latensi Deteksi = **2 detik** (efek filter persistensi $N=3$) | Sample Recall = 90,0%.
   - **Rule Default**: Event Recall = **100% (1/1)** | Latensi Deteksi = **0 detik** (langsung memicu threshold SpO2 < 90) | Sample Recall = 100,0%.
   - **DenseVAE (v2)**: `UNVERIFIED` (bobot model tidak diekspor ke disk; dilarang melatih ulang sesuai batasan).
