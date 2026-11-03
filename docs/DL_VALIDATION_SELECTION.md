# Catatan Seleksi Model Validasi Deep Learning

**Sumber**: Log eksekusi run seleksi (di luar repo) (`task-790.log`), dievaluasi sebelum penilaian held-out test set.

---

## 1. Tabel Seleksi Model pada Set Validasi (PR-AUC on Full Validation Partition)

Metrik seleksi primer sesuai pra-registrasi ([`docs/DL_PREREGISTRATION.md`](DL_PREREGISTRATION.md) Bagian 4) dievaluasi pada seluruh partisi validasi (56.309 baris: 25.128 normal dan 31.181 abnormal). Untuk model Deep Learning, angka merupakan rata-rata $\pm$ standar deviasi dari 5 random seed (`42, 43, 44, 45, 46`):

| Model | Varian Fitur | Tipe Model | Val PR-AUC | Val ROC-AUC | Val PR-AUC SD | Val ROC-AUC SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Engine (v3)** | v3 | baseline | 0,8480 | 0,7099 | — | — |
| **Isolation Forest (v2)** | v2 | baseline | 0,8335 | 0,7635 | — | — |
| **Rule Engine (v2)** | v2 | baseline | 0,8279 | 0,6748 | — | — |
| **Local Outlier Factor (v3)** | v3 | baseline | 0,7851 | 0,7450 | — | — |
| **One-Class SVM (v2)** | v2 | baseline | 0,7751 | 0,6984 | — | — |
| **One-Class SVM (v3)** | v3 | baseline | 0,7659 | 0,6918 | — | — |
| **Isolation Forest (v3)** | v3 | baseline | 0,7519 | 0,7361 | — | — |
| **DenseAE (v2)** | **v2** | **deep_learning** | **0,7511** | **0,6621** | **0,0636** | **0,0973** |
| **Local Outlier Factor (v2)** | v2 | baseline | 0,7419 | 0,6973 | — | — |
| **DenseVAE (v3)** | v3 | deep_learning | 0,7410 | 0,6752 | 0,0168 | 0,0279 |
| **DenseVAE (v2)** | v2 | deep_learning | 0,7279 | 0,6362 | 0,0386 | 0,0476 |
| **DenseAE (v3)** | v3 | deep_learning | 0,7254 | 0,6552 | 0,0126 | 0,0187 |

---

## 2. Kriteria Kelayakan Pasang ke Perangkat (*Domain-Shift Sanity Check*)

Sesuai Pasal 7 pra-registrasi ([`docs/DL_PREREGISTRATION.md`](DL_PREREGISTRATION.md)), model deep learning wajib lolos uji kelayakan *resting stream* sintetis pada suhu kulit perifer 33 °C (target alarm palsu $< 5,0\%$):

| Model | Varian Fitur | Alarm% @ 31 °C | Alarm% @ 33 °C (Target $<5\%$) | Alarm% @ 35 °C | Alarm% @ 36,5 °C | Keputusan Pra-Registrasi 33 °C |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseAE (v3)** | v3 | 99,50% | 99,50% | 0,00% | 0,00% | **FAIL (Non-Deployable)** |
| **DenseVAE (v3)** | v3 | 0,00% | 98,75% | 26,75% | 0,00% | **FAIL (Non-Deployable)** |
| **DenseAE (v2)** | **v2** | **0,00%** | **0,00%** | **0,00%** | **0,00%** | **PASS (Deployable)** |
| **DenseVAE (v2)** | **v2** | 0,00% | 0,00% | 0,00% | 0,00% | **PASS (Deployable)** |

---

## 3. Catatan Pertimbangan Seleksi Final

1. **Besaran Selisih PR-AUC Validasi vs Variabilitas Seed**:
   Selisih PR-AUC validasi antara `DenseAE (v2)` ($0,7511$) dan `DenseVAE (v3)` ($0,7410$) adalah sebesar $+0,0101$ ($1,01\%$). Selisih ini **lebih kecil daripada standar deviasi antar-seed** pada `DenseAE (v2)` ($\pm 0,0636$).
2. **Dasar Pemilihan Varian v2**:
   Meskipun varian tri-kanal v3 (`DenseVAE v3`) memiliki metrik validasi yang bersaing ketat dan deviasi antar-seed yang lebih rendah ($\pm 0,0168$), varian v3 gagal total pada kriteria kelayakan pakai hardware akibat pergeseran distribusi suhu inti tubuh klinis vs suhu kontak kulit perangkat IoT (98,75%–99,50% alarm palsu pada 33 °C).
   Oleh karena itu, `DenseAE (v2)` dipilih sebagai pemenang pra-registrasi definitif berdasarkan pemenuhan **kriteria layak pakai** (*deployment criterion*) bebas bias suhu perifer.
