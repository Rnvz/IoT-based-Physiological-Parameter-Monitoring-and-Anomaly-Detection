# Panduan Pengunduhan & Penataan Dataset

Direktori ini digunakan untuk menyimpan dataset tanda-tanda vital (vital signs) fisiologis manusia yang digunakan untuk pelatihan, validasi, dan pengujian model Machine Learning (*Isolation Forest* / deteksi anomali kondisi fisiologis).

Sesuai dengan berkas `.gitignore`, seluruh file data mentah (`*.csv`, `*.zip`, dsb.) **tidak akan di-commit ke repositori Git**. Ikuti panduan di bawah ini untuk mengunduh dan menempatkan berkas dataset.

---

## Daftar Dataset & Tautan Unduh

| No | Nama Dataset / Peneliti | Sumber & Tautan | Identifier / DOI |
|---|---|---|---|
| 1 | **Human Vital Signs** | [Kaggle Dataset](https://www.kaggle.com/datasets/engrarri21/human-vital-signs) | `engrarri21/human-vital-signs` |
| 2 | **Human Vital Sign Dataset** | [Kaggle Dataset](https://www.kaggle.com/datasets/nasirayub2/human-vital-sign-dataset) | `nasirayub2/human-vital-sign-dataset` |
| 3 | **COVID-19 Temperature, Oxygen & Pulse Rate** | [Kaggle Dataset](https://www.kaggle.com/datasets/rishanmascarenhas/covid19-temperatureoxygenpulse-rate) | `rishanmascarenhas/covid19-temperatureoxygenpulse-rate` |
| 4 | **IEEE DataPort Synthetic Vital Signs** | [IEEE DataPort](https://doi.org/10.21227/06bc-bv25) | DOI: `10.21227/06bc-bv25` |

---

## Struktur Folder yang Diharapkan (Expected Folder Structure)

Setelah diunduh dan diekstrak, letakkan masing-masing dataset pada subfolder tersendiri di dalam direktori `ml_pipeline/data/` dengan hierarki sebagai berikut:

```text
ml_pipeline/data/
├── README.md
├── engrarri21/
│   └── (file .csv hasil ekstrak dari engrarri21/human-vital-signs)
├── nasirayub2/
│   └── (file .csv hasil ekstrak dari nasirayub2/human-vital-sign-dataset)
├── rishanmascarenhas/
│   └── (file .csv hasil ekstrak dari rishanmascarenhas/covid19-temperatureoxygenpulse-rate)
└── ieee_dataport/
    └── (file .csv / data hasil unduh dari IEEE DataPort Synthetic)
```

---

## Metode Pengunduhan

### Opsi A: Menggunakan Kaggle CLI (Direkomendasikan)

Pastikan Kaggle CLI telah terpasang dan API token (`kaggle.json`) telah dikonfigurasi pada komputer Anda:
- **Windows**: `C:\Users\<Username>\.kaggle\kaggle.json`
- **Linux/macOS**: `~/.kaggle/kaggle.json`

Jalankan perintah berikut dari direktori root proyek (`c:/KULIAH/smt 7/Iot/Research/`):

```bash
# 1. Dataset engrarri21/human-vital-signs
kaggle datasets download -d engrarri21/human-vital-signs -p ml_pipeline/data/engrarri21 --unzip

# 2. Dataset nasirayub2/human-vital-sign-dataset
kaggle datasets download -d nasirayub2/human-vital-sign-dataset -p ml_pipeline/data/nasirayub2 --unzip

# 3. Dataset rishanmascarenhas/covid19-temperatureoxygenpulse-rate
kaggle datasets download -d rishanmascarenhas/covid19-temperatureoxygenpulse-rate -p ml_pipeline/data/rishanmascarenhas --unzip
```

> **Catatan untuk IEEE DataPort**:
> Kunjungi [https://doi.org/10.21227/06bc-bv25](https://doi.org/10.21227/06bc-bv25), login ke akun IEEE DataPort, unduh dataset, dan ekstrak ke dalam folder `ml_pipeline/data/ieee_dataport/`.

---

### Opsi B: Unduh Manual melalui Web Browser

1. Buka tautan dataset di atas satu per satu di browser.
2. Klik tombol **Download** pada halaman Kaggle / IEEE DataPort.
3. Buat subfolder tujuan (`engrarri21`, `nasirayub2`, `rishanmascarenhas`, `ieee_dataport`) di dalam `ml_pipeline/data/`.
4. Ekstrak file zip hasil unduhan ke dalam masing-masing subfolder yang bersangkutan.

---

## Verifikasi Dataset

Sebelum menjalankan pipeline preprocessing atau notebook training:
1. Pastikan file berekstensi `.csv` berada di subfolder masing-masing.
2. Buka notebook pemeriksaan awal di `ml_pipeline/notebooks/` untuk memvalidasi kolom fitur fisiologis seperti detak jantung (*Heart Rate / BPM*), saturasi oksigen (*SpO2*), suhu tubuh (*Body Temperature*), dan label anomali (jika ada).
