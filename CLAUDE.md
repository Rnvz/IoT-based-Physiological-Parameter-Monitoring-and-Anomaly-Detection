# Project Context & Agent Guide: Sistem Pemantauan dan Deteksi Anomali Parameter Fisiologis Berbasis IoT

Dokumen ini adalah acuan utama bagi AI Agent dan developer dalam mengembangkan sistem pemantauan dan deteksi anomali parameter fisiologis menggunakan ESP32, sinyal PPG, SpO₂, suhu kulit, dan Machine Learning (Isolation Forest).

---

## 1. Ringkasan Proyek

- **Nama Proyek**: Sistem Pemantauan dan Deteksi Anomali Parameter Fisiologis Berbasis IoT Menggunakan Sinyal PPG, SpO₂, Suhu, dan Machine Learning
- **Tujuan**: Membangun sistem prototipe IoT (*Desktop Demo Testbed* utama, dengan opsi implementasi *Wearable* spesifik jika memungkinkan) untuk pemantauan parameter fisiologis *real-time* dan *early-warning* non-klinis guna mendeteksi penyimpangan kombinasi parameter fisiologis terhadap pola normal menggunakan Machine Learning multivariat (*Feature Fusion*).
- **Masalah yang Diselesaikan**:
  - Pendekatan ambang batas (*threshold*) konvensional berbasis satu parameter sering memicu *false alarm* dan tidak peka terhadap kombinasi perubahan simultan antar parameter.
  - Artefak pergerakan (*motion artifact*) dan kontak sensor yang tidak stabil pada pergelangan tangan sering disalahartikan sebagai anomali medis. Sistem ini menambahkan lapisan *Signal Quality Assessment* (SQA) dan *persistent anomaly logic*.
- **Target Pengguna**: Peneliti, pengembang, dan pengguna umum untuk pemantauan fisiologis harian/kondisi istirahat (*non-clinical use case*).
- **Disclaimer Wajib**: **Sistem ini BUKAN alat diagnosis medis**. Status "Pola Menyimpang" hanya mencerminkan deviasi dari distribusi data normal yang dipelajari model, bukan vonis penyakit tertentu (misal: aritmia, hipoksia, demam, dll).

---

## 2. Arsitektur High-Level

Aliran data sistem dirancang secara berjenjang dari sisi *edge* hingga *backend* dan *dashboard*:

```
[ MAX30102 (PPG, HR, SpO2) ]  ──(I2C)─────┐
                                           ├──> [ ESP32-WROOM Node ]
[ DS18B20 (Skin Temp) ]       ──(OneWire)─┘         │
                                                    │ (Wi-Fi: MQTT / HTTP-WS)
                                                    ▼
                                          [ Python Backend Server ]
                                          ├── Signal Quality Assessment (SQA)
                                          ├── Temporal Feature Fusion Engine
                                          ├── Isolation Forest Inference
                                          └── Baseline Threshold Comparator
                                                    │
                      ┌─────────────────────────────┴─────────────────────────────┐
                      ▼                                                           ▼
         [ Feedback to ESP32 ]                                           [ PC Dashboard ]
     ├── OLED: HR, SpO2, Temp, Status                                    ├── Real-time Parameter Charts
     └── Active Buzzer: Persistent Anomaly Alarm                         ├── Anomaly & SQA Scores
                                                                         └── Device Connection Health
```

---

## 3. Tech Stack & Rekomendasi Teknis

| Komponen | Teknologi Terpilih / Usulan | Justifikasi Singkat |
| :--- | :--- | :--- |
| **Firmware Framework** | **PlatformIO + C++ (Arduino Framework for ESP32)** | Kompatibilitas luas dengan library `SparkFun MAX3010X`, `DallasTemperature`/`OneWire`, dan `Adafruit SSD1306`. PlatformIO mempermudah manajemen dependensi. |
| **Protokol IoT** | **MQTT (Mosquitto) / HTTP + WebSocket** | **Rekomendasi Utama**: MQTT untuk pub/sub telemetri hemat daya & dua arah (telemetri sensor ke backend, feedback status ke ESP32). Alternatif: FastAPI WebSocket jika ingin arsitektur tanpa broker terpisah. |
| **Backend API / Runtime** | **Python 3.10+ (FastAPI)** | Performa tinggi asynchronous I/O, ekosistem ML Python yang kuat, serta dukungan native untuk WebSocket/REST endpoint. |
| **Data Processing & ML** | **Scikit-Learn, Pandas, NumPy, SciPy** | `IsolationForest` untuk *unsupervised anomaly detection*, `scipy.signal` untuk filter sinyal/SQA dasar, `pandas` untuk manipulasi dataset & temporal sliding window. |
| **PC Dashboard** | **Streamlit** (atau **React / Vite + Tailwind + Chart.js**) | **Rekomendasi Fase 1**: Streamlit untuk visualisasi data riset yang cepat dibangun; dapat ditingkatkan ke React/Web SPA jika dibutuhkan latensi render ultra-responsif. |
| **Database / Logging** | **SQLite** / file CSV terstruktur | Cukup untuk logging sesi eksperimen lokal prototipe tanpa overhead database server skala enterprise. |

---

## 4. Struktur Folder Standar

```
Research/
├── CLAUDE.md                    # File acuan utama agent & developer
├── REQUIREMENTS.md              # Spesifikasi kebutuhan & constraint teknis
├── ARCHITECTURE.md              # Detail arsitektur, diagram Mermaid, & skema data
├── ROADMAP.md                   # Rencana tahapan pengerjaan (MVP to evaluation)
├── SECURITY.md                  # Kebijakan kredensial, autentikasi, & safety hardware
├── proposal.pdf                 # Dokumen proposal asli
│
├── firmware/                    # Kode embedded mikrokontroler (ESP32)
│   ├── platformio.ini           # Konfigurasi PlatformIO & dependensi lib
│   ├── include/                 # Header files (*.h)
│   │   ├── config.h             # Pinout, interval sampling, dsb (bebas secret)
│   │   └── sensors.h
│   └── src/                     # Source files (*.cpp)
│       ├── main.cpp             # Setup, loop, Wi-Fi reconnection
│       ├── sensor_manager.cpp   # Driver MAX30102 & DS18B20
│       ├── display_manager.cpp  # Driver OLED SSD1306
│       └── comm_client.cpp      # Klien MQTT / WebSocket
│
├── backend/                     # Layanan backend pemrosesan data & inferensi
│   ├── app/
│   │   ├── main.py              # Entrypoint server (FastAPI / MQTT service)
│   │   ├── core/                # Konfigurasi, logging, singleton
│   │   ├── services/
│   │   │   ├── sqa.py           # Signal Quality Assessment (PPG validation)
│   │   │   ├── feature_fusion.py# Temporal feature extraction (rolling window)
│   │   │   └── anomaly_model.py # Wrapper Isolation Forest & Threshold baseline
│   │   └── schemas/             # Pydantic models / payload schema
│   └── requirements.txt         # Dependensi Python backend
│
├── ml_pipeline/                 # Eksperimen data offline & pelatihan model
│   ├── data/                    # Folder dataset (Kaggle: engrarri21, nasirayub2, rishan & IEEE DataPort)
│   ├── notebooks/               # Jupyter/Colab notebook untuk eksplorasi
│   ├── src/                     # Script preprocessing, training, & export model
│   │   ├── clean_datasets.py    # Normalisasi HR, SpO2, Temp
│   │   └── train_isolation_forest.py
│   └── models/                  # Serialized model (*.joblib / *.pkl)
│
├── dashboard/                   # Antarmuka monitoring real-time pada PC
│   ├── app.py                   # Streamlit app atau frontend dashboard
│   └── requirements.txt
│
└── tests/                       # Unit & integration tests
    ├── test_sqa.py              # Uji logika deteksi kualitas sinyal
    ├── test_features.py         # Uji kalkulasi fitur temporal
    └── test_model.py            # Uji konsistensi output model
```

---

## 5. Konvensi Coding & Kontribusi

### Python (Backend & ML)
- Mengikuti pedoman **PEP 8**.
- Wajib menggunakan **Type Hints** (`typing`) pada setiap fungsi publik.
- Menggunakan `snake_case` untuk nama fungsi/variabel, dan `PascalCase` untuk nama *class*.
- Gunakan docstring format Google / Sphinx untuk modul dan fungsi inti.

### C++ / Embedded (ESP32)
- Hindari fungsi `delay()` yang memblokir ekosistem; gunakan penjadwalan non-blocking berbasis `millis()`.
- Pisahkan logika pembacaan sensor, tampilan OLED, dan transmisi jaringan ke dalam modul/fungsi terpisah.
- Gunakan `snake_case` untuk variabel dan fungsi, `kCamelCase` atau `UPPER_CASE` untuk konstanta pin/konfigurasi.

### Git & Format Commit
- Format commit mengacu pada **Conventional Commits**:
  - `feat: <deskripsi>`: Fitur baru
  - `fix: <deskripsi>`: Perbaikan bug
  - `docs: <deskripsi>`: Perubahan dokumentasi
  - `refactor: <deskripsi>`: Refaktorisasi kode tanpa merubah fungsionalitas
  - `test: <deskripsi>`: Penambahan atau perbaikan unit test

---

## 6. Batasan & Otoritas Agent (Agent Boundaries)

### Yang Boleh Dilakukan Agent Secara Otomatis:
1. Membuat dan memperbarui file dokumentasi (`*.md`), diagram Mermaid, dan spesifikasi.
2. Menulis kode sumber modular di folder `backend/`, `ml_pipeline/`, `dashboard/`, dan `tests/`.
3. Menulis kerangka kode firmware di folder `firmware/`.
4. Menjalankan unit test lokal (pytest), linter, dan script preprocessing dataset.

### Yang Harus Meminta Konfirmasi Pengguna Dahulu:
1. Menjalankan perintah kompilasi/flashing firmware ke port USB fisik mikrokontroler.
2. Mengubah skema kontrak payload data sensor antara ESP32 dan backend.
3. Menghapus file dataset, file model terlatih, atau dependensi sistem.
4. Membuat perubahan signifikan pada arsitektur atau pemilihan protokol komunikasi.

### Larangan Mutlak:
1. **Dilarang keras melakukan commit atau mengekspos kredensial** (SSID Wi-Fi, password, token API) ke dalam repository Git. Semua kredensial wajib menggunakan file `.env` atau template `config.h.example`.
2. **Dilarang mematikan atau membypass validasi Signal Quality Assessment (SQA)** pada jalur inferensi tanpa peringatan eksplisit.
3. **Dilarang menambahkan label diagnosis penyakit medis** pada sistem (misal: "Anda terdeteksi Bradikardia/Hipoksia"). Status hanya boleh: `NORMAL`, `POLA MENYIMPANG / ANOMALI`, atau `KUALITAS SINYAL RENDAH / CEK SENSOR`.
