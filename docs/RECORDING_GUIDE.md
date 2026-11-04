# Panduan Perekaman Sesi Telemetri IoT (Dataset Assignment 2 & DL v2)

Panduan operasional praktis untuk merekam sesi fisiologis dari hardware (ESP32 + MAX30102 + DS18B20 + DHT22) ke backend FastAPI secara terisolasi per sesi tanpa mencemari dataset produksi.

---

## 1. Menjalankan Backend per Sesi Perekaman

Setiap sesi perekaman **wajib** menggunakan direktori penyimpanan CSV (`CSV_DIR`) terpisah agar data tidak saling menimpa atau tercampur. Gunakan mode `DECISION_MODE=RULE` agar aktuator (LED/Buzzer) bekerja stabil berbasis rule baku saat merekam data baseline untuk pelatihan deep learning (`MODEL_KIND=dl`).

### Sintaks Eksekusi Backend (Linux / macOS / Git Bash):

```bash
# Sesi 1 (misalnya: Sesi Istirahat / Duduk Tenang)
CSV_DIR=./data_sessions/session_001 DECISION_MODE=RULE MODEL_KIND=dl uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

> **Catatan:** Backend akan otomatis membuat folder `data_sessions/session_001/` dan file `readings.csv` di dalamnya. Bila sesi selesai, hentikan server dengan `Ctrl+C`. Untuk sesi berikutnya, ganti path ke `session_002`, `session_003`, dst.

---

## 2. Menandai Skenario Perekaman (`tag_scenario.py`)

Setelah backend berjalan dan device terhubung ke MQTT, tandai konteks aktivitas secara dinamis menggunakan CLI `tag_scenario.py` di terminal terpisah.

### Perintah Penandaan:
```bash
# Menandai skenario aktif dengan catatan konteks
.venv/bin/python scripts/tag_scenario.py REST --note "Kamar tidur, posisi duduk rileks"

# Memverifikasi skenario yang sedang aktif di backend
.venv/bin/python scripts/tag_scenario.py --show
```

### Daftar Kode Skenario yang Valid:
* `REST`: Duduk diam, rileks, bernapas normal (baseline utama).
* `TYPING`: Bekerja/mengetik di meja komputer (gerakan tangan minimal).
* `LIGHT_ACTIVITY`: Berdiri, meregangkan tubuh, atau berjalan perlahan di dalam ruangan.
* `MOTION`: Gerakan tangan disengaja (uji ketahanan artefak gerak).
* `FINGER_SHIFT`: Menggeser letak sensor pada jari tanpa melepasnya.
* `SENSOR_OFF`: Sensor dilepas dari jari (hanya jika memang menguji transisi lepas).
* `OTHER`: Kondisi khusus lainnya.

---

## 3. Rekomendasi Daftar Sesi Perekaman

Untuk memenuhi syarat partisi `ml_eval` (Train, Validation, Test), diperlukan **minimal 3 sesi terpisah** dengan durasi memadai:

| Sesi | Target Durasi | Label Skenario | Kondisi Lingkungan & Subjek | Tujuan Pelatihan |
| :--- | :--- | :--- | :--- | :--- |
| **Sesi 1** (`session_001`) | 10 – 15 menit (~600–900 sampel) | `REST` | Ruangan A (suhu kamar sejuk ~24–26 °C), duduk tenang | Baseline normal subjek (Train partition) |
| **Sesi 2** (`session_002`) | 10 – 15 menit (~600–900 sampel) | `TYPING` / `LIGHT_ACTIVITY` | Ruangan A, aktivitas mengetik / membaca di meja | Variabilitas denyut wajar (Train / Val) |
| **Sesi 3** (`session_003`) | 10 – 15 menit (~600–900 sampel) | `REST` atau `LIGHT_ACTIVITY` | Ruangan B atau waktu berbeda (variasi suhu/kelembapan DHT) | Evaluasi generalisasi lingkungan (Test partition) |
| **Sesi 4 (Opsional)** | 5 – 10 menit | `MOTION` / `FINGER_SHIFT` | Gerakan terencana atau penyesuaian sensor | Uji ketahanan terhadap false alarm |

> [!IMPORTANT]
> **Pemisahan Anomali Buatan:** Jangan mencampur pelepasan jari acak atau anomali buatan ke dalam Sesi 1–3 yang dialokasikan sebagai data normal untuk melatih Unsupervised Autoencoder (`dl_v2`). Skenario pelepasan jari (`SENSOR_OFF`) hanya boleh dilakukan jika sesi tersebut memang khusus direkam untuk uji robustness.

---

## 4. Verifikasi Kualitas Data Setelah Tiap Sesi

Segera setelah sesi dihentikan (`Ctrl+C`), periksa integritas dan statistik data sebelum melanjutkan ke sesi berikutnya:

### A. Cek Kualitas Sesi & Kelayakan `ml_eval`:
```bash
.venv/bin/python scripts/check_recording.py -i data_sessions/session_001/
```
Output akan mengonfirmasi:
* Jumlah baris valid (`finger=true`, nilai sensor $>0$).
* Persentase nilai DHT22 null (`hum`, `room_temp`).
* Deteksi nilai macet (*stuck sensor* $> 30$ detik berturut-turut).
* Indikasi data simulator vs hardware asli.
* Status kelayakan terhadap syarat `ml_eval` (panjang minimal $\ge 300$ sampel).

### B. Cek Distribusi Fisiologis & Sugesti Baseline:
```bash
.venv/bin/python scripts/baseline_stats.py -i data_sessions/session_001/readings.csv
```
Pastikan:
* `heart_rate`: Rentang normal istirahat (55 – 100 bpm), standar deviasi wajar ($> 1.0$).
* `spo2`: Rentang 95% – 100%.
* `temp` (DS18B20): Rentang kulit yang stabil (~31 – 35 °C).
* `room_temp` & `hum`: Terisi stabil dari sensor DHT22.
