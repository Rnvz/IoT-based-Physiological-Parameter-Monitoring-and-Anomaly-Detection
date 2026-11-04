# Protokol Perekaman Telemetri Fisiologis & Lingkungan (DRAFT)

> **STATUS DOKUMEN: DRAFT UNTUK PERSETUJUAN PENGGUNA**  
> *Dokumen ini merupakan panduan protokol pengumpulan data yang dirancang sebelum perekaman berlangsung. Dokumen ini TIDAK memuat klaim hasil eksperimen final.*

---

## 1. Pernyataan Ruang Lingkup & Etika

1. **Subjek Tunggal (*Single-Subject Baseline*):** Seluruh data telemetri yang dikumpulkan berasal dari 1 subjek sukarelawan tunggal ($n=1$) dalam kondisi sadar penuh.
2. **Karakter Label Non-Klinis:** Label kondisi (`NORMAL`, `ANOMALOUS`, `EXCLUDED`) merupakan kategori operasional perangkat telemetri IoT dan variasi aktivitas fisik, **BUKAN** alat diagnosis medis atau klinis.
3. **Anomali Injeksi:** Anomali fisiologis kritis (seperti desaturasi oksigen drastis atau lonjakan aritmia ekstrem) tidak direplikasi secara fisik pada tubuh manusia, melainkan dievaluasi secara terpisah melalui injeksi sinyal sintetis terstandarisasi ([ml_eval/inject.py](file:///home/ywenanta/Projects/IoT/ml_eval/inject.py)) yang dilaporkan dengan label eksplisit: `SINTETIS / INJEKSI`.
4. **Prinsip Keselamatan (Safety First):** **DILARANG** melakukan prosedur yang membahayakan tubuh, seperti menahan napas (apnea buatan), membatasi sirkulasi darah (turniket/ikatan jari), hiperventilasi paksa, atau olahraga ekstrem yang memicu pingsan.

---

## 2. Daftar Skenario Perekaman Aman

Seluruh skenario yang diizinkan merupakan aktivitas harian dengan risiko minimal:

| Kode Skenario | Deskripsi Prosedur | Perubahan Fisiologis / Lingkungan yang Diharapkan | Kategori Label Beku |
| :--- | :--- | :--- | :--- |
| `REST` | Duduk tenang di kursi, bernapas santai dan rileks. | HR stabil (60–85 bpm), SpO2 96–99%, Temp stabil. | **NORMAL** |
| `TYPING` | Duduk di meja kerja sambil mengetik komputer atau membaca. | HR stabil dengan sedikit variasi ringan (65–90 bpm). | **NORMAL** |
| `WALK_STAIRS` | Berjalan santai di ruangan atau naik-turun tangga perlahan (1–2 menit). | Kenaikan bertahap HR (90–120 bpm), kembali normal bertahap. | **NORMAL** (Aktivitas Fisik Wajar) |
| `COLD_HANDS` | Tangan menyentuh permukaan/gelas air dingin sebelum sensor dipasang. | Suhu kulit DS18B20 turun sementara (~28–31 °C), HR stabil. | **NORMAL** (Perturbasi Termal Alami) |
| `FAN_AC` | Duduk di dekat hembusan kipas angin atau AC ruangan. | DHT22 mendeteksi penurunan `room_temp` atau variasi `hum`. | **NORMAL** (Variasi Mikroklimat) |
| `FINGER_SHIFT` | Menggeser sedikit posisi sensor MAX30102 pada jari tanpa melepas. | Potensi artefak optik sementara pada SpO2/HR ($<5$ detik). | **ARTIFAK** (Evaluasi Kekebalan Alarm) |
| `MOTION` | Gerakan tangan santai (melambaikan tangan atau gestur ringan). | Potensi fluktuasi transien sinyal PPG. | **ARTIFAK** (Evaluasi Kekebalan Alarm) |
| `DHT_UNPLUGGED` | Mencabut kabel data DHT22 selama sesi berjalan. | `hum` dan `room_temp` menjadi `null` / `None`. | **UJI FALLBACK S3** (Bukan Anomali Medis) |
| `SENSOR_OFF` | Sensor MAX30102 dilepas dari jari. | Flag `finger = False`. | **DIKECUALIKAN** (Dibuang oleh filter) |

---

## 3. Rencana Sesi, Sebaran Hari, dan Pembagian Peran

Perekaman dilakukan minimal sebanyak **6 sesi terpisah** yang disebar dalam minimal 2–3 hari dan jam berbeda untuk menangkap variabilitas diurnal dan suhu lingkungan:

| ID Sesi Terencana | Target Durasi | Target Sampel (1 Hz) | Kondisi Waktu & Ruangan | Skenario Utama | Alokasi Partisi |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `session_001_rest_pagi` | 15 menit | ~900 baris | Pagi hari (~24 °C), Ruang Tidur | `REST` | **Train** |
| `session_002_typing_siang` | 15 menit | ~900 baris | Siang hari (~28 °C), Meja Kerja | `TYPING` | **Train** |
| `session_003_rest_malam` | 15 menit | ~900 baris | Malam hari (~26 °C), Ruang Santai | `REST` | **Train** |
| `session_004_env_ac_siang` | 15 menit | ~900 baris | Siang hari ber-AC (~22 °C), Ruangan B | `FAN_AC` + `REST` | **Validation** |
| `session_005_stairs_sore` | 12 menit | ~720 baris | Sore hari (~27 °C), Koridor & Tangga | `WALK_STAIRS` + `REST` | **Validation** |
| `session_006_cold_test` | 15 menit | ~900 baris | Waktu bebas, Ruang Uji | `COLD_HANDS` + `REST` | **Test (Beku)** |
| `session_007_robustness_test`| 10 menit | ~600 baris | Ruang Uji | `FINGER_SHIFT` + `DHT_UNPLUGGED` | **Test (Beku)** |

*Aturan Partisi Beku:*
* **Train Partition (Sesi 1, 2, 3):** Khusus data normal baseline untuk melatih model *unsupervised* (Isolation Forest, LOF, OCSVM, Elliptic Envelope, Autoencoder).
* **Validation Partition (Sesi 4, 5):** Khusus untuk pemilihan hiperparameter dan kalibrasi ambang peringatan ($p_{95}$) dan kritis ($p_{99}$).
* **Test Partition (Sesi 6, 7):** Dibekukan sepenuhnya dan **hanya dievaluasi satu kali** setelah pra-registrasi di-commit.

---

## 4. Kriteria Kualitas Sesi (*Session Acceptance Criteria*)

Setiap file rekaman sesi wajib diverifikasi menggunakan [scripts/validate_recording.py](file:///home/ywenanta/Projects/IoT/scripts/validate_recording.py). Sesi dinyatakan **GAGAL/DITOLAK** jika melanggar ambang batas berikut:

1. **Jeda Waktu (*Data Gap*):**
   * Tidak boleh ada celah data $> 3.0$ detik saat sensor aktif menempel.
   * Total baris dengan jeda $> 3.0$ s maksimal $0.0\%$ (koneksi Wi-Fi/MQTT harus kontinu).
2. **Integritas Penempelan Jari (*Finger Contact*):**
   * Persentase baris dengan `finger == True` wajib $\ge 95.0\%$.
3. **Validitas Nilai Sensor Vital:**
   * Nilai pembacaan `hr == 0.0`, `spo2 == 0.0`, atau `temp == 0.0` pada saat `finger == True` maksimal $< 2.0\%$.
4. **Integritas Sensor Lingkungan (DHT22):**
   * Persentase `hum` atau `room_temp` null / missing wajib $0.0\%$ (kecuali sesi yang secara terencana menandai skenario `DHT_UNPLUGGED`).
5. **Panjang Minimum Sesi:**
   * Minimal memiliki $\ge 300$ baris valid kontinu per sesi (~5 menit perekaman efektif) untuk menjamin stabilitas jendela kausal $W_{15}$.
6. **Rentang Fisiologis Subjek:**
   * Heart Rate valid berada pada rentang $45 \le \text{HR} \le 160$ bpm.
   * SpO2 valid berada pada rentang $90 \le \text{SpO2} \le 100\%$.
   * Suhu sensor kulit berada pada rentang $28.0 \le \text{Temp} \le 39.0$ °C.

---

## 5. Prosedur Operasional Standar (SOP) Sebelum & Saat Rekaman

1. **Isolasi Simulator:**
   * Pastikan skrip [scripts/simulate_device.py](file:///home/ywenanta/Projects/IoT/scripts/simulate_device.py) dimatikan total (`pkill -f simulate_device.py`).
2. **Verifikasi Jalur Komunikasi:**
   * Pastikan broker MQTT mengarah ke `broker.hivemq.com:1883`.
   * Pastikan `TOPIC_PREFIX` di `backend/.env` dan firmware ESP32 identik serta unik (tidak bertabrakan dengan pengguna MQTT publik lain).
3. **Penyimpanan Terisolasi:**
   * Jalankan backend dengan direktori terpisah per sesi:
     ```bash
     CSV_DIR=./data_sessions/session_001_rest_pagi DECISION_MODE=RULE MODEL_KIND=dl uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
     ```
4. **Penandaan Skenario:**
   * Segera kirimkan tag skenario aktif begitu perekaman dimulai:
     ```bash
     .venv/bin/python scripts/tag_scenario.py REST --note "Duduk rileks kamar tidur"
     ```
5. **Validasi Pasca-Sesi:**
   * Begitu sesi selesai (server dihentikan), jalankan validator:
     ```bash
     .venv/bin/python scripts/validate_recording.py -i ./data_sessions/session_001_rest_pagi/
     ```
   * Jika status validator menampilkan `REJECTED`, sesi harus diulang sebelum berpindah ke kondisi berikutnya.
