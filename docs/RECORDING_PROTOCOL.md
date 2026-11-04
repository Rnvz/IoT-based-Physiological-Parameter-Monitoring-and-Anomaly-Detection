# Protokol Perekaman Telemetri Fisiologis & Lingkungan (DRAFT v2)

> **STATUS DOKUMEN: DRAFT UNTUK PERSETUJUAN PENGGUNA**  
> *Dokumen ini merupakan panduan protokol pengumpulan data yang dirancang sebelum perekaman berlangsung. Dokumen ini TIDAK memuat klaim hasil eksperimen final.*

---

## 1. Pernyataan Ruang Lingkup & Etika

1. **Subjek Tunggal (*Single-Subject Baseline*):** Seluruh data telemetri fisiologis berasal dari 1 subjek sukarelawan tunggal ($n=1$) dalam kondisi sadar penuh dan sehat secara mandiri.
2. **Karakter Label Non-Klinis:** Kategori operasional perangkat telemetri IoT dan variasi aktivitas fisik **BUKAN** diagnosis medis atau klinis.
3. **Anomali Injeksi:** Anomali fisiologis kritis (seperti desaturasi oksigen SpO2 drastis atau takikardia ekstrem) tidak direplikasi secara fisik pada tubuh manusia, melainkan dievaluasi secara terpisah melalui injeksi sinyal sintetis terstandarisasi ([ml_eval/inject.py](file:///home/ywenanta/Projects/IoT/ml_eval/inject.py)) yang dilaporkan dengan label eksplisit: `SINTETIS / INJEKSI`.
4. **Prinsip Keselamatan Mutlak (Safety First):** **DILARANG KERAS** dan **TIDAK BOLEH** menahan napas (apnea buatan), hiperventilasi, melakukan manuver Valsalva, latihan fisik berat, membatasi sirkulasi darah (turniket/ikatan jari), atau melakukan tindakan fisik apa pun yang sengaja dilakukan untuk menurunkan SpO2.

---

## 2. Pemetaan Skenario Baku API ke Label

> **STATUS PEMETAAN: DRAFT, MENUNGGU PERSETUJUAN PENGGUNA**  
> *Sesuai kontrak skema backend [backend/app/api/session.py](file:///home/ywenanta/Projects/IoT/backend/app/api/session.py), endpoint `/session/scenario` hanya menerima 7 enum nilai `ScenarioType` (`REST`, `TYPING`, `LIGHT_ACTIVITY`, `MOTION`, `FINGER_SHIFT`, `SENSOR_OFF`, `OTHER`). Seluruh skenario di luar skenario dasar menggunakan `OTHER` dengan parameter `note`.*

Tabel pemetaan taksonomi label draft:

| Nilai `scenario` (API) | Parameter `note` | Deskripsi Prosedur Fisik | Kategori Label Taksonomi |
| :--- | :--- | :--- | :--- |
| `REST` | *Kosong / deskripsi ruangan* | Duduk tenang di kursi, bernapas santai dan rileks. | **Normal** |
| `TYPING` | *Kosong / deskripsi meja* | Duduk di meja kerja sambil mengetik komputer atau membaca. | **Normal** |
| `LIGHT_ACTIVITY` | *Kosong / aktivitas* | Berdiri, meregangkan tubuh, atau berjalan santai di ruangan. | **Normal** |
| `OTHER` | `"FAN_AC"` | Duduk di dekat hembusan kipas angin atau perubahan AC ruangan. | **Normal** (Variasi Mikroklimat) |
| `MOTION` | *Opsional* | Gerakan tangan terencana saat memakai sensor. | **Positif Anomali Pola Data** |
| `FINGER_SHIFT` | *Opsional* | Menggeser sedikit letak sensor MAX30102 pada jari tanpa melepas. | **Positif Anomali Pola Data** |
| `OTHER` | `"DHT_UNPLUGGED"` | Mencabut kabel data DHT22 selama sesi berjalan. | **Positif Anomali Pola Data** (Sensor Fallback) |
| `OTHER` | `"WALK_STAIRS"` | Berjalan santai dan naik-turun tangga perlahan (1–2 menit). | **Dilaporkan Terpisah, Bukan Anomali** |
| `OTHER` | `"COLD_HANDS"` | Tangan dingin sebelum sensor dipasang. | **Dilaporkan Terpisah, Bukan Anomali** |
| `SENSOR_OFF` | *Opsional* | Sensor MAX30102 dilepas dari jari (`finger=False`). | **Dikecualikan (Tidak Sampai ke Model)** |

> [!IMPORTANT]
> **Catatan Teknis Gate Input Model:**  
> Skenario `SENSOR_OFF` (`finger=False`) serta baris data dengan nilai sensor vital $0.0$ **TIDAK PERNAH sampai ke model inferensi ML/DL** (langsung diabaikan oleh gate validitas `finger == True and hr > 0 and spo2 > 0 and temp > 0`). Oleh karena itu, baris-baris tersebut **BUKAN merupakan kelas positif anomali**, melainkan data idle/artefak yang secara otomatis dibuang sebelum evaluasi skor anomali.

---

## 3. Cara Pemisahan Sesi & Prosedur Antar-Sesi

> [!WARNING]
> **Temuan Kritis Arsitektur Penyimpanan Telemetri Idle:**  
> Kode backend pada [`backend/app/mqtt/consumer.py` baris 125–139](file:///home/ywenanta/Projects/IoT/backend/app/mqtt/consumer.py#L125-L139) menunjukkan bahwa pemanggilan `await self.repo.save_reading(reading)` dieksekusi **SEBELUM** evaluasi kondisi jari pada baris 149 (`if not telemetry.finger`).  
> Artinya, saat perangkat dalam kondisi menyala tetapi idle (jari dilepas, `finger=False`), ESP32 tetap mem-publish paket telemetri 1 Hz dan **backend terus menyimpan baris `finger=False` setiap detik ke dalam `readings.csv`**.  
> **Konsekuensi:**  
> 1. Tidak akan pernah terjadi jeda waktu $> 600\text{ s}$ di `readings.csv` selama perangkat menyala dan terhubung ke MQTT broker.
> 2. Membiarkan backend dan perangkat terus menyala antar-sesi akan membanjiri `readings.csv` dengan baris `finger=False`, yang langsung menurunkan rasio `finger_true_pct` ke bawah batas minimum $95.0\%$, menyebabkan sesi **DITOLAK (`REJECT`)** oleh validator.
> 3. Oleh karena itu, **DILARANG MENGANDALKAN JEDA 600 DETIK** untuk pemisahan sesi pada file tunggal!

### Prosedur Wajib Antar-Sesi (Isolasi Fisik):
1. **Satu Direktori Terpisah per Sesi (`CSV_DIR` Baru):**
   Setiap sesi perekaman wajib dijalankan dengan variabel lingkungan `CSV_DIR` yang baru dan unik (contoh: `CSV_DIR=./data_sessions/session_001_rest_pagi`).
2. **Penghentian Backend Segera Setelah Sesi Selesai:**
   Segera setelah target durasi sesi tercapai, matikan server backend (`Ctrl+C` pada terminal uvicorn) atau cabut daya perangkat ESP32 agar transmisi idle `finger=False` tidak tertulis ke file sesi.
3. **Validasi Mandiri Sebelum Sesi Berikutnya:**
   Jalankan `scripts/validate_recording.py` pada direktori sesi tersebut. Perekaman sesi berikutnya hanya boleh dimulai jika validator menghasilkan status `ACCEPT`.

---

## 4. Rencana Sesi, Sebaran Hari, dan Pembagian Peran

Perekaman dilakukan minimal sebanyak **6 sesi terpisah** yang disebar dalam minimal 2–3 hari dan jam berbeda:

| ID Sesi Terencana | Target Durasi | Target Sampel (1 Hz) | Kondisi Waktu & Ruangan | Skenario API + Note | Alokasi Partisi |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `session_001_rest_pagi` | 15 menit | ~900 baris | Pagi (~24 °C), Kamar Tidur | `REST` | **Train** |
| `session_002_typing_siang` | 15 menit | ~900 baris | Siang (~28 °C), Meja Kerja | `TYPING` | **Train** |
| `session_003_rest_malam` | 15 menit | ~900 baris | Malam (~26 °C), Ruang Santai | `REST` | **Train** |
| `session_004_env_ac_siang` | 15 menit | ~900 baris | Siang ber-AC (~22 °C), Ruangan B | `OTHER` (`"FAN_AC"`) | **Validation** |
| `session_005_stairs_sore` | 12 menit | ~720 baris | Sore (~27 °C), Koridor & Tangga | `OTHER` (`"WALK_STAIRS"`) | **Validation** |
| `session_006_cold_test` | 15 menit | ~900 baris | Bebas, Ruang Uji | `OTHER` (`"COLD_HANDS"`) | **Test (Beku)** |
| `session_007_robustness_test`| 10 menit | ~600 baris | Ruang Uji | `FINGER_SHIFT` & `DHT_UNPLUGGED` | **Test (Beku)** |

*Aturan Partisi Beku:*
* **Train Partition (Sesi 1, 2, 3):** Khusus data normal baseline untuk melatih model *unsupervised* (Isolation Forest, LOF, OCSVM, Autoencoder).
* **Validation Partition (Sesi 4, 5):** Khusus untuk pemilihan varian fitur (S3 vs S5) dan kalibrasi ambang batas $p_{95}$ dan $p_{99}$.
* **Test Partition (Sesi 6, 7):** Dibekukan sepenuhnya dan **hanya dievaluasi satu kali** setelah dokumen pra-registrasi dikomit ke Git.

---

## 5. Konfigurasi Fitur S3 dan S5

Daftar kolom fitur persis yang digunakan dalam pipeline evaluasi:
* **Varian `S3` (Fisiologis Murni):**
  Kolom persis: `hr`, `spo2`, `temp` (3 dimensi).
* **Varian `S5` (Fisiologis + Lingkungan):**
  Kolom persis: `hr`, `spo2`, `temp`, `room_temp`, `hum` (5 dimensi, yaitu `S3` + `room_temp` + `hum`).
  **TANPA kolom selisih `temp - room_temp`**.

Pemilihan antara varian `S3` dan `S5` dilakukan sepenuhnya pada set validasi berdasarkan PR-AUC anomali injeksi dan stabilitas FAR pada sesi perubahan suhu ruangan.

---

## 6. Definisi Operasional: Uji Ketahanan Fluktuasi Suhu Ruang

* **Kondisi Eksperimen Operasional:**
  * Subjek melakukan sesi istirahat tenang (`REST`) pada dua kondisi lingkungan dengan perbedaan suhu terukur:
    1. Ruang Hangat/Biasa: `room_temp` berada pada rentang $27.0 - 30.0$ °C.
    2. Ruang Sejuk/AC: `room_temp` berada pada rentang $21.0 - 24.0$ °C (selisih termal $\Delta \text{room\_temp} \ge 4.0$ °C).
* **Kriteria Keberhasilan:**
  * Model dinyatakan **tahan fluktuasi suhu ruang** jika pada sesi istirahat sejuk (`session_004`), tingkat False Alarm Rate (FAR) **tidak meningkat melampaui target uji operasional** ($< 2.0$ alarm/jam), membuktikan bahwa penurunan suhu ruangan tidak disalahartikan sebagai anomali fisiologis tubuh (seperti hipotermia).

---

## 7. Metrik False Alarm Rate (FAR) & Catatan Resolusi

* **Lingkup Penghitungan:** FAR dihitung **HANYA pada sesi test jinak** (*benign test session*, misal sesi istirahat/mengetik normal pada Test set).
* **Rumus Perhitungan:**
  $$\text{FAR} = \frac{\text{Jumlah Event Alarm Palsu (Severity } \ge 1\text{)}}{\text{Total Durasi Sesi Jinak (Jam)}}$$
* **Catatan Keterbatasan Resolusi Pengukuran:**
  * Jika total durasi sesi uji jinak adalah 30 menit ($0.5$ jam), maka resolusi terkecil adalah $1 / 0.5 = 2.0$ alarm/jam (terjadinya 1 event saja langsung bernilai 2.0 alarm/jam).
  * Jika total durasi adalah 1 jam, resolusinya adalah 1.0 alarm/jam.
  * Oleh karena itu, kriteria **$\text{FAR} < 2.0$ alarm/jam** dituliskan sebagai **target uji operasional dengan catatan resolusi pengukuran**, bukan klaim bukti sebelum data direkam.

---

## 8. Prosedur Operasional Standar (SOP) Sebelum & Saat Rekaman

1. **Pencadangan Data Lama:**
   * Sebelum memulai perekaman baru, backup file data lama yang ada:
     ```bash
     cp data/readings.csv data/readings.csv.bak_$(date +%Y%m%d_%H%M%S)
     ```
2. **Isolasi Simulator:**
   * Pastikan skrip simulator dimatikan sepenuhnya:
     ```bash
     pkill -f simulate_device.py
     ```
3. **Konfigurasi Jalur Komunikasi:**
   * Pastikan broker mengarah ke `broker.hivemq.com:1883`.
   * Nilai sementara `TOPIC_PREFIX` dan `API_KEY` pada `backend/.env` **tidak boleh dikomit ke Git**.
4. **Penyimpanan Terisolasi:**
   * Jalankan backend dengan direktori terpisah per sesi:
     ```bash
     CSV_DIR=./data_sessions/session_001_rest_pagi DECISION_MODE=RULE MODEL_KIND=dl uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
     ```
5. **Penandaan Skenario:**
   * Kirim tag skenario segera setelah perekaman dimulai:
     ```bash
     .venv/bin/python scripts/tag_scenario.py REST --note "Kamar tidur pagi hari"
     ```
6. **Validasi Pasca-Sesi:**
   * Begitu sesi selesai (server dihentikan), jalankan validator:
     ```bash
     .venv/bin/python scripts/validate_recording.py -i ./data_sessions/session_001_rest_pagi/ --expected-device-id esp32-01 --scenario-label REST
     ```
   * Jika validator menghasilkan status `REJECT`, sesi wajib diulang sebelum berpindah ke skenario berikutnya.

---

## 9. Lembar Catatan Sesi (Session Log Sheet) & Isolasi Folder

> [!IMPORTANT]
> **Rasional Arsitektur Metadata (`note` vs `/export/dataset.csv`):**  
> Audit kode menunjukkan bahwa endpoint `/session/scenario` menyimpan parameter `note` ke dalam `scenario_events.csv`, namun fungsi `export_dataset()` pada backend **TIDAK mengikutsertakan kolom `note` ke dalam `/export/dataset.csv`** (hanya mengekspor string enum `scenario`).  
> Oleh karena itu, pelacakan konteks fisik dan variasi mikroklimat **wajib mengandalkan isolasi direktori dan lembar catatan sesi**, dengan aturan:
> 1. **Satu Skenario per Sesi:** Satu sesi perekaman hanya menjalankan satu jenis skenario tunggal.
> 2. **Satu `CSV_DIR` per Sesi:** Setiap sesi disimpan ke direktori terisolasi baru.
> 3. **Nama Folder Memuat Skenario:** Format folder wajib `session_<XXX>_<skenario>_<kondisi>/` (mis. `session_001_rest_pagi/`).
> 4. **Pengisian Lembar Catatan Sesi:** Setiap sesi dicatat secara manual ke dalam lembar catatan sesi di bawah ini.

### Template Lembar Catatan Sesi:

| ID Sesi / Direktori | Skenario API | Tanggal & Jam (WIB) | Target Durasi | Suhu Ruang (°C) | Kelembapan (%) | Catatan Kondisi Riil / Observasi Fisik | Status Validator |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `session_001_rest_pagi` | `REST` | *Contoh: 2026-11-05 07:15* | 15 menit | ~24.5 | ~58 | Duduk santai di kamar tidur, bernapas tenang | `ACCEPT` |
| `session_002_typing_siang` | `TYPING` | *Contoh: 2026-11-05 13:00* | 15 menit | ~28.0 | ~62 | Mengetik laporan di meja kerja | `ACCEPT` |
| `session_003_rest_malam` | `REST` | *Contoh: 2026-11-05 21:00* | 15 menit | ~26.0 | ~60 | Duduk rileks malam hari | `ACCEPT` |
| `session_004_env_ac_siang` | `OTHER` (`FAN_AC`) | *Contoh: 2026-11-06 14:00* | 15 menit | ~22.0 | ~50 | Ruangan ber-AC sejuk, hembusan stabil | `ACCEPT` |
| `session_005_stairs_sore` | `OTHER` (`WALK_STAIRS`) | *Contoh: 2026-11-06 17:00* | 12 menit | ~27.0 | ~65 | Naik turun tangga perlahan (bukan latihan berat) | `ACCEPT` |
| `session_006_cold_test` | `OTHER` (`COLD_HANDS`) | *Contoh: 2026-11-07 10:00* | 15 menit | ~25.0 | ~55 | Tangan dingin sebelum pasang sensor | `ACCEPT` |
| `session_007_robustness_test` | `FINGER_SHIFT` | *Contoh: 2026-11-07 15:00* | 10 menit | ~25.0 | ~55 | Sensor digeser sedikit pada jari tanpa melepas | `ACCEPT` |
