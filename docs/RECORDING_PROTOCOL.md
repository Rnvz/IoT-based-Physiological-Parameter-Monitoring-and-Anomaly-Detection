# Protokol Perekaman Telemetri Fisiologis & Lingkungan (DRAFT v2)

> **STATUS DOKUMEN: DRAFT UNTUK PERSETUJUAN PENGGUNA**  
> *Dokumen ini merupakan panduan protokol pengumpulan data yang dirancang sebelum perekaman berlangsung. Dokumen ini TIDAK memuat klaim hasil eksperimen final.*

---

## 1. Pernyataan Ruang Lingkup & Etika

1. **Subjek Tunggal (*Single-Subject Baseline*):** Seluruh data telemetri fisiologis berasal dari 1 subjek sukarelawan tunggal ($n=1$) dalam kondisi sadar penuh dan sehat secara mandiri.
2. **Karakter Label Non-Klinis:** Label kondisi (`NORMAL`, `ANOMALOUS`, `EXCLUDED`) merupakan kategori operasional perangkat telemetri IoT dan variasi aktivitas fisik, **BUKAN** diagnosis medis atau klinis.
3. **Anomali Injeksi:** Anomali fisiologis kritis (seperti desaturasi oksigen SpO2 drastis atau takikardia ekstrem) tidak direplikasi secara fisik pada tubuh manusia, melainkan dievaluasi secara terpisah melalui injeksi sinyal sintetis terstandarisasi ([ml_eval/inject.py](file:///home/ywenanta/Projects/IoT/ml_eval/inject.py)) yang dilaporkan dengan label eksplisit: `SINTETIS / INJEKSI`.
4. **Prinsip Keselamatan (Safety First):** **DILARANG** melakukan prosedur yang membahayakan tubuh, seperti menahan napas (apnea buatan), membatasi sirkulasi darah (turniket/ikatan jari), hiperventilasi paksa, atau olahraga ekstrem yang memicu pingsan.

---

## 2. Pemetaan Skenario Baku API ke Label (Dibekukan)

Sesuai kontrak skema backend [backend/app/api/session.py](file:///home/ywenanta/Projects/IoT/backend/app/api/session.py), endpoint `/session/scenario` **hanya menerima 7 enum nilai `ScenarioType`**. Seluruh skenario di luar 6 skenario dasar wajib menggunakan nilai `OTHER` dengan parameter `note` berformat string tetap.

Tabel pemetaan berikut dibekukan (*frozen*) sebelum perekaman dimulai:

| Nilai `scenario` (API) | Parameter `note` (Format Tetap) | Deskripsi Prosedur Fisik | Kategori Label Beku |
| :--- | :--- | :--- | :--- |
| `REST` | *Kosong / deskripsi ruangan* | Duduk tenang di kursi, bernapas santai dan rileks. | **NORMAL** |
| `TYPING` | *Kosong / deskripsi meja* | Duduk di meja kerja sambil mengetik komputer atau membaca. | **NORMAL** |
| `LIGHT_ACTIVITY` | *Kosong / aktivitas* | Berdiri, meregangkan tubuh, atau berjalan santai di ruangan. | **NORMAL** |
| `OTHER` | `"WALK_STAIRS"` | Berjalan santai dan naik-turun tangga perlahan (1–2 menit). | **NORMAL** (Aktivitas Fisik Wajar) |
| `OTHER` | `"COLD_HANDS"` | Tangan menyentuh permukaan/gelas air dingin sebelum sensor dipasang. | **NORMAL** (Perturbasi Termal Alami) |
| `OTHER` | `"FAN_AC"` | Duduk di dekat hembusan kipas angin atau AC ruangan. | **NORMAL** (Variasi Mikroklimat) |
| `FINGER_SHIFT` | *Opsional* | Menggeser sedikit letak sensor MAX30102 pada jari tanpa melepas. | **ANOMALI JINAK** (Evaluasi False Alarm) |
| `MOTION` | *Opsional* | Gerakan tangan terencana saat memakai sensor. | **ANOMALI JINAK** (Evaluasi False Alarm) |
| `OTHER` | `"DHT_UNPLUGGED"` | Mencabut kabel data DHT22 selama sesi berjalan. | **UJI FALLBACK S3** (Bukan Medis) |
| `SENSOR_OFF` | *Opsional* | Sensor MAX30102 dilepas dari jari (`finger=False`). | **DIKECUALIKAN** (Dibuang oleh filter) |

---

## 3. Cara Pemisahan Sesi & Integritas Jeda Waktu

Protokol ini menyelaraskan dua batasan temporal:
1. **Syarat Kualitas Sesi Aktif ([scripts/validate_recording.py](file:///home/ywenanta/Projects/IoT/scripts/validate_recording.py)):** Di dalam satu sesi aktif, jeda waktu antar sampel berurutan wajib $\le 3.0$ detik. Adanya celah $> 3.0$ detik mengindikasikan koneksi terputus dan menyebabkan sesi ditolak (`REJECT`).
2. **Segmentasi Sesi Otomatis ([ml_eval/sessions.py](file:///home/ywenanta/Projects/IoT/ml_eval/sessions.py)):** Fungsi `segment_sessions()` otomatis memotong sesi baru jika mendeteksi selisih waktu antar baris $\Delta t > 600.0$ detik (10 menit).

### Aturan Operasional Pemisahan Sesi:
* **Pemisahan Fisik (Direktori Terpisah):** Setiap sesi direkam ke subfolder terpisah via `CSV_DIR` (misal: `data_sessions/session_001_rest/`). Dalam direktori ini, data direkam dari awal hingga akhir tanpa jeda $> 3.0$ s.
* **Pemisahan Temporal (File Gabungan):** Jika sesi-sesi digabungkan ke dalam satu file dataset besar, jeda henti antar sesi **wajib minimal $> 600.0$ detik** (10 menit) dengan transmisi dimatikan (`SENSOR_OFF`), sehingga algoritma segmentasi `ml_eval` secara deterministik menetapkan ID sesi terpisah (`session_001`, `session_002`, dst.) tanpa memicu alarm jeda intra-sesi.

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

## 5. Pemilihan Varian Fitur (S3 vs S5) & Representasi Suhu Relatif

1. **Penetapan Varian Tanpa Asumsi Subjektif:**
   * Tidak ada klaim apriori mengenai "varian utama". Pemilihan antara varian **`S3`** (fisiologis murni: `hr`, `spo2`, `temp`) dan **`S5`** (fisiologis + lingkungan: `hr`, `spo2`, `temp`, suhu lingkungan, kelembapan) dilakukan **sepenuhnya berdasarkan kriteria performa validasi** yang ditetapkan di muka:
     * Rasio False Alarm Rate (FAR) pada sesi validasi ber-AC.
     * PR-AUC pada validasi anomali injeksi.
2. **Representasi Suhu Relatif Ruang (Anti-Multikolinearitas):**
   * Untuk varian lingkungan, **hanya digunakan satu representasi suhu relatif ruang**, yaitu selisih termal:
     $$\text{temp\_sub\_room} = \text{temp} - \text{room\_temp}$$
   * **DILARANG** memasukkan `temp`, `room_temp`, dan `temp_sub_room` secara bersamaan ke dalam vektor input karena ketiga variabel tersebut memiliki ketergantungan linier identik ($X_1 - X_2 = X_3$) yang merusak kestabilan matriks kovariansi dan bobot model.

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
     .venv/bin/python scripts/validate_recording.py -i ./data_sessions/session_001_rest_pagi/ --expected-device-id esp32-01
     ```
   * Jika validator menghasilkan status `REJECT`, sesi wajib diulang sebelum berpindah ke skenario berikutnya.
