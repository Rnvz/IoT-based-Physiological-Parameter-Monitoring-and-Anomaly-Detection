# PHYSIO MONITOR: NOTE UNTUK MEMBANGUN BACKEND, ML, DAN DATABASE

Tempel seluruh isi file ini di awal sesi baru. Pertahankan keputusan di bawah ini kecuali ada instruksi baru dari pengguna. Pengguna berbahasa Indonesia. Jangan mengulang diskusi dasar; kerjakan bertahap, berikan kode lengkap, dan sebutkan file mana yang dibuat atau diubah. Tanyakan paling banyak satu pertanyaan, hanya jika benar-benar menghambat.

Catatan status: firmware dan dashboard sudah dibuat tetapi **belum di-compile dan belum diuji** di perangkat. Backend, ML, dan database **belum ada** (ini yang akan dibangun).

---

## 1. Tujuan dan arsitektur final

Sistem pemantauan dan deteksi anomali parameter fisiologis berbasis IoT (proposal skripsi S1): "Sistem Pemantauan dan Deteksi Anomali Parameter Fisiologis Berbasis IoT Menggunakan Analisis Sinyal PPG, Suhu, dan Machine Learning". Fokus: software engineering, kualitas arsitektur, maintainability, pengujian, employability.

```
ESP32 -> MQTT (HiveMQ publik) -> Python (Paho + FastAPI + ML) -> Supabase PostgreSQL + CSV lokal -> Dashboard (HTML/CSS/JS + Chart.js)
```

- Dashboard mengakses backend lewat REST API, bukan langsung ke database. (Dashboard juga punya mode langsung MQTT tanpa backend; itu sudah selesai dan tidak perlu diubah.)
- InfluxDB dan Docker **tidak dipakai** (keputusan lama sudah digantikan). Docker tidak wajib.
- Keputusan anomali dibuat oleh **model ML di backend**, bukan ambang di firmware.
- Penyimpanan data mentah **tidak boleh bergantung** pada keberhasilan model. Jika model error, data sensor tetap tersimpan.
- Kembangkan dan uji lokal dahulu (Python 3.11 + venv). Resource harus gratis, tanpa trial dan tanpa risiko biaya tak terduga. Backend lokal diperbolehkan. Cloudflare Tunnel boleh untuk demo; jangan menjanjikan layanan gratis selalu aktif 24/7.

## 2. Tech stack backend

Python 3.11, FastAPI, Uvicorn, Pydantic, Paho-MQTT (gunakan `mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)` pada Paho 2.x), pandas, NumPy, scikit-learn, Joblib, Matplotlib, Seaborn, Jupyter Notebook, Pytest, python-dotenv. Klien database: supabase-py atau psycopg (pilih yang paling sederhana dan jelaskan alasannya).

Struktur folder:

```
physio-monitor/
├── firmware/physio_monitor.ino          (sudah ada)
├── frontend/                            (sudah ada)
├── backend/
│   ├── app/
│   │   ├── main.py                      (FastAPI, lifespan menjalankan MQTT consumer)
│   │   ├── config.py                    (baca .env)
│   │   ├── api/                         (router: latest, history, predictions, device, actuator)
│   │   ├── mqtt/                        (consumer.py, publisher.py)
│   │   ├── ml/                          (inference.py, severity.py)
│   │   ├── database/
│   │   │   ├── interfaces.py            (kontrak repository)
│   │   │   ├── supabase_repository.py
│   │   │   ├── csv_repository.py
│   │   │   └── factory.py
│   │   └── schemas/                     (Pydantic)
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── ml/
│   ├── notebooks/
│   ├── training/
│   ├── evaluation/
│   └── artifacts/                       (model .joblib)
├── docs/
├── .gitignore
└── README.md
```

## 3. Kontrak MQTT (WAJIB konsisten dengan firmware)

Broker: `broker.hivemq.com`, port 1883 (TCP, tanpa TLS). Dashboard memakai `wss://broker.hivemq.com:8884/mqtt`. HiveMQ publik hanya untuk prototipe; siapa pun yang tahu prefix dapat membaca dan mengirim perintah. Semua topik dibentuk dari satu `TOPIC_PREFIX` (format `physiomonitor/<id-acak-panjang>`), dibaca dari `.env`, dan harus **persis sama** dengan di firmware dan `frontend/js/config.js`. Nilai prefix diisi pengguna.

| Topik | Arah | Isi |
|---|---|---|
| `<prefix>/data` | ESP32 ke backend | Data sensor (tiap sekitar 1 detik) |
| `<prefix>/result` | backend ke ESP32 | Hasil model |
| `<prefix>/cmd` | backend ke ESP32 | Perintah lampu dan tes buzzer |
| `<prefix>/actuator` | ESP32 ke backend | Konfirmasi status aktuator |

**Payload `/data` (tidak boleh diubah):**
```json
{"hr":75.0,"spo2":98.0,"temp":36.5,"hum":62.3,"room_temp":29.1,"finger":true,"status":"NORMAL"}
```
- `hr`, `spo2`, `temp` bernilai `0.0` jika belum tersedia. **0.0 bukan nilai fisiologis nyata.**
- `hum` dan `room_temp` bernilai `null` jika DHT22 gagal. Jangan diganti nol.
- `finger:false` berarti HR dan SpO2 tidak valid: jangan lakukan inferensi dan jangan simpan seolah nilai valid.
- Field `status` adalah status lokal firmware (NORMAL, ANOMALI, MENDETEKSI, MENUNGGU SENSOR). **Bukan keputusan model**; simpan sebagai `fw_status`, jangan campur dengan hasil model.
- `temp` adalah suhu **permukaan kulit** (DS18B20), bukan suhu inti tubuh.

**Payload `/result` (backend publish; firmware hanya membaca `severity`):**
```json
{"label":"NORMAL","severity":0,"score":0.12,"model_version":"if_v1"}
```
- `severity`: 0 = normal, 1 = batas/waspada, 2 = anomali. Firmware AUTO: 0 hijau, 1 kuning, 2 atau lebih merah.
- Publish **hanya untuk data valid** (finger true dan hr, spo2, temp valid). Publish ulang setiap 1 sampai 2 detik selama ada data valid: firmware menganggap hasil lebih tua dari **10 detik** sebagai tidak ada (lampu kuning "NO_RESULT").
- Kondisi NO_FINGER ditangani lokal oleh firmware (lampu biru redup); backend tidak perlu mengirim hasil.

**Payload `/cmd` (backend publish):**
```json
{"mode":"AUTO","seq":101}
{"mode":"MANUAL","color":"RED","brightness":60,"seq":102}
{"buzzer_test":true,"seq":103}
```
- `color` hanya: OFF, GREEN, RED, BLUE, YELLOW, WHITE. `brightness` 0 sampai 100. Backend harus memvalidasi.
- `seq` harus **integer 32-bit** (maks 2147483647) dan berbeda dari perintah sebelumnya (firmware membuang seq yang sama dengan sebelumnya). **Jangan pakai timestamp milidetik.** Pakai penghitung yang disimpan di database dan dinaikkan satu per satu, dimulai dari titik awal acak.
- Firmware: MANUAL kembali ke AUTO sendiri setelah 5 menit; tes buzzer mati sendiri setelah 30 detik. Tes buzzer hanya untuk memastikan buzzer berfungsi (toggle).

**Payload `/actuator` (ESP32 publish saat berubah dan tiap 5 detik):**
```json
{"mode":"AUTO","led":{"color":"GREEN","brightness":40},"buzzer":{"test":false},"source":"ML_RESULT","ack_seq":103,"manual_left_s":0}
```
- Yang dilaporkan adalah perintah yang dijalankan firmware, bukan hasil pengukuran kecerahan atau volume. Buzzer aktif: tidak ada volume.
- `source`: ML_RESULT, LOCAL_NO_FINGER, NO_RESULT, MANUAL.
- Simpan setiap perintah dan konfirmasi (`ack_seq`) ke database sebagai riwayat.

**Keputusan buzzer belum dikunci:** buzzer saat ini masih mengikuti status lokal firmware (ambang). Jangan mengubah perilaku buzzer tanpa konfirmasi pengguna.

## 4. Kontrak REST API (dipanggil dashboard mode API)

Dashboard sudah memanggil endpoint ini (`frontend/js/app.js`). Format JSON harus persis.

| Endpoint | Respons / body |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /latest` | `{"reading":{ts,hr,spo2,temp,hum,room_temp,finger},"prediction":{ts,label,severity,score}\|null}` |
| `GET /history?minutes=30` | `{"items":[{ts,hr,spo2,temp,hum,room_temp,finger}]}` (hum dan room_temp ditambahkan untuk grafik suhu ruangan di dashboard) |
| `GET /predictions?limit=100` | `{"items":[{ts,label,severity,score,model_version}]}` |
| `GET /device/status` | `{"online":bool,"age_s":number}` (online jika ada `/data` dalam 15 detik terakhir) |
| `GET /actuator/status` | `{"mode","led":{"color","brightness"},"buzzer":{"test":bool},"source","manual_left_s"}` |
| `POST /actuator/mode` | body `{"mode":"AUTO"}` |
| `POST /actuator/led` | body `{"color","brightness"}`, otomatis beralih ke MANUAL |
| `POST /actuator/buzzer` | body `{"test":true}` |

- Endpoint `POST` wajib autentikasi header `X-API-Key` (nilai dari `.env`), validasi isi, dan batas laju. CORS hanya untuk origin dashboard yang sah (dari `.env`).
- Timestamp: ISO 8601 UTC. ESP32 belum punya jam nyata, jadi backend memberi timestamp saat pesan diterima.
- `label` yang dikenali dashboard: NORMAL, ANOMALY, NO_FINGER, NO_DATA, SENSOR_ERROR.

## 5. Database: Supabase PostgreSQL + CSV lokal

- Akses database **hanya lewat Repository Pattern**. Kontrak: `save_reading()`, `get_latest()`, `get_history()`, `save_prediction()`, `get_predictions()`, `save_command()`, `save_actuator_status()`, `get_actuator_status()`. Implementasi: `supabase_repository.py` dan `csv_repository.py`; dipilih lewat `factory.py` dari `.env`.
- **Satu sumber kebenaran:** Supabase sebagai utama, CSV sebagai ekspor/pencatatan dataset. Tangani duplikasi (kunci unik `device_id` + `ts`).
- Kunci service Supabase hanya di backend (`.env`), **tidak pernah** di firmware atau frontend. Jangan commit `.env`.
- Setahu saya project Supabase gratis bisa di-pause jika lama tidak aktif dan kuotanya terbatas: **periksa kebijakan terbaru** sebelum demo dan siapkan cadangan (CSV lokal).

Usulan skema (boleh disesuaikan, jelaskan jika berubah):

```sql
create table readings (
  id bigint generated always as identity primary key,
  device_id text not null default 'esp32-01',
  ts timestamptz not null,
  hr real, spo2 real, temp real, hum real, room_temp real,
  finger boolean not null,
  fw_status text,
  unique (device_id, ts)
);

create table predictions (
  id bigint generated always as identity primary key,
  device_id text not null default 'esp32-01',
  ts timestamptz not null,
  reading_id bigint references readings(id),
  label text not null,
  severity smallint not null,
  score real,
  model_version text not null
);

create table actuator_events (
  id bigint generated always as identity primary key,
  ts timestamptz not null default now(),
  kind text not null,          -- 'cmd' atau 'status'
  seq integer,
  payload jsonb not null
);
```

Aturan penyimpanan: simpan data mentah setelah validasi tipe dan rentang. Simpan HR dan SpO2 sebagai `null` (bukan 0) jika tidak valid. Simpan hasil model terpisah dari data mentah. Pertimbangkan batas tulis (misalnya simpan tiap 1 sampai 5 detik) dan retry saat database gagal.

## 6. Machine Learning

- **Deteksi anomali tanpa label**, dilatih **offline** (bukan tiap pesan MQTT). Model utama: **Isolation Forest**. Pembanding: **LOF (`novelty=True`)** dan **One-Class SVM**. Protokol evaluasi dan fitur harus sama untuk ketiganya.
- Fitur awal: `hr`, `spo2`, `temp` (kulit), `room_temp`, `hum`, dan `thermal_gradient = temp - room_temp`. **Bandingkan model dengan dan tanpa fitur lingkungan**; manfaat thermal gradient harus dievaluasi, bukan dianggap otomatis benar. Risiko: model dilatih di satu lingkungan akan menandai anomali saat kondisi ruangan berubah.
- Simpan preprocessing (misalnya scaler) dan model sebagai **satu pipeline** dengan Joblib, beserta `model_version`.
- Konversi skor menjadi severity: 0, 1, 2 memakai ambang pada skor anomali yang dikalibrasi dari data validasi normal (misalnya persentil). Dokumentasikan ambangnya.
- Data tidak valid (NO_FINGER, NO_DATA, SENSOR_ERROR, hr/spo2 sama dengan 0) **tidak boleh** dianggap NORMAL maupun diproses sebagai nilai nol.
- Hasil ML = anomali pola data, **bukan diagnosis medis**.
- **Data:** dataset bawaan (kolom: HR, SpO2, Systolic/Diastolic BP, Body Temperature, Fall Detection, Predicted Disease, Data Accuracy, kolom alert) tampak sintetis. Kolom BP, Fall Detection, dan Blood Pressure Alert **tidak dipakai** karena alat tidak mengukurnya. Dataset tidak punya kelembapan dan suhu ruangan. Karena ada pergeseran distribusi antara dataset dan sensor ESP32, kumpulkan **setidaknya 1.000 rekaman dari alat sendiri** (konsisten, ada timestamp, duplikasi ditangani) dengan keragaman kondisi ruangan, posisi jari, dan waktu.
- **Evaluasi:** karena tanpa label, jelaskan apa yang bisa dan tidak bisa diklaim. Jika label dataset dibuat dari ambang yang sama, evaluasinya tidak independen. Periksa class imbalance, kebocoran data, pemisahan train/test, serta precision, recall, F1, dan confusion matrix bila ada label. Uji dengan skenario artefak aman (jari bergeser, jari dingin, sensor dilepas). **Jangan memancing kondisi berbahaya** pada diri sendiri atau relawan. Jika data berasal dari orang lain, perlu persetujuan dan anonimisasi.
- Kode ML ada di `ml/` (notebook, `training/`, `evaluation/`, `artifacts/`); backend hanya memuat artefak untuk inferensi.

## 7. Konsumer MQTT di backend

- Subscribe `/data` dan `/actuator`; publish `/result` dan `/cmd`. Subscribe diulang pada `on_connect` (langganan hilang saat koneksi putus). Reconnect otomatis, client id unik, QoS 0.
- Alur per pesan `/data`: parse JSON, validasi (Pydantic), beri timestamp, simpan data mentah, lalu (jika valid) inferensi, simpan prediksi, publish `/result`. Kegagalan di langkah model/publish tidak boleh menghentikan penyimpanan data mentah.
- Tangani JSON rusak, field hilang, nilai di luar rentang, pesan duplikat, dan kegagalan database (log + retry). Gunakan `logging`.
- Boleh satu proses (consumer dijalankan lewat `lifespan` FastAPI) atau dua proses; pilih yang paling sederhana.

## 8. Konfigurasi `.env.example`

```
MQTT_BROKER=broker.hivemq.com
MQTT_PORT=1883
TOPIC_PREFIX=physiomonitor/GANTI-ID-UNIK
DEVICE_ID=esp32-01
DB_BACKEND=supabase            # supabase atau csv
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
CSV_DIR=./data
API_KEY=
CORS_ORIGINS=http://localhost:8080
MODEL_PATH=../ml/artifacts/pipeline.joblib
MODEL_VERSION=if_v1
```

## 9. Urutan pengerjaan yang disarankan

1. `config.py`, skema Pydantic, antarmuka repository, `csv_repository.py` (agar bisa jalan tanpa Supabase).
2. Implementasi `supabase_repository.py` + SQL skema.
3. Konsumer MQTT: terima `/data`, validasi, simpan data mentah. Uji dengan data tiruan.
4. Pelatihan dan evaluasi model (notebook), simpan pipeline, lalu modul inferensi dan konversi severity.
5. Publisher `/result` dan `/cmd` (seq 32-bit), subscriber `/actuator`.
6. Endpoint FastAPI sesuai kontrak di bagian 4, autentikasi, dan CORS.
7. Tes Pytest (validasi, repository, severity, API) dan tes integrasi (termasuk data tidak valid dan koneksi gagal).
8. Uji ujung ke ujung dengan ESP32 dan dashboard (ubah `MODE` di `frontend/js/config.js` menjadi `"api"` dan isi `API_BASE`).

## 10. Prinsip pengembangan

- Pisahkan tanggung jawab; konfigurasi lewat environment; jangan commit rahasia (`.env`, token, password WiFi).
- Kode stabil, mudah diuji, dan diberi komentar yang berguna. Jangan mengubah pin ESP32, topik, format payload, atau layout OLED tanpa kebutuhan yang jelas.
- Jangan menyatakan komponen sudah selesai sebelum diuji.

## 11. Keputusan yang masih terbuka

- Perilaku buzzer final (mengikuti hasil model atau tetap ambang lokal).
- Nilai `TOPIC_PREFIX` final dan apakah pindah ke broker privat (autentikasi, TLS).
- Project Supabase (region, kebijakan pause) dan strategi cadangan CSV.
- Target evaluasi final model dan apakah menambah MPU6050 (fall detection) atau NTP di ESP32.
- Deployment akhir (setelah MVP lokal stabil).
