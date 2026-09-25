# REQUIREMENTS: Sistem Pemantauan dan Deteksi Anomali Parameter Fisiologis Berbasis IoT

Dokumen ini memuat spesifikasi kebutuhan fungsional, non-fungsional, batasan teknis, serta asumsi-asumsi perancangan sistem yang diekstrak langsung dari dokumen proposal proyek.

---

## 1. Kebutuhan Fungsional (Functional Requirements)

### 1.1 Akuisisi & Pengolahan Sinyal pada Edge (ESP32)
- **FR-01 (Akuisisi PPG & Oksimetri)**: Sistem harus dapat membaca data mentah PPG (*Photoplethysmography*), estimasi *Heart Rate/Pulse Rate*, dan saturasi oksigen (*SpO₂*) secara kontinu dari sensor MAX30102 via antarmuka I²C.
- **FR-02 (Pengukuran Suhu Kulit)**: Sistem harus dapat membaca suhu permukaan kulit secara berkala dari sensor digital DS18B20 via protokol 1-Wire.
- **FR-03 (Display Lokal OLED)**: Sistem harus menampilkan parameter berikut pada layar OLED I²C di pergelangan tangan:
  - Nilai *Heart Rate* (BPM)
  - Nilai *SpO₂* (%)
  - Nilai Suhu Permukaan Kulit (°C)
  - Label Status Sistem (`NORMAL`, `CEK SENSOR`, atau `ANOMALI`)
  - Apabila kualitas sinyal PPG terdeteksi rendah/buruk, OLED menampilkan `HR: -- BPM`, `SpO2: -- %`, dan status `CEK SENSOR`.
- **FR-04 (Peringatan Lokal Buzzer)**: Sistem harus membunyikan *active buzzer* sebagai alarm peringatan lokal hanya jika kedua kondisi terpenuhi:
  1. *Signal Quality* = `GOOD` (kualitas sinyal memadai), **DAN**
  2. *Model Detection* = `Pola Menyimpang` (*Anomaly detected*).
- **FR-05 (Anti-False Alarm / Persistent Anomaly)**: Sistem lokal harus menerapkan mekanisme *persistent anomaly* (debouncing) di mana buzzer tidak boleh aktif seketika hanya karena satu sampel anomali tunggal, melainkan harus bertahan anomali selama sejumlah *window* pengamatan berturut-turut.
- **FR-06 (Transmisi Nirkabel)**: ESP32 harus membungkus pembacaan sensor dengan informasi *timestamp* dan status pembacaan, lalu mentransmisikannya melalui jaringan Wi-Fi ke server/komputer lokal.
- **FR-07 (Edge Fail-Safe)**: Apabila koneksi Wi-Fi ke backend terputus, ESP32 tetap harus menjalankan pembacaan sensor lokal dan menampilkan nilai pada OLED secara *standalone* (meskipun deteksi ML dari backend tidak tersedia).

### 1.2 Pemrosesan Data & Machine Learning (Backend Python)
- **FR-08 (Signal Quality Assessment - SQA)**: Backend harus melakukan validasi kualitas sinyal PPG sebelum data dimasukkan ke dalam model ML, untuk mendeteksi *motion artifact*, pergeseran sensor, atau hilangnya kontak kulit.
- **FR-09 (Feature Fusion & Temporal Feature Extraction)**: Backend harus mengekstrak dan menggabungkan fitur multivariat secara dinamis berdasarkan data runtun waktu (*time-series*):
  - Parameter instan: *Heart Rate*, *SpO₂*, *Temperature*
  - Parameter temporal delta: *Heart Rate Delta*, *SpO₂ Delta*, *Temperature Delta*
  - Parameter statistik temporal: *Moving Average* (HR, SpO₂, Temp), *Variance* (HR, SpO₂, Temp), dan *Rate of Change*
  - Skor kualitas sinyal (*Signal Quality Score*)
- **FR-10 (Unsupervised Anomaly Detection)**: Backend harus mengimplementasikan algoritma **Isolation Forest** untuk menghitung *anomaly score* dan mengkategorikan kondisi ke dalam:
  - `Pola Normal`: Kombinasi parameter berada dalam sebaran distribusi data normal pelatihan.
  - `Pola Menyimpang`: Kombinasi parameter menunjukkan penyimpangan terhadap distribusi normal dengan kualitas sinyal baik.
  - `Kualitas Sinyal Rendah`: Kualitas sinyal tidak memadai sehingga proses deteksi anomali ditangguhkan/diberi label cek sensor.
- **FR-11 (Baseline Comparison)**: Backend harus menyediakan metode deteksi *baseline* berbasis ambang batas sederhana (*simple threshold*) per parameter independen untuk dibandingkan dengan performa deteksi multivariat Isolation Forest.
- **FR-12 (Strategi Multi-Dataset & Harmonisasi Fitur)**: Pipeline harus mampu mengintegrasikan, membersihkan, menangani *missing values*, dan menyelaraskan kombinasi dataset multi-sumber sesuai peran masing-masing:
  1. **Dataset Inti Real-Time [engrarri21]** ([Kaggle Link](https://www.kaggle.com/datasets/engrarri21/human-vital-signs)):
     - Kolom: `[Time, HR, RESP, SpO2, TEMP, OUTPUT]`
     - Peran: **Wajib (Primary)** — Struktur paling identik dengan proposal; label `OUTPUT` relevan sebagai ground-truth dasar.
  2. **Dataset Sebaran Pola Normal [nasirayub2]** ([Kaggle Link](https://www.kaggle.com/datasets/nasirayub2/human-vital-sign-dataset)):
     - Karakteristik: Volume besar (~38 MB), *timestamp* riil kontinu.
     - Peran: **Wajib (Variasi Normal)** — Memperkaya variasi distribusi data normal pada tahap unsupervised training.
  3. **Dataset Suplemen Numerik [rishanmascarenhas]** ([Kaggle Link](https://www.kaggle.com/datasets/rishanmascarenhas/covid19-temperatureoxygenpulse-rate)):
     - Kolom: `[Oxygen, PulseRate, Temperature, Result]`
     - Peran: **Opsional (Fitur Saja)** — Hanya fitur numeriknya yang diekstrak untuk melatih model unsupervised; label `Result` (status COVID) **tidak digunakan** untuk evaluasi anomali karena tidak mewakili konsep pola fisiologis menyimpang umum.
  4. **Dataset Sintetis Multi-Level [IEEE DataPort]** (*"Synthetic Dataset for Patient Vitals Monitoring with Scenarios"*):
     - Karakteristik: 200 simulasi pasien, 5.000 titik per pasien (>1 juta points), fitur SpO2, HR, Temp, rolling trends, serta noise Gaussian.
     - Peran: **Data Tambahan (Anomali Bertingkat)** — Mengakomodasi 5 kelas anomali (*Normal, Mild Abnormality, Moderate Abnormality, Critical Condition, Outlier*) yang selaras dengan konsep deteksi dini sistem.
  5. **Harmonisasi Kolom & Eliminasi RESP**:
     - Menyeragamkan penamaan fitur menjadi `Heart_Rate`, `SpO2`, dan `Temperature`.
     - Mengabaikan fitur `RESP` (karena tidak ada sensor fisik pernapasan pada prototype).

### 1.3 Dashboard Pemantauan PC
- **FR-13 (Visualisasi Real-Time)**: Dashboard PC harus menampilkan:
  - Nilai terkini *Heart Rate*, *SpO₂*, dan *Temperature*
  - Grafik tren perubahan parameter terhadap waktu (*time-series chart*)
  - *Signal Quality Score* (SQA)
  - *Anomaly Score* dari Isolation Forest
  - Status operasional sistem (`NORMAL`, `POLA MENYIMPANG`, `KUALITAS SINYAL RENDAH`)
  - Status koneksi ESP32 (Online/Offline) dan *Timestamp* terakhir.

---

## 2. Kebutuhan Non-Fungsional (Non-Functional Requirements)

- **NFR-01 (Latency)**: Latensi pengiriman data sensor dari ESP32 hingga pemrosesan model dan feedback status ke perangkat/dashboard tidak boleh melebihi **1.5 detik**.
- **NFR-02 (Reliability & Kestabilan)**: Sistem harus tahan terhadap *network glitch* lokal dengan logika auto-reconnect Wi-Fi pada ESP32 tanpa menyebabkan mikrokontroler mengalami *freeze* / *panic crash*.
- **NFR-03 (Metrik Evaluasi Ilmiah)**:
  - Evaluasi performa ML (pada dataset berlabel): *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix*.
  - Evaluasi performa sistem IoT: *End-to-End Latency*, *Packet Loss Rate*, dan *Sensor Reading Stability*.
- **NFR-04 (Interoperabilitas & Standar Data)**: Format pertukaran data telemetri harus menggunakan format standar JSON yang tervalidasi skemanya (*schema validation*).
- **NFR-05 (Usability)**: Antarmuka OLED harus tetap terbaca jelas dengan kontras memadai dan pembaruan tampilan minimal 1-2 kali per detik.

---

## 3. Batasan Teknis & Celah Data (Technical Constraints & Data Gaps)

1. **Batasan Daya & Catu Daya**: Prototipe dirancang ditenagai melalui koneksi kabel USB 5V (dapat menggunakan Power Bank untuk skenario portabel). Sensor beroperasi pada tegangan 3.3V dari regulator ESP32.
2. **Batasan Konektivitas**: Komunikasi menggunakan Wi-Fi 2.4 GHz (ESP32-WROOM hanya mendukung frekuensi 2.4 GHz 802.11 b/g/n, tidak mendukung 5 GHz). Jaringan harus berada dalam satu segmen LAN / Wi-Fi Access Point yang sama dengan server.
3. **Batasan Sensor Optik (PPG)**: MAX30102 yang dipasang pada pergelangan tangan sangat rentan terhadap *motion artifact* dan variasi tekanan kulit. Oleh karena itu, posisi dan tekanan *wrist strap* harus diatur stabil.
4. **Batasan Suhu**: Sensor DS18B20 mengukur suhu permukaan kulit luar (*skin surface temperature*), bukan suhu inti tubuh (*core body temperature*). Nilai normal kulit biasanya lebih rendah dari suhu oral/rektal (sekitar 31°C - 35°C).
5. **Batasan Fitur HRV & Respirasi**: Sistem tidak menggunakan fitur elektrokardiogram (ECG) maupun fitur HRV berbasis beat-to-beat seperti *IBI, RMSSD,* atau *SDNN*, karena dataset publik yang dipakai berupa parameter ringkasan (*aggregated tabular values*). Parameter respirasi (*RESP*) diabaikan.
6. **Batasan Medis (Non-Clinical)**: Sistem hanya sebagai pendeteksi anomali statistik distribusi data, bukan alat diagnosis medis.
7. **Kesenjangan Domain Data (Domain Gap)**:
   - Dataset publik (Kaggle & IEEE DataPort) umumnya diperoleh dari instrumen medis/pulse oximeter jari klinis, sedangkan prototipe menggunakan sensor optik MAX30102 pada permukaan pergelangan tangan (*wrist-worn*). Karakteristik noise, baseline drift, dan peredaman sinyal akan berbeda (sesuai Batasan Proposal Poin 7).
8. **Inkonsistensi Makna Label Ground-Truth**:
   - Dataset sekunder tidak memiliki label tri-state yang 100% kongruen dengan status operasional sistem (`Normal` vs `Pola Menyimpang` vs `Kualitas Sinyal Rendah`). Label `Result` pada dataset COVID hanya merefleksikan diagnosis virus spesifik, bukan deviasi fisiologis umum.
9. **Opsi Riset Lanjutan (Post-MVP)**:
   - Jika di masa mendatang dibutuhkan Signal Quality Assessment tingkat lanjut berbasis raw waveform PPG, dapat dieksplorasi dataset **PTT-PPG (PhysioNet)** yang menggunakan sensor MAX30101 dengan ground-truth ECG simultan.
10. **Fleksibilitas Form Factor Fisik (Desktop Demo Testbed vs Wearable Wrist)**:
   - **Tier 1 (Target Utama / Demo Standar)**: Format *Desktop Demo / Testbed Station* (ESP32, OLED, dan Buzzer di breadboard/meja, dengan probe MAX30102 mode jepit jari/fingertip clip dan DS18B20 kontak kulit). Format ini **sudah cukup dan valid** sebagai demo pembuktian sistem, sekaligus secara drastis meningkatkan *Signal-to-Noise Ratio* (SNR) dan memperkecil *domain gap* terhadap dataset klinis.
   - **Tier 2 (Target Lanjutan / Opsional)**: Format *Wearable Wrist-Strap* terintegrasi jika kendala mekanik pergelangan tangan (kestabilan tekanan strap dan penempelan probe) dapat diatasi dengan baik.

---

## 4. Asumsi Perancangan Sistem (Technical Assumptions)

Berikut adalah asumsi-asumsi teknis yang diambil berdasarkan celah yang belum dispesifikasikan secara eksplisit dalam proposal:

- **[ASUMSI-01: Topologi Jaringan]**: Server backend Python berjalan pada PC/Laptop yang terhubung ke satu Wi-Fi router atau Hotspot lokal bersama dengan ESP32. IP address server diketahui dan statis/diatur melalui konfigurasi.
- **[ASUMSI-02: Frekuensi Sampling & Transmisi]**:
  - Sensor MAX30102 disampel pada frekuensi internal 50–100 Hz untuk kalkulasi algoritma deteksi denyut, namun paket telemetri yang dikirimkan ke server backend diagregasikan pada frekuensi **1 Hz (1 detik sekali)** agar tidak membebani Wi-Fi dan antrean backend.
- **[ASUMSI-03: Kriteria SQA (Signal Quality Assessment)]**:
  - SQA awal pada backend ditentukan berdasarkan:
    1. Validitas rentang fisiologis dasar ($30 \le \text{HR} \le 220$ BPM, $70\% \le \text{SpO}_2 \le 100\%$, $25^\circ\text{C} \le \text{Temp} \le 42^\circ\text{C}$).
    2. Fluktuasi mendadak/diskontinuitas yang melampaui batas wajar biologis (misal HR melompat > 40 BPM dalam 1 detik).
    3. Skor SQA dihitung dalam skala 0.0 – 1.0 (nilai $\ge 0.7$ dianggap `GOOD`).
- **[ASUMSI-04: Ukuran Sliding Window Fitur Temporal]**:
  - Fitur *moving average*, *variance*, dan *rate of change* dihitung menggunakan sliding window berukuran **10–30 detik (10–30 sampel @ 1 Hz)** untuk menangkap dinamika perubahan jangka pendek.
- **[ASUMSI-05: Ambang Batas Persistent Anomaly (Debounce)]**:
  - Buzzer dan status `POLA MENYIMPANG` dipertahankan aktif jika model Isolation Forest mengeluarkan prediksi anomali selama minimal **3 hingga 5 kali pembacaan berturut-turut** (3–5 detik), guna mencegah alarm palsu akibat fluktuasi sesaat.
- **[ASUMSI-06: Strategi Multi-Tier Pelatihan & Evaluasi Dataset]**:
  - **Training Set Utama**: Kombinasi `engrarri21` + `nasirayub2` (data riil) untuk mempelajari distribusi variasi pola normal fisiologis.
  - **Training Variasi Anomali**: Dataset sintetis `IEEE DataPort` untuk memperkaya respons model terhadap anomali bertingkat (*mild/moderate/critical*).
  - **Training Suplemen**: Nilai numerik `rishanmascarenhas` (tanpa label).
  - **Evaluasi Akhir Model**: Menggunakan subset validasi terpisah dari `engrarri21` dan skenario fisik 7 kondisi riil dari perangkat ESP32 prototipe.

