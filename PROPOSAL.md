# Proposal Proyek IoT

## Judul
**Sistem Pemantauan dan Deteksi Anomali Parameter Fisiologis Berbasis IoT Menggunakan Sinyal PPG, SpO₂, Suhu, dan Machine Learning**

---

## Deskripsi Proyek

Proyek ini mengembangkan sistem pemantauan parameter fisiologis berbasis *Internet of Things* (IoT) secara *real-time* untuk mendeteksi penyimpangan kombinasi parameter fisiologis terhadap pola normal yang dipelajari oleh model *Machine Learning*. Sistem dirancang sebagai prototipe *monitoring* dan *early-warning* non-klinis yang menggabungkan pengukuran denyut, saturasi oksigen, dan suhu permukaan kulit.

Sistem menggunakan ESP32 *development board* berbasis **ESP32-WROOM** sebagai mikrokontroler utama. ESP32 bertugas membaca sensor, melakukan pengolahan data dasar, mengirimkan data melalui jaringan Wi-Fi, serta mengendalikan perangkat *output* lokal.

Sistem menggunakan dua sensor utama:
- **MAX30102** digunakan untuk memperoleh sinyal PPG serta parameter *Heart Rate/Pulse Rate* dan SpO₂.
- **DS18B20** digunakan untuk memperoleh suhu pada permukaan kulit.

MAX30102 dan DS18B20 terhubung ke ESP32 menggunakan antarmuka komunikasi yang sesuai. Untuk implementasi *wearable*, MAX30102 ditempatkan pada bagian bawah pergelangan tangan menggunakan *wrist strap* dengan posisi dan tekanan sensor yang relatif stabil.

Data sensor dikirim melalui Wi-Fi ke komputer/server lokal untuk diproses menggunakan *backend* berbasis Python. Sistem melakukan *preprocessing*, pemeriksaan kualitas sinyal PPG, penggabungan parameter fisiologis, serta ekstraksi fitur perubahan data terhadap waktu.

Parameter utama yang digunakan adalah **Heart Rate/Pulse Rate**, **SpO₂**, dan **Temperature**. Selain nilai pengukuran saat ini, sistem menghitung karakteristik temporal seperti:
- Perubahan nilai (*delta*)
- *Moving average*
- *Variance*
- *Rate of change*

Hal ini bertujuan membantu model mengenali pola perubahan yang tidak hanya bergantung pada satu nilai pengukuran.

Pemilihan fitur disesuaikan dengan data yang tersedia pada sensor dan dataset. Dataset yang ditemukan memiliki parameter seperti HR, SpO₂/Oxygen, PulseRate, Temperature, Time, serta kolom OUTPUT/Result sebagai informasi hasil atau label. Parameter **RESP** yang terdapat pada salah satu dataset tidak digunakan sebagai input utama sistem karena perangkat yang dirancang saat ini belum memiliki sensor respirasi khusus.

Model *Machine Learning* digunakan untuk mendeteksi apakah kombinasi parameter yang diamati berada dalam pola normal atau menunjukkan penyimpangan terhadap distribusi data normal yang dipelajari. Sebelum hasil deteksi digunakan, sistem melakukan pemeriksaan kualitas sinyal PPG agar gangguan akibat pergerakan atau kontak sensor yang tidak stabil tidak langsung dianggap sebagai penyimpangan fisiologis.

Sebagai implementasi *edge monitoring*, sistem dilengkapi dengan:
- **OLED I²C** untuk menampilkan *Heart Rate*, SpO₂, suhu, dan status pengamatan.
- **Buzzer aktif** sebagai peringatan lokal.

*Dashboard* pada komputer digunakan untuk menampilkan parameter secara *real-time*, grafik perubahan parameter, *anomaly score*, dan status sistem.

Sistem berfungsi sebagai *monitoring* dan *early-warning* terhadap penyimpangan pola parameter fisiologis, **bukan sebagai alat diagnosis medis**. Status "Pola Menyimpang" menunjukkan bahwa kombinasi data berbeda dari pola normal yang dipelajari model dan tidak menunjukkan bahwa pengguna mengalami penyakit tertentu.

---

## Model AI

Model utama menggunakan algoritma **Isolation Forest** untuk mendeteksi observasi yang menyimpang dari distribusi data normal. Pendekatan ini dipilih karena dapat digunakan sebagai *unsupervised anomaly detection*, sehingga sistem dapat mempelajari karakteristik pola normal tanpa harus memperoleh sejumlah besar data kondisi abnormal melalui eksperimen langsung pada manusia.

Model menggunakan pendekatan **Feature Fusion**, yaitu menggabungkan beberapa parameter fisiologis yang tersedia secara bersamaan:
- Heart Rate / Pulse Rate
- SpO₂ / Oxygen
- Temperature
- Perubahan Heart Rate
- Perubahan SpO₂
- Perubahan Temperature
- *Moving average* Heart Rate
- *Moving average* SpO₂
- *Moving average* Temperature
- *Variance* parameter
- *Rate of change* parameter
- Karakteristik temporal berdasarkan *timestamp*

Dengan pendekatan tersebut, model tidak hanya mempertimbangkan apakah satu nilai tinggi atau rendah, tetapi juga bagaimana beberapa parameter berubah secara bersamaan dan bagaimana perubahan tersebut berlangsung terhadap waktu.

Contohnya, perubahan Heart Rate yang terjadi secara bertahap dapat memiliki karakteristik berbeda dengan perubahan yang terjadi secara tiba-tiba. Demikian pula perubahan SpO₂ atau suhu dapat dianalisis berdasarkan nilai saat ini dan perubahan relatif terhadap beberapa pengamatan sebelumnya.

Data dari dataset digunakan sebagai sumber pembentukan pola dan validasi, dengan penyesuaian terhadap parameter yang benar-benar tersedia:
- **Dataset pertama** memiliki parameter: `Time`, `HR`, `RESP`, `SpO₂`, dan `TEMP`.
- **Dataset kedua** memiliki parameter: `Oxygen`, `PulseRate`, `Temperature`, dan `Result`.

Parameter yang memiliki makna serupa, seperti `HR`/`PulseRate` dan `SpO₂`/`Oxygen`, dapat diseragamkan menjadi nama fitur yang konsisten.

Model menghasilkan dua kategori utama:
- **Pola Normal**: Kombinasi parameter fisiologis berada dalam distribusi pola normal yang dipelajari oleh model.
- **Pola Menyimpang**: Kombinasi parameter menunjukkan penyimpangan terhadap distribusi pola normal model.

Selain Isolation Forest, digunakan metode **threshold sederhana sebagai baseline**. *Baseline* dapat menggunakan batas sederhana pada Heart Rate, SpO₂, atau perubahan parameter tertentu. Perbandingan dilakukan untuk melihat perbedaan karakteristik deteksi antara pendekatan satu parameter dan pendekatan multivariat berbasis *Machine Learning*.

Performa sistem dievaluasi menggunakan:
- Evaluasi Model: **Precision**, **Recall**, **F1-Score**, dan **Confusion Matrix** (apabila data pengujian memiliki label yang dapat digunakan).
- Evaluasi Sistem IoT: **Latency deteksi**, **kestabilan komunikasi IoT**, **packet loss**, dan **kestabilan pembacaan sensor**.

---

## Definisi Anomali

Dalam penelitian ini, anomali **tidak didefinisikan sebagai penyakit atau diagnosis medis**.

Anomali secara operasional didefinisikan sebagai kondisi ketika kombinasi nilai Heart Rate/Pulse Rate, SpO₂, suhu permukaan kulit, serta perubahan parameter terhadap waktu menunjukkan penyimpangan terhadap distribusi pola normal yang dipelajari oleh model.

Dengan demikian:
- Anomali tidak ditentukan hanya berdasarkan satu nilai Heart Rate.
- SpO₂ tidak digunakan sebagai diagnosis penyakit tertentu.
- Suhu yang diperoleh merupakan suhu permukaan kulit dan bukan suhu inti tubuh.
- Perubahan parameter akibat aktivitas normal tidak secara otomatis dianggap sebagai kondisi medis abnormal.
- Perubahan yang disebabkan oleh gangguan pembacaan sensor harus dibedakan melalui pemeriksaan kualitas sinyal.

Sistem membedakan tiga kondisi operasional:
1. **Pola Normal**: Kombinasi parameter berada dalam distribusi normal model.
2. **Pola Menyimpang**: Kombinasi parameter menunjukkan penyimpangan terhadap distribusi normal model dengan kualitas data yang memadai.
3. **Kualitas Sinyal Rendah**: Data PPG tidak cukup baik untuk digunakan sebagai dasar deteksi sehingga sistem meminta pengguna memperbaiki posisi atau kontak sensor.

Dengan mekanisme tersebut, *motion artifact*, sensor yang bergeser, atau kontak yang tidak stabil tidak langsung dikategorikan sebagai anomali fisiologis.

---

## Pengumpulan Data

Pengumpulan data dilakukan melalui dua sumber utama:

### 1. Fase Pelatihan dan Pembentukan Pola
Data sekunder diperoleh dari dataset fisiologis publik yang memiliki parameter yang relevan dengan perangkat.

#### Dataset 1:
| Parameter | Keterangan |
| :--- | :--- |
| `Time` | Waktu pengamatan |
| `HR` | Heart Rate |
| `RESP` | Respiratory Rate |
| `SpO₂` | Saturasi oksigen |
| `TEMP` | Temperature |
| `OUTPUT` | Hasil/label dataset |

#### Dataset 2:
| Parameter | Keterangan |
| :--- | :--- |
| `Oxygen` | Saturasi oksigen |
| `PulseRate` | Pulse/Heart Rate |
| `Temperature` | Temperature |
| `Result` | Hasil/label dataset |

Parameter yang memiliki arti serupa akan dinormalisasi agar memiliki format yang konsisten, misalnya:
- `HR` / `PulseRate` $\rightarrow$ `Heart_Rate`
- `SpO2` / `Oxygen` $\rightarrow$ `SpO2`
- `TEMP` / `Temperature` $\rightarrow$ `Temperature`

Parameter `RESP` tidak digunakan dalam model utama karena perangkat prototype tidak memiliki sensor respirasi.

#### Alur Pemrosesan Data:
$$\text{Dataset} \rightarrow \text{Data Cleaning} \rightarrow \text{Normalisasi Nama Parameter} \rightarrow \text{Missing Value Handling} \rightarrow \text{Feature Engineering} \rightarrow \text{Isolation Forest}$$

*Feature engineering* menghasilkan fitur tambahan seperti:
- Heart Rate, SpO₂, Temperature
- **+** Heart Rate Delta, SpO₂ Delta, Temperature Delta
- **+** Moving Average, Variance, Rate of Change

---

### 2. Fase Pengujian Real-Time
Pada fase pengujian, ESP32 membaca data secara langsung dari:
- **MAX30102**: $\rightarrow$ PPG, $\rightarrow$ Heart Rate, $\rightarrow$ SpO₂
- **DS18B20**: $\rightarrow$ Temperature

Data diberikan *timestamp* kemudian dikirimkan melalui Wi-Fi ke komputer/server lokal.

Pengujian fisik dilakukan dalam beberapa skenario aman, seperti:
1. Kondisi istirahat
2. Aktivitas mengetik
3. Aktivitas fisik ringan
4. Perubahan posisi tangan
5. Variasi posisi sensor pada pergelangan
6. Variasi tekanan *wrist strap*
7. Kondisi kontak sensor yang kurang stabil

Skenario tersebut digunakan untuk menguji stabilitas sensor, kualitas sinyal, komunikasi ESP32, dan respons sistem terhadap perubahan parameter, **bukan untuk sengaja menghasilkan kondisi medis abnormal**.

---

## Input Sistem

Input sistem dibagi menjadi *raw sensor input* dan *feature input*.

### Raw Sensor Input
- **MAX30102**:
  - Raw PPG Signal
  - Heart Rate / Pulse Rate
  - SpO₂
- **DS18B20**:
  - Temperature
- **ESP32**:
  - Timestamp
  - Status koneksi
  - Status pembacaan sensor

### Feature Input Machine Learning
Fitur yang digunakan oleh model meliputi:
- Heart Rate
- SpO₂
- Temperature
- Heart Rate Delta
- SpO₂ Delta
- Temperature Delta
- Moving Average Heart Rate
- Moving Average SpO₂
- Moving Average Temperature
- Heart Rate Variance
- SpO₂ Variance
- Temperature Variance
- Rate of Change
- Signal Quality Score

> [!NOTE]
> Model tidak menggunakan `IBI`, `RMSSD`, dan `SDNN`, karena kedua dataset yang ditemukan tidak menyediakan informasi *beat-to-beat interval* yang diperlukan untuk menghitung fitur tersebut.
> 
> *Raw PPG* tetap digunakan dalam sistem untuk *Signal Quality Assessment* dan validasi pembacaan, tetapi tidak digunakan sebagai fitur utama Isolation Forest jika data *training* yang tersedia hanya berupa parameter hasil ekstraksi seperti HR, SpO₂, dan Temperature.

---

## Output Sistem

Sistem menghasilkan dua lapis output, yaitu **Output Lokal pada perangkat ESP32** dan **Output pada Dashboard PC**.

### 1. Output Lokal — Perangkat ESP32

#### OLED I²C
OLED menampilkan:

```text
--------------------
PHYSIO MONITOR
--------------------
HR     : 78 BPM
SpO2   : 98 %
TEMP   : 32.4 C
STATUS : NORMAL
--------------------
```

Apabila kualitas PPG rendah:
```text
--------------------
PHYSIO MONITOR
--------------------
HR     : -- BPM
SpO2   : -- %
TEMP   : 32.4 C
STATUS : CEK SENSOR
--------------------
```

Apabila model mendeteksi penyimpangan:
```text
--------------------
PHYSIO MONITOR
--------------------
HR     : 105 BPM
SpO2   : 97 %
TEMP   : 32.8 C
STATUS : ANOMALI
--------------------
```

#### Buzzer Aktif
Buzzer digunakan sebagai peringatan lokal ketika:
$$\text{Signal Quality = GOOD} + \text{Model = Pola Menyimpang} \longrightarrow \textbf{BUZZER}$$

Untuk mengurangi *false alarm*, buzzer sebaiknya tidak aktif hanya berdasarkan satu pembacaan anomali. Sistem dapat menggunakan **persistent anomaly**, yaitu anomali harus muncul secara konsisten pada beberapa *window* pengamatan sebelum buzzer diaktifkan.

---

### 2. Output Dashboard PC

Dashboard menampilkan:
- Heart Rate / Pulse Rate
- SpO₂
- Temperature
- Grafik perubahan parameter terhadap waktu
- Signal Quality Score
- Anomaly Score
- Status sistem
- Timestamp
- Status koneksi ESP32

#### Status Dashboard:
- **NORMAL**: Parameter berada dalam distribusi pola normal model.
- **POLA MENYIMPANG**: Kombinasi parameter menunjukkan penyimpangan dari distribusi normal model.
- **KUALITAS SINYAL RENDAH**: Kualitas pembacaan PPG tidak memadai sehingga hasil deteksi tidak digunakan untuk menyatakan penyimpangan fisiologis.

---

## Batasan Sistem

1. **Bukan Alat Diagnosis Medis**: Sistem tidak digunakan sebagai alat diagnosis medis dan tidak digunakan untuk menentukan apakah pengguna mengalami aritmia, takikardia, bradikardia, hipoksia, demam, hipotermia, atau penyakit tertentu. Sistem hanya mendeteksi penyimpangan pola data terhadap distribusi yang dipelajari model.
2. **Karakteristik Sensor Optik (PPG)**: MAX30102 menghasilkan sinyal PPG optik, sehingga Heart Rate dan SpO₂ yang diperoleh merupakan hasil pengukuran berbasis sensor optik. Sistem tidak menggunakan ECG dan tidak menghitung fitur HRV seperti RMSSD atau SDNN karena dataset yang digunakan tidak menyediakan data IBI/beat-to-beat yang diperlukan.
3. **Suhu Permukaan Kulit**: DS18B20 digunakan untuk mengukur suhu pada lokasi pemasangan sensor, sehingga nilai tersebut diperlakukan sebagai suhu permukaan kulit dan bukan suhu inti tubuh.
4. **Kerentanan Motion Artifact**: MAX30102 yang dipasang pada pergelangan tangan rentan terhadap *motion artifact*, perubahan tekanan sensor, pergeseran posisi, dan perubahan kontak dengan kulit. Oleh karena itu, *Signal Quality Assessment* menjadi bagian penting sebelum hasil deteksi digunakan.
5. **Peniadaan Parameter Respirasi (RESP)**: Parameter RESP tidak digunakan sebagai input utama model, karena perangkat prototype tidak memiliki sensor khusus untuk mengukur *respiratory rate*. Meskipun RESP tersedia pada salah satu dataset, parameter tersebut tidak dapat secara langsung direproduksi oleh perangkat yang dikembangkan.
6. **Keterbatasan Dataset Sekunder**: Dataset yang digunakan memiliki parameter yang sudah diekstraksi seperti HR, PulseRate, SpO₂/Oxygen, dan Temperature. Oleh karena itu, fitur seperti IBI, RMSSD, dan SDNN tidak digunakan dalam model utama kecuali pada tahap pengembangan berikutnya tersedia dataset *raw PPG* yang memungkinkan fitur tersebut dihitung secara konsisten.
7. **Variabilitas Domain Data**: Terdapat kemungkinan perbedaan karakteristik antara dataset publik dan data MAX30102 + DS18B20, termasuk perbedaan sensor, metode pengukuran, kondisi lingkungan, populasi, dan karakteristik data. Oleh karena itu, hasil model tidak dapat dianggap berlaku secara umum untuk seluruh pengguna.
8. **Karakteristik Isolation Forest**: Isolation Forest mempelajari pola berdasarkan distribusi data *training*. Kondisi yang belum terdapat dalam data *training* dapat menghasilkan *anomaly score* tinggi meskipun belum tentu menunjukkan kondisi yang berbahaya.
9. **Ketergantungan Jaringan Wi-Fi untuk ML**: Sistem memerlukan koneksi Wi-Fi dan backend komputer untuk melakukan proses *Machine Learning*. Apabila koneksi terputus, fungsi *monitoring* lokal masih dapat menampilkan data sensor tertentu, tetapi deteksi berbasis model tidak dapat berjalan secara penuh.
10. **Protokol Pengujian Aman**: Pengujian tidak dilakukan dengan sengaja menghasilkan kondisi fisiologis abnormal atau kondisi medis tertentu. Pengujian difokuskan pada kondisi normal, variasi aktivitas, perubahan penggunaan sensor, dan gangguan kualitas sinyal.
