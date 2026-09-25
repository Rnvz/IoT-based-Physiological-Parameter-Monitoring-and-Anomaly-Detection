# SECURITY & SAFETY: Kebijakan Keamanan dan Keselamatan Sistem IoT

Dokumen ini mendefinisikan prinsip keamanan siber, privasi data fisiologis, manajemen kredensial, kebijakan pembaruan firmware (*OTA*), serta mekanisme proteksi keselamatan fisik (*hardware safety*) untuk prototipe IoT.

---

## 1. Model Ancaman & Konteks Riset (Threat Model)

Sistem ini beroperasi di lingkungan jaringan lokal (LAN/WLAN) untuk keperluan riset prototipe akademik *wearable monitoring*. Meskipun bukan perangkat medis klinis, data yang diakuisisi adalah parameter biometrik fisiologis personal (Heart Rate, SpO₂, dan Suhu Tubuh), sehingga aspek integritas data, kerahasiaan, dan keselamatan fisik perangkat menjadi prioritas.

---

## 2. Kebijakan Manajemen Kredensial & Secrets

1. **Prinsip Zero-Secrets in Git**:
   - Dilarang keras menuliskan SSID Wi-Fi, kata sandi, token autentikasi, atau path direktori sensitif ke dalam source code yang dilacak oleh Git.
   - Semua kredensial jaringan harus disimpan dalam:
     - Firmware: File lokal `include/secrets.h` yang terdaftar dalam `.gitignore`. Disediakan template `include/secrets.h.example` untuk acuan pengembang.
     - Backend & Dashboard: File lingkungan `.env` yang terdaftar dalam `.gitignore`. Disediakan template `.env.example`.
2. **Isolasi Lingkungan Pengembangan**:
   - Pengujian disarankan menggunakan jaringan Wi-Fi laboratorium/hotspot pribadi terisolasi, bukan jaringan publik terbuka tanpa enkripsi WPA2/WPA3.

---

## 3. Kebijakan Autentikasi & Otorisasi Perangkat (Device Auth)

Untuk mencegah data palsu (*data injection*) dan akses tidak sah:

1. **Pre-Shared Device Token (Bearer Token)**:
   - Setiap perangkat ESP32 memiliki identitas unik `device_id` dan rahasia pra-bagi `device_secret_token`.
   - Pada komunikasi MQTT: ESP32 menggunakan *Username* dan *Password* terenkripsi saat *handshake* dengan broker Mosquitto. Topik MQTT dibatasi melalui ACL (*Access Control List*), misal hanya boleh menerbitkan ke `physio/{device_id}/telemetry` dan berlangganan ke `physio/{device_id}/feedback`.
   - Pada komunikasi HTTP/WebSocket: Header HTTP `X-Device-Token` wajib disertakan pada setiap paket telemetri dan divalidasi oleh middleware backend.
2. **Validasi Skema & Sanitasi Input**:
   - Backend memvalidasi tipe data dan rentang fisik semua parameter yang masuk sebelum diproses. Payload dengan tipe data rusak atau nilai di luar batas fisik ekstrem yang tidak masuk akal akan ditolak secara otomatis untuk mencegah potensi serangan *denial of service* (DoS) berbasis *crash* memori.

---

## 4. Perlindungan Privasi Data Fisiologis Sensitif

1. **Penyimpanan Data Lokal (Local-Only Data Residency)**:
   - Seluruh data telemetri, log eksperimen, dan model Machine Learning hanya disimpan pada server lokal komputer pengguna. Tidak ada data fisiologis yang dikirimkan ke cloud pihak ketiga tanpa persetujuan eksplisit.
2. **Anonimisasi Data Subjek Uji**:
   - Log sesi pengujian fisik tidak boleh mencantumkan informasi identitas pribadi (*Personally Identifiable Information* - PII) seperti nama lengkap, NIK, atau kontak subjek uji.
   - Subjek uji hanya diidentifikasi dengan kode pseudonim (misal: `Subject_A_Resting`, `Subject_B_Typing`).
3. **Penyimpanan Terenkripsi (Opsional untuk Produksi)**:
   - Untuk data arsip jangka panjang, file basis data SQLite/CSV disarankan disimpan pada partisi drive yang terenkripsi (misal BitLocker / LUKS).

---

## 5. Kebijakan Pembaruan Firmware (Over-The-Air / OTA)

Pada tahap prototipe saat ini:

1. **Metode Flashing Utama (Wired UART)**:
   - Flashing firmware diutamakan melalui kabel USB UART langsung dari PC pengembang. Metode ini meminimalkan risiko *bricking* perangkat dan celah eksploitasi OTA nirkabel di lingkungan riset.
2. **Kebijakan jika Fitur OTA Diaktifkan (Future Phase)**:
   - **Kriptografi & Checksum**: Firmware binary yang diunduh wajib menyertakan verifikasi hash SHA-256 untuk memastikan integritas file dari kerusakan transmisi jaringan.
   - **Autentikasi OTA**: Endpoint pembaruan OTA pada ESP32 (ArduinoOTA / HTTP OTA) wajib dilindungi oleh kata sandi/token yang kuat.
   - **Dual-Partition Rollback**: ESP32 harus dikonfigurasi dengan skema partisi dua slot aplikasi (`app0` dan `app1` dengan partisi `otadata`). Jika firmware baru gagal boot atau mengalami *crash loop*, mikrokontroler secara otomatis melakukan *rollback* ke versi firmware sebelumnya yang stabil.

---

## 6. Keselamatan Fisik Perangkat Keras (Hardware Safety & Fail-Safe)

Mengingat perangkat bersentuhan langsung dengan kulit manusia:

1. **Proteksi Arus & Termal**:
   - Sensor MAX30102 dan DS18B20 beroperasi pada level tegangan rendah (3.3V DC). Pengaturan arus LED pada MAX30102 dibatasi pada tingkat aman standar untuk mencegah panas berlebih (*thermal discomfort*) pada permukaan kulit pergelangan tangan.
2. **Buzzer Failsafe & Anti-Ear Fatigue**:
   - Suara buzzer yang berbunyi terus-menerus dapat mengganggu pendengaran pengguna dan memicu kepanikan.
   - Wajib diimplementasikan **Software Timeout**: buzzer aktif maksimal selama **3–5 detik** per kejadian anomali, kemudian harus mati (*cooldown period*) minimal 10 detik sebelum diizinkan menyala kembali.
3. **Software Watchdog Timer (WDT)**:
   - ESP32 wajib mengaktifkan Task Watchdog Timer (TWDT) internal (interval ~5 detik). Jika terjadi *infinite loop* atau pembacaan sensor membeku, mikrokontroler akan me-reset dirinya sendiri secara aman tanpa memerlukan intervensi manual.
4. **Resiliensi Terputus Jaringan (Edge Autonomy)**:
   - Putusnya jaringan Wi-Fi tidak boleh mematikan fungsi pemantauan dasar pada OLED. ESP32 harus tetap menampilkan pembacaan sensor lokal dan menonaktifkan buzzer hingga status terkonfirmasi kembali oleh backend.
