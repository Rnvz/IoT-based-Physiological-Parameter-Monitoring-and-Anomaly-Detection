# Dokumentasi Konflik Firmware (WiFi Credentials)

> [!WARNING]
> File `firmware/physio_monitor/physio_monitor.ino` sengaja **TIDAK diedit secara otomatis** pada audit ini agar pemilik hardware / firmware engineer dapat memilih kredensial WiFi yang sesuai dengan lingkungan pengujian fisik masing-masing sebelum flashing ke ESP32.

---

## 1. Lokasi & Blok Konflik

- **File**: `firmware/physio_monitor/physio_monitor.ino`
- **Baris**: 14 - 25

### Kutipan Blok Konflik Mentah:

```cpp
<<<<<<< HEAD

// ============================================================
// WIFI
// ============================================================

const char* WIFI_SSID = "iQOO 12";
const char* WIFI_PASS = "Password123";
=======
const char* WIFI_SSID = "iQOO 15R";
const char* WIFI_PASS = "password";
>>>>>>> af17c59 (refactor: reorganize ML and DL into separate folders and clean source code comments)
```

---

## 2. Analisis Kedua Sisi

| Sisi Git | Kredensial WiFi | Catatan Konteks |
|---|---|---|
| **Sisi HEAD** (Branch Kerja Lokal) | SSID: `"iQOO 12"`<br>Password: `"Password123"` | Memiliki header komentar section `// WIFI`. Digunakan pada hotspot pengujian lokal developer pertama. |
| **Sisi af17c59** (Kolaborator) | SSID: `"iQOO 15R"`<br>Password: `"password"` | Berasal dari commit refactoring kolaborator yang menggunakan hotspot smartphone iQOO 15R. |

---

## 3. Langkah Resolusi untuk Pemilik Firmware

Pemilik hardware ESP32 dapat menyelesaikan konflik dengan langkah berikut:

### Langkah 1: Buka File Firmware
Buka [physio_monitor.ino](file:///home/ywenanta/Projects/IoT/firmware/physio_monitor/physio_monitor.ino#L14-L25) menggunakan text editor atau Arduino IDE.

### Langkah 2: Pilih Salah Satu Kredensial & Hapus Marker Konflik

#### Opsi A: Menggunakan Hotspot iQOO 12 (HEAD)
Ganti baris 14–25 menjadi:
```cpp
// ============================================================
// WIFI
// ============================================================

const char* WIFI_SSID = "iQOO 12";
const char* WIFI_PASS = "Password123";
```

#### Opsi B: Menggunakan Hotspot iQOO 15R (Kolaborator)
Ganti baris 14–25 menjadi:
```cpp
// ============================================================
// WIFI
// ============================================================

const char* WIFI_SSID = "iQOO 15R";
const char* WIFI_PASS = "password";
```

#### Opsi C: Menggunakan Kredensial WiFi Lokal Anda
```cpp
// ============================================================
// WIFI
// ============================================================

const char* WIFI_SSID = "SSID_WIFI_ANDA";
const char* WIFI_PASS = "PASSWORD_WIFI_ANDA";
```

Pastikan semua penanda konflik (`<<<<<<< HEAD`, `=======`, `>>>>>>> ...`) telah terhapus seutuhnya.

---

## 4. Verifikasi Kompilasi & Flashing

Setelah memilih kredensial, verifikasi sintaks dan kompilasi firmware:

### Menggunakan Arduino CLI:
```bash
# Pastikan core esp32 terpasang
arduino-cli core install esp32:esp32

# Kompilasi sketch
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/physio_monitor/physio_monitor.ino

# Upload ke ESP32 (sesuaikan port serial, mis. /dev/ttyUSB0)
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 firmware/physio_monitor/physio_monitor.ino
```

### Menggunakan Arduino IDE:
1. Buka `firmware/physio_monitor/physio_monitor.ino`.
2. Pilih Board: **ESP32 Dev Module**.
3. Pastikan library terpasang:
   - `DallasTemperature`
   - `DHT sensor library`
   - `ArduinoJson` (v6.x)
   - `PubSubClient`
4. Klik tombol **Verify** (Centang) untuk memastikan tidak ada error kompilasi.
5. Klik **Upload** untuk mem-flash program ke ESP32.
6. Buka Serial Monitor pada baud rate `115200` untuk memverifikasi koneksi WiFi dan MQTT terhubung.
