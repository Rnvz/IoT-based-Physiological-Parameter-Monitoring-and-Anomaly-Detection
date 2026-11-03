# Panduan Patch Firmware: Integrasi Buzzer Mengikuti Hasil Model (/result)

> [!IMPORTANT]
> **PEMBERITAHUAN PENTING**:
> Dokumen ini memuat patch kode persis (*unified diff*) untuk pemilik firmware ESP32 (`firmware/physio_monitor/physio_monitor.ino`).
> Sesuai batasan sistem, repositori backend tidak mengubah langsung file firmware. Patch ini harus diaplikasikan dan di-flash oleh pengembang firmware.

---

## 1. Latar Belakang & Perilaku Aktuator

Saat ini:
- **Lampu RGB** pada ESP32 sudah secara otomatis mengikuti hasil evaluasi cloud/backend (`/result`):
  - `lastSeverity == 0`: Hijau (40% brightness, `ML_RESULT`)
  - `lastSeverity == 1`: Kuning (70% brightness, `ML_RESULT`)
  - `lastSeverity == 2`: Merah (100% brightness, `ML_RESULT`)
  - Jika koneksi terputus atau tidak ada hasil selama $>10$ detik (`RESULT_TIMEOUT = 10000UL`), lampu otomatis kembali ke status otonom ambang lokal firmware.
- **Buzzer** saat ini hanya berbunyi jika status lokal firmware `statusSistem == "ANOMALI"`.
- Dengan patch ini, buzzer akan mengikuti `lastSeverity == 2` dari payload `/result`, dan otomatis kembali (*fallback*) ke ambang lokal jika tidak ada hasil `/result` selama $>10$ detik.

---

## 2. Karakteristik Sensitivitas DL vs Ambang Rule

> [!WARNING]
> **Karakteristik Deteksi Hipoksia DenseAE v2 (DL)**:
> - **Rule Engine Demo**: Ambang kritis (Severity 2) ditetapkan pada $\text{SpO}_2 < 90\%$.
> - **DenseAE v2 (DL)**: Karena fungsi penalti rekonstruksi kuadratik (*mean squared error* pada fitur ternormalisasi), DenseAE v2 memicu ambang kritis (Severity 2) pada **$\text{SpO}_2 \approx 93,1\%$**.
> - **Implikasi**: Ambang DL lebih agresif ($\text{SpO}_2 \approx 93\%$ vs rule $90\%$), namun belum terbukti lebih baik secara klinis.

---

## 3. Unified Diff Patch untuk `firmware/physio_monitor/physio_monitor.ino`

Berikut adalah patch persis untuk fungsi `updateBuzzer()` (sekitar baris 1314–1358):

```diff
--- a/firmware/physio_monitor/physio_monitor.ino
+++ b/firmware/physio_monitor/physio_monitor.ino
@@ -1331,7 +1331,22 @@
     }
   }
 
-  if (statusSistem != "ANOMALI") {
+  // Tentukan apakah kondisi alarm kritis aktif
+  bool isCriticalAlarm = false;
+
+  if (haveResult && (now - lastResultMs <= RESULT_TIMEOUT)) {
+    // Mengikuti /result dari backend/cloud jika data segar (<= 10 detik)
+    if (lastSeverity >= 2) {
+      isCriticalAlarm = true;
+    }
+  } else {
+    // Fallback otonom: kembali ke ambang batas lokal jika offline / timeout > 10 s
+    if (statusSistem == "ANOMALI") {
+      isCriticalAlarm = true;
+    }
+  }
+
+  if (!isCriticalAlarm) {
 
     digitalWrite(BUZZER_PIN, BUZZER_MATI);
 
```

---

## 4. Kebijakan Kendali Aktuator

Aktuasi buzzer sepenuhnya diatur secara otonom oleh firmware ESP32 berdasarkan tingkat keparahan (`severity`) yang diterima pada topik MQTT `/result` (serta fallback otonom ke ambang batas lokal saat timeout > 10 detik). Backend tidak mengontrol pin buzzer secara langsung dan tidak memerlukan variabel konfigurasi khusus (backend tidak menyimpan status kepatuhan aktuator).

