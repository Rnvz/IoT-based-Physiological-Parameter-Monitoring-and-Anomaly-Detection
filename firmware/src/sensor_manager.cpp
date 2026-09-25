#include "sensors.h"
#include "config.h"
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <OneWire.h>
#include <DallasTemperature.h>

// Objek sensor
MAX30105 particleSensor;
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

// Variabel untuk menghitung Heart Rate dan SpO2
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute;
int beatAvg;

bool sensors_init() {
    bool success = true;

    // Inisialisasi MAX30102
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
        Serial.println("MAX30102 tidak ditemukan. Periksa koneksi.");
        success = false;
    } else {
        // Konfigurasi MAX30102
        particleSensor.setup(MAX30102_LED_BRIGHTNESS, MAX30102_SAMPLE_RATE, MAX30102_LED_BRIGHTNESS, MAX30102_PULSE_WIDTH, MAX30102_ADC_RANGE);
        Serial.println("MAX30102 berhasil diinisialisasi.");
    }

    // Inisialisasi DS18B20
    ds18b20.begin();
    // Atur resolusi dan non-blocking mode
    ds18b20.setResolution(12);
    ds18b20.setWaitForConversion(false); 
    ds18b20.requestTemperatures(); // Minta suhu pertama kali
    Serial.println("DS18B20 berhasil diinisialisasi.");

    return success;
}

bool read_max30102(float &hr, float &spo2, bool &finger_detected, uint32_t &ppg_amplitude) {
    long irValue = particleSensor.getIR();
    
    // Asumsi ppg_amplitude dapat didekati dari nilai IR (ini bisa disesuaikan)
    ppg_amplitude = (uint32_t)irValue;

    if (irValue < IR_FINGER_THRESHOLD) {
        finger_detected = false;
        hr = 0.0;
        spo2 = 0.0;
        return true;
    }

    finger_detected = true;

    // Deteksi detak jantung
    if (checkForBeat(irValue) == true) {
        long delta = millis() - lastBeat;
        lastBeat = millis();

        beatsPerMinute = 60 / (delta / 1000.0);

        if (beatsPerMinute < 255 && beatsPerMinute > 20) {
            rates[rateSpot++] = (byte)beatsPerMinute;
            rateSpot %= RATE_SIZE;

            // Hitung rata-rata
            beatAvg = 0;
            for (byte x = 0; x < RATE_SIZE; x++)
                beatAvg += rates[x];
            beatAvg /= RATE_SIZE;
        }
    }
    
    hr = beatAvg;
    // SpO2 disederhanakan untuk contoh (perlu algoritma lebih kompleks untuk produksi)
    // Sebagai mock, set ke 98% jika terdeteksi jari
    spo2 = 98.0;

    return true;
}

bool read_ds18b20(float &temperature) {
    // Ambil hasil pembacaan suhu (karena non-blocking, ini adalah hasil request sebelumnya)
    float tempC = ds18b20.getTempCByIndex(0);
    
    // Minta konversi suhu berikutnya agar siap untuk pembacaan di cycle selanjutnya
    ds18b20.requestTemperatures();

    if (tempC != DEVICE_DISCONNECTED_C) {
        temperature = tempC;
        return true;
    }
    return false;
}

SensorData read_all_sensors() {
    SensorData data;
    data.max30102_ok = read_max30102(data.heart_rate, data.spo2, data.finger_detected, data.ppg_amplitude);
    data.ds18b20_ok = read_ds18b20(data.temperature);
    return data;
}
