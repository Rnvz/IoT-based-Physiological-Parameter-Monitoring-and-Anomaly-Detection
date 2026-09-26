#include <Arduino.h>
#include <Wire.h>
#include "config.h"
#include "secrets.h"
#include "sensors.h"
#include "display_manager.h"
#include "comm_client.h"

// State Global
SensorData current_sensor_data;
SystemStatus current_system_status = STATUS_NORMAL;

// Manajemen Waktu
unsigned long last_sensor_read_time = 0;
unsigned long last_oled_refresh_time = 0;

// State Buzzer
bool buzzer_is_active = false;
unsigned long buzzer_start_time = 0;
unsigned long last_buzzer_trigger_time = 0;

// Fungsi callback saat menerima feedback dari MQTT
void on_feedback_received(SystemStatus new_status, bool trigger_buzzer) {
    current_system_status = new_status;
    
    // Logika kontrol buzzer dengan cooldown dan timeout
    unsigned long current_time = millis();
    if (trigger_buzzer) {
        if (!buzzer_is_active && (current_time - last_buzzer_trigger_time >= BUZZER_COOLDOWN_MS)) {
            buzzer_is_active = true;
            buzzer_start_time = current_time;
            digitalWrite(BUZZER_PIN, HIGH);
            Serial.println("Buzzer diaktifkan (Anomali terdeteksi).");
        }
    } else {
        if (buzzer_is_active) {
            buzzer_is_active = false;
            digitalWrite(BUZZER_PIN, LOW);
            last_buzzer_trigger_time = current_time;
            Serial.println("Buzzer dimatikan oleh server.");
        }
    }
}

void setup() {
    Serial.begin(115200);
    Serial.println("Memulai sistem...");

    // Inisialisasi I2C
    Wire.begin(I2C_SDA, I2C_SCL);

    // Inisialisasi pin Buzzer
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    // Inisialisasi Layar OLED
    display_init();

    // Inisialisasi Sensor
    if (!sensors_init()) {
        Serial.println("Gagal menginisialisasi beberapa sensor!");
    }

    // Inisialisasi Komunikasi (WiFi dan MQTT)
    comm_init(on_feedback_received);
}

void loop() {
    unsigned long current_time = millis();

    // 1. Update Komunikasi (non-blocking)
    comm_loop();

    // 2. Baca Sensor & Kirim Data
    if (current_time - last_sensor_read_time >= SENSOR_READ_INTERVAL_MS) {
        last_sensor_read_time = current_time;

        current_sensor_data = read_all_sensors();

        // Update status lokal jika tidak ada jari
        if (!current_sensor_data.finger_detected) {
            current_system_status = STATUS_SIGNAL_QUALITY_LOW;
        } else if (current_system_status == STATUS_SIGNAL_QUALITY_LOW) {
            // Jika jari kembali terdeteksi dan status sebelumnya STATUS_SIGNAL_QUALITY_LOW, kembalikan ke STATUS_NORMAL
            // (Status deviasi hanya diubah oleh feedback dari server)
            current_system_status = STATUS_NORMAL;
        }

        // Kirim telemetri (hanya dikirim jika WiFi terhubung)
        comm_send_telemetry(current_sensor_data, current_system_status);
    }

    // 3. Update Tampilan OLED
    if (current_time - last_oled_refresh_time >= OLED_REFRESH_INTERVAL_MS) {
        last_oled_refresh_time = current_time;
        display_update(current_sensor_data, current_system_status, comm_is_wifi_connected());
    }

    // 4. Manajemen Timeout Buzzer (Keamanan)
    if (buzzer_is_active && (current_time - buzzer_start_time >= BUZZER_TIMEOUT_MS)) {
        buzzer_is_active = false;
        digitalWrite(BUZZER_PIN, LOW);
        last_buzzer_trigger_time = current_time;
        Serial.println("Buzzer dimatikan (Timeout keamanan tercapai).");
    }
}
