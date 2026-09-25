#ifndef CONFIG_H
#define CONFIG_H

// Konfigurasi Pin
#define I2C_SDA 21
#define I2C_SCL 22
#define DS18B20_PIN 4
#define BUZZER_PIN 25

// Konfigurasi OLED
#define OLED_WIDTH 128
#define OLED_HEIGHT 64
#define OLED_ADDR 0x3C

// Interval dan Waktu (dalam ms)
#define SENSOR_READ_INTERVAL_MS 1000
#define OLED_REFRESH_INTERVAL_MS 500
#define BUZZER_TIMEOUT_MS 3000
#define BUZZER_COOLDOWN_MS 10000
#define WIFI_RECONNECT_INTERVAL_MS 5000

// Konfigurasi MAX30102
#define MAX30102_LED_BRIGHTNESS 60
#define MAX30102_SAMPLE_RATE 100
#define MAX30102_PULSE_WIDTH 411
#define MAX30102_ADC_RANGE 4096

// Ambang Batas Sensor
#define IR_FINGER_THRESHOLD 50000

// Konfigurasi Komunikasi
#define DEVICE_ID "esp32_sensor_01"
#define MQTT_TOPIC_TELEMETRY "physio/" DEVICE_ID "/telemetry"
#define MQTT_TOPIC_FEEDBACK "physio/" DEVICE_ID "/feedback"

#endif // CONFIG_H
