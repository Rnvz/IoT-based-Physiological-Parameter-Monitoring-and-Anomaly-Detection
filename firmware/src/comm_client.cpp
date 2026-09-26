#include "comm_client.h"
#include "config.h"
#include "secrets.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

WiFiClient espClient;
PubSubClient mqttClient(espClient);
FeedbackCallback fb_callback = nullptr;

unsigned long last_wifi_reconnect_attempt = 0;
unsigned long wifi_backoff_delay = WIFI_RECONNECT_INTERVAL_MS;
const unsigned long MAX_BACKOFF_DELAY = 30000;
uint32_t sequence_id = 0;

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
    Serial.print("Pesan diterima di topik: ");
    Serial.println(topic);

    // Parsing JSON feedback
    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, payload, length);

    if (error) {
        Serial.print("Gagal parsing JSON: ");
        Serial.println(error.c_str());
        return;
    }

    // Mendapatkan data dari JSON feedback backend
    const char* status_str = doc["status"];
    bool buzzer_active = doc["buzzer_active"] | false;

    SystemStatus new_status = STATUS_NORMAL;
    if (status_str) {
        if (strcmp(status_str, "HIGH DEVIATION (SUSTAINED)") == 0) {
            new_status = STATUS_HIGH_DEVIATION_SUSTAINED;
        } else if (strcmp(status_str, "HIGH DEVIATION") == 0) {
            new_status = STATUS_HIGH_DEVIATION;
        } else if (strcmp(status_str, "LOW DEVIATION (SUSTAINED)") == 0) {
            new_status = STATUS_LOW_DEVIATION_SUSTAINED;
        } else if (strcmp(status_str, "LOW DEVIATION") == 0) {
            new_status = STATUS_LOW_DEVIATION;
        } else if (strcmp(status_str, "SIGNAL QUALITY LOW") == 0) {
            new_status = STATUS_SIGNAL_QUALITY_LOW;
        } else {
            new_status = STATUS_NORMAL;
        }
    }

    if (fb_callback) {
        fb_callback(new_status, buzzer_active);
    }
}

void connect_wifi_non_blocking() {
    unsigned long current_time = millis();
    if (WiFi.status() != WL_CONNECTED) {
        if (current_time - last_wifi_reconnect_attempt >= wifi_backoff_delay) {
            last_wifi_reconnect_attempt = current_time;
            Serial.print("Mencoba koneksi Wi-Fi ke ");
            Serial.println(WIFI_SSID);
            
            WiFi.disconnect();
            WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
            
            // Tambah backoff delay
            wifi_backoff_delay *= 2;
            if (wifi_backoff_delay > MAX_BACKOFF_DELAY) {
                wifi_backoff_delay = MAX_BACKOFF_DELAY;
            }
        }
    } else {
        // Reset backoff delay jika terhubung
        wifi_backoff_delay = WIFI_RECONNECT_INTERVAL_MS;
    }
}

void connect_mqtt() {
    if (!mqttClient.connected()) {
        Serial.print("Mencoba koneksi MQTT...");
        if (mqttClient.connect(DEVICE_ID, MQTT_USER, MQTT_PASSWORD)) {
            Serial.println("terhubung");
            mqttClient.subscribe(MQTT_TOPIC_FEEDBACK);
        } else {
            Serial.print("gagal, rc=");
            Serial.println(mqttClient.state());
        }
    }
}

void comm_init(FeedbackCallback callback) {
    fb_callback = callback;
    WiFi.mode(WIFI_STA);
    mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
    mqttClient.setCallback(mqtt_callback);
    
    // Inisiasi awal Wi-Fi
    connect_wifi_non_blocking();
}

void comm_loop() {
    connect_wifi_non_blocking();

    if (WiFi.status() == WL_CONNECTED) {
        if (!mqttClient.connected()) {
            // Kita coba koneksi MQTT secara berkala di sini (untuk sederhananya dicoba tiap loop jika putus)
            // Namun sebaiknya gunakan timer millis() agar tidak blocking jika broker mati.
            static unsigned long last_mqtt_reconnect = 0;
            if (millis() - last_mqtt_reconnect > 5000) {
                last_mqtt_reconnect = millis();
                connect_mqtt();
            }
        } else {
            mqttClient.loop();
        }
    }
}

void comm_send_telemetry(const SensorData& data, SystemStatus current_status) {
    if (WiFi.status() != WL_CONNECTED || !mqttClient.connected()) {
        return;
    }

    StaticJsonDocument<512> doc;
    doc["device_id"] = DEVICE_ID;
    doc["sequence_id"] = sequence_id++;
    doc["timestamp_ms"] = millis();
    
    // Sensor data
    JsonObject telemetry = doc.createNestedObject("telemetry");
    telemetry["heart_rate"] = data.heart_rate;
    telemetry["spo2"] = data.spo2;
    telemetry["temperature"] = data.ds18b20_ok ? data.temperature : 0.0;
    telemetry["ppg_amplitude"] = data.ppg_amplitude;
    telemetry["finger_detected"] = data.finger_detected;

    // Status
    const char* status_str = "NORMAL";
    if (current_status == STATUS_SIGNAL_QUALITY_LOW) status_str = "SIGNAL QUALITY LOW";
    else if (current_status == STATUS_LOW_DEVIATION) status_str = "LOW DEVIATION";
    else if (current_status == STATUS_LOW_DEVIATION_SUSTAINED) status_str = "LOW DEVIATION (SUSTAINED)";
    else if (current_status == STATUS_HIGH_DEVIATION) status_str = "HIGH DEVIATION";
    else if (current_status == STATUS_HIGH_DEVIATION_SUSTAINED) status_str = "HIGH DEVIATION (SUSTAINED)";
    doc["status"] = status_str;

    char jsonBuffer[512];
    serializeJson(doc, jsonBuffer);

    mqttClient.publish(MQTT_TOPIC_TELEMETRY, jsonBuffer);
}

bool comm_is_wifi_connected() {
    return WiFi.status() == WL_CONNECTED;
}
