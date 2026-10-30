// ============================================================
// PHYSIO MONITOR
// ESP32 + MAX30102 + DS18B20 + DHT22 + OLED SSD1306 + Buzzer
// + Lampu RGB KY-016 + MQTT
//
// Fungsi utama program:
// 1. Membaca detak jantung (HR) dan saturasi oksigen (SpO2)
//    dari sensor MAX30102.
// 2. Membaca suhu tubuh dari sensor DS18B20.
// 3. Membaca suhu dan kelembapan ruangan dari sensor DHT22.
// 4. Menampilkan hasil pengukuran pada layar OLED.
// 5. Membunyikan buzzer jika terdeteksi kondisi anomali.
// 6. Mengirim data ke broker MQTT dalam format JSON
//    setiap 1 detik.
// 7. Mengendalikan lampu RGB: mode AUTO (mengikuti hasil model
//    dari backend) dan mode MANUAL (perintah dari dashboard),
//    lalu melaporkan status lampu ke backend.
//
// Library tambahan yang dibutuhkan (Library Manager):
// - DHT sensor library + Adafruit Unified Sensor
// - ArduinoJson
// ============================================================


// ============================================================
// LIBRARY
// ============================================================

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <MAX30105.h>
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <DHT.h>
#include <ArduinoJson.h>


// ============================================================
// WIFI
// ============================================================

const char* WIFI_SSID = "iQOO 15R";
const char* WIFI_PASS = "password";


// ============================================================
// MQTT
// TOPIC_PREFIX harus unik dan sama persis dengan konfigurasi
// backend. Ganti GANTI-ID-UNIK dengan teks acak buatanmu.
// ============================================================

const char* MQTT_BROKER = "broker.hivemq.com";
const int MQTT_PORT = 1883;

#define TOPIC_PREFIX "physiomonitor/8009455779hsr801164603"

// Data sensor: ESP32 -> backend
const char* MQTT_TOPIC = TOPIC_PREFIX "/data";

// Hasil model: backend -> ESP32
const char* TOPIC_RESULT = TOPIC_PREFIX "/result";

// Perintah lampu: backend -> ESP32
const char* TOPIC_CMD = TOPIC_PREFIX "/cmd";

// Konfirmasi status lampu: ESP32 -> backend
const char* TOPIC_ACTUATOR = TOPIC_PREFIX "/actuator";

WiFiClient espClient;
PubSubClient mqttClient(espClient);


// ============================================================
// PIN
// ============================================================

// MAX30102 (I2C bus 0)
#define MAX_SDA 21
#define MAX_SCL 22

// DS18B20 (OneWire)
#define DS18B20_PIN 4

// OLED (I2C bus 1)
#define OLED_SDA 16
#define OLED_SCL 17

// Buzzer (active-low: LOW = bunyi, HIGH = diam)
#define BUZZER_PIN 25
#define BUZZER_NYALA LOW
#define BUZZER_MATI HIGH

// DHT22 (suhu dan kelembapan ruangan)
#define DHT_PIN 14
#define DHT_TYPE DHT22

// Lampu RGB KY-016 (common cathode). Verifikasi dengan wiring.
#define LED_R_PIN 26
#define LED_G_PIN 27
#define LED_B_PIN 32


// ============================================================
// OLED
// ============================================================

TwoWire oledWire = TwoWire(1);

Adafruit_SSD1306 display(
  128,
  64,
  &oledWire,
  -1
);


// ============================================================
// SENSOR OBJECT
// ============================================================

MAX30105 particleSensor;

OneWire oneWire(DS18B20_PIN);
DallasTemperature sensors(&oneWire);

DHT dht(DHT_PIN, DHT_TYPE);


// ============================================================
// MAX30102 CONFIG
// ============================================================

const byte LED_BRIGHTNESS = 0x3F;
const byte SAMPLE_AVERAGE = 4;
const byte LED_MODE = 2;
const int SAMPLE_RATE = 100;
const int PULSE_WIDTH = 411;
const int ADC_RANGE = 4096;

// Laju sampel efektif FIFO: 100 / 4 = 25 sampel/detik
const float FIFO_SAMPLE_RATE = 25.0;


// ============================================================
// FINGER DETECTION
// ============================================================

const long FINGER_THRESHOLD = 12000;
const unsigned long FINGER_TIMEOUT = 2000;

unsigned long lastFingerDetected = 0;
bool fingerPresent = false;


// ============================================================
// HEART RATE
// ============================================================

float bpmRaw = 0;
float bpmFiltered = 0;

unsigned long totalSamples = 0;
unsigned long lastBeatSample = 0;
bool firstBeatDetected = false;

#define HR_FILTER_SIZE 5

float hrHistory[HR_FILTER_SIZE] = {
  0, 0, 0, 0, 0
};

int hrHistoryIndex = 0;
int hrHistoryCount = 0;


// ============================================================
// SPO2 BUFFER
// ============================================================

#define SPO2_BUFFER_SIZE 100

uint32_t irBuffer[SPO2_BUFFER_SIZE];
uint32_t redBuffer[SPO2_BUFFER_SIZE];

int spo2BufferCount = 0;
int spo2NewSamples = 0;
bool spo2BufferReady = false;

int32_t spo2Raw = 0;
int8_t validSpO2 = 0;

#define SPO2_FILTER_SIZE 5

float spo2History[SPO2_FILTER_SIZE] = {
  0, 0, 0, 0, 0
};

int spo2HistoryIndex = 0;
int spo2HistoryCount = 0;

float spo2Filtered = 0;


// ============================================================
// TEMPERATURE (DS18B20 - suhu tubuh)
// ============================================================

float suhuAktual = 0;
bool suhuValid = false;

unsigned long waktuSuhuTerakhir = 0;
unsigned long waktuRequestSuhu = 0;
bool suhuSedangDiproses = false;

const unsigned long SUHU_INTERVAL = 1000;
const unsigned long SUHU_CONVERSION_TIME = 100;


// ============================================================
// DHT22 (suhu dan kelembapan ruangan)
// ============================================================

float kelembapan = 0;
float suhuRuangan = 0;
bool dhtValid = false;

unsigned long lastDHTRead = 0;

// DHT22 minimal 2 detik antar pembacaan
const unsigned long DHT_INTERVAL = 2000;


// ============================================================
// STATUS
// Status lokal firmware ini tetap dipakai untuk OLED dan buzzer
// (perilaku buzzer belum diubah). Keputusan anomali dari model
// diterima lewat topik result dan dipakai oleh lampu.
// Nilai: "MENUNGGU SENSOR", "MENDETEKSI", "NORMAL", "ANOMALI"
// ============================================================

String statusSistem = "MENUNGGU SENSOR";


// ============================================================
// ANOMALY THRESHOLD (dipakai buzzer lokal)
// ============================================================

const float HR_LOW = 50.0;
const float HR_HIGH = 100.0;
const float SPO2_LOW = 90.0;
const float TEMP_HIGH = 38.0;


// ============================================================
// TIMER
// ============================================================

unsigned long lastOLEDUpdate = 0;
unsigned long lastMQTTPublish = 0;
unsigned long lastWiFiAttempt = 0;
unsigned long lastMQTTAttempt = 0;
unsigned long lastBuzzerTime = 0;
unsigned long buzzerOffTime = 0;


// ============================================================
// CONNECTION STATE
// ============================================================

bool wifiConnecting = false;
bool mqttConnecting = false;


// ============================================================
// LAMPU RGB - PARAMETER
// ============================================================

#define LED_PWM_FREQ 5000
#define LED_PWM_BITS 8

// Koreksi kecerahan per kanal (persen). Pada 3.3V kanal hijau
// dan biru bisa tampak lebih redup. Sesuaikan setelah melihat
// warnanya langsung (maksimal 100).
const uint8_t GAIN_R = 60;
const uint8_t GAIN_G = 100;
const uint8_t GAIN_B = 100;

// MANUAL otomatis kembali ke AUTO setelah waktu ini
const unsigned long MANUAL_TIMEOUT = 300000UL;

// Hasil model lebih tua dari ini dianggap tidak ada
const unsigned long RESULT_TIMEOUT = 10000UL;

// Status lampu dipublish ulang tiap interval ini (heartbeat)
const unsigned long ACTUATOR_HEARTBEAT = 5000UL;


// ============================================================
// LAMPU RGB - WARNA
// ============================================================

struct NamedColor {
  const char* name;
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

const NamedColor COLORS[] = {
  {"OFF",    0,   0,   0},
  {"GREEN",  0,   255, 0},
  {"RED",    255, 0,   0},
  {"BLUE",   0,   0,   255},
  {"YELLOW", 255, 150, 0},
  {"WHITE",  255, 255, 255}
};

const int COLOR_COUNT = sizeof(COLORS) / sizeof(COLORS[0]);


// ============================================================
// LAMPU RGB - STATE
// ============================================================

enum LedMode { MODE_AUTO, MODE_MANUAL };

LedMode ledMode = MODE_AUTO;

char manualColor[12] = "GREEN";
uint8_t manualBrightness = 60;
unsigned long manualUntil = 0;

bool haveResult = false;
int lastSeverity = 0;
unsigned long lastResultMs = 0;

long lastCmdSeq = -1;

char currentColor[12] = "";
uint8_t currentBrightness = 255;
char currentSource[20] = "";

bool statusDirty = true;
unsigned long lastActuatorPublish = 0;
bool buzzerTest = false;
unsigned long buzzerTestUntil = 0;
const unsigned long BUZZER_TEST_TIMEOUT = 30000UL;

// ============================================================
// RESET VITALS
// Mengosongkan seluruh data HR dan SpO2 saat jari dilepas.
// ============================================================

void resetVitals() {

  bpmRaw = 0;
  bpmFiltered = 0;

  spo2Raw = 0;
  spo2Filtered = 0;
  validSpO2 = 0;

  for (int i = 0; i < HR_FILTER_SIZE; i++) {
    hrHistory[i] = 0;
  }

  hrHistoryIndex = 0;
  hrHistoryCount = 0;

  for (int i = 0; i < SPO2_FILTER_SIZE; i++) {
    spo2History[i] = 0;
  }

  spo2HistoryIndex = 0;
  spo2HistoryCount = 0;

  spo2BufferCount = 0;
  spo2NewSamples = 0;
  spo2BufferReady = false;

  firstBeatDetected = false;
  lastBeatSample = 0;
}


// ============================================================
// ADD HR HISTORY
// Moving average 5 nilai BPM terakhir. Nilai di luar 40-180 dibuang.
// ============================================================

void addHR(float value) {

  if (value < 40 || value > 180) {
    return;
  }

  hrHistory[hrHistoryIndex] = value;

  hrHistoryIndex++;

  if (hrHistoryIndex >= HR_FILTER_SIZE) {
    hrHistoryIndex = 0;
  }

  if (hrHistoryCount < HR_FILTER_SIZE) {
    hrHistoryCount++;
  }

  float total = 0;

  for (int i = 0; i < hrHistoryCount; i++) {
    total += hrHistory[i];
  }

  bpmFiltered = total / hrHistoryCount;
}


// ============================================================
// ADD SPO2 HISTORY
// Moving average SpO2. Nilai di luar 70-100 dibuang.
// ============================================================

void addSpO2(float value) {

  if (value < 70 || value > 100) {
    return;
  }

  spo2History[spo2HistoryIndex] = value;

  spo2HistoryIndex++;

  if (spo2HistoryIndex >= SPO2_FILTER_SIZE) {
    spo2HistoryIndex = 0;
  }

  if (spo2HistoryCount < SPO2_FILTER_SIZE) {
    spo2HistoryCount++;
  }

  float total = 0;

  for (int i = 0; i < spo2HistoryCount; i++) {
    total += spo2History[i];
  }

  spo2Filtered = total / spo2HistoryCount;
}


// ============================================================
// PROCESS HEART RATE
// Memakai sample count berdasarkan FIFO 25 Hz.
// ============================================================

void processHeartRate(uint32_t irValue) {

  bool beat = checkForBeat(irValue);

  if (!beat) {
    return;
  }

  unsigned long currentSample = totalSamples;

  Serial.print("[BEAT] Sample=");
  Serial.print(currentSample);

  // Detak pertama hanya sebagai referensi
  if (!firstBeatDetected) {

    firstBeatDetected = true;
    lastBeatSample = currentSample;

    Serial.println(" | FIRST BEAT");

    return;
  }

  unsigned long deltaSamples = currentSample - lastBeatSample;

  Serial.print(" | DeltaSamples=");
  Serial.print(deltaSamples);

  // 8 sampel = 187.5 BPM, 38 sampel = 39.5 BPM
  if (deltaSamples < 8) {

    Serial.println(" | REJECTED: TOO FAST");

    return;
  }

  if (deltaSamples > 38) {

    Serial.println(" | REJECTED: TOO SLOW");

    lastBeatSample = currentSample;

    return;
  }

  float newBPM = (60.0 * FIFO_SAMPLE_RATE) / deltaSamples;

  Serial.print(" | BPM_RAW=");
  Serial.print(newBPM, 1);

  if (newBPM >= 40 && newBPM <= 180) {

    bpmRaw = newBPM;

    addHR(newBPM);

    lastBeatSample = currentSample;

    Serial.print(" | BPM_FILTERED=");
    Serial.println(bpmFiltered, 1);

  } else {

    Serial.println(" | BPM REJECTED");
  }
}


// Deklarasi awal (dipanggil sebelum didefinisikan)
void calculateSpO2();


// ============================================================
// ADD SPO2 SAMPLE
// Setelah buffer penuh, SpO2 dihitung ulang tiap 25 sampel baru.
// ============================================================

void addSpO2Sample(
  uint32_t redValue,
  uint32_t irValue
) {

  // Tahap 1: isi buffer 100 sampel pertama
  if (!spo2BufferReady) {

    redBuffer[spo2BufferCount] = redValue;
    irBuffer[spo2BufferCount] = irValue;

    spo2BufferCount++;

    if (spo2BufferCount >= SPO2_BUFFER_SIZE) {

      spo2BufferReady = true;
      spo2NewSamples = 0;

      calculateSpO2();
    }

    return;
  }

  // Tahap 2: sliding window, geser 75 sampel terakhir ke depan
  if (spo2NewSamples == 0) {

    for (int i = 25; i < SPO2_BUFFER_SIZE; i++) {

      redBuffer[i - 25] = redBuffer[i];
      irBuffer[i - 25] = irBuffer[i];
    }
  }

  redBuffer[75 + spo2NewSamples] = redValue;
  irBuffer[75 + spo2NewSamples] = irValue;

  spo2NewSamples++;

  if (spo2NewSamples >= 25) {

    spo2NewSamples = 0;

    calculateSpO2();
  }
}


// ============================================================
// CALCULATE SPO2
// ============================================================

void calculateSpO2() {

  int32_t tempSpO2;
  int8_t tempValidSpO2;
  int32_t tempHeartRate;
  int8_t tempValidHeartRate;

  maxim_heart_rate_and_oxygen_saturation(
    irBuffer,
    SPO2_BUFFER_SIZE,
    redBuffer,
    &tempSpO2,
    &tempValidSpO2,
    &tempHeartRate,
    &tempValidHeartRate
  );

  // HR sengaja TIDAK memakai tempHeartRate (dari checkForBeat)
  if (
    tempValidSpO2 &&
    tempSpO2 >= 70 &&
    tempSpO2 <= 100
  ) {

    spo2Raw = tempSpO2;
    validSpO2 = 1;

    addSpO2((float)tempSpO2);

    Serial.print("[SpO2] RAW=");
    Serial.print(tempSpO2);
    Serial.print(" | FILTERED=");
    Serial.println(spo2Filtered, 1);

  } else {

    validSpO2 = 0;

    Serial.println("[SpO2] INVALID");
  }
}


// ============================================================
// PROCESS MAX30102 FIFO
// ============================================================

void processMAX30102() {

  particleSensor.check();

  while (particleSensor.available()) {

    uint32_t redValue = particleSensor.getFIFORed();
    uint32_t irValue = particleSensor.getFIFOIR();

    totalSamples++;

    // Deteksi jari
    if (irValue > FINGER_THRESHOLD) {

      fingerPresent = true;
      lastFingerDetected = millis();

      // HR dan SpO2 hanya dihitung jika jari menempel
      processHeartRate(irValue);

      addSpO2Sample(redValue, irValue);
    }

    particleSensor.nextSample();
  }
}


// ============================================================
// CHECK FINGER TIMEOUT
// ============================================================

void checkFingerTimeout() {

  if (!fingerPresent) {
    return;
  }

  if (millis() - lastFingerDetected > FINGER_TIMEOUT) {

    fingerPresent = false;

    resetVitals();

    statusSistem = "MENUNGGU SENSOR";
  }
}


// ============================================================
// TEMPERATURE (DS18B20) - non-blocking
// ============================================================

void updateTemperature() {

  unsigned long now = millis();

  // Tahap 1: minta konversi suhu
  if (
    !suhuSedangDiproses &&
    now - waktuSuhuTerakhir >= SUHU_INTERVAL
  ) {

    sensors.requestTemperatures();

    waktuRequestSuhu = now;
    waktuSuhuTerakhir = now;

    suhuSedangDiproses = true;
  }

  // Tahap 2: baca hasil setelah waktu konversi
  if (
    suhuSedangDiproses &&
    now - waktuRequestSuhu >= SUHU_CONVERSION_TIME
  ) {

    float temp = sensors.getTempCByIndex(0);

    if (
      temp != DEVICE_DISCONNECTED_C &&
      temp > -20 &&
      temp < 70
    ) {

      suhuAktual = temp;
      suhuValid = true;

    } else {

      suhuValid = false;
    }

    suhuSedangDiproses = false;
  }
}


// ============================================================
// DHT22 - suhu dan kelembapan ruangan
// Dibaca setiap DHT_INTERVAL (2 detik). Jika gagal, dhtValid
// menjadi false dan payload MQTT mengirim null.
// ============================================================

void updateDHT() {

  unsigned long now = millis();

  if (now - lastDHTRead < DHT_INTERVAL) {
    return;
  }

  lastDHTRead = now;

  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t) || h < 0 || h > 100) {

    dhtValid = false;

    Serial.println("[DHT22] Gagal membaca sensor");

    return;
  }

  kelembapan = h;
  suhuRuangan = t;
  dhtValid = true;
}


// ============================================================
// LAMPU RGB - OUTPUT
// Dua versi API LEDC: ESP32 core 3.x dan 2.x.
// ============================================================

const NamedColor* findColor(const char* name) {

  for (int i = 0; i < COLOR_COUNT; i++) {

    if (strcmp(COLORS[i].name, name) == 0) {
      return &COLORS[i];
    }
  }

  return nullptr;
}

void ledInit() {

#if ESP_ARDUINO_VERSION_MAJOR >= 3

  ledcAttach(LED_R_PIN, LED_PWM_FREQ, LED_PWM_BITS);
  ledcAttach(LED_G_PIN, LED_PWM_FREQ, LED_PWM_BITS);
  ledcAttach(LED_B_PIN, LED_PWM_FREQ, LED_PWM_BITS);

#else

  ledcSetup(0, LED_PWM_FREQ, LED_PWM_BITS);
  ledcSetup(1, LED_PWM_FREQ, LED_PWM_BITS);
  ledcSetup(2, LED_PWM_FREQ, LED_PWM_BITS);

  ledcAttachPin(LED_R_PIN, 0);
  ledcAttachPin(LED_G_PIN, 1);
  ledcAttachPin(LED_B_PIN, 2);

#endif
}

void ledWriteRaw(uint8_t r, uint8_t g, uint8_t b) {

#if ESP_ARDUINO_VERSION_MAJOR >= 3

  ledcWrite(LED_R_PIN, r);
  ledcWrite(LED_G_PIN, g);
  ledcWrite(LED_B_PIN, b);

#else

  ledcWrite(0, r);
  ledcWrite(1, g);
  ledcWrite(2, b);

#endif
}

// brightness 0-100 (persen)
void setLed(const char* colorName, uint8_t brightness) {

  const NamedColor* c = findColor(colorName);

  if (c == nullptr) {
    c = &COLORS[0];
  }

  if (brightness > 100) {
    brightness = 100;
  }

  uint32_t r = (uint32_t)c->r * brightness * GAIN_R / 10000;
  uint32_t g = (uint32_t)c->g * brightness * GAIN_G / 10000;
  uint32_t b = (uint32_t)c->b * brightness * GAIN_B / 10000;

  if (r > 255) r = 255;
  if (g > 255) g = 255;
  if (b > 255) b = 255;

  ledWriteRaw((uint8_t)r, (uint8_t)g, (uint8_t)b);
}


// ============================================================
// LAMPU RGB - PUBLISH STATUS
// Melaporkan perintah yang sedang dijalankan firmware, bukan
// hasil pengukuran kecerahan.
// ============================================================

void publishActuatorStatus() {

  if (!mqttClient.connected()) {
    return;
  }

  unsigned long now = millis();

  unsigned long manualLeft = 0;

  if (ledMode == MODE_MANUAL && manualUntil > now) {
    manualLeft = (manualUntil - now) / 1000;
  }

  char buf[260];

  snprintf(
    buf,
    sizeof(buf),
    "{\"mode\":\"%s\",\"led\":{\"color\":\"%s\",\"brightness\":%u},"
    "\"buzzer\":{\"test\":%s},"
    "\"source\":\"%s\",\"ack_seq\":%ld,\"manual_left_s\":%lu}",
    ledMode == MODE_AUTO ? "AUTO" : "MANUAL",
    currentColor,
    currentBrightness,
    buzzerTest ? "true" : "false",
    currentSource,
    lastCmdSeq,
    manualLeft
  );

  mqttClient.publish(TOPIC_ACTUATOR, buf);

  statusDirty = false;
  lastActuatorPublish = now;

  Serial.print("ACTUATOR -> ");
  Serial.println(buf);
}


// ============================================================
// LAMPU RGB - MQTT CALLBACK
// Menangani topik result (hasil model) dan cmd (perintah).
// ============================================================

void actuatorOnMessage(
  char* topic,
  byte* payload,
  unsigned int length
) {

#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  StaticJsonDocument<256> doc;
#endif

  DeserializationError err = deserializeJson(doc, payload, length);

  if (err) {

    Serial.print("JSON error: ");
    Serial.println(err.c_str());

    return;
  }


  // ---------------- HASIL MODEL ----------------

  if (strcmp(topic, TOPIC_RESULT) == 0) {

    lastSeverity = doc["severity"] | 0;
    lastResultMs = millis();
    haveResult = true;

    return;
  }


  // ---------------- PERINTAH ----------------

  if (strcmp(topic, TOPIC_CMD) == 0) {

    long seq = doc["seq"] | -1L;

    // Abaikan perintah duplikat (seq sama dengan yang terakhir)
    if (seq >= 0 && seq == lastCmdSeq) {
      return;
    }

    bool hasBuzzer = doc["buzzer_test"].is<bool>();

    if (hasBuzzer) {

      buzzerTest = doc["buzzer_test"].as<bool>();
      buzzerTestUntil = millis() + BUZZER_TEST_TIMEOUT;
    }

    const char* mode = doc["mode"] | "";

    if (mode[0] == '\0' && hasBuzzer) {

      lastCmdSeq = seq;
      statusDirty = true;

      return;
    }

    if (strcmp(mode, "AUTO") == 0) {

      ledMode = MODE_AUTO;

    } else if (strcmp(mode, "MANUAL") == 0) {

      const char* color = doc["color"] | manualColor;

      // Tolak warna di luar daftar
      if (findColor(color) == nullptr) {

        Serial.println("CMD ditolak: warna tidak dikenal");

        return;
      }

      int brightness = doc["brightness"] | (int)manualBrightness;

      strncpy(manualColor, color, sizeof(manualColor) - 1);
      manualColor[sizeof(manualColor) - 1] = '\0';

      manualBrightness = (uint8_t)constrain(brightness, 0, 100);

      ledMode = MODE_MANUAL;
      manualUntil = millis() + MANUAL_TIMEOUT;

    } else {

      Serial.println("CMD ditolak: mode tidak dikenal");

      return;
    }

    lastCmdSeq = seq;
    statusDirty = true;
  }
}


// ============================================================
// LAMPU RGB - SERVICE
// Dipanggil di setup() dan loop().
// ============================================================

void actuatorBegin() {

  ledInit();

  setLed("OFF", 0);

  mqttClient.setCallback(actuatorOnMessage);
}

// Langganan hilang saat koneksi putus, jadi harus diulang
// setiap MQTT berhasil terhubung kembali.
void actuatorOnMqttConnected() {

  mqttClient.subscribe(TOPIC_RESULT);
  mqttClient.subscribe(TOPIC_CMD);

  statusDirty = true;
}

void actuatorUpdate() {

  unsigned long now = millis();


  // MANUAL kedaluwarsa, kembali ke AUTO
  if (
    ledMode == MODE_MANUAL &&
    (long)(now - manualUntil) >= 0
  ) {

    ledMode = MODE_AUTO;

    statusDirty = true;
  }


  const char* color;
  uint8_t brightness;
  const char* source;


  if (ledMode == MODE_MANUAL) {

    color = manualColor;
    brightness = manualBrightness;
    source = "MANUAL";

  } else if (!fingerPresent) {

    // Kondisi teknis ditentukan lokal oleh firmware
    color = "BLUE";
    brightness = 15;
    source = "LOCAL_NO_FINGER";

  } else if (
    !haveResult ||
    now - lastResultMs > RESULT_TIMEOUT
  ) {

    color = "YELLOW";
    brightness = 40;
    source = "NO_RESULT";

  } else if (lastSeverity <= 0) {

    color = "GREEN";
    brightness = 40;
    source = "ML_RESULT";

  } else if (lastSeverity == 1) {

    color = "YELLOW";
    brightness = 70;
    source = "ML_RESULT";

  } else {

    color = "RED";
    brightness = 100;
    source = "ML_RESULT";
  }


  if (
    strcmp(color, currentColor) != 0 ||
    brightness != currentBrightness ||
    strcmp(source, currentSource) != 0
  ) {

    strncpy(currentColor, color, sizeof(currentColor) - 1);
    currentColor[sizeof(currentColor) - 1] = '\0';

    strncpy(currentSource, source, sizeof(currentSource) - 1);
    currentSource[sizeof(currentSource) - 1] = '\0';

    currentBrightness = brightness;

    setLed(color, brightness);

    statusDirty = true;
  }


  if (
    statusDirty ||
    now - lastActuatorPublish >= ACTUATOR_HEARTBEAT
  ) {

    publishActuatorStatus();
  }
}


// ============================================================
// WIFI SERVICE - non-blocking
// ============================================================

void serviceWiFi() {

  wl_status_t state = WiFi.status();

  if (state == WL_CONNECTED) {

    wifiConnecting = false;

    return;
  }

  // Sedang connecting: tunggu hingga 15 detik
  if (wifiConnecting) {

    if (millis() - lastWiFiAttempt > 15000) {

      Serial.println("WiFi timeout, retry...");

      WiFi.disconnect();

      wifiConnecting = false;

    } else {

      return;
    }
  }

  // Jeda minimal 5 detik antar percobaan
  if (millis() - lastWiFiAttempt < 5000) {

    return;
  }

  Serial.print("Connecting WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);

  WiFi.begin(WIFI_SSID, WIFI_PASS);

  lastWiFiAttempt = millis();

  wifiConnecting = true;
}


// ============================================================
// MQTT SERVICE
// ============================================================

void serviceMQTT() {

  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  if (mqttClient.connected()) {

    mqttClient.loop();

    mqttConnecting = false;

    return;
  }

  if (mqttConnecting) {
    return;
  }

  if (millis() - lastMQTTAttempt < 5000) {
    return;
  }

  Serial.println("Connecting MQTT...");

  mqttConnecting = true;

  lastMQTTAttempt = millis();

  String clientId =
    "PhysioMonitor-" +
    String((uint32_t)ESP.getEfuseMac(), HEX);

  if (mqttClient.connect(clientId.c_str())) {

    Serial.println("MQTT connected");

    // Langganan ulang topik result dan cmd
    actuatorOnMqttConnected();

    mqttConnecting = false;

  } else {

    Serial.print("MQTT failed, state=");
    Serial.println(mqttClient.state());

    mqttConnecting = false;
  }
}


// ============================================================
// STATUS (lokal, untuk OLED dan buzzer)
// ============================================================

void updateStatus() {

  if (!fingerPresent) {

    statusSistem = "MENUNGGU SENSOR";

    return;
  }

  if (bpmFiltered <= 0 || spo2Filtered <= 0) {

    statusSistem = "MENDETEKSI";

    return;
  }

  bool anomaly = false;

  if (bpmFiltered < HR_LOW || bpmFiltered > HR_HIGH) {
    anomaly = true;
  }

  if (spo2Filtered < SPO2_LOW) {
    anomaly = true;
  }

  if (suhuValid && suhuAktual > TEMP_HIGH) {
    anomaly = true;
  }

  if (anomaly) {
    statusSistem = "ANOMALI";
  } else {
    statusSistem = "NORMAL";
  }
}


// ============================================================
// BUZZER - beep 150 ms setiap 1000 ms saat ANOMALI
// (perilaku belum diubah; menunggu keputusan apakah buzzer
// mengikuti hasil model atau tetap ambang lokal)
// ============================================================

void updateBuzzer() {

  unsigned long now = millis();

  if (buzzerTest) {

    if ((long)(now - buzzerTestUntil) >= 0) {

      buzzerTest = false;
      statusDirty = true;

    } else {

      digitalWrite(BUZZER_PIN, BUZZER_NYALA);
      buzzerOffTime = 0;

      return;
    }
  }

  if (statusSistem != "ANOMALI") {

    digitalWrite(BUZZER_PIN, BUZZER_MATI);

    buzzerOffTime = 0;

    return;
  }

  if (buzzerOffTime != 0 && now >= buzzerOffTime) {

    digitalWrite(BUZZER_PIN, BUZZER_MATI);

    buzzerOffTime = 0;
  }

  if (buzzerOffTime == 0 && now - lastBuzzerTime >= 1000) {

    lastBuzzerTime = now;

    digitalWrite(BUZZER_PIN, BUZZER_NYALA);

    buzzerOffTime = now + 150;
  }
}


// ============================================================
// OLED - refresh setiap 200 ms
// Layout: judul, HR, SpO2, suhu tubuh, kelembapan, status.
// ============================================================

void updateOLED() {

  if (millis() - lastOLEDUpdate < 200) {
    return;
  }

  lastOLEDUpdate = millis();

  display.clearDisplay();

  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);

  // Title
  display.setCursor(0, 0);
  display.println("PHYSIO MONITOR");

  // HR
  display.setCursor(0, 11);
  display.print("HR: ");

  if (bpmFiltered > 0) {
    display.print(bpmFiltered, 1);
    display.println(" BPM");
  } else {
    display.println("--");
  }

  // SpO2
  display.setCursor(0, 21);
  display.print("SpO2: ");

  if (spo2Filtered > 0) {
    display.print(spo2Filtered, 1);
    display.println(" %");
  } else {
    display.println("--");
  }

  // Suhu tubuh
  display.setCursor(0, 31);
  display.print("Temp: ");

  if (suhuValid) {
    display.print(suhuAktual, 1);
    display.println(" C");
  } else {
    display.println("--");
  }

  // Kelembapan ruangan
  display.setCursor(0, 41);
  display.print("Hum: ");

  if (dhtValid) {
    display.print(kelembapan, 1);
    display.println(" %");
  } else {
    display.println("--");
  }

  // Status
  display.setCursor(0, 54);
  display.print(statusSistem);

  display.display();
}


// ============================================================
// MQTT PUBLISH - setiap 1 detik
// Contoh payload:
// {"hr":75.0,"spo2":98.0,"temp":36.5,"hum":62.3,
//  "room_temp":29.1,"finger":true,"status":"NORMAL"}
// hr/spo2/temp = 0.0 jika belum tersedia (kompatibel dashboard lama).
// hum/room_temp = null jika DHT22 gagal baca.
// Field status adalah status lokal firmware, bukan keputusan model.
// ============================================================

void publishMQTT() {

  if (!mqttClient.connected()) {
    return;
  }

  if (millis() - lastMQTTPublish < 1000) {
    return;
  }

  lastMQTTPublish = millis();

  String payload = "{";

  // HR
  payload += "\"hr\":";
  payload += (bpmFiltered > 0) ? String(bpmFiltered, 1) : String("0.0");

  // SpO2
  payload += ",\"spo2\":";
  payload += (spo2Filtered > 0) ? String(spo2Filtered, 1) : String("0.0");

  // Suhu tubuh
  payload += ",\"temp\":";
  payload += suhuValid ? String(suhuAktual, 1) : String("0.0");

  // Kelembapan ruangan
  payload += ",\"hum\":";
  payload += dhtValid ? String(kelembapan, 1) : String("null");

  // Suhu ruangan
  payload += ",\"room_temp\":";
  payload += dhtValid ? String(suhuRuangan, 1) : String("null");

  // Jari menempel atau tidak (penanda data valid untuk model)
  payload += ",\"finger\":";
  payload += fingerPresent ? "true" : "false";

  // Status
  payload += ",\"status\":\"";
  payload += statusSistem;
  payload += "\"}";

  mqttClient.publish(MQTT_TOPIC, payload.c_str());

  Serial.print("MQTT -> ");
  Serial.println(payload);
}


// ============================================================
// DEBUG SERIAL - setiap 1 detik
// ============================================================

void printDebug(uint32_t irValue, uint32_t redValue) {

  static unsigned long lastDebug = 0;

  if (millis() - lastDebug < 1000) {
    return;
  }

  lastDebug = millis();

  Serial.print("IR: ");
  Serial.print(irValue);

  Serial.print(" | RED: ");
  Serial.print(redValue);

  Serial.print(" | BPM: ");

  if (bpmFiltered > 0) {
    Serial.print(bpmFiltered, 1);
  } else {
    Serial.print("--");
  }

  Serial.print(" | SpO2: ");

  if (spo2Filtered > 0) {
    Serial.print(spo2Filtered, 1);
  } else {
    Serial.print("--");
  }

  Serial.print(" | Temp: ");

  if (suhuValid) {
    Serial.print(suhuAktual, 1);
  } else {
    Serial.print("--");
  }

  Serial.print(" | Hum: ");

  if (dhtValid) {
    Serial.print(kelembapan, 1);
  } else {
    Serial.print("--");
  }

  Serial.print(" | RoomTemp: ");

  if (dhtValid) {
    Serial.print(suhuRuangan, 1);
  } else {
    Serial.print("--");
  }

  Serial.print(" | Status: ");
  Serial.print(statusSistem);

  Serial.print(" | LED: ");
  Serial.print(ledMode == MODE_AUTO ? "AUTO " : "MANUAL ");
  Serial.print(currentColor);

  Serial.print(" | Samples: ");
  Serial.print(totalSamples);

  Serial.print(" | WiFi: ");
  Serial.print(WiFi.status() == WL_CONNECTED ? "OK" : "OFF");

  Serial.print(" | MQTT: ");
  Serial.println(mqttClient.connected() ? "OK" : "OFF");
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("========================================");
  Serial.println(" PHYSIO MONITOR");
  Serial.println(" ESP32 + MAX30102 + DS18B20 + DHT22");
  Serial.println(" HR + SpO2 + Suhu + Kelembapan + MQTT");
  Serial.println(" Lampu RGB AUTO / MANUAL");
  Serial.println("========================================");

  // Buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, BUZZER_MATI);

  // MAX30102
  Wire.begin(MAX_SDA, MAX_SCL);
  Wire.setClock(400000);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {

    Serial.println("MAX30102 tidak ditemukan!");

    while (1) {
      delay(1000);
    }
  }

  Serial.println("MAX30102 OK");

  particleSensor.setup(
    LED_BRIGHTNESS,
    SAMPLE_AVERAGE,
    LED_MODE,
    SAMPLE_RATE,
    PULSE_WIDTH,
    ADC_RANGE
  );

  particleSensor.setPulseAmplitudeRed(LED_BRIGHTNESS);
  particleSensor.setPulseAmplitudeIR(LED_BRIGHTNESS);
  particleSensor.setPulseAmplitudeGreen(0);

  // DS18B20
  sensors.begin();
  sensors.setResolution(9);
  sensors.setWaitForConversion(false);

  // DHT22
  dht.begin();
  lastDHTRead = millis();

  // OLED
  oledWire.begin(OLED_SDA, OLED_SCL, 400000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {

    Serial.println("OLED tidak ditemukan.");

  } else {

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("PHYSIO MONITOR");
    display.println();
    display.println("Initializing...");
    display.display();
  }

  // MQTT
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);

  // Payload JSON lebih panjang dari default 256 byte
  mqttClient.setBufferSize(512);

  // Lampu RGB dan callback MQTT (harus setelah setBufferSize)
  actuatorBegin();

  // WiFi
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);

  // Pembacaan suhu pertama langsung dimulai di loop pertama
  waktuSuhuTerakhir = millis() - SUHU_INTERVAL;

  Serial.println();
  Serial.println("System ready.");
  Serial.println("Tempelkan jari pada MAX30102.");
  Serial.println();
}


// ============================================================
// LOOP
// Seluruh tugas non-blocking.
// ============================================================

void loop() {

  // 1. Sensor MAX30102 (HR dan SpO2)
  processMAX30102();

  // 2. Cek jari dilepas
  checkFingerTimeout();

  // 3. Suhu tubuh DS18B20
  updateTemperature();

  // 4. Suhu dan kelembapan ruangan DHT22
  updateDHT();

  // 5. Status lokal (OLED dan buzzer)
  updateStatus();

  // 6. Lampu RGB (AUTO dari model / MANUAL dari dashboard)
  actuatorUpdate();

  // 7. WiFi
  serviceWiFi();

  // 8. MQTT
  serviceMQTT();

  // 9. Publish data sensor
  publishMQTT();

  // 10. OLED
  updateOLED();

  // 11. Buzzer
  updateBuzzer();

  // Debug: simpan nilai IR dan RED terakhir
  static uint32_t lastIR = 0;
  static uint32_t lastRED = 0;

  if (particleSensor.available()) {

    lastIR = particleSensor.getFIFOIR();
    lastRED = particleSensor.getFIFORed();
  }

  printDebug(lastIR, lastRED);
}
