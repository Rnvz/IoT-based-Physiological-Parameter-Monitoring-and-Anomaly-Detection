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

const char* MQTT_BROKER = "broker.hivemq.com";
const int MQTT_PORT = 1883;

#define TOPIC_PREFIX "physiomonitor/8009455779hsr801164603"

const char* MQTT_TOPIC = TOPIC_PREFIX "/data";

const char* TOPIC_RESULT = TOPIC_PREFIX "/result";

const char* TOPIC_CMD = TOPIC_PREFIX "/cmd";

const char* TOPIC_ACTUATOR = TOPIC_PREFIX "/actuator";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

#define MAX_SDA 21
#define MAX_SCL 22

#define DS18B20_PIN 4

#define OLED_SDA 16
#define OLED_SCL 17

#define BUZZER_PIN 25
#define BUZZER_NYALA LOW
#define BUZZER_MATI HIGH

#define DHT_PIN 14
#define DHT_TYPE DHT22

#define LED_R_PIN 26
#define LED_G_PIN 27
#define LED_B_PIN 32

TwoWire oledWire = TwoWire(1);

Adafruit_SSD1306 display(
  128,
  64,
  &oledWire,
  -1
);

MAX30105 particleSensor;

OneWire oneWire(DS18B20_PIN);
DallasTemperature sensors(&oneWire);

DHT dht(DHT_PIN, DHT_TYPE);

const byte LED_BRIGHTNESS = 0x3F;
const byte SAMPLE_AVERAGE = 4;
const byte LED_MODE = 2;
const int SAMPLE_RATE = 100;
const int PULSE_WIDTH = 411;
const int ADC_RANGE = 4096;

const float FIFO_SAMPLE_RATE = 25.0;

const long FINGER_THRESHOLD = 12000;
const unsigned long FINGER_TIMEOUT = 2000;

unsigned long lastFingerDetected = 0;
bool fingerPresent = false;

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

float suhuAktual = 0;
bool suhuValid = false;

unsigned long waktuSuhuTerakhir = 0;
unsigned long waktuRequestSuhu = 0;
bool suhuSedangDiproses = false;

const unsigned long SUHU_INTERVAL = 1000;
const unsigned long SUHU_CONVERSION_TIME = 100;

float kelembapan = 0;
float suhuRuangan = 0;
bool dhtValid = false;

unsigned long lastDHTRead = 0;

const unsigned long DHT_INTERVAL = 2000;

String statusSistem = "MENUNGGU SENSOR";

unsigned long lastOLEDUpdate = 0;
unsigned long lastMQTTPublish = 0;
unsigned long lastWiFiAttempt = 0;
unsigned long lastMQTTAttempt = 0;
unsigned long lastBuzzerTime = 0;
unsigned long buzzerOffTime = 0;

bool wifiConnecting = false;
bool mqttConnecting = false;

#define LED_PWM_FREQ 5000
#define LED_PWM_BITS 8

const uint8_t GAIN_R = 60;
const uint8_t GAIN_G = 100;
const uint8_t GAIN_B = 100;

const unsigned long MANUAL_TIMEOUT = 300000UL;

const unsigned long RESULT_TIMEOUT = 10000UL;

const unsigned long ACTUATOR_HEARTBEAT = 5000UL;

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
  {"WHITE",  255, 255, 255},
  {"PURPLE", 180, 0,   255}
};

const int COLOR_COUNT = sizeof(COLORS) / sizeof(COLORS[0]);

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

  haveResult = false;
  lastSeverity = 0;
  lastResultMs = 0;
}

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

  if (hrHistoryCount >= 3) {
    bpmFiltered = total / hrHistoryCount;
  } else {
    bpmFiltered = 0.0;
  }
}

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

void processHeartRate(uint32_t irValue) {

  bool beat = checkForBeat(irValue);

  if (!beat) {
    return;
  }

  unsigned long currentSample = totalSamples;

  Serial.print("[BEAT] Sample=");
  Serial.print(currentSample);

  if (!firstBeatDetected) {

    firstBeatDetected = true;
    lastBeatSample = currentSample;

    Serial.println(" | FIRST BEAT");

    return;
  }

  unsigned long deltaSamples = currentSample - lastBeatSample;

  Serial.print(" | DeltaSamples=");
  Serial.print(deltaSamples);

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

void calculateSpO2();

void addSpO2Sample(
  uint32_t redValue,
  uint32_t irValue
) {

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

void processMAX30102() {

  particleSensor.check();

  while (particleSensor.available()) {

    uint32_t redValue = particleSensor.getFIFORed();
    uint32_t irValue = particleSensor.getFIFOIR();

    totalSamples++;

    if (irValue > FINGER_THRESHOLD) {

      fingerPresent = true;
      lastFingerDetected = millis();

      processHeartRate(irValue);

      addSpO2Sample(redValue, irValue);
    }

    particleSensor.nextSample();
  }
}

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

void updateTemperature() {

  unsigned long now = millis();

  if (
    !suhuSedangDiproses &&
    now - waktuSuhuTerakhir >= SUHU_INTERVAL
  ) {

    sensors.requestTemperatures();

    waktuRequestSuhu = now;
    waktuSuhuTerakhir = now;

    suhuSedangDiproses = true;
  }

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

void publishActuatorStatus() {

  if (!mqttClient.connected()) {
    return;
  }

  unsigned long now = millis();

  unsigned long manualLeft = 0;

  if (ledMode == MODE_MANUAL && manualUntil > now) {
    manualLeft = (manualUntil - now) / 1000;
  }

  char buf[300];

  bool buzzerSounding = buzzerTest || (fingerPresent && haveResult && (now - lastResultMs <= RESULT_TIMEOUT) && (lastSeverity >= 2));

  snprintf(
    buf,
    sizeof(buf),
    "{\"mode\":\"%s\",\"led\":{\"color\":\"%s\",\"brightness\":%u},"
    "\"buzzer\":{\"test\":%s,\"active\":%s},"
    "\"source\":\"%s\",\"ack_seq\":%ld,\"manual_left_s\":%lu}",
    ledMode == MODE_AUTO ? "AUTO" : "MANUAL",
    currentColor,
    currentBrightness,
    buzzerTest ? "true" : "false",
    buzzerSounding ? "true" : "false",
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

  if (strcmp(topic, TOPIC_RESULT) == 0) {

    int newSev = doc["severity"] | 0;
    if (newSev != lastSeverity || !haveResult) {
      statusDirty = true;
    }
    lastSeverity = newSev;
    lastResultMs = millis();
    haveResult = true;

    return;
  }

  if (strcmp(topic, TOPIC_CMD) == 0) {

    long seq = doc["seq"] | -1L;

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

void actuatorBegin() {

  ledInit();

  setLed("OFF", 0);

  mqttClient.setCallback(actuatorOnMessage);
}

void actuatorOnMqttConnected() {

  mqttClient.subscribe(TOPIC_RESULT);
  mqttClient.subscribe(TOPIC_CMD);

  statusDirty = true;
}

void actuatorUpdate() {

  unsigned long now = millis();

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

  if (WiFi.status() != WL_CONNECTED) {

    color = "PURPLE";
    brightness = 50;
    source = "NO_WIFI";

  } else if (ledMode == MODE_MANUAL) {

    color = manualColor;
    brightness = manualBrightness;
    source = "MANUAL";

  } else if (!fingerPresent) {

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

void serviceWiFi() {

  wl_status_t state = WiFi.status();

  if (state == WL_CONNECTED) {

    wifiConnecting = false;

    return;
  }

  if (wifiConnecting) {

    if (millis() - lastWiFiAttempt > 15000) {

      Serial.println("WiFi timeout, retry...");

      WiFi.disconnect();

      wifiConnecting = false;

    } else {

      return;
    }
  }

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

    actuatorOnMqttConnected();

    mqttConnecting = false;

  } else {

    Serial.print("MQTT failed, state=");
    Serial.println(mqttClient.state());

    mqttConnecting = false;
  }
}

void updateStatus() {

  if (!fingerPresent) {

    statusSistem = "MENUNGGU SENSOR";

    return;
  }

  statusSistem = "MENDETEKSI";
}

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

  bool isModelAnomaly = fingerPresent && haveResult && (now - lastResultMs <= RESULT_TIMEOUT) && (lastSeverity >= 2);

  if (!isModelAnomaly) {

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

void updateOLED() {

  if (millis() - lastOLEDUpdate < 200) {
    return;
  }

  lastOLEDUpdate = millis();

  display.clearDisplay();

  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);

  display.setCursor(0, 0);
  display.println("PHYSIO MONITOR");

  if (WiFi.status() != WL_CONNECTED) {

    display.setCursor(0, 28);
    display.println("Trying to connect to");
    display.println("Wi-Fi...");

    display.display();

    return;
  }

  display.setCursor(0, 11);
  display.print("HR: ");

  if (bpmFiltered > 0) {
    display.print(bpmFiltered, 1);
    display.println(" BPM");
  } else {
    display.println("--");
  }

  display.setCursor(0, 21);
  display.print("SpO2: ");

  if (spo2Filtered > 0) {
    display.print(spo2Filtered, 1);
    display.println(" %");
  } else {
    display.println("--");
  }

  display.setCursor(0, 31);
  display.print("Temp: ");

  if (fingerPresent && suhuValid) {
    display.print(suhuAktual, 1);
    display.println(" C");
  } else {
    display.println("--");
  }

  display.setCursor(0, 41);
  display.print("Hum: ");

  if (dhtValid) {
    display.print(kelembapan, 1);
    display.println(" %");
  } else {
    display.println("--");
  }

  display.setCursor(0, 54);
  display.print(fingerPresent ? "MENDETEKSI" : "MENUNGGU SENSOR");

  display.display();
}

void publishMQTT() {

  if (!mqttClient.connected()) {
    return;
  }

  if (millis() - lastMQTTPublish < 1000) {
    return;
  }

  lastMQTTPublish = millis();

  String payload = "{";

  payload += "\"hr\":";
  payload += (bpmFiltered > 0) ? String(bpmFiltered, 1) : String("0.0");

  payload += ",\"spo2\":";
  payload += (spo2Filtered > 0) ? String(spo2Filtered, 1) : String("0.0");

  payload += ",\"temp\":";
  payload += suhuValid ? String(suhuAktual, 1) : String("0.0");

  payload += ",\"hum\":";
  payload += dhtValid ? String(kelembapan, 1) : String("null");

  payload += ",\"room_temp\":";
  payload += dhtValid ? String(suhuRuangan, 1) : String("null");

  payload += ",\"finger\":";
  payload += fingerPresent ? "true" : "false";

  payload += ",\"status\":\"";
  payload += statusSistem;
  payload += "\"}";

  mqttClient.publish(MQTT_TOPIC, payload.c_str());

  Serial.print("MQTT -> ");
  Serial.println(payload);
}

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

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, BUZZER_MATI);

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

  sensors.begin();
  sensors.setResolution(9);
  sensors.setWaitForConversion(false);

  dht.begin();
  lastDHTRead = millis();

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

  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);

  mqttClient.setBufferSize(512);

  actuatorBegin();

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);

  waktuSuhuTerakhir = millis() - SUHU_INTERVAL;

  Serial.println();
  Serial.println("System ready.");
  Serial.println("Tempelkan jari pada MAX30102.");
  Serial.println();
}

void loop() {

  processMAX30102();

  checkFingerTimeout();

  updateTemperature();

  updateDHT();

  updateStatus();

  actuatorUpdate();

  serviceWiFi();

  serviceMQTT();

  publishMQTT();

  updateOLED();

  updateBuzzer();

  static uint32_t lastIR = 0;
  static uint32_t lastRED = 0;

  if (particleSensor.available()) {

    lastIR = particleSensor.getFIFOIR();
    lastRED = particleSensor.getFIFORed();
  }

  printDebug(lastIR, lastRED);
}
