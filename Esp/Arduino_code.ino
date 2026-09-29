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


// ============================================================
// WIFI
// ============================================================

const char* WIFI_SSID = "iQOO 15R";
const char* WIFI_PASS = "password";


// ============================================================
// MQTT
// ============================================================

const char* MQTT_BROKER = "broker.hivemq.com";
const int MQTT_PORT = 1883;

const char* MQTT_TOPIC = "skripsi/physiomonitor/data";

WiFiClient espClient;
PubSubClient mqttClient(espClient);


// ============================================================
// PIN
// ============================================================

// MAX30102
#define MAX_SDA 21
#define MAX_SCL 22

// DS18B20
#define DS18B20_PIN 4

// OLED
#define OLED_SDA 16
#define OLED_SCL 17

// Buzzer
#define BUZZER_PIN 25

#define BUZZER_NYALA LOW
#define BUZZER_MATI HIGH


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


// ============================================================
// MAX30102 CONFIG
// ============================================================
//
// Konfigurasi ini sengaja mengikuti kode HR kamu yang sudah
// terbukti stabil.
//

const byte LED_BRIGHTNESS = 0x3F;
const byte SAMPLE_AVERAGE = 4;
const byte LED_MODE = 2;
const int SAMPLE_RATE = 100;
const int PULSE_WIDTH = 411;
const int ADC_RANGE = 4096;


// ============================================================
// IMPORTANT
// ============================================================
//
// Karena:
// sample rate = 100 Hz
// averaging   = 4
//
// FIFO output efektif:
// 100 / 4 = 25 samples/second
//
// Ini sesuai dengan FreqS = 25 pada spo2_algorithm.h
//

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


// ------------------------------------------------------------
// HR MOVING AVERAGE
// ------------------------------------------------------------

#define HR_FILTER_SIZE 5

float hrHistory[HR_FILTER_SIZE] = {
  0, 0, 0, 0, 0
};

int hrHistoryIndex = 0;

int hrHistoryCount = 0;


// ============================================================
// SP02 BUFFER
// ============================================================

#define SPO2_BUFFER_SIZE 100

uint32_t irBuffer[SPO2_BUFFER_SIZE];
uint32_t redBuffer[SPO2_BUFFER_SIZE];

int spo2BufferCount = 0;

int spo2NewSamples = 0;

bool spo2BufferReady = false;


// ============================================================
// SP02 RAW RESULT
// ============================================================

int32_t spo2Raw = 0;

int8_t validSpO2 = 0;


// ============================================================
// SP02 FILTER
// ============================================================

#define SPO2_FILTER_SIZE 5

float spo2History[SPO2_FILTER_SIZE] = {
  0, 0, 0, 0, 0
};

int spo2HistoryIndex = 0;

int spo2HistoryCount = 0;

float spo2Filtered = 0;


// ============================================================
// TEMPERATURE
// ============================================================

float suhuAktual = 0;

bool suhuValid = false;

unsigned long waktuSuhuTerakhir = 0;

unsigned long waktuRequestSuhu = 0;

bool suhuSedangDiproses = false;

const unsigned long SUHU_INTERVAL = 1000;

const unsigned long SUHU_CONVERSION_TIME = 100;


// ============================================================
// STATUS
// ============================================================

String statusSistem = "MENUNGGU SENSOR";


// ============================================================
// ANOMALY THRESHOLD
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
// RESET VITALS
// ============================================================

void resetVitals() {

  bpmRaw = 0;

  bpmFiltered = 0;

  spo2Raw = 0;

  spo2Filtered = 0;

  validSpO2 = 0;


  // HR history
  for (int i = 0; i < HR_FILTER_SIZE; i++) {
    hrHistory[i] = 0;
  }

  hrHistoryIndex = 0;
  hrHistoryCount = 0;


  // SpO2 history
  for (int i = 0; i < SPO2_FILTER_SIZE; i++) {
    spo2History[i] = 0;
  }

  spo2HistoryIndex = 0;
  spo2HistoryCount = 0;


  // SpO2 buffer
  spo2BufferCount = 0;
  spo2NewSamples = 0;
  spo2BufferReady = false;


  // Beat
  firstBeatDetected = false;
  lastBeatSample = 0;
}


// ============================================================
// ADD HR HISTORY
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

  bpmFiltered =
    total / hrHistoryCount;
}


// ============================================================
// ADD SPO2 HISTORY
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


  // ----------------------------------------------------------
  // Average
  // ----------------------------------------------------------

  float total = 0;

  for (int i = 0; i < spo2HistoryCount; i++) {
    total += spo2History[i];
  }

  spo2Filtered =
    total / spo2HistoryCount;
}


// ============================================================
// PROCESS HEART RATE
// ============================================================
//
// IMPORTANT:
//
// Kita memakai SAMPLE COUNT berdasarkan FIFO = 25 Hz.
// Jangan gunakan 100 Hz di sini karena sampleAverage = 4.
//

void processHeartRate(uint32_t irValue) {

  bool beat = checkForBeat(irValue);


  if (!beat) {
    return;
  }


  unsigned long currentSample =
    totalSamples;


  Serial.print("[BEAT] Sample=");
  Serial.print(currentSample);


  // ----------------------------------------------------------
  // FIRST BEAT
  // ----------------------------------------------------------

  if (!firstBeatDetected) {

    firstBeatDetected = true;

    lastBeatSample = currentSample;

    Serial.println(" | FIRST BEAT");

    return;
  }


  // ----------------------------------------------------------
  // SECOND / NEXT BEAT
  // ----------------------------------------------------------

  unsigned long deltaSamples =
    currentSample - lastBeatSample;


  Serial.print(" | DeltaSamples=");
  Serial.print(deltaSamples);


  // ----------------------------------------------------------
  // Valid beat interval
  //
  // 25 Hz:
  //
  // 8 samples  = 320 ms = 187.5 BPM
  // 38 samples = 1.52 s = 39.5 BPM
  // ----------------------------------------------------------

  if (deltaSamples < 8) {

    Serial.println(" | REJECTED: TOO FAST");

    return;
  }


  if (deltaSamples > 38) {

    Serial.println(" | REJECTED: TOO SLOW");

    // tetap gunakan beat sekarang
    // sebagai referensi berikutnya
    lastBeatSample = currentSample;

    return;
  }


  // ----------------------------------------------------------
  // HITUNG BPM
  // ----------------------------------------------------------

  float newBPM =
    (60.0 * FIFO_SAMPLE_RATE) /
    deltaSamples;


  Serial.print(" | BPM_RAW=");
  Serial.print(newBPM, 1);


  if (newBPM >= 40 &&
      newBPM <= 180) {

    bpmRaw = newBPM;

    addHR(newBPM);

    lastBeatSample = currentSample;


    Serial.print(
      " | BPM_FILTERED="
    );

    Serial.println(
      bpmFiltered,
      1
    );

  } else {

    Serial.println(
      " | BPM REJECTED"
    );
  }
}


// ============================================================
// PROCESS SP02 BUFFER
// ============================================================

void addSpO2Sample(
  uint32_t redValue,
  uint32_t irValue
) {

  // ==========================================================
  // FILL INITIAL BUFFER
  // ==========================================================

  if (!spo2BufferReady) {

    redBuffer[spo2BufferCount] =
      redValue;

    irBuffer[spo2BufferCount] =
      irValue;

    spo2BufferCount++;


    // --------------------------------------------------------
    // Buffer penuh
    // --------------------------------------------------------

    if (spo2BufferCount >= SPO2_BUFFER_SIZE) {

      spo2BufferReady = true;

      spo2NewSamples = 0;


      calculateSpO2();

    }

    return;
  }


  // ==========================================================
  // BUFFER SUDAH PENUH
  //
  // Setiap 25 sample:
  // - buang 25 sample lama
  // - simpan 25 sample baru
  // ==========================================================

  if (spo2NewSamples == 0) {

    // --------------------------------------------------------
    // Shift 75 sample terakhir
    // --------------------------------------------------------

    for (
      int i = 25;
      i < SPO2_BUFFER_SIZE;
      i++
    ) {

      redBuffer[i - 25] =
        redBuffer[i];

      irBuffer[i - 25] =
        irBuffer[i];
    }
  }


  // ----------------------------------------------------------
  // Masukkan sample baru
  // ----------------------------------------------------------

  redBuffer[75 + spo2NewSamples] =
    redValue;

  irBuffer[75 + spo2NewSamples] =
    irValue;


  spo2NewSamples++;


  // ----------------------------------------------------------
  // 25 sample baru lengkap
  // ----------------------------------------------------------

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


  // ----------------------------------------------------------
  // SpO2
  //
  // Untuk HR kita sengaja TIDAK menggunakan tempHeartRate.
  // HR tetap berasal dari checkForBeat() yang sudah terbukti
  // bekerja pada sensor kamu.
  // ----------------------------------------------------------

  if (
    tempValidSpO2 &&
    tempSpO2 >= 70 &&
    tempSpO2 <= 100
  ) {

    spo2Raw =
      tempSpO2;

    validSpO2 = 1;

    addSpO2(
      (float)tempSpO2
    );


    Serial.print(
      "[SpO2] RAW="
    );

    Serial.print(
      tempSpO2
    );

    Serial.print(
      " | FILTERED="
    );

    Serial.println(
      spo2Filtered,
      1
    );

  } else {

    validSpO2 = 0;

    Serial.println(
      "[SpO2] INVALID"
    );
  }
}


// ============================================================
// PROCESS MAX30102 FIFO
// ============================================================

void processMAX30102() {

  particleSensor.check();


  while (
    particleSensor.available()
  ) {

    uint32_t redValue =
      particleSensor.getFIFORed();

    uint32_t irValue =
      particleSensor.getFIFOIR();


    // --------------------------------------------------------
    // Sample counter
    // --------------------------------------------------------

    totalSamples++;


    // --------------------------------------------------------
    // FINGER
    // --------------------------------------------------------

    if (
      irValue >
      FINGER_THRESHOLD
    ) {

      fingerPresent = true;

      lastFingerDetected =
        millis();

    }


    // --------------------------------------------------------
    // HR
    // --------------------------------------------------------

    if (irValue > FINGER_THRESHOLD) {

      processHeartRate(
        irValue
      );
    }


    // --------------------------------------------------------
    // SpO2
    // --------------------------------------------------------

    if (irValue > FINGER_THRESHOLD) {

      addSpO2Sample(
        redValue,
        irValue
      );
    }


    // --------------------------------------------------------
    // Next FIFO sample
    // --------------------------------------------------------

    particleSensor.nextSample();
  }
}


// ============================================================
// CHECK FINGER
// ============================================================

void checkFingerTimeout() {

  if (!fingerPresent) {
    return;
  }


  if (
    millis() -
    lastFingerDetected >
    FINGER_TIMEOUT
  ) {

    fingerPresent = false;

    resetVitals();

    statusSistem =
      "MENUNGGU SENSOR";
  }
}


// ============================================================
// TEMPERATURE
// ============================================================

void updateTemperature() {

  unsigned long now =
    millis();


  // ==========================================================
  // START REQUEST
  // ==========================================================

  if (
    !suhuSedangDiproses &&
    now - waktuSuhuTerakhir >= SUHU_INTERVAL
  ) {

    sensors.requestTemperatures();

    waktuRequestSuhu = now;

    waktuSuhuTerakhir = now;

    suhuSedangDiproses = true;
  }


  // ==========================================================
  // READ AFTER CONVERSION
  // ==========================================================

  if (
    suhuSedangDiproses &&
    now - waktuRequestSuhu >=
    SUHU_CONVERSION_TIME
  ) {

    float temp =
      sensors.getTempCByIndex(0);


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
// WIFI SERVICE
// ============================================================

void serviceWiFi() {

  wl_status_t state =
    WiFi.status();


  // ==========================================================
  // CONNECTED
  // ==========================================================

  if (state == WL_CONNECTED) {

    wifiConnecting = false;

    return;
  }


  // ==========================================================
  // JIKA SEDANG CONNECTING
  // ==========================================================

  if (wifiConnecting) {

    if (
      millis() -
      lastWiFiAttempt >
      15000
    ) {

      Serial.println(
        "WiFi timeout, retry..."
      );

      WiFi.disconnect();

      wifiConnecting = false;

    } else {

      return;
    }
  }


  // ==========================================================
  // RETRY
  // ==========================================================

  if (
    millis() -
    lastWiFiAttempt <
    5000
  ) {

    return;
  }


  Serial.print(
    "Connecting WiFi: "
  );

  Serial.println(
    WIFI_SSID
  );


  WiFi.mode(WIFI_STA);

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASS
  );


  lastWiFiAttempt =
    millis();

  wifiConnecting = true;
}


// ============================================================
// MQTT CONNECT
// ============================================================

void serviceMQTT() {

  if (
    WiFi.status() != WL_CONNECTED
  ) {

    return;
  }


  if (
    mqttClient.connected()
  ) {

    mqttClient.loop();

    mqttConnecting = false;

    return;
  }


  if (mqttConnecting) {

    return;
  }


  if (
    millis() -
    lastMQTTAttempt <
    5000
  ) {

    return;
  }


  Serial.println(
    "Connecting MQTT..."
  );


  mqttConnecting = true;

  lastMQTTAttempt =
    millis();


  String clientId =
    "PhysioMonitor-" +
    String(
      (uint32_t)
      ESP.getEfuseMac(),
      HEX
    );


  if (
    mqttClient.connect(
      clientId.c_str()
    )
  ) {

    Serial.println(
      "MQTT connected"
    );

    mqttConnecting = false;

  } else {

    Serial.print(
      "MQTT failed, state="
    );

    Serial.println(
      mqttClient.state()
    );

    mqttConnecting = false;
  }
}


// ============================================================
// STATUS
// ============================================================

void updateStatus() {

  if (!fingerPresent) {

    statusSistem =
      "MENUNGGU SENSOR";

    return;
  }


  // ----------------------------------------------------------
  // Belum punya HR + SpO2
  // ----------------------------------------------------------

  if (
    bpmFiltered <= 0 ||
    spo2Filtered <= 0
  ) {

    statusSistem =
      "MENDETEKSI";

    return;
  }


  // ----------------------------------------------------------
  // ANOMALY
  // ----------------------------------------------------------

  bool anomaly = false;


  if (
    bpmFiltered < HR_LOW ||
    bpmFiltered > HR_HIGH
  ) {

    anomaly = true;
  }


  if (
    spo2Filtered < SPO2_LOW
  ) {

    anomaly = true;
  }


  if (
    suhuValid &&
    suhuAktual > TEMP_HIGH
  ) {

    anomaly = true;
  }


  if (anomaly) {

    statusSistem =
      "ANOMALI";

  } else {

    statusSistem =
      "NORMAL";
  }
}


// ============================================================
// BUZZER
// ============================================================

void updateBuzzer() {

  unsigned long now =
    millis();


  // ----------------------------------------------------------
  // Jika bukan anomaly
  // ----------------------------------------------------------

  if (
    statusSistem != "ANOMALI"
  ) {

    digitalWrite(
      BUZZER_PIN,
      BUZZER_MATI
    );

    buzzerOffTime = 0;

    return;
  }


  // ----------------------------------------------------------
  // Matikan buzzer setelah 150ms
  // ----------------------------------------------------------

  if (
    buzzerOffTime != 0 &&
    now >= buzzerOffTime
  ) {

    digitalWrite(
      BUZZER_PIN,
      BUZZER_MATI
    );

    buzzerOffTime = 0;
  }


  // ----------------------------------------------------------
  // Beep setiap 1000ms
  // ----------------------------------------------------------

  if (
    buzzerOffTime == 0 &&
    now - lastBuzzerTime >= 1000
  ) {

    lastBuzzerTime = now;

    digitalWrite(
      BUZZER_PIN,
      BUZZER_NYALA
    );

    buzzerOffTime =
      now + 150;
  }
}


// ============================================================
// OLED
// ============================================================

void updateOLED() {

  if (
    millis() -
    lastOLEDUpdate <
    200
  ) {

    return;
  }


  lastOLEDUpdate =
    millis();


  display.clearDisplay();

  display.setTextColor(
    SSD1306_WHITE
  );


  // ==========================================================
  // TITLE
  // ==========================================================

  display.setTextSize(1);

  display.setCursor(0, 0);

  display.println(
    "PHYSIO MONITOR"
  );


  // ==========================================================
  // HR
  // ==========================================================

  display.setCursor(0, 12);

  display.print("HR: ");

  if (bpmFiltered > 0) {

    display.print(
      bpmFiltered,
      1
    );

    display.println(
      " BPM"
    );

  } else {

    display.println(
      "--"
    );
  }


  // ==========================================================
  // SPO2
  // ==========================================================

  display.setCursor(0, 25);

  display.print(
    "SpO2: "
  );

  if (spo2Filtered > 0) {

    display.print(
      spo2Filtered,
      1
    );

    display.println(
      " %"
    );

  } else {

    display.println(
      "--"
    );
  }


  // ==========================================================
  // TEMPERATURE
  // ==========================================================

  display.setCursor(0, 38);

  display.print(
    "Temp: "
  );

  if (suhuValid) {

    display.print(
      suhuAktual,
      1
    );

    display.println(
      " C"
    );

  } else {

    display.println(
      "--"
    );
  }


  // ==========================================================
  // STATUS
  // ==========================================================

  display.setCursor(0, 51);

  display.print(
    statusSistem
  );


  display.display();
}


// ============================================================
// MQTT PUBLISH
// ============================================================

void publishMQTT() {

  if (
    !mqttClient.connected()
  ) {

    return;
  }


  if (
    millis() -
    lastMQTTPublish <
    1000
  ) {

    return;
  }


  lastMQTTPublish =
    millis();


  // ==========================================================
  // JSON
  // ==========================================================

  String payload = "{";


  // HR
  payload += "\"hr\":";

  if (bpmFiltered > 0) {

    payload +=
      String(
        bpmFiltered,
        1
      );

  } else {

    payload += "0.0";
  }


  // SpO2
  payload += ",\"spo2\":";

  if (spo2Filtered > 0) {

    payload +=
      String(
        spo2Filtered,
        1
      );

  } else {

    payload += "0.0";
  }


  // Temperature
  payload += ",\"temp\":";

  if (suhuValid) {

    payload +=
      String(
        suhuAktual,
        1
      );

  } else {

    payload += "0.0";
  }


  // Status
  payload +=
    ",\"status\":\"";

  payload +=
    statusSistem;

  payload +=
    "\"}";


  mqttClient.publish(
    MQTT_TOPIC,
    payload.c_str()
  );


  Serial.print(
    "MQTT -> "
  );

  Serial.println(
    payload
  );
}


// ============================================================
// DEBUG SERIAL
// ============================================================

void printDebug(uint32_t irValue,
                uint32_t redValue) {

  static unsigned long lastDebug = 0;


  if (
    millis() -
    lastDebug <
    1000
  ) {

    return;
  }


  lastDebug =
    millis();


  Serial.print(
    "IR: "
  );

  Serial.print(
    irValue
  );

  Serial.print(
    " | RED: "
  );

  Serial.print(
    redValue
  );

  Serial.print(
    " | BPM: "
  );

  if (bpmFiltered > 0) {

    Serial.print(
      bpmFiltered,
      1
    );

  } else {

    Serial.print("--");
  }


  Serial.print(
    " | SpO2: "
  );

  if (spo2Filtered > 0) {

    Serial.print(
      spo2Filtered,
      1
    );

  } else {

    Serial.print("--");
  }


  Serial.print(
    " | Temp: "
  );

  if (suhuValid) {

    Serial.print(
      suhuAktual,
      1
    );

  } else {

    Serial.print("--");
  }


  Serial.print(
    " | Status: "
  );

  Serial.print(
    statusSistem
  );


  Serial.print(
    " | Samples: "
  );

  Serial.print(
    totalSamples
  );


  Serial.print(
    " | WiFi: "
  );

  Serial.print(
    WiFi.status() ==
    WL_CONNECTED
      ? "OK"
      : "OFF"
  );


  Serial.print(
    " | MQTT: "
  );

  Serial.println(
    mqttClient.connected()
      ? "OK"
      : "OFF"
  );
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);


  Serial.println();

  Serial.println(
    "========================================"
  );

  Serial.println(
    " PHYSIO MONITOR"
  );

  Serial.println(
    " ESP32 + MAX30102 + DS18B20"
  );

  Serial.println(
    " HR + SpO2 + MQTT"
  );

  Serial.println(
    "========================================"
  );


  // ==========================================================
  // BUZZER
  // ==========================================================

  pinMode(
    BUZZER_PIN,
    OUTPUT
  );

  digitalWrite(
    BUZZER_PIN,
    BUZZER_MATI
  );


  // ==========================================================
  // MAX30102 I2C
  // ==========================================================

  Wire.begin(
    MAX_SDA,
    MAX_SCL
  );

  Wire.setClock(400000);


  if (
    !particleSensor.begin(
      Wire,
      I2C_SPEED_FAST
    )
  ) {

    Serial.println(
      "MAX30102 tidak ditemukan!"
    );

    while (1) {

      delay(1000);
    }
  }


  Serial.println(
    "MAX30102 OK"
  );


  // ==========================================================
  // MAX30102 CONFIG
  // ==========================================================

  particleSensor.setup(
    LED_BRIGHTNESS,
    SAMPLE_AVERAGE,
    LED_MODE,
    SAMPLE_RATE,
    PULSE_WIDTH,
    ADC_RANGE
  );


  particleSensor.setPulseAmplitudeRed(
    LED_BRIGHTNESS
  );

  particleSensor.setPulseAmplitudeIR(
    LED_BRIGHTNESS
  );

  particleSensor.setPulseAmplitudeGreen(
    0
  );


  // ==========================================================
  // DS18B20
  // ==========================================================

  sensors.begin();

  sensors.setResolution(9);

  sensors.setWaitForConversion(false);


  // ==========================================================
  // OLED
  // ==========================================================

  oledWire.begin(
    OLED_SDA,
    OLED_SCL,
    400000
  );


  if (
    !display.begin(
      SSD1306_SWITCHCAPVCC,
      0x3C
    )
  ) {

    Serial.println(
      "OLED tidak ditemukan."
    );

  } else {

    display.clearDisplay();

    display.setTextColor(
      SSD1306_WHITE
    );

    display.setTextSize(1);

    display.setCursor(0, 0);

    display.println(
      "PHYSIO MONITOR"
    );

    display.println();

    display.println(
      "Initializing..."
    );

    display.display();
  }


  // ==========================================================
  // MQTT
  // ==========================================================

  mqttClient.setServer(
    MQTT_BROKER,
    MQTT_PORT
  );


  // ==========================================================
  // WIFI
  // ==========================================================

  WiFi.mode(WIFI_STA);

  WiFi.setAutoReconnect(true);

  WiFi.persistent(false);


  // ==========================================================
  // INITIAL TEMPERATURE
  // ==========================================================

  waktuSuhuTerakhir =
    millis() -
    SUHU_INTERVAL;


  Serial.println();

  Serial.println(
    "System ready."
  );

  Serial.println(
    "Tempelkan jari pada MAX30102."
  );

  Serial.println();
}


// ============================================================
// LOOP
// ============================================================

void loop() {

  // ==========================================================
  // 1. SENSOR
  // ==========================================================

  processMAX30102();


  // ==========================================================
  // 2. FINGER TIMEOUT
  // ==========================================================

  checkFingerTimeout();


  // ==========================================================
  // 3. TEMPERATURE
  // ==========================================================

  updateTemperature();


  // ==========================================================
  // 4. STATUS
  // ==========================================================

  updateStatus();


  // ==========================================================
  // 5. WIFI
  // ==========================================================

  serviceWiFi();


  // ==========================================================
  // 6. MQTT
  // ==========================================================

  serviceMQTT();


  // ==========================================================
  // 7. MQTT PUBLISH
  // ==========================================================

  publishMQTT();


  // ==========================================================
  // 8. OLED
  // ==========================================================

  updateOLED();


  // ==========================================================
  // 9. BUZZER
  // ==========================================================

  updateBuzzer();


  // ==========================================================
  // DEBUG
  // ==========================================================

  static uint32_t lastIR = 0;
  static uint32_t lastRED = 0;

  if (particleSensor.available()) {

    lastIR =
      particleSensor.getFIFOIR();

    lastRED =
      particleSensor.getFIFORed();
  }

  printDebug(
    lastIR,
    lastRED
  );
}