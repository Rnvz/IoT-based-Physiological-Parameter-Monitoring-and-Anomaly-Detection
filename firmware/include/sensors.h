#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>

// Struktur data sensor
struct SensorData {
    float heart_rate;
    float spo2;
    float temperature;
    bool finger_detected;
    bool max30102_ok;
    bool ds18b20_ok;
    uint32_t ppg_amplitude;
};

// Deklarasi fungsi manajer sensor
bool sensors_init();
bool read_max30102(float &hr, float &spo2, bool &finger_detected, uint32_t &ppg_amplitude);
bool read_ds18b20(float &temperature);
SensorData read_all_sensors();

#endif // SENSORS_H
