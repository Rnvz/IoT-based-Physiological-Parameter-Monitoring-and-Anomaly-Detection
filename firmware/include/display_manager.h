#ifndef DISPLAY_MANAGER_H
#define DISPLAY_MANAGER_H

#include "sensors.h"

enum SystemStatus {
    NORMAL,
    CEK_SENSOR,
    ANOMALI
};

void display_init();
void display_update(const SensorData& data, SystemStatus status, bool wifi_connected);

#endif // DISPLAY_MANAGER_H
