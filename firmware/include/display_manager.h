#ifndef DISPLAY_MANAGER_H
#define DISPLAY_MANAGER_H

#include "sensors.h"

enum SystemStatus {
    STATUS_NORMAL,
    STATUS_LOW_DEVIATION,
    STATUS_LOW_DEVIATION_SUSTAINED,
    STATUS_HIGH_DEVIATION,
    STATUS_HIGH_DEVIATION_SUSTAINED,
    STATUS_SIGNAL_QUALITY_LOW
};

void display_init();
void display_update(const SensorData& data, SystemStatus status, bool wifi_connected);

#endif // DISPLAY_MANAGER_H
