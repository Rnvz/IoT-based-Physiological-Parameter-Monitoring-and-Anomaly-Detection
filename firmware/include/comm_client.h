#ifndef COMM_CLIENT_H
#define COMM_CLIENT_H

#include "sensors.h"
#include "display_manager.h" // Untuk SystemStatus

// Callback untuk menerima feedback dari server
typedef void (*FeedbackCallback)(SystemStatus new_status, bool buzzer_active);

void comm_init(FeedbackCallback callback);
void comm_loop();
void comm_send_telemetry(const SensorData& data, SystemStatus current_status);
bool comm_is_wifi_connected();

#endif // COMM_CLIENT_H
