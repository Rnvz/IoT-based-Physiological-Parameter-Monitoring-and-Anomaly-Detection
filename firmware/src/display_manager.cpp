#include "display_manager.h"
#include "config.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);

void display_init() {
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        Serial.println("Gagal menginisialisasi OLED!");
        return;
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("PHYSIO MONITOR");
    display.println("Memulai...");
    display.display();
}

void display_update(const SensorData& data, SystemStatus status, bool wifi_connected) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    
    // Header dan Indikator WiFi
    display.setCursor(0, 0);
    display.print("PHYSIO MONITOR");
    display.setCursor(OLED_WIDTH - 15, 0);
    if (wifi_connected) {
        display.print("W+"); // Terhubung
    } else {
        display.print("W-"); // Terputus
    }

    display.setCursor(0, 10);
    display.print("--------------------");

    display.setCursor(0, 20);
    if (!data.finger_detected) {
        display.println("HR     : -- BPM");
        display.println("SpO2   : -- %");
    } else {
        display.print("HR     : ");
        display.print((int)data.heart_rate);
        display.println(" BPM");

        display.print("SpO2   : ");
        display.print((int)data.spo2);
        display.println(" %");
    }

    display.setCursor(0, 40);
    display.print("TEMP   : ");
    if (data.ds18b20_ok) {
        display.print(data.temperature, 1);
        display.println(" C");
    } else {
        display.println("--.- C");
    }

    // Tampilkan Status
    display.setCursor(0, 52);
    display.print("STATUS : ");
    switch (status) {
        case NORMAL:
            display.println("NORMAL");
            break;
        case CEK_SENSOR:
            display.println("CEK SENSOR");
            break;
        case ANOMALI:
            display.println("ANOMALI");
            break;
    }

    display.display();
}
