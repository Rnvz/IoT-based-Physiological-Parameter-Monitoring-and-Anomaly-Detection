#!/usr/bin/env python3
"""
Simulasi Pengiriman Data Telemetri ESP32 ke Broker MQTT
Script ini menguji integrasi end-to-end antara layer firmware (simulasi)
dengan backend FastAPI & model ML Isolation Forest tanpa memerlukan perangkat keras fisik.

4 Skenario Pengujian:
1. Normal Stream (5 detik): Menguji status fisiologis normal.
2. SQA Noise / Finger Off (3 detik): Menguji filter kualitas sinyal.
3. Anomali Transien 1 Detik (Uji Strict Reset Debounce): Memastikan derau sesaat diredam.
4. Anomali Persisten (4 detik): Memastikan alarm aktif pada sampel ke-3 berturut-turut
   dan langsung mati saat kembali normal.
"""

import sys
import time
import json
import argparse
from typing import Dict, Any, List
import paho.mqtt.client as mqtt

# Konfigurasi Default
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1883
DEVICE_ID = "esp32_sensor_01"
TELEMETRY_TOPIC = f"physio/{DEVICE_ID}/telemetry"
FEEDBACK_TOPIC = f"physio/{DEVICE_ID}/feedback"

# Penyimpanan feedback yang diterima
received_feedbacks: List[Dict[str, Any]] = []


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[*] Terhubung ke broker MQTT (rc={rc})")
    client.subscribe(FEEDBACK_TOPIC)
    print(f"[*] Berlangganan topik feedback: {FEEDBACK_TOPIC}\n")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        received_feedbacks.append(data)
        seq = data.get("reply_sequence_id", "-")
        status = data.get("status", "unknown")
        buzz = data.get("buzzer_active", False)
        sqa = data.get("sqa_status", "-")
        score = data.get("anomaly_score", 0.0)
        display = data.get("display", {}).get("line_status", "-")

        buzz_str = "🔊 [BUZZER AKTIF!]" if buzz else "🔇 [Buzzer Mati]"
        print(f"  └── 📩 [RECV FEEDBACK] Seq={seq:2} | Status={status:<26} | SQA={sqa:<12} | Score={score:+.3f} | OLED={display:<12} | {buzz_str}")
    except Exception as e:
        print(f"  └── ⚠️ Gagal decode feedback: {e}")


def send_telemetry(client: mqtt.Client, topic: str, seq_id: int, hr: float, spo2: float, temp: float, amp: float, finger: bool, is_flat: bool = False, scenario_desc: str = ""):
    if is_flat:
        # Format persis seperti Esp/Arduino_code.ino
        status_str = "MENUNGGU_SENSOR" if not finger or hr <= 0 else ("ANOMALI" if hr < 50 or hr > 100 or temp >= 38.0 else "NORMAL")
        payload = {
            "hr": round(hr, 1),
            "spo2": round(spo2, 1),
            "temp": round(temp, 1),
            "status": status_str
        }
    else:
        payload = {
            "device_id": DEVICE_ID,
            "sequence_id": seq_id,
            "timestamp_ms": int(time.time() * 1000),
            "telemetry": {
                "heart_rate": round(hr, 1),
                "spo2": round(spo2, 1),
                "temperature": round(temp, 1),
                "ppg_amplitude": round(amp, 1),
                "finger_detected": finger
            },
            "status": "normal"
        }
    
    desc_str = f" ({scenario_desc})" if scenario_desc else ""
    print(f"[*] 🚀 [SEND] Seq={seq_id:2} | HR={hr:5.1f} bpm | SpO2={spo2:5.1f}% | Temp={temp:4.1f}°C | Finger={'YES' if finger else 'NO '}{desc_str}")
    client.publish(topic, json.dumps(payload))



def run_simulation(host: str, port: int, topic: str = TELEMETRY_TOPIC, is_flat: bool = False):
    print("=" * 75)
    print(f"  SIMULASI PENGUJIAN INTEGRASI FIRMWARE ESP32 -> MQTT -> ML BACKEND")
    print(f"  Broker Target: {host}:{port}")
    print(f"  Topik Publish: {topic}")
    print(f"  Format Mode  : {'Flat JSON (Arduino_code.ino)' if is_flat else 'Canonical Hierarchical JSON'}")
    print("=" * 75 + "\n")

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    else:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(host, port, keepalive=60)
    except Exception as e:
        print(f"[!] GAGAL terhubung ke broker MQTT di {host}:{port}: {e}")
        print("    Pastikan broker MQTT aktif dan dapat diakses.")
        sys.exit(1)

    client.loop_start()
    time.sleep(1.0)  # Beri jeda untuk koneksi broker

    seq = 1

    # --------------------------------------------------------------------------
    # SKENARIO 1: Normal Stream (5 Sampel)
    # --------------------------------------------------------------------------
    print("--- [SKENARIO 1] Data Fisiologis Normal (5 Sampel @ 1 Hz) ---")
    for _ in range(5):
        send_telemetry(client, topic, seq, hr=75.0, spo2=98.0, temp=36.5, amp=2500.0, finger=True, is_flat=is_flat, scenario_desc="Normal")
        seq += 1
        time.sleep(1.0)

    # --------------------------------------------------------------------------
    # SKENARIO 2: SQA Noise / Finger Off (3 Sampel)
    # --------------------------------------------------------------------------
    print("\n--- [SKENARIO 2] Filter Kualitas Sinyal SQA / Sensor Lepas (3 Sampel) ---")
    for _ in range(3):
        send_telemetry(client, topic, seq, hr=0.0, spo2=98.0 if is_flat else 0.0, temp=36.5, amp=0.0, finger=False, is_flat=is_flat, scenario_desc="Finger OFF")
        seq += 1
        time.sleep(1.0)

    # --------------------------------------------------------------------------
    # SKENARIO 3: Derau Anomali Transien / 1 Sampel (Uji Strict Reset Debounce)
    # --------------------------------------------------------------------------
    print("\n--- [SKENARIO 3] Anomali Transien 1 Detik (Uji Peredaman Derau/Debounce) ---")
    # 1 Sampel anomali tajam
    send_telemetry(client, topic, seq, hr=140.0, spo2=88.0, temp=39.0, amp=2000.0, finger=True, is_flat=is_flat, scenario_desc="Spike Terisolasi")
    seq += 1
    time.sleep(1.0)
    # Langsung kembali normal (counter harus langsung reset ke 0, buzzer TIDAK boleh aktif)
    send_telemetry(client, topic, seq, hr=74.0, spo2=98.5, temp=36.6, amp=2500.0, finger=True, is_flat=is_flat, scenario_desc="Kembali Normal")
    seq += 1
    time.sleep(1.0)

    # --------------------------------------------------------------------------
    # SKENARIO 4: Anomali Persisten (11 Sampel -> Uji Status SUSTAINED >= 10 Sampel) & Pemulihan
    # --------------------------------------------------------------------------
    print("\n--- [SKENARIO 4] Anomali Persisten (11 Sampel -> Uji Status SUSTAINED >= 10 Sampel) ---")
    for i in range(11):
        send_telemetry(client, topic, seq, hr=145.0 + i * 1, spo2=86.0, temp=39.5, amp=1800.0, finger=True, is_flat=is_flat, scenario_desc=f"Deviasi Persisten #{i+1}")
        seq += 1
        time.sleep(1.0)

    # 1 Sampel pemulihan normal (Strict Reset)
    send_telemetry(client, topic, seq, hr=75.0, spo2=98.0, temp=36.5, amp=2400.0, finger=True, is_flat=is_flat, scenario_desc="Pemulihan Normal (Strict Reset)")
    seq += 1
    time.sleep(1.5)

    client.loop_stop()
    client.disconnect()

    print("\n" + "=" * 75)
    print("  HASIL EVALUASI SIMULASI")
    print(f"  Total Telemetri Terkirim: {seq - 1}")
    print(f"  Total Feedback Diterima : {len(received_feedbacks)}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulasi Perangkat ESP32 MQTT Telemetry")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host MQTT broker (default: localhost atau broker.hivemq.com)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port MQTT broker (default: 1883)")
    parser.add_argument("--topic", default=TELEMETRY_TOPIC, help="Topik MQTT untuk publish data")
    parser.add_argument("--flat", action="store_true", help="Gunakan format Flat JSON persis seperti Esp/Arduino_code.ino")
    args = parser.parse_args()

    run_simulation(args.host, args.port, topic=args.topic, is_flat=args.flat)

