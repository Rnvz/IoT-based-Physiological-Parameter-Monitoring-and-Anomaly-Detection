#!/usr/bin/env python3
"""
Simulasi Pengiriman Data Telemetri ESP32 ke Broker MQTT
Script ini mensimulasikan firmware ESP32 yang terhubung ke broker MQTT (HiveMQ)
dan berkomunikasi dengan backend FastAPI, Rule Engine, dan ML Anomaly Detector.

Kontrak Topik:
  - Publish   : <prefix>/data
  - Subscribe : <prefix>/result
  - Subscribe : <prefix>/cmd
  - Publish   : <prefix>/actuator (konfirmasi status aktuator)

Fitur Pengujian:
  1. Pengiriman data normal (vitals stabil, DHT22 aktif).
  2. Pengiriman sensor lepas / finger=False (hr/spo2/temp=0.0).
  3. DHT22 sensor failure (hum & room_temp = None / null).
  4. Anomali transien vs anomali persisten.
  5. Konfirmasi perintah /cmd dan publish ke /actuator dengan ack_seq.
  6. Mode --fast-1200: Mengirim 1.200 sampel cepat untuk verifikasi kuota dataset skripsi.
"""

import sys
import time
import json
import random
import argparse
from typing import Dict, Any, List, Optional
import paho.mqtt.client as mqtt

DEFAULT_BROKER = "broker.hivemq.com"
DEFAULT_PORT = 1883
DEFAULT_PREFIX = "physiomonitor/test"
DEVICE_ID = "esp32-01"

received_results: List[Dict[str, Any]] = []
received_cmds: List[Dict[str, Any]] = []

def create_mqtt_client(prefix: str):
    data_topic = f"{prefix}/data"
    result_topic = f"{prefix}/result"
    cmd_topic = f"{prefix}/cmd"
    actuator_topic = f"{prefix}/actuator"

    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"[*] Terhubung ke broker MQTT (rc={rc})")
        client.subscribe([(result_topic, 0), (cmd_topic, 0)])
        print(f"[*] Berlangganan topik: {result_topic} dan {cmd_topic}\n")

    def on_message(client, userdata, msg):
        topic = msg.topic
        payload_str = msg.payload.decode("utf-8", errors="replace")
        try:
            data = json.loads(payload_str)
        except json.JSONDecodeError:
            print(f"  └── [Rx {topic}] Non-JSON payload: {payload_str}")
            return

        if topic == result_topic:
            received_results.append(data)
            label = data.get("label", "UNKNOWN")
            sev = data.get("severity", -1)
            score = data.get("score")
            score_str = f"{score:+.4f}" if score is not None else "None"
            ver = data.get("model_version", "-")
            color_tag = "\033[92m" if sev == 0 else ("\033[93m" if sev == 1 else "\033[91m")
            reset = "\033[0m"
            print(f"  └── 📩 [RECV /result] {color_tag}Label={label:<8} | Severity={sev} | Score={score_str:<8} | Ver={ver}{reset}")

        elif topic == cmd_topic:
            received_cmds.append(data)
            seq = data.get("seq")
            mode = data.get("mode", "AUTO")
            color = data.get("color", "-")
            bright = data.get("brightness", "-")
            buzz = data.get("buzzer_test", False)
            print(f"  └── ⚙️ [RECV /cmd] Seq={seq} | Mode={mode} | Color={color} | Brightness={bright} | BuzzerTest={buzz}")

            # Kirim konfirmasi balik ke <prefix>/actuator
            ack_payload = {
                "mode": mode,
                "led": {"color": color if color != "-" else "GREEN", "brightness": bright if bright != "-" else 40},
                "buzzer": {"test": buzz},
                "source": "MANUAL" if mode == "MANUAL" else "ML_RESULT",
                "ack_seq": seq,
                "manual_left_s": 300 if mode == "MANUAL" else 0
            }
            client.publish(actuator_topic, json.dumps(ack_payload))
            print(f"      └── 🚀 [ACK /actuator] Sent confirmation for seq {seq}")

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    else:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message
    return client

def send_data_packet(
    client: mqtt.Client,
    topic: str,
    device_id: str,
    hr: Optional[float],
    spo2: Optional[float],
    temp: Optional[float],
    hum: Optional[float],
    room_temp: Optional[float],
    finger: bool,
    status: str = "NORMAL",
    desc: str = ""
):
    payload = {
        "device_id": device_id,
        "hr": hr if hr is not None else 0.0,
        "spo2": spo2 if spo2 is not None else 0.0,
        "temp": temp if temp is not None else 0.0,
        "hum": hum,
        "room_temp": room_temp,
        "finger": finger,
        "status": status
    }
    client.publish(topic, json.dumps(payload))
    hr_str = f"{hr:5.1f}" if hr is not None else "  -- "
    spo2_str = f"{spo2:4.0f}%" if spo2 is not None else " --%"
    temp_str = f"{temp:4.1f}°C" if temp is not None else " --.-"
    desc_str = f" ({desc})" if desc else ""
    print(f"[*] 🚀 [SEND /data] HR={hr_str} | SpO2={spo2_str} | Temp={temp_str} | Finger={'YES' if finger else 'NO '}{desc_str}")

def run_fast_1200(client: mqtt.Client, topic: str, device_id: str, base_temp: float = 33.0):
    """
    Mengirim 1.200 sampel data fisiologis dengan variasi realistis.
    Memenuhi kebutuhan minimal 1.000 baris dataset untuk skripsi.
    """
    total_records = 1200
    print("\n" + "=" * 75)
    print(f"  MODE CEPAT: MENGIRIM {total_records} SAMPEL KE {topic} (Suhu Kulit Dasar: {base_temp}°C)")
    print("=" * 75)

    base_hr = 75.0
    base_spo2 = 98.0
    base_hum = 60.0
    base_room = 28.5

    start_time = time.time()
    for i in range(total_records):
        # 1. 90% waktu: data fisiologis normal dengan sedikit variasi alami
        if i % 100 in range(85, 90):
            # 5 sampel jari lepas (finger=False)
            hr, spo2, temp = 0.0, 0.0, 0.0
            finger = False
            status = "NO_FINGER"
        elif i % 100 in range(50, 53):
            # 3 sampel anomali transien (HR tinggi / SpO2 turun sedikit)
            hr = round(base_hr + random.uniform(35.0, 45.0), 1)
            spo2 = round(base_spo2 - random.uniform(6.0, 9.0), 1)
            temp = round(base_temp + random.uniform(1.2, 2.0), 1)
            finger = True
            status = "NORMAL"
        elif i % 150 == 0:
            # DHT22 sensor glitch (null humidity & room_temp)
            hr = round(base_hr + random.uniform(-4.0, 4.0), 1)
            spo2 = round(min(100.0, base_spo2 + random.uniform(-1.0, 1.0)), 1)
            temp = round(base_temp + random.uniform(-0.3, 0.3), 1)
            finger = True
            status = "NORMAL"
            send_data_packet(client, topic, device_id, hr, spo2, temp, None, None, finger, status)
            continue
        else:
            # Normal
            hr = round(base_hr + random.uniform(-5.0, 5.0), 1)
            spo2 = round(min(100.0, base_spo2 + random.uniform(-1.0, 1.0)), 1)
            temp = round(base_temp + random.uniform(-0.4, 0.4), 1)
            finger = True
            status = "NORMAL"

        hum = round(base_hum + random.uniform(-2.0, 2.0), 1)
        room_temp = round(base_room + random.uniform(-0.5, 0.5), 1)

        send_data_packet(client, topic, device_id, hr, spo2, temp, hum, room_temp, finger, status)
        if (i + 1) % 100 == 0:
            print(f"  └── Progress: {i + 1}/{total_records} sampel terkirim...")
        time.sleep(0.01)  # Interval cepat 10ms per record

    elapsed = time.time() - start_time
    print(f"\n[✓] Selesai mengirim {total_records} rekor dalam {elapsed:.2f} detik (~{total_records/elapsed:.1f} msg/s)!\n")

def run_interactive_scenarios(client: mqtt.Client, topic: str, device_id: str, base_temp: float = 33.0, duration_s: int = 60):
    """
    Menjalankan 5 skenario terstruktur @ 1 Hz selama ~60 detik untuk pengujian normal.
    """
    print(f"\n--- [SKENARIO 1] Data Normal (35 Sampel -> Melampaui ML Warmup 15 Sampel) ---")
    for i in range(35):
        t_noise = round(base_temp + (i % 3) * 0.1, 1)
        send_data_packet(
            client, topic, device_id,
            hr=75.0 + (i % 3), spo2=98.0, temp=t_noise,
            hum=62.0, room_temp=28.5, finger=True,
            desc=f"Sampel #{i+1}"
        )
        time.sleep(1.0)

    print("\n--- [SKENARIO 2] Finger Off / Sensor Lepas (5 Sampel -> Reset Buffer, No /result) ---")
    for i in range(5):
        send_data_packet(
            client, topic, device_id,
            hr=0.0, spo2=0.0, temp=0.0,
            hum=62.0, room_temp=28.5, finger=False,
            status="MENUNGGU SENSOR", desc=f"Finger OFF #{i+1}"
        )
        time.sleep(1.0)

    print("\n--- [SKENARIO 3] Anomali Transien 1 Detik & Pemulihan (5 Sampel) ---")
    send_data_packet(
        client, topic, device_id,
        hr=145.0, spo2=88.0, temp=39.0,
        hum=60.0, room_temp=28.5, finger=True,
        desc="Spike 1 Detik"
    )
    time.sleep(1.0)
    for i in range(4):
        send_data_packet(
            client, topic, device_id,
            hr=75.0 + i, spo2=98.0, temp=base_temp,
            hum=60.0, room_temp=28.5, finger=True,
            desc=f"Normal Pulih #{i+1}"
        )
        time.sleep(1.0)

    print("\n--- [SKENARIO 4] Anomali Kritis Berkelanjutan (10 Sampel -> Severity 2) ---")
    for i in range(10):
        send_data_packet(
            client, topic, device_id,
            hr=150.0 + i * 2, spo2=85.0, temp=39.5,
            hum=60.0, room_temp=28.5, finger=True,
            desc=f"Critical Anomaly #{i+1}"
        )
        time.sleep(1.0)

    print("\n--- [SKENARIO 5] Pemulihan Kembali Normal (5 Sampel) ---")
    for i in range(5):
        send_data_packet(
            client, topic, device_id,
            hr=76.0, spo2=98.0, temp=base_temp,
            hum=62.0, room_temp=28.5, finger=True,
            desc=f"Recovery #{i+1}"
        )
        time.sleep(1.0)

def main():
    parser = argparse.ArgumentParser(description="Simulasi ESP32 Telemetry untuk Backend Rebuild")
    parser.add_argument("--host", "--broker", default=DEFAULT_BROKER, help=f"Host broker MQTT (default: {DEFAULT_BROKER})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port broker (default: {DEFAULT_PORT})")
    parser.add_argument("--prefix", "--topic-prefix", default=DEFAULT_PREFIX, help=f"Topic prefix (default: {DEFAULT_PREFIX})")
    parser.add_argument("--device-id", default=DEVICE_ID, help=f"Device ID (default: {DEVICE_ID})")
    parser.add_argument("--base-temp", type=float, default=33.0, help="Suhu dasar kulit realistis (default: 33.0 C)")
    parser.add_argument("--fast-1200", action="store_true", help="Kirim 1.200 sampel cepat untuk verifikasi dataset")
    args = parser.parse_args()

    topic = f"{args.prefix.rstrip('/')}/data"
    client = create_mqtt_client(args.prefix.rstrip('/'))

    print("=" * 75)
    print(f"  ESP32 SIMULATOR (PHYSIOMONITOR REBUILD)")
    print(f"  Broker   : {args.host}:{args.port}")
    print(f"  Data     : {topic}")
    print(f"  Device   : {args.device_id}")
    print(f"  Base Temp: {args.base_temp}°C")
    print("=" * 75)

    try:
        client.connect(args.host, args.port, keepalive=60)
    except Exception as e:
        print(f"[!] Gagal koneksi ke broker MQTT: {e}")
        sys.exit(1)

    client.loop_start()
    time.sleep(1.5)  # Tunggu koneksi terjalin

    try:
        if args.fast_1200:
            run_fast_1200(client, topic, args.device_id, base_temp=args.base_temp)
        else:
            run_interactive_scenarios(client, topic, args.device_id, base_temp=args.base_temp)
    except KeyboardInterrupt:
        print("\n[*] Simulasi dihentikan pengguna.")
    finally:
        time.sleep(1.0)
        client.loop_stop()
        client.disconnect()
        print(f"\n[*] Selesai. Total /result diterima: {len(received_results)}, /cmd diterima: {len(received_cmds)}")

if __name__ == "__main__":
    main()
