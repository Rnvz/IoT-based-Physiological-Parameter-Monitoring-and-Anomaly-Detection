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

Tampilan:
  - Format tabel terstruktur, fixed-width (109 kolom), single-row per sampel.
  - Bebas dari emoji, menggunakan teks dan ANSI color yang selaras tanpa distorsi lebar.
  - Sinkronisasi event-driven (tidak ada teks bertumpuk atau terpotong).
"""

import sys
import time
import json
import random
import argparse
import threading
from typing import Dict, Any, List, Optional
import paho.mqtt.client as mqtt

DEFAULT_BROKER = "broker.hivemq.com"
DEFAULT_PORT = 1883
DEFAULT_PREFIX = "physiomonitor/test"
DEVICE_ID = "esp32-01"

# Layout Grid Konstan (Total Lebar: 109 Karakter)
INNER_WIDTH = 105
BOX_SEP = "+" + "-" * (INNER_WIDTH + 2) + "+"
SEP = "+" + "-"*6 + "+" + "-"*10 + "+" + "-"*6 + "+" + "-"*8 + "+" + "-"*16 + "+" + "-"*8 + "+" + "-"*12 + "+" + "-"*9 + "+" + "-"*24 + "+"

c_no, c_hr, c_spo2, c_skin, c_dht, c_fin, c_sev, c_sc, c_desc = (
    "No", "HR (bpm)", "SpO2", "Skin", "DHT22 (Room)", "Finger", "Severity", "Score", "Catatan / Status"
)
HDR = f"| {c_no:^4} | {c_hr:^8} | {c_spo2:^4} | {c_skin:^6} | {c_dht:^14} | {c_fin:^6} | {c_sev:^10} | {c_sc:^7} | {c_desc:<22} |"

# Threading & State Global
connected_event = threading.Event()
result_event = threading.Event()
latest_result: Dict[str, Any] = {}
received_results: List[Dict[str, Any]] = []
received_cmds: List[Dict[str, Any]] = []


def print_banner(host: str, port: int, topic: str, prefix: str, device_id: str, base_temp: float):
    print(BOX_SEP)
    t1 = "ESP32 TELEMETRY SIMULATOR - PHYSIOMONITOR REBUILD"
    t2 = f"Broker   : {host}:{port}"
    t3 = f"Topik TX : {topic}"
    t4 = f"Topik RX : {prefix}/result, {prefix}/cmd"
    t5 = f"Perangkat: {device_id} | Suhu Kulit Dasar: {base_temp:.1f} C"
    for line in [t1, t2, t3, t4, t5]:
        print(f"| {line[:INNER_WIDTH]:<{INNER_WIDTH}} |")
    print(BOX_SEP)


def print_status_msg(msg: str):
    clean_msg = msg[:INNER_WIDTH]
    print(f"| {clean_msg:<{INNER_WIDTH}} |")


def print_section(title: str, use_color: bool = True):
    print(SEP)
    clean_title = title[:INNER_WIDTH]
    if use_color:
        print(f"| \033[1;36m{clean_title:<{INNER_WIDTH}}\033[0m |")
    else:
        print(f"| {clean_title:<{INNER_WIDTH}} |")
    print(SEP)


def print_actuator_box(mode: str, color: str, brightness: Any, buzzer: bool, seq: Any, use_color: bool = True):
    buzz_str = "ON" if buzzer else "OFF"
    text = f"AKTUATOR CMD: Mode={mode} | LED={color} ({brightness}%) | Buzzer={buzz_str} | Seq={seq} -> ACK terkirim"
    clean_text = text[:INNER_WIDTH]
    print(SEP)
    if use_color:
        print(f"| \033[1;33m{clean_text:<{INNER_WIDTH}}\033[0m |")
    else:
        print(f"| {clean_text:<{INNER_WIDTH}} |")
    print(SEP)


def print_row(
    sample_no: int,
    hr: Optional[float],
    spo2: Optional[float],
    temp: Optional[float],
    hum: Optional[float],
    room_temp: Optional[float],
    finger: bool,
    sev_str: str,
    score_str: str,
    desc: str,
    sev_code: int = 0,
    use_color: bool = True
):
    hr_str = f"{hr:5.1f}" if (hr is not None and hr > 0) else "--"
    spo2_str = f"{int(spo2):3d}%" if (spo2 is not None and spo2 > 0) else "--"
    skin_str = f"{temp:4.1f} C" if (temp is not None and temp > 0) else "--"

    if room_temp is not None and hum is not None:
        dht_str = f"{room_temp:4.1f} C / {hum:2.0f}%"
    elif room_temp is not None:
        dht_str = f"{room_temp:4.1f} C / --"
    else:
        dht_str = "-- / --"

    finger_str = "ON" if finger else "OFF"

    # Format teks severity (persis 10 karakter) sebelum dibungkus ANSI escape code
    sev_padded = f"{sev_str:<10}"
    if use_color:
        if not finger:
            colored_sev = f"\033[90m{sev_padded}\033[0m"
        elif sev_code == 0:
            colored_sev = f"\033[92m{sev_padded}\033[0m"
        elif sev_code == 1:
            colored_sev = f"\033[93m{sev_padded}\033[0m"
        elif sev_code == 2:
            colored_sev = f"\033[91m{sev_padded}\033[0m"
        else:
            colored_sev = f"\033[96m{sev_padded}\033[0m"
    else:
        colored_sev = sev_padded

    clean_desc = desc[:22]
    row = (
        f"| {sample_no:4d} | {hr_str:>8} | {spo2_str:>4} | {skin_str:^6} | "
        f"{dht_str:^14} | {finger_str:^6} | {colored_sev} | {score_str:>7} | "
        f"{clean_desc:<22} |"
    )
    print(row)


def print_summary(total_sent: int, total_res: int, total_cmd: int, elapsed_s: float):
    l1 = "RINGKASAN SIMULASI"
    l2 = f"Total Data Terkirim    : {total_sent} sampel ({elapsed_s:.1f} detik)"
    l3 = f"Total /result Diterima : {total_res} respon"
    l4 = f"Total /cmd Diterima    : {total_cmd} perintah aktuator (ACK terkonfirmasi)"
    for line in [l1, l2, l3, l4]:
        print(f"| {line[:INNER_WIDTH]:<{INNER_WIDTH}} |")
    print(BOX_SEP)


def create_mqtt_client(prefix: str, use_color: bool = True) -> mqtt.Client:
    result_topic = f"{prefix}/result"
    cmd_topic = f"{prefix}/cmd"
    actuator_topic = f"{prefix}/actuator"

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe([(result_topic, 0), (cmd_topic, 0)])
            connected_event.set()
        else:
            print_status_msg(f"[ERROR] Gagal terhubung ke broker MQTT (rc={rc})")

    def on_message(client, userdata, msg):
        global latest_result
        topic = msg.topic
        payload_str = msg.payload.decode("utf-8", errors="replace")
        try:
            data = json.loads(payload_str)
        except json.JSONDecodeError:
            return

        if topic == result_topic:
            received_results.append(data)
            latest_result = data
            result_event.set()

        elif topic == cmd_topic:
            received_cmds.append(data)
            seq = data.get("seq")
            mode = data.get("mode", "AUTO")
            color = data.get("color", "-")
            bright = data.get("brightness", "-")
            buzz = data.get("buzzer_test", False)

            # Kirim konfirmasi balik (ACK) ke <prefix>/actuator
            ack_payload = {
                "mode": mode,
                "led": {"color": color if color != "-" else "GREEN", "brightness": bright if bright != "-" else 40},
                "buzzer": {"test": buzz},
                "source": "MANUAL" if mode == "MANUAL" else "ML_RESULT",
                "ack_seq": seq,
                "manual_left_s": 300 if mode == "MANUAL" else 0
            }
            client.publish(actuator_topic, json.dumps(ack_payload))
            print_actuator_box(mode, color, bright, buzz, seq, use_color=use_color)

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
    desc: str = "",
    sample_no: int = 1,
    target_interval_s: float = 1.0,
    quiet: bool = False,
    use_color: bool = True
):
    global latest_result
    result_event.clear()
    t0 = time.time()

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

    sev_code = 0
    sev_str = "--"
    score_str = "--"

    if finger:
        # Tunggu respons dari backend FastAPI jika mode interaktif/normal (interval >= 0.2s)
        if target_interval_s >= 0.2:
            got_resp = result_event.wait(timeout=0.45)
        else:
            got_resp = result_event.is_set()

        if got_resp and latest_result:
            sev_code = latest_result.get("severity", -1)
            score = latest_result.get("score")
            score_str = f"{score:+.4f}" if score is not None else "--"
            if sev_code == 0:
                sev_str = "0 [NORMAL]"
            elif sev_code == 1:
                sev_str = "1 [WARN]"
            elif sev_code == 2:
                sev_str = "2 [ALERT]"
            else:
                sev_str = f"{sev_code} [RESP]"
        else:
            sev_code = -1
            sev_str = "-- [WAIT]" if target_interval_s >= 0.2 else "-- [FAST]"
            score_str = "--"
    else:
        # Backend tidak mengirim /result saat sensor dilepas (finger=False)
        sev_code = -2
        sev_str = "-- [OFF]"
        score_str = "--"

    if not quiet:
        print_row(
            sample_no=sample_no,
            hr=hr,
            spo2=spo2,
            temp=temp,
            hum=hum,
            room_temp=room_temp,
            finger=finger,
            sev_str=sev_str,
            score_str=score_str,
            desc=desc,
            sev_code=sev_code,
            use_color=use_color
        )

    elapsed = time.time() - t0
    sleep_time = max(0.0, target_interval_s - elapsed)
    if sleep_time > 0:
        time.sleep(sleep_time)


def run_interactive_scenarios(
    client: mqtt.Client,
    topic: str,
    device_id: str,
    base_temp: float = 33.0,
    use_color: bool = True
) -> int:
    """
    Menjalankan 5 skenario telemetri terstruktur @ 1 Hz (Total 60 sampel).
    """
    sample_no = 0

    # Skenario 1: Data Normal
    print_section("SKENARIO 1/5: Baseline Normal (35 Sampel -> Melampaui ML Warm-up 15 Sampel)", use_color=use_color)
    for i in range(35):
        sample_no += 1
        t_noise = round(base_temp + (i % 3) * 0.1, 1)
        send_data_packet(
            client, topic, device_id,
            hr=75.0 + (i % 3), spo2=98.0, temp=t_noise,
            hum=62.0, room_temp=28.5, finger=True,
            desc=f"Sampel #{i+1}",
            sample_no=sample_no,
            use_color=use_color
        )

    # Skenario 2: Sensor Lepas (Finger Off)
    print_section("SKENARIO 2/5: Finger Off / Sensor Lepas (5 Sampel -> Reset Buffer, No /result)", use_color=use_color)
    for i in range(5):
        sample_no += 1
        send_data_packet(
            client, topic, device_id,
            hr=0.0, spo2=0.0, temp=0.0,
            hum=62.0, room_temp=28.5, finger=False,
            status="MENUNGGU SENSOR",
            desc=f"Finger OFF #{i+1}",
            sample_no=sample_no,
            use_color=use_color
        )

    # Skenario 3: Anomali Transien 1 Detik
    print_section("SKENARIO 3/5: Anomali Transien 1 Detik & Pemulihan (5 Sampel)", use_color=use_color)
    sample_no += 1
    send_data_packet(
        client, topic, device_id,
        hr=145.0, spo2=88.0, temp=39.0,
        hum=60.0, room_temp=28.5, finger=True,
        desc="Spike 1 Detik",
        sample_no=sample_no,
        use_color=use_color
    )
    for i in range(4):
        sample_no += 1
        send_data_packet(
            client, topic, device_id,
            hr=75.0 + i, spo2=98.0, temp=base_temp,
            hum=60.0, room_temp=28.5, finger=True,
            desc=f"Normal Pulih #{i+1}",
            sample_no=sample_no,
            use_color=use_color
        )

    # Skenario 4: Anomali Kritis Berkelanjutan
    print_section("SKENARIO 4/5: Anomali Kritis Berkelanjutan (10 Sampel -> Severity 2 Alert)", use_color=use_color)
    for i in range(10):
        sample_no += 1
        send_data_packet(
            client, topic, device_id,
            hr=150.0 + i * 2, spo2=85.0, temp=39.5,
            hum=60.0, room_temp=28.5, finger=True,
            desc=f"Critical Alert #{i+1}",
            sample_no=sample_no,
            use_color=use_color
        )

    # Skenario 5: Pemulihan Akhir
    print_section("SKENARIO 5/5: Pemulihan Akhir ke Normal (5 Sampel)", use_color=use_color)
    for i in range(5):
        sample_no += 1
        send_data_packet(
            client, topic, device_id,
            hr=76.0, spo2=98.0, temp=base_temp,
            hum=62.0, room_temp=28.5, finger=True,
            desc=f"Recovery #{i+1}",
            sample_no=sample_no,
            use_color=use_color
        )

    return sample_no


def run_fast_1200(
    client: mqtt.Client,
    topic: str,
    device_id: str,
    base_temp: float = 33.0,
    use_color: bool = True
) -> int:
    """
    Mengirim 1.200 sampel cepat untuk verifikasi kuota dataset skripsi.
    Menampilkan progress snapshot setiap 100 sampel tanpa membanjiri terminal.
    """
    total_records = 1200
    print_section(f"MODE CEPAT: Mengirim {total_records} Sampel Telemetri (Batch Skripsi)", use_color=use_color)

    base_hr = 75.0
    base_spo2 = 98.0
    base_hum = 60.0
    base_room = 28.5

    for i in range(total_records):
        sample_no = i + 1
        if i % 100 in range(85, 90):
            hr, spo2, temp = 0.0, 0.0, 0.0
            finger = False
            status = "NO_FINGER"
        elif i % 100 in range(50, 53):
            hr = round(base_hr + random.uniform(35.0, 45.0), 1)
            spo2 = round(base_spo2 - random.uniform(6.0, 9.0), 1)
            temp = round(base_temp + random.uniform(1.2, 2.0), 1)
            finger = True
            status = "NORMAL"
        else:
            hr = round(base_hr + random.uniform(-5.0, 5.0), 1)
            spo2 = round(min(100.0, base_spo2 + random.uniform(-1.0, 1.0)), 1)
            temp = round(base_temp + random.uniform(-0.4, 0.4), 1)
            finger = True
            status = "NORMAL"

        hum = round(base_hum + random.uniform(-2.0, 2.0), 1)
        room_temp = round(base_room + random.uniform(-0.5, 0.5), 1)

        is_snapshot = (sample_no % 100 == 0) or (sample_no == 1)
        send_data_packet(
            client, topic, device_id,
            hr, spo2, temp, hum, room_temp, finger,
            status=status,
            desc=f"Progress {sample_no}/{total_records}",
            sample_no=sample_no,
            target_interval_s=0.01,
            quiet=not is_snapshot,
            use_color=use_color
        )

    return total_records


def main():
    parser = argparse.ArgumentParser(description="Simulasi ESP32 Telemetry untuk Backend Rebuild")
    parser.add_argument("--host", "--broker", default=DEFAULT_BROKER, help=f"Host broker MQTT (default: {DEFAULT_BROKER})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port broker (default: {DEFAULT_PORT})")
    parser.add_argument("--prefix", "--topic-prefix", default=DEFAULT_PREFIX, help=f"Topic prefix (default: {DEFAULT_PREFIX})")
    parser.add_argument("--device-id", default=DEVICE_ID, help=f"Device ID (default: {DEVICE_ID})")
    parser.add_argument("--base-temp", type=float, default=33.0, help="Suhu dasar kulit realistis (default: 33.0 C)")
    parser.add_argument("--fast-1200", action="store_true", help="Kirim 1.200 sampel cepat untuk verifikasi dataset")
    parser.add_argument("--no-color", action="store_true", help="Nonaktifkan warna ANSI di terminal")
    args = parser.parse_args()

    use_color = (not args.no_color) and sys.stdout.isatty()
    clean_prefix = args.prefix.rstrip("/")
    topic = f"{clean_prefix}/data"

    print_banner(
        host=args.host,
        port=args.port,
        topic=topic,
        prefix=clean_prefix,
        device_id=args.device_id,
        base_temp=args.base_temp
    )

    client = create_mqtt_client(clean_prefix, use_color=use_color)
    print_status_msg("[STATUS] Menghubungkan ke broker MQTT...")

    try:
        client.connect(args.host, args.port, keepalive=60)
    except Exception as e:
        print_status_msg(f"[ERROR] Gagal koneksi ke broker MQTT: {e}")
        print(BOX_SEP)
        sys.exit(1)

    client.loop_start()

    # Tunggu konfirmasi koneksi dan subscription selesai
    if not connected_event.wait(timeout=10.0):
        print_status_msg("[ERROR] Timeout (10s) menunggu koneksi dan subscription broker MQTT.")
        print(BOX_SEP)
        client.loop_stop()
        sys.exit(1)

    print_status_msg("[STATUS] Terhubung ke broker MQTT & subscription aktif.")
    print(SEP)
    print(HDR)

    start_time = time.time()
    total_sent = 0

    try:
        if args.fast_1200:
            total_sent = run_fast_1200(client, topic, args.device_id, base_temp=args.base_temp, use_color=use_color)
        else:
            total_sent = run_interactive_scenarios(client, topic, args.device_id, base_temp=args.base_temp, use_color=use_color)
        print(SEP)
    except KeyboardInterrupt:
        print(SEP)
        print_status_msg("[INFO] Simulasi dihentikan oleh pengguna (Ctrl+C).")
        print(BOX_SEP)
    finally:
        elapsed = time.time() - start_time
        time.sleep(0.5)
        client.loop_stop()
        client.disconnect()
        print_summary(
            total_sent=total_sent,
            total_res=len(received_results),
            total_cmd=len(received_cmds),
            elapsed_s=elapsed
        )


if __name__ == "__main__":
    main()
