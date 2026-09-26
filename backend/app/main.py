from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import paho.mqtt.client as mqtt

from app.core.config import settings
from app.schemas.telemetry import (
    TelemetryPayload,
    FeedbackPayload,
    FeedbackDisplay,
    DashboardPayload,
)
from app.services.sqa import SignalQualityAssessor
from app.services.feature_fusion import FeatureFusionEngine
from app.services.anomaly_model import AnomalyDetector, PersistentAnomalyEvaluator

logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Global instances
sqa_service = SignalQualityAssessor()
feature_engine = FeatureFusionEngine()
anomaly_detector: Optional[AnomalyDetector] = None
persistent_evaluator = PersistentAnomalyEvaluator()
mqtt_client: Optional[mqtt.Client] = None
main_loop: Optional[asyncio.AbstractEventLoop] = None
last_telemetry: Dict[str, TelemetryPayload] = {}
last_arrival_time: Dict[str, float] = {}
msg_counter: int = 0

TABLE_HEADER = (
    f"\n\033[1m\033[96m  WAKTU        INTERVAL   HR INSTAN    d_HR     MA_HR    VAR_HR   SpO2    SUHU     SQA           STATUS           HASIL MODEL ML      BUZZER\033[0m\n"
    f"\033[2m " + "-" * 130 + "\033[0m"
)




class ConnectionManager:
    def __init__(self):
        self.device_connections: List[WebSocket] = []
        self.dashboard_connections: List[WebSocket] = []

    async def connect_device(self, websocket: WebSocket):
        await websocket.accept()
        self.device_connections.append(websocket)

    def disconnect_device(self, websocket: WebSocket):
        if websocket in self.device_connections:
            self.device_connections.remove(websocket)

    async def connect_dashboard(self, websocket: WebSocket):
        await websocket.accept()
        self.dashboard_connections.append(websocket)

    def disconnect_dashboard(self, websocket: WebSocket):
        if websocket in self.dashboard_connections:
            self.dashboard_connections.remove(websocket)

    async def broadcast_to_dashboards(self, message: dict):
        for connection in list(self.dashboard_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Dashboard WS error: {e}")
                self.disconnect_dashboard(connection)


manager = ConnectionManager()


def process_telemetry(payload_dict: dict, publish_mqtt: bool = False) -> Tuple[FeedbackPayload, DashboardPayload]:
    """
    Pipeline pemrosesan terpadu:
    1. Parse & validasi TelemetryPayload
    2. Evaluasi kualitas sinyal (SQA)
    3. Ekstraksi fitur temporal (Feature Fusion 13-dim)
    4. Evaluasi model ML (Isolation Forest + Calibrated Threshold)
    5. Evaluasi debounce anomali persisten (Strict Reset)
    6. Pembuatan respons feedback perangkat & broadcast dashboard
    """
    telemetry = TelemetryPayload.model_validate(payload_dict)
    device_id = telemetry.device_id

    # 1. SQA Assessment
    prev_telemetry = last_telemetry.get(device_id)
    sqa_result = sqa_service.assess(telemetry, prev_telemetry)
    sqa_status = sqa_result.state

    # 2. Feature Fusion
    features = feature_engine.process(telemetry)

    # 3. Anomaly Detection & Strict Reset Debounce
    is_anomaly = False
    anomaly_score = 0.0
    should_buzz = False

    if sqa_status == "GOOD" and features is not None and anomaly_detector is not None:
        anom_result = anomaly_detector.predict(features)
        eval_result = persistent_evaluator.evaluate(anom_result.is_anomaly, anom_result.severity)
        is_anomaly = eval_result.is_persistent_anomaly
        anomaly_score = anom_result.anomaly_score
        should_buzz = eval_result.should_buzz
        status = eval_result.severity
        line_status = eval_result.oled_status
    else:
        # Jika SQA POOR atau buffer fitur belum penuh, redam alarm
        should_buzz = False
        if sqa_status != "GOOD":
            status = "SIGNAL QUALITY LOW"
            line_status = "SIG QUAL LOW"
        else:
            status = "NORMAL"
            line_status = "NORMAL"

    # Simpan telemetri terakhir
    last_telemetry[device_id] = telemetry

    seq_id = telemetry.sequence_id if telemetry.sequence_id is not None else telemetry.network.sequence_id

    feedback = FeedbackPayload(
        device_id=device_id,
        reply_sequence_id=seq_id,
        status=status,
        buzzer_active=should_buzz,
        sqa_status=sqa_status,
        display=FeedbackDisplay(
            line_status=line_status,
            oled_mode="STANDARD",
        ),
        anomaly_score=float(anomaly_score),
    )

    dash_payload = DashboardPayload(
        timestamp=telemetry.timestamp,
        metrics={
            "heart_rate": telemetry.raw_sensors.heart_rate,
            "spo2": telemetry.raw_sensors.spo2,
            "temperature": telemetry.raw_sensors.temperature,
        },
        temporal_features=features.tolist() if features is not None else None,
        diagnostics={
            "sqa_score": sqa_result.score,
            "sqa_status": sqa_status,
            "anomaly_score": float(anomaly_score),
            "is_anomaly": is_anomaly,
            "severity": status,
        },
        system={
            "rssi": telemetry.network.wifi_rssi,
            "device_id": device_id,
        },
    )

    # Kirim umpan balik ke topik MQTT perangkat
    if publish_mqtt and mqtt_client is not None:
        try:
            feedback_topic = f"physio/{device_id}/feedback"
            feedback_json = feedback.model_dump_json()
            mqtt_client.publish(feedback_topic, feedback_json)
            logger.info(
                f"MQTT Feedback sent to {feedback_topic} | status={status}, buzz={should_buzz}, score={anomaly_score:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to publish MQTT feedback: {e}")

    # Broadcast ke WebSocket dashboard jika aktif
    if main_loop and main_loop.is_running() and manager.dashboard_connections:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_to_dashboards(dash_payload.model_dump()),
            main_loop,
        )

    return feedback, dash_payload


def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    logger.info(f"Connected to MQTT broker ({settings.mqtt_host}:{settings.mqtt_port}) with rc={rc}")
    topics = [
        (settings.mqtt_topic, 0),
        ("physio/+/telemetry", 0),
    ]
    client.subscribe(topics)
    logger.info(f"Subscribed to MQTT topics: {[t[0] for t in topics]}")


def on_mqtt_message(client, userdata, message):
    global msg_counter
    try:
        now = time.time()
        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        payload_str = message.payload.decode("utf-8")
        payload_dict = json.loads(payload_str)

        # Ekstrak device_id jika ada di topik, atau beri nama default untuk device teman
        topic_parts = message.topic.split("/")
        if len(topic_parts) >= 3 and "device_id" not in payload_dict:
            payload_dict["device_id"] = topic_parts[1]
        elif "device_id" not in payload_dict:
            payload_dict["device_id"] = "esp32_hardware"

        device_id = payload_dict["device_id"]

        # Hitung interval waktu penerimaan (delta time)
        prev_time = last_arrival_time.get(device_id)
        if prev_time is not None:
            delta_s = now - prev_time
            if delta_s > 2.5:
                dt_str = f"\033[93m{delta_s:5.2f}s [!]\033[0m"
            else:
                dt_str = f"\033[96m{delta_s:5.2f}s  \033[0m"
        else:
            dt_str = "\033[2m   --   \033[0m"
        last_arrival_time[device_id] = now

        # Jalankan pipeline pemrosesan telemetri
        should_publish_feedback = message.topic.startswith("physio/")
        feedback, dash = process_telemetry(payload_dict, publish_mqtt=should_publish_feedback)
        telemetry = last_telemetry.get(device_id)

        # ANSI Colors
        CLR_RESET = "\033[0m"
        CLR_BOLD = "\033[1m"
        CLR_DIM = "\033[2m"
        CLR_RED = "\033[91m"
        CLR_GREEN = "\033[92m"
        CLR_YELLOW = "\033[93m"
        CLR_CYAN = "\033[96m"
        CLR_WHITE = "\033[97m"
        CLR_BG_RED = "\033[41m\033[97m\033[1m"
        CLR_BG_GREEN = "\033[42m\033[30m\033[1m"
        CLR_BG_YELLOW = "\033[43m\033[30m\033[1m"
        CLR_BG_MAGENTA = "\033[45m\033[97m\033[1m"

        hr_val = telemetry.raw_sensors.heart_rate if telemetry else 0.0
        spo2_val = telemetry.raw_sensors.spo2 if telemetry else 0.0
        temp_val = telemetry.raw_sensors.temperature if telemetry else 0.0

        # Ekstrak Fitur Temporal Jendela Geser (Debug Temporal Dynamics)
        if dash.temporal_features is not None:
            delta_hr = float(dash.temporal_features[3])
            ma_hr = float(dash.temporal_features[6])
            var_hr = float(dash.temporal_features[9])
        elif feature_engine and len(feature_engine.buffer) >= 2:
            data_buf = np.array(feature_engine.buffer)
            delta_hr = float(data_buf[-1, 0] - data_buf[-2, 0])
            ma_hr = float(np.mean(data_buf[:, 0]))
            var_hr = float(np.var(data_buf[:, 0], ddof=1))
        elif feature_engine and len(feature_engine.buffer) == 1:
            data_buf = np.array(feature_engine.buffer)
            delta_hr = 0.0
            ma_hr = float(data_buf[0, 0])
            var_hr = 0.0
        else:
            delta_hr, ma_hr, var_hr = 0.0, 0.0, 0.0

        # HR Instan Formatting
        if hr_val <= 0 or feedback.status == "SIGNAL QUALITY LOW":
            hr_str = f"{CLR_DIM}  -- bpm {CLR_RESET}"
        elif hr_val < 50 or hr_val > 110:
            hr_str = f"{CLR_RED}{CLR_BOLD}{hr_val:5.1f} bpm{CLR_RESET}"
        else:
            hr_str = f"{CLR_WHITE}{CLR_BOLD}{hr_val:5.1f} bpm{CLR_RESET}"

        # Debug: ΔHR Formatting
        if hr_val <= 0 or feedback.status == "SIGNAL QUALITY LOW":
            d_hr_str = f"{CLR_DIM}   --  {CLR_RESET}"
        elif abs(delta_hr) > 15.0:
            d_hr_str = f"{CLR_RED}{CLR_BOLD}{delta_hr:+6.1f} {CLR_RESET}"
        else:
            d_hr_str = f"{CLR_CYAN}{delta_hr:+6.1f} {CLR_RESET}"

        # Debug: MA_HR Formatting
        if hr_val <= 0 or feedback.status == "SIGNAL QUALITY LOW":
            ma_hr_str = f"{CLR_DIM}   --  {CLR_RESET}"
        else:
            ma_hr_str = f"{CLR_WHITE}{ma_hr:5.1f} {CLR_RESET}"

        # Debug: VAR_HR Formatting
        if hr_val <= 0 or feedback.status == "SIGNAL QUALITY LOW":
            var_hr_str = f"{CLR_DIM}   --   {CLR_RESET}"
        elif var_hr > 50.0:
            var_hr_str = f"{CLR_RED}{CLR_BOLD}{var_hr:6.1f} {CLR_RESET}"
        else:
            var_hr_str = f"{CLR_WHITE}{var_hr:6.1f} {CLR_RESET}"

        # SpO2 Formatting
        spo2_str = f"{CLR_WHITE}{spo2_val:4.0f}%{CLR_RESET}"

        # Suhu Formatting
        if temp_val >= 38.0:
            temp_str = f"{CLR_RED}{CLR_BOLD}{temp_val:4.1f} C{CLR_RESET}"
        else:
            temp_str = f"{CLR_WHITE}{temp_val:4.1f} C{CLR_RESET}"

        # SQA Formatting
        if feedback.sqa_status == "GOOD":
            sqa_str = f"{CLR_GREEN}GOOD{CLR_RESET}"
        else:
            sqa_str = f"{CLR_YELLOW}POOR{CLR_RESET}"

        # Status Fisiologis Formatting (NEWS-inspired Non-Diagnostic Severity)
        if feedback.status == "HIGH DEVIATION (SUSTAINED)":
            stat_str = f"{CLR_BG_RED} HIGH DEV (S) {CLR_RESET}"
        elif feedback.status == "HIGH DEVIATION":
            stat_str = f"{CLR_BG_RED}   HIGH DEV   {CLR_RESET}"
        elif feedback.status == "LOW DEVIATION (SUSTAINED)":
            stat_str = f"{CLR_BG_MAGENTA} LOW DEV (S)  {CLR_RESET}"
        elif feedback.status == "LOW DEVIATION":
            stat_str = f"{CLR_BG_YELLOW}   LOW DEV    {CLR_RESET}"
        elif feedback.status == "SIGNAL QUALITY LOW":
            stat_str = f"\033[43m\033[30m\033[1m SIG QUAL LOW {CLR_RESET}"
        else:
            stat_str = f"{CLR_BG_GREEN}    NORMAL    {CLR_RESET}"

        # Model ML Result / Buffer Status
        buf_len = len(feature_engine.buffer) if feature_engine else 0
        buf_max = feature_engine.window_size if feature_engine else 15
        thresh_val = getattr(anomaly_detector, "anomaly_threshold", 0.0629) if anomaly_detector else 0.0629

        if buf_len < buf_max:
            ml_str = f"{CLR_DIM}Buffer ({buf_len:2d}/{buf_max:2d})   {CLR_RESET}"
        else:
            if feedback.anomaly_score < thresh_val:
                ml_str = f"{CLR_RED}{CLR_BOLD}Skor: {feedback.anomaly_score:+.3f} [!] {CLR_RESET}"
            else:
                ml_str = f"{CLR_GREEN}Skor: {feedback.anomaly_score:+.3f}     {CLR_RESET}"

        # Buzzer State Formatting
        if feedback.buzzer_active:
            buzz_str = f"{CLR_BG_RED}  AKTIF   {CLR_RESET}"
        else:
            buzz_str = f"{CLR_DIM}   Mati   {CLR_RESET}"

        # Tampilkan Header setiap 25 baris
        if msg_counter % 25 == 0:
            print(TABLE_HEADER, flush=True)
        msg_counter += 1

        # Baris Tabular Rapi dengan Kolom Debug Temporal
        row = (
            f" {now_str}   {dt_str}  {hr_str}  {d_hr_str}  {ma_hr_str}  {var_hr_str}   "
            f"{spo2_str}  {temp_str}   {sqa_str}   {stat_str}   {ml_str}   {buzz_str}"
        )
        print(row, flush=True)
        logger.debug(f"MQTT Rx: dev={device_id} status={feedback.status} hr={hr_val} temp={temp_val} buzz={feedback.buzzer_active}")

    except Exception as e:
        logger.error(f"Error handling MQTT message on {message.topic}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global anomaly_detector, mqtt_client, main_loop
    import uuid
    logger.info("Starting up FastAPI application...")
    main_loop = asyncio.get_running_loop()

    # Inisialisasi model ML Fase 1
    anomaly_detector = AnomalyDetector()

    # Inisialisasi MQTT client background worker dengan unique Client ID untuk broker HiveMQ publik
    client_id = f"backend-physio-{uuid.uuid4().hex[:8]}"
    if hasattr(mqtt, "CallbackAPIVersion"):
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    else:
        mqtt_client = mqtt.Client(client_id=client_id)

    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message

    try:
        logger.info(f"Connecting to MQTT broker at {settings.mqtt_host}:{settings.mqtt_port} (Client ID: {client_id})...")
        mqtt_client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=60)
        mqtt_client.loop_start()
        logger.info("MQTT background loop started successfully.")
    except Exception as e:
        logger.warning(f"Could not connect to MQTT broker ({e}). Background MQTT worker will remain idle.")

    yield

    logger.info("Shutting down FastAPI application...")
    if mqtt_client:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logger.info("MQTT client disconnected successfully.")
        except Exception as e:
            logger.error(f"Error disconnecting MQTT client: {e}")


app = FastAPI(title="Physio Monitor Backend", lifespan=lifespan)


@app.get("/health")
def health_check():
    is_mqtt_connected = bool(mqtt_client and mqtt_client.is_connected())
    return {
        "status": "ok",
        "mqtt_connected": is_mqtt_connected,
        "model_loaded": bool(anomaly_detector and anomaly_detector.model is not None),
        "threshold": getattr(anomaly_detector, "anomaly_threshold", 0.0) if anomaly_detector else 0.0,
    }


@app.get("/api/status")
def get_status():
    return {"last_readings": {k: v.model_dump() for k, v in last_telemetry.items()}}


@app.websocket("/ws/device")
async def websocket_device_endpoint(websocket: WebSocket):
    await manager.connect_device(websocket)
    device_id = "unknown"
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload_dict = json.loads(data)
                feedback, _ = process_telemetry(payload_dict, publish_mqtt=False)
                device_id = feedback.device_id
                await websocket.send_json(feedback.model_dump())
            except json.JSONDecodeError:
                logger.error("Invalid JSON received over WebSocket")
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
    except WebSocketDisconnect:
        manager.disconnect_device(websocket)
        logger.info(f"Device {device_id} disconnected from WebSocket")
        feature_engine.reset()


@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    await manager.connect_dashboard(websocket)
    try:
        while True:
            # Tetap jaga koneksi aktif
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_dashboard(websocket)
