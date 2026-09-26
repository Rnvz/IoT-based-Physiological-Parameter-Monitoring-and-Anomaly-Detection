from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
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
        eval_result = persistent_evaluator.evaluate(anom_result.is_anomaly)
        is_anomaly = eval_result.is_persistent_anomaly
        anomaly_score = anom_result.anomaly_score
        should_buzz = eval_result.should_buzz
    else:
        # Jika SQA POOR atau buffer fitur belum penuh, redam alarm
        should_buzz = False

    # Simpan telemetri terakhir
    last_telemetry[device_id] = telemetry

    # Status untuk firmware ESP32 & tampilan OLED
    if sqa_status != "GOOD":
        status = "cek_sensor"
        line_status = "CEK SENSOR"
    elif is_anomaly:
        status = "anomaly"
        line_status = "ANOMALI"
    else:
        status = "normal"
        line_status = "NORMAL"

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
    try:
        payload_str = message.payload.decode("utf-8")
        payload_dict = json.loads(payload_str)

        # Ekstrak device_id jika ada di topik, atau beri nama default untuk device teman
        topic_parts = message.topic.split("/")
        if len(topic_parts) >= 3 and "device_id" not in payload_dict:
            payload_dict["device_id"] = topic_parts[1]
        elif "device_id" not in payload_dict:
            payload_dict["device_id"] = "esp32_hardware"

        # Log payload mentah yang diterima dari device
        logger.info(f"📡 [MQTT DITERIMA] Topik: {message.topic} | Payload: {payload_str}")

        # Jangan kirim feedback MQTT jika device belum subscribe (hanya kirim jika topic physio/)
        should_publish_feedback = message.topic.startswith("physio/")
        feedback, dash = process_telemetry(payload_dict, publish_mqtt=should_publish_feedback)

        # Log hasil inferensi ML secara eksplisit
        thresh_val = getattr(anomaly_detector, "anomaly_threshold", 0.0) if anomaly_detector else 0.0
        logger.info(
            f"🔬 [INFERENSI ML] Device: {feedback.device_id} | Status: {feedback.status} | SQA: {feedback.sqa_status} | "
            f"Anomaly Score: {feedback.anomaly_score:+.4f} (Thresh FPR 13%: {thresh_val:.4f}) | "
            f"Buzzer Alarm: {feedback.buzzer_active} | [Catatan: SpO2 masih statis 98.0%]"
        )
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
