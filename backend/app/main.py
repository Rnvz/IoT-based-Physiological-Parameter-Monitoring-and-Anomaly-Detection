from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import asyncio
import json
import logging
from typing import Dict, List
from app.core.config import settings
from app.schemas.telemetry import TelemetryPayload, FeedbackPayload, DashboardPayload
from app.services.sqa import SignalQualityAssessor
from app.services.feature_fusion import FeatureFusionEngine
from app.services.anomaly_model import AnomalyDetector, PersistentAnomalyEvaluator

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Global instances
sqa_service = SignalQualityAssessor()
feature_engine = FeatureFusionEngine()
anomaly_detector = None
persistent_evaluator = PersistentAnomalyEvaluator()

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
        for connection in self.dashboard_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Dashboard WS error: {e}")

manager = ConnectionManager()

last_telemetry: Dict[str, TelemetryPayload] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global anomaly_detector
    logger.info("Starting up FastAPI application...")
    anomaly_detector = AnomalyDetector()
    yield
    logger.info("Shutting down FastAPI application...")

app = FastAPI(title="Physio Monitor Backend", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/status")
def get_status():
    return {"last_readings": {k: v.dict() for k, v in last_telemetry.items()}}

@app.websocket("/ws/device")
async def websocket_device_endpoint(websocket: WebSocket):
    await manager.connect_device(websocket)
    device_id = "unknown"
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload_dict = json.loads(data)
                telemetry = TelemetryPayload(**payload_dict)
                device_id = telemetry.device_id
                
                # SQA
                prev_telemetry = last_telemetry.get(device_id)
                sqa_result = sqa_service.assess(telemetry, prev_telemetry)
                
                # Feature Fusion
                features = feature_engine.process(telemetry)
                
                # Anomaly Detection
                is_anomaly = False
                anomaly_score = 0.0
                sqa_status = sqa_result.state
                
                if features is not None and sqa_result.state == "GOOD":
                    anom_result = anomaly_detector.predict(features)
                    eval_result = persistent_evaluator.evaluate(anom_result.is_anomaly)
                    is_anomaly = eval_result.is_persistent_anomaly
                    anomaly_score = anom_result.anomaly_score
                    should_buzz = eval_result.should_buzz
                else:
                    should_buzz = False

                # Save last telemetry
                last_telemetry[device_id] = telemetry

                # Send feedback to device
                feedback = FeedbackPayload(
                    device_id=device_id,
                    reply_sequence_id=telemetry.network.sequence_id,
                    status="OK",
                    buzzer_active=should_buzz,
                    sqa_status=sqa_status,
                    display={
                        "line_status": "NORMAL" if not is_anomaly else "ANOMALI",
                        "oled_mode": "STANDARD"
                    }
                )
                await websocket.send_json(feedback.dict())

                # Send to dashboards
                dash_payload = DashboardPayload(
                    timestamp=telemetry.timestamp,
                    metrics={
                        "heart_rate": telemetry.raw_sensors.heart_rate,
                        "spo2": telemetry.raw_sensors.spo2,
                        "temperature": telemetry.raw_sensors.temperature
                    },
                    temporal_features=features.tolist() if features is not None else None,
                    diagnostics={
                        "sqa_score": sqa_result.score,
                        "sqa_status": sqa_status,
                        "anomaly_score": anomaly_score,
                        "is_anomaly": is_anomaly
                    },
                    system={
                        "rssi": telemetry.network.wifi_rssi,
                        "device_id": device_id
                    }
                )
                await manager.broadcast_to_dashboards(dash_payload.dict())

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
            except Exception as e:
                logger.error(f"Processing error: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect_device(websocket)
        logger.info(f"Device {device_id} disconnected")
        feature_engine.reset()
