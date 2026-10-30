import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from app.config import settings
from app.mqtt.consumer import MQTTService
from app.database.factory import get_repository

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

mqtt_service: Optional[MQTTService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mqtt_service
    logger.info("Initializing IoT Physio Monitor Backend (Phase 1)...")
    loop = asyncio.get_running_loop()
    mqtt_service = MQTTService(loop=loop)
    mqtt_service.start()
    logger.info(
        f"MQTT Service started. Broker: {settings.mqtt_broker}:{settings.mqtt_port}, "
        f"Topic prefix: {settings.topic_prefix}, Decision mode: {settings.decision_mode}"
    )

    yield

    logger.info("Shutting down IoT Physio Monitor Backend...")
    if mqtt_service:
        mqtt_service.stop()
        logger.info("MQTT Service stopped.")

app = FastAPI(title="Physio Monitor Backend", lifespan=lifespan)

@app.get("/health")
def health_check():
    is_connected = bool(mqtt_service and mqtt_service.client.is_connected())
    model_loaded = bool(mqtt_service and mqtt_service.ml_service and mqtt_service.ml_service.model is not None)
    return {
        "status": "ok",
        "mqtt_connected": is_connected,
        "decision_mode": settings.decision_mode,
        "model_loaded": model_loaded,
        "model_version": settings.model_version,
        "db_backend": settings.db_backend,
    }
