import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.mqtt.consumer import MQTTService
from app.api import health_router, telemetry_router, actuator_router, session_router

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

mqtt_service: Optional[MQTTService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mqtt_service

    # Security check: Refuse to start if API_KEY is empty
    configured_key = settings.api_key.strip() if settings.api_key else ""
    if not configured_key:
        raise RuntimeError("API_KEY must not be empty. Server refusing to start with unsecured routes.")

    logger.info("Initializing IoT Physio Monitor Backend...")
    try:
        loop = asyncio.get_running_loop()
        mqtt_service = MQTTService(loop=loop)
        mqtt_service.start()
        logger.info(
            f"MQTT Service started. Broker: {settings.mqtt_broker}:{settings.mqtt_port}, "
            f"Topic prefix: {settings.topic_prefix}, Decision mode: {settings.decision_mode}"
        )
    except Exception as e:
        logger.warning(f"Could not start MQTT service ({e}). Continuing in standalone API mode.")

    yield

    logger.info("Shutting down IoT Physio Monitor Backend...")
    if mqtt_service:
        mqtt_service.stop()
        logger.info("MQTT Service stopped.")

app = FastAPI(title="Physio Monitor Backend", lifespan=lifespan)

# CORS Configuration (strictly from CORS_ORIGINS, never "*")
allowed_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# Global exception handler to prevent leaking stack traces
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Wire API Routers
app.include_router(health_router)
app.include_router(telemetry_router)
app.include_router(actuator_router)
app.include_router(session_router)
