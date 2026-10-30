from app.api.health import router as health_router
from app.api.telemetry import router as telemetry_router
from app.api.actuator import router as actuator_router

__all__ = ["health_router", "telemetry_router", "actuator_router"]
