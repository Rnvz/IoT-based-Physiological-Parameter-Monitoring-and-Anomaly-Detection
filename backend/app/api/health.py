from fastapi import APIRouter
from app.config import settings

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    """Exact contract from section 4: {\"status\":\"ok\"}"""
    return {"status": "ok"}

@router.get("/health/details")
def health_details():
    from app.main import mqtt_service
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
