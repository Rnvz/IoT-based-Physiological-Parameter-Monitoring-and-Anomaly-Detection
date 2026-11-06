from typing import Optional
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
    
    model_loaded = False
    fallback_reason: Optional[str] = None
    model_version = settings.model_version

    if mqtt_service and mqtt_service.ml_service:
        svc = mqtt_service.ml_service
        if hasattr(svc, "is_loaded"):
            model_loaded = bool(svc.is_loaded)
        elif hasattr(svc, "model"):
            model_loaded = bool(svc.model is not None)
        elif hasattr(svc, "ort_session") or hasattr(svc, "numpy_model"):
            model_loaded = bool(svc.ort_session is not None or svc.numpy_model is not None)

        if hasattr(svc, "model_version") and svc.model_version:
            model_version = svc.model_version
    elif getattr(settings, "model_kind", "ml").lower() == "dl":
        model_version = "ae_v1"

    if not model_loaded:
        fallback_reason = "Model artifacts not loaded or ML service uninitialized"

    return {
        "status": "ok",
        "mqtt_connected": is_connected,
        "decision_mode": settings.decision_mode,
        "model_loaded": model_loaded,
        "fallback_reason": fallback_reason,
        "model_version": model_version,
        "db_backend": settings.db_backend,
    }
