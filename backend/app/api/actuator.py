import json
from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import settings
from app.database.factory import get_repository
from app.api.deps import verify_api_key, rate_limit_post

router = APIRouter(prefix="/actuator", tags=["actuator"])

LedColor = Literal["OFF", "GREEN", "RED", "BLUE", "YELLOW", "WHITE"]

class ModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["AUTO", "MANUAL"]

class LedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    color: LedColor
    brightness: int = Field(ge=0, le=100)

class BuzzerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test: bool

def _get_publisher():
    from app.main import mqtt_service
    if mqtt_service and mqtt_service.publisher:
        return mqtt_service.publisher
    return None

@router.get("/status")
async def get_actuator_status():
    """
    Returns latest actuator status from ESP32.
    Shape: {"mode","led":{"color","brightness"},"buzzer":{"test":bool},"source","ack_seq","manual_left_s"}
    """
    repo = get_repository()
    latest_event = await repo.get_latest_actuator_event(kind="status")

    if latest_event and latest_event.payload_json:
        try:
            data = json.loads(latest_event.payload_json)
            return {
                "mode": data.get("mode", "AUTO"),
                "led": data.get("led", {"color": "GREEN", "brightness": 40}),
                "buzzer": data.get("buzzer", {"test": False}),
                "source": data.get("source", "ML_RESULT"),
                "ack_seq": data.get("ack_seq"),
                "manual_left_s": data.get("manual_left_s", 0)
            }
        except Exception:
            pass

    # Default fallback shape
    return {
        "mode": "AUTO",
        "led": {"color": "GREEN", "brightness": 40},
        "buzzer": {"test": False},
        "source": "NO_RESULT",
        "ack_seq": None,
        "manual_left_s": 0
    }

@router.post("/mode", dependencies=[Depends(rate_limit_post)])
async def set_mode(body: ModeRequest, _auth: str = Depends(verify_api_key)):
    """Sets actuator mode (AUTO or MANUAL). Protected with X-API-Key."""
    publisher = _get_publisher()
    if not publisher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MQTT publisher service unavailable"
        )

    cmd = await publisher.publish_command(mode=body.mode)
    if not cmd:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to publish mode command"
        )
    return cmd.model_dump(exclude_none=True)

@router.post("/led", dependencies=[Depends(rate_limit_post)])
async def set_led(body: LedRequest, _auth: str = Depends(verify_api_key)):
    """Sets LED color and brightness, automatically switching to MANUAL mode."""
    publisher = _get_publisher()
    if not publisher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MQTT publisher service unavailable"
        )

    cmd = await publisher.publish_command(
        mode="MANUAL",
        color=body.color,
        brightness=body.brightness
    )
    if not cmd:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to publish LED command"
        )
    return cmd.model_dump(exclude_none=True)

@router.post("/buzzer", dependencies=[Depends(rate_limit_post)])
async def set_buzzer(body: BuzzerRequest, _auth: str = Depends(verify_api_key)):
    """Triggers buzzer test. Protected with X-API-Key."""
    publisher = _get_publisher()
    if not publisher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MQTT publisher service unavailable"
        )

    cmd = await publisher.publish_command(buzzer_test=body.test)
    if not cmd:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to publish buzzer command"
        )
    return cmd.model_dump(exclude_none=True)
