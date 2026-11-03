import json
from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import settings
from app.database.factory import get_repository
from app.api.deps import verify_api_key, verify_read_api_key, rate_limit_post

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
async def get_actuator_status(_auth: Optional[str] = Depends(verify_read_api_key)):
    """
    Returns latest actuator status from ESP32.
    Shape: {"mode","led":{"color","brightness"},"buzzer":{"test":bool},"source","ack_seq","manual_left_s"}
    """
    repo = get_repository()
    latest_status = await repo.get_latest_actuator_event(kind="status")
    latest_cmd = await repo.get_latest_actuator_event(kind="cmd")

    status_data = {}
    if latest_status and latest_status.payload_json:
        try:
            status_data = json.loads(latest_status.payload_json)
        except Exception:
            pass

    cmd_data = {}
    if latest_cmd and latest_cmd.payload_json:
        try:
            cmd_data = json.loads(latest_cmd.payload_json)
        except Exception:
            pass

    cmd_seq = latest_cmd.seq if latest_cmd else None
    ack_seq = status_data.get("ack_seq")

    cmd_is_newer = False
    if latest_cmd:
        if not latest_status:
            cmd_is_newer = True
        elif cmd_seq is not None and (ack_seq is None or cmd_seq > ack_seq):
            cmd_is_newer = True

    mode = status_data.get("mode", "AUTO")
    led = dict(status_data.get("led", {"color": "GREEN", "brightness": 40}))
    buzzer = dict(status_data.get("buzzer", {"test": False}))
    source = status_data.get("source", "ML_RESULT" if latest_status else "NO_RESULT")
    manual_left_s = status_data.get("manual_left_s", 0)

    if cmd_is_newer and cmd_data:
        if "mode" in cmd_data and cmd_data["mode"]:
            mode = cmd_data["mode"]
        if mode == "MANUAL":
            source = "MANUAL"
            manual_left_s = 300
        elif mode == "AUTO":
            source = "ML_RESULT"
            manual_left_s = 0

        if "color" in cmd_data and cmd_data["color"]:
            led["color"] = cmd_data["color"]
        if "brightness" in cmd_data and cmd_data["brightness"] is not None:
            led["brightness"] = cmd_data["brightness"]
        if "buzzer_test" in cmd_data and cmd_data["buzzer_test"] is not None:
            buzzer["test"] = cmd_data["buzzer_test"]

    return {
        "mode": mode,
        "led": led,
        "buzzer": buzzer,
        "source": source,
        "ack_seq": ack_seq,
        "manual_left_s": manual_left_s
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
