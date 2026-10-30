from datetime import datetime, timezone
from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field
from fastapi import APIRouter, Depends

from app.database.factory import get_repository
from app.database.interfaces import ScenarioEventRecord
from app.api.deps import verify_api_key, verify_read_api_key

router = APIRouter(prefix="/session", tags=["session"])

# Strict benign scenarios only: never add scenarios implying provoked abnormal physiology
ScenarioType = Literal[
    "REST",
    "TYPING",
    "LIGHT_ACTIVITY",
    "MOTION",
    "FINGER_SHIFT",
    "SENSOR_OFF",
    "OTHER"
]

class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: ScenarioType
    note: Optional[str] = Field(default=None, max_length=200)

@router.post("/scenario")
async def set_scenario(
    payload: ScenarioRequest,
    _auth: str = Depends(verify_api_key)
):
    """
    Sets current benign scenario tag for own-device recordings.
    Protected with X-API-Key.
    Scenario remains active until explicitly updated.
    """
    repo = get_repository()
    ts = datetime.now(timezone.utc).isoformat()
    record = ScenarioEventRecord(
        ts=ts,
        scenario=payload.scenario,
        note=payload.note
    )
    await repo.save_scenario_event(record)

    return {
        "status": "ok",
        "scenario": record.scenario,
        "note": record.note,
        "ts": record.ts
    }

@router.get("/scenario")
async def get_scenario(
    _auth: Optional[str] = Depends(verify_read_api_key)
):
    """
    Fetches the current active scenario tag.
    Respects READ_AUTH_REQUIRED configuration.
    """
    repo = get_repository()
    current = await repo.get_current_scenario()

    return {
        "scenario": current.scenario,
        "note": current.note,
        "ts": current.ts
    }
