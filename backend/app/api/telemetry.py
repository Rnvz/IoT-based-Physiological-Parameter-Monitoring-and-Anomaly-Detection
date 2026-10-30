import os
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, Query, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.config import settings
from app.database.factory import get_repository
from app.api.deps import verify_api_key, verify_read_api_key

router = APIRouter(tags=["telemetry"])

@router.get("/latest")
async def get_latest(
    device_id: Optional[str] = None,
    _auth: Optional[str] = Depends(verify_read_api_key)
):
    """
    Returns latest sensor reading and latest prediction.
    Shape: {"reading": {...}|null, "prediction": {...}|null}
    """
    target_device = device_id or settings.device_id
    repo = get_repository()

    reading = await repo.get_latest_reading(target_device)
    prediction = await repo.get_latest_prediction(target_device)

    reading_dict = None
    if reading:
        reading_dict = {
            "ts": reading.ts,
            "hr": reading.hr,
            "spo2": reading.spo2,
            "temp": reading.temp,
            "hum": reading.hum,
            "room_temp": reading.room_temp,
            "finger": reading.finger,
        }

    prediction_dict = None
    if prediction:
        prediction_dict = {
            "ts": prediction.ts,
            "label": prediction.label,
            "severity": prediction.published_severity,
            "score": prediction.ml_score,
        }

    return {
        "reading": reading_dict,
        "prediction": prediction_dict,
    }

@router.get("/history")
async def get_history(
    minutes: int = Query(default=30, ge=1, le=1440, description="History window in minutes (1-1440)"),
    device_id: Optional[str] = None,
    _auth: Optional[str] = Depends(verify_read_api_key)
):
    """
    Returns historical readings for device within the last N minutes.
    Shape: {"items": [{ts, hr, spo2, temp, hum, room_temp, finger}]}
    """
    target_device = device_id or settings.device_id
    repo = get_repository()
    readings = await repo.get_history(target_device, minutes=minutes)

    items = [
        {
            "ts": r.ts,
            "hr": r.hr,
            "spo2": r.spo2,
            "temp": r.temp,
            "hum": r.hum,
            "room_temp": r.room_temp,
            "finger": r.finger,
        }
        for r in readings
    ]
    return {"items": items}

@router.get("/predictions")
async def get_predictions(
    limit: int = Query(default=100, ge=1, le=1000, description="Limit predictions count (1-1000)"),
    device_id: Optional[str] = None,
    _auth: Optional[str] = Depends(verify_read_api_key)
):
    """
    Returns recent predictions for device.
    Shape: {"items": [{ts, label, severity, score, model_version, rule_severity, ml_severity, decision_mode, decision_source}]}
    """
    target_device = device_id or settings.device_id
    repo = get_repository()
    preds = await repo.get_predictions(target_device, limit=limit)

    items = [
        {
            "ts": p.ts,
            "label": p.label,
            "severity": p.published_severity,
            "score": p.ml_score,
            "model_version": p.model_version,
            "rule_severity": p.rule_severity,
            "ml_severity": p.ml_severity,
            "decision_mode": p.decision_mode,
            "decision_source": p.decision_source,
        }
        for p in preds
    ]
    return {"items": items}

@router.get("/device/status")
async def get_device_status(
    device_id: Optional[str] = None,
    _auth: Optional[str] = Depends(verify_read_api_key)
):
    """
    Returns online status and age in seconds.
    Online if last /data arrived within 15 seconds.
    Shape: {"online": bool, "age_s": number}
    """
    target_device = device_id or settings.device_id
    repo = get_repository()
    latest = await repo.get_latest_reading(target_device)

    if not latest or not latest.ts:
        return {"online": False, "age_s": -1.0}

    try:
        dt = datetime.fromisoformat(latest.ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_s = max(0.0, (now - dt).total_seconds())
        online = age_s <= 15.0
        return {
            "online": online,
            "age_s": round(age_s, 1),
        }
    except Exception:
        return {"online": False, "age_s": -1.0}

@router.get("/export/readings.csv")
async def export_readings_csv(_auth: str = Depends(verify_api_key)):
    """
    Streams readings.csv dataset for research/course submission.
    Protected with X-API-Key.
    """
    csv_file = os.path.abspath(os.path.join(settings.csv_dir, "readings.csv"))
    
    if not os.path.exists(csv_file):
        async def empty_stream():
            yield "ts,device_id,hr,spo2,temp,hum,room_temp,finger,fw_status\n"
        return StreamingResponse(
            empty_stream(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=readings.csv"}
        )

    def iterfile():
        with open(csv_file, "r", encoding="utf-8") as f:
            for line in f:
                yield line

    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=readings.csv"}
    )

@router.get("/export/dataset.csv")
async def export_dataset_csv(_auth: str = Depends(verify_api_key)):
    """
    Streams consolidated course dataset with as-of joined actuator state.
    Header: ts,hr,spo2,temp,hum,room_temp,finger,fw_status,rule_severity,ml_severity,published_severity,decision_mode,mode,led_color,brightness,buzzer_test,source
    Protected with X-API-Key.
    """
    repo = get_repository()
    stream_generator = repo.stream_dataset_export()

    return StreamingResponse(
        stream_generator,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset.csv"}
    )
