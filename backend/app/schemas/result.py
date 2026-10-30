from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal

ResultLabel = Literal["NORMAL", "ANOMALY", "NO_DATA", "SENSOR_ERROR"]

class ResultPayload(BaseModel):
    """
    Schema for outgoing /result MQTT messages to ESP32.
    Format: {"label":"NORMAL","severity":0,"score":0.12,"model_version":"if_v1"}
    """
    model_config = ConfigDict(extra="ignore")

    label: str = "NORMAL"
    severity: int = Field(default=0, ge=0, le=2)
    score: Optional[float] = None
    model_version: str = "if_v1"
