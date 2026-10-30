from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any

class ActuatorStatusPayload(BaseModel):
    """
    Schema for incoming /actuator MQTT messages from ESP32.
    Format: {"mode":"AUTO","led":{"color":"GREEN","brightness":40},"buzzer":{"test":false},"source":"ML_RESULT","ack_seq":103,"manual_left_s":0}
    """
    model_config = ConfigDict(extra="ignore")

    mode: str = "AUTO"
    led: Optional[Dict[str, Any]] = None
    buzzer: Optional[Dict[str, Any]] = None
    source: str = "ML_RESULT"
    ack_seq: Optional[int] = None
    manual_left_s: int = 0
