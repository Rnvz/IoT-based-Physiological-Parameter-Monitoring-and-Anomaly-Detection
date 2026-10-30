from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal

LedColor = Literal["OFF", "GREEN", "RED", "BLUE", "YELLOW", "WHITE"]

class CommandPayload(BaseModel):
    """
    Schema for outgoing /cmd MQTT messages to ESP32.
    Format:
    {"mode":"AUTO","seq":101}
    {"mode":"MANUAL","color":"RED","brightness":60,"seq":102}
    {"buzzer_test":true,"seq":103}
    """
    model_config = ConfigDict(extra="ignore")

    mode: Optional[Literal["AUTO", "MANUAL"]] = None
    color: Optional[LedColor] = None
    brightness: Optional[int] = Field(default=None, ge=0, le=100)
    buzzer_test: Optional[bool] = None
    seq: int = Field(ge=1, le=2147483647)
