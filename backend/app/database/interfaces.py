from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ReadingRecord(BaseModel):
    ts: str
    device_id: str
    hr: Optional[float] = None
    spo2: Optional[float] = None
    temp: Optional[float] = None
    hum: Optional[float] = None
    room_temp: Optional[float] = None
    finger: bool = False
    fw_status: Optional[str] = "NORMAL"

class PredictionRecord(BaseModel):
    ts: str
    device_id: str
    rule_severity: Optional[int] = None
    ml_severity: Optional[int] = None
    ml_score: Optional[float] = None
    published_severity: int = 0
    decision_mode: str = "RULE"
    decision_source: str = "RULE"
    label: str = "NORMAL"
    model_version: str = "if_v1"

class ActuatorEventRecord(BaseModel):
    ts: str
    kind: str
    seq: Optional[int] = None
    payload_json: str

class ScenarioEventRecord(BaseModel):
    ts: str
    scenario: str
    note: Optional[str] = None

class BaseRepository(ABC):
    @abstractmethod
    async def save_reading(self, reading: ReadingRecord) -> bool:
        """Save raw sensor reading before any inference."""
        pass

    @abstractmethod
    async def get_latest_reading(self, device_id: str) -> Optional[ReadingRecord]:
        """Fetch the most recent reading for a given device."""
        pass

    @abstractmethod
    async def get_latest_prediction(self, device_id: str) -> Optional[PredictionRecord]:
        """Fetch the most recent prediction for a given device."""
        pass

    @abstractmethod
    async def get_history(self, device_id: str, minutes: int = 30) -> List[ReadingRecord]:
        """Fetch readings history for a given device within the last N minutes."""
        pass

    @abstractmethod
    async def save_prediction(self, prediction: PredictionRecord) -> bool:
        """Save model / rule prediction output."""
        pass

    @abstractmethod
    async def get_predictions(self, device_id: str, limit: int = 100) -> List[PredictionRecord]:
        """Fetch latest predictions for a given device."""
        pass

    @abstractmethod
    async def save_actuator_event(self, event: ActuatorEventRecord) -> bool:
        """Save command sent or actuator status received."""
        pass

    @abstractmethod
    async def get_latest_actuator_event(self, kind: Optional[str] = None) -> Optional[ActuatorEventRecord]:
        """Fetch the most recent actuator event."""
        pass

    @abstractmethod
    def stream_dataset_export(self):
        """Stream complete dataset export with as-of join for research/course submission."""
        pass

    @abstractmethod
    async def save_scenario_event(self, event: ScenarioEventRecord) -> bool:
        """Save scenario tag change."""
        pass

    @abstractmethod
    async def get_current_scenario(self) -> ScenarioEventRecord:
        """Fetch the current active scenario event."""
        pass
