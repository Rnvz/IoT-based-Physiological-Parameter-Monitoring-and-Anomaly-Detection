from typing import Optional, List
from pydantic import BaseModel
from app.core.config import settings
from app.schemas.telemetry import TelemetryPayload

class SQAResult(BaseModel):
    """
    Hasil dari Signal Quality Assessment (SQA).
    """
    score: float
    state: str
    reasons: List[str]

class SignalQualityAssessor:
    """
    Layanan untuk menilai kualitas sinyal dari data telemetri yang masuk.
    """
    def __init__(self):
        self.hr_min = settings.hr_min
        self.hr_max = settings.hr_max
        self.spo2_min = settings.spo2_min
        self.spo2_max = settings.spo2_max
        self.temp_min = settings.temp_min
        self.temp_max = settings.temp_max
        self.max_hr_jump = settings.max_hr_jump
        self.good_threshold = settings.sqa_good_threshold

    def assess(self, telemetry: TelemetryPayload, previous: Optional[TelemetryPayload] = None) -> SQAResult:
        score = 1.0
        reasons = []

        sensors = telemetry.raw_sensors
        
        # Check 1: Physiological range
        if not (self.hr_min <= sensors.heart_rate <= self.hr_max):
            score -= 0.3
            reasons.append("HR out of range")
            
        if not (self.spo2_min <= sensors.spo2 <= self.spo2_max):
            score -= 0.3
            reasons.append("SpO2 out of range")
            
        if not (self.temp_min <= sensors.temperature <= self.temp_max):
            score -= 0.2
            reasons.append("Temperature out of range")
            
        # Check 2: Sudden jump detection (if previous exists)
        if previous:
            time_diff = (telemetry.timestamp - previous.timestamp) / 1000.0  # seconds
            if time_diff > 0:
                hr_jump = abs(sensors.heart_rate - previous.raw_sensors.heart_rate)
                if hr_jump > self.max_hr_jump:
                    score -= 0.4
                    reasons.append("Sudden HR jump")
                    
        # Check 3: PPG amplitude threshold
        if telemetry.sensor_status.ppg_amplitude < 100:  # arbitrary minimum valid threshold
            score -= 0.5
            reasons.append("Low PPG amplitude")

        score = max(0.0, score)
        state = "GOOD" if score >= self.good_threshold else "POOR_QUALITY"
        
        return SQAResult(score=score, state=state, reasons=reasons)
