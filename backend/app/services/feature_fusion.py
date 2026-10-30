import numpy as np
from collections import deque
from typing import Optional
from app.schemas.telemetry import TelemetryPayload
from app.core.config import settings

class FeatureFusionEngine:
    """
    Mesin fusi fitur temporal untuk ekstraksi fitur dalam jendela waktu geser (sliding window).
    Menghasilkan vektor fitur 13 dimensi (tanpa Signal_Quality_Score).
    """
    def __init__(self, window_size: int = settings.sliding_window_size):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def process(self, telemetry: TelemetryPayload) -> Optional[np.ndarray]:
        sensors = telemetry.raw_sensors
        
        data_point = np.array([sensors.heart_rate, sensors.spo2, sensors.temperature])
        self.buffer.append(data_point)
        
        if len(self.buffer) < self.window_size:
            return None
            
        return self._compute_features()
        
    def _compute_features(self) -> np.ndarray:
        data = np.array(self.buffer)
        
        current_values = data[-1]
        
        delta = data[-1] - data[-2]
        
        moving_avg = np.mean(data, axis=0)
        
        variance = np.var(data, axis=0, ddof=1)
        
        rate_of_change = (
            abs(delta[0]) / max(moving_avg[0], 1e-6) +
            abs(delta[1]) / max(moving_avg[1], 1e-6) +
            abs(delta[2]) / max(moving_avg[2], 1e-6)
        )
        
        feature_vector = np.concatenate([
            current_values,
            delta,
            moving_avg,
            variance,
            [rate_of_change]
        ])
        
        return feature_vector
        
    def reset(self):
        """Menghapus isi buffer."""
        self.buffer.clear()
