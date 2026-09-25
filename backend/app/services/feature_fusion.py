import numpy as np
from collections import deque
from typing import Optional
from app.schemas.telemetry import TelemetryPayload
from app.core.config import settings

class FeatureFusionEngine:
    """
    Mesin fusi fitur temporal untuk ekstraksi fitur dalam jendela waktu geser (sliding window).
    Menghasilkan vektor fitur 14 dimensi.
    """
    def __init__(self, window_size: int = settings.sliding_window_size):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def process(self, telemetry: TelemetryPayload) -> Optional[np.ndarray]:
        sensors = telemetry.raw_sensors
        
        # We store tuples of (hr, spo2, temp)
        data_point = np.array([sensors.heart_rate, sensors.spo2, sensors.temperature])
        self.buffer.append(data_point)
        
        if len(self.buffer) < self.window_size:
            return None
            
        return self._compute_features()
        
    def _compute_features(self) -> np.ndarray:
        data = np.array(self.buffer)  # shape: (window_size, 3)
        
        # 1-3: Current raw values (mean of the window or just the last)
        # Using the last point as current value
        current_values = data[-1]
        
        # 4-6: Moving average
        moving_avg = np.mean(data, axis=0)
        
        # 7-9: Variance
        variance = np.var(data, axis=0)
        
        # 10-12: Delta (max - min in window)
        delta = np.max(data, axis=0) - np.min(data, axis=0)
        
        # 13-14: Rate of change (slope) for HR and SpO2
        x = np.arange(self.window_size)
        hr_slope = np.polyfit(x, data[:, 0], 1)[0]
        spo2_slope = np.polyfit(x, data[:, 1], 1)[0]
        
        feature_vector = np.concatenate([
            current_values,
            moving_avg,
            variance,
            delta,
            [hr_slope, spo2_slope]
        ])
        
        return feature_vector
        
    def reset(self):
        """Menghapus isi buffer."""
        self.buffer.clear()
