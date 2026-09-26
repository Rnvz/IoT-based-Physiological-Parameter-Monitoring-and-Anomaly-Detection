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
        
        # We store tuples of (hr, spo2, temp)
        data_point = np.array([sensors.heart_rate, sensors.spo2, sensors.temperature])
        self.buffer.append(data_point)
        
        if len(self.buffer) < self.window_size:
            return None
            
        return self._compute_features()
        
    def _compute_features(self) -> np.ndarray:
        data = np.array(self.buffer)  # shape: (window_size, 3) -> [HR, SpO2, Temp]
        
        # x1-x3: Current raw values (latest reading)
        current_values = data[-1]  # [HR, SpO2, Temp]
        
        # x4-x6: Delta (difference between last two readings)
        delta = data[-1] - data[-2]  # [HR_Delta, SpO2_Delta, Temp_Delta]
        
        # x7-x9: Moving average over window
        moving_avg = np.mean(data, axis=0)  # [MA_HR, MA_SpO2, MA_Temp]
        
        # x10-x12: Variance over window
        variance = np.var(data, axis=0, ddof=1)  # [Var_HR, Var_SpO2, Var_Temp]
        
        # x13: Rate of change (normalized composite)
        rate_of_change = (
            abs(delta[0]) / max(moving_avg[0], 1e-6) +
            abs(delta[1]) / max(moving_avg[1], 1e-6) +
            abs(delta[2]) / max(moving_avg[2], 1e-6)
        )
        
        # 13-dimensional feature vector (Signal_Quality_Score removed)
        feature_vector = np.concatenate([
            current_values,    # x1-x3
            delta,             # x4-x6
            moving_avg,        # x7-x9
            variance,          # x10-x12
            [rate_of_change]   # x13
        ])
        
        return feature_vector
        
    def reset(self):
        """Menghapus isi buffer."""
        self.buffer.clear()
