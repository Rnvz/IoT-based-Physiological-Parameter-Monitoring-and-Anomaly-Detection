import time
from collections import deque
from typing import Optional, Dict
import numpy as np
from app.config import settings

class DeviceRollingWindow:
    """
    Maintains a rolling window of vital sign readings for a single device.
    Extracts 13-dimensional temporal feature vector when buffer is full (15 samples).
    """

    def __init__(self, device_id: str, window_size: int = settings.window_size, max_gap_seconds: float = settings.max_message_gap_seconds):
        self.device_id = device_id
        self.window_size = window_size
        self.max_gap_seconds = max_gap_seconds
        self.buffer = deque(maxlen=window_size)
        self.last_timestamp: Optional[float] = None

    def reset(self):
        """Resets the buffer and timestamp tracking."""
        self.buffer.clear()
        self.last_timestamp = None

    def push(self, hr: float, spo2: float, temp: float, timestamp: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Pushes a valid reading into the window.
        Resets automatically if the time gap from the last message exceeds max_gap_seconds.
        Returns 13-dim feature vector if buffer has reached window_size, otherwise None.
        """
        now = timestamp if timestamp is not None else time.time()

        if self.last_timestamp is not None:
            gap = now - self.last_timestamp
            if gap > self.max_gap_seconds:
                self.reset()

        self.last_timestamp = now
        self.buffer.append(np.array([hr, spo2, temp], dtype=np.float64))

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
            abs(delta[0]) / max(float(moving_avg[0]), 1e-6) +
            abs(delta[1]) / max(float(moving_avg[1]), 1e-6) +
            abs(delta[2]) / max(float(moving_avg[2]), 1e-6)
        )

        return np.concatenate([
            current_values,
            delta,
            moving_avg,
            variance,
            [rate_of_change]
        ])

class DeviceWindowManager:
    """Manages independent rolling windows for each device_id."""

    def __init__(self, window_size: int = settings.window_size, max_gap_seconds: float = settings.max_message_gap_seconds):
        self.window_size = window_size
        self.max_gap_seconds = max_gap_seconds
        self._windows: Dict[str, DeviceRollingWindow] = {}

    def get_window(self, device_id: str) -> DeviceRollingWindow:
        if device_id not in self._windows:
            self._windows[device_id] = DeviceRollingWindow(
                device_id=device_id,
                window_size=self.window_size,
                max_gap_seconds=self.max_gap_seconds
            )
        return self._windows[device_id]

    def reset_device(self, device_id: str):
        if device_id in self._windows:
            self._windows[device_id].reset()

    def reset_all(self):
        for window in self._windows.values():
            window.reset()
        self._windows.clear()
