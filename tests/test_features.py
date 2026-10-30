import time
import numpy as np
from app.services.feature_fusion import FeatureFusionEngine
from app.ml.device_window import DeviceRollingWindow, DeviceWindowManager

# --- Legacy FeatureFusionEngine Tests ---
def test_buffer_not_ready_returns_none(sample_normal_telemetry):
    engine = FeatureFusionEngine(window_size=3)
    res = engine.process(sample_normal_telemetry)
    assert res is None

def test_buffer_full_returns_feature_vector(sample_normal_telemetry):
    engine = FeatureFusionEngine(window_size=3)
    engine.process(sample_normal_telemetry)
    engine.process(sample_normal_telemetry)
    res = engine.process(sample_normal_telemetry)
    assert res is not None
    assert isinstance(res, np.ndarray)

def test_feature_vector_has_13_dimensions(sample_normal_telemetry):
    engine = FeatureFusionEngine(window_size=3)
    engine.process(sample_normal_telemetry)
    engine.process(sample_normal_telemetry)
    res = engine.process(sample_normal_telemetry)
    assert len(res) == 13

def test_delta_calculation_correctness(sample_normal_telemetry):
    engine = FeatureFusionEngine(window_size=2)
    sample1 = sample_normal_telemetry.model_copy(deep=True)
    sample1.raw_sensors.heart_rate = 70.0
    
    sample2 = sample_normal_telemetry.model_copy(deep=True)
    sample2.raw_sensors.heart_rate = 80.0
    
    engine.process(sample1)
    res = engine.process(sample2)
    # Delta HR = 80 - 70 = 10.0 (index 3: Heart_Rate_Delta)
    assert res[3] == 10.0

def test_moving_average_correctness(sample_normal_telemetry):
    engine = FeatureFusionEngine(window_size=2)
    sample1 = sample_normal_telemetry.model_copy(deep=True)
    sample1.raw_sensors.heart_rate = 70.0
    sample2 = sample_normal_telemetry.model_copy(deep=True)
    sample2.raw_sensors.heart_rate = 80.0
    
    engine.process(sample1)
    res = engine.process(sample2)
    # Mean HR should be 75.0 (index 6: MA_Heart_Rate)
    assert res[6] == 75.0

def test_reset_clears_buffer(sample_normal_telemetry):
    engine = FeatureFusionEngine(window_size=3)
    engine.process(sample_normal_telemetry)
    engine.reset()
    assert len(engine.buffer) == 0


# --- New Rebuilt DeviceRollingWindow & DeviceWindowManager Tests ---

def test_device_rolling_window_warmup():
    """Window of size 15 must return None for samples 1..14."""
    window = DeviceRollingWindow(device_id="esp32-01", window_size=15)
    for i in range(14):
        feat = window.push(hr=75.0, spo2=98.0, temp=36.5, timestamp=100.0 + i)
        assert feat is None, f"Expected None during warmup at step {i+1}"
    
    feat15 = window.push(hr=75.0, spo2=98.0, temp=36.5, timestamp=114.0)
    assert feat15 is not None
    assert len(feat15) == 13
    assert isinstance(feat15, np.ndarray)

def test_device_rolling_window_math_correctness():
    """Verify 13 feature dimensions math."""
    window = DeviceRollingWindow(device_id="esp32-01", window_size=2)
    window.push(hr=70.0, spo2=97.0, temp=36.0, timestamp=10.0)
    feat = window.push(hr=80.0, spo2=99.0, temp=37.0, timestamp=11.0)
    
    assert feat is not None
    # Current [0:3]
    assert np.allclose(feat[0:3], [80.0, 99.0, 37.0])
    # Delta [3:6]
    assert np.allclose(feat[3:6], [10.0, 2.0, 1.0])
    # MA [6:9]
    assert np.allclose(feat[6:9], [75.0, 98.0, 36.5])
    # Variance [9:12] (sample variance for [70, 80] is 50.0)
    assert feat[9] == 50.0

def test_device_rolling_window_reset_on_gap():
    """Verify window resets automatically if gap exceeds max_gap_seconds (5.0s)."""
    window = DeviceRollingWindow(device_id="esp32-01", window_size=3, max_gap_seconds=5.0)
    window.push(hr=75.0, spo2=98.0, temp=36.5, timestamp=10.0)
    window.push(hr=75.0, spo2=98.0, temp=36.5, timestamp=11.0)
    assert len(window.buffer) == 2

    # Now jump 6 seconds ahead (> 5.0s max gap)
    window.push(hr=75.0, spo2=98.0, temp=36.5, timestamp=17.5)
    # Gap triggered a reset before appending, so buffer length is now 1
    assert len(window.buffer) == 1

def test_device_window_manager_isolation():
    """Verify per-device isolation between different device_ids."""
    manager = DeviceWindowManager(window_size=3)
    w_dev1 = manager.get_window("esp32-01")
    w_dev2 = manager.get_window("esp32-02")

    w_dev1.push(hr=70.0, spo2=98.0, temp=36.5, timestamp=1.0)
    w_dev1.push(hr=71.0, spo2=98.0, temp=36.5, timestamp=2.0)

    assert len(w_dev1.buffer) == 2
    assert len(w_dev2.buffer) == 0

    manager.reset_device("esp32-01")
    assert len(w_dev1.buffer) == 0
