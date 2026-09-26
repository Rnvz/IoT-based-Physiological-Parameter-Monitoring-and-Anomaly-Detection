import numpy as np
from app.services.feature_fusion import FeatureFusionEngine

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
