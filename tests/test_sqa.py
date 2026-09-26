from app.services.sqa import SignalQualityAssessor

def test_normal_reading_returns_good_quality(sample_normal_telemetry):
    assessor = SignalQualityAssessor()
    result = assessor.assess(sample_normal_telemetry)
    assert result.score == 1.0
    assert result.state == "GOOD"
    assert len(result.reasons) == 0

def test_out_of_range_hr_returns_poor_quality(sample_normal_telemetry):
    assessor = SignalQualityAssessor()
    sample_normal_telemetry.raw_sensors.heart_rate = 20.0
    result = assessor.assess(sample_normal_telemetry)
    assert "HR out of range" in result.reasons
    assert result.score < 1.0

def test_out_of_range_spo2_returns_poor_quality(sample_normal_telemetry):
    assessor = SignalQualityAssessor()
    sample_normal_telemetry.raw_sensors.spo2 = 65.0
    result = assessor.assess(sample_normal_telemetry)
    assert "SpO2 out of range" in result.reasons

def test_sudden_hr_jump_returns_poor_quality(sample_normal_telemetry):
    assessor = SignalQualityAssessor()
    prev = sample_normal_telemetry.model_copy(deep=True)
    prev.timestamp = 0
    prev.raw_sensors.heart_rate = 75.0
    
    current = sample_normal_telemetry.model_copy(deep=True)
    current.timestamp = 1000
    current.raw_sensors.heart_rate = 120.0
    
    result = assessor.assess(current, prev)
    assert "Sudden HR jump" in result.reasons

def test_zero_ppg_amplitude_returns_poor_quality(sample_normal_telemetry):
    assessor = SignalQualityAssessor()
    sample_normal_telemetry.sensor_status.ppg_amplitude = 50.0
    result = assessor.assess(sample_normal_telemetry)
    assert "Low PPG amplitude" in result.reasons

def test_boundary_values(sample_normal_telemetry):
    assessor = SignalQualityAssessor()
    sample_normal_telemetry.raw_sensors.heart_rate = assessor.hr_min
    result = assessor.assess(sample_normal_telemetry)
    assert result.state == "GOOD"
