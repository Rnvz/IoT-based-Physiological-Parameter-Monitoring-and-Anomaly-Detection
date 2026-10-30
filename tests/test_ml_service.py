import pytest
from app.ml.severity import ThresholdEvaluator, DevicePersistentEvaluator
from app.ml.inference import MLService, get_ml_service

def test_threshold_evaluator_sign_convention_and_mapping():
    evaluator = ThresholdEvaluator()
    # Ensure thresholds loaded from file or defaults
    assert evaluator.thr_013 > evaluator.thr_005

    # Score well above thr_013 -> Normal (0)
    sev, label = evaluator.score_to_raw_severity(evaluator.thr_013 + 0.1)
    assert sev == 0
    assert label == "NORMAL"

    # Score between thr_005 and thr_013 -> Warning (1)
    mid_score = (evaluator.thr_013 + evaluator.thr_005) / 2.0
    sev, label = evaluator.score_to_raw_severity(mid_score)
    assert sev == 1
    assert label == "ANOMALY"

    # Score below thr_005 -> Critical (2)
    sev, label = evaluator.score_to_raw_severity(evaluator.thr_005 - 0.05)
    assert sev == 2
    assert label == "ANOMALY"

def test_device_persistent_evaluator_debounce_and_strict_reset():
    evaluator = DevicePersistentEvaluator(debounce_count=3)
    
    # 1st anomaly: suppressed to 1
    sev1 = evaluator.evaluate(raw_severity=2)
    assert sev1 == 1
    assert evaluator.consecutive_count == 1

    # 2nd anomaly: suppressed to 1
    sev2 = evaluator.evaluate(raw_severity=2)
    assert sev2 == 1
    assert evaluator.consecutive_count == 2

    # 3rd anomaly: reached debounce limit (3) -> emits full severity 2
    sev3 = evaluator.evaluate(raw_severity=2)
    assert sev3 == 2
    assert evaluator.consecutive_count == 3

    # Normal reading (0) immediately causes strict reset
    sev_norm = evaluator.evaluate(raw_severity=0)
    assert sev_norm == 0
    assert evaluator.consecutive_count == 0

def test_ml_service_warmup_returns_no_data():
    """During warmup (< 15 samples), MLService must return NO_DATA, never NORMAL."""
    service = MLService()
    dev = "test_device_warmup"

    for i in range(14):
        res = service.process_reading(
            device_id=dev,
            hr=75.0,
            spo2=98.0,
            temp=36.5,
            finger=True,
            timestamp=100.0 + i
        )
        assert res.label == "NO_DATA", f"Expected NO_DATA during warmup at sample {i+1}"
        assert res.raw_severity is None
        assert res.is_warmup is True

    # 15th sample completes the window
    res15 = service.process_reading(
        device_id=dev,
        hr=75.0,
        spo2=98.0,
        temp=36.5,
        finger=True,
        timestamp=114.0
    )
    assert res15.is_warmup is False
    assert res15.raw_severity is not None
    assert res15.raw_score is not None

def test_ml_service_finger_false_resets_window():
    service = MLService()
    dev = "test_device_finger_off"

    # Push 10 samples
    for i in range(10):
        service.process_reading(device_id=dev, hr=75.0, spo2=98.0, temp=36.5, finger=True, timestamp=10.0 + i)

    # Finger off reading
    res_off = service.process_reading(device_id=dev, hr=None, spo2=None, temp=36.5, finger=False)
    assert res_off.label == "NO_DATA"

    # Next reading should start warm-up from 0 again
    res_next = service.process_reading(device_id=dev, hr=75.0, spo2=98.0, temp=36.5, finger=True, timestamp=30.0)
    assert res_next.label == "NO_DATA"
    assert res_next.is_warmup is True
