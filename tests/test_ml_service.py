from app.ml.severity import ThresholdEvaluator, DevicePersistentEvaluator
from app.ml.inference import MLService

def test_threshold_evaluator_sign_convention_and_mapping():
    evaluator = ThresholdEvaluator()
    assert evaluator.thr_013 > evaluator.thr_005

    sev, label = evaluator.score_to_raw_severity(evaluator.thr_013 + 0.1)
    assert sev == 0
    assert label == "NORMAL"

    mid_score = (evaluator.thr_013 + evaluator.thr_005) / 2.0
    sev, label = evaluator.score_to_raw_severity(mid_score)
    assert sev == 1
    assert label == "ANOMALY"

    sev, label = evaluator.score_to_raw_severity(evaluator.thr_005 - 0.05)
    assert sev == 2
    assert label == "ANOMALY"

def test_device_persistent_evaluator_debounce_and_strict_reset():
    evaluator = DevicePersistentEvaluator(debounce_count=3)
    
    # Critical sequence (2): suppressed for samples 1 and 2, triggers on sample 3
    sev1 = evaluator.evaluate(raw_severity=2)
    assert sev1 == 0, f"Expected 0 for 1st consecutive anomaly, got {sev1}"
    assert evaluator.consecutive_count == 1

    sev2 = evaluator.evaluate(raw_severity=2)
    assert sev2 == 0, f"Expected 0 for 2nd consecutive anomaly, got {sev2}"
    assert evaluator.consecutive_count == 2

    sev3 = evaluator.evaluate(raw_severity=2)
    assert sev3 == 2, f"Expected 2 on 3rd consecutive anomaly, got {sev3}"
    assert evaluator.consecutive_count == 3

    sev_norm = evaluator.evaluate(raw_severity=0)
    assert sev_norm == 0
    assert evaluator.consecutive_count == 0

def test_device_persistent_evaluator_single_spike_suppressed():
    evaluator = DevicePersistentEvaluator(debounce_count=3)

    # 1-sample spike in warning zone (severity 1) must be suppressed
    sev = evaluator.evaluate(raw_severity=1)
    assert sev == 0, f"Expected single spike to be suppressed (0), got {sev}"
    assert evaluator.consecutive_count == 1

    # Returns to normal -> resets
    assert evaluator.evaluate(raw_severity=0) == 0
    assert evaluator.consecutive_count == 0

    # 3 consecutive warning anomalies -> triggers 1 on 3rd
    assert evaluator.evaluate(raw_severity=1) == 0
    assert evaluator.evaluate(raw_severity=1) == 0
    assert evaluator.evaluate(raw_severity=1) == 1


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

    for i in range(10):
        service.process_reading(device_id=dev, hr=75.0, spo2=98.0, temp=36.5, finger=True, timestamp=10.0 + i)

    res_off = service.process_reading(device_id=dev, hr=None, spo2=None, temp=36.5, finger=False)
    assert res_off.label == "NO_DATA"

    res_next = service.process_reading(device_id=dev, hr=75.0, spo2=98.0, temp=36.5, finger=True, timestamp=30.0)
    assert res_next.label == "NO_DATA"
    assert res_next.is_warmup is True


def test_default_artifact_paths_exist():
    import os
    from app.config import settings

    assert os.path.exists(settings.model_path), f"Default model_path does not exist: {settings.model_path}"
    assert os.path.exists(settings.scaler_path), f"Default scaler_path does not exist: {settings.scaler_path}"
    assert os.path.exists(settings.thresholds_path), f"Default thresholds_path does not exist: {settings.thresholds_path}"
    assert os.path.exists(settings.model_dir), f"Default model_dir does not exist: {settings.model_dir}"
    assert os.path.exists(os.path.join(settings.model_dir, "metadata.json")), f"metadata.json missing in {settings.model_dir}"

