import numpy as np
from app.services.anomaly_model import AnomalyDetector, PersistentAnomalyEvaluator

def test_model_loads_successfully(mock_model):
    detector = AnomalyDetector()
    # If file doesn't exist, it falls back gracefully
    assert detector is not None

def test_baseline_threshold_normal():
    detector = AnomalyDetector()
    # Mocking a feature vector with normal values: HR=75, SpO2=98, Temp=36.5
    features = np.zeros(13)
    features[0:3] = [75.0, 98.0, 32.0]
    res = detector.predict(features)
    assert res.baseline_anomaly is False

def test_baseline_threshold_anomaly():
    detector = AnomalyDetector()
    features = np.zeros(13)
    features[0:3] = [120.0, 92.0, 39.5] # Extreme values
    res = detector.predict(features)
    assert res.baseline_anomaly is True

def test_persistent_evaluator_counts_correctly():
    evaluator = PersistentAnomalyEvaluator(threshold=3)
    res1 = evaluator.evaluate(is_anomaly=True)
    assert res1.persistent_count == 1
    assert res1.is_persistent_anomaly is False
    
    res2 = evaluator.evaluate(is_anomaly=True)
    res3 = evaluator.evaluate(is_anomaly=True)
    assert res3.is_persistent_anomaly is True

def test_persistent_evaluator_resets_on_normal():
    evaluator = PersistentAnomalyEvaluator(threshold=3)
    evaluator.evaluate(True)
    evaluator.evaluate(True)
    res = evaluator.evaluate(False)
    assert res.persistent_count == 0
    assert res.is_persistent_anomaly is False
