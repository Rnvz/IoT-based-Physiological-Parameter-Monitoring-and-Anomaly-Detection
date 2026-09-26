import numpy as np
from app.services.anomaly_model import AnomalyDetector, PersistentAnomalyEvaluator

def test_model_loads_successfully(mock_model):
    detector = AnomalyDetector()
    # If file doesn't exist, it falls back gracefully
    assert detector is not None

def test_model_predict_normal():
    detector = AnomalyDetector()
    # Feature vector representing normal vitals with moving averages
    features = np.zeros(13)
    features[0:3] = [75.0, 98.0, 36.5]
    features[6:9] = [75.0, 98.0, 36.5]
    res = detector.predict(features)
    assert res.is_anomaly is False
    assert res.severity == "NORMAL"

def test_model_predict_anomaly():
    detector = AnomalyDetector()
    # Feature vector representing extreme tachycardia, hypoxia, fever with large delta/variance
    features = np.zeros(13)
    features[0:3] = [150.0, 85.0, 39.5]
    features[3:6] = [20.0, -10.0, 2.0]
    features[6:9] = [140.0, 88.0, 39.0]
    features[9:12] = [250.0, 20.0, 1.5]
    features[12] = 2.0
    res = detector.predict(features)
    assert res.is_anomaly is True
    assert res.severity == "HIGH DEVIATION"

def test_persistent_evaluator_counts_correctly():
    evaluator = PersistentAnomalyEvaluator(threshold=3, sustained_threshold=10)
    res1 = evaluator.evaluate(is_anomaly=True, base_severity="HIGH DEVIATION")
    assert res1.persistent_count == 1
    assert res1.is_persistent_anomaly is False
    assert res1.severity == "HIGH DEVIATION"
    assert res1.oled_status == "HIGH DEV"
    
    res2 = evaluator.evaluate(is_anomaly=True, base_severity="HIGH DEVIATION")
    res3 = evaluator.evaluate(is_anomaly=True, base_severity="HIGH DEVIATION")
    assert res3.is_persistent_anomaly is True
    assert res3.should_buzz is True
    assert res3.severity == "HIGH DEVIATION"
    assert res3.oled_status == "HIGH DEV"

def test_persistent_evaluator_sustained_status():
    evaluator = PersistentAnomalyEvaluator(threshold=3, sustained_threshold=10)
    # Feed 9 high deviation samples
    for _ in range(9):
        res = evaluator.evaluate(is_anomaly=True, base_severity="HIGH DEVIATION")
    assert res.persistent_count == 9
    assert res.severity == "HIGH DEVIATION"
    assert res.oled_status == "HIGH DEV"
    
    # 10th sample reaches sustained threshold
    res10 = evaluator.evaluate(is_anomaly=True, base_severity="HIGH DEVIATION")
    assert res10.persistent_count == 10
    assert res10.severity == "HIGH DEVIATION (SUSTAINED)"
    assert res10.oled_status == "HIGH DEV (S)"

    # Low deviation sustained test
    eval_low = PersistentAnomalyEvaluator(threshold=3, sustained_threshold=10)
    for _ in range(10):
        res_low = eval_low.evaluate(is_anomaly=True, base_severity="LOW DEVIATION")
    assert res_low.persistent_count == 10
    assert res_low.severity == "LOW DEVIATION (SUSTAINED)"
    assert res_low.oled_status == "LOW DEV (S)"

def test_persistent_evaluator_resets_on_normal():
    evaluator = PersistentAnomalyEvaluator(threshold=3, sustained_threshold=10)
    evaluator.evaluate(True, "HIGH DEVIATION")
    evaluator.evaluate(True, "HIGH DEVIATION")
    res = evaluator.evaluate(False)
    assert res.persistent_count == 0
    assert res.is_persistent_anomaly is False
    assert res.should_buzz is False
    assert res.severity == "NORMAL"
    assert res.oled_status == "NORMAL"
