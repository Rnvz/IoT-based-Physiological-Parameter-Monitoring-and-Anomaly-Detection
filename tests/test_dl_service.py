import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.ml.dl_service import DLService, DeviceDLState, DLInferenceResult

try:
    from dl.artifacts.dl_v1.numpy_forward import EdgeAutoencoderInference
except ImportError:
    from ml.artifacts.dl_v1.numpy_forward import EdgeAutoencoderInference

def test_dl_service_matches_numpy_forward_on_fixed_samples():
    service = DLService()
    edge = EdgeAutoencoderInference()

    test_cases = [
        {"Heart_Rate": 75.0, "SpO2": 98.0},
        {"Heart_Rate": 110.0, "SpO2": 95.0},
        {"Heart_Rate": 60.0, "SpO2": 89.0},
    ]

    for tc in test_cases:
        dl_res = service.process_reading(
            device_id="test-dev",
            hr=tc["Heart_Rate"],
            spo2=tc["SpO2"],
            temp=36.5,
            finger=True
        )
        edge_res = edge.predict_sample(tc)
        
        # Parity difference must be less than 1e-5
        assert dl_res.raw_score == pytest.approx(edge_res["score"], abs=1e-5)

def test_dl_service_persistence_escalation_n3():
    service = DLService()
    dev = "dev-persistence"
    service.reset_device(dev)

    # Normal sample (severity 0)
    res0 = service.process_reading(dev, hr=75.0, spo2=98.0, finger=True)
    assert res0.raw_severity == 0
    assert res0.debounced_severity == 0
    assert res0.label == "NORMAL"

    # Step 1: Anomaly crossing warning threshold
    # HR=105, SpO2=95 -> score > thr_warn (0.013571)
    res1 = service.process_reading(dev, hr=105.0, spo2=95.0, finger=True)
    assert res1.raw_severity >= 1
    # Severity must NOT rise yet (consecutive=1 < 3)
    assert res1.debounced_severity == 0
    assert res1.label == "NORMAL"

    # Step 2: Anomaly continues
    res2 = service.process_reading(dev, hr=105.0, spo2=95.0, finger=True)
    assert res2.raw_severity >= 1
    # Severity must NOT rise yet (consecutive=2 < 3)
    assert res2.debounced_severity == 0
    assert res2.label == "NORMAL"

    # Step 3: 3rd consecutive anomaly
    res3 = service.process_reading(dev, hr=105.0, spo2=95.0, finger=True)
    assert res3.raw_severity >= 1
    # Severity rises after N=3 consecutive samples!
    assert res3.debounced_severity >= 1
    assert res3.label == "ANOMALY"

def test_dl_service_reset_on_finger_false():
    service = DLService()
    dev = "dev-finger"
    service.reset_device(dev)

    # 3 consecutive anomalies to escalate severity
    for _ in range(3):
        res = service.process_reading(dev, hr=115.0, spo2=90.0, finger=True)
    assert res.debounced_severity >= 1

    # Finger removed -> reset
    res_off = service.process_reading(dev, hr=None, spo2=None, finger=False)
    assert res_off.label == "NO_DATA"

    # Reattach finger with normal vitals -> immediately 0
    res_reattach = service.process_reading(dev, hr=75.0, spo2=98.0, finger=True)
    assert res_reattach.raw_severity == 0
    assert res_reattach.debounced_severity == 0

def test_dl_service_reset_on_data_gap():
    service = DLService()
    dev = "dev-gap"
    service.reset_device(dev)

    t0 = 1000.0
    # 2 warning samples at t=1000 and t=1001
    service.process_reading(dev, hr=105.0, spo2=95.0, finger=True, timestamp=t0)
    res2 = service.process_reading(dev, hr=105.0, spo2=95.0, finger=True, timestamp=t0 + 1.0)
    assert res2.debounced_severity == 0

    # Gap of 10 seconds (> 5.0s max gap)
    # Consecutive counter must reset, so this is sample 1, not sample 3!
    res3 = service.process_reading(dev, hr=105.0, spo2=95.0, finger=True, timestamp=t0 + 11.0)
    assert res3.debounced_severity == 0
    state = service._get_state(dev)
    assert state.consecutive_warn == 1

def test_dl_service_fallback_on_exception():
    service = DLService()
    dev = "dev-exc"
    service.reset_device(dev)

    # Simulate ONNX session raising an error
    with patch.object(service.ort_session, "run", side_effect=RuntimeError("Simulated ONNX error")):
        res = service.process_reading(dev, hr=75.0, spo2=98.0, finger=True)
        assert res.label == "SENSOR_ERROR"
        assert res.raw_severity is None
        assert res.debounced_severity is None
