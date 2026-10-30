import pytest
from app.rules.arbitrator import arbitrate, ArbitrationResult
from app.ml.inference import MLInferenceResult

def test_arbitrator_rule_mode():
    ml_res = MLInferenceResult(
        raw_score=0.35,
        raw_severity=0,
        debounced_severity=0,
        label="NORMAL",
        model_version="if_v1",
        is_warmup=False
    )
    # Mode RULE: published follows rule_severity even if ML differs
    res = arbitrate(
        rule_severity=2,
        rule_label="ANOMALY",
        ml_result=ml_res,
        mode="RULE"
    )
    assert res.decision_mode == "RULE"
    assert res.decision_source == "RULE"
    assert res.published_severity == 2
    assert res.published_label == "ANOMALY"
    assert res.rule_severity == 2
    assert res.ml_severity == 0
    assert res.ml_score == 0.35
    assert res.is_fallback is False

def test_arbitrator_ml_mode_available():
    ml_res = MLInferenceResult(
        raw_score=0.05,
        raw_severity=2,
        debounced_severity=2,
        label="ANOMALY",
        model_version="if_v1",
        is_warmup=False
    )
    # Mode ML: published follows ML severity
    res = arbitrate(
        rule_severity=0,
        rule_label="NORMAL",
        ml_result=ml_res,
        mode="ML"
    )
    assert res.decision_mode == "ML"
    assert res.decision_source == "ML"
    assert res.published_severity == 2
    assert res.published_label == "ANOMALY"
    assert res.rule_severity == 0
    assert res.ml_severity == 2
    assert res.ml_score == 0.05
    assert res.is_fallback is False

def test_arbitrator_ml_mode_fallback_on_warmup():
    """When ML is in warm-up (is_warmup=True, raw_severity=None), must fall back to RULE."""
    ml_res = MLInferenceResult(
        label="NO_DATA",
        raw_severity=None,
        is_warmup=True
    )
    res = arbitrate(
        rule_severity=1,
        rule_label="ANOMALY",
        ml_result=ml_res,
        mode="ML"
    )
    assert res.decision_mode == "ML"
    assert res.decision_source == "RULE_FALLBACK"
    assert res.published_severity == 1
    assert res.published_label == "ANOMALY"
    assert res.rule_severity == 1
    assert res.ml_severity is None
    assert res.is_fallback is True

def test_arbitrator_ml_mode_fallback_on_none():
    """When ml_result is completely None (e.g. error or sensor error), fall back to RULE."""
    res = arbitrate(
        rule_severity=2,
        rule_label="ANOMALY",
        ml_result=None,
        mode="ML"
    )
    assert res.decision_mode == "ML"
    assert res.decision_source == "RULE_FALLBACK"
    assert res.published_severity == 2
    assert res.published_label == "ANOMALY"
    assert res.is_fallback is True
