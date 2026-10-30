import pytest
from app.rules.engine import evaluate_rules
from app.config import settings

def test_rules_normal_values():
    sev, label = evaluate_rules(hr=75.0, spo2=98.0, temp=36.5)
    assert sev == 0
    assert label == "NORMAL"

def test_rules_missing_values_return_no_data():
    sev, label = evaluate_rules(hr=None, spo2=98.0, temp=36.5)
    assert sev == 0
    assert label == "NO_DATA"

    sev, label = evaluate_rules(hr=75.0, spo2=None, temp=36.5)
    assert sev == 0
    assert label == "NO_DATA"

    sev, label = evaluate_rules(hr=75.0, spo2=98.0, temp=None)
    assert sev == 0
    assert label == "NO_DATA"

def test_rules_severity_1_warning_band():
    # HR in warning band [50..55] or [95..100]
    sev, _ = evaluate_rules(hr=53.0, spo2=98.0, temp=36.5)
    assert sev == 1

    sev, _ = evaluate_rules(hr=97.0, spo2=98.0, temp=36.5)
    assert sev == 1

    # SpO2 in warning band [90..94]
    sev, _ = evaluate_rules(hr=75.0, spo2=92.0, temp=36.5)
    assert sev == 1

    # Skin temp in warning band [37.5..38.0] or [35.0..35.5]
    sev, _ = evaluate_rules(hr=75.0, spo2=98.0, temp=37.7)
    assert sev == 1

    sev, _ = evaluate_rules(hr=75.0, spo2=98.0, temp=35.2)
    assert sev == 1

def test_rules_severity_2_critical_alert():
    # HR critical (< 50 or > 100)
    sev, label = evaluate_rules(hr=45.0, spo2=98.0, temp=36.5)
    assert sev == 2
    assert label == "ANOMALY"

    sev, label = evaluate_rules(hr=130.0, spo2=98.0, temp=36.5)
    assert sev == 2
    assert label == "ANOMALY"

    # SpO2 critical (< 90)
    sev, label = evaluate_rules(hr=75.0, spo2=88.0, temp=36.5)
    assert sev == 2
    assert label == "ANOMALY"

    # Skin temp critical (>= 38.0 or <= 35.0)
    sev, label = evaluate_rules(hr=75.0, spo2=98.0, temp=38.5)
    assert sev == 2
    assert label == "ANOMALY"

    sev, label = evaluate_rules(hr=75.0, spo2=98.0, temp=34.5)
    assert sev == 2
    assert label == "ANOMALY"

def test_rules_boundary_precedence():
    """Severity 2 (critical) must take precedence over Severity 1 (warning)."""
    # SpO2 is critical (<90), while HR is only warning (97.0)
    sev, _ = evaluate_rules(hr=97.0, spo2=85.0, temp=36.5)
    assert sev == 2
