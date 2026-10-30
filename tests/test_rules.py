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

def test_rules_severity_1_warning_band(mocker):
    sev, _ = evaluate_rules(hr=53.0, spo2=98.0, temp=36.5)
    assert sev == 1

    sev, _ = evaluate_rules(hr=97.0, spo2=98.0, temp=36.5)
    assert sev == 1

    sev, _ = evaluate_rules(hr=75.0, spo2=92.0, temp=36.5)
    assert sev == 1

    sev, _ = evaluate_rules(hr=75.0, spo2=98.0, temp=37.7)
    assert sev == 1

    mocker.patch.object(settings, "rule_temp_low_warn", 32.0)
    sev, _ = evaluate_rules(hr=75.0, spo2=98.0, temp=31.8)
    assert sev == 1

def test_rules_severity_2_critical_alert(mocker):
    sev, label = evaluate_rules(hr=45.0, spo2=98.0, temp=36.5)
    assert sev == 2
    assert label == "ANOMALY"

    sev, label = evaluate_rules(hr=130.0, spo2=98.0, temp=36.5)
    assert sev == 2
    assert label == "ANOMALY"

    sev, label = evaluate_rules(hr=75.0, spo2=88.0, temp=36.5)
    assert sev == 2
    assert label == "ANOMALY"

    sev, label = evaluate_rules(hr=75.0, spo2=98.0, temp=38.5)
    assert sev == 2
    assert label == "ANOMALY"

    mocker.patch.object(settings, "rule_temp_low_alert", 30.0)
    sev, label = evaluate_rules(hr=75.0, spo2=98.0, temp=29.5)
    assert sev == 2
    assert label == "ANOMALY"

def test_rules_realistic_skin_temperature_regression():
    """
    Regression test: realistic skin temperatures (31.0 - 34.5 C) on healthy fingers
    must NOT constantly trigger false alarms when low-temp rules are disabled by default.
    """
    assert settings.rule_temp_low_alert is None
    assert settings.rule_temp_low_warn is None

    realistic_temperatures = [31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5]
    for t in realistic_temperatures:
        sev, label = evaluate_rules(hr=75.0, spo2=98.0, temp=t)
        assert sev == 0, f"Expected severity 0 for realistic skin temp {t}°C, got {sev}"
        assert label == "NORMAL"

def test_rules_boundary_precedence():
    """Severity 2 (critical) must take precedence over Severity 1 (warning)."""
    sev, _ = evaluate_rules(hr=97.0, spo2=85.0, temp=36.5)
    assert sev == 2
