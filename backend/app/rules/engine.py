from typing import Optional, Tuple
from app.config import settings

def evaluate_rules(
    hr: Optional[float],
    spo2: Optional[float],
    temp: Optional[float]
) -> Tuple[int, str]:
    """
    Evaluates rule-based severity from configurable thresholds in config/settings.
    No hardcoding.
    Returns: (severity: 0|1|2, label: "NORMAL"|"ANOMALY"|"NO_DATA")
    """
    if hr is None or spo2 is None or temp is None:
        return 0, "NO_DATA"

    # Severity 2 (Critical Alert):
    # HR < 50 or > 100; SpO2 < 90; Temp >= 38.0 or <= 35.0
    if (
        hr < settings.rule_hr_low_alert
        or hr > settings.rule_hr_high_alert
        or spo2 < settings.rule_spo2_alert
        or temp >= settings.rule_temp_high_alert
        or temp <= settings.rule_temp_low_alert
    ):
        return 2, "ANOMALY"

    # Severity 1 (Warning / Borderline Band):
    # HR < 55 or > 95; SpO2 < 94; Temp >= 37.5 or <= 35.5
    if (
        hr < settings.rule_hr_low_warn
        or hr > settings.rule_hr_high_warn
        or spo2 < settings.rule_spo2_warn
        or temp >= settings.rule_temp_high_warn
        or temp <= settings.rule_temp_low_warn
    ):
        return 1, "ANOMALY"

    return 0, "NORMAL"
