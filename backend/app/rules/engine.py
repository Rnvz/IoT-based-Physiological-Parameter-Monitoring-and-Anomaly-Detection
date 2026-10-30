from typing import Optional, Tuple
from app.config import settings

def evaluate_rules(
    hr: Optional[float],
    spo2: Optional[float],
    temp: Optional[float]
) -> Tuple[int, str]:
    """
    Evaluates rule-based severity from configurable thresholds in config/settings.
    NOTE: Thresholds are demo values to be set from the user's own baseline recordings
    and are not clinical.
    Returns: (severity: 0|1|2, label: "NORMAL"|"ANOMALY"|"NO_DATA")
    """
    if hr is None or spo2 is None or temp is None:
        return 0, "NO_DATA"

    temp_crit = False
    if settings.rule_temp_high_alert is not None and temp >= settings.rule_temp_high_alert:
        temp_crit = True
    if settings.rule_temp_low_alert is not None and temp <= settings.rule_temp_low_alert:
        temp_crit = True

    if (
        (settings.rule_hr_low_alert is not None and hr < settings.rule_hr_low_alert)
        or (settings.rule_hr_high_alert is not None and hr > settings.rule_hr_high_alert)
        or (settings.rule_spo2_alert is not None and spo2 < settings.rule_spo2_alert)
        or temp_crit
    ):
        return 2, "ANOMALY"

    temp_warn = False
    if settings.rule_temp_high_warn is not None and temp >= settings.rule_temp_high_warn:
        temp_warn = True
    if settings.rule_temp_low_warn is not None and temp <= settings.rule_temp_low_warn:
        temp_warn = True

    if (
        (settings.rule_hr_low_warn is not None and hr < settings.rule_hr_low_warn)
        or (settings.rule_hr_high_warn is not None and hr > settings.rule_hr_high_warn)
        or (settings.rule_spo2_warn is not None and spo2 < settings.rule_spo2_warn)
        or temp_warn
    ):
        return 1, "ANOMALY"

    return 0, "NORMAL"
