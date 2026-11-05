"""
ml_eval.inject: Labelled synthetic anomaly injection on copies of test sessions.

Injects three canonical failure modes:
1. HR_SPIKE   : Tachycardia / sudden heart rate spike.
2. SPO2_DROP  : Acute hypoxemia desaturation (< 90%).
3. FLAT_SIGNAL: Sensor flatline / freeze (constant signal with zero variance).

EVALUATION PROTOCOL RULES:
- Never modifies original test DataFrames (operates strictly on deep copies).
- Labels every sample with is_synthetic (True/False) and anomaly_type.
- Real recordings and synthetic injections must always be reported separately.
"""

from typing import Tuple, List, Optional
import copy
import pandas as pd
import numpy as np

def inject_synthetic_anomalies(
    test_df: pd.DataFrame,
    seed: int = 42,
    episode_length: int = 20,
    anomalies: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates two distinct evaluation test sets from test_df:
    1. real_test_df     : Clean copy of physical test recordings (y_true=0, is_synthetic=False).
    2. injected_test_df : Copy with labelled synthetic anomalies injected.

    The original DataFrame is never modified.
    """
    if test_df.empty:
        raise ValueError("Cannot inject anomalies into an empty DataFrame.")

    if anomalies is None:
        anomalies = ["HR_SPIKE", "SPO2_DROP", "FLAT_SIGNAL"]

    rng = np.random.default_rng(seed)

    real_test = test_df.copy(deep=True)
    real_test["y_true"] = 0
    real_test["is_synthetic"] = False
    real_test["anomaly_type"] = "NONE"

    injected_test = test_df.copy(deep=True)
    injected_test["y_true"] = 0
    injected_test["is_synthetic"] = False
    injected_test["anomaly_type"] = "NONE"

    for col in ["hr", "spo2", "temp"]:
        injected_test[col] = injected_test[col].astype(float)

    sessions = injected_test["session_id"].unique() if "session_id" in injected_test.columns else ["default"]

    for sid in sessions:
        mask = injected_test["session_id"] == sid if "session_id" in injected_test.columns else slice(None)
        sess_indices = injected_test[mask].index.tolist()
        n_samples = len(sess_indices)

        if n_samples < episode_length * 2:
            continue

        max_episodes = min(len(anomalies), n_samples // (episode_length + 15))
        if max_episodes < 1:
            continue

        selected_anomalies = anomalies[:max_episodes]
        step = n_samples // (len(selected_anomalies) + 1)

        for i, a_type in enumerate(selected_anomalies):
            start_pos = (i + 1) * step
            end_pos = min(start_pos + episode_length, n_samples - 1)
            target_idx = sess_indices[start_pos:end_pos]

            if len(target_idx) == 0:
                continue

            if a_type == "HR_SPIKE":
                spike = rng.uniform(45.0, 60.0)
                injected_test.loc[target_idx, "hr"] = np.clip(
                    injected_test.loc[target_idx, "hr"] + spike, 40.0, 185.0
                )
            elif a_type == "SPO2_DROP":
                target_spo2 = rng.uniform(81.0, 87.0)
                injected_test.loc[target_idx, "spo2"] = target_spo2
            elif a_type == "FLAT_SIGNAL":
                freeze_hr = float(injected_test.loc[target_idx[0], "hr"])
                freeze_spo2 = float(injected_test.loc[target_idx[0], "spo2"])
                freeze_temp = float(injected_test.loc[target_idx[0], "temp"])
                injected_test.loc[target_idx, "hr"] = freeze_hr
                injected_test.loc[target_idx, "spo2"] = freeze_spo2
                injected_test.loc[target_idx, "temp"] = freeze_temp

            injected_test.loc[target_idx, "y_true"] = 1
            injected_test.loc[target_idx, "is_synthetic"] = True
            injected_test.loc[target_idx, "anomaly_type"] = a_type

    return real_test, injected_test
