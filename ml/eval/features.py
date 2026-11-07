"""
ml_eval.features: Causal rolling window feature engineering.

IMPORTANT PROTOCOL RULE:
1. Features are strictly CAUSAL: computed solely from past and current samples (t-14 to t).
2. Features NEVER cross session boundaries: window buffers reset between sessions.
3. Warm-up rows (first 14 samples of each contiguous segment) are DROPPED.
   Zero-filling warm-up rows is strictly prohibited as it injects artificial anomalies.
4. Window buffers reset immediately if the gap between consecutive samples exceeds 5.0s,
   or if finger is false.

Feature Sets:
- Feature Set 1 (CORE): 12 features (hr, spo2, temp, plus rolling mean, var, delta of each).
  NOTE: Feature Set 1 is the primary set DIRECTLY COMPARABLE to the Rule-Based Baseline,
  as the clinical rule engine operates solely on HR, SpO2, and skin temperature.
- Feature Set 2 (EXTENDED - Contextual Only): 24 features (Core 12 + room_temp, hum, temp_sub_room, plus their rolling stats).
  NOTE: Parameter lingkungan (humidity & room_temp) dialokasikan sebagai data pendukung/kontekstual dan tidak digunakan untuk melatih model fisiologis utama.
"""

from collections import deque
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from ml.eval.sessions import parse_timestamp

DEFAULT_WINDOW_SIZE = 15
DEFAULT_SAMPLE_GAP_SECONDS = 5.0

CORE_FEATURE_NAMES: List[str] = [
    "hr", "spo2", "temp",
    "hr_mean", "spo2_mean", "temp_mean",
    "hr_var", "spo2_var", "temp_var",
    "hr_delta", "spo2_delta", "temp_delta"
]

EXTENDED_FEATURE_NAMES: List[str] = CORE_FEATURE_NAMES + [
    "room_temp", "hum", "temp_sub_room",
    "room_temp_mean", "hum_mean", "temp_sub_room_mean",
    "room_temp_var", "hum_var", "temp_sub_room_var",
    "room_temp_delta", "hum_delta", "temp_sub_room_delta"
]

def extract_session_features(
    session_df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    max_gap_seconds: float = DEFAULT_SAMPLE_GAP_SECONDS,
    feature_set: str = "core"
) -> pd.DataFrame:
    """
    Computes causal rolling window features strictly within a SINGLE session.
    Drops warm-up rows where len(window) < window_size (never zero-fills).
    Resets buffer on gaps > max_gap_seconds or finger == False.
    """
    if session_df.empty:
        return pd.DataFrame()

    df = session_df.copy()
    if "dt" not in df.columns:
        df["dt"] = df["ts"].apply(parse_timestamp)
    df = df.sort_values("dt").reset_index(drop=True)

    buffer = deque(maxlen=window_size)
    last_dt: Optional[pd.Timestamp] = None
    output_rows = []

    has_ambient = "room_temp" in df.columns and "hum" in df.columns

    for i in range(len(df)):
        row = df.iloc[i]
        curr_dt = row["dt"]

        finger_val = row.get("finger", True)
        if isinstance(finger_val, str):
            finger_ok = finger_val.strip().lower() in ("true", "1", "t", "yes")
        else:
            finger_ok = bool(finger_val)

        if last_dt is not None:
            gap = (curr_dt - last_dt).total_seconds()
            if gap > max_gap_seconds or gap < 0:
                buffer.clear()

        if not finger_ok:
            buffer.clear()
            last_dt = curr_dt
            continue

        hr = float(row["hr"])
        spo2 = float(row["spo2"])
        temp = float(row["temp"])

        if has_ambient and not pd.isna(row["room_temp"]) and not pd.isna(row["hum"]):
            room = float(row["room_temp"])
            hum = float(row["hum"])
        else:
            room = 25.0
            hum = 50.0

        temp_sub_room = temp - room
        vec = np.array([hr, spo2, temp, room, hum, temp_sub_room], dtype=np.float64)

        buffer.append(vec)
        last_dt = curr_dt

        if len(buffer) < window_size:
            continue

        buf_arr = np.array(buffer)

        curr = buf_arr[-1]
        prev = buf_arr[-2]
        delta = curr - prev
        mean = np.mean(buf_arr, axis=0)
        var = np.var(buf_arr, axis=0, ddof=1) if window_size > 1 else np.zeros(6)

        feat_dict: Dict[str, Any] = {
            "ts": row["ts"],
            "session_id": row["session_id"] if "session_id" in row else "unknown",
            "scenario": row.get("scenario", "UNLABELED"),
            "hr": curr[0],
            "spo2": curr[1],
            "temp": curr[2],
            "hr_mean": mean[0],
            "spo2_mean": mean[1],
            "temp_mean": mean[2],
            "hr_var": var[0],
            "spo2_var": var[1],
            "temp_var": var[2],
            "hr_delta": delta[0],
            "spo2_delta": delta[1],
            "temp_delta": delta[2],
        }

        if "y_true" in row:
            feat_dict["y_true"] = int(row["y_true"])
        if "is_synthetic" in row:
            feat_dict["is_synthetic"] = bool(row["is_synthetic"])
        if "anomaly_type" in row:
            feat_dict["anomaly_type"] = str(row["anomaly_type"])

        if feature_set.lower() == "extended":
            feat_dict.update({
                "room_temp": curr[3],
                "hum": curr[4],
                "temp_sub_room": curr[5],
                "room_temp_mean": mean[3],
                "hum_mean": mean[4],
                "temp_sub_room_mean": mean[5],
                "room_temp_var": var[3],
                "hum_var": var[4],
                "temp_sub_room_var": var[5],
                "room_temp_delta": delta[3],
                "hum_delta": delta[4],
                "temp_sub_room_delta": delta[5],
            })

        output_rows.append(feat_dict)

    return pd.DataFrame(output_rows)

def extract_features(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    max_gap_seconds: float = DEFAULT_SAMPLE_GAP_SECONDS,
    feature_set: str = "core"
) -> pd.DataFrame:
    """
    Computes causal window features across all sessions in df.
    Processes each session INDEPENDENTLY to guarantee zero cross-session leakage.
    """
    if "session_id" not in df.columns:
        raise ValueError("DataFrame must contain 'session_id' to extract session-isolated features.")

    session_dfs = []
    for sid in df["session_id"].unique():
        sub_df = df[df["session_id"] == sid]
        feat_sub = extract_session_features(
            sub_df,
            window_size=window_size,
            max_gap_seconds=max_gap_seconds,
            feature_set=feature_set
        )
        if not feat_sub.empty:
            session_dfs.append(feat_sub)

    if not session_dfs:
        return pd.DataFrame()

    return pd.concat(session_dfs, ignore_index=True)
