"""
ml_eval.sessions: Dataset loading, cleaning, and session segmentation.

A session is defined as a contiguous run of rows from the physical device
where the timestamp gap between consecutive readings is strictly below a
configurable threshold (default: 10 minutes = 600 seconds).

Only rows with finger=True and valid physiological vitals (HR, SpO2, Temp)
are retained for model training and evaluation.
"""

import sys
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np

HR_MIN, HR_MAX = 30.0, 220.0
SPO2_MIN, SPO2_MAX = 70.0, 100.0
TEMP_MIN, TEMP_MAX = 25.0, 45.0

DEFAULT_MAX_GAP_SECONDS = 600.0

def parse_timestamp(val: Any) -> datetime:
    """Parses ISO timestamp string to timezone-aware UTC datetime."""
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    s = str(val).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise ValueError(f"Could not parse timestamp '{val}': {e}")

def load_dataset(source: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Loads dataset from file path or DataFrame.
    Validates required columns: ts, hr, spo2, temp, finger.
    Sorts chronologically by ts.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    required_cols = {"ts", "hr", "spo2", "temp", "finger"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if "scenario" not in df.columns:
        df["scenario"] = "UNLABELED"
    else:
        df["scenario"] = df["scenario"].fillna("UNLABELED").astype(str)

    df["dt"] = df["ts"].apply(parse_timestamp)
    df = df.sort_values("dt").reset_index(drop=True)
    return df

def clean_for_modelling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only rows with finger=true and valid physiological values.
    Converts hr, spo2, temp to numeric.
    """
    df = df.copy()

    def parse_finger(v: Any) -> bool:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if pd.isna(v):
            return False
        return str(v).strip().lower() in ("true", "1", "t", "yes")

    df["finger_bool"] = df["finger"].apply(parse_finger)
    df["hr_num"] = pd.to_numeric(df["hr"], errors="coerce")
    df["spo2_num"] = pd.to_numeric(df["spo2"], errors="coerce")
    df["temp_num"] = pd.to_numeric(df["temp"], errors="coerce")

    if "room_temp" in df.columns:
        df["room_temp_num"] = pd.to_numeric(df["room_temp"], errors="coerce")
    else:
        df["room_temp_num"] = np.nan

    if "hum" in df.columns:
        df["hum_num"] = pd.to_numeric(df["hum"], errors="coerce")
    else:
        df["hum_num"] = np.nan

    valid_mask = (
        (df["finger_bool"] == True) &
        (df["hr_num"] >= HR_MIN) & (df["hr_num"] <= HR_MAX) &
        (df["spo2_num"] >= SPO2_MIN) & (df["spo2_num"] <= SPO2_MAX) &
        (df["temp_num"] >= TEMP_MIN) & (df["temp_num"] <= TEMP_MAX)
    )

    clean_df = df[valid_mask].copy()

    clean_df["hr"] = clean_df["hr_num"]
    clean_df["spo2"] = clean_df["spo2_num"]
    clean_df["temp"] = clean_df["temp_num"]
    clean_df["room_temp"] = clean_df["room_temp_num"]
    clean_df["hum"] = clean_df["hum_num"]
    clean_df["finger"] = True

    drop_cols = ["finger_bool", "hr_num", "spo2_num", "temp_num", "room_temp_num", "hum_num"]
    clean_df = clean_df.drop(columns=[c for c in drop_cols if c in clean_df.columns])
    clean_df = clean_df.reset_index(drop=True)
    return clean_df

def segment_sessions(df: pd.DataFrame, max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS) -> pd.DataFrame:
    """
    Assigns session_id to consecutive rows where time gap <= max_gap_seconds.
    Rows must already be sorted chronologically and cleaned.
    """
    if df.empty:
        df_out = df.copy()
        df_out["session_id"] = pd.Series(dtype=str)
        return df_out

    df = df.copy()
    if "dt" not in df.columns:
        df["dt"] = df["ts"].apply(parse_timestamp)
        df = df.sort_values("dt").reset_index(drop=True)

    session_ids = []
    current_session_idx = 1
    session_ids.append(f"session_{current_session_idx:03d}")

    for i in range(1, len(df)):
        gap = (df["dt"].iloc[i] - df["dt"].iloc[i - 1]).total_seconds()
        if gap > max_gap_seconds or gap < 0:
            current_session_idx += 1
        session_ids.append(f"session_{current_session_idx:03d}")

    df["session_id"] = session_ids
    return df

def summarize_sessions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics for segmented sessions:
    total sessions, duration per session, and rows per scenario.
    """
    if df.empty or "session_id" not in df.columns:
        return {
            "total_sessions": 0,
            "total_valid_rows": 0,
            "sessions": {}
        }

    summary = {
        "total_sessions": int(df["session_id"].nunique()),
        "total_valid_rows": int(len(df)),
        "sessions": {}
    }

    for sid, group in df.groupby("session_id", sort=True):
        dts = group["dt"].tolist() if "dt" in group.columns else [parse_timestamp(x) for x in group["ts"]]
        start_ts = dts[0].isoformat()
        end_ts = dts[-1].isoformat()
        duration_s = max(0.0, (dts[-1] - dts[0]).total_seconds())

        scenario_counts = group["scenario"].value_counts().to_dict()

        summary["sessions"][str(sid)] = {
            "rows": int(len(group)),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_seconds": round(duration_s, 2),
            "scenarios": scenario_counts
        }

    return summary

def load_and_segment_sessions(
    source: Union[str, pd.DataFrame],
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS
) -> pd.DataFrame:
    """
    Convenience end-to-end function: loads, cleans, and segments sessions.
    """
    raw_df = load_dataset(source)
    cleaned_df = clean_for_modelling(raw_df)
    segmented_df = segment_sessions(cleaned_df, max_gap_seconds=max_gap_seconds)
    return segmented_df

def main():
    parser = argparse.ArgumentParser(description="Session segmentation and validation for physiological datasets.")
    parser.add_argument("--input", "-i", required=True, help="Path to exported dataset CSV (e.g. data/dataset.csv)")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP_SECONDS, help="Max gap seconds between samples before splitting session (default: 600)")
    parser.add_argument("--output", "-o", default=None, help="Optional output path to save segmented CSV")
    args = parser.parse_args()

    print(f"Loading and segmenting dataset from: {args.input}")
    df = load_and_segment_sessions(args.input, max_gap_seconds=args.max_gap)
    summary = summarize_sessions(df)

    print("=" * 80)
    print("SESSION SEGMENTATION SUMMARY")
    print("=" * 80)
    print(f"Total sessions identified : {summary['total_sessions']}")
    print(f"Total valid modelled rows : {summary['total_valid_rows']}")
    print("-" * 80)
    for sid, info in summary["sessions"].items():
        dur_min = info["duration_seconds"] / 60.0
        scen_str = ", ".join(f"{k}: {v}" for k, v in info["scenarios"].items())
        print(f"[{sid}] {info['rows']:>5} rows | Duration: {dur_min:>5.1f} min | Scenarios: {scen_str}")
    print("=" * 80)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved segmented dataset with 'session_id' to: {args.output}")

if __name__ == "__main__":
    main()
