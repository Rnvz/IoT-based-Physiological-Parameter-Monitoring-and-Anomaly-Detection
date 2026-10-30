#!/usr/bin/env python3
"""
Dataset Quality and Integrity Checker for Physiological Telemetry CSVs.
Validates recorded datasets against thesis evaluation protocol requirements.

Reports:
  - Total rows and valid rows (finger=True and non-null HR/SpO2/Temp)
  - First and last timestamps, total duration
  - Median and maximum timestamp gap
  - High-frequency burst detection (>3 rows in any 1-second window)
  - Duplicate timestamp count
  - Valid rows grouped by benign scenario (if column exists)
  - Longest contiguous valid stretch (gap <= 5.0s without sensor off)
  - Progress against 1,000 valid-row thesis target
  - Clear PASS / FAIL line

Usage:
    python scripts/dataset_check.py -i data/readings.csv
    python scripts/dataset_check.py -i data/dataset_export.csv
"""

import os
import sys
import csv
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import statistics

TARGET_VALID_ROWS = 1000
MAX_CONTIGUOUS_GAP_SECONDS = 5.0

def parse_iso_ts(ts_str: str) -> Optional[datetime]:
    """Parse ISO 8601 string to datetime object."""
    if not ts_str:
        return None
    try:
        # Handle trailing Z
        clean_ts = ts_str.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def is_valid_number(val: Any) -> bool:
    """Check if value is a valid non-null numeric reading."""
    if val is None:
        return False
    s = str(val).strip().lower()
    if s in ("", "null", "none", "nan"):
        return False
    try:
        f = float(s)
        return f > 0.0
    except ValueError:
        return False

def is_finger_present(val: Any) -> bool:
    """Check if finger flag is true."""
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("true", "1", "t", "yes")

def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """Analyzes CSV dataset and returns quality and integrity metrics."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    rows_data: List[Dict[str, Any]] = []
    has_scenario_col = False

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            has_scenario_col = "scenario" in reader.fieldnames

        for idx, row in enumerate(reader):
            ts_str = row.get("ts", "")
            dt = parse_iso_ts(ts_str)

            hr_raw = row.get("hr")
            spo2_raw = row.get("spo2")
            temp_raw = row.get("temp")
            finger_raw = row.get("finger")
            scenario_raw = row.get("scenario") if has_scenario_col else None

            # Valid reading criteria: finger is true AND hr, spo2, temp are valid non-null floats
            valid = (
                is_finger_present(finger_raw) and
                is_valid_number(hr_raw) and
                is_valid_number(spo2_raw) and
                is_valid_number(temp_raw)
            )

            rows_data.append({
                "line_no": idx + 2,
                "ts_str": ts_str,
                "dt": dt,
                "is_valid": valid,
                "scenario": scenario_raw
            })

    total_rows = len(rows_data)
    valid_rows = sum(1 for r in rows_data if r["is_valid"])
    invalid_rows = total_rows - valid_rows

    if total_rows == 0:
        return {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "first_ts": None,
            "last_ts": None,
            "duration_s": 0.0,
            "median_gap_s": None,
            "max_gap_s": None,
            "burst_detected": False,
            "max_rows_in_1s": 0,
            "burst_windows_count": 0,
            "duplicate_ts_count": 0,
            "has_scenario_col": has_scenario_col,
            "scenario_counts": {},
            "longest_valid_stretch": 0,
            "target_met": False,
            "pass_check": False,
            "failure_reasons": ["Dataset file is empty (0 rows)."]
        }

    # Timestamp sorting and gap calculation
    dated_rows = [r for r in rows_data if r["dt"] is not None]
    dated_rows.sort(key=lambda r: r["dt"])

    first_ts = dated_rows[0]["ts_str"] if dated_rows else None
    last_ts = dated_rows[-1]["ts_str"] if dated_rows else None
    duration_s = (dated_rows[-1]["dt"] - dated_rows[0]["dt"]).total_seconds() if len(dated_rows) > 1 else 0.0

    # Gaps between consecutive timestamps
    gaps: List[float] = []
    for i in range(1, len(dated_rows)):
        gap = (dated_rows[i]["dt"] - dated_rows[i-1]["dt"]).total_seconds()
        gaps.append(gap)

    median_gap_s = statistics.median(gaps) if gaps else None
    max_gap_s = max(gaps) if gaps else None

    # Duplicate timestamps count
    all_ts = [r["ts_str"] for r in rows_data if r["ts_str"]]
    duplicate_ts_count = len(all_ts) - len(set(all_ts))

    # High frequency burst check: > 3 rows in any 1-second window
    burst_windows_count = 0
    max_rows_in_1s = 0
    n_d = len(dated_rows)

    right = 0
    for left in range(n_d):
        t_left = dated_rows[left]["dt"]
        while right < n_d and (dated_rows[right]["dt"] - t_left).total_seconds() <= 1.0:
            right += 1
        window_count = right - left
        if window_count > max_rows_in_1s:
            max_rows_in_1s = window_count
        if window_count > 3:
            burst_windows_count += 1

    burst_detected = (max_rows_in_1s > 3)

    # Valid rows per scenario
    scenario_counts: Dict[str, int] = {}
    if has_scenario_col:
        for r in rows_data:
            if r["is_valid"]:
                sc = r["scenario"] if r["scenario"] else "UNLABELED"
                scenario_counts[sc] = scenario_counts.get(sc, 0) + 1

    # Longest contiguous valid stretch (gap <= 5.0s between consecutive valid samples)
    longest_valid_stretch = 0
    current_stretch = 0
    prev_valid_dt: Optional[datetime] = None

    for r in dated_rows:
        if r["is_valid"]:
            if current_stretch == 0 or prev_valid_dt is None:
                current_stretch = 1
            else:
                gap = (r["dt"] - prev_valid_dt).total_seconds()
                if gap <= MAX_CONTIGUOUS_GAP_SECONDS:
                    current_stretch += 1
                else:
                    current_stretch = 1
            if current_stretch > longest_valid_stretch:
                longest_valid_stretch = current_stretch
            prev_valid_dt = r["dt"]
        else:
            current_stretch = 0
            prev_valid_dt = None

    target_met = (valid_rows >= TARGET_VALID_ROWS)
    failure_reasons = []
    if not target_met:
        failure_reasons.append(f"Target 1,000 valid rows not met ({valid_rows}/{TARGET_VALID_ROWS}).")

    pass_check = (len(failure_reasons) == 0)

    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "duration_s": duration_s,
        "median_gap_s": median_gap_s,
        "max_gap_s": max_gap_s,
        "burst_detected": burst_detected,
        "max_rows_in_1s": max_rows_in_1s,
        "burst_windows_count": burst_windows_count,
        "duplicate_ts_count": duplicate_ts_count,
        "has_scenario_col": has_scenario_col,
        "scenario_counts": scenario_counts,
        "longest_valid_stretch": longest_valid_stretch,
        "target_met": target_met,
        "pass_check": pass_check,
        "failure_reasons": failure_reasons
    }

def format_duration(seconds: float) -> str:
    """Format duration into human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{seconds:.1f}s ({h}h {m}m {s}s)"
    elif m > 0:
        return f"{seconds:.1f}s ({m}m {s}s)"
    else:
        return f"{seconds:.1f}s"

def print_report(res: Dict[str, Any], file_path: str):
    """Prints the formatted dataset report."""
    sep = "=" * 80
    print(sep)
    print(f"DATASET QUALITY & INTEGRITY REPORT: {os.path.basename(file_path)}")
    print(f"Source file: {os.path.abspath(file_path)}")
    print(sep)

    print(f"Total rows                   : {res['total_rows']:,}")
    pct_valid = (res['valid_rows'] / res['total_rows'] * 100) if res['total_rows'] > 0 else 0.0
    pct_invalid = (res['invalid_rows'] / res['total_rows'] * 100) if res['total_rows'] > 0 else 0.0
    print(f"Valid rows (finger & vitals) : {res['valid_rows']:,} ({pct_valid:.1f}%)")
    print(f"Invalid rows                 : {res['invalid_rows']:,} ({pct_invalid:.1f}%)")
    print(f"First timestamp              : {res['first_ts'] or 'N/A'}")
    print(f"Last timestamp               : {res['last_ts'] or 'N/A'}")
    print(f"Total duration               : {format_duration(res['duration_s'])}")

    if res['median_gap_s'] is not None:
        print(f"Median timestamp gap         : {res['median_gap_s']:.2f}s")
        print(f"Maximum timestamp gap        : {res['max_gap_s']:.2f}s")
    else:
        print("Median timestamp gap         : N/A")
        print("Maximum timestamp gap        : N/A")

    if res['burst_detected']:
        print(f"Burst data (>3 rows/1s)      : DETECTED (max {res['max_rows_in_1s']} rows/sec, {res['burst_windows_count']} burst windows)")
        print("  --> WARNING: High row rate indicates simulator fast mode or unthrottled bursts.")
    else:
        print(f"Burst data (>3 rows/1s)      : None detected (max {res['max_rows_in_1s']} rows/sec)")

    print(f"Duplicate timestamps         : {res['duplicate_ts_count']}")
    print(f"Longest contiguous stretch   : {res['longest_valid_stretch']:,} valid rows (gap <= {MAX_CONTIGUOUS_GAP_SECONDS:.1f}s)")

    if res['has_scenario_col']:
        print("Valid rows by scenario       :")
        if res['scenario_counts']:
            for sc, count in sorted(res['scenario_counts'].items()):
                print(f"  - {sc:<20s} : {count:,} rows")
        else:
            print("  (No valid rows with scenario labels)")
    else:
        print("Valid rows by scenario       : N/A (no scenario column in CSV)")

    target_str = "MET" if res['target_met'] else "NOT MET"
    print(f"Target 1,000 valid rows      : {target_str} ({res['valid_rows']:,} / {TARGET_VALID_ROWS:,})")

    print("-" * 80)
    if res['pass_check']:
        print("RESULT                       : PASS")
    else:
        print(f"RESULT                       : FAIL ({'; '.join(res['failure_reasons'])})")
    print(sep)

def main():
    parser = argparse.ArgumentParser(description="Check dataset quality and integrity for thesis evaluation.")
    parser.add_argument("-i", "--input", required=True, help="Path to readings.csv or export dataset CSV")
    args = parser.parse_args()

    try:
        report = analyze_dataset(args.input)
        print_report(report, args.input)
        sys.exit(0 if report["pass_check"] else 1)
    except Exception as e:
        print(f"[ERROR] Failed to analyze dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
