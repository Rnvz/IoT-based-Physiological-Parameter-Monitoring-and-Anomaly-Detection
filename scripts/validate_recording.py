#!/usr/bin/env python3
"""
scripts/validate_recording.py: Read-only Telemetry Recording Session Validator.

Validates recorded session CSVs against quality requirements from docs/RECORDING_PROTOCOL.md:
  1. Segments data into recording sessions (gap > 600s splits distinct sessions).
  2. Per session reports:
     - Total rows and session duration.
     - Timestamp gaps > 3.0 seconds (count and max gap within session).
     - Percentage of rows with finger == True (must be >= 95.0%).
     - Percentage of rows with hr, spo2, or temp == 0.0 when finger == True (must be < 2.0%).
     - Percentage of rows with null/missing hum or room_temp (must be 0.0% unless allowed).
     - Feature ranges (min, max, mean, std) for HR, SpO2, temp, hum, room_temp.
  3. Rejects any session that violates protocol criteria.

Usage:
    python scripts/validate_recording.py -i data_sessions/session_001/
    python scripts/validate_recording.py -i tests/fixtures/sample_fixture.csv --fixture
"""

import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Protocol acceptance criteria thresholds
DEFAULT_MIN_SESSION_ROWS = 100
RECOMMENDED_MIN_ROWS = 300
MIN_FINGER_TRUE_PCT = 95.0
MAX_ZERO_VITALS_PCT = 2.0
MAX_INTRA_SESSION_GAP_SECONDS = 3.0
INTER_SESSION_SPLIT_GAP_SECONDS = 600.0


def parse_timestamp(val: Any) -> Optional[datetime]:
    """Parses timestamp string to timezone-aware UTC datetime."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    s = str(val).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def resolve_csv_path(input_path: str) -> str:
    """
    Resolves input path to a target CSV file.
    If input_path is a directory, searches for 'readings.csv' or the first CSV file found.
    """
    abs_path = os.path.abspath(input_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if os.path.isfile(abs_path):
        if not abs_path.lower().endswith(".csv"):
            raise ValueError(f"Input file is not a CSV: {input_path}")
        return abs_path

    if os.path.isdir(abs_path):
        candidate = os.path.join(abs_path, "readings.csv")
        if os.path.isfile(candidate):
            return candidate

        csv_files = [os.path.join(abs_path, f) for f in sorted(os.listdir(abs_path)) if f.lower().endswith(".csv")]
        if csv_files:
            return csv_files[0]

        raise FileNotFoundError(f"No CSV files found in directory: {input_path}")

    raise ValueError(f"Invalid path type for: {input_path}")


def validate_single_session_df(
    session_id: str,
    df_session: pd.DataFrame,
    min_rows: int = DEFAULT_MIN_SESSION_ROWS,
    allow_null_dht: bool = False
) -> Dict[str, Any]:
    """
    Validates a single segmented session DataFrame against protocol rules.
    """
    total_rows = len(df_session)
    if total_rows == 0:
        return {
            "session_id": session_id,
            "total_rows": 0,
            "valid_rows": 0,
            "duration_seconds": 0.0,
            "gaps_gt_3s_count": 0,
            "max_gap_seconds": 0.0,
            "finger_true_pct": 0.0,
            "zero_vitals_pct": 0.0,
            "null_hum_pct": 100.0,
            "null_room_temp_pct": 100.0,
            "feature_ranges": {},
            "status": "REJECT",
            "reasons": ["Sesi tidak memiliki baris data (0 rows)."]
        }

    # Timestamp & Intra-session gaps
    dts = df_session["_dt"].dropna().sort_values().reset_index(drop=True)
    duration_s = 0.0
    gaps_gt_3s = 0
    max_gap_s = 0.0

    if len(dts) > 1:
        duration_s = max(0.0, (dts.iloc[-1] - dts.iloc[0]).total_seconds())
        for i in range(1, len(dts)):
            gap = (dts.iloc[i] - dts.iloc[i - 1]).total_seconds()
            if gap > max_gap_s:
                max_gap_s = gap
            if gap > MAX_INTRA_SESSION_GAP_SECONDS:
                gaps_gt_3s += 1
    elif len(dts) == 1:
        duration_s = 1.0

    # % finger == True
    finger_count = int(df_session["_finger"].sum())
    finger_true_pct = round(float(finger_count / total_rows) * 100, 2)

    # % zero vitals when finger == True
    finger_mask = (df_session["_finger"] == True)
    if finger_count > 0:
        zero_vitals_mask = (
            (df_session["_hr"].isna() | (df_session["_hr"] <= 0.0)) |
            (df_session["_spo2"].isna() | (df_session["_spo2"] <= 0.0)) |
            (df_session["_temp"].isna() | (df_session["_temp"] <= 0.0))
        ) & finger_mask
        zero_vitals_count = int(zero_vitals_mask.sum())
        zero_vitals_pct = round(float(zero_vitals_count / finger_count) * 100, 2)
    else:
        zero_vitals_pct = 100.0

    # % null hum and room_temp
    def calc_null_pct(s: pd.Series) -> float:
        if len(s) == 0:
            return 100.0
        missing = s.isna() | (s <= 0.0)
        return round(float(missing.sum() / len(s)) * 100, 2)

    null_hum_pct = calc_null_pct(df_session["_hum"])
    null_room_temp_pct = calc_null_pct(df_session["_room_temp"])

    # Valid rows
    valid_mask = (
        finger_mask &
        (df_session["_hr"] > 0.0) &
        (df_session["_spo2"] > 0.0) &
        (df_session["_temp"] > 0.0)
    )
    valid_rows = int(valid_mask.sum())

    # Feature ranges (min, max, mean, std)
    metrics_to_calc = [
        ("HR", df_session.loc[valid_mask, "_hr"], "bpm"),
        ("SpO2", df_session.loc[valid_mask, "_spo2"], "%"),
        ("temp", df_session.loc[valid_mask, "_temp"], "°C"),
        ("hum", df_session.loc[df_session["_hum"].notna() & (df_session["_hum"] > 0), "_hum"], "%"),
        ("room_temp", df_session.loc[df_session["_room_temp"].notna() & (df_session["_room_temp"] > 0), "_room_temp"], "°C"),
    ]

    feature_ranges: Dict[str, Dict[str, Any]] = {}
    for name, s, unit in metrics_to_calc:
        s_clean = s.dropna()
        if len(s_clean) > 0:
            feature_ranges[name] = {
                "count": int(len(s_clean)),
                "min": round(float(s_clean.min()), 2),
                "max": round(float(s_clean.max()), 2),
                "mean": round(float(s_clean.mean()), 2),
                "std": round(float(s_clean.std(ddof=1)), 2) if len(s_clean) > 1 else 0.0,
                "unit": unit
            }
        else:
            feature_ranges[name] = {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "unit": unit
            }

    # Evaluate acceptance criteria
    reasons: List[str] = []
    if valid_rows < min_rows:
        reasons.append(f"Baris valid ({valid_rows}) kurang dari batas minimum ({min_rows} baris).")

    if gaps_gt_3s > 0:
        reasons.append(f"Terdeteksi {gaps_gt_3s} kejadian jeda waktu > 3.0 s (maksimum jeda: {max_gap_s:.2f} s).")

    if finger_true_pct < MIN_FINGER_TRUE_PCT:
        reasons.append(f"Persentase finger=True ({finger_true_pct:.1f}%) di bawah batas minimum {MIN_FINGER_TRUE_PCT}%.")

    if zero_vitals_pct >= MAX_ZERO_VITALS_PCT:
        reasons.append(f"Persentase nilai 0.0 pada sensor vital ({zero_vitals_pct:.1f}%) melebihi batas {MAX_ZERO_VITALS_PCT}%.")

    if not allow_null_dht:
        if null_hum_pct > 0.0:
            reasons.append(f"Kelembapan (hum) memiliki {null_hum_pct:.1f}% data null/hilang.")
        if null_room_temp_pct > 0.0:
            reasons.append(f"Suhu ruangan (room_temp) memiliki {null_room_temp_pct:.1f}% data null/hilang.")

    status = "ACCEPT" if len(reasons) == 0 else "REJECT"

    return {
        "session_id": session_id,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "duration_seconds": round(duration_s, 2),
        "gaps_gt_3s_count": gaps_gt_3s,
        "max_gap_seconds": round(max_gap_s, 2),
        "finger_true_pct": finger_true_pct,
        "zero_vitals_pct": zero_vitals_pct,
        "null_hum_pct": null_hum_pct,
        "null_room_temp_pct": null_room_temp_pct,
        "feature_ranges": feature_ranges,
        "status": status,
        "reasons": reasons
    }


def validate_recording(
    input_path: str,
    min_rows: int = DEFAULT_MIN_SESSION_ROWS,
    allow_null_dht: bool = False,
    is_fixture: bool = False
) -> Dict[str, Any]:
    """
    Main validator function: loads CSV, segments into sessions, and validates each session.
    """
    csv_file = resolve_csv_path(input_path)
    df = pd.read_csv(csv_file)

    if len(df) == 0:
        empty_res = validate_single_session_df("session_001", pd.DataFrame(), min_rows=min_rows)
        return {
            "csv_file": csv_file,
            "is_fixture": is_fixture,
            "total_rows": 0,
            "total_sessions": 0,
            "status": "REJECT",
            "sessions": {"session_001": empty_res},
            "reasons": ["File CSV kosong (0 baris)."]
        }

    # Normalize column names
    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)

    def parse_finger(v: Any) -> bool:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if pd.isna(v):
            return False
        return str(v).strip().lower() in ("true", "1", "t", "yes")

    df["_finger"] = df["finger"].apply(parse_finger) if "finger" in df.columns else False
    df["_hr"] = pd.to_numeric(df.get("hr", pd.Series(dtype=float)), errors="coerce")
    df["_spo2"] = pd.to_numeric(df.get("spo2", pd.Series(dtype=float)), errors="coerce")
    df["_temp"] = pd.to_numeric(df.get("temp", pd.Series(dtype=float)), errors="coerce")
    df["_hum"] = pd.to_numeric(df.get("hum", pd.Series(dtype=float)), errors="coerce")
    df["_room_temp"] = pd.to_numeric(df.get("room_temp", pd.Series(dtype=float)), errors="coerce")

    if "ts" in df.columns:
        df["_dt"] = df["ts"].apply(parse_timestamp)
        df = df.sort_values("_dt").reset_index(drop=True)
    else:
        df["_dt"] = None

    # Segment into sessions
    # If session_id column already exists, use it; otherwise segment on gap > 600s
    if "session_id" in df.columns:
        df["_session_id"] = df["session_id"].astype(str)
    else:
        session_ids = []
        cur_idx = 1
        session_ids.append(f"session_{cur_idx:03d}")
        for i in range(1, len(df)):
            dt_prev = df.loc[i - 1, "_dt"]
            dt_cur = df.loc[i, "_dt"]
            if dt_prev is not None and dt_cur is not None:
                gap = (dt_cur - dt_prev).total_seconds()
                if gap > INTER_SESSION_SPLIT_GAP_SECONDS or gap < 0:
                    cur_idx += 1
            session_ids.append(f"session_{cur_idx:03d}")
        df["_session_id"] = session_ids

    # Validate each session
    session_results: Dict[str, Dict[str, Any]] = {}
    overall_reasons: List[str] = []

    for sid, group in df.groupby("_session_id", sort=True):
        res_s = validate_single_session_df(
            session_id=str(sid),
            df_session=group.reset_index(drop=True),
            min_rows=min_rows,
            allow_null_dht=allow_null_dht
        )
        session_results[str(sid)] = res_s
        if res_s["status"] == "REJECT":
            for r in res_s["reasons"]:
                overall_reasons.append(f"[{sid}] {r}")

    overall_status = "ACCEPT" if len(overall_reasons) == 0 else "REJECT"

    return {
        "csv_file": csv_file,
        "is_fixture": is_fixture,
        "total_rows": len(df),
        "total_sessions": len(session_results),
        "status": overall_status,
        "sessions": session_results,
        "reasons": overall_reasons
    }


def print_report(res: Dict[str, Any]):
    """Prints a clear structured report to stdout."""
    print("=" * 80)
    if res.get("is_fixture", False) or "fixture" in res["csv_file"].lower():
        print("LAPORAN VALIDASI SESI TELEMETRI [LABEL: FIXTURE, BUKAN HASIL]")
    else:
        print("LAPORAN VALIDASI SESI TELEMETRI (PROTOKOL PEREKAMAN V2)")
    print("=" * 80)
    print(f"File CSV Target    : {res['csv_file']}")
    print(f"Total Baris Data   : {res['total_rows']:,}")
    print(f"Total Sesi Ditemukan: {res['total_sessions']}")
    print("-" * 80)

    for sid, s in res["sessions"].items():
        dur_m = s["duration_seconds"] / 60.0
        status_tag = "[ACCEPT]" if s["status"] == "ACCEPT" else "[REJECT]"
        print(f"\n>> {status_tag} {sid.upper()} | {s['valid_rows']}/{s['total_rows']} baris valid | Durasi: {s['duration_seconds']:.1f}s ({dur_m:.1f} m)")
        print(f"   - Jeda > 3.0s              : {s['gaps_gt_3s_count']} kejadian (Maks gap: {s['max_gap_seconds']:.2f}s)")
        print(f"   - % finger == True         : {s['finger_true_pct']:.2f}%")
        print(f"   - % Vital == 0.0           : {s['zero_vitals_pct']:.2f}%")
        print(f"   - % Null hum / room_temp   : {s['null_hum_pct']:.1f}% / {s['null_room_temp_pct']:.1f}%")

        # Ranges
        print("   - Rentang Fitur:")
        for fname, fstats in s["feature_ranges"].items():
            if fstats["count"] > 0:
                print(f"     * {fname:<10}: min={fstats['min']:>6.2f}, mean={fstats['mean']:>6.2f}, max={fstats['max']:>6.2f}, std={fstats['std']:>5.2f} {fstats['unit']}")
            else:
                print(f"     * {fname:<10}: N/A (0 data)")

        if s["status"] == "REJECT":
            print("   - Alasan Penolakan:")
            for r in s["reasons"]:
                print(f"     ! {r}")

    print("\n" + "=" * 80)
    print("KESIMPULAN VALIDASI PROTOKOL PEREKAMAN:")
    if res["status"] == "ACCEPT":
        print(">>> HASIL AKHIR: [ACCEPT / DITERIMA] Seluruh sesi memenuhi kriteria protokol.")
    else:
        print(">>> HASIL AKHIR: [REJECT / DITOLAK] Ditemukan sesi yang gagal memenuhi syarat.")
        for r in res["reasons"]:
            print(f"    * {r}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Validasi sesi rekaman telemetri per-sesi sesuai docs/RECORDING_PROTOCOL.md.")
    parser.add_argument("--input", "-i", required=True, help="Path ke file CSV atau direktori sesi")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_SESSION_ROWS, help=f"Minimum baris valid per sesi (default: {DEFAULT_MIN_SESSION_ROWS})")
    parser.add_argument("--allow-null-dht", action="store_true", help="Izinkan null pada sensor DHT22 (untuk skenario DHT_UNPLUGGED)")
    parser.add_argument("--fixture", action="store_true", help="Tandai eksplisit sebagai data fixture pengujian")
    args = parser.parse_args()

    try:
        res = validate_recording(
            input_path=args.input,
            min_rows=args.min_rows,
            allow_null_dht=args.allow_null_dht,
            is_fixture=args.fixture
        )
        print_report(res)
        if res["status"] != "ACCEPT":
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Gagal memvalidasi sesi: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
