"""
scripts/check_recording.py: Dataset Quality & Session Integrity Checker (Read-Only).

Evaluates recorded physiological telemetry CSVs (per session folder or direct CSV file):
  1. Valid rows (finger=True, hr > 0, spo2 > 0, temp > 0).
  2. Percentage of null/missing values for humidity (hum) and ambient temperature (room_temp).
  3. Range (min, max) and standard deviation (std) for HR, SpO2, temp, hum, room_temp.
  4. Number of rows with stuck/frozen sensor values (> 30 seconds consecutive identical vitals).
  5. Simulator data indicators (device_id patterns, synthetic constant values, zero variance).
  6. ml_eval session qualification (minimum valid rows >= 15 for windowing, >= 300 for representative session).

Usage:
    python scripts/check_recording.py -i data_sessions/session_001/
    python scripts/check_recording.py -i tests/fixtures/sample_fixture.csv
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

ML_EVAL_MIN_WINDOW_ROWS = 15
ML_EVAL_RECOMMENDED_SESSION_ROWS = 300
STUCK_SENSOR_THRESHOLD_SECONDS = 30.0

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

def analyze_recording(input_path: str) -> Dict[str, Any]:
    """
    Performs read-only quality and integrity audit on a recording CSV or folder.
    Returns structured results dictionary.
    """
    csv_file = resolve_csv_path(input_path)
    df = pd.read_csv(csv_file)

    total_rows = len(df)
    if total_rows == 0:
        return {
            "csv_file": csv_file,
            "total_rows": 0,
            "valid_rows": 0,
            "valid_ratio": 0.0,
            "ml_eval_qualified": False,
            "ml_eval_reasons": ["Dataset file is empty (0 rows)."],
            "null_percentages": {"hum": 100.0, "room_temp": 100.0},
            "sensor_stats": {},
            "stuck_rows_count": 0,
            "stuck_streaks_count": 0,
            "simulator_indicators": {
                "is_simulator": False,
                "reasons": ["Dataset is empty."]
            }
        }

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
    else:
        df["_dt"] = None

    valid_mask = (
        (df["_finger"] == True) &
        (df["_hr"] > 0.0) &
        (df["_spo2"] > 0.0) &
        (df["_temp"] > 0.0)
    )
    valid_rows = int(valid_mask.sum())
    valid_ratio = round(float(valid_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

    def calc_null_pct(series: pd.Series) -> float:
        if len(series) == 0:
            return 100.0
        missing = series.isna() | (series <= 0.0)
        return round(float(missing.sum() / len(series)) * 100, 2)

    hum_null_pct = calc_null_pct(df["_hum"])
    room_temp_null_pct = calc_null_pct(df["_room_temp"])

    metrics_to_compute = [
        ("HR", df.loc[valid_mask, "_hr"], "bpm"),
        ("SpO2", df.loc[valid_mask, "_spo2"], "%"),
        ("temp", df.loc[valid_mask, "_temp"], "°C"),
        ("hum", df.loc[df["_hum"].notna() & (df["_hum"] > 0), "_hum"], "%"),
        ("room_temp", df.loc[df["_room_temp"].notna() & (df["_room_temp"] > 0), "_room_temp"], "°C"),
    ]

    sensor_stats: Dict[str, Dict[str, Any]] = {}
    for name, s, unit in metrics_to_compute:
        s_clean = s.dropna()
        if len(s_clean) > 0:
            min_v = float(s_clean.min())
            max_v = float(s_clean.max())
            std_v = float(s_clean.std(ddof=1)) if len(s_clean) > 1 else 0.0
            mean_v = float(s_clean.mean())
            sensor_stats[name] = {
                "count": int(len(s_clean)),
                "min": round(min_v, 2),
                "max": round(max_v, 2),
                "mean": round(mean_v, 2),
                "std": round(std_v, 2),
                "unit": unit
            }
        else:
            sensor_stats[name] = {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "unit": unit
            }

    stuck_rows_count = 0
    stuck_streaks_count = 0
    streak_rows = 0
    streak_flagged = False
    streak_start_dt: Optional[datetime] = None
    prev_tuple: Optional[Tuple[float, float, float]] = None

    valid_df = df[valid_mask].copy().reset_index(drop=True)

    for i in range(len(valid_df)):
        cur_tuple = (
            round(float(valid_df.loc[i, "_hr"]), 2),
            round(float(valid_df.loc[i, "_spo2"]), 2),
            round(float(valid_df.loc[i, "_temp"]), 2)
        )
        cur_dt = valid_df.loc[i, "_dt"]

        if prev_tuple is not None and cur_tuple == prev_tuple:
            streak_rows += 1
            elapsed_s = (cur_dt - streak_start_dt).total_seconds() if (cur_dt and streak_start_dt) else float(streak_rows)
            if elapsed_s >= STUCK_SENSOR_THRESHOLD_SECONDS:
                if not streak_flagged:
                    stuck_streaks_count += 1
                    streak_flagged = True
                stuck_rows_count += 1
        else:
            streak_rows = 1
            streak_flagged = False
            streak_start_dt = cur_dt
            prev_tuple = cur_tuple

    sim_reasons: List[str] = []
    is_simulator = False

    if "device_id" in df.columns:
        dev_ids = df["device_id"].dropna().astype(str).unique().tolist()
        sim_keywords = ["sim", "mock", "fake", "test", "dummy"]
        for d in dev_ids:
            if any(k in d.lower() for k in sim_keywords):
                sim_reasons.append(f"Kolom device_id teridentifikasi simulator: '{d}'")
                is_simulator = True

    if sensor_stats.get("temp", {}).get("count", 0) > 30:
        t_std = sensor_stats["temp"]["std"]
        if t_std is not None and t_std == 0.0:
            sim_reasons.append("Suhu tubuh (temp) memiliki std = 0.00 (nilai konstan mutlak, khas simulasi)")
            is_simulator = True

    if sensor_stats.get("HR", {}).get("count", 0) > 30:
        hr_std = sensor_stats["HR"]["std"]
        if hr_std is not None and hr_std == 0.0:
            sim_reasons.append("Detak jantung (HR) memiliki std = 0.00 (nilai konstan mutlak)")
            is_simulator = True

    if sensor_stats.get("room_temp", {}).get("count", 0) > 50:
        rt_std = sensor_stats["room_temp"]["std"]
        rt_mean = sensor_stats["room_temp"]["mean"]
        if rt_std == 0.0 and rt_mean in (24.0, 25.0):
            sim_reasons.append(f"room_temp konstan persis {rt_mean} °C tanpa fluktuasi mikroklimat")
            is_simulator = True

    if not is_simulator:
        sim_summary = "Hardware Asli (Tidak terdeteksi pola artifisial konstan/device_id simulator)"
    else:
        sim_summary = f"Terindikasi Simulator ({'; '.join(sim_reasons)})"

    ml_eval_reasons: List[str] = []
    if valid_rows < ML_EVAL_MIN_WINDOW_ROWS:
        ml_eval_reasons.append(
            f"Baris valid ({valid_rows}) kurang dari window_size minimal ({ML_EVAL_MIN_WINDOW_ROWS})."
        )
    elif valid_rows < ML_EVAL_RECOMMENDED_SESSION_ROWS:
        ml_eval_reasons.append(
            f"Baris valid ({valid_rows}) di bawah panjang sesi rekomendasi ({ML_EVAL_RECOMMENDED_SESSION_ROWS} baris/~5 menit)."
        )

    if stuck_rows_count > 60:
        ml_eval_reasons.append(
            f"Sensor macet terdeteksi pada {stuck_rows_count} baris (>30s nilai identik berturut-turut)."
        )

    ml_eval_qualified = (len(ml_eval_reasons) == 0)

    return {
        "csv_file": csv_file,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "valid_ratio": valid_ratio,
        "null_percentages": {
            "hum": hum_null_pct,
            "room_temp": room_temp_null_pct
        },
        "sensor_stats": sensor_stats,
        "stuck_rows_count": stuck_rows_count,
        "stuck_streaks_count": stuck_streaks_count,
        "simulator_indicators": {
            "is_simulator": is_simulator,
            "summary": sim_summary,
            "reasons": sim_reasons
        },
        "ml_eval_qualified": ml_eval_qualified,
        "ml_eval_reasons": ml_eval_reasons
    }

def print_report(res: Dict[str, Any]):
    """Prints a structured text report to stdout."""
    print("=" * 80)
    print("LAPORAN AUDIT PEREKAMAN SESI TELEMETRI (READ-ONLY)")
    print("=" * 80)
    print(f"File CSV Target          : {res['csv_file']}")
    print(f"Total Baris Terbaca      : {res['total_rows']:,}")
    print(f"Baris Valid (finger & >0): {res['valid_rows']:,} ({res['valid_ratio']:.1f}%)")
    print("-" * 80)

    print("1. INTEGRITAS SENSOR LINGKUNGAN (DHT22):")
    print(f"   - Kelembapan (hum) Null / Missing  : {res['null_percentages']['hum']:.1f}%")
    print(f"   - Suhu Ruang (room_temp) Null      : {res['null_percentages']['room_temp']:.1f}%")
    print("-" * 80)

    print("2. STATISTIK DESKRIPTIF SENSOR:")
    header = f"   {'Sensor':<12} | {'N Valid':>8} | {'Min':>7} | {'Mean':>7} | {'Max':>7} | {'Std':>7} | {'Satuan':<6}"
    print(header)
    print("   " + "-" * 70)
    for name, s in res["sensor_stats"].items():
        if s["count"] > 0:
            print(f"   {name:<12} | {s['count']:>8} | {s['min']:>7.2f} | {s['mean']:>7.2f} | {s['max']:>7.2f} | {s['std']:>7.2f} | {s['unit']:<6}")
        else:
            print(f"   {name:<12} | {0:>8} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {s['unit']:<6}")
    print("-" * 80)

    print("3. DETEKSI NILAI SENSOR MACET / FROZEN (>30 DETIK BERULANG):")
    print(f"   - Jumlah Kejadian Macet (>30s)    : {res['stuck_streaks_count']} kejadian")
    print(f"   - Total Baris Terindikasi Macet   : {res['stuck_rows_count']} baris")
    if res["stuck_rows_count"] == 0:
        print("   - Status: BAIK (Sensor aktif bervariasi secara fisiologis)")
    else:
        print("   - Peringatan: Terdapat nilai sensor statis tanpa fluktuasi fisiologis.")
    print("-" * 80)

    print("4. INDIKASI DATA SIMULATOR VS HARDWARE ASLI:")
    sim = res["simulator_indicators"]
    if sim["is_simulator"]:
        print(f"   - Status : [PERINGATAN] {sim['summary']}")
        for r in sim["reasons"]:
            print(f"     * {r}")
    else:
        print(f"   - Status : [OK] {sim['summary']}")
    print("-" * 80)

    print("5. EVALUASI SYARAT ML_EVAL (PANJANG MINIMAL & KELAYAKAN):")
    if res["ml_eval_qualified"]:
        print("   - Status Kelayakan : [LULUS] Sesi memenuhi syarat pemodelan ml_eval.")
        print(f"   - Panjang Valid    : {res['valid_rows']} baris (>= {ML_EVAL_RECOMMENDED_SESSION_ROWS} rekomendasi).")
    else:
        print("   - Status Kelayakan : [BELUM MEMENUHI]")
        for reason in res["ml_eval_reasons"]:
            print(f"     * {reason}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="Audit kualitas perekaman sesi telemetri IoT untuk ml_eval."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path ke file CSV rekaman atau folder sesi (mis. data_sessions/session_001/)"
    )
    args = parser.parse_args()

    try:
        results = analyze_recording(args.input)
        print_report(results)
    except Exception as e:
        print(f"[ERROR] Gagal menganalisis perekaman: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
