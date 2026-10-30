#!/usr/bin/env python3
"""
Baseline Statistics & Threshold Calibration Suggester
Membaca dataset perekaman readings.csv (data mentah dengan finger=True),
menghitung statistik deskriptif per sensor (HR, SpO2, Skin Temp),
dan memberikan SARAN ambang batas warning & alert berbasis deviasi standar (mean +/- k*std).

Catatan: Script ini HANYA memberikan saran dan TIDAK PERNAH mengedit .env secara otomatis.
Ambang batas ini ditujukan untuk kalibrasi baseline mandiri, bukan diagnosis klinis.
"""

import sys
import os
import csv
import argparse
import numpy as np
from typing import List, Dict, Any, Optional

def parse_args():
    parser = argparse.ArgumentParser(description="Analisis Baseline Perekaman Sensor Fisiologis")
    parser.add_argument(
        "--input", "-i",
        default="data/readings.csv",
        help="Path ke file readings.csv (default: data/readings.csv)"
    )
    parser.add_argument(
        "--k-warn",
        type=float,
        default=2.0,
        help="Pengali standar deviasi untuk batas warning (default: 2.0 -> ~95%% interval)"
    )
    parser.add_argument(
        "--k-alert",
        type=float,
        default=3.0,
        help="Pengali standar deviasi untuk batas alert (default: 3.0 -> ~99.7%% interval)"
    )
    return parser.parse_args()

def analyze_metric(values: List[float], name: str, unit: str, k_warn: float, k_alert: float) -> Optional[Dict[str, Any]]:
    if not values:
        return None

    arr = np.array(values, dtype=np.float64)
    count = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if count > 1 else 0.0
    min_val = float(np.min(arr))
    p5_val = float(np.percentile(arr, 5))
    p50_val = float(np.percentile(arr, 50))
    p95_val = float(np.percentile(arr, 95))
    max_val = float(np.max(arr))

    # Suggested bounds
    warn_low = mean_val - k_warn * std_val
    warn_high = mean_val + k_warn * std_val
    alert_low = mean_val - k_alert * std_val
    alert_high = mean_val + k_alert * std_val

    # Physiological clamp for SpO2 (cannot exceed 100%)
    if "SpO2" in name:
        warn_high = min(100.0, warn_high)
        alert_high = min(100.0, alert_high)

    return {
        "name": name,
        "unit": unit,
        "count": count,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "p5": p5_val,
        "p50": p50_val,
        "p95": p95_val,
        "max": max_val,
        "warn_low": warn_low,
        "warn_high": warn_high,
        "alert_low": alert_low,
        "alert_high": alert_high,
    }

def print_report(metrics_data: List[Dict[str, Any]], k_warn: float, k_alert: float, total_rows: int, valid_finger_rows: int):
    print("=" * 78)
    print("  STATISTIK BASELINE PEREKAMAN SENSOR FISIOLOGIS (NON-KLINIS)")
    print("=" * 78)
    print(f"Total baris dibaca         : {total_rows:,}")
    print(f"Baris dengan finger=True   : {valid_finger_rows:,}")
    print(f"Faktor pengali deviasi (k) : k_warn={k_warn:.1f}σ, k_alert={k_alert:.1f}σ")
    print("-" * 78)

    header = f"{'Metrik':<12} | {'N':>6} | {'Mean':>6} | {'Std':>5} | {'Min':>5} | {'p5':>5} | {'p50':>5} | {'p95':>5} | {'Max':>5} | {'Satuan':<6}"
    print(header)
    print("-" * 78)

    for m in metrics_data:
        print(
            f"{m['name']:<12} | {m['count']:>6} | {m['mean']:>6.1f} | {m['std']:>5.2f} | "
            f"{m['min']:>5.1f} | {m['p5']:>5.1f} | {m['p50']:>5.1f} | {m['p95']:>5.1f} | {m['max']:>5.1f} | {m['unit']:<6}"
        )
    print("-" * 78)

    print("\n" + "=" * 78)
    print("  USULAN AMBANG BATAS RULE ENGINE (SARAN KALIBRASI MANDIRI)")
    print("=" * 78)
    for m in metrics_data:
        print(f"\n[{m['name']}] (Baseline: mean={m['mean']:.1f}, std={m['std']:.2f})")
        print(f"  • Warning Band (Severity 1) : < {m['warn_low']:.1f} atau > {m['warn_high']:.1f} {m['unit']}")
        print(f"  • Alert Band   (Severity 2) : < {m['alert_low']:.1f} atau > {m['alert_high']:.1f} {m['unit']}")

    print("\n" + "-" * 78)
    print("Catatan Penting:")
    print("1. Nilai di atas adalah estimasi statistik (mean ± k*std) dari dataset perekaman.")
    print("2. Nilai ini HANYA SARAN untuk demo/tugas akhir dan TIDAK BOLEH dianggap rujukan medis.")
    print("3. File .env TIDAK diubah secara otomatis. Salin nilai yang diinginkan ke .env jika diperlukan.")
    print("=" * 78 + "\n")

def main():
    args = parse_args()
    filepath = os.path.abspath(args.input)

    if not os.path.exists(filepath):
        print(f"[!] File tidak ditemukan: {filepath}")
        print("    Jalankan perekaman atau simulasi terlebih dahulu untuk menghasilkan readings.csv.")
        sys.exit(1)

    hr_list = []
    spo2_list = []
    temp_list = []
    total_rows = 0
    valid_finger_rows = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            finger_str = row.get("finger", "").strip().lower()
            if finger_str not in ("true", "1", "t"):
                continue

            valid_finger_rows += 1

            # HR
            hr_raw = row.get("hr", "").strip()
            if hr_raw:
                try:
                    hr_val = float(hr_raw)
                    if hr_val > 0.0:
                        hr_list.append(hr_val)
                except ValueError:
                    pass

            # SpO2
            spo2_raw = row.get("spo2", "").strip()
            if spo2_raw:
                try:
                    spo2_val = float(spo2_raw)
                    if spo2_val > 0.0:
                        spo2_list.append(spo2_val)
                except ValueError:
                    pass

            # Temp
            temp_raw = row.get("temp", "").strip()
            if temp_raw:
                try:
                    temp_val = float(temp_raw)
                    if temp_val > 0.0:
                        temp_list.append(temp_val)
                except ValueError:
                    pass

    metrics_to_print = []
    for values, name, unit in [
        (hr_list, "Heart Rate", "bpm"),
        (spo2_list, "SpO2", "%"),
        (temp_list, "Skin Temp", "°C")
    ]:
        res = analyze_metric(values, name, unit, args.k_warn, args.k_alert)
        if res:
            metrics_to_print.append(res)

    if not metrics_to_print:
        print(f"[!] Tidak ada data valid dengan finger=True pada {filepath}.")
        sys.exit(0)

    print_report(metrics_to_print, args.k_warn, args.k_alert, total_rows, valid_finger_rows)

if __name__ == "__main__":
    main()
