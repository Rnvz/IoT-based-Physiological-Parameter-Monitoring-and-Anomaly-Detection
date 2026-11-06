# Model Card: ae_v1 (V2)

## Model Overview
- **Architecture**: Compact Dense Autoencoder ($2 \to 16 \to 8 \to 2 \to 8 \to 16 \to 2$)
- **Parameters**: 533 parameters (< 10,000 parameter edge limit)
- **Primary Task**: Unsupervised physiological anomaly detection (reconstruction MSE error)
- **Features Used**: Heart_Rate, SpO2
- **Export Formats**: ONNX (`model.onnx`) and NumPy Fallback (`numpy_forward.py`)

## Clinical & Operational Disclaimer
> [!CAUTION]
> **NON-CLINICAL DEMO SYSTEM ONLY**
> This model is trained on public research datasets and is NOT a medical device.
> It MUST NOT be used for clinical diagnosis, patient triage, or life-critical medical monitoring.

## Public Training Data & Limitations
1. **Multi-Source Composition**:
   - `nasirayub2` (cross-sectional, 200,020 rows): Synthetic/heuristic multi-risk dataset. Labels derived from hypertension criteria (systolic/diastolic blood pressure) and BMI, NOT vitals.
   - `engrarri21` (longitudinal, 21,523 rows): Single-subject ICU streaming recording with high exact duplicate fraction (86.0%). Untuk interpretasi metrik uji PR-AUC tanpa bias duplikasi, lihat subset `engrarri21_unique` pada `docs/AOL_TEST_REPORT.md`.
   - `gourangomandal` (cross-sectional, 60,000 rows): Extreme label circularity (95.5% agreement with simple threshold rules).
2. **Core vs. Skin Temperature Discrepancy**:
   - Public datasets record **core/oral body temperature** (~36.5–37.5 °C).
   - Physical IoT sensors (DS18B20 or thermistors on finger/wrist) measure **peripheral skin contact temperature** (~31.0–35.0 °C).
   - Models trained with core temperature features (`v3`) will perceive peripheral skin readings as severe hypothermia unless temperature-free (`v2`) or compensated.

## Domain Shift Sanity Checks
- **Resting Stream Benchmark**: Replayed at 31 °C, 33 °C, 35 °C, and 36.5 °C (passed with 0.0% alarms at 33 °C).
- **Persistence Filtering**: $N=3$ consecutive anomaly samples required to trigger alert state.
- **Detailed Test Verification**: See [AOL_TEST_REPORT.md](../../docs/AOL_TEST_REPORT.md) for full metrics, bootstrap CIs, and domain sensitivity injection testing.

## Parity Verification
- **PyTorch vs ONNX Max Absolute Difference**: Verified $< 10^{-5}$ pada sampel uji standar.
- **PyTorch vs NumPy Forward Max Absolute Difference**: Verified $< 10^{-5}$ pada sampel uji standar.
- **ONNX vs NumPy Forward Parity (10.000 points)**:
  - **Max Absolute Difference ($\text{score} \le 10.0$)**: $4.77 \times 10^{-6}$
  - **Max Relative Difference (all points)**: $4.19 \times 10^{-4}$ ($0.0419\%$)
  - **Threshold Boundary Region ($[0.005, 0.05]$ covering $\text{warn}=0.0136$ & $\text{crit}=0.0248$)**:
    - Max Absolute Difference: $4.43 \times 10^{-7}$
    - Max Relative Difference: $2.18 \times 10^{-5}$ ($0.00218\%$)
- **Evaluasi Paritas Grid Komprehensif (4.551 Titik: HR 40–150 $\times$ SpO2 80–100)**:
  - **Selisih Maks Skor Terukur**:
    - ONNX vs NumPy: $1,53 \times 10^{-5}$ ($1,5 \times 10^{-5}$)
    - PyTorch vs ONNX: $1,14 \times 10^{-5}$
    - PyTorch vs NumPy: $1,14 \times 10^{-5}$
  - **Status Toleransi & Keputusan Severity**:
    - Toleransi rencana $10^{-5}$ terlewati pada skor mentah ($1,5 \times 10^{-5}$ ONNX-NumPy dan $1,14 \times 10^{-5}$ PyTorch-ONNX).
    - Keputusan severity antara backend produksi **ONNX vs NumPy adalah 100% SAMA** (0 dari 4.551 titik berbeda).
    - Keputusan severity antara **PyTorch vs ONNX/NumPy BERBEDA pada 1 titik** (HR=111,0 BPM, SpO2=97,0%: skor PyTorch $0,02480478$ [Severity 2] vs ONNX $0,02480471$ [Severity 1], berjarak hanya $4,66 \times 10^{-8}$ dari batas kritis $0,02480476$).
- **Conclusion**: Pada kedua runtime produksi yang dideploy (ONNX dan NumPy fallback), kesepakatan klasifikasi adalah 100% identik tanpa perbedaan severity sama sekali.

