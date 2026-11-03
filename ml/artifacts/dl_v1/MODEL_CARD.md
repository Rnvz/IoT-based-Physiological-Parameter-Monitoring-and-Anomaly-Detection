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
   - `engrarri21` (longitudinal, 21,523 rows): Single-subject ICU streaming recording with high exact duplicate fraction (86.0%).
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
- **PyTorch vs ONNX Max Absolute Difference**: Verified $< 10^{-5}$
- **PyTorch vs NumPy Forward Max Absolute Difference**: Verified $< 10^{-5}$
- **ONNX vs NumPy Forward (Python 3.14 venv)**: Verified $8.02 \times 10^{-6}$ (< $10^{-5}$)
