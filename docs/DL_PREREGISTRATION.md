# Deep Learning Benchmark & Deployment Pre-Registration Protocol

> [!IMPORTANT]
> **PRE-REGISTRATION STATUS: FROZEN BEFORE TEST EVALUATION**
> This document pre-registers all evaluation criteria, feature sets, threshold calibration rules, and deployment criteria BEFORE scoring on the held-out test partition. Under the scientific protocol, the test partition is evaluated exactly once without post-hoc modification.

---

## 1. Scope & Objective

Benchmarking compact deep-learning anomaly detection architectures (Dense Autoencoder, Variational Autoencoder) against deterministic rule-based logic and classical shallow baselines (Isolation Forest, LOF, One-Class SVM) on harmonized multi-source physiological datasets (`nasirayub2`, `engrarri21`, `gourangomandal`).

---

## 2. Feature Variants

Two standardized feature variants are evaluated:
1. **Variant 3 (`v3`) — Full Tri-Channel Vitals**:
   - Features: `Heart_Rate`, `SpO2`, `Temperature`.
   - Normalized using `StandardScaler` fitted **strictly on normal train rows** (`OUTPUT == 0`).
2. **Variant 2 (`v2`) — Dual-Channel Vitals (Temperature-Free)**:
   - Features: `Heart_Rate`, `SpO2`.
   - Designed to eliminate sensitivity to the domain discrepancy between clinical core body temperature (public data ~36.5–37.5 °C) and peripheral skin contact temperature (IoT device ~31.0–35.0 °C).

---

## 3. Model Architectures & Parameter Budget

All deep learning models must satisfy a strict edge parameter budget ($\le 10,000$ parameters) suitable for microcontrollers or local edge gateways:

1. **Dense Autoencoder (`DenseAE`)**:
   - Compression bottleneck: $3 \to 16 \to 8 \to 2 \to 8 \to 16 \to 3$ (for `v3`) and $2 \to 16 \to 8 \to 2 \to 8 \to 16 \to 2$ (for `v2`).
   - Activations: GELU or ReLU with Dropout ($p=0.1$) and LayerNorm/BatchNorm.
   - Anomaly Score: Mean Squared Reconstruction Error $\|x - \hat{x}\|_2^2$.
2. **Variational Autoencoder (`DenseVAE`)**:
   - Encoder: $D_{in} \to 16 \to 8 \to (\mu=2, \log\sigma^2=2)$.
   - Decoder: $2 \to 8 \to 16 \to D_{in}$.
   - Loss: ELBO = Reconstruction MSE + $\beta \cdot D_{KL}(q(z|x) \parallel \mathcal{N}(0, I))$ with $\beta=0.001$.
   - Anomaly Score: Reconstruction MSE error $\|x - \hat{x}\|_2^2$.

---

## 4. Pre-Registered Validation Selection Metric

- **Primary Selection Metric**: **PR-AUC (Precision-Recall Area Under Curve)** evaluated on the full **Validation Partition** (including normal and abnormal samples).
- **Early Stopping**: Monitored on normal validation reconstruction loss (`patience=5`, minimum $\Delta = 10^{-4}$).
- **Random Seeds**: 5 fixed seeds (`42, 43, 44, 45, 46`). Results reported as Mean $\pm$ Standard Deviation.

---

## 5. Standardized Severity Threshold Calibration

Operating thresholds are calibrated strictly on **Normal Validation Rows** (`OUTPUT == 0`):
- **Warning Threshold (Severity 1)**: 95th percentile ($P_{95}$) of normal validation anomaly scores (targeting $\le 5\%$ nominal false alarm rate).
- **Alert / Critical Threshold (Severity 2)**: 99th percentile ($P_{99}$) of normal validation anomaly scores (targeting $\le 1\%$ nominal false alarm rate).

Test rows and abnormal validation rows are never inspected during threshold calibration.

---

## 6. Temporal Persistence Filter

To suppress transient single-sample noise:
- **Persistence Length**: $N = 3$ consecutive samples above threshold required to trigger state transition.
- **Reset**: Immediate return to normal if an incoming reading falls below the warning threshold.

---

## 7. Pre-Registered Deployment Criterion (Domain Shift Sanity Check)

To be certified as **Deployable to Edge Devices**, a model variant must pass the following domain-shift sanity check:

> **Criterion**: "A variant is deployable to edge hardware only if its false alarm rate on a synthetic resting stream at 33 °C (peripheral skin baseline) is strictly below 5.0%."

- **Synthetic Resting Stream Specification**:
  - Sample size: 400 contiguous samples at 1 Hz.
  - Vitals: Heart Rate $72 \pm 3$ BPM, SpO2 $98 \pm 0.5\%$, Temperature fixed at $33.0 \pm 0.3$ °C.
- Variants failing this criterion (due to core-vs-skin temperature mismatch) are mechanically flagged as **NON-DEPLOYABLE** for direct physical skin monitoring without calibration.

---

## 8. Binding Single-Pass Test Statement

The held-out test partition (`splits/public_split.json`, SHA256: `15a424ff1118cac8dfe3cbd20d80f028863b46edc235db24a2dd50adef6d1635`) is scored exactly **ONCE**. No hyperparameter tuning, feature re-engineering, or threshold adjustments are permitted post-evaluation.
