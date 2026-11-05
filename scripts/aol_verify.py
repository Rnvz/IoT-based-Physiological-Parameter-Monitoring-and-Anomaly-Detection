"""
scripts/aol_verify.py: Comprehensive verification audit runner for AOL DL benchmark.
Computes all split counts, uniqueness audits, tied-score ROC metrics,
bootstrap CI audits, and domain sensitivity tests (grid, 1D sweep, graded injections).
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.rules.engine import evaluate_rules
from ml.artifacts.dl_v1.numpy_forward import (
    EdgeAutoencoderInference,
    layer_norm_np,
    gelu_np,
    WARN_THRESHOLD,
    CRIT_THRESHOLD,
    SCALER_MEAN,
    SCALER_SCALE
)
from ml_eval.public_split import load_public_split
from ml.dl.evaluate import compute_metrics, patient_group_bootstrap

def main():
    print("=" * 80)
    print("AOL VERIFICATION AUDIT (dl-public)")
    print("=" * 80)

    harm_path = os.path.join(PROJECT_ROOT, "ml_pipeline/data/harmonized_vitals.csv")
    split_path = os.path.join(PROJECT_ROOT, "splits/public_split.json")

    harm = pd.read_csv(harm_path)
    harm["row_id"] = np.arange(len(harm))
    split_meta, split_sha = load_public_split(split_path)

    print(f"Dataset : {harm_path} (Total rows: {len(harm)})")
    print(f"Split   : {split_path} (SHA256: {split_sha})")

    train_idx = split_meta["indices"]["train"]
    val_idx = split_meta["indices"]["val"]
    test_idx = split_meta["indices"]["test"]

    train_df = harm.iloc[train_idx].reset_index(drop=True)
    val_df = harm.iloc[val_idx].reset_index(drop=True)
    test_df = harm.iloc[test_idx].reset_index(drop=True)

    norm_train = train_df[train_df["OUTPUT"] == 0]
    norm_val = val_df[val_df["OUTPUT"] == 0]

    scaler_v3 = StandardScaler().fit(norm_train[["Heart_Rate", "SpO2", "Temperature"]].values)
    scaler_v2 = StandardScaler().fit(norm_train[["Heart_Rate", "SpO2"]].values)

    results_payload = {
        "split_sha256": split_sha,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print("\n" + "=" * 80)
    print("TASK 1: RECALCULATED COUNTS (PER SOURCE x SPLIT x CLASS)")
    print("=" * 80)

    sources = ["nasirayub2", "engrarri21", "gourangomandal"]
    splits = ["train", "val", "test"]
    count_rows = []

    for src in sources:
        src_df = harm[harm["Source"] == src]
        for sp in splits:
            idx = set(split_meta["indices"][sp])
            sp_df = src_df[src_df["row_id"].isin(idx)]
            n_tot = len(sp_df)
            n_norm = int(np.sum(sp_df["OUTPUT"] == 0))
            n_abn = int(np.sum(sp_df["OUTPUT"] == 1))
            prev = n_abn / n_tot if n_tot > 0 else 0.0
            count_rows.append({
                "Source": src,
                "Split": sp,
                "Normal_0": n_norm,
                "Abnormal_1": n_abn,
                "Total": n_tot,
                "Prevalence_pct": round(prev * 100, 2)
            })

    for sp in splits:
        idx = set(split_meta["indices"][sp])
        sp_df = harm[harm["row_id"].isin(idx)]
        n_tot = len(sp_df)
        n_norm = int(np.sum(sp_df["OUTPUT"] == 0))
        n_abn = int(np.sum(sp_df["OUTPUT"] == 1))
        prev = n_abn / n_tot if n_tot > 0 else 0.0
        count_rows.append({
            "Source": "POOLED",
            "Split": sp,
            "Normal_0": n_norm,
            "Abnormal_1": n_abn,
            "Total": n_tot,
            "Prevalence_pct": round(prev * 100, 2)
        })

    df_counts = pd.DataFrame(count_rows)
    print(df_counts.to_string(index=False))
    results_payload["counts_table"] = count_rows

    test_eng = harm[(harm["Source"] == "engrarri21") & (harm["row_id"].isin(set(test_idx)))]
    vitals_cols = ["Heart_Rate", "SpO2", "Temperature"]
    all_cols = ["Patient_ID", "Heart_Rate", "SpO2", "Temperature", "OUTPUT"]

    n_eng_tot = len(test_eng)
    n_eng_vitals_uniq = len(test_eng.drop_duplicates(subset=vitals_cols))
    n_eng_all_uniq = len(test_eng.drop_duplicates(subset=all_cols))

    eng_norm = test_eng[test_eng["OUTPUT"] == 0]
    n_eng_norm_tot = len(eng_norm)
    n_eng_norm_vitals_uniq = len(eng_norm.drop_duplicates(subset=vitals_cols))
    n_eng_norm_all_uniq = len(eng_norm.drop_duplicates(subset=all_cols))

    print("\n--- ENGRARRI21 TEST SET UNIQUENESS AUDIT ---")
    print(f"Total test rows               : {n_eng_tot}")
    print(f"Unique rows (vitals cols)     : {n_eng_vitals_uniq} (Duplicate: {100 - n_eng_vitals_uniq/n_eng_tot*100:.2f}%)")
    print(f"Unique rows (all cols)        : {n_eng_all_uniq} (Duplicate: {100 - n_eng_all_uniq/n_eng_tot*100:.2f}%)")
    print(f"Normal test rows (OUTPUT=0)   : {n_eng_norm_tot}")
    print(f"Unique normal (vitals cols)   : {n_eng_norm_vitals_uniq} (Duplicate: {100 - n_eng_norm_vitals_uniq/n_eng_norm_tot*100:.2f}%)")
    print(f"Unique normal (all cols)      : {n_eng_norm_all_uniq} (Duplicate: {100 - n_eng_norm_all_uniq/n_eng_norm_tot*100:.2f}%)")

    results_payload["engrarri21_uniqueness"] = {
        "total_test": n_eng_tot,
        "unique_vitals_test": n_eng_vitals_uniq,
        "unique_all_cols_test": n_eng_all_uniq,
        "normal_test": n_eng_norm_tot,
        "unique_normal_vitals_test": n_eng_norm_vitals_uniq,
        "unique_normal_all_cols_test": n_eng_norm_all_uniq
    }

    print("\n--- RULE VS OUTPUT AGREEMENT (CIRCULARITY AUDIT ON REAL DATA) ---")
    circ_records = {}
    for src in sources:
        sdf = harm[harm["Source"] == src]
        agreements = []
        for _, row in sdf.iterrows():
            hr = row["Heart_Rate"]
            spo2 = row["SpO2"] if not np.isnan(row["SpO2"]) else None
            temp = row["Temperature"] if not np.isnan(row["Temperature"]) else None
            sev, _ = evaluate_rules(hr, spo2, temp)
            rule_pred = 1 if sev >= 1 else 0
            agreements.append(rule_pred == int(row["OUTPUT"]))
        acc = float(np.mean(agreements) * 100)
        circ_records[src] = round(acc, 2)
        print(f"Source: {src:15s} | N={len(sdf):6d} | Rule vs OUTPUT Agreement: {acc:.2f}%")
    results_payload["rule_circularity_agreement_pct"] = circ_records

    print("\n" + "=" * 80)
    print("TASK 2: COMPREHENSIVE TEST METRICS (WITH TIED-SCORE ROC REC@5%FPR)")
    print("=" * 80)

    def score_rule(df, variant="v3"):
        scores = []
        for _, row in df.iterrows():
            hr = row["Heart_Rate"]
            spo2 = row["SpO2"]
            temp = row["Temperature"] if variant == "v3" else 36.5
            sev, _ = evaluate_rules(hr, spo2, temp)
            score = 1.0 if sev == 2 else (0.5 if sev == 1 else 0.0)
            scores.append(score)
        return np.array(scores, dtype=np.float64)

    print("Training shallow baselines on train normal...")
    if_v3 = IsolationForest(n_estimators=100, random_state=42).fit(scaler_v3.transform(norm_train[["Heart_Rate", "SpO2", "Temperature"]].values))
    if_v2 = IsolationForest(n_estimators=100, random_state=42).fit(scaler_v2.transform(norm_train[["Heart_Rate", "SpO2"]].values))

    sub_idx = np.random.default_rng(42).choice(len(norm_train), size=25000, replace=False)
    lof_v3 = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(scaler_v3.transform(norm_train.iloc[sub_idx][["Heart_Rate", "SpO2", "Temperature"]].values))
    lof_v2 = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(scaler_v2.transform(norm_train.iloc[sub_idx][["Heart_Rate", "SpO2"]].values))

    ocsvm_v3 = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(scaler_v3.transform(norm_train.iloc[sub_idx][["Heart_Rate", "SpO2", "Temperature"]].values))
    ocsvm_v2 = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(scaler_v2.transform(norm_train.iloc[sub_idx][["Heart_Rate", "SpO2"]].values))

    weights_path = os.path.join(PROJECT_ROOT, "ml/artifacts/dl_v1/model_weights.npz")
    infer_ae_v2 = EdgeAutoencoderInference(weights_path)
    w_ae = infer_ae_v2.w

    def score_dense_ae_v2(X_scaled):
        h = X_scaled @ w_ae["encoder.0.weight"].T + w_ae["encoder.0.bias"]
        h = layer_norm_np(h, w_ae["encoder.1.weight"], w_ae["encoder.1.bias"])
        h = gelu_np(h)
        h = h @ w_ae["encoder.4.weight"].T + w_ae["encoder.4.bias"]
        h = layer_norm_np(h, w_ae["encoder.5.weight"], w_ae["encoder.5.bias"])
        h = gelu_np(h)
        z = h @ w_ae["encoder.7.weight"].T + w_ae["encoder.7.bias"]
        d = z @ w_ae["decoder.0.weight"].T + w_ae["decoder.0.bias"]
        d = layer_norm_np(d, w_ae["decoder.1.weight"], w_ae["decoder.1.bias"])
        d = gelu_np(d)
        d = d @ w_ae["decoder.4.weight"].T + w_ae["decoder.4.bias"]
        d = layer_norm_np(d, w_ae["decoder.5.weight"], w_ae["decoder.5.bias"])
        d = gelu_np(d)
        rec = d @ w_ae["decoder.7.weight"].T + w_ae["decoder.7.bias"]
        return np.mean((X_scaled - rec) ** 2, axis=-1)

    models_to_eval = [
        ("Rule Engine (v3)", "v3", lambda df, X: score_rule(df, "v3")),
        ("Rule Engine (v2)", "v2", lambda df, X: score_rule(df, "v2")),
        ("Isolation Forest (v3)", "v3", lambda df, X: -if_v3.decision_function(X)),
        ("Isolation Forest (v2)", "v2", lambda df, X: -if_v2.decision_function(X)),
        ("Local Outlier Factor (v3)", "v3", lambda df, X: -lof_v3.decision_function(X)),
        ("Local Outlier Factor (v2)", "v2", lambda df, X: -lof_v2.decision_function(X)),
        ("One-Class SVM (v3)", "v3", lambda df, X: -ocsvm_v3.decision_function(X)),
        ("One-Class SVM (v2)", "v2", lambda df, X: -ocsvm_v2.decision_function(X)),
        ("DenseAE (v2) [Seed 42]", "v2", lambda df, X: score_dense_ae_v2(X)),
    ]

    eng_test_unique_df = test_eng.drop_duplicates(subset=vitals_cols).reset_index(drop=True)
    subsets_eval = [
        ("POOLED", test_df),
        ("nasirayub2", test_df[test_df["Source"] == "nasirayub2"].reset_index(drop=True)),
        ("engrarri21", test_eng.reset_index(drop=True)),
        ("engrarri21_unique", eng_test_unique_df),
        ("gourangomandal", test_df[test_df["Source"] == "gourangomandal"].reset_index(drop=True)),
    ]

    metrics_table = []

    for m_name, variant, score_fn in models_to_eval:
        scaler = scaler_v3 if variant == "v3" else scaler_v2
        cols = ["Heart_Rate", "SpO2", "Temperature"] if variant == "v3" else ["Heart_Rate", "SpO2"]

        val_X = scaler.transform(norm_val[cols].values)
        val_scores = score_fn(norm_val, val_X)
        warn_th = float(np.percentile(val_scores, 95.0)) if "Rule" not in m_name else 0.5

        for s_name, s_df in subsets_eval:
            s_y = s_df["OUTPUT"].values
            s_X = scaler.transform(s_df[cols].values)
            s_scores = score_fn(s_df, s_X)

            m = compute_metrics(s_y, s_scores, warn_th)
            prev = float(np.mean(s_y == 1))
            pr_minus_prev = (m["pr_auc"] - prev) if m["pr_auc"] is not None else None

            metrics_table.append({
                "Model": m_name,
                "Source": s_name,
                "Prevalence": round(prev, 4),
                "PR-AUC": round(m["pr_auc"], 4) if m["pr_auc"] is not None else "-",
                "PR - Prev": round(pr_minus_prev, 4) if pr_minus_prev is not None else "-",
                "ROC-AUC": round(m["roc_auc"], 4) if m["roc_auc"] is not None else "-",
                "Rec@5%FPR": round(m["recall_at_5pct_fpr"], 4) if m["recall_at_5pct_fpr"] is not None else "-",
                "F1": round(m["f1"], 4) if m["f1"] is not None else "-"
            })

    df_metrics = pd.DataFrame(metrics_table)
    print(df_metrics.to_string(index=False))
    results_payload["metrics_table"] = metrics_table

    print("\n" + "=" * 80)
    print("TASK 3: BOOTSTRAP RESAMPLING AUDIT FOR ENGRARRI21")
    print("=" * 80)
    print("Code inspection of patient_group_bootstrap in ml/dl/evaluate.py:")
    print("When n_pts == 1 (engrarri21 has Patient_ID = 'engrarri21_1' for all rows):")
    print("  -> Execution falls to line 150: block_size = 50, n_blocks = n_samples // 50")
    print("  -> Resamples contiguous 50-row blocks WITHIN the same patient's stream.")
    print("Audit verdict:")
    print("  [!] CI label: 'tidak valid: 1 subjek, 86% duplikat'.")
    print("  Because block bootstrap inside a single individual cannot estimate across-patient variability.")
    print("\nMetrics on engrarri21 UNIQUE rows only (comparison):")
    df_eng_comp = df_metrics[df_metrics["Source"].isin(["engrarri21", "engrarri21_unique"])].copy()
    print(df_eng_comp[["Model", "Source", "PR-AUC", "PR - Prev", "ROC-AUC", "Rec@5%FPR", "F1"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK 5: SENSITIVITY TESTING (BUKAN HASIL TEST) — DENSEAE v2 SEED 42 vs RULE")
    print("=" * 80)

    print("\n--- 5a. FALSE ALARM GRID (20 SEEDS x 400 SAMPLES, PERSISTENCE N=3) ---")
    hr_means = [55, 60, 72, 85, 95]
    spo2_means = [95, 96, 98]
    grid_results = []

    for hr_m in hr_means:
        for sp_m in spo2_means:
            ae_warn_shares = []
            ae_crit_shares = []
            rule_raw_warn_shares = []
            rule_raw_crit_shares = []
            rule_n3_warn_shares = []
            rule_n3_crit_shares = []

            for seed in range(20):
                rng = np.random.default_rng(seed)
                hr_seq = rng.normal(hr_m, 3.0, size=400)
                spo2_seq = np.clip(rng.normal(sp_m, 0.5, size=400), 80.0, 100.0)

                ae_infer = EdgeAutoencoderInference(weights_path)
                ae_sevs = []
                for h_val, s_val in zip(hr_seq, spo2_seq):
                    p = ae_infer.predict_sample({"Heart_Rate": h_val, "SpO2": s_val})
                    ae_sevs.append(p["severity"])

                ae_warn_shares.append(np.mean(np.array(ae_sevs) >= 1) * 100)
                ae_crit_shares.append(np.mean(np.array(ae_sevs) == 2) * 100)

                raw_sevs = []
                n3_sevs = []
                rule_crit_count = 0
                rule_warn_count = 0
                rule_active_sev = 0

                for h_val, s_val in zip(hr_seq, spo2_seq):
                    sev, _ = evaluate_rules(h_val, s_val, 36.5)
                    raw_sevs.append(sev)

                    if sev == 2:
                        rule_crit_count += 1
                        rule_warn_count += 1
                    elif sev == 1:
                        rule_crit_count = 0
                        rule_warn_count += 1
                    else:
                        rule_crit_count = 0
                        rule_warn_count = 0

                    if rule_crit_count >= 3:
                        rule_active_sev = 2
                    elif rule_warn_count >= 3:
                        rule_active_sev = 1
                    elif rule_warn_count == 0:
                        rule_active_sev = 0
                    n3_sevs.append(rule_active_sev)

                rule_raw_warn_shares.append(np.mean(np.array(raw_sevs) >= 1) * 100)
                rule_raw_crit_shares.append(np.mean(np.array(raw_sevs) == 2) * 100)
                rule_n3_warn_shares.append(np.mean(np.array(n3_sevs) >= 1) * 100)
                rule_n3_crit_shares.append(np.mean(np.array(n3_sevs) == 2) * 100)

            grid_results.append({
                "HR_mean": hr_m,
                "SpO2_mean": sp_m,
                "DenseAE_N3_Sev>=1_%": round(float(np.mean(ae_warn_shares)), 2),
                "DenseAE_N3_Sev2_%": round(float(np.mean(ae_crit_shares)), 2),
                "Rule_Raw_Sev>=1_%": round(float(np.mean(rule_raw_warn_shares)), 2),
                "Rule_Raw_Sev2_%": round(float(np.mean(rule_raw_crit_shares)), 2),
                "Rule_N3_Sev>=1_%": round(float(np.mean(rule_n3_warn_shares)), 2),
                "Rule_N3_Sev2_%": round(float(np.mean(rule_n3_crit_shares)), 2)
            })

    df_grid = pd.DataFrame(grid_results)
    print(df_grid.to_string(index=False))
    results_payload["grid_sensitivity"] = grid_results

    print("\n--- 5b. 1D THRESHOLD CROSSING SWEEPS ---")
    hr_sweep = np.linspace(40.0, 140.0, 1001)
    hr_ae_scores = []
    for h in hr_sweep:
        raw = np.array([[h, 98.0]], dtype=np.float32)
        norm = (raw - SCALER_MEAN) / SCALER_SCALE
        sc = score_dense_ae_v2(norm)[0]
        hr_ae_scores.append(sc)
    hr_ae_scores = np.array(hr_ae_scores)

    hr_low_warn = hr_sweep[np.where(hr_ae_scores >= WARN_THRESHOLD)[0]]
    hr_low_warn_val = hr_low_warn[hr_low_warn < 76][-1] if any(hr_low_warn < 76) else None
    hr_high_warn_val = hr_low_warn[hr_low_warn > 76][0] if any(hr_low_warn > 76) else None

    hr_low_crit = hr_sweep[np.where(hr_ae_scores >= CRIT_THRESHOLD)[0]]
    hr_low_crit_val = hr_low_crit[hr_low_crit < 76][-1] if any(hr_low_crit < 76) else None
    hr_high_crit_val = hr_low_crit[hr_low_crit > 76][0] if any(hr_low_crit > 76) else None

    spo2_sweep = np.linspace(80.0, 100.0, 201)
    spo2_ae_scores = []
    for sp in spo2_sweep:
        raw = np.array([[72.0, sp]], dtype=np.float32)
        norm = (raw - SCALER_MEAN) / SCALER_SCALE
        sc = score_dense_ae_v2(norm)[0]
        spo2_ae_scores.append(sc)
    spo2_ae_scores = np.array(spo2_ae_scores)

    spo2_cross_warn = spo2_sweep[np.where(spo2_ae_scores >= WARN_THRESHOLD)[0]]
    spo2_warn_val = spo2_cross_warn[-1] if len(spo2_cross_warn) > 0 else None

    spo2_cross_crit = spo2_sweep[np.where(spo2_ae_scores >= CRIT_THRESHOLD)[0]]
    spo2_crit_val = spo2_cross_crit[-1] if len(spo2_cross_crit) > 0 else None

    print(f"DenseAE v2 Thresholds: Warning={WARN_THRESHOLD:.6f}, Critical={CRIT_THRESHOLD:.6f}")
    print(f"Rule Thresholds      : HR Warn=[<55, >95], HR Crit=[<50, >100], SpO2 Warn=[<94], SpO2 Crit=[<90]\n")
    print("Crossing Thresholds Side-by-Side:")
    print(f"  - HR Low Warn  : DenseAE={hr_low_warn_val:.1f} BPM  vs  Rule=55.0 BPM")
    print(f"  - HR High Warn : DenseAE={hr_high_warn_val:.1f} BPM  vs  Rule=95.0 BPM")
    print(f"  - HR Low Crit  : DenseAE={hr_low_crit_val:.1f} BPM  vs  Rule=50.0 BPM")
    print(f"  - HR High Crit : DenseAE={hr_high_crit_val:.1f} BPM  vs  Rule=100.0 BPM")
    print(f"  - SpO2 Low Warn: DenseAE={spo2_warn_val:.1f}%     vs  Rule=94.0%")
    print(f"  - SpO2 Low Crit: DenseAE={spo2_crit_val:.1f}%     vs  Rule=90.0%")

    results_payload["1d_sweep"] = {
        "hr_low_warn_ae": hr_low_warn_val,
        "hr_high_warn_ae": hr_high_warn_val,
        "hr_low_crit_ae": hr_low_crit_val,
        "hr_high_crit_ae": hr_high_crit_val,
        "spo2_warn_ae": spo2_warn_val,
        "spo2_crit_ae": spo2_crit_val,
    }

    print("\n--- 5c. GRADED ANOMALY INJECTIONS (20 EPISODES PER LEVEL) ---")
    hr_spikes = [10.0, 20.0, 30.0, 40.0]
    spo2_drops = [95.0, 93.0, 91.0, 89.0]
    injection_records = []

    for delta_hr in hr_spikes:
        ae_recalls = []
        ae_latencies = []
        rule_raw_recalls = []
        rule_raw_latencies = []
        rule_n3_recalls = []
        rule_n3_latencies = []

        for ep in range(20):
            rng = np.random.default_rng(ep + 100)
            base_hr = rng.normal(72.0, 2.0, size=20) + delta_hr
            base_spo2 = np.clip(rng.normal(98.0, 0.4, size=20), 80.0, 100.0)

            ae_infer = EdgeAutoencoderInference(weights_path)
            ae_hits = []
            for h, s in zip(base_hr, base_spo2):
                p = ae_infer.predict_sample({"Heart_Rate": h, "SpO2": s})
                ae_hits.append(p["severity"] >= 1)
            hit = any(ae_hits)
            ae_recalls.append(1.0 if hit else 0.0)
            if hit:
                ae_latencies.append(ae_hits.index(True))

            rule_raw_hits = []
            rule_n3_hits = []
            c_w = 0
            for h, s in zip(base_hr, base_spo2):
                sev, _ = evaluate_rules(h, s, 36.5)
                rule_raw_hits.append(sev >= 1)
                c_w = (c_w + 1) if sev >= 1 else 0
                rule_n3_hits.append(c_w >= 3)

            r_raw_hit = any(rule_raw_hits)
            rule_raw_recalls.append(1.0 if r_raw_hit else 0.0)
            if r_raw_hit:
                rule_raw_latencies.append(rule_raw_hits.index(True))

            r_n3_hit = any(rule_n3_hits)
            rule_n3_recalls.append(1.0 if r_n3_hit else 0.0)
            if r_n3_hit:
                rule_n3_latencies.append(rule_n3_hits.index(True))

        injection_records.append({
            "Type": f"HR_SPIKE (+{delta_hr:.0f} BPM)",
            "Target_Vitals": f"HR ~{72+delta_hr:.0f}, SpO2 ~98",
            "DenseAE_N3_Recall": round(float(np.mean(ae_recalls)), 2),
            "DenseAE_N3_Med_Lat": int(np.median(ae_latencies)) if len(ae_latencies) > 0 else "-",
            "Rule_Raw_Recall": round(float(np.mean(rule_raw_recalls)), 2),
            "Rule_Raw_Med_Lat": int(np.median(rule_raw_latencies)) if len(rule_raw_latencies) > 0 else "-",
            "Rule_N3_Recall": round(float(np.mean(rule_n3_recalls)), 2),
            "Rule_N3_Med_Lat": int(np.median(rule_n3_latencies)) if len(rule_n3_latencies) > 0 else "-"
        })

    for target_sp in spo2_drops:
        ae_recalls = []
        ae_latencies = []
        rule_raw_recalls = []
        rule_raw_latencies = []
        rule_n3_recalls = []
        rule_n3_latencies = []

        for ep in range(20):
            rng = np.random.default_rng(ep + 200)
            base_hr = rng.normal(72.0, 2.0, size=20)
            base_spo2 = rng.normal(target_sp, 0.4, size=20)

            ae_infer = EdgeAutoencoderInference(weights_path)
            ae_hits = []
            for h, s in zip(base_hr, base_spo2):
                p = ae_infer.predict_sample({"Heart_Rate": h, "SpO2": s})
                ae_hits.append(p["severity"] >= 1)
            hit = any(ae_hits)
            ae_recalls.append(1.0 if hit else 0.0)
            if hit:
                ae_latencies.append(ae_hits.index(True))

            rule_raw_hits = []
            rule_n3_hits = []
            c_w = 0
            for h, s in zip(base_hr, base_spo2):
                sev, _ = evaluate_rules(h, s, 36.5)
                rule_raw_hits.append(sev >= 1)
                c_w = (c_w + 1) if sev >= 1 else 0
                rule_n3_hits.append(c_w >= 3)

            r_raw_hit = any(rule_raw_hits)
            rule_raw_recalls.append(1.0 if r_raw_hit else 0.0)
            if r_raw_hit:
                rule_raw_latencies.append(rule_raw_hits.index(True))

            r_n3_hit = any(rule_n3_hits)
            rule_n3_recalls.append(1.0 if r_n3_hit else 0.0)
            if r_n3_hit:
                rule_n3_latencies.append(rule_n3_hits.index(True))

        injection_records.append({
            "Type": f"SPO2_DROP (-> {target_sp:.0f}%)",
            "Target_Vitals": f"HR ~72, SpO2 ~{target_sp:.0f}",
            "DenseAE_N3_Recall": round(float(np.mean(ae_recalls)), 2),
            "DenseAE_N3_Med_Lat": int(np.median(ae_latencies)) if len(ae_latencies) > 0 else "-",
            "Rule_Raw_Recall": round(float(np.mean(rule_raw_recalls)), 2),
            "Rule_Raw_Med_Lat": int(np.median(rule_raw_latencies)) if len(rule_raw_latencies) > 0 else "-",
            "Rule_N3_Recall": round(float(np.mean(rule_n3_recalls)), 2),
            "Rule_N3_Med_Lat": int(np.median(rule_n3_latencies)) if len(rule_n3_latencies) > 0 else "-"
        })

    df_inj = pd.DataFrame(injection_records)
    print(df_inj.to_string(index=False))
    results_payload["graded_injections"] = injection_records

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "aol_verification.json")
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[✓] Full verification payload saved to: {json_path}")

if __name__ == "__main__":
    main()
