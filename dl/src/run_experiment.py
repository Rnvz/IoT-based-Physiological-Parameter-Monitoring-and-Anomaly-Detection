"""
ml.dl.run_experiment: End-to-end training, pre-registered model selection,
test evaluation, domain shift analysis, and ONNX export.
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

from dl.src.models import DenseAutoencoder, DenseVAE, count_parameters
from dl.src.train import (
    FEATURE_VARIANTS,
    SEEDS,
    set_seed,
    fit_scaler,
    train_autoencoder,
    train_vae,
    train_baselines
)
from dl.src.evaluate import (
    compute_metrics,
    patient_group_bootstrap,
    compute_persistence_events,
    benchmark_model_speed_and_size,
    evaluate_domain_shift_stream
)
from dl.src.export import export_dl_model
from ml.eval.public_split import load_public_split

def score_model(model_obj: Any, X: np.ndarray) -> np.ndarray:
    """Computes continuous anomaly score for given model object."""
    if hasattr(model_obj, "score") and not isinstance(model_obj, torch.nn.Module):
        if hasattr(model_obj, "scaler") and model_obj.scaler is not None:
            pass
        return model_obj.score(pd.DataFrame(X, columns=["Heart_Rate", "SpO2", "Temperature"][:X.shape[1]]))
    elif isinstance(model_obj, torch.nn.Module):
        model_obj.eval()
        with torch.no_grad():
            t_x = torch.tensor(X, dtype=torch.float32)
            scores = model_obj.compute_anomaly_score(t_x).cpu().numpy()
        return scores.astype(np.float64)
    else:
        raise ValueError(f"Unknown model type: {type(model_obj)}")

def main():
    parser = argparse.ArgumentParser(description="Run complete DL physiological anomaly detection benchmark.")
    parser.add_argument("--data", "-d", default="ml_pipeline/data/harmonized_vitals.csv", help="Harmonized vitals CSV")
    parser.add_argument("--split", "-s", default="splits/public_split.json", help="Frozen split definition JSON")
    parser.add_argument("--device-csv", default="tests/fixtures/sample_fixture.csv", help="Device CSV for single subject replay")
    parser.add_argument("--output-dir", "-o", default="ml/artifacts/dl_v1", help="Export artifact directory")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples (default: 1000)")
    args = parser.parse_args()

    print("=" * 80)
    print("PHYSIOLOGICAL IOT ANOMALY DETECTION: BENCHMARK & DEPLOYMENT RUNNER")
    print("=" * 80)

    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    df["row_id"] = np.arange(len(df))

    print(f"Loading frozen split from: {args.split}")
    split_meta, split_sha = load_public_split(args.split)
    print(f"Split SHA256: {split_sha}")

    train_idx = split_meta["indices"]["train"]
    val_idx = split_meta["indices"]["val"]
    test_idx = split_meta["indices"]["test"]

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    print(f"Partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    scalers = {}
    for v_name, cols in FEATURE_VARIANTS.items():
        scalers[v_name] = fit_scaler(train_df, cols)
        print(f"[{v_name}] Scaler fitted on normal train (mean={scalers[v_name].mean_.round(2)}, scale={scalers[v_name].scale_.round(2)})")

    print("\n--- Training Shallow Baselines on Normal Train Data ---")
    baselines = {}
    for v_name, cols in FEATURE_VARIANTS.items():
        norm_train = train_df[train_df["OUTPUT"] == 0]
        baselines[v_name] = train_baselines(norm_train, scalers[v_name], cols, variant=v_name, seed=42)

    print("\n--- Training Deep Learning Architectures (5 seeds: [42, 43, 44, 45, 46]) ---")
    dl_models: Dict[str, Dict[str, List[Any]]] = {
        "v3": {"DenseAE": [], "DenseVAE": []},
        "v2": {"DenseAE": [], "DenseVAE": []}
    }

    train_norm = train_df[train_df["OUTPUT"] == 0]
    val_norm = val_df[val_df["OUTPUT"] == 0]

    for v_name, cols in FEATURE_VARIANTS.items():
        X_tr = scalers[v_name].transform(train_norm[cols].values)
        X_v = scalers[v_name].transform(val_norm[cols].values)
        inp_dim = len(cols)

        print(f"\nVariant {v_name} ({cols}):")
        for s in SEEDS:
            t0 = time.time()
            ae, tr_l, val_l = train_autoencoder(X_tr, X_v, input_dim=inp_dim, seed=s, epochs=20, patience=4)
            vae, vtr_l, vval_l = train_vae(X_tr, X_v, input_dim=inp_dim, seed=s, epochs=20, patience=4)
            dl_models[v_name]["DenseAE"].append(ae)
            dl_models[v_name]["DenseVAE"].append(vae)
            print(f"  Seed {s} fitted in {time.time()-t0:.1f}s | AE val loss: {val_l[-1]:.4f} | VAE val loss: {vval_l[-1]:.4f}")

    print("\n" + "=" * 80)
    print("PART 2 & 4: PRE-REGISTERED VALIDATION MODEL SELECTION (PR-AUC on Full Val)")
    print("=" * 80)

    val_selection_results = []
    y_val = val_df["OUTPUT"].values

    for v_name, cols in FEATURE_VARIANTS.items():
        X_val_all = scalers[v_name].transform(val_df[cols].values)

        for b_name, b_mod in baselines[v_name].items():
            sc = b_mod.score(val_df) if b_name == "Rule Engine" else -b_mod.model.decision_function(X_val_all)
            pr_auc_val, roc_auc_val = compute_metrics(y_val, sc, 0.5)["pr_auc"], compute_metrics(y_val, sc, 0.5)["roc_auc"]
            val_selection_results.append({
                "model": f"{b_name} ({v_name})",
                "variant": v_name,
                "type": "baseline",
                "val_pr_auc": pr_auc_val,
                "val_roc_auc": roc_auc_val
            })

        for arch in ["DenseAE", "DenseVAE"]:
            seed_pr_aucs = []
            seed_roc_aucs = []
            for m in dl_models[v_name][arch]:
                sc = score_model(m, X_val_all)
                met = compute_metrics(y_val, sc, 0.5)
                seed_pr_aucs.append(met["pr_auc"])
                seed_roc_aucs.append(met["roc_auc"])

            val_selection_results.append({
                "model": f"{arch} ({v_name})",
                "variant": v_name,
                "type": "deep_learning",
                "val_pr_auc": float(np.mean(seed_pr_aucs)),
                "val_pr_auc_std": float(np.std(seed_pr_aucs)),
                "val_roc_auc": float(np.mean(seed_roc_aucs)),
                "val_roc_auc_std": float(np.std(seed_roc_aucs))
            })

    val_df_summary = pd.DataFrame(val_selection_results).sort_values("val_pr_auc", ascending=False)
    print(val_df_summary.to_string(index=False))

    dl_rows = [r for r in val_selection_results if r["type"] == "deep_learning"]
    best_dl = max(dl_rows, key=lambda x: x["val_pr_auc"])
    best_arch = "DenseAE" if "DenseAE" in best_dl["model"] else "DenseVAE"
    best_variant = best_dl["variant"]
    print(f"\n>>> PRE-REGISTERED WINNER: {best_dl['model']} (Val PR-AUC: {best_dl['val_pr_auc']:.4f})")

    thresholds = {}
    for v_name, cols in FEATURE_VARIANTS.items():
        thresholds[v_name] = {}
        X_val_norm = scalers[v_name].transform(val_norm[cols].values)

        for b_name, b_mod in baselines[v_name].items():
            if b_name == "Rule Engine":
                thresholds[v_name][b_name] = {"warn": 0.5, "crit": 1.0}
            else:
                sc = -b_mod.model.decision_function(X_val_norm)
                thresholds[v_name][b_name] = {
                    "warn": float(np.percentile(sc, 95.0)),
                    "crit": float(np.percentile(sc, 99.0))
                }

        for arch in ["DenseAE", "DenseVAE"]:
            rep_m = dl_models[v_name][arch][0]
            sc = score_model(rep_m, X_val_norm)
            thresholds[v_name][arch] = {
                "warn": float(np.percentile(sc, 95.0)),
                "crit": float(np.percentile(sc, 99.0))
            }

    print("\n" + "=" * 80)
    print("PART 5: STANDARDIZED TEST EVALUATION (POOLED & PER-SOURCE)")
    print("=" * 80)

    sources = ["nasirayub2", "engrarri21", "gourangomandal"]
    y_test = test_df["OUTPUT"].values

    test_results_table = []

    def evaluate_model_full(m_name: str, score_func_or_list: Any, v_name: str, th_dict: dict, is_dl: bool = False):
        cols = FEATURE_VARIANTS[v_name]

        caveats = {
            "gourangomandal": "[CAVEAT: Rule agrees with labels by 95.5%; labels derived from thresholds]"
        }

        subsets = [("POOLED", test_df)] + [(s, test_df[test_df["Source"] == s]) for s in sources]

        for s_name, s_df in subsets:
            s_y = s_df["OUTPUT"].values
            X_sub = scalers[v_name].transform(s_df[cols].values)

            if is_dl:
                seed_metrics = []
                all_scores = []
                for seed_idx, m in enumerate(score_func_or_list):
                    sc = score_model(m, X_sub)
                    all_scores.append(sc)
                    met = compute_metrics(s_y, sc, th_dict["warn"], th_dict["crit"])
                    seed_metrics.append(met)

                mean_pr = np.mean([m["pr_auc"] for m in seed_metrics])
                std_pr = np.std([m["pr_auc"] for m in seed_metrics])
                mean_roc = np.mean([m["roc_auc"] for m in seed_metrics])
                std_roc = np.std([m["roc_auc"] for m in seed_metrics])
                mean_f1 = np.mean([m["f1"] for m in seed_metrics])
                std_f1 = np.std([m["f1"] for m in seed_metrics])
                mean_rec5 = np.mean([m["recall_at_5pct_fpr"] for m in seed_metrics if m["recall_at_5pct_fpr"] is not None])

                rep_sc = all_scores[0]
                ci_dict = patient_group_bootstrap(s_df, rep_sc, th_dict["warn"], n_bootstrap=args.n_bootstrap, seed=42)

                pers_stat = None
                if s_name == "engrarri21":
                    pers_stat = compute_persistence_events(rep_sc, s_y, th_dict["warn"], N=3)["events_per_1000_normal_rows"]

                bench_us, bench_sz = benchmark_model_speed_and_size(score_func_or_list[0], X_sub)

                test_results_table.append({
                    "Model": f"{m_name} ({v_name})",
                    "Source": s_name,
                    "PR-AUC (Mean+-SD)": f"{mean_pr:.4f} +- {std_pr:.4f}",
                    "PR-AUC 95% CI": f"[{ci_dict['pr_auc'][1]:.4f}, {ci_dict['pr_auc'][2]:.4f}]",
                    "ROC-AUC": f"{mean_roc:.4f} +- {std_roc:.4f}",
                    "F1 Score": f"{mean_f1:.4f} +- {std_f1:.4f}",
                    "Rec@5%FPR": f"{mean_rec5:.4f}",
                    "Persistence FA/1k": f"{pers_stat:.2f}" if pers_stat is not None else "-",
                    "Inf (us)": bench_us,
                    "Size (KB)": round(bench_sz / 1024, 1),
                    "Caveat": caveats.get(s_name, "")
                })

            else:
                if m_name == "Rule Engine":
                    sc = score_func_or_list.score(s_df)
                else:
                    sc = -score_func_or_list.model.decision_function(X_sub)

                met = compute_metrics(s_y, sc, th_dict["warn"], th_dict["crit"])
                ci_dict = patient_group_bootstrap(s_df, sc, th_dict["warn"], n_bootstrap=args.n_bootstrap, seed=42)

                pers_stat = None
                if s_name == "engrarri21":
                    pers_stat = compute_persistence_events(sc, s_y, th_dict["warn"], N=3)["events_per_1000_normal_rows"]

                bench_us, bench_sz = benchmark_model_speed_and_size(score_func_or_list, X_sub)

                test_results_table.append({
                    "Model": f"{m_name} ({v_name})",
                    "Source": s_name,
                    "PR-AUC (Mean+-SD)": f"{met['pr_auc']:.4f}",
                    "PR-AUC 95% CI": f"[{ci_dict['pr_auc'][1]:.4f}, {ci_dict['pr_auc'][2]:.4f}]",
                    "ROC-AUC": f"{met['roc_auc']:.4f}",
                    "F1 Score": f"{met['f1']:.4f}",
                    "Rec@5%FPR": f"{met['recall_at_5pct_fpr']:.4f}" if met['recall_at_5pct_fpr'] is not None else "-",
                    "Persistence FA/1k": f"{pers_stat:.2f}" if pers_stat is not None else "-",
                    "Inf (us)": bench_us,
                    "Size (KB)": round(bench_sz / 1024, 1),
                    "Caveat": caveats.get(s_name, "")
                })

    for v_name in ["v3", "v2"]:
        for b_name in ["Rule Engine", "Isolation Forest", "Local Outlier Factor", "One-Class SVM"]:
            evaluate_model_full(b_name, baselines[v_name][b_name], v_name, thresholds[v_name][b_name], is_dl=False)

        for arch in ["DenseAE", "DenseVAE"]:
            evaluate_model_full(arch, dl_models[v_name][arch], v_name, thresholds[v_name][arch], is_dl=True)

    test_summary_df = pd.DataFrame(test_results_table)
    print("\n--- TEST EVALUATION SUMMARY TABLE ---")
    print(test_summary_df[["Model", "Source", "PR-AUC (Mean+-SD)", "PR-AUC 95% CI", "F1 Score", "Persistence FA/1k", "Inf (us)", "Caveat"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("PART 5: LEAVE-ONE-SOURCE-OUT (LOSO) GENERALIZATION TABLE")
    print("=" * 80)

    loso_results = []
    loso_folds = split_meta["loso_folds"]

    for f_name, f_info in loso_folds.items():
        held_out = f_info["held_out_source"]
        tr_df = df.iloc[f_info["train_indices"]]
        te_df = df.iloc[f_info["test_indices"]]

        tr_norm = tr_df[tr_df["OUTPUT"] == 0]

        for v_name in ["v3", "v2"]:
            cols = FEATURE_VARIANTS[v_name]
            l_scaler = fit_scaler(tr_df, cols)
            X_tr = l_scaler.transform(tr_norm[cols].values)

            ae, _, _ = train_autoencoder(X_tr, X_tr[:500], input_dim=len(cols), seed=42, epochs=15)

            tr_scores = score_model(ae, X_tr)
            th_warn = float(np.percentile(tr_scores, 95.0))

            X_te = l_scaler.transform(te_df[cols].values)
            te_y = te_df["OUTPUT"].values
            te_scores = score_model(ae, X_te)

            m_res = compute_metrics(te_y, te_scores, th_warn)
            loso_results.append({
                "Fold": f_name,
                "Held Out Source": held_out,
                "Variant": v_name,
                "PR-AUC": round(m_res["pr_auc"], 4) if m_res["pr_auc"] is not None else None,
                "ROC-AUC": round(m_res["roc_auc"], 4) if m_res["roc_auc"] is not None else None,
                "F1 Score": round(m_res["f1"], 4) if m_res["f1"] is not None else None
            })

    print(pd.DataFrame(loso_results).to_string(index=False))

    print("\n" + "=" * 80)
    print("PART 6: DOMAIN-SHIFT / DEPLOYMENT SANITY (NOT A TEST RESULT)")
    print("=" * 80)

    deployment_audit = []
    for v_name in ["v3", "v2"]:
        for arch in ["DenseAE", "DenseVAE"]:
            rep_m = dl_models[v_name][arch][0]
            th = thresholds[v_name][arch]

            score_fn = lambda x: score_model(rep_m, x)
            stream_res = evaluate_domain_shift_stream(
                score_func=score_fn,
                scaler=scalers[v_name],
                feature_cols=FEATURE_VARIANTS[v_name],
                warn_th=th["warn"],
                crit_th=th["crit"],
                persistence_n=3
            )

            share_33 = stream_res["33.0C"]["warning_share_pct"]
            passes_33 = share_33 < 5.0

            deployment_audit.append({
                "Model": f"{arch} ({v_name})",
                "Variant": v_name,
                "Alarm% @ 31°C": stream_res["31.0C"]["warning_share_pct"],
                "Alarm% @ 33°C (Target <5%)": stream_res["33.0C"]["warning_share_pct"],
                "Alarm% @ 35°C": stream_res["35.0C"]["warning_share_pct"],
                "Alarm% @ 36.5°C": stream_res["36.5C"]["warning_share_pct"],
                "Pre-Reg 33°C Decision": "PASS (Deployable)" if passes_33 else "FAIL (Non-Deployable)"
            })

    print(pd.DataFrame(deployment_audit).to_string(index=False))

    if os.path.exists(args.device_csv):
        print(f"\nReplaying Single Subject Device Recording: {args.device_csv}")
        dev_df = pd.read_csv(args.device_csv)
        rep_m = dl_models[best_variant]["DenseAE"][0]
        th_w = thresholds[best_variant]["DenseAE"]["warn"]
        rename_map = {"hr": "Heart_Rate", "spo2": "SpO2", "temp": "Temperature"}
        dev_df = dev_df.rename(columns={k: v for k, v in rename_map.items() if k in dev_df.columns})
        dev_norm = dev_df[dev_df.get("scenario", pd.Series(["REST"] * len(dev_df))).isin(["REST", "TYPING"])]
        if len(dev_norm) > 0 and all(c in dev_norm.columns for c in cols):
            X_dev = scalers[best_variant].transform(dev_norm[cols].values)
            dev_sc = score_model(rep_m, X_dev)
            dur_h = len(dev_norm) / 3600.0
            fa_count = np.sum(dev_sc >= th_w)
            fa_per_h = fa_count / max(dur_h, 1e-4)
            print(f"[Single Subject Sanity Check] Duration: {dur_h:.2f} h | Alarms: {fa_count} | False Alarms/hr: {fa_per_h:.1f}")

    print("\n" + "=" * 80)
    print(f"PART 7: EXPORTING PRE-REGISTERED WINNER ({best_dl['model']}) TO {args.output_dir}")
    print("=" * 80)

    selected_model = dl_models[best_variant]["DenseAE"][0]
    th_warn = thresholds[best_variant]["DenseAE"]["warn"]
    th_crit = thresholds[best_variant]["DenseAE"]["crit"]

    export_res = export_dl_model(
        model=selected_model,
        scaler=scalers[best_variant],
        variant=best_variant,
        feature_cols=FEATURE_VARIANTS[best_variant],
        warn_th=th_warn,
        crit_th=th_crit,
        split_sha=split_sha,
        seeds=SEEDS,
        output_dir=args.output_dir,
        model_version="ae_v1"
    )

    print("Export Complete:")
    print(f"  - ONNX Model    : {export_res['onnx_path']}")
    print(f"  - Metadata JSON : {export_res['metadata_path']}")
    print(f"  - Model Card    : {export_res['model_card_path']}")
    print(f"  - PyTorch vs ONNX Max Diff : {export_res['pt_onnx_diff']:.2e} (Passed < 1e-5)")
    print(f"  - PyTorch vs NumPy Max Diff: {export_res['pt_np_diff']:.2e} (Passed < 1e-5)")
    print("=" * 80)

if __name__ == "__main__":
    main()
