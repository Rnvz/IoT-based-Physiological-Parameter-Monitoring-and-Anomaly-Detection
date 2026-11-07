import os
import sys
import time
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve,
    precision_recall_curve, accuracy_score, balanced_accuracy_score
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dl.src.models import FlexibleDenseAutoencoder
from ml.src.feature_engineering import FEATURE_COLS

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan Device: {device}")

    # 1. Memuat dataset hasil refaktorisasi feature engineering
    train_path = os.path.join(PROJECT_ROOT, "ml/data/features_train.csv")
    val_path = os.path.join(PROJECT_ROOT, "ml/data/features_val.csv")
    test_path = os.path.join(PROJECT_ROOT, "ml/data/features_test.csv")

    print(f"Memuat data: {train_path}, {val_path}, {test_path}...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_norm_mask = (df_train["OUTPUT"] == 0).values
    val_norm_mask = (df_val["OUTPUT"] == 0).values
    test_norm_mask = (df_test["OUTPUT"] == 0).values

    X_train_norm = df_train.loc[train_norm_mask, FEATURE_COLS].values
    X_val_norm = df_val.loc[val_norm_mask, FEATURE_COLS].values
    X_val_all = df_val[FEATURE_COLS].values
    X_test_all = df_test[FEATURE_COLS].values

    y_val = df_val["OUTPUT"].values
    y_test = df_test["OUTPUT"].values

    print("\n" + "="*70)
    print("DISTRIBUSI DATASET SETELAH PEMBERSIHAN & REFAKTORISASI")
    print("="*70)
    print(f"Train Normal: {len(X_train_norm):,} baris")
    print(f"Val Total   : {len(y_val):,} (Normal: {val_norm_mask.sum():,}, Anomaly: {(~val_norm_mask).sum():,})")
    print(f"Test Total  : {len(y_test):,} (Normal: {test_norm_mask.sum():,}, Anomaly: {(~test_norm_mask).sum():,})")
    print(f"Sumber Data : {list(df_test['Source'].unique())}")
    print("="*70 + "\n")

    # 2. Inisialisasi Model FlexibleDenseAutoencoder
    model = FlexibleDenseAutoencoder(input_dim=13, bottleneck_dim=4, hidden_dim=32, dropout=0.05).to(device)

    # 3. Training Loop (HANYA pada data Normal)
    tr_tensor = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    va_tensor = torch.tensor(X_val_norm, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(tr_tensor, tr_tensor), batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience = 5
    no_imp = 0
    max_epochs = 30

    print("Memulai pelatihan FlexibleDenseAutoencoder pada data Normal...")
    for epoch in range(max_epochs):
        model.train()
        t_loss = 0.0
        for bx, _ in loader:
            optimizer.zero_grad()
            recon = model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        with torch.no_grad():
            va_loss = criterion(model(va_tensor), va_tensor).item()

        avg_t_loss = t_loss / len(loader)
        print(f"  Epoch {epoch+1:2d}/{max_epochs} | Train Loss: {avg_t_loss:.6f} | Val Normal Loss: {va_loss:.6f}")

        if va_loss < best_loss - 1e-5:
            best_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping tercapai pada epoch {epoch+1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # 4. Inferensi Anomaly Score (MSE)
    t_va_all = torch.tensor(X_val_all, dtype=torch.float32).to(device)
    t_te_all = torch.tensor(X_test_all, dtype=torch.float32).to(device)

    scores_val = model.compute_anomaly_score(t_va_all).cpu().numpy()
    scores_test = model.compute_anomaly_score(t_te_all).cpu().numpy()

    roc_val = float(roc_auc_score(y_val, scores_val))
    pr_val = float(average_precision_score(y_val, scores_val))
    roc_test = float(roc_auc_score(y_test, scores_test))
    pr_test = float(average_precision_score(y_test, scores_test))

    print("\n" + "="*70)
    print("AREA UNDER CURVE (AUROC & AUPRC)")
    print("="*70)
    print(f"Validation Set -> ROC-AUC: {roc_val:.4f} | PR-AUC: {pr_val:.4f}")
    print(f"Test Set       -> ROC-AUC: {roc_test:.4f} | PR-AUC: {pr_test:.4f}")
    print("="*70 + "\n")

    # 5. Strategi Pemilihan Ambang Batas Optimal pada Validation Set
    print("="*85)
    print("OPTIMASI STRATEGI AMBANG BATAS (THRESHOLD) PADA VALIDATION SET")
    print("="*85)

    # A. Strategi 1: Youden's J Index (J = Sensitivity + Specificity - 1 = TPR - FPR)
    fpr_v, tpr_v, thrs_roc = roc_curve(y_val, scores_val)
    j_scores = tpr_v - fpr_v
    best_j_idx = int(np.argmax(j_scores))
    tau_youden = float(thrs_roc[best_j_idx])
    val_j_opt = float(j_scores[best_j_idx])
    val_tpr_youden = float(tpr_v[best_j_idx])
    val_fpr_youden = float(fpr_v[best_j_idx])

    # B. Strategi 2: Best F1-Score pada Validation Set
    p_v, r_v, thrs_pr = precision_recall_curve(y_val, scores_val)
    f1_v = 2 * p_v * r_v / (p_v + r_v + 1e-12)
    best_f1_idx = int(np.argmax(f1_v))
    tau_best_f1 = float(thrs_pr[best_f1_idx])
    val_f1_opt = float(f1_v[best_f1_idx])
    val_rec_best_f1 = float(r_v[best_f1_idx])
    val_prec_best_f1 = float(p_v[best_f1_idx])

    # C. Referensi Strategi Lama: Fixed 10% Normal Quantile
    val_norm_scores = scores_val[val_norm_mask]
    tau_quantile_10 = float(np.percentile(val_norm_scores, 90.0))

    print(f"1. Youden's J Index Optimal (Val) : tau = {tau_youden:.6f} | Val J = {val_j_opt:.4f} (TPR={val_tpr_youden*100:.2f}%, FPR={val_fpr_youden*100:.2f}%)")
    print(f"2. Best F1-Score Optimal (Val)   : tau = {tau_best_f1:.6f} | Val F1 = {val_f1_opt:.4f} (Recall={val_rec_best_f1*100:.2f}%, Prec={val_prec_best_f1*100:.2f}%)")
    print(f"3. Fixed Normal Quantile 10% (Ref): tau = {tau_quantile_10:.6f} (Sebelumnya konservatif di Test FPR < 2%)")
    print("="*85 + "\n")

    # 6. Evaluasi Test Set untuk Setiap Strategi Threshold
    strategies = {
        "youden_j": {
            "name": "Youden's J Index (Optimal)",
            "tau": tau_youden,
            "description": "Memaksimalkan (Sensitivity + Specificity - 1) pada Val"
        },
        "best_f1": {
            "name": "Best F1-Score (Optimal)",
            "tau": tau_best_f1,
            "description": "Memaksimalkan Harmonic Mean Precision-Recall pada Val"
        },
        "fixed_quantile_10": {
            "name": "Fixed Quantile 10% (Reference)",
            "tau": tau_quantile_10,
            "description": "Persentil 90% Validation Normal murni"
        }
    }

    eval_test_results = {}
    print("="*110)
    print(f"{'Strategi Threshold':<25} | {'Tau':<12} | {'Act FPR':<9} | {'Recall':<8} | {'Precision':<9} | {'F1-Score':<8} | {'Bal Acc':<8} | {'Accuracy':<8}")
    print("="*110)

    for strat_key, strat_info in strategies.items():
        thr = strat_info["tau"]
        pred_test = (scores_test >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
        rec = float(recall_score(y_test, pred_test))
        prec = float(precision_score(y_test, pred_test, zero_division=0))
        f1 = float(f1_score(y_test, pred_test, zero_division=0))
        act_fpr = float(fp / (tn + fp))
        spec = float(tn / (tn + fp))
        acc = float(accuracy_score(y_test, pred_test))
        bal_acc = float(balanced_accuracy_score(y_test, pred_test))

        eval_test_results[strat_key] = {
            "strategy_name": strat_info["name"],
            "description": strat_info["description"],
            "threshold": thr,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "actual_fpr": act_fpr,
            "recall": rec,
            "specificity": spec,
            "precision": prec,
            "f1": f1,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "breakdown": {}
        }

        # Breakdown per sumber data
        for src in sorted(df_test['Source'].unique()):
            s_mask = (df_test['Source'] == src).values
            s_y = y_test[s_mask]
            s_pred = pred_test[s_mask]
            s_norm_mask = (s_y == 0)

            s_fpr = float((s_pred[s_norm_mask]).mean()) if s_norm_mask.sum() > 0 else 0.0
            s_rec = float(recall_score(s_y, s_pred)) if (s_y == 1).sum() > 0 else 0.0
            s_prec = float(precision_score(s_y, s_pred, zero_division=0))
            s_f1 = float(f1_score(s_y, s_pred, zero_division=0))

            eval_test_results[strat_key]["breakdown"][src] = {
                "total": int(len(s_y)),
                "normal": int(s_norm_mask.sum()),
                "anomaly": int((s_y == 1).sum()),
                "fpr": s_fpr,
                "recall": s_rec,
                "precision": s_prec,
                "f1": s_f1
            }

        print(f"{strat_info['name']:<25} | {thr:<12.6f} | {act_fpr*100:6.2f}%   | {rec*100:6.2f}%  | {prec*100:6.2f}%   | {f1:<8.4f} | {bal_acc*100:6.2f}%  | {acc*100:6.2f}%")

    print("="*110 + "\n")

    # 7. Tampilkan Breakdown Per-Sumber untuk Strategi Optimal
    for strat_key in ["youden_j", "best_f1"]:
        strat = eval_test_results[strat_key]
        print(f"\nBreakdown Per-Sumber: {strat['strategy_name']} (tau = {strat['threshold']:.6f}):")
        print("-" * 85)
        print(f"{'Sumber Dataset':<18} | {'Total':<7} | {'Normal':<7} | {'Anom':<7} | {'FPR':<8} | {'Recall':<8} | {'Precision':<9} | {'F1-Score':<8}")
        print("-" * 85)
        for src, bd in strat["breakdown"].items():
            print(f"{src:<18} | {bd['total']:<7} | {bd['normal']:<7} | {bd['anomaly']:<7} | {bd['fpr']*100:6.2f}% | {bd['recall']*100:6.2f}% | {bd['precision']*100:6.2f}%   | {bd['f1']:<8.4f}")
        print("-" * 85)

    # 8. Simpan Model Weights dan Summary
    save_path = os.path.join(PROJECT_ROOT, "dl/artifacts/dl_v1/dense_autoencoder_kaggle.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights tersimpan di: {save_path}")

    summary_path = os.path.join(PROJECT_ROOT, "dl/artifacts/dl_v1/kaggle_eval_summary.json")
    summary_data = {
        "roc_auc_val": roc_val,
        "pr_auc_val": pr_val,
        "roc_auc_test": roc_test,
        "pr_auc_test": pr_test,
        "optimal_thresholds": {
            "youden_j": tau_youden,
            "best_f1": tau_best_f1,
            "fixed_quantile_10": tau_quantile_10
        },
        "test_evaluations": eval_test_results
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary evaluasi threshold optimal tersimpan di: {summary_path}")

if __name__ == "__main__":
    main()
