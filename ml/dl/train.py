"""
ml.dl.train: Training pipelines for Deep Learning models and classical baselines.

Protocol Enforcements:
- Features: per-row only (no temporal autocorrelation leakage).
- StandardScaler: fit strictly on TRAIN NORMAL (OUTPUT == 0) rows only.
- Early stopping: strictly on normal validation loss.
- 5 fixed seeds: [42, 43, 44, 45, 46].
"""

import os
import sys
import random
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    torch = None
    nn = None
    TensorDataset = None
    DataLoader = None

# Ensure backend rule engine is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.rules.engine import evaluate_rules
from ml.dl.models import DenseAutoencoder, DenseVAE

FEATURE_VARIANTS = {
    "v3": ["Heart_Rate", "SpO2", "Temperature"],
    "v2": ["Heart_Rate", "SpO2"]
}

SEEDS = [42, 43, 44, 45, 46]


def set_seed(seed: int):
    """Sets random seeds across all libraries for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """
    Fits StandardScaler EXCLUSIVELY on normal train rows (OUTPUT == 0).
    """
    normal_train = train_df[train_df["OUTPUT"] == 0]
    if len(normal_train) == 0:
        raise ValueError("Cannot fit scaler: no normal (OUTPUT == 0) rows found in train_df.")

    scaler = StandardScaler()
    scaler.fit(normal_train[feature_cols].values)
    return scaler


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    seed: int = 42,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5
) -> Tuple[DenseAutoencoder, List[float], List[float]]:
    """
    Trains DenseAutoencoder on normal train data with early stopping on normal val loss.
    """
    set_seed(seed)
    model = DenseAutoencoder(input_dim=input_dim, bottleneck_dim=2, dropout=0.05)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve_epochs = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / max(1, n_batches)
        train_losses.append(avg_train_loss)

        # Validation on normal validation set
        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = criterion(val_recon, val_tensor).item()
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, train_losses, val_losses


def train_vae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    seed: int = 42,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    beta: float = 0.001
) -> Tuple[DenseVAE, List[float], List[float]]:
    """
    Trains DenseVAE on normal train data with early stopping on normal val loss.
    Loss = MSE reconstruction + beta * KL divergence.
    """
    set_seed(seed)
    model = DenseVAE(input_dim=input_dim, bottleneck_dim=2, dropout=0.05)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    no_improve_epochs = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            recon_loss = nn.functional.mse_loss(recon, batch_x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + beta * kl_loss
            total_loss.backward()
            optimizer.step()

            total_train_loss += recon_loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / max(1, n_batches)
        train_losses.append(avg_train_loss)

        # Validation on normal validation set
        model.eval()
        with torch.no_grad():
            val_recon, _, _ = model(val_tensor)
            val_loss = nn.functional.mse_loss(val_recon, val_tensor).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, train_losses, val_losses


# ==============================================================================
# BASELINES
# ==============================================================================

class RuleBaselineWrapper:
    """Clinical / Heuristic Rule Engine baseline."""
    def __init__(self, variant: str = "v3"):
        self.variant = variant
        self.name = f"Rule Engine ({variant})"

    def score(self, data: Any) -> np.ndarray:
        scores = []
        if isinstance(data, pd.DataFrame):
            for _, r in data.iterrows():
                hr = r["Heart_Rate"]
                spo2 = r["SpO2"]
                temp = r["Temperature"] if self.variant == "v3" and "Temperature" in r else 36.5
                sev, _ = evaluate_rules(hr, spo2, temp)
                scores.append(0.0 if sev == 0 else (0.5 if sev == 1 else 1.0))
        else:
            arr = np.asarray(data)
            for r in arr:
                hr = float(r[0])
                spo2 = float(r[1])
                temp = float(r[2]) if self.variant == "v3" and len(r) > 2 else 36.5
                sev, _ = evaluate_rules(hr, spo2, temp)
                scores.append(0.0 if sev == 0 else (0.5 if sev == 1 else 1.0))
        return np.array(scores, dtype=np.float64)


class ShallowBaselineWrapper:
    """Wrapper for Isolation Forest, LOF, and One-Class SVM."""
    def __init__(self, name: str, model: Any, scaler: StandardScaler, feature_cols: List[str]):
        self.name = name
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols

    def score(self, data: Any) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            X = self.scaler.transform(data[self.feature_cols].values)
        else:
            X = np.asarray(data)
        raw = self.model.decision_function(X)
        # Invert so higher score = more anomalous
        return -raw.astype(np.float64)


def train_baselines(
    train_normal_df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str],
    variant: str = "v3",
    seed: int = 42,
    subsample_size: int = 25000
) -> Dict[str, Any]:
    """
    Fits shallow baselines strictly on normal train rows.
    Uses stratified/random subsample for kernel methods (LOF, OC-SVM) for efficiency.
    """
    X_train = scaler.transform(train_normal_df[feature_cols].values)

    if len(X_train) > subsample_size:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(len(X_train), size=subsample_size, replace=False)
        X_sub = X_train[sub_idx]
    else:
        X_sub = X_train

    # 1. Rule Engine
    rule_model = RuleBaselineWrapper(variant=variant)

    # 2. Isolation Forest
    iforest = IsolationForest(n_estimators=100, random_state=seed, contamination="auto")
    iforest.fit(X_train)
    if_wrapper = ShallowBaselineWrapper(f"Isolation Forest ({variant})", iforest, scaler, feature_cols)

    # 3. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto")
    lof.fit(X_sub)
    lof_wrapper = ShallowBaselineWrapper(f"Local Outlier Factor ({variant})", lof, scaler, feature_cols)

    # 4. One-Class SVM
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_sub)
    ocsvm_wrapper = ShallowBaselineWrapper(f"One-Class SVM ({variant})", ocsvm, scaler, feature_cols)

    return {
        "Rule Engine": rule_model,
        "Isolation Forest": if_wrapper,
        "Local Outlier Factor": lof_wrapper,
        "One-Class SVM": ocsvm_wrapper
    }
