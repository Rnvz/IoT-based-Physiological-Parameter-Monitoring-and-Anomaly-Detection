"""
ml_eval.baselines: Baseline models and standardized validation threshold selection.

Models:
1. Rule Engine    : Clinical deterministic logic (reuses backend/app/rules/engine.py).
2. Isolation Forest: Partition-based tree isolation.
3. Local Outlier Factor (LOF): Density-based local outlier detection (novelty=True).
4. One-Class SVM  : Support vector boundary estimation with RBF kernel.

TRAINING PROTOCOL RULES:
- Models and scaler are fitted EXCLUSIVELY on NORMAL train sessions (scenario REST or TYPING).
- StandardScaler is fitted on train only, never validation or test.
- Operating thresholds for all models are calibrated EXCLUSIVELY on NORMAL VALIDATION sessions.
- Test sessions are never seen during training or threshold calibration.
"""

import os
import sys
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Ensure backend app is importable for rule engine reuse
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app.rules.engine import evaluate_rules
except ImportError:
    # Fallback to local import if backend is already on path
    from backend.app.rules.engine import evaluate_rules

from ml_eval.features import CORE_FEATURE_NAMES, EXTENDED_FEATURE_NAMES


NORMAL_SCENARIOS = {"REST", "TYPING"}


class BaselineModelWrapper:
    """Wrapper encapsulating a baseline model, scaler, operating threshold, and feature set."""

    def __init__(self, name: str, model: Any, scaler: Optional[StandardScaler], feature_names: List[str]):
        self.name = name
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold: float = 0.5
        self.target_fpr: float = 0.05

    def score(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Computes continuous anomaly score for each row.
        Higher score = more anomalous.
        """
        if self.name == "Rule Engine":
            scores = []
            for _, r in df_features.iterrows():
                sev, _ = evaluate_rules(r["hr"], r["spo2"], r["temp"])
                # 0: normal (0.0), 1: warning (0.5), 2: critical (1.0)
                scores.append(0.0 if sev == 0 else (0.5 if sev == 1 else 1.0))
            return np.array(scores, dtype=np.float64)

        X = df_features[self.feature_names].values
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Scikit-learn convention: decision_function returns negative values for anomalies.
        # We invert it: anomaly_score = -decision_function(X) so higher = more anomalous.
        raw = self.model.decision_function(X)
        return -raw.astype(np.float64)

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Predicts binary anomaly status (0 = Normal, 1 = Anomaly)
        using the frozen operating threshold.
        """
        scores = self.score(df_features)
        return (scores >= self.threshold).astype(int)


def train_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_set: str = "core",
    target_fpr: float = 0.05,
    seed: int = 42
) -> Dict[str, BaselineModelWrapper]:
    """
    Fits baseline models on NORMAL train data and selects operating thresholds on VAL data.

    Arguments:
        train_df   : Extracted feature rows for train sessions.
        val_df     : Extracted feature rows for validation sessions.
        feature_set: 'core' (12 features) or 'extended' (24 features).
        target_fpr : Target False Positive Rate on normal validation data (default: 0.05).
        seed       : Random seed for reproducibility.

    Returns:
        Dict mapping model name to BaselineModelWrapper.
    """
    feature_names = CORE_FEATURE_NAMES if feature_set.lower() == "core" else EXTENDED_FEATURE_NAMES

    # 1. Filter Train data to NORMAL scenarios only (REST or TYPING)
    train_normal_mask = train_df["scenario"].isin(NORMAL_SCENARIOS)
    if train_normal_mask.sum() == 0:
        # Fallback if unlabelled or different scenario tags
        train_normal = train_df
    else:
        train_normal = train_df[train_normal_mask]

    X_train = train_normal[feature_names].values

    # 2. Fit StandardScaler on TRAIN ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Instantiate and fit ML models on Train
    models = {
        "Rule Engine": BaselineModelWrapper("Rule Engine", None, None, feature_names),
        "Isolation Forest": BaselineModelWrapper(
            "Isolation Forest",
            IsolationForest(n_estimators=100, random_state=seed, contamination="auto"),
            scaler,
            feature_names
        ),
        "Local Outlier Factor": BaselineModelWrapper(
            "Local Outlier Factor",
            LocalOutlierFactor(
                n_neighbors=min(20, max(1, len(X_train_scaled) - 1)),
                novelty=True,
                contamination="auto"
            ),
            scaler,
            feature_names
        ),
        "One-Class SVM": BaselineModelWrapper(
            "One-Class SVM",
            OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
            scaler,
            feature_names
        )
    }

    # Fit ML models
    models["Isolation Forest"].model.fit(X_train_scaled)
    models["Local Outlier Factor"].model.fit(X_train_scaled)
    models["One-Class SVM"].model.fit(X_train_scaled)

    # 4. Calibrate operating threshold EXCLUSIVELY on Validation sessions
    val_normal_mask = val_df["scenario"].isin(NORMAL_SCENARIOS)
    if val_normal_mask.sum() == 0:
        val_normal = val_df
    else:
        val_normal = val_df[val_normal_mask]

    for name, wrapper in models.items():
        wrapper.target_fpr = target_fpr
        if name == "Rule Engine":
            wrapper.threshold = 0.5  # Fixed deterministic boundary
        else:
            val_scores = wrapper.score(val_normal)
            # Pick threshold at (1 - target_fpr) percentile of normal validation scores
            pct = 100.0 * (1.0 - target_fpr)
            wrapper.threshold = float(np.percentile(val_scores, pct))

    return models
