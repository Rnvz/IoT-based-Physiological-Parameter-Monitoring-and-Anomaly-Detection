"""
ml_eval.run: End-to-end evaluation runner.

Orchestrates:
1. Loading and session segmentation (ml_eval.sessions)
2. Frozen session-wise partitioning into Train/Val/Test (ml_eval.split)
3. Causal feature extraction without cross-session leakage (ml_eval.features)
4. Baseline model fitting and threshold calibration (ml_eval.baselines)
5. Labelled synthetic anomaly injection on test copies (ml_eval.inject)
6. Standardized evaluation metrics computation (ml_eval.metrics)
7. Writing results/<split_sha>/report.json and report.md

CRITICAL PROTOCOL REQUIREMENT:
Every output MUST start with the banner "SYNTHETIC FIXTURE DATA - NOT A RESULT"
when the input is a fixture (detected via CLI flag, file path, or dataset metadata).
The report must list the split hash and all core library versions.
"""

import os
import sys
import json
import argparse
import platform
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import sklearn
import joblib

from ml.eval.sessions import load_and_segment_sessions, summarize_sessions
from ml.eval.split import freeze_split, load_frozen_split
from ml.eval.features import extract_features, CORE_FEATURE_NAMES, EXTENDED_FEATURE_NAMES
from ml.eval.baselines import train_baselines
from ml.eval.inject import inject_synthetic_anomalies
from ml.eval.metrics import evaluate_model

BANNER_TEXT = "SYNTHETIC FIXTURE DATA - NOT A RESULT"
BANNER_BLOCK = f"""================================================================================
{BANNER_TEXT}
================================================================================"""

def get_library_versions() -> Dict[str, str]:
    """Captures exact runtime library versions for protocol reproducibility."""
    versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "os": platform.platform()
    }
    return versions

def detect_is_fixture(source_path: str, df: Optional[pd.DataFrame] = None, force_fixture: bool = False) -> bool:
    """
    Detects if the dataset is synthetic fixture data.
    Checks:
    1. Explicit force_fixture flag (--fixture)
    2. File path contains 'fixture' or 'synthetic'
    3. DataFrame contains metadata or columns indicating fixture
    """
    if force_fixture:
        return True

    lower_path = source_path.lower()
    if "fixture" in lower_path or "synthetic" in lower_path:
        return True

    if df is not None:
        if "is_fixture" in df.columns:
            return True
        if "scenario" in df.columns and any(df["scenario"].astype(str).str.upper().str.contains("FIXTURE")):
            return True

    if os.path.exists(source_path):
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if "fixture" in first_line.lower() or "synthetic" in first_line.lower():
                    return True
        except Exception:
            pass

    return False

def generate_markdown_report(report_dict: Dict[str, Any]) -> str:
    """Formats evaluation results dictionary into a structured Markdown report."""
    lines = []
    is_fixture = report_dict.get("is_fixture", False)

    if is_fixture:
        lines.append(f"# {BANNER_TEXT}\n")
        lines.append("> [!WARNING]")
        lines.append(f"> **{BANNER_TEXT}**")
        lines.append("> This report was generated on synthetic fixture data solely to verify")
        lines.append("> the evaluation harness mechanics. Under the frozen protocol, NO empirical claim")
        lines.append("> of model superiority or operational performance is made.\n")
    else:
        lines.append("# Physiological Anomaly Detection Evaluation Report\n")

    lines.append("## Protocol & Environment Metadata")
    lines.append(f"- **Split SHA256**: `{report_dict['split_hash']}`")
    lines.append(f"- **Evaluated At**: `{report_dict['evaluated_at']}`")
    lines.append(f"- **Protocol Version**: `{report_dict['protocol_version']}`")
    lines.append(f"- **Feature Set**: `{report_dict['feature_set']}` ({report_dict['feature_count']} features)")
    lines.append(f"- **Target Val FPR**: `{report_dict['target_val_fpr']}`")
    lines.append(f"- **Target Recall**: `{report_dict['target_recall']}`\n")

    lines.append("### Runtime Dependencies")
    for lib, ver in report_dict["environment"].items():
        lines.append(f"- **{lib}**: `{ver}`")
    lines.append("")

    lines.append("## Partitioning Overview")
    splits = report_dict["splits"]
    lines.append(f"- **Train Sessions** ({len(splits['train_sessions'])}): {', '.join(splits['train_sessions'])}")
    lines.append(f"- **Val Sessions**   ({len(splits['val_sessions'])}): {', '.join(splits['val_sessions'])}")
    lines.append(f"- **Test Sessions**  ({len(splits['test_sessions'])}): {', '.join(splits['test_sessions'])}")
    lines.append(f"- **Total Rows Evaluated**: {report_dict['dataset_summary']['total_valid_rows']}\n")

    lines.append("## Baseline Comparison Table")
    lines.append(
        "| Model | Threshold | PR-AUC | ROC-AUC | F1-Score | Prec | Rec | Benign FA/h | FA/h @ 90% Rec | Mean Latency (s) | Inference (µs) | Size (KB) |"
    )
    lines.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )

    for m_name, res in report_dict["models"].items():
        syn = res["metrics_synthetic"]
        ben = res["metrics_benign_real"]
        lat = res["detection_latency"]
        comp = res["computational_footprint"]

        pr_auc = f"{syn['pr_auc']:.4f}" if syn['pr_auc'] is not None else "N/A"
        roc_auc = f"{syn['roc_auc']:.4f}" if syn['roc_auc'] is not None else "N/A"
        f1 = f"{syn['f1_score']:.4f}"
        prec = f"{syn['precision']:.4f}"
        rec = f"{syn['recall']:.4f}"
        th = f"{res['operating_threshold']:.4f}"
        fa_h = f"{ben['false_alarms_per_hour']:.2f}"
        fa_rec = f"{ben['false_alarms_per_hour_at_90_recall']:.2f}" if ben['false_alarms_per_hour_at_90_recall'] is not None else "N/A"
        mean_lat = f"{lat['mean_latency_seconds']:.2f}" if lat['mean_latency_seconds'] is not None else "N/A"
        inf_us = f"{comp['inference_time_us']['mean_us']:.1f}"
        sz_kb = f"{comp['serialized_size_kb']:.1f}" if comp['serialized_size_kb'] is not None else "Code only"

        lines.append(
            f"| **{m_name}** | {th} | {pr_auc} | {roc_auc} | {f1} | {prec} | {rec} | {fa_h} | {fa_rec} | {mean_lat} | {inf_us} | {sz_kb} |"
        )
    lines.append("")

    lines.append("## Notes on Comparability")
    lines.append("- **Rule Engine**: Evaluates non-clinical demo thresholds deterministically on raw HR, SpO2, and temperature.")
    lines.append("- **ML Baselines**: Trained exclusively on normal train sessions (REST / TYPING).")
    lines.append("- **Operating Threshold**: Calibrated strictly on normal validation sessions at target FPR.")
    lines.append("- **Synthetic Anomalies**: Injected into isolated copies of test sessions; originals remain pristine.")

    return "\n".join(lines) + "\n"

def run_pipeline(
    dataset_path: str,
    split_path: str = "splits/split.json",
    output_dir: str = "results",
    seed: int = 42,
    force_split: bool = False,
    is_fixture: Optional[bool] = None,
    feature_set: str = "core",
    target_fpr: float = 0.05,
    target_recall: float = 0.90,
    max_gap_seconds: float = 600.0,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Executes steps 1-6 end to end and writes results/<split_sha>/report.json and report.md.
    """
    raw_df = pd.read_csv(dataset_path)

    fixture_status = detect_is_fixture(
        source_path=dataset_path,
        df=raw_df,
        force_fixture=bool(is_fixture)
    )

    if fixture_status and not quiet:
        print(BANNER_BLOCK)

    if not quiet:
        print(f"Loading and segmenting dataset from: {dataset_path}")

    segmented_df = load_and_segment_sessions(raw_df, max_gap_seconds=max_gap_seconds)
    summary = summarize_sessions(segmented_df)

    if not quiet:
        print(f"Identified {summary['total_sessions']} sessions ({summary['total_valid_rows']} valid rows).")

    if os.path.exists(split_path) and not force_split:
        if not quiet:
            print(f"Loading existing frozen split from: {split_path}")
        split_meta, split_sha = load_frozen_split(split_path)
    else:
        if not quiet:
            print(f"Creating new frozen split at: {split_path} (seed={seed})")
        split_meta, split_sha = freeze_split(
            df=segmented_df,
            output_path=split_path,
            seed=seed,
            force=force_split
        )

    if not quiet:
        print(f"Split SHA256: {split_sha}")
        print(f"Train sessions: {split_meta['train_sessions']}")
        print(f"Val sessions  : {split_meta['val_sessions']}")
        print(f"Test sessions : {split_meta['test_sessions']}")

    train_raw = segmented_df[segmented_df["session_id"].isin(split_meta["train_sessions"])].copy()
    val_raw = segmented_df[segmented_df["session_id"].isin(split_meta["val_sessions"])].copy()
    test_raw = segmented_df[segmented_df["session_id"].isin(split_meta["test_sessions"])].copy()

    if train_raw.empty or val_raw.empty or test_raw.empty:
        raise ValueError(
            f"One or more partitions is empty! Train rows: {len(train_raw)}, "
            f"Val rows: {len(val_raw)}, Test rows: {len(test_raw)}. "
            f"Check that session IDs in {split_path} match the dataset."
        )

    if not quiet:
        print(f"Extracting '{feature_set}' rolling features (window=15, causal, session-isolated)...")

    train_feats = extract_features(train_raw, feature_set=feature_set)
    val_feats = extract_features(val_raw, feature_set=feature_set)

    if train_feats.empty or val_feats.empty:
        raise ValueError("Feature extraction yielded 0 rows for train or validation sets.")

    if not quiet:
        print(f"Fitting baselines on normal train data and calibrating thresholds on val (target_fpr={target_fpr})...")

    models = train_baselines(
        train_df=train_feats,
        val_df=val_feats,
        feature_set=feature_set,
        target_fpr=target_fpr,
        seed=seed
    )

    if not quiet:
        print("Injecting synthetic anomalies into copies of test sessions (originals unchanged)...")

    real_test_raw, injected_test_raw = inject_synthetic_anomalies(
        test_df=test_raw,
        seed=seed,
        episode_length=20
    )

    real_test_feats = extract_features(real_test_raw, feature_set=feature_set)
    injected_test_feats = extract_features(injected_test_raw, feature_set=feature_set)

    if not quiet:
        print("Scoring all baseline models using standardized metrics...")

    evaluated_models: Dict[str, Any] = {}
    for m_name, wrapper in models.items():
        res = evaluate_model(
            model_wrapper=wrapper,
            real_test_df=real_test_feats,
            injected_test_df=injected_test_feats,
            target_recall=target_recall
        )
        evaluated_models[m_name] = res

    feat_names = CORE_FEATURE_NAMES if feature_set.lower() == "core" else EXTENDED_FEATURE_NAMES
    report_dict: Dict[str, Any] = {
        "banner": BANNER_TEXT if fixture_status else None,
        "is_fixture": fixture_status,
        "protocol_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split_hash": split_sha,
        "split_path": os.path.abspath(split_path),
        "seed": seed,
        "feature_set": feature_set,
        "feature_count": len(feat_names),
        "target_val_fpr": target_fpr,
        "target_recall": target_recall,
        "environment": get_library_versions(),
        "dataset_summary": summary,
        "splits": {
            "train_sessions": split_meta["train_sessions"],
            "val_sessions": split_meta["val_sessions"],
            "test_sessions": split_meta["test_sessions"]
        },
        "models": evaluated_models
    }

    dest_dir = os.path.join(output_dir, split_sha)
    os.makedirs(dest_dir, exist_ok=True)

    json_path = os.path.join(dest_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    md_content = generate_markdown_report(report_dict)
    md_path = os.path.join(dest_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    if not quiet:
        print("\n" + "=" * 80)
        print("EVALUATION RUN COMPLETE")
        print("=" * 80)
        print(f"Results written to: {dest_dir}")
        print(f"  - JSON Report    : {json_path}")
        print(f"  - Markdown Report: {md_path}")
        print("=" * 80)
        print(md_content)

    return report_dict

def main():
    parser = argparse.ArgumentParser(description="Run standardized ML evaluation protocol.")
    parser.add_argument("--dataset", "-d", required=True, help="Path to input dataset CSV (real or synthetic fixture)")
    parser.add_argument("--split", "-s", default="splits/split.json", help="Path to split.json (default: splits/split.json)")
    parser.add_argument("--output-dir", "-o", default="results", help="Directory to save output reports (default: results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--force", action="store_true", help="Force recalculating and overwriting split file")
    parser.add_argument("--fixture", action="store_true", help="Force treatment of dataset as synthetic fixture")
    parser.add_argument("--feature-set", choices=["core", "extended"], default="core", help="Feature set (default: core)")
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Target validation FPR for threshold calibration (default: 0.05)")
    parser.add_argument("--target-recall", type=float, default=0.90, help="Target recall for false alarm rate calculation (default: 0.90)")
    parser.add_argument("--max-gap", type=float, default=600.0, help="Max gap seconds for session segmentation (default: 600)")

    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        split_path=args.split,
        output_dir=args.output_dir,
        seed=args.seed,
        force_split=args.force,
        is_fixture=args.fixture if args.fixture else None,
        feature_set=args.feature_set,
        target_fpr=args.target_fpr,
        target_recall=args.target_recall,
        max_gap_seconds=args.max_gap
    )

if __name__ == "__main__":
    main()
