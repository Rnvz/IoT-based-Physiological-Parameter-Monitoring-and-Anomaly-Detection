"""
ml_eval.public_split: Frozen dataset partitioning for public cross-sectional and longitudinal datasets.

Splitting Rules:
1. Cross-sectional sources (nasirayub2, gourangomandal):
   - Group-split strictly by Patient_ID with fixed seed (60% Train, 20% Val, 20% Test).
   - Zero patient overlap between partitions.
2. Longitudinal streaming source (engrarri21):
   - Strictly chronological blocks: 60% Train, 20% Val, 20% Test.
   - 60-row purge buffer (gap) inserted between Train and Val, and between Val and Test.
   - Purge buffers are completely excluded from all sets to prevent boundary autocorrelation leakage.
3. Leave-One-Source-Out (LOSO) folds defined for domain generalization testing.
4. Refuses to overwrite splits/public_split.json without --force.
"""

import os
import sys
import json
import random
import hashlib
import argparse
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 hex digest of file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def split_public_dataset(
    df: pd.DataFrame,
    seed: int = 42,
    gap_rows: int = 60
) -> Dict[str, Any]:
    """
    Computes deterministic index partitioning across all sources in harmonized_vitals.
    Returns dictionary with indices and summary counts.
    """
    required_cols = {"Patient_ID", "Heart_Rate", "SpO2", "Temperature", "OUTPUT", "Source"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Add row_idx for exact index tracking
    df = df.copy()
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    train_indices = []
    val_indices = []
    test_indices = []

    source_stats = {}

    for src in sorted(df["Source"].unique()):
        src_df = df[df["Source"] == src]
        src_rows = src_df["row_id"].values

        if src == "engrarri21":
            # Longitudinal / sequential block split
            N = len(src_rows)
            n_tr = int(round(0.60 * N))
            n_val = int(round(0.20 * N))

            tr_idx = src_rows[0:n_tr].tolist()
            # 60 row gap
            val_start = n_tr + gap_rows
            val_idx = src_rows[val_start:val_start + n_val].tolist()
            # 60 row gap
            te_start = val_start + n_val + gap_rows
            te_idx = src_rows[te_start:].tolist()

            source_stats[src] = {
                "type": "chronological_blocks",
                "total_rows": N,
                "gap_rows": gap_rows,
                "train_rows": len(tr_idx),
                "val_rows": len(val_idx),
                "test_rows": len(te_idx),
                "purged_rows": N - (len(tr_idx) + len(val_idx) + len(te_idx))
            }
        else:
            # Cross-sectional: Group split by Patient_ID
            patients = sorted(src_df["Patient_ID"].unique())
            rng = random.Random(seed)
            shuffled_patients = list(patients)
            rng.shuffle(shuffled_patients)

            n_pts = len(shuffled_patients)
            n_tr_pts = int(round(0.60 * n_pts))
            n_val_pts = int(round(0.20 * n_pts))

            train_pts = set(shuffled_patients[:n_tr_pts])
            val_pts = set(shuffled_patients[n_tr_pts:n_tr_pts + n_val_pts])
            test_pts = set(shuffled_patients[n_tr_pts + n_val_pts:])

            tr_idx = src_df[src_df["Patient_ID"].isin(train_pts)]["row_id"].tolist()
            val_idx = src_df[src_df["Patient_ID"].isin(val_pts)]["row_id"].tolist()
            te_idx = src_df[src_df["Patient_ID"].isin(test_pts)]["row_id"].tolist()

            source_stats[src] = {
                "type": "group_by_patient",
                "total_rows": len(src_df),
                "total_patients": n_pts,
                "train_patients": len(train_pts),
                "val_patients": len(val_pts),
                "test_patients": len(test_pts),
                "train_rows": len(tr_idx),
                "val_rows": len(val_idx),
                "test_rows": len(te_idx)
            }

        train_indices.extend(tr_idx)
        val_indices.extend(val_idx)
        test_indices.extend(te_idx)

    # Class breakdown per partition
    def class_breakdown(indices: List[int]) -> Dict[str, Any]:
        sub = df.loc[indices]
        return {
            "total": len(sub),
            "normal_output_0": int((sub["OUTPUT"] == 0).sum()),
            "abnormal_output_1": int((sub["OUTPUT"] == 1).sum()),
            "by_source": {
                s: {
                    "total": int((sub["Source"] == s).sum()),
                    "output_0": int(((sub["Source"] == s) & (sub["OUTPUT"] == 0)).sum()),
                    "output_1": int(((sub["Source"] == s) & (sub["OUTPUT"] == 1)).sum())
                }
                for s in sorted(df["Source"].unique())
            }
        }

    # Leave-One-Source-Out (LOSO) folds
    sources = sorted(df["Source"].unique().tolist())
    loso_folds = {}
    for held_out in sources:
        train_srcs = [s for s in sources if s != held_out]
        fold_train_idx = df[df["Source"].isin(train_srcs)]["row_id"].tolist()
        fold_test_idx = df[df["Source"] == held_out]["row_id"].tolist()
        loso_folds[f"leave_out_{held_out}"] = {
            "held_out_source": held_out,
            "train_sources": train_srcs,
            "train_rows": len(fold_train_idx),
            "test_rows": len(fold_test_idx),
            "train_indices": fold_train_idx,
            "test_indices": fold_test_idx
        }

    split_dict = {
        "protocol_version": "public_v1",
        "seed": seed,
        "gap_rows": gap_rows,
        "total_dataset_rows": len(df),
        "total_train_rows": len(train_indices),
        "total_val_rows": len(val_indices),
        "total_test_rows": len(test_indices),
        "source_breakdown": source_stats,
        "class_breakdown": {
            "train": class_breakdown(train_indices),
            "val": class_breakdown(val_indices),
            "test": class_breakdown(test_indices)
        },
        "indices": {
            "train": sorted(train_indices),
            "val": sorted(val_indices),
            "test": sorted(test_indices)
        },
        "loso_folds": loso_folds
    }

    return split_dict


def freeze_public_split(
    df: pd.DataFrame,
    output_path: str = "splits/public_split.json",
    seed: int = 42,
    force: bool = False
) -> Tuple[Dict[str, Any], str]:
    """
    Creates and freezes public_split.json.
    Refuses overwrite unless force=True.
    Returns (split_dict, sha256_hash).
    """
    abs_path = os.path.abspath(output_path)
    if os.path.exists(abs_path) and not force:
        existing_sha = compute_file_sha256(abs_path)
        raise FileExistsError(
            f"Frozen public split already exists at '{output_path}' (SHA256: {existing_sha}). "
            f"Pass force=True or --force only if explicitly resetting the frozen benchmark."
        )

    split_dict = split_public_dataset(df, seed=seed)

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    json_str = json.dumps(split_dict, indent=2, sort_keys=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(json_str + "\n")

    sha256_hash = compute_file_sha256(abs_path)
    return split_dict, sha256_hash


def load_public_split(split_path: str = "splits/public_split.json") -> Tuple[Dict[str, Any], str]:
    """Loads frozen public split and verifies its SHA256 digest."""
    abs_path = os.path.abspath(split_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Frozen split not found at: '{split_path}'. Run ml_eval.public_split first.")

    sha256 = compute_file_sha256(abs_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta, sha256


def main():
    parser = argparse.ArgumentParser(description="Freeze public dataset train/val/test and LOSO splits.")
    parser.add_argument("--input", "-i", default="ml_pipeline/data/harmonized_vitals.csv", help="Path to harmonized dataset")
    parser.add_argument("--output", "-o", default="splits/public_split.json", help="Path to output split.json")
    parser.add_argument("--seed", type=int, default=42, help="Seed for cross-sectional patient shuffle")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing frozen split")
    args = parser.parse_args()

    print(f"Loading harmonized vitals from: {args.input}")
    df = pd.read_csv(args.input)

    try:
        split_dict, sha256 = freeze_public_split(df, output_path=args.output, seed=args.seed, force=args.force)
    except FileExistsError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("FROZEN PUBLIC DATASET SPLIT CREATED")
    print("=" * 80)
    print(f"File Path    : {args.output}")
    print(f"SHA256 Hash  : {sha256}")
    print(f"Total Rows   : {split_dict['total_dataset_rows']}")
    print(f"Train Rows   : {split_dict['total_train_rows']}")
    print(f"Val Rows     : {split_dict['total_val_rows']}")
    print(f"Test Rows    : {split_dict['total_test_rows']}")
    print("-" * 80)
    for src, info in split_dict["source_breakdown"].items():
        print(f"[{src}] Type: {info['type']} | Train: {info['train_rows']}, Val: {info['val_rows']}, Test: {info['test_rows']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
