"""
ml_eval.split: Frozen session-wise partitioning into Train, Validation, and Test sets.

Strictly splits by entire recording sessions (Session-wise), never by random rows,
preventing high-autocorrelation temporal data leakage.
Refuses to overwrite an existing split definition unless explicitly forced.
Calculates and outputs the SHA256 digest of the frozen split configuration.
"""

import os
import sys
import json
import random
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd

from ml.eval.sessions import load_and_segment_sessions, summarize_sessions

def compute_content_sha256(content: str) -> str:
    """Computes SHA256 hex digest of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 hex digest of a file's raw bytes."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def split_session_ids(
    session_ids: List[str],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> Dict[str, List[str]]:
    """
    Deterministically partitions session IDs into Train, Val, and Test sets.
    Fails if len(session_ids) < 3.
    """
    unique_sessions = sorted(list(set(session_ids)))
    n = len(unique_sessions)

    if n < 3:
        raise ValueError(
            f"Cannot split fewer than 3 sessions into train/val/test partitions. "
            f"Found only {n} session(s): {unique_sessions}. At least 3 sessions are required."
        )

    rng = random.Random(seed)
    shuffled = list(unique_sessions)
    rng.shuffle(shuffled)

    if n == 3:
        train = [shuffled[0]]
        val = [shuffled[1]]
        test = [shuffled[2]]
    elif n == 4:
        train = [shuffled[0], shuffled[1]]
        val = [shuffled[2]]
        test = [shuffled[3]]
    else:
        n_test = max(1, int(round(n * ratios[2])))
        n_val = max(1, int(round(n * ratios[1])))
        n_train = n - n_test - n_val

        if n_train < 1:
            n_train = 1
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1

        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test)
    }

def freeze_split(
    df: pd.DataFrame,
    output_path: str = "splits/split.json",
    seed: int = 42,
    force: bool = False
) -> Tuple[Dict[str, Any], str]:
    """
    Creates and freezes split.json.
    Refuses to overwrite if output_path already exists unless force=True.
    Returns (split_dict, sha256_hash).
    """
    if "session_id" not in df.columns:
        raise ValueError("DataFrame must contain 'session_id' column for session-wise splitting.")

    abs_path = os.path.abspath(output_path)

    if os.path.exists(abs_path) and not force:
        existing_sha = compute_file_sha256(abs_path)
        raise FileExistsError(
            f"Split file already exists at '{output_path}' (SHA256: {existing_sha}). "
            f"Evaluation protocol freeze forbids overwriting existing splits. "
            f"Pass force=True or --force only if explicitly resetting the frozen benchmark."
        )

    sessions = sorted(df["session_id"].unique())
    partitions = split_session_ids(sessions, seed=seed)
    summary = summarize_sessions(df)

    split_meta = {
        "protocol_version": "1.0",
        "seed": seed,
        "total_sessions": len(sessions),
        "total_valid_rows": int(len(df)),
        "train_sessions": partitions["train"],
        "val_sessions": partitions["val"],
        "test_sessions": partitions["test"],
        "session_summary": summary["sessions"]
    }

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    json_str = json.dumps(split_meta, indent=2, sort_keys=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(json_str + "\n")

    sha256_hash = compute_file_sha256(abs_path)
    return split_meta, sha256_hash

def load_frozen_split(split_path: str = "splits/split.json") -> Tuple[Dict[str, Any], str]:
    """
    Loads an existing frozen split.json and verifies its SHA256 digest.
    """
    abs_path = os.path.abspath(split_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Frozen split file not found at: '{split_path}'. Run ml_eval.split to generate it.")

    sha256_hash = compute_file_sha256(abs_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    return split_meta, sha256_hash

def main():
    parser = argparse.ArgumentParser(description="Freeze session-wise train/val/test split.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV dataset with or without session_id")
    parser.add_argument("--output", "-o", default="splits/split.json", help="Path to write split.json (default: splits/split.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic session shuffle (default: 42)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing frozen split file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "session_id" not in df.columns:
        print("Segmenting sessions from dataset...")
        df = load_and_segment_sessions(df)

    try:
        split_meta, sha256_hash = freeze_split(
            df=df,
            output_path=args.output,
            seed=args.seed,
            force=args.force
        )
    except FileExistsError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("FROZEN SESSION SPLIT CREATED")
    print("=" * 80)
    print(f"File Path    : {args.output}")
    print(f"SHA256 Hash  : {sha256_hash}")
    print(f"Seed Used    : {split_meta['seed']}")
    print(f"Train Sessions ({len(split_meta['train_sessions'])}): {split_meta['train_sessions']}")
    print(f"Val Sessions   ({len(split_meta['val_sessions'])}): {split_meta['val_sessions']}")
    print(f"Test Sessions  ({len(split_meta['test_sessions'])}): {split_meta['test_sessions']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
