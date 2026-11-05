"""
Unit tests for ml_eval.split: session-wise partitioning and freeze mechanics.
"""

import os
import pytest
import pandas as pd

from ml.eval.split import (
    split_session_ids,
    freeze_split,
    load_frozen_split,
    compute_file_sha256
)


def test_split_by_session_exclusivity():
    """Verify that sessions are strictly partitioned into disjoint sets."""
    sessions = [f"session_{i:03d}" for i in range(1, 11)]  # 10 sessions
    splits = split_session_ids(sessions, seed=42)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Disjointness
    assert train_set.isdisjoint(val_set)
    assert val_set.isdisjoint(test_set)
    assert train_set.isdisjoint(test_set)

    # Union covers all sessions
    assert (train_set | val_set | test_set) == set(sessions)

    # All partitions have at least 1 session
    assert len(train_set) >= 1
    assert len(val_set) >= 1
    assert len(test_set) >= 1


def test_split_fails_with_fewer_than_3_sessions():
    """Verify that split_session_ids raises ValueError when fewer than 3 sessions exist."""
    with pytest.raises(ValueError, match="Cannot split fewer than 3 sessions"):
        split_session_ids(["session_001", "session_002"], seed=42)

    with pytest.raises(ValueError, match="Cannot split fewer than 3 sessions"):
        split_session_ids(["session_001"], seed=42)


def test_split_determinism():
    """Verify that the same seed produces identical splits and different seeds yield different splits."""
    sessions = [f"session_{i:03d}" for i in range(1, 15)]

    split_a = split_session_ids(sessions, seed=42)
    split_b = split_session_ids(sessions, seed=42)
    split_c = split_session_ids(sessions, seed=999)

    assert split_a == split_b
    assert split_a != split_c


def test_freeze_split_refuses_overwrite(tmp_path, synthetic_fixture_df):
    """Verify that freeze_split raises FileExistsError on existing file unless force=True."""
    from ml.eval.sessions import load_and_segment_sessions

    segmented = load_and_segment_sessions(synthetic_fixture_df)
    target_json = tmp_path / "splits" / "split.json"

    # First write must succeed
    meta1, sha1 = freeze_split(segmented, output_path=str(target_json), seed=42, force=False)
    assert os.path.exists(target_json)
    assert sha1 == compute_file_sha256(str(target_json))

    # Second write without force must raise FileExistsError
    with pytest.raises(FileExistsError, match="Split file already exists"):
        freeze_split(segmented, output_path=str(target_json), seed=42, force=False)

    # Second write WITH force=True must succeed
    meta2, sha2 = freeze_split(segmented, output_path=str(target_json), seed=42, force=True)
    assert sha2 == sha1


def test_load_frozen_split(tmp_path, synthetic_fixture_df):
    """Verify loading and hash verification of frozen split."""
    from ml.eval.sessions import load_and_segment_sessions

    segmented = load_and_segment_sessions(synthetic_fixture_df)
    target_json = tmp_path / "split.json"

    meta_saved, sha_saved = freeze_split(segmented, output_path=str(target_json), seed=42)
    meta_loaded, sha_loaded = load_frozen_split(str(target_json))

    assert sha_saved == sha_loaded
    assert meta_saved["train_sessions"] == meta_loaded["train_sessions"]
    assert meta_saved["val_sessions"] == meta_loaded["val_sessions"]
    assert meta_saved["test_sessions"] == meta_loaded["test_sessions"]
