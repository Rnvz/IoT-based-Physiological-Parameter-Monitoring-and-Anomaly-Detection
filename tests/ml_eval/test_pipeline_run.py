"""
Integration tests for ml_eval.run: end-to-end pipeline execution, fixture banner, and reports.
"""

import os
import json
import pytest

from ml.eval.run import run_pipeline, BANNER_TEXT


def test_pipeline_fixture_banner_and_report_generation(tmp_path, synthetic_fixture_csv, capsys):
    """
    Verify that:
    1. The banner 'SYNTHETIC FIXTURE DATA - NOT A RESULT' is displayed for fixtures.
    2. Results are saved to results/<split_sha>/report.json and report.md.
    3. The report lists split hash and environment library versions.
    """
    out_dir = str(tmp_path / "results")
    split_file = str(tmp_path / "splits" / "split.json")

    report = run_pipeline(
        dataset_path=synthetic_fixture_csv,
        split_path=split_file,
        output_dir=out_dir,
        seed=42,
        is_fixture=True,
        feature_set="core",
        quiet=False
    )

    captured = capsys.readouterr()
    # 1. Output must display the banner
    assert BANNER_TEXT in captured.out

    # 2. Results written to results/<split_sha>/
    split_sha = report["split_hash"]
    dest_dir = os.path.join(out_dir, split_sha)
    json_path = os.path.join(dest_dir, "report.json")
    md_path = os.path.join(dest_dir, "report.md")

    assert os.path.exists(json_path)
    assert os.path.exists(md_path)

    # 3. Check JSON report contents
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["banner"] == BANNER_TEXT
    assert data["is_fixture"] is True
    assert data["split_hash"] == split_sha
    assert "environment" in data
    assert "scikit_learn" in data["environment"]
    assert "numpy" in data["environment"]
    assert "pandas" in data["environment"]
    assert "joblib" in data["environment"]
    assert "python" in data["environment"]

    # All baseline models must be evaluated
    expected_models = {"Rule Engine", "Isolation Forest", "Local Outlier Factor", "One-Class SVM"}
    assert set(data["models"].keys()) == expected_models

    # Check Markdown report contents
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert BANNER_TEXT in md_text
    assert split_sha in md_text


def test_pipeline_determinism(tmp_path, synthetic_fixture_csv):
    """Verify that running the pipeline twice with the same seed produces identical metrics."""
    out_dir_1 = str(tmp_path / "run_1")
    out_dir_2 = str(tmp_path / "run_2")
    split_file_1 = str(tmp_path / "split_1.json")
    split_file_2 = str(tmp_path / "split_2.json")

    rep1 = run_pipeline(
        dataset_path=synthetic_fixture_csv,
        split_path=split_file_1,
        output_dir=out_dir_1,
        seed=42,
        is_fixture=True,
        quiet=True
    )

    rep2 = run_pipeline(
        dataset_path=synthetic_fixture_csv,
        split_path=split_file_2,
        output_dir=out_dir_2,
        seed=42,
        is_fixture=True,
        quiet=True
    )

    assert rep1["split_hash"] == rep2["split_hash"]
    assert rep1["splits"] == rep2["splits"]

    # Check model thresholds and metrics match exactly
    for m in rep1["models"]:
        assert rep1["models"][m]["operating_threshold"] == rep2["models"][m]["operating_threshold"]
        assert rep1["models"][m]["metrics_synthetic"] == rep2["models"][m]["metrics_synthetic"]
