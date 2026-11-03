import os
import tempfile
import pytest
from app.database.csv_repository import CSVRepository
from app.database.interfaces import PredictionRecord, ReadingRecord

@pytest.mark.anyio
async def test_csv_repository_reads_legacy_10_column_predictions():
    """Verify that an existing/legacy predictions.csv (10 columns, no DL fields) is read seamlessly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_file = os.path.join(tmp_dir, "predictions.csv")
        legacy_header = "ts,device_id,rule_severity,ml_severity,ml_score,published_severity,decision_mode,decision_source,label,model_version\n"
        legacy_row = "2026-01-01T10:00:00+00:00,esp32-01,0,0,0.2500,0,RULE,RULE,NORMAL,if_v1\n"
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write(legacy_header)
            f.write(legacy_row)

        repo = CSVRepository(csv_dir=tmp_dir)
        preds = await repo.get_predictions("esp32-01")
        assert len(preds) == 1
        p = preds[0]
        assert p.ts == "2026-01-01T10:00:00+00:00"
        assert p.device_id == "esp32-01"
        assert p.rule_severity == 0
        assert p.ml_severity == 0
        assert p.ml_score == 0.2500
        assert p.published_severity == 0
        assert p.decision_mode == "RULE"
        assert p.decision_source == "RULE"
        assert p.label == "NORMAL"
        assert p.model_version == "if_v1"
        assert p.dl_score is None
        assert p.dl_severity is None
        assert p.inference_ms is None

@pytest.mark.anyio
async def test_csv_repository_reads_and_writes_13_column_dl_predictions():
    """Verify writing and reading predictions.csv with DL columns (13 columns)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = CSVRepository(csv_dir=tmp_dir)
        # Pre-seed with 13-column header
        with open(repo.predictions_file, "w", encoding="utf-8") as f:
            f.write("ts,device_id,rule_severity,ml_severity,ml_score,published_severity,decision_mode,decision_source,label,model_version,dl_score,dl_severity,inference_ms\n")

        rec = PredictionRecord(
            ts="2026-01-01T11:00:00+00:00",
            device_id="esp32-01",
            rule_severity=1,
            ml_severity=1,
            ml_score=0.0150,
            published_severity=1,
            decision_mode="ML",
            decision_source="ML",
            label="ANOMALY",
            model_version="ae_v1",
            dl_score=0.015123,
            dl_severity=1,
            inference_ms=2.34
        )
        ok = await repo.save_prediction(rec)
        assert ok is True

        with open(repo.predictions_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        cols = lines[1].strip().split(",")
        assert len(cols) == 13
        assert cols[10] == "0.015123"
        assert cols[11] == "1"
        assert cols[12] == "2.34"

        preds = await repo.get_predictions("esp32-01")
        assert len(preds) == 1
        p = preds[0]
        assert p.dl_score == 0.015123
        assert p.dl_severity == 1
        assert p.inference_ms == 2.34

def test_dataset_export_preserves_all_legacy_columns():
    """Verify /export/dataset.csv stream preserves all legacy columns."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = CSVRepository(csv_dir=tmp_dir)
        stream = repo.stream_dataset_export()
        header_line = next(stream).strip()
        expected_cols = [
            "ts", "hr", "spo2", "temp", "hum", "room_temp", "finger", "fw_status",
            "rule_severity", "ml_severity", "published_severity", "decision_mode",
            "scenario", "mode", "led_color", "brightness", "buzzer_test", "source"
        ]
        for col in expected_cols:
            assert col in header_line, f"Missing legacy column {col} in dataset export"
