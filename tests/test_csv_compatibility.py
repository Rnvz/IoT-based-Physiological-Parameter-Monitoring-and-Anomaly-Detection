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

@pytest.mark.anyio
async def test_csv_repository_migrates_legacy_10_col_header_to_13_on_dl_write():
    """Verify in-place non-destructive migration from 10-col legacy to 13-col DL format."""
    from app.config import settings
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "predictions.csv")
        legacy_header = "ts,device_id,rule_severity,ml_severity,ml_score,published_severity,decision_mode,decision_source,label,model_version\n"
        row1 = "2026-11-03T10:00:00Z,dev1,0,0,0.1000,0,RULE,RULE,NORMAL,iso_forest_v1\n"
        row2 = "2026-11-03T10:00:01Z,dev1,1,1,0.6000,1,ML,ML,WARNING,iso_forest_v1\n"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(legacy_header + row1 + row2)

        prev_kind = settings.model_kind
        settings.model_kind = "dl"
        try:
            repo = CSVRepository(csv_dir=tmp_dir)
            p1 = PredictionRecord(
                ts="2026-11-03T10:00:02Z",
                device_id="dev1",
                rule_severity=0,
                ml_severity=0,
                ml_score=0.0123,
                published_severity=0,
                decision_mode="ML",
                decision_source="DL",
                label="NORMAL",
                model_version="dense_ae_v2",
                dl_score=0.012345,
                dl_severity=0,
                inference_ms=1.52
            )
            p2 = PredictionRecord(
                ts="2026-11-03T10:00:03Z",
                device_id="dev1",
                rule_severity=2,
                ml_severity=2,
                ml_score=0.0891,
                published_severity=2,
                decision_mode="ML",
                decision_source="DL",
                label="ALERT",
                model_version="dense_ae_v2",
                dl_score=0.089123,
                dl_severity=2,
                inference_ms=1.84
            )
            assert await repo.save_prediction(p1) is True
            assert await repo.save_prediction(p2) is True

            with open(csv_path, "r", encoding="utf-8") as f:
                lines = [line.strip().split(",") for line in f if line.strip()]

            # Header + 4 rows
            assert len(lines) == 5
            assert len(lines[0]) == 13
            assert lines[0][10:] == ["dl_score", "dl_severity", "inference_ms"]

            # Legacy rows padded
            assert len(lines[1]) == 13
            assert lines[1][0] == "2026-11-03T10:00:00Z"
            assert lines[1][10:] == ["", "", ""]

            assert len(lines[2]) == 13
            assert lines[2][0] == "2026-11-03T10:00:01Z"
            assert lines[2][10:] == ["", "", ""]

            # New DL rows
            assert len(lines[3]) == 13
            assert lines[3][10:] == ["0.012345", "0", "1.52"]

            assert len(lines[4]) == 13
            assert lines[4][10:] == ["0.089123", "2", "1.84"]

            # Reverse scenario: repo initialized with model_kind="ml" on 13-col file
            settings.model_kind = "ml"
            repo_ml = CSVRepository(csv_dir=tmp_dir)

            r = ReadingRecord(
                ts="2026-11-03T10:00:02Z",
                device_id="dev1",
                hr=75.0,
                spo2=98.0,
                temp=36.5,
                hum=50.0,
                room_temp=25.0,
                finger=True,
                fw_status="NORMAL"
            )
            assert await repo_ml.save_reading(r) is True

            p_ml = PredictionRecord(
                ts="2026-11-03T10:00:04Z",
                device_id="dev1",
                rule_severity=0,
                ml_severity=0,
                ml_score=0.1500,
                published_severity=0,
                decision_mode="ML",
                decision_source="ML",
                label="NORMAL",
                model_version="iso_forest_v1",
                dl_score=None,
                dl_severity=None,
                inference_ms=None
            )
            assert await repo_ml.save_prediction(p_ml) is True

            preds = await repo_ml.get_predictions("dev1")
            assert len(preds) == 5
            assert preds[0].dl_score is None
            assert preds[2].dl_score == 0.012345
            assert preds[3].dl_severity == 2
            assert preds[4].dl_score is None

            export_stream = list(repo_ml.stream_dataset_export())
            assert len(export_stream) == 2  # header + reading row
            assert "75.0,98.0" in export_stream[1]
        finally:
            settings.model_kind = prev_kind

