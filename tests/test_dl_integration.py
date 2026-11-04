import os
import json
import tempfile
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.database.csv_repository import CSVRepository
from app.mqtt.consumer import MQTTService
from app.config import settings
from app.main import app
from app.ml.dl_service import DLService

@pytest.mark.anyio
async def test_dl_dashboard_contract_fields_in_latest_and_predictions():
    """Verify that /latest and /predictions include all DL fields when model_kind=dl."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = CSVRepository(csv_dir=tmp_dir)
        service = MQTTService.__new__(MQTTService)
        service.loop = None
        service._row_count = 0
        service.prefix = "test"
        service.data_topic = "test/data"
        service.actuator_topic = "test/actuator"
        service.publisher = MagicMock()
        service.ml_service = DLService()
        service.repo = repo

        with patch.object(settings, "model_kind", "dl"), patch.object(settings, "decision_mode", "ML"):
            # Send one valid reading
            sample = {"hr": 75.0, "spo2": 98.0, "temp": 36.5, "finger": True, "device_id": "esp32-01"}
            await service._handle_data_message(sample)

            with patch("app.api.telemetry.get_repository", return_value=repo):
                client = TestClient(app)
                res_latest = client.get("/latest")
                assert res_latest.status_code == 200
                data_latest = res_latest.json()

                pred = data_latest["prediction"]
                assert pred is not None
                assert "dl_score" in pred
                assert "dl_severity" in pred
                assert "inference_ms" in pred
                assert "margin" in pred
                assert pred["model_version"] == "ae_v1"
                assert pred["decision_mode"] == "ML"
                assert pred["decision_source"] == "ML"

                res_preds = client.get("/predictions")
                assert res_preds.status_code == 200
                data_preds = res_preds.json()
                assert len(data_preds["items"]) == 1
                p_item = data_preds["items"][0]
                assert "dl_score" in p_item
                assert "dl_severity" in p_item
                assert "inference_ms" in p_item
                assert "margin" in p_item

@pytest.mark.anyio
async def test_dl_inference_exception_triggers_rule_fallback_and_preserves_raw_storage():
    """Verify that an exception during DL inference triggers RULE_FALLBACK and preserves readings.csv."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = CSVRepository(csv_dir=tmp_dir)
        service = MQTTService.__new__(MQTTService)
        service.loop = None
        service._row_count = 0
        service.prefix = "test"
        service.data_topic = "test/data"
        service.actuator_topic = "test/actuator"
        service.publisher = MagicMock()
        service.ml_service = DLService()
        service.repo = repo

        with patch.object(settings, "model_kind", "dl"), patch.object(settings, "decision_mode", "ML"):
            # Mock process_reading to raise an unexpected runtime exception
            with patch.object(service.ml_service, "process_reading", side_effect=RuntimeError("Simulated DL crash")):
                # HR 105, SpO2 88 is rule severity 2
                sample = {"hr": 105.0, "spo2": 88.0, "temp": 36.5, "finger": True, "device_id": "esp32-01"}
                await service._handle_data_message(sample)

            # 1. Raw storage in readings.csv must NOT stop
            with open(repo.readings_file, "r", encoding="utf-8") as f:
                readings = f.read()
            assert "105.0" in readings
            assert "88.0" in readings

            # 2. Prediction record must show fallback
            preds = await repo.get_predictions("esp32-01")
            assert len(preds) == 1
            p = preds[0]
            assert p.decision_mode == "ML"
            assert p.decision_source == "RULE_FALLBACK"
            assert p.published_severity == 2
            assert p.rule_severity == 2

            # 3. /result payload published must be rule severity
            service.publisher.publish_result.assert_called_once()
            called_payload = service.publisher.publish_result.call_args[0][0]
            assert called_payload.severity == 2

@pytest.mark.anyio
async def test_dl_finger_false_resets_and_omits_result():
    """Verify finger=False resets DL state and omits /result publish."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = CSVRepository(csv_dir=tmp_dir)
        service = MQTTService.__new__(MQTTService)
        service.loop = None
        service._row_count = 0
        service.prefix = "test"
        service.data_topic = "test/data"
        service.actuator_topic = "test/actuator"
        service.publisher = MagicMock()
        service.ml_service = DLService()
        service.repo = repo

        with patch.object(settings, "model_kind", "dl"), patch.object(settings, "decision_mode", "ML"):
            sample = {"hr": None, "spo2": None, "temp": None, "finger": False, "device_id": "esp32-01"}
            await service._handle_data_message(sample)

            # Raw storage must store the reading
            with open(repo.readings_file, "r", encoding="utf-8") as f:
                readings = f.read()
            assert "False" in readings

            # /result must NOT be published
            service.publisher.publish_result.assert_not_called()
            # State must be reset
            state = service.ml_service._get_state("esp32-01")
            assert state.active_severity == 0
            assert state.consecutive_warn == 0
