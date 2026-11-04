import json
import os
import tempfile
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.database.csv_repository import CSVRepository
from app.mqtt.consumer import MQTTService
from app.main import app
from app.ml.inference import get_ml_service

@pytest.mark.anyio
async def test_golden_rule_execution_and_outputs():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "golden_rule_run.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = CSVRepository(csv_dir=tmp_dir)
        service = MQTTService.__new__(MQTTService)
        service.loop = None
        service._row_count = 0
        service.prefix = "test"
        service.data_topic = "test/data"
        service.actuator_topic = "test/actuator"
        service.publisher = MagicMock()
        service.ml_service = get_ml_service()
        service.repo = repo

        for item, fixed_ts in golden["telemetries"]:
            mock_dt = datetime.fromisoformat(fixed_ts)
            with patch("app.mqtt.consumer.datetime") as mock_datetime:
                mock_datetime.now.return_value = mock_dt
                await service._handle_data_message(item)

        with open(repo.readings_file, "r", encoding="utf-8") as f:
            readings_content = f.read()
        with open(repo.predictions_file, "r", encoding="utf-8") as f:
            preds_content = f.read()

        with patch("app.api.telemetry.get_repository", return_value=repo):
            client = TestClient(app)
            latest_res = client.get("/latest").json()
            preds_res = client.get("/predictions").json()

        assert readings_content == golden["readings_csv"]
        assert preds_content == golden["predictions_csv"]
        assert latest_res == golden["latest_response"]
        assert preds_res == golden["predictions_response"]
