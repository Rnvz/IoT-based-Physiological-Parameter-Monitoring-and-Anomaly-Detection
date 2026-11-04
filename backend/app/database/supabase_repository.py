import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from app.config import settings
from app.database.interfaces import (
    BaseRepository,
    ReadingRecord,
    PredictionRecord,
    ActuatorEventRecord,
    ScenarioEventRecord,
)
from app.database.csv_repository import CSVRepository

logger = logging.getLogger(__name__)

class SupabaseRepository(BaseRepository):
    """
    Supabase REST API repository with Single Consolidated Table ('physio_records')
    and dual-storage local CSV backup.
    
    1. Saves all events locally to CSV (ensuring CSV export & offline work).
    2. Sends a single integrated record to Supabase ('physio_records') ONLY when
       the model evaluation confirms a 'NORMAL' or 'ANOMALY' classification.
       Each record includes:
         - Sensor Readings (hr, spo2, temp, hum, room_temp)
         - Model Decision (label, severity, ml_score)
         - Auto Actuator Response (led_color, buzzer_active)
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_service_key: Optional[str] = None,
        csv_dir: str = "./data",
    ):
        self.url = (supabase_url or settings.supabase_url or "").rstrip("/")
        self.key = supabase_service_key or settings.supabase_service_key or ""
        self.csv_repo = CSVRepository(csv_dir=csv_dir)
        self.enabled = bool(self.url and self.key)

        # Cache reading terakhir untuk digabungkan saat prediksi model terbit
        self._last_reading: Optional[ReadingRecord] = None

        # Cache status aktuator aktual dari ESP32
        self._current_led_color = "GREEN"
        self._current_buzzer_active = False

        if not self.enabled:
            logger.warning("[SupabaseRepo] SUPABASE_URL or SUPABASE_SERVICE_KEY not configured. Running in CSV-only mode.")
        else:
            logger.info(f"[SupabaseRepo] Connected to Supabase at {self.url} (Single-Table Mode: physio_records)")

    async def _post_rest(self, table: str, payload: Dict[str, Any]) -> bool:
        """Sends an async POST request to Supabase PostgREST endpoint."""
        if not self.enabled:
            return False

        endpoint = f"{self.url}/rest/v1/{table}"
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        try:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(endpoint, json=payload, headers=headers)
                    if resp.status_code in (200, 201, 204):
                        return True
                    else:
                        logger.error(f"[SupabaseRepo] Error inserting to {table}: {resp.status_code} - {resp.text}")
                        return False
            except ImportError:
                import urllib.request
                import urllib.error

                data_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")

                def _sync_request():
                    with urllib.request.urlopen(req, timeout=5) as response:
                        return response.status in (200, 201, 204)

                return await asyncio.to_thread(_sync_request)

        except Exception as e:
            logger.error(f"[SupabaseRepo] Network exception pushing to {table}: {e}")
            return False

    async def save_reading(self, reading: ReadingRecord) -> bool:
        """
        Saves raw reading to local CSV.
        Caches reading to be consolidated into the single table when model prediction arrives.
        """
        saved_local = await self.csv_repo.save_reading(reading)
        if reading.finger and reading.hr is not None and reading.spo2 is not None:
            self._last_reading = reading
        return saved_local

    async def save_prediction(self, prediction: PredictionRecord) -> bool:
        """
        Saves prediction output to local CSV.
        Consolidates Sensor + Model + Auto Actuator into 1 table in Supabase ('physio_records')
        ONLY when label is 'NORMAL' or 'ANOMALY'.
        """
        saved_local = await self.csv_repo.save_prediction(prediction)

        if not self.enabled:
            return saved_local

        # KETENTUAN 1: Hanya data yang sudah memiliki tanda NORMAL atau ANOMALY
        if prediction.label not in ("NORMAL", "ANOMALY"):
            return saved_local

        # Tentukan respon aktuator secara AUTO dari hasil keputusan model
        sev = prediction.published_severity
        if sev <= 0:
            auto_led = "GREEN"
            auto_buzzer = False
        elif sev == 1:
            auto_led = "YELLOW"
            auto_buzzer = False
        else:
            auto_led = "RED"
            auto_buzzer = True

        # Ambil data sensor terkini
        reading = self._last_reading
        hr = reading.hr if reading else None
        spo2 = reading.spo2 if reading else None
        temp = reading.temp if reading else None
        hum = reading.hum if reading else None
        room_temp = reading.room_temp if reading else None

        # Baris Tunggal Konsolidasi
        single_record_payload = {
            "ts": prediction.ts,
            "device_id": prediction.device_id,
            "hr": hr,
            "spo2": spo2,
            "temp": temp,
            "hum": hum,
            "room_temp": room_temp,
            "label": prediction.label,
            "severity": sev,
            "ml_score": prediction.ml_score,
            "led_color": auto_led,
            "buzzer_active": auto_buzzer,
        }

        # Kirim secara asinkron ke Supabase tabel tunggal 'physio_records'
        asyncio.create_task(self._post_rest("physio_records", single_record_payload))
        logger.debug(f"[SupabaseRepo] Pushed single record to physio_records: {prediction.label} (Sev {sev})")

        return saved_local

    async def save_actuator_event(self, event: ActuatorEventRecord) -> bool:
        """
        Saves actuator event to local CSV.
        Updates internal actuator cache for synchronization.
        """
        saved_local = await self.csv_repo.save_actuator_event(event)

        try:
            parsed = json.loads(event.payload_json)
            if isinstance(parsed, dict) and parsed.get("mode") == "AUTO":
                led = parsed.get("led", {})
                if isinstance(led, dict):
                    self._current_led_color = led.get("color", self._current_led_color)
                buzzer = parsed.get("buzzer", {})
                if isinstance(buzzer, dict):
                    self._current_buzzer_active = bool(buzzer.get("active", False))
        except Exception:
            pass

        return saved_local

    async def get_latest_reading(self, device_id: str) -> Optional[ReadingRecord]:
        return await self.csv_repo.get_latest_reading(device_id)

    async def get_latest_prediction(self, device_id: str) -> Optional[PredictionRecord]:
        return await self.csv_repo.get_latest_prediction(device_id)

    async def get_history(self, device_id: str, minutes: int = 30) -> List[ReadingRecord]:
        return await self.csv_repo.get_history(device_id, minutes=minutes)

    async def get_predictions(self, device_id: str, limit: int = 100) -> List[PredictionRecord]:
        return await self.csv_repo.get_predictions(device_id, limit=limit)

    async def get_latest_actuator_event(self, kind: Optional[str] = None) -> Optional[ActuatorEventRecord]:
        return await self.csv_repo.get_latest_actuator_event(kind=kind)

    def stream_dataset_export(self):
        """Streams CSV dataset directly from local CSV storage for course export/submission."""
        return self.csv_repo.stream_dataset_export()

    async def save_scenario_event(self, event: ScenarioEventRecord) -> bool:
        return await self.csv_repo.save_scenario_event(event)

    async def get_current_scenario() -> ScenarioEventRecord:
        return await self.csv_repo.get_current_scenario()
